from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from cattle_weight.cv.segmentation import (
    bbox_from_mask,
    contour_from_mask,
    required_pvtv2_segmentation_source,
    should_require_pvtv2_inference,
)
from cattle_weight.features.schema import (
    KEYPOINT_LINKS,
    PAPER_FEATURES,
    STATS,
    build_feature_specs,
)
from cattle_weight.io import read_json, read_parquet, registry_dir, reports_dir, write_parquet, write_yaml
from cattle_weight.progress import ProgressLogger, logger


def extract_features(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    index = read_parquet(registry_dir(config) / "dataset_index.parquet")
    cv = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    keypoints = read_parquet(registry_dir(config) / "keypoint_metadata.parquet")
    depth = read_parquet(registry_dir(config) / "depth_metadata.parquet")
    pointcloud = read_parquet(registry_dir(config) / "pointcloud_metadata.parquet")
    assert_feature_inputs_ready(config, cv, keypoints, depth, pointcloud)
    merged = (
        index[["image_id", "dataset", "weight_kg", "data_version"]]
        .merge(cv, on=["image_id", "dataset"], how="left")
        .merge(keypoints[["image_id", "keypoint_status", "keypoint_path"]], on="image_id", how="left")
        .merge(pointcloud[["image_id", "pointcloud_status", "pointcloud_path"]], on="image_id", how="left")
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    feature_cfg = config.get("features", {})
    log_every = int(feature_cfg.get("log_every", 100))
    workers = max(1, int(os.environ.get("CATTLE_WEIGHT_FEATURE_WORKERS", feature_cfg.get("workers", 1))))
    schema_version = feature_cfg.get("schema_version", "paper_84_v1")
    config_hash = config.get("_config_hash")
    progress = ProgressLogger("feature extraction", total=len(merged), log_every=log_every)
    success_count = 0
    failed_count = 0
    logger().info(
        "Feature extraction setup | total_rows=%d | schema=%s | workers=%d | log_every=%d",
        len(merged),
        schema_version,
        workers,
        log_every,
    )
    records = merged.to_dict(orient="records")
    for row_order, record in enumerate(records):
        record["_row_order"] = row_order
    if workers == 1:
        result_iter = (
            extract_feature_row(record, config_hash=config_hash, schema_version=schema_version)
            for record in records
        )
        for completed, result in enumerate(result_iter, start=1):
            success_count, failed_count = collect_feature_result(
                result,
                rows,
                failures,
                progress,
                completed,
                success_count,
                failed_count,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_image_id = {
                executor.submit(extract_feature_row, record, config_hash=config_hash, schema_version=schema_version): record["image_id"]
                for record in records
            }
            for completed, future in enumerate(as_completed(future_to_image_id), start=1):
                result = future.result()
                success_count, failed_count = collect_feature_result(
                    result,
                    rows,
                    failures,
                    progress,
                    completed,
                    success_count,
                    failed_count,
                )
    rows = sorted(rows, key=lambda item: item["_row_order"])
    for row in rows:
        row.pop("_row_order", None)
    features_df = pd.DataFrame(rows)

    missing = features_df[PAPER_FEATURES].isna().mean().reset_index()
    missing.columns = ["feature", "missing_fraction"]
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    missing.to_csv(reports_dir(config) / "feature_missingness.csv", index=False)
    features_df[PAPER_FEATURES].describe().T.to_csv(reports_dir(config) / "feature_summary.csv")
    write_schema(config)
    if features_df[PAPER_FEATURES].isna().all(axis=None) and config.get("features", {}).get("fail_on_all_missing", True):
        raise RuntimeError("All extracted paper features are missing. Check upstream CV artifacts.")
    outputs: dict[str, pd.DataFrame] = {}
    for dataset, part in features_df.groupby("dataset"):
        path = registry_dir(config) / f"features_{dataset.lower()}.parquet"
        write_parquet(part.reset_index(drop=True), path)
        outputs[dataset] = part.reset_index(drop=True)
        dataset_counts = part["feature_status"].value_counts(dropna=False).to_dict()
        logger().info("Feature table written | dataset=%s | rows=%d | status_counts=%s | path=%s", dataset, len(part), dataset_counts, path)
    pd.DataFrame(failures).to_csv(reports_dir(config) / "feature_extraction_failures.csv", index=False)
    status_counts = features_df["feature_status"].value_counts(dropna=False).to_dict()
    logger().info(
        "Feature extraction complete | rows=%d | status_counts=%s | failures=%d",
        len(features_df),
        status_counts,
        len(failures),
    )
    return outputs


def collect_feature_result(
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    progress: ProgressLogger,
    completed: int,
    success_count: int,
    failed_count: int,
) -> tuple[int, int]:
    if result["feature_status"] == "success":
        success_count += 1
    else:
        failed_count += 1
        failures.append({"image_id": result["image_id"], "dataset": result["dataset"], "reason": result["failure_reason"]})
        logger().warning(
            "Feature extraction failed | image_id=%s | dataset=%s | reason=%s",
            result["image_id"],
            result["dataset"],
            result["failure_reason"],
        )
    rows.append(result)
    progress.log(
        completed,
        success=success_count,
        failed=failed_count,
        current=str(result["image_id"]),
        extra=f"item_seconds={result.pop('_item_seconds'):.1f}",
    )
    return success_count, failed_count


def extract_feature_row(record: dict[str, Any], *, config_hash: str | None, schema_version: str) -> dict[str, Any]:
    item_started = time.perf_counter()
    base = {
        "_row_order": int(record["_row_order"]),
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "weight_kg": record["weight_kg"],
        "data_version": record.get("data_version"),
        "feature_schema_version": schema_version,
        "config_hash": config_hash,
    }
    try:
        if record.get("segmentation_status") != "success":
            raise RuntimeError("segmentation_missing")
        if record.get("keypoint_status") != "success":
            raise RuntimeError("keypoints_missing")
        if record.get("pointcloud_status") != "success":
            raise RuntimeError("pointcloud_missing")
        mask = load_mask(record["mask_path"])
        contour = contour_from_mask(mask)
        keypoint_array = load_keypoint_array(record["keypoint_path"])
        pc = np.load(record["pointcloud_path"])
        features = compute_all_features(mask, contour, keypoint_array, pc)
        status = "success"
        reason = None
    except Exception as exc:
        features = {name: np.nan for name in PAPER_FEATURES}
        status = "failed"
        reason = str(exc)
    return {
        **base,
        "feature_status": status,
        "failure_reason": reason,
        **features,
        "_item_seconds": time.perf_counter() - item_started,
    }


def assert_feature_inputs_ready(
    config: dict[str, Any],
    cv: pd.DataFrame,
    keypoints: pd.DataFrame,
    depth: pd.DataFrame,
    pointcloud: pd.DataFrame,
) -> None:
    config_hash = config.get("_config_hash")
    cv_success = cv[cv["segmentation_status"] == "success"]
    if cv_success.empty:
        raise RuntimeError("Feature extraction requires successful segmentation rows.")
    if should_require_pvtv2_inference(config):
        expected_source = required_pvtv2_segmentation_source(config)
        invalid_sources = sorted(set(cv_success.loc[cv_success["segmentation_source"] != expected_source, "segmentation_source"].astype(str)))
        if invalid_sources:
            raise RuntimeError(f"Feature extraction requires frozen PVTv2-B2 saliency masks; found sources={invalid_sources}.")
    invalid_seg_hash = sorted(set(cv_success.loc[cv_success["config_hash"] != config_hash, "config_hash"].astype(str)))
    if invalid_seg_hash:
        raise RuntimeError(f"Feature extraction requires current-config segmentation artifacts; found config_hash={invalid_seg_hash}.")

    depth_success = depth[depth["depth_status"] == "success"]
    if depth_success.empty:
        raise RuntimeError("Feature extraction requires successful depth rows.")
    if config.get("pointcloud", {}).get("require_depth_pro", True):
        invalid_depth = sorted(set(depth_success.loc[depth_success["depth_provider"] != "depth_pro", "depth_provider"].astype(str)))
        if invalid_depth:
            raise RuntimeError(f"Feature extraction requires depth_provider=depth_pro; found providers={invalid_depth}.")
    invalid_depth_hash = sorted(set(depth_success.loc[depth_success["config_hash"] != config_hash, "config_hash"].astype(str)))
    if invalid_depth_hash:
        raise RuntimeError(f"Feature extraction requires current-config depth artifacts; found config_hash={invalid_depth_hash}.")

    keypoint_success = keypoints[keypoints["keypoint_status"] == "success"]
    pointcloud_success = pointcloud[pointcloud["pointcloud_status"] == "success"]
    if keypoint_success.empty:
        raise RuntimeError("Feature extraction requires successful keypoint rows.")
    if pointcloud_success.empty:
        raise RuntimeError("Feature extraction requires successful point-cloud rows.")
    invalid_keypoint_hash = sorted(set(keypoint_success.loc[keypoint_success["config_hash"] != config_hash, "config_hash"].astype(str)))
    if invalid_keypoint_hash:
        raise RuntimeError(f"Feature extraction requires current-config keypoint artifacts; found config_hash={invalid_keypoint_hash}.")
    invalid_pointcloud_hash = sorted(set(pointcloud_success.loc[pointcloud_success["config_hash"] != config_hash, "config_hash"].astype(str)))
    if invalid_pointcloud_hash:
        raise RuntimeError(f"Feature extraction requires current-config point-cloud artifacts; found config_hash={invalid_pointcloud_hash}.")


def write_schema(config: dict[str, Any]) -> None:
    specs = [
        {
            "name": spec.name,
            "family": spec.family,
            "units": spec.units,
            "coordinate_frame": spec.coordinate_frame,
            "source_artifact": spec.source_artifact,
            "paper_derived": spec.paper_derived,
            "missingness_rule": spec.missingness_rule,
        }
        for spec in build_feature_specs()
    ]
    write_yaml({"schema_version": config.get("features", {}).get("schema_version", "paper_84_v1"), "features": specs}, registry_dir(config) / "feature_schema.yaml")


def load_mask(path: str | Path) -> np.ndarray:
    if not path:
        raise FileNotFoundError("Mask path is missing")
    return np.asarray(Image.open(path).convert("L")) > 0


def load_keypoint_array(path: str | Path) -> np.ndarray:
    payload = read_json(Path(path))
    keypoints = np.asarray(payload["keypoints"], dtype=float)
    if keypoints.shape != (20, 3):
        raise ValueError(f"Expected AnimalPose keypoints with shape (20, 3), got {keypoints.shape}")
    return keypoints


def compute_all_features(mask: np.ndarray, contour: np.ndarray, keypoints: np.ndarray, pointcloud: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(body_measurement_features(keypoints))
    features.update(basic_contour_features(mask, contour))
    features.update(hu_moment_features(mask))
    features.update(prefix_stats("mask_centroid_dist", centroid_distances(contour)))
    features.update(prefix_stats("pc_centroid_dist", centroid_distances(pointcloud)))
    features.update(mask_convex_hull_features(mask, contour))
    features.update(pointcloud_convex_hull_features(pointcloud))
    features.update(prefix_stats("mask_beam_angle", beam_angles(contour)))
    features.update(prefix_stats("pc_beam_angle", beam_angles(pointcloud[:, :2] if len(pointcloud) else np.empty((0, 2)))))
    features.update(prefix_stats("mask_chain_code", chain_codes(contour)))
    features.update(prefix_stats("pc_chain_code", chain_codes(pointcloud[:, :2] if len(pointcloud) else np.empty((0, 2)))))
    features.update(keypoint_distance_features(keypoints))
    missing = sorted(set(PAPER_FEATURES) - set(features))
    if missing:
        raise AssertionError(f"Feature implementation missing columns: {missing}")
    return {name: float(features[name]) for name in PAPER_FEATURES}


def body_measurement_features(keypoints: np.ndarray) -> dict[str, float]:
    nose = keypoints[4, :2]
    throat = keypoints[5, :2]
    tailbase = keypoints[6, :2]
    withers = keypoints[7, :2]
    front_paws = keypoints[[16, 17], :2]
    back_paws = keypoints[[18, 19], :2]
    front_elbows = keypoints[[8, 9], :2]
    body_length = euclidean(nose, tailbase)
    tail_to_neck = euclidean(throat, tailbase)
    withers_height = float(np.min(np.abs(front_paws[:, 1] - withers[1])))
    hip_height = float(np.min(np.abs(back_paws[:, 1] - tailbase[1])))
    elbow_midpoint = front_elbows.mean(axis=0)
    heart_girth = euclidean(withers, elbow_midpoint)
    cylinder_height = euclidean(tailbase, withers)
    cylinder_volume = math.pi * (heart_girth / 2.0) ** 2 * cylinder_height
    return {
        "bm_heart_girth_px": heart_girth,
        "bm_withers_height_px": withers_height,
        "bm_hip_height_px": hip_height,
        "bm_tail_distance_to_neck_px": tail_to_neck,
        "bm_body_length_px": body_length,
        "bm_cylinder_volume_px3": cylinder_volume,
    }


def basic_contour_features(mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    perimeter = ordered_path_length(contour)
    return {
        "bc_bbox_width_px": float(x1 - x0 + 1) if not np.isnan(x0) else np.nan,
        "bc_bbox_height_px": float(y1 - y0 + 1) if not np.isnan(y0) else np.nan,
        "bc_contour_perimeter_px": perimeter,
    }


def hu_moment_features(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {f"hm_{idx}": np.nan for idx in range(1, 8)}
    x = xs.astype(float)
    y = ys.astype(float)
    m00 = float(len(x))
    xbar = float(x.mean())
    ybar = float(y.mean())

    def mu(p: int, q: int) -> float:
        return float(((x - xbar) ** p * (y - ybar) ** q).sum())

    def eta(p: int, q: int) -> float:
        return mu(p, q) / (m00 ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    hu = [
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ]
    return {f"hm_{idx}": signed_log_hu(value) for idx, value in enumerate(hu, start=1)}


def signed_log_hu(value: float) -> float:
    if value == 0:
        return 0.0
    return float(-math.copysign(1.0, value) * math.log10(abs(value)))


def centroid_distances(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=float)
    centroid = points.mean(axis=0)
    return np.linalg.norm(points - centroid, axis=1)


def prefix_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {f"{prefix}_{name}": np.nan for name in STATS}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def mask_convex_hull_features(mask: np.ndarray, contour: np.ndarray) -> dict[str, float]:
    area, perimeter = hull_area_perimeter(contour)
    mask_area = float(mask.sum())
    return {
        "mask_hull_area_px2": area,
        "mask_hull_perimeter_px": perimeter,
        "mask_solidity": mask_area / area if area and not np.isnan(area) else np.nan,
    }


def pointcloud_convex_hull_features(pointcloud: np.ndarray) -> dict[str, float]:
    if len(pointcloud) == 0:
        return {"pc_hull_area": np.nan, "pc_hull_perimeter": np.nan, "pc_solidity": np.nan}
    projected = pointcloud[:, :2]
    area, perimeter = hull_area_perimeter(projected)
    unique_xy = len(np.unique(np.round(projected, decimals=4), axis=0))
    solidity = unique_xy / area if area and not np.isnan(area) else np.nan
    return {"pc_hull_area": area, "pc_hull_perimeter": perimeter, "pc_solidity": float(solidity)}


def hull_area_perimeter(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return (np.nan, np.nan)
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError:
        return (np.nan, np.nan)
    return float(hull.volume), float(hull.area)


def beam_angles(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.asarray([], dtype=float)
    ordered = order_points_by_angle(points[:, :2])
    centroid = ordered.mean(axis=0)
    vectors = ordered - centroid
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    diffs = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return np.degrees(np.abs(diffs))


def chain_codes(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=float)
    ordered = order_points_by_angle(points[:, :2])
    deltas = np.diff(np.vstack([ordered, ordered[0]]), axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    codes = np.mod(np.round((angles + math.pi) / (math.pi / 4.0)), 8)
    return codes.astype(float)


def ordered_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.nan
    ordered = order_points_by_angle(points[:, :2])
    deltas = np.diff(np.vstack([ordered, ordered[0]]), axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def order_points_by_angle(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def keypoint_distance_features(keypoints: np.ndarray) -> dict[str, float]:
    features = {}
    for name, idx_a, idx_b in KEYPOINT_LINKS:
        features[name] = euclidean(keypoints[idx_a, :2], keypoints[idx_b, :2])
    return features


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
