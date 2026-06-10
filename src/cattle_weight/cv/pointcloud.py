from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cattle_weight.cv.segmentation import required_pvtv2_segmentation_source, should_require_pvtv2_inference
from cattle_weight.io import artifacts_dir, read_parquet, registry_dir, write_parquet
from cattle_weight.progress import ProgressLogger, logger


def depth_to_point_cloud(depth: np.ndarray, mask: np.ndarray, focal_length_px: float) -> np.ndarray:
    if focal_length_px <= 0 or np.isnan(focal_length_px):
        raise ValueError(f"Invalid focal length: {focal_length_px}")
    height, width = depth.shape
    ys, xs = np.where(mask & (depth > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float64)
    cx = width / 2.0
    cy = height / 2.0
    x = (xs - cx) * z / focal_length_px
    y = (ys - cy) * z / focal_length_px
    return np.column_stack([x, y, z]).astype(np.float32)


def cleanup_point_cloud(points: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    counts = {
        "point_count_after_mask": int(len(points)),
        "point_count_after_downsample": 0,
        "point_count_after_z_cleanup": 0,
    }
    if len(points) == 0:
        return points, counts
    cleaned = points.copy()
    cleaned[:, 2] = -cleaned[:, 2]
    cleaned[:, 2] = cleaned[:, 2] - cleaned[:, 2].min()
    rng = np.random.default_rng(seed)
    sample_count = max(1, int(round(len(cleaned) * fraction)))
    sample_count = min(sample_count, len(cleaned))
    idx = rng.choice(len(cleaned), size=sample_count, replace=False)
    cleaned = cleaned[np.sort(idx)]
    counts["point_count_after_downsample"] = int(len(cleaned))
    if len(cleaned) >= 3:
        threshold = otsu_threshold(cleaned[:, 2])
        low_cluster = cleaned[cleaned[:, 2] <= threshold]
        high_cluster = cleaned[cleaned[:, 2] > threshold]
        if len(low_cluster) >= max(3, len(high_cluster)):
            cleaned = low_cluster
        elif len(high_cluster) >= 3:
            cleaned = high_cluster
    counts["point_count_after_z_cleanup"] = int(len(cleaned))
    return cleaned.astype(np.float32), counts


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    if len(values) == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=min(bins, max(2, len(values))))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    sum_total = float((hist * centers).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_threshold = float(centers[0])
    for count, center in zip(hist, centers):
        weight_bg += float(count)
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += float(count) * float(center)
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_threshold = float(center)
    return best_threshold


def run_pointcloud(config: dict[str, Any]) -> pd.DataFrame:
    cv = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    depth = read_parquet(registry_dir(config) / "depth_metadata.parquet")
    require_pvtv2 = should_require_pvtv2_inference(config)
    if require_pvtv2:
        assert_pvtv2_pointcloud_inputs(cv)
    merged = cv.merge(depth[["image_id", "depth_map_path", "focal_length_px", "depth_status", "depth_provider"]], on="image_id", how="left")
    seed = int(config.get("run", {}).get("random_seed", 0))
    fraction = float(config.get("pointcloud", {}).get("downsample_fraction", 0.02))
    pc_cfg = config.get("pointcloud", {})
    metadata_path = registry_dir(config) / "pointcloud_metadata.parquet"
    checkpoint_every = int(pc_cfg.get("checkpoint_every", 100))
    current_image_ids = set(merged["image_id"].astype(str))
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    overwrite_existing = bool(pc_cfg.get("overwrite_existing", False))
    require_depth_pro = bool(pc_cfg.get("require_depth_pro", config.get("depth", {}).get("provider") == "depth_pro" and not config.get("depth", {}).get("allow_fallback", False)))
    if metadata_path.exists() and not overwrite_existing:
        previous = pd.read_parquet(metadata_path)
        current_rows = {str(item["image_id"]): item for item in merged.to_dict(orient="records")}
        for item in previous.to_dict("records"):
            image_id = str(item.get("image_id"))
            pointcloud_path = item.get("pointcloud_path")
            if (
                image_id in current_image_ids
                and item.get("pointcloud_status") == "success"
                and pointcloud_path
                and Path(pointcloud_path).exists()
                and is_trusted_pointcloud_resume_row(
                    item,
                    current_rows.get(image_id),
                    config,
                    require_pvtv2=require_pvtv2,
                    require_depth_pro=require_depth_pro,
                )
            ):
                rows.append(item)
                completed.add(image_id)
    processed = 0
    success_count = sum(1 for item in rows if item.get("pointcloud_status") == "success")
    failed_count = sum(1 for item in rows if item.get("pointcloud_status") == "failed")
    skipped_count = len(completed)
    log_every = int(config.get("pointcloud", {}).get("log_every", checkpoint_every or 100))
    progress = ProgressLogger("pointcloud", total=len(merged), log_every=log_every)
    logger().info(
        "Pointcloud setup | total_rows=%d | existing_success=%d | to_process=%d | downsample_fraction=%s | "
        "checkpoint_every=%d | log_every=%d | metadata=%s",
        len(merged),
        len(completed),
        len(merged) - len(completed),
        fraction,
        checkpoint_every,
        log_every,
        metadata_path,
    )
    if completed:
        progress.log(
            skipped_count,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            extra="resumed_existing_success",
            force=True,
        )
    for row_idx, row in merged.iterrows():
        image_id = row["image_id"]
        if image_id in completed:
            continue
        dataset = row["dataset"]
        output_path = artifacts_dir(config) / "pointclouds" / dataset / f"{safe_name(image_id)}_pointcloud.npy"
        mask_size = 0
        item_started = time.perf_counter()
        extra = None
        try:
            if row["segmentation_status"] != "success" or row["depth_status"] != "success":
                raise RuntimeError("segmentation_or_depth_failed")
            if require_pvtv2 and row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
                raise RuntimeError(f"pointcloud_requires_pvtv2_mask got {row.get('segmentation_source')}")
            if require_depth_pro and row.get("depth_provider") != "depth_pro":
                raise RuntimeError(f"pointcloud_requires_depth_pro got {row.get('depth_provider')}")
            mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 0
            mask_size = int(mask.size)
            depth_map = np.load(row["depth_map_path"])
            if mask.shape != depth_map.shape:
                resized = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                    (depth_map.shape[1], depth_map.shape[0]),
                    Image.Resampling.NEAREST,
                )
                mask = np.asarray(resized) > 0
            points = depth_to_point_cloud(depth_map, mask, float(row["focal_length_px"]))
            cleaned, counts = cleanup_point_cloud(points, fraction, seed + int(row_idx))
            if len(cleaned) == 0:
                raise RuntimeError("empty_pointcloud_after_cleanup")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, cleaned)
            status = "success"
            reason = None
            file_kb = output_path.stat().st_size / 1024
            extra = (
                f"points_mask={counts['point_count_after_mask']} | "
                f"points_downsample={counts['point_count_after_downsample']} | "
                f"points_final={counts['point_count_after_z_cleanup']} | file_kb={file_kb:.1f}"
            )
        except Exception as exc:
            output_path = None
            counts = {"point_count_after_mask": 0, "point_count_after_downsample": 0, "point_count_after_z_cleanup": 0}
            status = "failed"
            reason = str(exc)
            failed_count += 1
            logger().warning("Pointcloud failed | image_id=%s | dataset=%s | reason=%s", image_id, dataset, reason)
        else:
            success_count += 1
        rows.append(
            {
                "image_id": image_id,
                "dataset": dataset,
                "dataset_version": row.get("dataset_version"),
                "transform_version": row.get("transform_version"),
                "config_hash": config.get("_config_hash"),
                "provider": row.get("depth_provider"),
                "pointcloud_status": status,
                "failure_reason": reason,
                "pointcloud_path": str(output_path.resolve()) if output_path else None,
                "point_count_before_mask": mask_size,
                **counts,
            }
        )
        processed += 1
        item_seconds = time.perf_counter() - item_started
        detail = f"item_seconds={item_seconds:.1f}"
        if extra:
            detail = f"{detail} | {extra}"
        progress.log(
            skipped_count + processed,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            current=str(image_id),
            extra=detail,
        )
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            write_parquet(pd.DataFrame(rows), metadata_path)
            logger().info(
                "Pointcloud checkpoint written | processed_this_run=%d | total_rows=%d | metadata=%s",
                processed,
                len(rows),
                metadata_path,
            )
    out = pd.DataFrame(rows)
    write_parquet(out, metadata_path)
    status_counts = out["pointcloud_status"].value_counts(dropna=False).to_dict() if "pointcloud_status" in out else {}
    logger().info("Pointcloud complete | rows=%d | status_counts=%s | metadata=%s", len(out), status_counts, metadata_path)
    return out


def assert_pvtv2_pointcloud_inputs(cv: pd.DataFrame) -> None:
    success = cv[cv["segmentation_status"] == "success"]
    if success.empty:
        raise RuntimeError("Point-cloud generation requires successful PVTv2 segmentation rows.")
    expected_source = required_pvtv2_segmentation_source()
    invalid_sources = sorted(set(success.loc[success["segmentation_source"] != expected_source, "segmentation_source"].astype(str)))
    if invalid_sources:
        raise RuntimeError(f"Point-cloud generation requires frozen PVTv2-B2 saliency masks; found sources={invalid_sources}.")


def is_trusted_pointcloud_resume_row(
    pointcloud_row: dict[str, Any],
    current_row: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    require_pvtv2: bool,
    require_depth_pro: bool,
) -> bool:
    if current_row is None:
        return False
    pointcloud_path = pointcloud_row.get("pointcloud_path")
    if not pointcloud_path or not Path(str(pointcloud_path)).exists():
        return False
    if pointcloud_row.get("pointcloud_status") != "success":
        return False
    if pointcloud_row.get("config_hash") != config.get("_config_hash") and not row_is_reusable_by_stable_input(pointcloud_row):
        return False
    if require_pvtv2 and current_row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
        return False
    if require_depth_pro and current_row.get("depth_provider") != "depth_pro":
        return False
    return True


def row_is_reusable_by_stable_input(row: dict[str, Any]) -> bool:
    marker = row.get("reusable_by_stable_input_config", False)
    stable_hash = row.get("stable_input_config_hash")
    return bool(marker) and stable_hash not in {None, "", "nan"}


def safe_name(image_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in image_id)
