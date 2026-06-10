from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from cattle_weight.features.schema import ANIMALPOSE_KEYPOINTS, KEYPOINT_LINKS
from cattle_weight.config import REPO_ROOT
from cattle_weight.cv.animalpose_direct import infer_direct_animalpose, infer_direct_animalpose_batch
from cattle_weight.cv.segmentation import required_pvtv2_segmentation_source, should_require_pvtv2_inference
from cattle_weight.io import artifacts_dir, read_parquet, registry_dir, reports_dir, write_json, write_parquet


DIRECT_ANIMALPOSE_PROVIDER = "mmpose_checkpoint_direct_simplebaseline"


def validate_keypoints(keypoints: np.ndarray) -> None:
    if keypoints.shape != (20, 3):
        raise ValueError(f"AnimalPose keypoints must have shape (20, 3), got {keypoints.shape}")


def assert_pvtv2_keypoint_inputs(artifacts: pd.DataFrame) -> None:
    success = artifacts[artifacts["segmentation_status"] == "success"]
    if success.empty:
        raise RuntimeError("Keypoint inference requires successful PVTv2 segmentation rows.")
    expected_source = required_pvtv2_segmentation_source()
    invalid_sources = sorted(set(success.loc[success["segmentation_source"] != expected_source, "segmentation_source"].astype(str)))
    if invalid_sources:
        raise RuntimeError(f"Keypoint inference requires frozen PVTv2-B2 saliency masks; found sources={invalid_sources}.")


def direct_checkpoint_required(kp_cfg: dict[str, Any]) -> bool:
    return bool(kp_cfg.get("provider") == "mmpose" and kp_cfg.get("force_direct_checkpoint", False) and not kp_cfg.get("allow_fallback", False))


def is_trusted_keypoint_resume_row(
    row: dict[str, Any],
    config: dict[str, Any],
    current_image_ids: set[str],
    expected_provider: str | None = None,
) -> bool:
    image_id = str(row.get("image_id"))
    if image_id not in current_image_ids:
        return False
    if row.get("keypoint_status") != "success":
        return False
    if expected_provider and row.get("provider") != expected_provider:
        return False
    path = row.get("keypoint_path")
    if not path or not Path(str(path)).exists():
        return False
    if row.get("config_hash") == config.get("_config_hash"):
        return True
    return row_is_reusable_by_stable_input(row)


def row_is_reusable_by_stable_input(row: dict[str, Any]) -> bool:
    marker = row.get("reusable_by_stable_input_config", False)
    stable_hash = row.get("stable_input_config_hash")
    return bool(marker) and stable_hash not in {None, "", "nan"}


def run_keypoints(config: dict[str, Any]) -> pd.DataFrame:
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    artifacts = artifacts.merge(
        sticker[["image_id", "normalized_image_path", "raw_image_path", "scale"]],
        on="image_id",
        how="left",
    )
    require_pvtv2 = should_require_pvtv2_inference(config)
    if require_pvtv2:
        assert_pvtv2_keypoint_inputs(artifacts)
    kp_cfg = config.get("keypoints", {})
    metadata_path = registry_dir(config) / "keypoint_metadata.parquet"
    current_image_ids = set(artifacts["image_id"].astype(str))
    overwrite_existing = bool(kp_cfg.get("overwrite_existing", False))
    existing_rows: dict[str, dict[str, Any]] = {}
    expected_provider = DIRECT_ANIMALPOSE_PROVIDER if direct_checkpoint_required(kp_cfg) else None
    if metadata_path.exists() and not overwrite_existing:
        existing = read_parquet(metadata_path)
        for item in existing.to_dict(orient="records"):
            image_id = str(item["image_id"])
            if is_trusted_keypoint_resume_row(item, config, current_image_ids, expected_provider):
                existing_rows[image_id] = item
    rows: list[dict[str, Any]] = list(existing_rows.values())
    processed = 0
    checkpoint_every = int(kp_cfg.get("checkpoint_every", 100))
    if kp_cfg.get("provider") == "mmpose" and kp_cfg.get("force_direct_checkpoint", False):
        return run_direct_keypoints_batched(config, artifacts, existing_rows, rows, metadata_path)
    for _, row in artifacts.iterrows():
        image_id = row["image_id"]
        if image_id in existing_rows:
            continue
        dataset = row["dataset"]
        output_path = artifacts_dir(config) / "keypoints" / dataset / f"{safe_name(image_id)}.json"
        try:
            if row["segmentation_status"] != "success":
                raise RuntimeError("segmentation_failed")
            if require_pvtv2 and row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
                raise RuntimeError(f"keypoints_require_pvtv2_mask got {row.get('segmentation_source')}")
            if output_path.exists() and not overwrite_existing:
                payload = read_existing_keypoints(output_path)
                keypoints = np.asarray(payload["keypoints"], dtype=float)
                provider = str(payload.get("provider", "existing_keypoints"))
                provider_status = str(payload.get("provider_status", "existing"))
            elif kp_cfg.get("provider") == "mmpose" and kp_cfg.get("force_direct_checkpoint", False):
                keypoints = infer_direct_mmpose_checkpoint(config, row)
                provider = "mmpose_checkpoint_direct_simplebaseline"
                provider_status = "mmpose_api_bypassed_by_config"
            elif kp_cfg.get("provider") == "mmpose" and not kp_cfg.get("allow_fallback", False):
                try:
                    keypoints = infer_mmpose_keypoints(config, row)
                    provider = "mmpose_animalpose"
                    provider_status = "full_provider"
                except RuntimeError as exc:
                    if "mmcv._ext" not in str(exc) and "MultiScaleDeformableAttention" not in str(exc):
                        raise
                    keypoints = infer_direct_mmpose_checkpoint(config, row)
                    provider = "mmpose_checkpoint_direct_simplebaseline"
                    provider_status = "mmpose_api_blocked_by_mmcv_ext"
            else:
                keypoints = fallback_keypoints_from_bbox(row)
                provider = "fallback_bbox_geometry"
                provider_status = "fallback_partial_reproduction"
            validate_keypoints(keypoints)
            payload = {
                "image_id": image_id,
                "provider": provider,
                "provider_status": provider_status,
                "keypoint_names": ANIMALPOSE_KEYPOINTS,
                "keypoints": keypoints.tolist(),
            }
            write_json(payload, output_path)
            status = "success"
            reason = None
        except Exception as exc:
            status = "failed"
            provider = kp_cfg.get("provider", "mmpose")
            reason = str(exc)
            output_path = None
        rows.append(
            {
                "image_id": image_id,
                "dataset": dataset,
                "dataset_version": row.get("dataset_version"),
                "transform_version": row.get("transform_version"),
                "config_hash": config.get("_config_hash"),
                "provider": provider,
                "keypoint_status": status,
                "failure_reason": reason,
                "keypoint_path": str(output_path.resolve()) if output_path else None,
            }
        )
        processed += 1
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            write_parquet(pd.DataFrame(rows), metadata_path)
    out = pd.DataFrame(rows)
    out = out.drop_duplicates("image_id", keep="last").reset_index(drop=True)
    write_parquet(out, metadata_path)
    write_keypoint_review_overlays(config, artifacts, out)
    return out


def run_direct_keypoints_batched(
    config: dict[str, Any],
    artifacts: pd.DataFrame,
    existing_rows: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    metadata_path: Path,
) -> pd.DataFrame:
    kp_cfg = config.get("keypoints", {})
    require_pvtv2 = should_require_pvtv2_inference(config)
    batch_size = int(kp_cfg.get("batch_size", 16))
    checkpoint_every = int(kp_cfg.get("checkpoint_every", 100))
    pending: list[pd.Series] = []
    processed = 0

    def flush() -> None:
        nonlocal pending, processed, rows
        if not pending:
            return
        valid_rows: list[pd.Series] = []
        image_paths: list[str] = []
        bboxes: list[np.ndarray] = []
        use_raw = bool(kp_cfg.get("use_raw_image_for_inference", False))
        scales: list[float] = []
        overwrite_existing = bool(kp_cfg.get("overwrite_existing", False))
        for item in pending:
            output_path = artifacts_dir(config) / "keypoints" / item["dataset"] / f"{safe_name(item['image_id'])}.json"
            if output_path.exists() and not overwrite_existing and not direct_checkpoint_required(kp_cfg):
                payload = read_existing_keypoints(output_path)
                keypoints = np.asarray(payload["keypoints"], dtype=float)
                append_keypoint_row(rows, config, item, output_path, str(payload.get("provider", "existing_keypoints")), "existing", "success", None)
                processed += 1
                continue
            valid_rows.append(item)
            scale = float(item.get("scale", 1.0) or 1.0)
            if use_raw and pd.notna(item.get("raw_image_path")) and scale > 0:
                image_paths.append(str(item["raw_image_path"]))
                bboxes.append(np.array([item["bbox_x0"], item["bbox_y0"], item["bbox_x1"], item["bbox_y1"]], dtype=float) / scale)
                scales.append(scale)
            else:
                image_paths.append(str(item["normalized_image_path"]))
                bboxes.append(np.array([item["bbox_x0"], item["bbox_y0"], item["bbox_x1"], item["bbox_y1"]], dtype=float))
                scales.append(1.0)
        if valid_rows:
            try:
                predictions = infer_direct_animalpose_batch(
                    image_paths,
                    bboxes,
                    str(kp_cfg["checkpoint_path"]),
                    str(kp_cfg.get("device", "cpu")),
                    batch_size=batch_size,
                )
            except Exception as exc:
                for item in valid_rows:
                    append_keypoint_row(rows, config, item, None, "mmpose_checkpoint_direct_simplebaseline", "batch_failed", "failed", str(exc))
                    processed += 1
            else:
                for item, keypoints, scale in zip(valid_rows, predictions, scales):
                    keypoints = keypoints.copy()
                    keypoints[:, :2] *= scale
                    output_path = artifacts_dir(config) / "keypoints" / item["dataset"] / f"{safe_name(item['image_id'])}.json"
                    validate_keypoints(keypoints)
                    payload = {
                        "image_id": item["image_id"],
                        "provider": "mmpose_checkpoint_direct_simplebaseline",
                        "provider_status": "mmpose_api_bypassed_by_config",
                        "keypoint_names": ANIMALPOSE_KEYPOINTS,
                        "keypoints": keypoints.tolist(),
                    }
                    write_json(payload, output_path)
                    append_keypoint_row(rows, config, item, output_path, "mmpose_checkpoint_direct_simplebaseline", "mmpose_api_bypassed_by_config", "success", None)
                    processed += 1
        pending = []
        if checkpoint_every > 0 and processed % checkpoint_every < batch_size:
            write_parquet(pd.DataFrame(rows).drop_duplicates("image_id", keep="last"), metadata_path)

    for _, row in artifacts.iterrows():
        image_id = row["image_id"]
        if image_id in existing_rows:
            continue
        if row["segmentation_status"] != "success":
            append_keypoint_row(rows, config, row, None, "mmpose_checkpoint_direct_simplebaseline", "segmentation_failed", "failed", "segmentation_failed")
            processed += 1
            continue
        if require_pvtv2 and row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
            append_keypoint_row(
                rows,
                config,
                row,
                None,
                "mmpose_checkpoint_direct_simplebaseline",
                "invalid_segmentation_source",
                "failed",
                f"keypoints_require_pvtv2_mask got {row.get('segmentation_source')}",
            )
            processed += 1
            continue
        if pd.isna(row.get("normalized_image_path")):
            append_keypoint_row(rows, config, row, None, "mmpose_checkpoint_direct_simplebaseline", "missing_image", "failed", "normalized_image_path_missing")
            processed += 1
            continue
        pending.append(row)
        if len(pending) >= batch_size:
            flush()
    flush()
    out = pd.DataFrame(rows).drop_duplicates("image_id", keep="last").reset_index(drop=True)
    write_parquet(out, metadata_path)
    write_keypoint_review_overlays(config, artifacts, out)
    return out


def write_keypoint_review_overlays(config: dict[str, Any], artifacts: pd.DataFrame, keypoint_metadata: pd.DataFrame) -> pd.DataFrame:
    candidates = build_keypoint_review_candidates(config, artifacts, keypoint_metadata)
    if candidates.empty:
        return candidates
    output_root = reports_dir(config) / "keypoint_review"
    output_root.mkdir(parents=True, exist_ok=True)
    max_overlay_side = int(config.get("keypoints", {}).get("review", {}).get("max_overlay_side", 1600))
    rows: list[dict[str, Any]] = []
    for item in candidates.to_dict(orient="records"):
        category_dir = output_root / str(item["review_category"])
        category_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = category_dir / f"{int(item['selection_rank']):02d}_{safe_name(str(item['image_id']))}.png"
        write_keypoint_overlay(item["normalized_image_path"], item["keypoint_path"], overlay_path, max_overlay_side, item)
        rows.append({**item, "overlay_path": str(overlay_path.resolve())})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "keypoint_review_manifest.csv", index=False)
    return manifest


def build_keypoint_review_candidates(config: dict[str, Any], artifacts: pd.DataFrame, keypoint_metadata: pd.DataFrame) -> pd.DataFrame:
    review_cfg = config.get("keypoints", {}).get("review", {})
    random_count = int(review_cfg.get("random_count", 10))
    anomaly_count = int(review_cfg.get("area_anomaly_count", 10))
    complexity_count = int(review_cfg.get("complexity_count", 10))
    low_confidence_count = int(review_cfg.get("low_confidence_count", 10))
    seed = int(config.get("run", {}).get("random_seed", 0))

    keypoint_review = keypoint_metadata[["image_id", "keypoint_status", "provider", "keypoint_path"]].rename(
        columns={"provider": "keypoint_provider"}
    )
    merged = artifacts.merge(
        keypoint_review,
        on="image_id",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if row.get("segmentation_status") != "success" or row.get("keypoint_status") != "success":
            continue
        image_path = row.get("normalized_image_path")
        keypoint_path = row.get("keypoint_path")
        if not image_path or pd.isna(image_path) or not Path(str(image_path)).exists():
            continue
        if not keypoint_path or pd.isna(keypoint_path) or not Path(str(keypoint_path)).exists():
            continue
        payload = read_existing_keypoints(Path(str(keypoint_path)))
        keypoints = np.asarray(payload["keypoints"], dtype=float)
        try:
            validate_keypoints(keypoints)
        except ValueError:
            continue
        area = float(row.get("mask_area_px", 0) or 0)
        contour_count = float(row.get("contour_point_count", 0) or 0)
        rows.append(
            {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "provider": row.get("keypoint_provider"),
                "normalized_image_path": str(image_path),
                "keypoint_path": str(keypoint_path),
                "bbox_x0": row.get("bbox_x0"),
                "bbox_y0": row.get("bbox_y0"),
                "bbox_x1": row.get("bbox_x1"),
                "bbox_y1": row.get("bbox_y1"),
                "mask_area_px": area,
                "complexity_score": contour_count / np.sqrt(area) if area > 0 else 0.0,
                "min_keypoint_confidence": float(np.nanmin(keypoints[:, 2])) if len(keypoints) else np.nan,
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["area_anomaly_score"] = robust_anomaly_score(candidates["mask_area_px"].to_numpy(dtype=float))

    selected_parts: list[pd.DataFrame] = []
    stable = candidates.sort_values(["dataset", "image_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if random_count > 0 and not stable.empty:
        positions = np.sort(rng.choice(len(stable), size=min(random_count, len(stable)), replace=False))
        selected_parts.append(with_keypoint_review_category(stable.iloc[positions], "random"))
    if anomaly_count > 0:
        selected_parts.append(
            with_keypoint_review_category(
                candidates.sort_values(["area_anomaly_score", "image_id"], ascending=[False, True]).head(anomaly_count),
                "area_anomaly",
            )
        )
    if complexity_count > 0:
        selected_parts.append(
            with_keypoint_review_category(
                candidates.sort_values(["complexity_score", "image_id"], ascending=[False, True]).head(complexity_count),
                "high_mask_complexity",
            )
        )
    if low_confidence_count > 0 and candidates["min_keypoint_confidence"].notna().any():
        selected_parts.append(
            with_keypoint_review_category(
                candidates.sort_values(["min_keypoint_confidence", "image_id"], ascending=[True, True]).head(low_confidence_count),
                "low_confidence",
            )
        )
    if not selected_parts:
        return candidates.iloc[0:0].copy()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_rank"] = selected.groupby("review_category").cumcount() + 1
    return selected.reset_index(drop=True)


def with_keypoint_review_category(rows: pd.DataFrame, category: str) -> pd.DataFrame:
    part = rows.copy().reset_index(drop=True)
    part["review_category"] = category
    return part


def robust_anomaly_score(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(len(values), dtype=float)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = mad * 1.4826 if mad > 0 else float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scores = np.abs(values - median) / scale
    scores[~np.isfinite(scores)] = 0.0
    return scores


def write_keypoint_overlay(
    image_path: str | Path,
    keypoint_path: str | Path,
    output_path: Path,
    max_side: int,
    metadata: dict[str, Any],
) -> None:
    image = Image.open(image_path).convert("RGB")
    keypoints = np.asarray(read_existing_keypoints(Path(keypoint_path))["keypoints"], dtype=float)
    scale = 1.0
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        image = image.resize(size, Image.Resampling.BILINEAR)
    keypoints = keypoints.copy()
    keypoints[:, :2] *= scale
    draw = ImageDraw.Draw(image)
    bbox_values = [metadata.get(name) for name in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]]
    if all(pd.notna(value) for value in bbox_values):
        x0, y0, x1, y1 = [float(value) * scale for value in bbox_values]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 190, 0), width=2)
    for _, idx_a, idx_b in KEYPOINT_LINKS:
        xa, ya = keypoints[idx_a, :2]
        xb, yb = keypoints[idx_b, :2]
        draw.line((float(xa), float(ya), float(xb), float(yb)), fill=(0, 180, 255), width=2)
    for idx, (x, y, score) in enumerate(keypoints):
        radius = 3 if score >= 0 else 2
        color = (255, 64, 64) if score < 0 else (20, 220, 120)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0))
        if idx < 10:
            draw.text((x + 4, y + 4), str(idx), fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def append_keypoint_row(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    row: pd.Series,
    output_path: Path | None,
    provider: str,
    provider_status: str,
    status: str,
    reason: str | None,
) -> None:
    rows.append(
        {
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "dataset_version": row.get("dataset_version"),
            "transform_version": row.get("transform_version"),
            "config_hash": config.get("_config_hash"),
            "provider": provider,
            "provider_status": provider_status,
            "keypoint_status": status,
            "failure_reason": reason,
            "keypoint_path": str(output_path.resolve()) if output_path else None,
        }
    )


def read_existing_keypoints(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_mmpose_keypoints(config: dict[str, Any], row: pd.Series) -> np.ndarray:
    kp_cfg = config.get("keypoints", {})
    config_path = kp_cfg.get("config_path")
    checkpoint_path = kp_cfg.get("checkpoint_path")
    if not config_path or not checkpoint_path:
        raise RuntimeError(
            "MMPose keypoints require keypoints.config_path and keypoints.checkpoint_path "
            "for an AnimalPose-compatible 20-keypoint model."
        )
    image_path = row.get("normalized_image_path")
    if pd.isna(image_path):
        raise RuntimeError("normalized_image_path is missing for MMPose inference.")
    model = get_mmpose_model(str(config_path), str(checkpoint_path), str(kp_cfg.get("device", "cpu")))
    bbox = np.array([[row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]]], dtype=np.float32)
    try:
        from mmpose.apis import inference_topdown
    except ImportError as exc:
        raise RuntimeError("MMPose is not installed. Install mmpose/mmcv or enable keypoints.allow_fallback.") from exc
    results = inference_topdown(model, str(image_path), bboxes=bbox)
    if not results:
        raise RuntimeError("MMPose produced no keypoint result.")
    pred = results[0].pred_instances
    points = np.asarray(pred.keypoints[0], dtype=float)
    if hasattr(pred, "keypoint_scores"):
        scores = np.asarray(pred.keypoint_scores[0], dtype=float)
    else:
        scores = np.ones(points.shape[0], dtype=float)
    if points.shape[0] != 20:
        raise RuntimeError(f"Expected 20 AnimalPose keypoints from MMPose, got {points.shape[0]}.")
    return np.column_stack([points[:, 0], points[:, 1], scores])


def infer_direct_mmpose_checkpoint(config: dict[str, Any], row: pd.Series) -> np.ndarray:
    kp_cfg = config.get("keypoints", {})
    checkpoint_path = kp_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("Direct AnimalPose checkpoint inference requires keypoints.checkpoint_path.")
    image_path = row.get("normalized_image_path")
    if pd.isna(image_path):
        raise RuntimeError("normalized_image_path is missing for direct AnimalPose inference.")
    bbox = np.array([row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]], dtype=float)
    return infer_direct_animalpose(str(image_path), bbox, str(checkpoint_path), str(kp_cfg.get("device", "cpu")))


@lru_cache(maxsize=2)
def get_mmpose_model(config_path: str, checkpoint_path: str, device: str) -> Any:
    try:
        from mmpose.apis import init_model
    except ImportError as exc:
        raise RuntimeError("MMPose is not installed. Install mmpose/mmcv or enable keypoints.allow_fallback.") from exc
    resolved_config = resolve_mmpose_config_path(config_path)
    resolved_checkpoint = resolve_model_path(checkpoint_path)
    if not resolved_config.exists():
        raise FileNotFoundError(f"MMPose config file does not exist: {resolved_config}")
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"MMPose checkpoint file does not exist: {resolved_checkpoint}")
    return init_model(str(resolved_config), str(resolved_checkpoint), device=device)


def resolve_mmpose_config_path(config_path: str) -> Path:
    if config_path == "mmpose://animalpose/resnet50":
        import mmpose

        return (
            Path(mmpose.__file__).resolve().parent
            / ".mim"
            / "configs"
            / "animal_2d_keypoint"
            / "topdown_heatmap"
            / "animalpose"
            / "td-hm_res50_8xb64-210e_animalpose-256x256.py"
        )
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_model_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def fallback_keypoints_from_bbox(row: pd.Series) -> np.ndarray:
    x0 = float(row["bbox_x0"])
    y0 = float(row["bbox_y0"])
    x1 = float(row["bbox_x1"])
    y1 = float(row["bbox_y1"])
    if any(np.isnan([x0, y0, x1, y1])):
        raise RuntimeError("bbox_missing")
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    points = [
        (x1 - 0.13 * w, y0 + 0.17 * h, 0.55),
        (x1 - 0.08 * w, y0 + 0.20 * h, 0.55),
        (x1 - 0.20 * w, y0 + 0.15 * h, 0.50),
        (x1 - 0.03 * w, y0 + 0.18 * h, 0.50),
        (x1 + 0.02 * w, y0 + 0.34 * h, 0.60),
        (x1 - 0.12 * w, y0 + 0.52 * h, 0.60),
        (x0 + 0.05 * w, y0 + 0.42 * h, 0.60),
        (x0 + 0.34 * w, y0 + 0.10 * h, 0.60),
        (x1 - 0.30 * w, y0 + 0.72 * h, 0.55),
        (x1 - 0.18 * w, y0 + 0.72 * h, 0.55),
        (x0 + 0.22 * w, y0 + 0.72 * h, 0.55),
        (x0 + 0.36 * w, y0 + 0.72 * h, 0.55),
        (x1 - 0.30 * w, y0 + 0.88 * h, 0.55),
        (x1 - 0.18 * w, y0 + 0.88 * h, 0.55),
        (x0 + 0.22 * w, y0 + 0.88 * h, 0.55),
        (x0 + 0.36 * w, y0 + 0.88 * h, 0.55),
        (x1 - 0.30 * w, y1, 0.55),
        (x1 - 0.18 * w, y1, 0.55),
        (x0 + 0.22 * w, y1, 0.55),
        (x0 + 0.36 * w, y1, 0.55),
    ]
    return np.asarray(points, dtype=float)


def safe_name(image_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in image_id)
