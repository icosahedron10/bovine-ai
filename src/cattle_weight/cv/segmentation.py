from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from cattle_weight.cv.pvtv2 import (
    PVT_SEGMENTATION_ALGORITHM_VERSION,
    PVT_SEGMENTATION_SOURCE,
    predict_pvtv2_b2_mask,
    predict_pvtv2_b2_masks,
)
from cattle_weight.cv.sticker import safe_stem
from cattle_weight.io import artifacts_dir, read_parquet, registry_dir, reports_dir, write_json, write_parquet
from cattle_weight.progress import ProgressLogger, logger


def required_pvtv2_segmentation_source(config: dict[str, Any] | None = None) -> str:
    return PVT_SEGMENTATION_SOURCE


def required_pvtv2_algorithm_version(config: dict[str, Any] | None = None) -> str:
    return PVT_SEGMENTATION_ALGORITHM_VERSION


def run_segmentation(config: dict[str, Any]) -> pd.DataFrame:
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    seg_cfg = config.get("segmentation", {})
    provider_name = str(seg_cfg.get("provider", "pvtv2_b2"))
    metadata_path = registry_dir(config) / "cv_artifacts.parquet"
    current_image_ids = set(sticker["image_id"].astype(str))
    overwrite_existing = bool(seg_cfg.get("overwrite_existing", False))
    require_pvtv2 = should_require_pvtv2_inference(config)
    existing_rows: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        existing = read_parquet(metadata_path).drop_duplicates("image_id", keep="last")
        for item in existing.to_dict(orient="records"):
            image_id = str(item["image_id"])
            if is_trusted_segmentation_resume_row(item, config, current_image_ids, require_pvtv2):
                existing_rows[image_id] = item
    rows: list[dict[str, Any]] = list(existing_rows.values())
    processed = 0
    checkpoint_every = int(config.get("segmentation", {}).get("checkpoint_every", 100))
    log_every = int(config.get("segmentation", {}).get("log_every", checkpoint_every or 100))
    batch_size = max(1, int(seg_cfg.get("frozen_pvtv2", {}).get("batch_size", seg_cfg.get("batch_size", 64))))
    postprocess_workers = max(1, int(seg_cfg.get("postprocess_workers", min(8, batch_size))))
    success_count = int(sum(1 for item in rows if item.get("segmentation_status") == "success"))
    failed_count = int(sum(1 for item in rows if item.get("segmentation_status") == "failed"))
    skipped_count = len(existing_rows)
    progress = ProgressLogger("segmentation", total=len(sticker), log_every=log_every)
    logger().info(
        "Segmentation setup | rows=%d | resumed_rows=%d | provider=%s | require_pvtv2=%s | "
        "overwrite_existing=%s | batch_size=%d | postprocess_workers=%d | checkpoint_every=%d | metadata=%s",
        len(sticker),
        skipped_count,
        provider_name,
        require_pvtv2,
        overwrite_existing,
        batch_size,
        postprocess_workers,
        checkpoint_every,
        metadata_path,
    )
    if skipped_count:
        progress.log(
            skipped_count,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            extra="resumed_existing_rows",
            force=True,
        )

    pending_pvtv2: list[pd.Series] = []

    def flush_pvtv2() -> None:
        nonlocal pending_pvtv2, processed, success_count, failed_count, rows
        if not pending_pvtv2:
            return
        if len(pending_pvtv2) == 1:
            item = pending_pvtv2[0]
            try:
                mask = predict_pvtv2_b2_mask(item["normalized_image_path"], config)
                result = successful_pvtv2_segmentation_row(config, item, mask)
            except Exception as row_exc:
                result = failed_segmentation_row(config, item, provider_name, str(row_exc))
            rows.append(result)
            processed, success_count, failed_count = record_segmentation_progress(
                rows,
                result,
                processed,
                skipped_count,
                success_count,
                failed_count,
                checkpoint_every,
                metadata_path,
                progress,
            )
            pending_pvtv2 = []
            return
        try:
            masks = predict_pvtv2_b2_masks([item["normalized_image_path"] for item in pending_pvtv2], config)
        except Exception as exc:
            logger().warning("Batched PVTv2 segmentation failed; retrying rows individually | rows=%d | reason=%s", len(pending_pvtv2), exc)
            for item in pending_pvtv2:
                try:
                    mask = predict_pvtv2_b2_mask(item["normalized_image_path"], config)
                    result = successful_pvtv2_segmentation_row(config, item, mask)
                except Exception as row_exc:
                    result = failed_segmentation_row(config, item, provider_name, str(row_exc))
                rows.append(result)
                processed, success_count, failed_count = record_segmentation_progress(
                    rows,
                    result,
                    processed,
                    skipped_count,
                    success_count,
                    failed_count,
                    checkpoint_every,
                    metadata_path,
                    progress,
                )
        else:
            if postprocess_workers > 1 and len(pending_pvtv2) > 1:
                with ThreadPoolExecutor(max_workers=postprocess_workers) as executor:
                    results = list(
                        executor.map(
                            lambda args: finalize_pvtv2_segmentation_row(config, args[0], args[1], provider_name),
                            zip(pending_pvtv2, masks),
                        )
                    )
            else:
                results = [
                    finalize_pvtv2_segmentation_row(config, item, mask, provider_name)
                    for item, mask in zip(pending_pvtv2, masks)
                ]
            for result in results:
                rows.append(result)
                processed, success_count, failed_count = record_segmentation_progress(
                    rows,
                    result,
                    processed,
                    skipped_count,
                    success_count,
                    failed_count,
                    checkpoint_every,
                    metadata_path,
                    progress,
                )
        pending_pvtv2 = []

    for _, row in sticker.iterrows():
        image_id = str(row["image_id"])
        if image_id in existing_rows:
            continue
        if provider_name == "pvtv2_b2" and row["detection_status"] == "success":
            pending_pvtv2.append(row)
            if len(pending_pvtv2) >= batch_size:
                flush_pvtv2()
            continue
        flush_pvtv2()
        result = run_single_segmentation_row(config, row, provider_name, overwrite_existing, require_pvtv2)
        rows.append(result)
        processed, success_count, failed_count = record_segmentation_progress(
            rows,
            result,
            processed,
            skipped_count,
            success_count,
            failed_count,
            checkpoint_every,
            metadata_path,
            progress,
        )
    flush_pvtv2()
    artifacts = pd.DataFrame(rows).drop_duplicates("image_id", keep="last").reset_index(drop=True)
    write_parquet(artifacts, metadata_path)
    progress.log(
        len(sticker),
        success=int((artifacts["segmentation_status"] == "success").sum()) if len(artifacts) else 0,
        failed=int((artifacts["segmentation_status"] == "failed").sum()) if len(artifacts) else 0,
        skipped=skipped_count,
        force=True,
    )
    logger().info(
        "Segmentation complete | rows=%d | status_counts=%s | metadata=%s",
        len(artifacts),
        artifacts["segmentation_status"].value_counts(dropna=False).to_dict() if len(artifacts) else {},
        metadata_path,
    )
    return artifacts


def is_trusted_segmentation_resume_row(
    row: dict[str, Any],
    config: dict[str, Any],
    current_image_ids: set[str],
    require_pvtv2: bool,
) -> bool:
    image_id = str(row.get("image_id"))
    if image_id not in current_image_ids:
        return False
    if row.get("segmentation_status") != "success":
        return False
    if row.get("config_hash") != config.get("_config_hash"):
        return False
    if require_pvtv2 and row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
        return False
    if require_pvtv2 and row.get("segmentation_algorithm_version") != required_pvtv2_algorithm_version(config):
        return False
    mask_path = row.get("mask_path")
    return bool(mask_path) and Path(str(mask_path)).exists()


def run_single_segmentation_row(
    config: dict[str, Any],
    row: pd.Series,
    provider_name: str,
    overwrite_existing: bool,
    require_pvtv2: bool,
) -> dict[str, Any]:
    image_id = row["image_id"]
    dataset = row["dataset"]
    output_mask = artifacts_dir(config) / "segmentation" / dataset / f"{safe_stem(image_id)}_mask.png"
    try:
        if row["detection_status"] != "success":
            raise RuntimeError("sticker_normalization_failed")
        normalized_mask = row.get("normalized_mask_path")
        seg_cfg = config.get("segmentation", {})
        if provider_name == "provided_mask_baseline":
            if pd.isna(normalized_mask) or not normalized_mask or not Path(str(normalized_mask)).exists():
                raise RuntimeError("provided_mask_baseline requires normalized_mask_path")
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            if overwrite_existing or not output_mask.exists():
                Image.open(normalized_mask).convert("L").save(output_mask)
            source = "provided_mask_baseline"
        elif output_mask.exists() and not overwrite_existing and not require_pvtv2:
            source = "existing_mask"
        elif pd.notna(normalized_mask) and seg_cfg.get("use_provided_masks_as_fallback", False):
            mask = Image.open(normalized_mask).convert("L")
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_mask)
            source = "provided_mask_ablation"
        elif provider_name == "pvtv2_b2":
            mask = predict_pvtv2_b2_mask(row["normalized_image_path"], config)
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(output_mask)
            source = required_pvtv2_segmentation_source(config)
        elif seg_cfg.get("allow_fallback", False):
            mask = fallback_segment_from_image(row["normalized_image_path"])
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(output_mask)
            source = "fallback_image_threshold"
        else:
            raise RuntimeError(f"{provider_name} inference is unavailable and fallback is disabled.")
        if require_pvtv2 and source != required_pvtv2_segmentation_source(config):
            raise RuntimeError(f"Headline segmentation requires frozen PVTv2-B2 saliency masks, got source={source}")
        return successful_segmentation_row(config, row, output_mask, source)
    except Exception as exc:
        return failed_segmentation_row(config, row, provider_name, str(exc))


def successful_pvtv2_segmentation_row(config: dict[str, Any], row: pd.Series, mask: np.ndarray) -> dict[str, Any]:
    output_mask = artifacts_dir(config) / "segmentation" / row["dataset"] / f"{safe_stem(row['image_id'])}_mask.png"
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask_array = np.asarray(mask, dtype=bool)
    Image.fromarray((mask_array.astype(np.uint8) * 255)).save(output_mask)
    return successful_segmentation_row_from_mask(config, row, output_mask, required_pvtv2_segmentation_source(config), mask_array)


def finalize_pvtv2_segmentation_row(
    config: dict[str, Any],
    row: pd.Series,
    mask: np.ndarray,
    provider_name: str,
) -> dict[str, Any]:
    try:
        return successful_pvtv2_segmentation_row(config, row, mask)
    except Exception as exc:
        return failed_segmentation_row(config, row, provider_name, str(exc))


def successful_segmentation_row(config: dict[str, Any], row: pd.Series, output_mask: Path, source: str) -> dict[str, Any]:
    mask_array = np.asarray(Image.open(output_mask).convert("L")) > 0
    return successful_segmentation_row_from_mask(config, row, output_mask, source, mask_array)


def successful_segmentation_row_from_mask(
    config: dict[str, Any],
    row: pd.Series,
    output_mask: Path,
    source: str,
    mask_array: np.ndarray,
) -> dict[str, Any]:
    mask_array = np.asarray(mask_array, dtype=bool)
    bbox = bbox_from_mask(mask_array)
    contour = contour_from_mask(mask_array)
    return segmentation_metadata_row(
        config,
        row,
        source,
        "success",
        None,
        output_mask,
        bbox,
        int(mask_array.sum()),
        int(len(contour)),
    )


def failed_segmentation_row(config: dict[str, Any], row: pd.Series, source: str, reason: str) -> dict[str, Any]:
    return segmentation_metadata_row(
        config,
        row,
        source,
        "failed",
        reason,
        None,
        (np.nan, np.nan, np.nan, np.nan),
        0,
        0,
    )


def segmentation_metadata_row(
    config: dict[str, Any],
    row: pd.Series,
    source: str,
    status: str,
    reason: str | None,
    output_mask: Path | None,
    bbox: tuple[float, float, float, float],
    area: int,
    contour_point_count: int,
) -> dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "dataset": row["dataset"],
        "dataset_version": row.get("data_version"),
        "transform_version": row.get("transform_version"),
        "config_hash": config.get("_config_hash"),
        "provider": source,
        "segmentation_source": source,
        "segmentation_algorithm_version": (
            required_pvtv2_algorithm_version(config) if source == required_pvtv2_segmentation_source(config) else source
        ),
        "segmentation_status": status,
        "failure_reason": reason,
        "mask_path": str(output_mask.resolve()) if output_mask else None,
        "bbox_x0": bbox[0],
        "bbox_y0": bbox[1],
        "bbox_x1": bbox[2],
        "bbox_y1": bbox[3],
        "mask_area_px": area,
        "contour_point_count": contour_point_count,
    }


def record_segmentation_progress(
    rows: list[dict[str, Any]],
    item: dict[str, Any],
    processed: int,
    skipped_count: int,
    success_count: int,
    failed_count: int,
    checkpoint_every: int,
    metadata_path: Path,
    progress: ProgressLogger,
) -> tuple[int, int, int]:
    processed += 1
    if item["segmentation_status"] == "success":
        success_count += 1
    else:
        failed_count += 1
        logger().warning(
            "Segmentation failed | image_id=%s | dataset=%s | reason=%s",
            item["image_id"],
            item["dataset"],
            item["failure_reason"],
        )
    progress.log(
        processed + skipped_count,
        success=success_count,
        failed=failed_count,
        skipped=skipped_count,
        current=str(item["image_id"]),
    )
    if checkpoint_every > 0 and processed % checkpoint_every == 0:
        write_parquet(pd.DataFrame(rows).drop_duplicates("image_id", keep="last"), metadata_path)
        logger().info(
            "Segmentation checkpoint written | processed_this_run=%d | total_rows=%d | metadata=%s",
            processed,
            len(rows),
            metadata_path,
        )
    return processed, success_count, failed_count


def run_mask_sanity_checks(config: dict[str, Any]) -> pd.DataFrame:
    seg_cfg = config.get("segmentation", {})
    sanity_cfg = seg_cfg.get("sanity_checks", {})
    if not sanity_cfg.get("enabled", not seg_cfg.get("allow_fallback", False)):
        return pd.DataFrame()
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    summary = validate_segmentation_masks(config, artifacts, sticker)
    failures = summary[summary["sanity_status"] != "accepted"]
    if not failures.empty:
        raise RuntimeError(
            f"Mask sanity checks failed for {len(failures)} rows. "
            f"See {reports_dir(config) / 'mask_sanity_summary.csv'}."
        )
    return summary


def run_mask_review(config: dict[str, Any]) -> pd.DataFrame:
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    selected = select_mask_review_rows(config, artifacts, sticker)
    if selected.empty:
        raise RuntimeError("No successful masks were available for review.")
    output_root = reports_dir(config) / "mask_review"
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    max_overlay_side = int(config.get("segmentation", {}).get("review", {}).get("max_overlay_side", 1600))
    for item in selected.to_dict(orient="records"):
        category_dir = output_root / str(item["review_category"])
        category_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = category_dir / f"{int(item['selection_rank']):02d}_{safe_stem(str(item['image_id']))}.png"
        write_mask_overlay(item["normalized_image_path"], item["mask_path"], overlay_path, max_overlay_side)
        rows.append({**item, "overlay_path": str(overlay_path.resolve())})
    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "mask_review_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def select_mask_review_rows(config: dict[str, Any], artifacts: pd.DataFrame, sticker: pd.DataFrame) -> pd.DataFrame:
    review_cfg = config.get("segmentation", {}).get("review", {})
    random_count = int(review_cfg.get("random_count", 10))
    anomaly_count = int(review_cfg.get("area_anomaly_count", 10))
    complexity_count = int(review_cfg.get("complexity_count", 10))
    seed = int(config.get("run", {}).get("random_seed", 0))
    candidates = build_mask_review_candidates(config, artifacts, sticker)
    if candidates.empty:
        return candidates

    selected_parts: list[pd.DataFrame] = []
    stable = candidates.sort_values(["dataset", "image_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if random_count > 0 and not stable.empty:
        positions = np.sort(rng.choice(len(stable), size=min(random_count, len(stable)), replace=False))
        selected_parts.append(with_review_category(stable.iloc[positions], "random"))
    if anomaly_count > 0:
        anomaly = candidates.sort_values(["area_anomaly_score", "image_id"], ascending=[False, True]).head(anomaly_count)
        selected_parts.append(with_review_category(anomaly, "area_anomaly"))
    if complexity_count > 0:
        complexity = candidates.sort_values(["complexity_score", "image_id"], ascending=[False, True]).head(complexity_count)
        selected_parts.append(with_review_category(complexity, "high_complexity"))
    if not selected_parts:
        return candidates.iloc[0:0].copy()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_rank"] = selected.groupby("review_category").cumcount() + 1
    return selected.reset_index(drop=True)


def build_mask_review_candidates(config: dict[str, Any], artifacts: pd.DataFrame, sticker: pd.DataFrame) -> pd.DataFrame:
    required_artifact_cols = ["image_id", "dataset", "segmentation_source", "segmentation_status", "mask_path", "mask_area_px"]
    required_sticker_cols = ["image_id", "dataset", "detection_status", "normalized_image_path", "normalized_width", "normalized_height"]
    missing_artifacts = [col for col in required_artifact_cols if col not in artifacts.columns]
    missing_sticker = [col for col in required_sticker_cols if col not in sticker.columns]
    if missing_artifacts:
        raise RuntimeError(f"Segmentation metadata is missing required review columns: {missing_artifacts}")
    if missing_sticker:
        raise RuntimeError(f"Sticker metadata is missing required review columns: {missing_sticker}")
    merged = sticker[required_sticker_cols].merge(artifacts[required_artifact_cols], on=["image_id", "dataset"], how="inner")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if row.get("detection_status") != "success" or row.get("segmentation_status") != "success":
            continue
        image_path = row.get("normalized_image_path")
        mask_path = row.get("mask_path")
        if not image_path or pd.isna(image_path) or not Path(str(image_path)).exists():
            continue
        if not mask_path or pd.isna(mask_path) or not Path(str(mask_path)).exists():
            continue
        image_area = float(row.get("normalized_width", 0) or 0) * float(row.get("normalized_height", 0) or 0)
        mask_fraction = float(row.get("mask_area_px", 0) / image_area) if image_area > 0 else np.nan
        rows.append(
            {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "segmentation_source": row["segmentation_source"],
                "normalized_image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_area_px": int(row.get("mask_area_px", 0)),
                "mask_area_fraction": mask_fraction,
                "complexity_score": mask_complexity_score(mask_path, config),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    fractions = candidates["mask_area_fraction"].to_numpy(dtype=float)
    finite = fractions[np.isfinite(fractions)]
    if len(finite) == 0:
        candidates["area_anomaly_score"] = np.nan
        return candidates
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = mad * 1.4826 if mad > 0 else float(np.std(finite))
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    candidates["area_anomaly_score"] = np.abs(candidates["mask_area_fraction"] - median) / scale
    return candidates


def with_review_category(rows: pd.DataFrame, category: str) -> pd.DataFrame:
    part = rows.copy().reset_index(drop=True)
    part["review_category"] = category
    return part


def mask_complexity_score(mask_path: str | Path, config: dict[str, Any]) -> float:
    max_side = int(config.get("segmentation", {}).get("review", {}).get("complexity_max_side", 768))
    mask_image = Image.open(mask_path).convert("L")
    if max(mask_image.size) > max_side:
        scale = max_side / max(mask_image.size)
        size = (max(1, int(round(mask_image.width * scale))), max(1, int(round(mask_image.height * scale))))
        mask_image = mask_image.resize(size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image) > 0
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    contour = contour_from_mask(mask)
    perimeter = float(len(contour))
    hull_area = convex_hull_area(contour)
    solidity_penalty = 0.0
    if hull_area and np.isfinite(hull_area):
        solidity = min(1.0, area / hull_area)
        solidity_penalty = 10.0 * max(0.0, 1.0 - solidity)
    return float(perimeter / np.sqrt(area) + solidity_penalty)


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return float("nan")
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError:
        return float("nan")
    return float(hull.volume)


def write_mask_overlay(image_path: str | Path, mask_path: str | Path, output_path: Path, max_side: int) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    base = image.convert("RGBA")
    mask_arr = np.asarray(mask) > 0
    overlay_arr = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    overlay_arr[mask_arr] = np.asarray([255, 48, 64, 92], dtype=np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    composited = Image.alpha_composite(base, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composited.convert("RGB").save(output_path)


def run_mask_freeze(config: dict[str, Any], accepted_by: str = "human_review") -> dict[str, Any]:
    review_manifest = reports_dir(config) / "mask_review" / "mask_review_manifest.csv"
    if not review_manifest.exists():
        raise RuntimeError(f"Mask review manifest is required before freezing masks: {review_manifest}")
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    successes = artifacts[artifacts["segmentation_status"] == "success"].copy()
    if successes.empty:
        raise RuntimeError("No successful segmentation rows are available to freeze.")
    masks: list[dict[str, Any]] = []
    for item in successes.sort_values(["dataset", "image_id"]).to_dict(orient="records"):
        mask_path = item.get("mask_path")
        if not mask_path or pd.isna(mask_path) or not Path(str(mask_path)).exists():
            raise RuntimeError(f"Cannot freeze missing mask for image_id={item.get('image_id')}: {mask_path}")
        masks.append(
            {
                "image_id": item["image_id"],
                "dataset": item["dataset"],
                "segmentation_source": item.get("segmentation_source"),
                "mask_path": str(Path(str(mask_path)).resolve()),
                "mask_sha256": sha256_file(Path(str(mask_path))),
            }
        )
    manifest = {
        "status": "frozen",
        "accepted_by": accepted_by,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run", {}).get("run_id"),
        "config_hash": config.get("_config_hash"),
        "random_seed": config.get("run", {}).get("random_seed"),
        "review_manifest_path": str(review_manifest.resolve()),
        "review_row_count": int(len(pd.read_csv(review_manifest))),
        "mask_count": len(masks),
        "source_counts": successes["segmentation_source"].value_counts(dropna=False).to_dict(),
        "masks": masks,
    }
    output_path = reports_dir(config) / "mask_review" / "frozen_mask_manifest.json"
    write_json(manifest, output_path)
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_segmentation_masks(config: dict[str, Any], artifacts: pd.DataFrame, sticker: pd.DataFrame) -> pd.DataFrame:
    required_artifact_cols = [
        "image_id",
        "dataset",
        "dataset_version",
        "transform_version",
        "config_hash",
        "segmentation_source",
        "segmentation_status",
        "mask_path",
        "mask_area_px",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "contour_point_count",
    ]
    missing_cols = [col for col in required_artifact_cols if col not in artifacts.columns]
    if missing_cols:
        raise RuntimeError(f"Segmentation metadata is missing required columns: {missing_cols}")

    required_sticker_cols = ["image_id", "dataset", "detection_status", "normalized_width", "normalized_height"]
    missing_sticker_cols = [col for col in required_sticker_cols if col not in sticker.columns]
    if missing_sticker_cols:
        raise RuntimeError(f"Sticker metadata is missing required columns: {missing_sticker_cols}")

    sanity_cfg = config.get("segmentation", {}).get("sanity_checks", {})
    min_fraction = float(sanity_cfg.get("min_area_fraction", 0.005))
    max_fraction = float(sanity_cfg.get("max_area_fraction", 0.98))
    min_median = float(sanity_cfg.get("min_median_area_fraction", 0.02))
    max_median = float(sanity_cfg.get("max_median_area_fraction", 0.85))
    require_pvtv2 = should_require_pvtv2_inference(config)
    configured_provider = required_pvtv2_segmentation_source(config)

    merged = sticker[required_sticker_cols].merge(artifacts, on=["image_id", "dataset"], how="left", suffixes=("_sticker", ""))
    rows: list[dict[str, Any]] = []
    accepted_fractions: list[float] = []
    for _, row in merged.iterrows():
        if row.get("detection_status") != "success":
            continue
        status = "accepted"
        reasons: list[str] = []
        mask_path = row.get("mask_path")
        if pd.isna(row.get("segmentation_status")):
            reasons.append("missing_segmentation_metadata")
        elif row.get("segmentation_status") != "success":
            reasons.append(f"segmentation_status={row.get('segmentation_status')}")
        if require_pvtv2 and row.get("segmentation_source") != required_pvtv2_segmentation_source(config):
            reasons.append(f"segmentation_source={row.get('segmentation_source')}")
        for col in ["dataset_version", "transform_version", "config_hash", "mask_area_px", "contour_point_count"]:
            if pd.isna(row.get(col)):
                reasons.append(f"missing_{col}")
        if not mask_path or pd.isna(mask_path) or not Path(str(mask_path)).exists():
            reasons.append("missing_mask_file")
            mask_area = 0
            mask_fraction = np.nan
        else:
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            mask_area = int(mask.sum())
            image_area = float(row.get("normalized_width", 0) or 0) * float(row.get("normalized_height", 0) or 0)
            mask_fraction = float(mask_area / image_area) if image_area > 0 else np.nan
            if mask_area <= 0:
                reasons.append("empty_mask")
            if not np.isfinite(mask_fraction):
                reasons.append("missing_normalized_dimensions")
            elif mask_fraction < min_fraction or mask_fraction > max_fraction:
                reasons.append(f"mask_area_fraction={mask_fraction:.6f}")
            else:
                accepted_fractions.append(mask_fraction)
        if reasons:
            status = "failed"
        rows.append(
            {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "segmentation_source": row.get("segmentation_source"),
                "segmentation_status": row.get("segmentation_status"),
                "sanity_status": status,
                "failure_reason": ";".join(reasons) if reasons else None,
                "mask_path": mask_path,
                "mask_area_px": mask_area,
                "mask_area_fraction": mask_fraction,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("Mask sanity checks found no sticker-normalized rows to validate.")
    if accepted_fractions:
        median_fraction = float(np.median(accepted_fractions))
        if median_fraction < min_median or median_fraction > max_median:
            summary.loc[len(summary)] = {
                "image_id": "__dataset_distribution__",
                "dataset": "all",
                "segmentation_source": configured_provider,
                "segmentation_status": "success",
                "sanity_status": "failed",
                "failure_reason": f"median_mask_area_fraction={median_fraction:.6f}",
                "mask_path": None,
                "mask_area_px": 0,
                "mask_area_fraction": median_fraction,
            }
    else:
        summary.loc[len(summary)] = {
            "image_id": "__dataset_distribution__",
            "dataset": "all",
            "segmentation_source": configured_provider,
            "segmentation_status": "failed",
            "sanity_status": "failed",
            "failure_reason": "no_accepted_mask_area_fractions",
            "mask_path": None,
            "mask_area_px": 0,
            "mask_area_fraction": np.nan,
        }
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir(config) / "mask_sanity_summary.csv", index=False)
    return summary


def should_require_pvtv2_inference(config: dict[str, Any]) -> bool:
    seg_cfg = config.get("segmentation", {})
    if "require_pvtv2_inference" in seg_cfg:
        return bool(seg_cfg["require_pvtv2_inference"])
    return seg_cfg.get("provider", "pvtv2_b2") == "pvtv2_b2" and not seg_cfg.get("allow_fallback", False)


def fallback_segment_from_image(image_path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    gray = arr.mean(axis=2)
    threshold = np.percentile(gray, 45)
    mask = gray < threshold
    labeled, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError(f"Fallback segmentation found no foreground for {image_path}")
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    largest = int(np.argmax(sizes) + 1)
    return ndimage.binary_fill_holes(labeled == largest)


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.empty((0, 2), dtype=float)
    eroded = ndimage.binary_erosion(mask)
    boundary = mask ^ eroded
    ys, xs = np.where(boundary)
    return np.column_stack([xs, ys]).astype(float)
