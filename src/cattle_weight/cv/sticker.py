from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from cattle_weight.io import artifacts_dir, figures_dir, read_parquet, registry_dir, reports_dir, write_parquet
from cattle_weight.progress import ProgressLogger, logger


def detect_sticker(image_path: str | Path, mask_path: str | Path | None = None) -> dict[str, Any]:
    if mask_path:
        try:
            return detect_sticker_from_mask(mask_path)
        except Exception:
            pass
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        np.maximum(max_channel, 1.0),
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )
    colored = (saturation > 0.35) & (max_channel > 80)
    labeled, count = ndimage.label(colored)
    best: dict[str, Any] | None = None
    height, width = colored.shape
    for label in range(1, count + 1):
        ys, xs = np.where(labeled == label)
        area = int(len(xs))
        if area < 20 or area > width * height * 0.05:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        box_w = max(1, x1 - x0 + 1)
        box_h = max(1, y1 - y0 + 1)
        aspect = box_w / box_h
        if aspect < 0.55 or aspect > 1.8:
            continue
        radius = math.sqrt(area / math.pi)
        bbox_radius = (box_w + box_h) / 4.0
        circularity_error = abs(radius - bbox_radius) / max(radius, 1.0)
        score = circularity_error + abs(1.0 - aspect) * 0.25
        candidate = {
            "center_x": float(xs.mean()),
            "center_y": float(ys.mean()),
            "radius_px": float(radius),
            "area_px": area,
            "method": "color_component_circle",
            "score": float(score),
        }
        if best is None or score < best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError(f"Sticker detection failed for {image_path}")
    return best


def detect_sticker_from_mask(mask_path: str | Path) -> dict[str, Any]:
    arr = np.asarray(Image.open(mask_path).convert("RGB"))
    sticker_colors = np.asarray([[0, 117, 255], [255, 240, 0]], dtype=np.int16)
    pixels = arr.astype(np.int16)
    matches = np.zeros(arr.shape[:2], dtype=bool)
    for color in sticker_colors:
        matches |= np.linalg.norm(pixels - color[None, None, :], axis=2) <= 35
    labeled, count = ndimage.label(matches)
    if count == 0:
        raise RuntimeError(f"No sticker class found in mask: {mask_path}")
    best_label = None
    best_area = 0
    for label in range(1, count + 1):
        area = int((labeled == label).sum())
        if area > best_area:
            best_area = area
            best_label = label
    ys, xs = np.where(labeled == best_label)
    radius = math.sqrt(best_area / math.pi)
    return {
        "center_x": float(xs.mean()),
        "center_y": float(ys.mean()),
        "radius_px": float(radius),
        "area_px": int(best_area),
        "method": "annotation_sticker_class",
        "score": 0.0,
    }


def normalize_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path | None,
    sticker: dict[str, Any],
    output_image_path: Path,
    output_mask_path: Path | None,
    target_radius_px: float,
) -> dict[str, Any]:
    radius = float(sticker["radius_px"])
    if radius <= 0:
        raise ValueError(f"Sticker radius must be positive, got {radius}")
    scale = target_radius_px / radius
    image = Image.open(image_path).convert("RGB")
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    max_output_pixels = float(sticker.get("max_output_pixels", 0) or 0)
    if max_output_pixels and new_size[0] * new_size[1] > max_output_pixels:
        raise RuntimeError(
            f"normalized image would have {new_size[0] * new_size[1]} pixels, above configured max_output_pixels={max_output_pixels:.0f}"
        )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    resized_image = image.resize(new_size, Image.Resampling.BILINEAR)
    if output_image_path.suffix.lower() in {".jpg", ".jpeg"}:
        resized_image.save(output_image_path, quality=92)
    else:
        resized_image.save(output_image_path)
    normalized_mask_path = None
    if mask_path and output_mask_path:
        mask = load_cattle_mask_image(mask_path)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.resize(new_size, Image.Resampling.NEAREST).save(output_mask_path)
        normalized_mask_path = str(output_mask_path.resolve())
    return {
        "scale": float(scale),
        "normalized_width": int(new_size[0]),
        "normalized_height": int(new_size[1]),
        "normalized_sticker_radius_px": float(radius * scale),
        "normalized_image_path": str(output_image_path.resolve()),
        "normalized_mask_path": normalized_mask_path,
    }


def run_sticker_normalization(config: dict[str, Any]) -> pd.DataFrame:
    index = read_parquet(registry_dir(config) / "dataset_index.parquet")
    current_image_ids = set(index["image_id"].astype(str))
    target = float(config.get("sticker", {}).get("target_radius_px", 100.0))
    metadata_path = registry_dir(config) / "sticker_metadata.parquet"
    existing_rows: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        existing = read_parquet(metadata_path).drop_duplicates("image_id", keep="last")
        for item in existing.to_dict(orient="records"):
            image_id = str(item["image_id"])
            if image_id in current_image_ids and existing_sticker_row_complete(item):
                existing_rows[image_id] = item
    rows: list[dict[str, Any]] = list(existing_rows.values())
    base = artifacts_dir(config) / "sticker_normalized"
    image_extension = str(config.get("sticker", {}).get("normalized_image_extension", ".jpg"))
    checkpoint_every = int(config.get("sticker", {}).get("checkpoint_every", 100))
    log_every = int(config.get("sticker", {}).get("log_every", checkpoint_every or 100))
    worker_count = max(1, int(config.get("sticker", {}).get("workers", 1)))
    progress = ProgressLogger("sticker normalization", total=len(index), log_every=log_every)
    logger().info(
        "Sticker normalization setup | rows=%d | resumed_rows=%d | workers=%d | checkpoint_every=%d | "
        "log_every=%d | output_extension=%s | metadata=%s",
        len(index),
        len(existing_rows),
        worker_count,
        checkpoint_every,
        log_every,
        image_extension,
        metadata_path,
    )
    processed = 0
    success_count = int(sum(1 for item in rows if item.get("detection_status") == "success"))
    failed_count = int(sum(1 for item in rows if item.get("detection_status") == "failed"))
    skipped_count = len(existing_rows)
    if skipped_count:
        progress.log(
            skipped_count,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            extra="resumed_existing_rows",
            force=True,
        )
    tasks: list[dict[str, Any]] = []
    for _, row in index.iterrows():
        image_id = str(row["image_id"])
        if image_id in existing_rows:
            continue
        tasks.append(build_sticker_task(row, base, image_extension, target, config))

    if worker_count == 1 or len(tasks) <= 1:
        for task in tasks:
            item, failure = normalize_sticker_task(task)
            rows.append(item)
            processed, success_count, failed_count = record_sticker_progress(
                rows,
                item,
                failure,
                processed,
                skipped_count,
                success_count,
                failed_count,
                checkpoint_every,
                metadata_path,
                progress,
            )
    else:
        max_workers = min(worker_count, len(tasks))
        logger().info("Sticker normalization worker pool starting | workers=%d | tasks=%d", max_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(normalize_sticker_task, task) for task in tasks]
            for future in as_completed(futures):
                item, failure = future.result()
                rows.append(item)
                processed, success_count, failed_count = record_sticker_progress(
                    rows,
                    item,
                    failure,
                    processed,
                    skipped_count,
                    success_count,
                    failed_count,
                    checkpoint_every,
                    metadata_path,
                    progress,
                )
    metadata = pd.DataFrame(rows).drop_duplicates("image_id", keep="last").reset_index(drop=True)
    write_parquet(metadata, metadata_path)
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    failure_rows = metadata[metadata["detection_status"] == "failed"]
    failure_report = failure_rows[["image_id", "dataset", "failure_reason"]].rename(columns={"failure_reason": "reason"})
    failure_report.to_csv(reports_dir(config) / "sticker_detection_failures.csv", index=False)
    write_sticker_reports(metadata, config)
    success_rate = float((metadata["detection_status"] == "success").mean()) if len(metadata) else 0.0
    min_success_rate = float(config.get("sticker", {}).get("min_success_rate", 0.98))
    if success_rate < min_success_rate:
        write_sticker_failure_note(success_rate, min_success_rate, config)
    progress.log(
        len(index),
        success=int((metadata["detection_status"] == "success").sum()) if len(metadata) else 0,
        failed=int((metadata["detection_status"] == "failed").sum()) if len(metadata) else 0,
        skipped=skipped_count,
        force=True,
    )
    logger().info(
        "Sticker normalization complete | rows=%d | status_counts=%s | success_rate=%.3f | metadata=%s",
        len(metadata),
        metadata["detection_status"].value_counts(dropna=False).to_dict() if len(metadata) else {},
        success_rate,
        metadata_path,
    )
    return metadata


def build_sticker_task(
    row: pd.Series,
    base: Path,
    image_extension: str,
    target_radius_px: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_id = str(row["image_id"])
    dataset = str(row["dataset"])
    image_stem = safe_stem(image_id)
    out_image = base / dataset / "images" / f"{image_stem}{image_extension}"
    out_mask = base / dataset / "masks" / f"{image_stem}_mask.png" if pd.notna(row.get("mask_path")) else None
    raw_mask_path = row.get("mask_path") if pd.notna(row.get("mask_path")) else None
    return {
        "image_id": image_id,
        "dataset": dataset,
        "data_version": row.get("data_version"),
        "image_path": str(row["image_path"]),
        "mask_path": str(raw_mask_path) if raw_mask_path else None,
        "out_image": out_image,
        "out_mask": out_mask,
        "target_radius_px": target_radius_px,
        "max_output_pixels": config.get("sticker", {}).get("max_output_pixels", 150_000_000),
        "transform_version": config.get("_config_hash"),
    }


def normalize_sticker_task(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    dataset = task["dataset"]
    image_id = task["image_id"]
    try:
        sticker = detect_sticker(task["image_path"], task.get("mask_path"))
        normalized = normalize_image_and_mask(
            task["image_path"],
            task.get("mask_path"),
            {**sticker, "max_output_pixels": task["max_output_pixels"]},
            Path(task["out_image"]),
            Path(task["out_mask"]) if task.get("out_mask") else None,
            float(task["target_radius_px"]),
        )
        status = "success"
        reason = None
        failure = None
    except Exception as exc:
        sticker = {
            "center_x": np.nan,
            "center_y": np.nan,
            "radius_px": np.nan,
            "area_px": np.nan,
            "method": "failed",
            "score": np.nan,
        }
        normalized = {
            "scale": np.nan,
            "normalized_width": np.nan,
            "normalized_height": np.nan,
            "normalized_sticker_radius_px": np.nan,
            "normalized_image_path": None,
            "normalized_mask_path": None,
        }
        status = "failed"
        reason = str(exc)
        failure = {"image_id": image_id, "dataset": dataset, "reason": reason}

    return (
        {
            "image_id": image_id,
            "dataset": dataset,
            "data_version": task.get("data_version"),
            "raw_image_path": task["image_path"],
            "raw_mask_path": task.get("mask_path"),
            "raw_sticker_radius_px": sticker["radius_px"],
            "raw_sticker_center_x": sticker["center_x"],
            "raw_sticker_center_y": sticker["center_y"],
            "detection_method": sticker["method"],
            "detection_score": sticker["score"],
            "target_sticker_radius_px": task["target_radius_px"],
            "detection_status": status,
            "failure_reason": reason,
            **normalized,
            "transform_version": task.get("transform_version"),
        },
        failure,
    )


def record_sticker_progress(
    rows: list[dict[str, Any]],
    item: dict[str, Any],
    failure: dict[str, Any] | None,
    processed: int,
    skipped_count: int,
    success_count: int,
    failed_count: int,
    checkpoint_every: int,
    metadata_path: Path,
    progress: ProgressLogger,
) -> tuple[int, int, int]:
    processed += 1
    if item["detection_status"] == "success":
        success_count += 1
    else:
        failed_count += 1
    if failure:
        logger().warning(
            "Sticker normalization failed | image_id=%s | dataset=%s | reason=%s",
            failure["image_id"],
            failure["dataset"],
            failure["reason"],
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
            "Sticker checkpoint written | processed_this_run=%d | total_rows=%d | metadata=%s",
            processed,
            len(rows),
            metadata_path,
        )
    return processed, success_count, failed_count


def safe_stem(image_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in image_id)


def existing_artifacts_present(row: dict[str, Any]) -> bool:
    image_path = row.get("normalized_image_path")
    if not image_path or not Path(str(image_path)).exists():
        return False
    mask_path = row.get("normalized_mask_path")
    if mask_path and str(mask_path) != "nan" and not Path(str(mask_path)).exists():
        return False
    return row.get("detection_status") == "success"


def existing_sticker_row_complete(row: dict[str, Any]) -> bool:
    if row.get("detection_status") == "success":
        return existing_artifacts_present(row)
    if row.get("detection_status") == "failed":
        reason = row.get("failure_reason")
        return reason not in {None, "", "nan"}
    return False


def load_cattle_mask_image(mask_path: str | Path) -> Image.Image:
    image = Image.open(mask_path).convert("RGB")
    arr = np.asarray(image)
    cattle_colors = np.asarray([[255, 30, 249], [0, 255, 0]], dtype=np.int16)
    pixels = arr.astype(np.int16)
    matches = np.zeros(arr.shape[:2], dtype=bool)
    for color in cattle_colors:
        matches |= np.linalg.norm(pixels - color[None, None, :], axis=2) <= 45
    if not matches.any():
        gray = np.asarray(image.convert("L"))
        matches = gray > 0
    return Image.fromarray((matches.astype(np.uint8) * 255), mode="L")


def write_sticker_reports(metadata: pd.DataFrame, config: dict[str, Any]) -> None:
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    cols = [
        "image_id",
        "dataset",
        "raw_sticker_radius_px",
        "normalized_sticker_radius_px",
        "detection_status",
        "failure_reason",
    ]
    metadata[cols].to_csv(reports_dir(config) / "sticker_radius_before_after.csv", index=False)
    success = metadata[metadata["detection_status"] == "success"]
    if not success.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(success["raw_sticker_radius_px"], alpha=0.65, label="before", bins=min(20, max(4, len(success) // 2)))
        plt.hist(success["normalized_sticker_radius_px"], alpha=0.65, label="after", bins=min(20, max(4, len(success) // 2)))
        plt.axvline(float(config.get("sticker", {}).get("target_radius_px", 100.0)), color="black", linewidth=1)
        plt.xlabel("Sticker radius (px)")
        plt.ylabel("Images")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir(config) / "sticker_radius_before_after.png")
        plt.close()


def write_sticker_failure_note(success_rate: float, min_success_rate: float, config: dict[str, Any]) -> None:
    path = reports_dir(config) / "sticker_detection_success_note.md"
    path.write_text(
        "\n".join(
            [
                "# Sticker Detection Success",
                "",
                f"Observed automatic sticker detection success rate: {success_rate:.3f}.",
                f"Configured target success rate: {min_success_rate:.3f}.",
                "",
                "Failures are listed in `reports/sticker_detection_failures.csv`.",
            ]
        ),
        encoding="utf-8",
    )
