from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cattle_weight.config import REPO_ROOT
from cattle_weight.cv.segmentation import run_mask_sanity_checks, should_require_pvtv2_inference
from cattle_weight.io import artifacts_dir, read_parquet, registry_dir, reports_dir, write_json, write_parquet
from cattle_weight.progress import ProgressLogger, logger


def run_depth(config: dict[str, Any]) -> pd.DataFrame:
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    merged = artifacts.merge(
        sticker[["image_id", "normalized_image_path", "raw_image_path", "scale"]],
        on="image_id",
        how="left",
    )
    depth_cfg = config.get("depth", {})
    if should_require_pvtv2_inference(config):
        sanity = run_mask_sanity_checks(config)
        accepted_ids = set(sanity.loc[sanity["sanity_status"] == "accepted", "image_id"].astype(str))
        merged = merged[merged["image_id"].astype(str).isin(accepted_ids)].reset_index(drop=True)
        if merged.empty:
            raise RuntimeError("No rows have accepted PVTv2 masks for Depth Pro inference.")
    metadata_path = registry_dir(config) / "depth_metadata.parquet"
    index = read_parquet(registry_dir(config) / "dataset_index.parquet")
    current_image_ids = set(merged["image_id"].astype(str)) & set(index["image_id"].astype(str))
    expected_provider = "depth_pro" if depth_cfg.get("provider") == "depth_pro" and not depth_cfg.get("allow_fallback", False) else None
    overwrite_existing = bool(depth_cfg.get("overwrite_existing", False))
    existing_rows: dict[str, dict[str, Any]] = {}
    if metadata_path.exists() and not overwrite_existing:
        existing = read_parquet(metadata_path)
        for item in existing.to_dict(orient="records"):
            image_id = str(item["image_id"])
            if is_trusted_depth_resume_row(item, config, current_image_ids, expected_provider):
                existing_rows[image_id] = item
    rows: list[dict[str, Any]] = list(existing_rows.values())
    processed = 0
    checkpoint_every = int(depth_cfg.get("checkpoint_every", 50))
    log_every = int(depth_cfg.get("log_every", checkpoint_every or 25))
    success_count = sum(1 for item in rows if item.get("depth_status") == "success")
    failed_count = sum(1 for item in rows if item.get("depth_status") == "failed")
    skipped_count = len(existing_rows)
    progress = ProgressLogger("depth", total=len(merged), log_every=log_every)
    logger().info(
        "Depth setup | total_rows=%d | existing_success=%d | to_process=%d | provider=%s | allow_fallback=%s | "
        "input_max_side=%s | output_max_side=%s | checkpoint_every=%d | log_every=%d | metadata=%s",
        len(merged),
        len(existing_rows),
        len(merged) - len(existing_rows),
        depth_cfg.get("provider"),
        depth_cfg.get("allow_fallback", False),
        depth_cfg.get("input_max_side"),
        depth_cfg.get("output_max_side"),
        checkpoint_every,
        log_every,
        metadata_path,
    )
    if existing_rows:
        progress.log(
            skipped_count,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            extra="resumed_existing_success",
            force=True,
        )
    for _, row in merged.iterrows():
        image_id = row["image_id"]
        if image_id in existing_rows:
            continue
        dataset = row["dataset"]
        output_path = artifacts_dir(config) / "depth" / dataset / f"{safe_name(image_id)}_depth.npy"
        item_started = time.perf_counter()
        extra = None
        try:
            if row["segmentation_status"] != "success":
                raise RuntimeError("segmentation_failed")
            if depth_cfg.get("provider") == "depth_pro" and not depth_cfg.get("allow_fallback", False):
                depth, focal = infer_depth_pro(config, row)
                provider = "depth_pro"
                status = "success"
                reason = None
            elif output_path.exists():
                depth = np.load(output_path)
                focal = float(row.get("focal_length_px", np.nan))
                if np.isnan(focal):
                    focal = float(max(depth.shape))
                provider = "existing_depth"
                status = "success"
                reason = None
            else:
                depth = fallback_depth_map(
                    row["normalized_image_path"],
                    row["mask_path"],
                    int(depth_cfg.get("fallback_max_side", 512)),
                )
                focal = float(max(depth.shape))
                provider = "fallback_dummy_relative_depth"
                status = "success"
                reason = None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, depth.astype(np.float32))
            file_mb = output_path.stat().st_size / 1024 / 1024
            extra = f"shape={depth.shape[1]}x{depth.shape[0]} | file_mb={file_mb:.2f}"
        except Exception as exc:
            output_path = None
            focal = np.nan
            status = "failed"
            provider = depth_cfg.get("provider", "depth_pro")
            reason = str(exc)
            failed_count += 1
            logger().warning("Depth failed | image_id=%s | dataset=%s | reason=%s", image_id, dataset, reason)
        else:
            success_count += 1
        rows.append(
            {
                "image_id": image_id,
                "dataset": dataset,
                "dataset_version": row.get("dataset_version"),
                "transform_version": row.get("transform_version"),
                "config_hash": config.get("_config_hash"),
                "provider": provider,
                "depth_provider": provider,
                "depth_status": status,
                "failure_reason": reason,
                "depth_map_path": str(output_path.resolve()) if output_path else None,
                "focal_length_px": focal,
            }
        )
        processed += 1
        item_seconds = time.perf_counter() - item_started
        completed = skipped_count + processed
        detail = f"item_seconds={item_seconds:.1f}"
        if extra:
            detail = f"{detail} | {extra}"
        progress.log(
            completed,
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            current=str(image_id),
            extra=detail,
        )
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            write_parquet(pd.DataFrame(rows).drop_duplicates("image_id", keep="last"), metadata_path)
            logger().info(
                "Depth checkpoint written | processed_this_run=%d | total_rows=%d | metadata=%s",
                processed,
                len(rows),
                metadata_path,
            )
    out = pd.DataFrame(rows).drop_duplicates("image_id", keep="last").reset_index(drop=True)
    write_parquet(out, metadata_path)
    status_counts = out["depth_status"].value_counts(dropna=False).to_dict() if "depth_status" in out else {}
    provider_counts = out["depth_provider"].value_counts(dropna=False).to_dict() if "depth_provider" in out else {}
    logger().info(
        "Depth complete | rows=%d | status_counts=%s | provider_counts=%s | metadata=%s",
        len(out),
        status_counts,
        provider_counts,
        metadata_path,
    )
    return out


def is_trusted_depth_resume_row(
    row: dict[str, Any],
    config: dict[str, Any],
    current_image_ids: set[str],
    expected_provider: str | None = None,
) -> bool:
    image_id = str(row.get("image_id"))
    if image_id not in current_image_ids:
        return False
    if row.get("depth_status") != "success":
        return False
    provider = row.get("depth_provider") or row.get("provider")
    if expected_provider and provider != expected_provider:
        return False
    path = row.get("depth_map_path")
    if not path or not Path(str(path)).exists():
        return False
    if row.get("config_hash") == config.get("_config_hash"):
        return True
    return row_is_reusable_by_stable_input(row)


def row_is_reusable_by_stable_input(row: dict[str, Any]) -> bool:
    marker = row.get("reusable_by_stable_input_config", False)
    stable_hash = row.get("stable_input_config_hash")
    return bool(marker) and stable_hash not in {None, "", "nan"}


def run_depth_gpu_check(config: dict[str, Any], image_index: int = 0) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Depth GPU check requires torch.") from exc
    depth_cfg = config.get("depth", {})
    requested_device = str(depth_cfg.get("device") or "")
    if not requested_device.startswith("cuda"):
        raise RuntimeError(f"Depth GPU check requires depth.device to be CUDA, got {requested_device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Depth GPU check requires CUDA, but torch.cuda.is_available() is false.")

    rows = depth_check_rows(config)
    if not rows:
        raise RuntimeError("No successful segmentation rows are available for Depth Pro GPU check.")
    if image_index < 0 or image_index >= len(rows):
        raise IndexError(f"--image-index {image_index} is outside available rows 0..{len(rows) - 1}.")
    row = rows[image_index]

    torch.cuda.reset_peak_memory_stats()
    before_allocated = int(torch.cuda.memory_allocated())
    model, transform = load_depth_pro_model(str(depth_cfg.get("checkpoint_path", "")), requested_device)
    try:
        model_device = next(model.parameters()).device
    except Exception as exc:
        raise RuntimeError("Unable to inspect Depth Pro model parameter device.") from exc
    if model_device.type != "cuda":
        raise RuntimeError(f"Depth Pro model parameters are not on CUDA: {model_device}")

    masked = depth_input_image(config, row)
    tensor = transform(masked)
    if not hasattr(tensor, "to"):
        raise RuntimeError("Depth Pro transform did not return a tensor-like object with .to().")
    tensor = tensor.to(requested_device)
    input_device = getattr(tensor, "device", None)
    if input_device is None or input_device.type != "cuda":
        raise RuntimeError(f"Depth Pro input tensor is not on CUDA: {input_device}")

    with torch.no_grad():
        prediction = model.infer(tensor)
    if "depth" not in prediction:
        raise RuntimeError("Depth Pro prediction did not contain a `depth` output.")
    depth = prediction["depth"]
    depth_device = getattr(depth, "device", None)
    if depth_device is None or depth_device.type != "cuda":
        raise RuntimeError(f"Depth Pro depth output is not on CUDA before CPU conversion: {depth_device}")

    after_allocated = int(torch.cuda.memory_allocated())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    if max(after_allocated, peak_allocated) <= 0:
        raise RuntimeError("CUDA memory allocation remained zero during Depth Pro GPU check.")
    payload = {
        "status": "success",
        "image_id": row["image_id"],
        "requested_device": requested_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_parameter_device": str(model_device),
        "input_tensor_device": str(input_device),
        "depth_output_device_before_cpu": str(depth_device),
        "cuda_memory_allocated_before": before_allocated,
        "cuda_memory_allocated_after": after_allocated,
        "cuda_max_memory_allocated": peak_allocated,
    }
    write_json(payload, reports_dir(config) / "depth_gpu_check.json")
    return payload


def depth_check_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    sticker = read_parquet(registry_dir(config) / "sticker_metadata.parquet")
    merged = artifacts.merge(
        sticker[["image_id", "normalized_image_path", "raw_image_path", "scale"]],
        on="image_id",
        how="left",
    )
    success = merged[merged["segmentation_status"] == "success"].sort_values(["dataset", "image_id"])
    if should_require_pvtv2_inference(config):
        success = success[success["segmentation_source"] == "pvtv2_b2_frozen_saliency"]
    return success.to_dict(orient="records")


def depth_input_image(config: dict[str, Any], row: dict[str, Any] | pd.Series) -> Image.Image:
    depth_cfg = config.get("depth", {})
    use_raw = bool(depth_cfg.get("use_raw_image_for_inference", False))
    scale = float(row.get("scale", 1.0) or 1.0)
    if use_raw and pd.notna(row.get("raw_image_path")) and scale > 0:
        image = masked_image_for_depth(row["raw_image_path"], row["mask_path"])
    else:
        image = masked_image_for_depth(row["normalized_image_path"], row["mask_path"])
    max_side = int(depth_cfg.get("input_max_side", 0) or 0)
    if max_side > 0:
        image, _ = resize_for_depth(image, max_side)
    return image


def infer_depth_pro(config: dict[str, Any], row: pd.Series) -> tuple[np.ndarray, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Depth Pro requires torch. Install Apple's Depth Pro package or enable depth.allow_fallback.") from exc
    depth_cfg = config.get("depth", {})
    device = depth_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_depth_pro_model(str(depth_cfg.get("checkpoint_path", "")), str(device))
    use_raw = bool(depth_cfg.get("use_raw_image_for_inference", False))
    scale = float(row.get("scale", 1.0) or 1.0)
    if use_raw and pd.notna(row.get("raw_image_path")) and scale > 0:
        masked = masked_image_for_depth(row["raw_image_path"], row["mask_path"])
        output_size = Image.open(row["mask_path"]).size
        focal_scale = scale
    else:
        masked = masked_image_for_depth(row["normalized_image_path"], row["mask_path"])
        output_size = masked.size
        focal_scale = 1.0
    max_side = int(depth_cfg.get("input_max_side", 0) or 0)
    if max_side > 0:
        masked, resize_scale = resize_for_depth(masked, max_side)
        focal_scale /= resize_scale
    tensor = transform(masked)
    if hasattr(tensor, "to"):
        tensor = tensor.to(device)
    with torch.no_grad():
        prediction = model.infer(tensor)
    if "depth" not in prediction:
        raise RuntimeError("Depth Pro prediction did not contain a `depth` output.")
    depth = prediction["depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr.squeeze()
    output_max_side = int(depth_cfg.get("output_max_side", 0) or 0)
    should_restore_output_size = output_max_side <= 0 or max(output_size) <= output_max_side
    if should_restore_output_size and (depth_arr.shape[1], depth_arr.shape[0]) != output_size:
        depth_image = Image.fromarray(depth_arr)
        depth_arr = np.asarray(depth_image.resize(output_size, Image.Resampling.BILINEAR), dtype=np.float32)
    focal_value = prediction.get("focallength_px", prediction.get("focal_length_px"))
    if focal_value is None:
        raise RuntimeError("Depth Pro prediction did not include focal length in pixels.")
    if hasattr(focal_value, "detach"):
        focal_value = focal_value.detach().cpu().item()
    return depth_arr, float(focal_value) * focal_scale


def create_depth_pro_model(depth_pro: Any, device: str, checkpoint_path: str | None = None) -> tuple[Any, Any]:
    if not hasattr(depth_pro, "create_model_and_transforms"):
        raise RuntimeError("Installed depth_pro package does not expose create_model_and_transforms().")
    logger().info("Loading Depth Pro model | device=%s | checkpoint=%s", device, checkpoint_path or "default")
    if checkpoint_path:
        import depth_pro.depth_pro as depth_pro_module

        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Depth Pro checkpoint does not exist: {path}")
        config = replace(depth_pro_module.DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(path))
        model, transform = depth_pro.create_model_and_transforms(config=config)
    else:
        model, transform = depth_pro.create_model_and_transforms()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = "unknown"
    logger().info("Depth Pro model ready | requested_device=%s | model_parameter_device=%s", device, model_device)
    return model, transform


@lru_cache(maxsize=1)
def load_depth_pro_model(checkpoint_path: str, device: str) -> tuple[Any, Any]:
    try:
        import depth_pro
    except ImportError as exc:
        raise RuntimeError("Depth Pro is not installed. Install Apple's Depth Pro package or enable depth.allow_fallback.") from exc
    checkpoint = checkpoint_path or None
    return create_depth_pro_model(depth_pro, device, checkpoint)


def masked_image_for_depth(image_path: str | Path, mask_path: str | Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image) > 0
    arr = np.asarray(image).copy()
    arr[~mask] = 0
    return Image.fromarray(arr)


def resize_for_depth(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BILINEAR), scale


def fallback_depth_map(image_path: str | Path, mask_path: str | Path, max_side: int = 0) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if max_side and max(image.size) > max_side:
        image, _ = resize_for_depth(image, max_side)
    width, height = image.size
    yy, xx = np.mgrid[0:height, 0:width]
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image) > 0
    center_x = width / 2.0
    center_y = height / 2.0
    radial = np.sqrt(((xx - center_x) / max(width, 1)) ** 2 + ((yy - center_y) / max(height, 1)) ** 2)
    depth = 1.0 + radial
    depth[~mask] = 0.0
    return depth


def safe_name(image_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in image_id)
