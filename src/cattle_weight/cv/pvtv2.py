from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


PVT_SEGMENTATION_SOURCE = "pvtv2_b2_frozen_saliency"
PVT_SEGMENTATION_ALGORITHM_VERSION = "pvtv2_saliency_grid_v2"


def require_torch_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        import timm
    except ImportError as exc:
        raise RuntimeError(
            f"Frozen PVTv2-B2 segmentation requires torch and timm. Missing package: {exc.name}"
        ) from exc
    return torch, functional, timm


def predict_pvtv2_b2_mask(image_path: str | Path, config: dict[str, Any]) -> np.ndarray:
    return predict_pvtv2_b2_masks([image_path], config)[0]


def predict_pvtv2_b2_masks(image_paths: list[str | Path], config: dict[str, Any]) -> list[np.ndarray]:
    if not image_paths:
        return []
    seg_cfg = config.get("segmentation", {})
    frozen_cfg = seg_cfg.get("frozen_pvtv2", {})
    model_name = str(frozen_cfg.get("model_name", "pvt_v2_b2"))
    pretrained = bool(frozen_cfg.get("pretrained", True))
    image_size = int(frozen_cfg.get("image_size", 384))
    batch_size = max(1, int(frozen_cfg.get("batch_size", seg_cfg.get("batch_size", 64))))

    torch, _, _ = require_torch_stack()
    device = resolve_device(torch, str(frozen_cfg.get("device", "auto")))
    torch, functional, model, device = load_frozen_pvtv2_encoder(model_name, pretrained, device)

    masks: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        original_sizes: list[tuple[int, int]] = []
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                original_size = image.size
                image.draft("RGB", (image_size, image_size))
                image = image.convert("RGB")
                original_sizes.append(original_size)
                resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                tensors.append(image_to_tensor(resized, torch))
        tensor = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            features = model(tensor)
            saliencies = saliency_batch_from_features(torch, functional, features, image_size)

        for saliency, original_size in zip(saliencies, original_sizes):
            low_res_mask = mask_from_saliency(saliency, config)
            masks.append(resize_binary_mask(low_res_mask, original_size))
    return masks


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


@lru_cache(maxsize=4)
def load_frozen_pvtv2_encoder(model_name: str, pretrained: bool, device: str) -> tuple[Any, Any, Any, str]:
    torch, functional, timm = require_torch_stack()
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
    except Exception as exc:
        weights = "pretrained weights" if pretrained else "model definition"
        raise RuntimeError(
            f"Unable to load frozen {model_name} {weights}. "
            "Full reproduction requires public PVTv2 weights to be available to timm."
        ) from exc
    model.to(device)
    model.eval()
    return torch, functional, model, device


def resolve_device(torch: Any, configured: str) -> str:
    if configured in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen PVTv2-B2 segmentation was configured for CUDA, but CUDA is not available.")
    return configured


def saliency_from_features(torch: Any, functional: Any, features: list[Any], image_size: int) -> np.ndarray:
    batch = saliency_batch_from_features(torch, functional, features, image_size)
    if len(batch) != 1:
        raise RuntimeError(f"Expected a single PVTv2 saliency map, got {len(batch)}.")
    return batch[0]


def saliency_batch_from_features(torch: Any, functional: Any, features: list[Any], image_size: int) -> list[np.ndarray]:
    if not features:
        raise RuntimeError("Frozen PVTv2-B2 produced no feature maps.")
    maps: list[np.ndarray] = []
    for feature in features:
        activation = feature.float().abs().mean(dim=1, keepdim=True)
        activation = functional.interpolate(
            activation,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        arr = activation[:, 0].detach().cpu().numpy().astype(np.float32)
        maps.append(np.stack([normalize_saliency(item, allow_flat=True) for item in arr], axis=0))
    averaged = np.mean(np.stack(maps, axis=0), axis=0)
    return [normalize_saliency(item) for item in averaged]


def mask_from_saliency(saliency: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    seg_cfg = (config or {}).get("segmentation", {})
    frozen_cfg = seg_cfg.get("frozen_pvtv2", {})
    threshold_method = str(frozen_cfg.get("threshold_method", "otsu"))
    saliency = normalize_saliency(np.asarray(saliency, dtype=np.float32))
    if threshold_method == "otsu":
        threshold = otsu_threshold(saliency)
    elif threshold_method == "fixed":
        threshold = float(frozen_cfg.get("fixed_threshold", seg_cfg.get("inference_threshold", 0.5)))
    elif threshold_method == "percentile":
        threshold = float(np.percentile(saliency, float(frozen_cfg.get("threshold_percentile", 60.0))))
    else:
        raise ValueError(f"Unsupported frozen PVTv2 threshold_method: {threshold_method}")
    return postprocess_saliency_mask(saliency >= threshold, frozen_cfg.get("postprocess", {}))


def normalize_saliency(saliency: np.ndarray, allow_flat: bool = False) -> np.ndarray:
    arr = np.asarray(saliency, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError("Frozen PVTv2-B2 saliency contains no finite values.")
    low = float(finite.min())
    high = float(finite.max())
    if high - low <= 1e-8:
        if allow_flat:
            return np.zeros_like(arr, dtype=np.float32)
        raise RuntimeError("Frozen PVTv2-B2 saliency is empty or flat; cannot derive a mask.")
    normalized = (arr - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("Cannot threshold empty frozen PVTv2 saliency.")
    hist, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = int(hist.sum())
    if total == 0:
        raise RuntimeError("Cannot threshold empty frozen PVTv2 saliency.")
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
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_var:
            best_var = variance
            best_threshold = float(center)
    return best_threshold


def postprocess_saliency_mask(mask: np.ndarray, postprocess_cfg: dict[str, Any] | None = None) -> np.ndarray:
    cfg = postprocess_cfg or {}
    out = np.asarray(mask, dtype=bool)
    if not out.any():
        raise RuntimeError("Frozen PVTv2-B2 saliency produced an empty foreground mask.")
    if bool(cfg.get("fill_holes", True)):
        out = ndimage.binary_fill_holes(out)
    if bool(cfg.get("largest_component", True)):
        labeled, count = ndimage.label(out)
        if count == 0:
            raise RuntimeError("Frozen PVTv2-B2 saliency produced no connected foreground components.")
        sizes = ndimage.sum(out, labeled, range(1, count + 1))
        largest = int(np.argmax(sizes) + 1)
        out = labeled == largest
    min_area_px = int(cfg.get("min_area_px", 1))
    if int(out.sum()) < min_area_px:
        raise RuntimeError(
            f"Frozen PVTv2-B2 saliency mask area {int(out.sum())} is below min_area_px={min_area_px}."
        )
    return out.astype(bool)


def image_to_tensor(image: Image.Image, torch: Any) -> Any:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))
