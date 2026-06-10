from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cattle_weight.config import REPO_ROOT


def require_pose_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torchvision
    except ImportError as exc:
        raise RuntimeError(f"Direct AnimalPose inference requires torch and torchvision. Missing: {exc.name}") from exc
    return torch, nn, torchvision


def build_simplebaseline_res50() -> Any:
    torch, nn, torchvision = require_pose_torch()

    class KeypointHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.final_layer = nn.Conv2d(256, 20, kernel_size=1, stride=1, padding=0)

        def forward(self, x: Any) -> Any:
            return self.final_layer(self.deconv_layers(x))

    class SimpleBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torchvision.models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()
            self.keypoint_head = KeypointHead()

        def forward(self, x: Any) -> Any:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return self.keypoint_head(x)

    return SimpleBaseline()


@lru_cache(maxsize=2)
def load_direct_animalpose_model(checkpoint_path: str, device: str) -> Any:
    torch, _, _ = require_pose_torch()
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"AnimalPose checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model = build_simplebaseline_res50()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_direct_animalpose(
    image_path: str | Path,
    bbox: np.ndarray,
    checkpoint_path: str,
    device: str,
) -> np.ndarray:
    torch, _, _ = require_pose_torch()
    model = load_direct_animalpose_model(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = expand_bbox(bbox, image.width, image.height)
    crop = image.crop((x0, y0, x1, y1)).resize((256, 256), Image.Resampling.BILINEAR)
    tensor = image_to_tensor(crop, torch).to(device)
    with torch.no_grad():
        heatmaps = model(tensor[None, :, :, :])[0].detach().cpu().numpy()
    points = heatmap_argmax_to_image_points(heatmaps, (x0, y0, x1, y1))
    return points


def infer_direct_animalpose_batch(
    image_paths: list[str],
    bboxes: list[np.ndarray],
    checkpoint_path: str,
    device: str,
    batch_size: int = 16,
) -> list[np.ndarray]:
    torch, _, _ = require_pose_torch()
    model = load_direct_animalpose_model(checkpoint_path, device)
    outputs: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_boxes = bboxes[start : start + batch_size]
        tensors = []
        crop_boxes: list[tuple[int, int, int, int]] = []
        for image_path, bbox in zip(batch_paths, batch_boxes):
            image = Image.open(image_path).convert("RGB")
            crop_box = expand_bbox(bbox, image.width, image.height)
            crop = image.crop(crop_box).resize((256, 256), Image.Resampling.BILINEAR)
            tensors.append(image_to_tensor(crop, torch))
            crop_boxes.append(crop_box)
        with torch.no_grad():
            batch = torch.stack(tensors, dim=0).to(device)
            heatmaps_batch = model(batch).detach().cpu().numpy()
        for heatmaps, crop_box in zip(heatmaps_batch, crop_boxes):
            outputs.append(heatmap_argmax_to_image_points(heatmaps, crop_box))
    return outputs


def expand_bbox(bbox: np.ndarray, width: int, height: int, scale: float = 1.25) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    side_w = box_w * scale
    side_h = box_h * scale
    new_x0 = max(0, int(round(cx - side_w / 2.0)))
    new_x1 = min(width, int(round(cx + side_w / 2.0)))
    new_y0 = max(0, int(round(cy - side_h / 2.0)))
    new_y1 = min(height, int(round(cy + side_h / 2.0)))
    return new_x0, new_y0, max(new_x0 + 1, new_x1), max(new_y0 + 1, new_y1)


def image_to_tensor(image: Image.Image, torch: Any) -> Any:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))


def heatmap_argmax_to_image_points(heatmaps: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop_w = max(1.0, float(x1 - x0))
    crop_h = max(1.0, float(y1 - y0))
    keypoints = []
    heat_h, heat_w = heatmaps.shape[1:]
    for heatmap in heatmaps:
        flat_idx = int(np.argmax(heatmap))
        yy, xx = np.unravel_index(flat_idx, heatmap.shape)
        score = float(heatmap[yy, xx])
        image_x = x0 + (xx + 0.5) * crop_w / heat_w
        image_y = y0 + (yy + 0.5) * crop_h / heat_h
        keypoints.append((image_x, image_y, score))
    return np.asarray(keypoints, dtype=float)
