from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cattle_weight.config import REPO_ROOT


def generate_smoke_dataset(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {})
    root_value = data_cfg.get("local_root") or data_cfg.get("smoke_root") or "data_registry/smoke_dataset"
    root = Path(root_value)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    count = int(data_cfg.get("smoke_images_per_dataset", 16))
    seed = int(config.get("run", {}).get("random_seed", 0))
    rng = np.random.default_rng(seed)
    for dataset in data_cfg.get("datasets", ["B3", "B4"]):
        images_dir = root / dataset / "images"
        masks_dir = root / dataset / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            width = 260 + int(rng.integers(-12, 13))
            height = 180 + int(rng.integers(-8, 9))
            body_w = 105 + idx * 5 + (6 if dataset == "B4" else 0)
            body_h = 58 + idx * 2 + (4 if dataset == "B4" else 0)
            weight = int(round(80 + body_w * 1.05 + body_h * 0.9 + (idx % 4) * 3))
            if dataset == "B4":
                weight = int(round(weight * 0.82))

            image = Image.new("RGB", (width, height), (238, 241, 235))
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(image)
            mask_draw = ImageDraw.Draw(mask)
            x0 = int((width - body_w) / 2)
            y0 = int((height - body_h) / 2)
            x1 = x0 + body_w
            y1 = y0 + body_h
            body_box = (x0, y0, x1, y1)
            head_box = (x1 - 18, y0 + 10, x1 + 28, y0 + 42)
            cattle_color = (92, 79, 65)
            draw.ellipse(body_box, fill=cattle_color)
            draw.ellipse(head_box, fill=cattle_color)
            mask_draw.ellipse(body_box, fill=255)
            mask_draw.ellipse(head_box, fill=255)
            for leg_x in [x0 + 20, x0 + 42, x1 - 38, x1 - 18]:
                draw.rectangle((leg_x, y1 - 8, leg_x + 10, min(height - 8, y1 + 44)), fill=cattle_color)
                mask_draw.rectangle((leg_x, y1 - 8, leg_x + 10, min(height - 8, y1 + 44)), fill=255)

            sticker_radius = 10 + (idx % 3)
            sticker_center = (x0 + int(body_w * 0.58), y0 + int(body_h * 0.42))
            sx, sy = sticker_center
            sticker_box = (
                sx - sticker_radius,
                sy - sticker_radius,
                sx + sticker_radius,
                sy + sticker_radius,
            )
            draw.ellipse(sticker_box, fill=(220, 35, 35), outline=(255, 255, 255), width=1)

            name = f"{dataset}_weight_{weight:03d}_img_{idx:03d}.png"
            image.save(images_dir / name)
            mask.save(masks_dir / name.replace(".png", "_mask.png"))
    return root
