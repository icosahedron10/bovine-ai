from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cattle_weight.config import load_config
from cattle_weight.depth.providers import load_depth_pro_model, masked_image_for_depth, resize_for_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a Depth Pro preview for one indexed image.")
    parser.add_argument("--config", default="configs/reproduction_b3_b4.yaml")
    parser.add_argument("--image-index", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/depth_previews")
    return parser.parse_args()


def depth_to_turbo_png(depth: np.ndarray) -> Image.Image:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency/environment guard
        raise RuntimeError(f"OpenCV is required to save the depth colormap: {exc}") from exc

    if depth.ndim == 3:
        depth = np.squeeze(depth)

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise RuntimeError("Depth Pro returned no finite positive depth values.")

    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        lo = float(depth[valid].min())
        hi = float(depth[valid].max())

    scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[~valid] = 0
    return Image.fromarray(colored_rgb)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    depth_cfg = config.get("depth", {})

    import torch

    requested_device = str(depth_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    cv = pd.read_parquet(Path("data_registry") / "cv_artifacts.parquet")
    sticker = pd.read_parquet(Path("data_registry") / "sticker_metadata.parquet")
    merged = cv.merge(
        sticker[["image_id", "normalized_image_path", "raw_image_path", "scale"]],
        on="image_id",
        how="left",
    )
    merged = merged[merged["segmentation_status"] == "success"].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No successful segmentation rows are available for Depth Pro preview.")
    if args.image_index < 0 or args.image_index >= len(merged):
        raise IndexError(f"image-index {args.image_index} is outside 0..{len(merged) - 1}")

    row = merged.iloc[args.image_index]
    model, transform = load_depth_pro_model(str(depth_cfg.get("checkpoint_path", "")), requested_device)

    use_raw = bool(depth_cfg.get("use_raw_image_for_inference", False))
    image_path = row["raw_image_path"] if use_raw and pd.notna(row.get("raw_image_path")) else row["normalized_image_path"]
    image = masked_image_for_depth(image_path, row["mask_path"])
    max_side = int(depth_cfg.get("input_max_side", 0) or 0)
    if max_side > 0:
        image, _ = resize_for_depth(image, max_side)

    tensor = transform(image)
    if hasattr(tensor, "to"):
        tensor = tensor.to(requested_device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        prediction = model.infer(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    depth = prediction["depth"].detach().float().cpu().numpy()
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    valid = np.isfinite(depth) & (depth > 0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_path = out_dir / f"image_{args.image_index}_depth_turbo.png"
    input_path = out_dir / f"image_{args.image_index}_masked_input.png"

    depth_to_turbo_png(depth).save(depth_path)
    image.save(input_path)

    lo, hi = np.percentile(depth[valid], [2, 98])
    print(f"image_id={row['image_id']}")
    print(f"inference_image_size={image.size[0]}x{image.size[1]}")
    print(f"depth_shape={depth.shape[1]}x{depth.shape[0]}")
    print(f"depth_min_m={float(depth[valid].min()):.6f}")
    print(f"depth_p02_m={float(lo):.6f}")
    print(f"depth_p98_m={float(hi):.6f}")
    print(f"depth_max_m={float(depth[valid].max()):.6f}")
    print(f"depth_preview={depth_path.resolve()}")
    print(f"masked_input={input_path.resolve()}")
    print(f"tensor_device={getattr(tensor, 'device', 'unknown')}")
    print(f"output_depth_device={getattr(prediction['depth'], 'device', 'unknown')}")


if __name__ == "__main__":
    main()
