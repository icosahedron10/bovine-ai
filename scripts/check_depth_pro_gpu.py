from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cattle_weight.config import load_config
from cattle_weight.depth.providers import load_depth_pro_model, masked_image_for_depth, resize_for_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether Depth Pro inference is running on CUDA.")
    parser.add_argument("--config", default="configs/reproduction_b3_b4.yaml")
    parser.add_argument("--image-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    depth_cfg = config.get("depth", {})

    import torch

    print(f"torch_cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
    requested_device = str(depth_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"requested_device={requested_device}")

    cv = pd.read_parquet(Path("data_registry") / "cv_artifacts.parquet")
    sticker = pd.read_parquet(Path("data_registry") / "sticker_metadata.parquet")
    merged = cv.merge(
        sticker[["image_id", "normalized_image_path", "raw_image_path", "scale"]],
        on="image_id",
        how="left",
    )
    merged = merged[merged["segmentation_status"] == "success"].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No successful segmentation rows are available for Depth Pro GPU check.")
    row = merged.iloc[args.image_index]

    model, transform = load_depth_pro_model(str(depth_cfg.get("checkpoint_path", "")), requested_device)
    model_device = next(model.parameters()).device
    print(f"model_parameter_device={model_device}")

    use_raw = bool(depth_cfg.get("use_raw_image_for_inference", False))
    image_path = row["raw_image_path"] if use_raw and pd.notna(row.get("raw_image_path")) else row["normalized_image_path"]
    image = masked_image_for_depth(image_path, row["mask_path"])
    max_side = int(depth_cfg.get("input_max_side", 0) or 0)
    if max_side > 0:
        image, _ = resize_for_depth(image, max_side)
    print(f"image_id={row['image_id']}")
    print(f"inference_image_size={image.size[0]}x{image.size[1]}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

    transform_started = time.perf_counter()
    tensor = transform(image)
    if hasattr(tensor, "to"):
        tensor = tensor.to(requested_device)
    transform_seconds = time.perf_counter() - transform_started
    print(f"input_tensor_device={getattr(tensor, 'device', 'unknown')}")

    forward_started = time.perf_counter()
    with torch.no_grad():
        prediction = model.infer(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_started

    after_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    peak_alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    depth = prediction["depth"]
    print(f"output_depth_device={getattr(depth, 'device', 'unknown')}")
    print(f"transform_seconds={transform_seconds:.3f}")
    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"cuda_memory_allocated_before_mb={before_alloc / 1024 / 1024:.1f}")
    print(f"cuda_memory_allocated_after_mb={after_alloc / 1024 / 1024:.1f}")
    print(f"cuda_memory_reserved_before_mb={before_reserved / 1024 / 1024:.1f}")
    print(f"cuda_memory_reserved_after_mb={after_reserved / 1024 / 1024:.1f}")
    print(f"cuda_peak_memory_allocated_mb={peak_alloc / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
