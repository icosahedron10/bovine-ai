from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cattle_weight.io import read_parquet, registry_dir, version_from_frame, write_parquet


def create_splits(config: dict[str, Any]) -> pd.DataFrame:
    index = read_parquet(registry_dir(config) / "dataset_index.parquet")
    seed = int(config.get("run", {}).get("random_seed", 0))
    test_size = float(config.get("model", {}).get("test_size", 0.2))
    rows: list[dict[str, Any]] = []
    for dataset, part in index.groupby("dataset"):
        ordered = part.sort_values("image_id").reset_index(drop=True)
        rng = np.random.default_rng(seed + stable_dataset_offset(dataset))
        permuted = rng.permutation(len(ordered))
        test_count = max(1, int(round(len(ordered) * test_size)))
        if len(ordered) > 1:
            test_count = min(test_count, len(ordered) - 1)
        test_indices = set(permuted[:test_count].tolist())
        for idx, row in ordered.iterrows():
            split = "test" if idx in test_indices else "train"
            rows.append(
                {
                    "image_id": row["image_id"],
                    "dataset": dataset,
                    "split": split,
                    "random_seed": seed,
                    "test_size": test_size,
                }
            )
    splits = pd.DataFrame(rows)
    splits["split_version"] = version_from_frame(splits, ["image_id", "dataset", "split", "random_seed", "test_size"])
    write_parquet(splits, registry_dir(config) / "splits.parquet")
    return splits


def stable_dataset_offset(dataset: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(dataset))
