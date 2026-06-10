from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cattle_weight.io import read_parquet, registry_dir, reports_dir


def write_depth_completion_summary(config: dict[str, Any]) -> pd.DataFrame:
    depth = read_parquet(registry_dir(config) / "depth_metadata.parquet")
    cv_path = registry_dir(config) / "cv_artifacts.parquet"
    if cv_path.exists():
        cv = read_parquet(cv_path)
        join_cols = ["image_id"]
        extra_cols = [
            col
            for col in ["image_id", "segmentation_source", "segmentation_status", "mask_path", "mask_area_px"]
            if col in cv.columns
        ]
        if extra_cols != ["image_id"]:
            depth = depth.merge(cv[extra_cols].drop_duplicates("image_id"), on=join_cols, how="left")

    required = ["dataset", "depth_provider", "depth_status"]
    missing = [col for col in required if col not in depth.columns]
    if missing:
        raise RuntimeError(f"depth_metadata.parquet is missing required columns: {missing}")

    audited = depth.copy()
    audited["depth_map_exists"] = audited["depth_map_path"].map(lambda value: bool(value) and Path(str(value)).exists())
    audited["mask_exists"] = audited.get("mask_path", pd.Series([None] * len(audited))).map(
        lambda value: bool(value) and Path(str(value)).exists()
    )
    group_cols = [
        col
        for col in [
            "dataset",
            "depth_provider",
            "depth_status",
            "segmentation_source",
            "segmentation_status",
            "failure_reason",
        ]
        if col in audited.columns
    ]
    summary = (
        audited.groupby(group_cols, dropna=False)
        .agg(
            rows=("image_id", "count"),
            depth_map_exists=("depth_map_exists", "sum"),
            mask_exists=("mask_exists", "sum"),
        )
        .reset_index()
    )
    summary["depth_map_missing"] = summary["rows"] - summary["depth_map_exists"]
    summary["mask_missing"] = summary["rows"] - summary["mask_exists"]
    summary = summary.sort_values(["dataset", "depth_provider", "depth_status", "rows"], ascending=[True, True, True, False])
    path = reports_dir(config) / "depth_completion_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
