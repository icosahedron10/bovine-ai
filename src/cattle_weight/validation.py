from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cattle_weight.cv.pvtv2 import PVT_SEGMENTATION_SOURCE
from cattle_weight.io import read_parquet, registry_dir
from cattle_weight.reports.status import write_headline_reproduction_status


EXPECTED_INDEX_COUNTS = {"B3": 2500, "B4": 1900}
SAMPLE_FLOOR_TRAIN = 500
SAMPLE_FLOOR_TEST = 100
DIRECT_ANIMALPOSE_PROVIDER = "mmpose_checkpoint_direct_simplebaseline"


def validate_headline(config: dict[str, Any], *, write_status: bool = True) -> dict[str, Any]:
    blocking: list[str] = []
    partial: list[str] = []
    details: dict[str, Any] = {
        "expected_segmentation_source": PVT_SEGMENTATION_SOURCE,
        "expected_depth_provider": "depth_pro",
        "expected_keypoint_provider": DIRECT_ANIMALPOSE_PROVIDER,
        "indexed_counts": {},
        "trusted_counts": {},
        "sample_floor": {},
        "missing_artifacts": [],
        "stale_artifacts": [],
        "invalid_artifacts": [],
    }

    if config.get("run", {}).get("mode") != "full":
        blocking.append(f"run.mode must be full, got {config.get('run', {}).get('mode')!r}")
    if bool(config.get("data", {}).get("synthetic_smoke", False)):
        blocking.append("data.synthetic_smoke must be false for headline validation")

    tables = load_headline_tables(config, blocking)
    if not tables:
        return write_or_return_status(config, "blocked", blocking, partial, details, write_status)

    index = tables["index"]
    splits = tables["splits"]
    cv = tables["cv"]
    keypoints = tables["keypoints"]
    depth = tables["depth"]
    pointcloud = tables["pointcloud"]
    features = tables["features"]

    validate_index_counts(index, details, partial)
    validate_success_artifacts(config, "segmentation", cv, "segmentation_status", blocking, details)
    validate_success_artifacts(config, "keypoints", keypoints, "keypoint_status", blocking, details)
    validate_success_artifacts(config, "depth", depth, "depth_status", blocking, details)
    validate_success_artifacts(config, "pointcloud", pointcloud, "pointcloud_status", blocking, details)
    validate_success_artifacts(config, "features", features, "feature_status", blocking, details)

    validate_segmentation_sources(cv, blocking, details)
    validate_keypoint_sources(keypoints, blocking, details)
    validate_depth_sources(depth, blocking, details)

    trusted = trusted_artifact_intersection(config, index, splits, cv, keypoints, depth, pointcloud, features)
    details["trusted_counts"] = {
        dataset: int(len(ids))
        for dataset, ids in trusted["trusted_ids_by_dataset"].items()
    }
    floor_reasons = validate_sample_floor(splits, trusted["trusted_ids_by_dataset"], details)
    if floor_reasons:
        if any(trusted["trusted_ids_by_dataset"].values()):
            partial.extend(floor_reasons)
        else:
            blocking.extend(floor_reasons)

    if blocking:
        status = "blocked"
        metrics_allowed = False
        reasons = blocking
    elif partial:
        status = "partial"
        metrics_allowed = False
        reasons = partial
    else:
        status = "complete"
        metrics_allowed = True
        reasons = []

    payload = {
        "status": status,
        "metrics_allowed": metrics_allowed,
        "stage": "validate headline",
        "reason": "; ".join(reasons) if reasons else None,
        "details": details,
    }
    if write_status:
        write_headline_reproduction_status(
            config,
            status,
            "validate headline",
            payload["reason"],
            metrics_allowed=metrics_allowed,
            details=details,
        )
    return payload


def load_headline_tables(config: dict[str, Any], blocking: list[str]) -> dict[str, pd.DataFrame]:
    reg = registry_dir(config)
    required_paths = {
        "index": reg / "dataset_index.parquet",
        "splits": reg / "splits.parquet",
        "cv": reg / "cv_artifacts.parquet",
        "keypoints": reg / "keypoint_metadata.parquet",
        "depth": reg / "depth_metadata.parquet",
        "pointcloud": reg / "pointcloud_metadata.parquet",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, path in required_paths.items():
        if not path.exists():
            blocking.append(f"missing required metadata: {path}")
            continue
        tables[name] = read_parquet(path)
    feature_frames = []
    for dataset in config.get("data", {}).get("datasets", ["B3", "B4"]):
        path = reg / f"features_{str(dataset).lower()}.parquet"
        if not path.exists():
            blocking.append(f"missing required feature table: {path}")
            continue
        feature_frames.append(read_parquet(path))
    if feature_frames:
        tables["features"] = pd.concat(feature_frames, ignore_index=True)
    if blocking:
        return {}
    return tables


def validate_index_counts(index: pd.DataFrame, details: dict[str, Any], partial: list[str]) -> None:
    for dataset, expected in EXPECTED_INDEX_COUNTS.items():
        count = int(index.loc[index["dataset"] == dataset, "image_id"].nunique())
        details["indexed_counts"][dataset] = count
        if count < expected:
            partial.append(f"{dataset} indexed count {count} is below required {expected}")


def validate_success_artifacts(
    config: dict[str, Any],
    artifact_name: str,
    frame: pd.DataFrame,
    status_column: str,
    blocking: list[str],
    details: dict[str, Any],
) -> None:
    if status_column not in frame.columns:
        blocking.append(f"{artifact_name} metadata missing status column {status_column}")
        return
    success = frame[frame[status_column] == "success"].copy()
    if success.empty:
        blocking.append(f"{artifact_name} metadata has no successful rows")
        return
    stale = stale_success_rows(config, success)
    if stale:
        details["stale_artifacts"].extend({"artifact": artifact_name, **item} for item in stale[:20])
        blocking.append(f"{artifact_name} metadata contains stale config_hash rows")
    path_column = artifact_path_column(artifact_name)
    if path_column:
        missing = missing_success_paths(success, path_column)
        if missing:
            details["missing_artifacts"].extend({"artifact": artifact_name, **item} for item in missing[:20])
            blocking.append(f"{artifact_name} metadata references missing artifact files")


def stale_success_rows(config: dict[str, Any], success: pd.DataFrame) -> list[dict[str, Any]]:
    current_hash = config.get("_config_hash")
    if "config_hash" not in success.columns:
        return [{"image_id": str(row.get("image_id")), "config_hash": None} for _, row in success.iterrows()]
    stale_rows: list[dict[str, Any]] = []
    for _, row in success.iterrows():
        if row.get("config_hash") == current_hash:
            continue
        if row_is_reusable_by_stable_input(row):
            continue
        stale_rows.append({"image_id": str(row.get("image_id")), "config_hash": row.get("config_hash")})
    return stale_rows


def row_is_reusable_by_stable_input(row: pd.Series | dict[str, Any]) -> bool:
    marker = row.get("reusable_by_stable_input_config", False)
    stable_hash = row.get("stable_input_config_hash")
    return bool(marker) and stable_hash not in {None, "", "nan"}


def artifact_path_column(artifact_name: str) -> str | None:
    return {
        "segmentation": "mask_path",
        "keypoints": "keypoint_path",
        "depth": "depth_map_path",
        "pointcloud": "pointcloud_path",
    }.get(artifact_name)


def missing_success_paths(success: pd.DataFrame, path_column: str) -> list[dict[str, Any]]:
    if path_column not in success.columns:
        return [{"image_id": str(row.get("image_id")), "path": None} for _, row in success.iterrows()]
    missing = []
    for _, row in success.iterrows():
        path_value = row.get(path_column)
        if not path_value or pd.isna(path_value) or not Path(str(path_value)).exists():
            missing.append({"image_id": str(row.get("image_id")), "path": str(path_value)})
    return missing


def validate_segmentation_sources(cv: pd.DataFrame, blocking: list[str], details: dict[str, Any]) -> None:
    success = cv[cv["segmentation_status"] == "success"]
    invalid = sorted(set(success.loc[success["segmentation_source"] != PVT_SEGMENTATION_SOURCE, "segmentation_source"].astype(str)))
    if invalid:
        details["invalid_artifacts"].append({"artifact": "segmentation", "invalid_sources": invalid})
        blocking.append(f"successful segmentation rows must use {PVT_SEGMENTATION_SOURCE}; found {invalid}")


def validate_keypoint_sources(keypoints: pd.DataFrame, blocking: list[str], details: dict[str, Any]) -> None:
    success = keypoints[keypoints["keypoint_status"] == "success"]
    if "provider" not in success.columns:
        blocking.append("keypoint metadata is missing provider")
        return
    invalid = sorted(set(success.loc[success["provider"] != DIRECT_ANIMALPOSE_PROVIDER, "provider"].astype(str)))
    if invalid:
        details["invalid_artifacts"].append({"artifact": "keypoints", "invalid_providers": invalid})
        blocking.append(f"successful keypoint rows must use direct AnimalPose checkpoint provider; found {invalid}")


def validate_depth_sources(depth: pd.DataFrame, blocking: list[str], details: dict[str, Any]) -> None:
    success = depth[depth["depth_status"] == "success"]
    if "depth_provider" not in success.columns:
        blocking.append("depth metadata is missing depth_provider")
        return
    invalid = sorted(set(success.loc[success["depth_provider"] != "depth_pro", "depth_provider"].astype(str)))
    if invalid:
        details["invalid_artifacts"].append({"artifact": "depth", "invalid_providers": invalid})
        blocking.append(f"successful depth rows must use depth_provider=depth_pro; found {invalid}")


def trusted_artifact_intersection(
    config: dict[str, Any],
    index: pd.DataFrame,
    splits: pd.DataFrame,
    cv: pd.DataFrame,
    keypoints: pd.DataFrame,
    depth: pd.DataFrame,
    pointcloud: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    current_hash = config.get("_config_hash")
    trusted_cv = trusted_ids(cv, "segmentation_status", current_hash)
    if "segmentation_source" in cv.columns:
        trusted_cv &= set(cv.loc[cv["segmentation_source"] == PVT_SEGMENTATION_SOURCE, "image_id"].astype(str))
    else:
        trusted_cv = set()
    trusted_keypoints = trusted_ids(keypoints, "keypoint_status", current_hash)
    if "provider" in keypoints.columns:
        trusted_keypoints &= set(keypoints.loc[keypoints["provider"] == DIRECT_ANIMALPOSE_PROVIDER, "image_id"].astype(str))
    else:
        trusted_keypoints = set()
    trusted_depth = trusted_ids(depth, "depth_status", current_hash)
    if "depth_provider" in depth.columns:
        trusted_depth &= set(depth.loc[depth["depth_provider"] == "depth_pro", "image_id"].astype(str))
    else:
        trusted_depth = set()
    trusted_pointcloud = trusted_ids(pointcloud, "pointcloud_status", current_hash)
    trusted_features = trusted_ids(features, "feature_status", current_hash)
    split_ids = set(splits["image_id"].astype(str))

    by_dataset: dict[str, set[str]] = {}
    for dataset in config.get("data", {}).get("datasets", ["B3", "B4"]):
        index_ids = set(index.loc[index["dataset"] == dataset, "image_id"].astype(str))
        by_dataset[str(dataset)] = (
            index_ids
            & split_ids
            & trusted_cv
            & trusted_keypoints
            & trusted_depth
            & trusted_pointcloud
            & trusted_features
        )
    return {"trusted_ids_by_dataset": by_dataset}


def trusted_ids(frame: pd.DataFrame, status_column: str, current_hash: str | None) -> set[str]:
    if status_column not in frame.columns or "image_id" not in frame.columns:
        return set()
    success = frame[frame[status_column] == "success"]
    if "config_hash" not in success.columns:
        return set()
    rows = []
    for _, row in success.iterrows():
        if row.get("config_hash") == current_hash or row_is_reusable_by_stable_input(row):
            rows.append(str(row["image_id"]))
    return set(rows)


def validate_sample_floor(splits: pd.DataFrame, trusted_ids_by_dataset: dict[str, set[str]], details: dict[str, Any]) -> list[str]:
    reasons = []
    for dataset in ["B3", "B4"]:
        ids = trusted_ids_by_dataset.get(dataset, set())
        part = splits[(splits["dataset"] == dataset) & (splits["image_id"].astype(str).isin(ids))]
        train_count = int(part.loc[part["split"] == "train", "image_id"].nunique())
        test_count = int(part.loc[part["split"] == "test", "image_id"].nunique())
        details["sample_floor"][dataset] = {
            "trusted_train": train_count,
            "trusted_test": test_count,
            "required_train": SAMPLE_FLOOR_TRAIN,
            "required_test": SAMPLE_FLOOR_TEST,
        }
        if train_count < SAMPLE_FLOOR_TRAIN:
            reasons.append(f"{dataset} trusted train rows {train_count} below sample floor {SAMPLE_FLOOR_TRAIN}")
        if test_count < SAMPLE_FLOOR_TEST:
            reasons.append(f"{dataset} trusted test rows {test_count} below sample floor {SAMPLE_FLOOR_TEST}")
    return reasons


def write_or_return_status(
    config: dict[str, Any],
    status: str,
    blocking: list[str],
    partial: list[str],
    details: dict[str, Any],
    write_status: bool,
) -> dict[str, Any]:
    reasons = blocking or partial
    payload = {
        "status": status,
        "metrics_allowed": False,
        "stage": "validate headline",
        "reason": "; ".join(reasons) if reasons else None,
        "details": details,
    }
    if write_status:
        write_headline_reproduction_status(
            config,
            status,
            "validate headline",
            payload["reason"],
            metrics_allowed=False,
            details=details,
        )
    return payload
