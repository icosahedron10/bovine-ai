from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cattle_weight.config import REPO_ROOT


def output_root(config: dict[str, Any]) -> Path:
    value = config.get("run", {}).get("output_dir", ".")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def registry_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "data_registry"


def reports_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "reports"


def figures_dir(config: dict[str, Any]) -> Path:
    return reports_dir(config) / "figures"


def models_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "models"


def artifacts_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "artifacts"


def ensure_base_dirs(config: dict[str, Any]) -> None:
    for path in [
        registry_dir(config),
        reports_dir(config),
        figures_dir(config),
        models_dir(config),
        artifacts_dir(config),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            f"Writing {path} requires pyarrow or fastparquet. Install project dependencies first."
        ) from exc


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    return pd.read_parquet(path)


def write_json(data: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(data: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def get_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def artifact_metadata(
    config: dict[str, Any],
    data_version: str | None = None,
    split_version: str | None = None,
    feature_schema_version: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run", {}).get("run_id", "run"),
        "config_hash": config.get("_config_hash"),
        "data_version": data_version,
        "split_version": split_version,
        "feature_schema_version": feature_schema_version,
        "model_parameters": model_parameters or {},
        "random_seed": config.get("run", {}).get("random_seed"),
        "code_commit": get_git_commit(),
    }


def version_from_frame(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "empty"
    selected = df[columns] if columns else df
    hashed = pd.util.hash_pandas_object(selected.sort_index(axis=1), index=False)
    return f"sha256:{int(hashed.sum()):x}"
