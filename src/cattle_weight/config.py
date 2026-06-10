from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    default = load_yaml(DEFAULT_CONFIG)
    if path.resolve() == DEFAULT_CONFIG.resolve():
        cfg = default
    else:
        cfg = deep_merge(default, load_yaml(path))
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(config: dict[str, Any]) -> str:
    public = {k: v for k, v in config.items() if not k.startswith("_")}
    dumped = yaml.safe_dump(public, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def apply_cli_overrides(
    config: dict[str, Any],
    dataset_root: str | None = None,
    kaggle_ref: str | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    if dataset_root:
        cfg.setdefault("data", {})["local_root"] = dataset_root
    if kaggle_ref:
        cfg.setdefault("data", {})["kaggle_dataset"] = kaggle_ref
    if output_dir:
        cfg.setdefault("run", {})["output_dir"] = output_dir
    if seed is not None:
        cfg.setdefault("run", {})["random_seed"] = int(seed)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def resolve_path(config: dict[str, Any], path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    output_dir = Path(config.get("run", {}).get("output_dir", "."))
    return (REPO_ROOT / output_dir / path).resolve()
