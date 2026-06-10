from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from cattle_weight.config import REPO_ROOT
from cattle_weight.io import output_root


PRESERVED_WORKSPACE_NAMES = {
    ".git",
    "src",
    "tests",
    "configs",
    "docs",
    "README.md",
    "REPRODUCE_PAPER.md",
    "HANDOFF.md",
    "checkpoints",
}

PRESERVED_DATA_REGISTRY_NAMES = {
    "kaggle_download",
    ".kaggle",
    "kaggle.json",
}


def clean_generated(
    config: dict[str, Any],
    *,
    include_kaggle_cache: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    targets = generated_cleanup_targets(config, include_kaggle_cache=include_kaggle_cache)
    for target in targets:
        print(f"clean-generated target: {target}")
    if dry_run:
        return targets
    for target in targets:
        remove_path(target)
    return targets


def generated_cleanup_targets(config: dict[str, Any], *, include_kaggle_cache: bool = False) -> list[Path]:
    root = output_root(config)
    guard_output_root(root)
    run_id = str(config.get("run", {}).get("run_id", "run"))
    candidates = [
        root / "artifacts",
        root / "reports",
        root / "models" / run_id,
    ]
    data_registry = root / "data_registry"
    if data_registry.exists():
        if include_kaggle_cache:
            candidates.append(data_registry)
        else:
            candidates.extend(
                child
                for child in sorted(data_registry.iterdir())
                if child.name not in PRESERVED_DATA_REGISTRY_NAMES
            )
    else:
        candidates.append(data_registry)
    targets = [path.resolve() for path in candidates if path.exists()]
    for target in targets:
        guard_cleanup_target(target, root)
    return targets


def guard_output_root(root: Path) -> None:
    resolved_root = root.resolve()
    if resolved_root == REPO_ROOT.resolve():
        raise RuntimeError(
            "Refusing to clean generated artifacts with run.output_dir set to the workspace root. "
            "Use an isolated run output directory."
        )
    if is_preserved_workspace_path(resolved_root):
        raise RuntimeError(f"Refusing to clean preserved workspace path: {resolved_root}")


def guard_cleanup_target(target: Path, output_dir: Path) -> None:
    resolved = target.resolve()
    output = output_dir.resolve()
    workspace = REPO_ROOT.resolve()
    if not is_relative_to(resolved, output):
        raise RuntimeError(f"Cleanup target is outside configured output directory: {resolved}")
    if not (is_relative_to(resolved, workspace) or is_relative_to(resolved, output)):
        raise RuntimeError(f"Cleanup target is outside both workspace and configured output directory: {resolved}")
    if is_preserved_workspace_path(resolved):
        raise RuntimeError(f"Refusing to clean preserved workspace path: {resolved}")
    if resolved.name in PRESERVED_DATA_REGISTRY_NAMES:
        raise RuntimeError(f"Refusing to clean preserved data cache path without destructive flag: {resolved}")


def is_preserved_workspace_path(path: Path) -> bool:
    workspace = REPO_ROOT.resolve()
    resolved = path.resolve()
    for name in PRESERVED_WORKSPACE_NAMES:
        preserved = (workspace / name).resolve()
        if resolved == preserved or is_relative_to(resolved, preserved):
            return True
    return False


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
