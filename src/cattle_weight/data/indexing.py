from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cattle_weight.config import REPO_ROOT
from cattle_weight.data.smoke import generate_smoke_dataset
from cattle_weight.io import figures_dir, registry_dir, reports_dir, version_from_frame, write_parquet


WEIGHT_PATTERNS = [
    re.compile(r"(?:weight|wt|kg)[_\-\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*kg", re.IGNORECASE),
    re.compile(r"_(?:s|side)_(\d+(?:\.\d+)?)_", re.IGNORECASE),
]
FALLBACK_WEIGHT_PATTERN = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d+)?)(?!\d)")


def parse_weight_from_filename(path: str | Path) -> float:
    stem = Path(path).stem
    for pattern in WEIGHT_PATTERNS:
        match = pattern.search(stem)
        if match:
            weight = float(match.group(1))
            if weight <= 0:
                raise ValueError(f"Parsed non-positive weight from {path}: {weight}")
            return weight
    fallback_candidates = [
        float(match.group(1))
        for match in FALLBACK_WEIGHT_PATTERN.finditer(stem)
        if 20 <= float(match.group(1)) <= 1000
    ]
    if fallback_candidates:
        return fallback_candidates[0]
    raise ValueError(f"Could not parse cattle weight from filename: {path}")


def materialize_dataset_root(config: dict[str, Any], dataset_root: str | None = None) -> Path:
    data_cfg = config.get("data", {})
    datasets = [str(dataset) for dataset in data_cfg.get("datasets", ["B3", "B4"])]
    if data_cfg.get("synthetic_smoke"):
        return resolve_dataset_root(generate_smoke_dataset(config), datasets)
    root_value = dataset_root or data_cfg.get("local_root")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = REPO_ROOT / root
        if not root.exists():
            raise FileNotFoundError(f"Configured dataset root does not exist: {root}")
        return resolve_dataset_root(root, datasets)
    kaggle_ref = data_cfg.get("kaggle_dataset")
    if not kaggle_ref:
        raise RuntimeError(
            "No dataset root or Kaggle dataset reference configured. Set data.local_root "
            "or data.kaggle_dataset."
        )
    downloaded = download_kaggle_dataset(
        kaggle_ref,
        registry_dir(config) / "kaggle_download",
        datasets=datasets,
    )
    return resolve_dataset_root(downloaded, datasets)


def download_kaggle_dataset(dataset_ref: str, destination: Path, datasets: list[str] | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    requested = [str(dataset) for dataset in (datasets or []) if str(dataset)]
    if requested and extracted_dataset_dirs_exist(destination, requested):
        return destination
    if not os.environ.get("KAGGLE_USERNAME") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise RuntimeError(
            "Kaggle access requested but credentials are missing. Set KAGGLE_USERNAME and "
            "KAGGLE_KEY, or create ~/.kaggle/kaggle.json, or configure data.local_root."
        )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle access requires the kaggle package to be installed.") from exc
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=str(destination), unzip=False)
    if requested and extracted_dataset_dirs_exist(destination, requested):
        return destination
    for archive in destination.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            if requested:
                extract_requested_dataset_members(handle, destination, requested)
            else:
                extract_archive_members_safely(handle, destination, handle.infolist())
    return destination


def extracted_dataset_dirs_exist(root: Path, datasets: list[str]) -> bool:
    dataset_root = resolve_dataset_root(root, datasets)
    for dataset in datasets:
        try:
            dataset_dir = discover_dataset_dir(dataset_root, dataset)
        except FileNotFoundError:
            return False
        if not any(path.is_file() for path in dataset_dir.rglob("*")):
            return False
    return True


def extract_requested_dataset_members(handle: zipfile.ZipFile, destination: Path, datasets: list[str]) -> None:
    normalized = {dataset.lower() for dataset in datasets}
    members = [
        member
        for member in handle.infolist()
        if archive_member_matches_dataset(member.filename, normalized)
    ]
    if not members:
        raise RuntimeError(
            f"Kaggle archive did not contain requested dataset folders: {sorted(normalized)}"
        )
    for member in members:
        extract_archive_member_safely(handle, destination, member)


def extract_archive_members_safely(handle: zipfile.ZipFile, destination: Path, members: list[zipfile.ZipInfo]) -> None:
    for member in members:
        extract_archive_member_safely(handle, destination, member)


def extract_archive_member_safely(handle: zipfile.ZipFile, destination: Path, member: zipfile.ZipInfo) -> None:
    destination_root = destination.resolve()
    target_path = (destination_root / member.filename).resolve()
    try:
        target_path.relative_to(destination_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to extract archive member outside destination: {member.filename}") from exc
    if member.is_dir():
        target_path.mkdir(parents=True, exist_ok=True)
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with handle.open(member) as source, target_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def archive_member_matches_dataset(member_name: str, normalized_datasets: set[str]) -> bool:
    parts = [part.lower() for part in Path(member_name).parts]
    return "pixel" in parts and any(part in normalized_datasets for part in parts)


def resolve_dataset_root(root: Path, datasets: list[str]) -> Path:
    candidates = dataset_root_candidates(root, datasets)
    if not candidates:
        return root
    return sorted(candidates, key=dataset_root_preference)[0]


def dataset_root_candidates(root: Path, datasets: list[str]) -> list[Path]:
    normalized = {dataset.lower() for dataset in datasets}
    candidates: list[Path] = []
    if dataset_root_contains(root, normalized):
        candidates.append(root)
    for path in root.rglob("*"):
        if path.is_dir() and dataset_root_contains(path, normalized):
            candidates.append(path)
    return candidates


def dataset_root_contains(root: Path, normalized_datasets: set[str]) -> bool:
    try:
        child_names = {child.name.lower() for child in root.iterdir() if child.is_dir()}
    except OSError:
        return False
    return normalized_datasets.issubset(child_names)


def dataset_root_preference(path: Path) -> tuple[int, int, str]:
    parts = [part.lower() for part in path.parts]
    if path.name.lower() == "pixel":
        kind = 0
    elif "pixel" in parts:
        kind = 1
    else:
        kind = 2
    return kind, len(parts), str(path)


def discover_dataset_dir(root: Path, dataset: str) -> Path:
    candidates = [
        root / dataset,
        root / dataset.lower(),
        root / dataset.upper(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == dataset.lower()]
    if matches:
        return sorted(matches, key=dataset_dir_preference)[0]
    raise FileNotFoundError(f"Could not find dataset directory for {dataset} under {root}")


def dataset_dir_preference(path: Path) -> tuple[int, int, str]:
    parts = [part.lower() for part in path.parts]
    if path.parent.name.lower() == "pixel":
        kind = 0
    elif "pixel" in parts:
        kind = 1
    elif "vector" in parts:
        kind = 3
    else:
        kind = 2
    return kind, len(parts), str(path)


def discover_images(dataset_dir: Path, extensions: list[str]) -> list[Path]:
    normalized = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in normalized
        and not any(part.lower() in {"masks", "mask", "annotations", "labels"} for part in path.parts)
        and "mask" not in path.stem.lower()
        and is_side_view_image(path)
    )


def build_mask_lookup(dataset_dir: Path, keywords: list[str]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        lowered_parent = str(path.parent).lower()
        if "annotation" not in lowered_parent and not any(keyword.lower() in lowered_name for keyword in keywords):
            continue
        if lowered_name.endswith("___fuse.png"):
            lookup[lowered_name[: -len("___fuse.png")]] = path
        if lowered_name.endswith("_mask.png"):
            lookup[f"{lowered_name[: -len('_mask.png')]}.png"] = path
        if lowered_name.endswith("_mask.jpg"):
            lookup[f"{lowered_name[: -len('_mask.jpg')]}.jpg"] = path
        if lowered_name.endswith("_mask.jpeg"):
            lookup[f"{lowered_name[: -len('_mask.jpeg')]}.jpeg"] = path
        lookup[lowered_name] = path
    return lookup


def discover_mask_for_image(
    image_path: Path,
    dataset_dir: Path,
    keywords: list[str],
    mask_lookup: dict[str, Path] | None = None,
) -> Path | None:
    if mask_lookup is not None:
        direct = mask_lookup.get(image_path.name.lower())
        if direct:
            return direct
        fuse = mask_lookup.get(f"{image_path.name.lower()}___fuse.png")
        if fuse:
            return fuse
    image_name = image_path.name.lower()
    expected_fuse = f"{image_path.name}___fuse.png".lower()
    candidates: list[Path] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        lowered_parent = str(path.parent).lower()
        if lowered_name == expected_fuse:
            candidates.append(path)
        elif image_name in lowered_name and any(
            keyword.lower() in lowered_name or keyword.lower() in lowered_parent for keyword in keywords
        ):
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def is_side_view_image(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if {"rear", "back", "back_2"} & lowered_parts:
        return False
    name = path.name.lower()
    return "_s_" in name or "_side_" in name or "side" in lowered_parts or "b3" in lowered_parts or "b4" in lowered_parts


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def build_dataset_index(config: dict[str, Any], dataset_root: str | None = None) -> pd.DataFrame:
    root = materialize_dataset_root(config, dataset_root)
    data_cfg = config.get("data", {})
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in data_cfg.get("datasets", ["B3", "B4"]):
        dataset_dir = discover_dataset_dir(root, dataset)
        images = discover_images(dataset_dir, data_cfg.get("image_extensions", [".jpg", ".png"]))
        mask_lookup = build_mask_lookup(dataset_dir, data_cfg.get("mask_keywords", ["mask"]))
        if not images:
            failures.append({"dataset": dataset, "path": str(dataset_dir), "reason": "no_images_found"})
        for image_path in images:
            try:
                weight = parse_weight_from_filename(image_path)
                width, height = read_image_size(image_path)
                mask_path = discover_mask_for_image(
                    image_path,
                    dataset_dir,
                    data_cfg.get("mask_keywords", ["mask"]),
                    mask_lookup,
                )
            except Exception as exc:
                failures.append({"dataset": dataset, "path": str(image_path), "reason": str(exc)})
                continue
            rel = image_path.relative_to(root)
            image_id = f"{dataset}:{rel.as_posix()}"
            rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "weight_kg": float(weight),
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()) if mask_path else None,
                    "width": int(width),
                    "height": int(height),
                    "source_root": str(root.resolve()),
                    "status": "indexed",
                }
            )
    index = pd.DataFrame(rows)
    if index.empty:
        raise RuntimeError(f"No valid images indexed from {root}. Failures: {failures[:5]}")
    index = apply_dataset_sampling(index, config)
    index["data_version"] = version_from_frame(index, ["image_id", "weight_kg", "image_path", "mask_path", "width", "height"])
    reg = registry_dir(config)
    write_parquet(index, reg / "dataset_index.parquet")
    pd.DataFrame(failures).to_csv(reg / "index_failures.csv", index=False)
    write_dataset_summary(index, config)
    return index


def apply_dataset_sampling(index: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    per_dataset = data_cfg.get("sample_per_dataset")
    if per_dataset in {None, ""}:
        return index.reset_index(drop=True)
    sample_size = int(per_dataset)
    if sample_size <= 0:
        raise ValueError(f"data.sample_per_dataset must be positive, got {sample_size}")
    seed = int(config.get("run", {}).get("random_seed", 0))
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    for dataset in sorted(index["dataset"].dropna().unique()):
        part = index[index["dataset"] == dataset].sort_values("image_id").reset_index(drop=True)
        if len(part) <= sample_size:
            sampled = part
        else:
            positions = np.sort(rng.choice(len(part), size=sample_size, replace=False))
            sampled = part.iloc[positions]
        sampled_parts.append(sampled)
    if not sampled_parts:
        return index.iloc[0:0].reset_index(drop=True)
    return pd.concat(sampled_parts, ignore_index=True)


def write_dataset_summary(index: pd.DataFrame, config: dict[str, Any]) -> None:
    rows = []
    for dataset, part in index.groupby("dataset"):
        dims = part[["width", "height"]].describe().to_dict()
        rows.append(
            {
                "dataset": dataset,
                "image_count": int(len(part)),
                "mask_count": int(part["mask_path"].notna().sum()),
                "min_weight": float(part["weight_kg"].min()),
                "max_weight": float(part["weight_kg"].max()),
                "mean_weight": float(part["weight_kg"].mean()),
                "median_weight": float(part["weight_kg"].median()),
                "width_mean": float(dims["width"]["mean"]),
                "height_mean": float(dims["height"]["mean"]),
                "width_min": int(part["width"].min()),
                "width_max": int(part["width"].max()),
                "height_min": int(part["height"].min()),
                "height_max": int(part["height"].max()),
            }
        )
    summary = pd.DataFrame(rows)
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir(config) / "b3_b4_dataset_summary.csv", index=False)
    write_dataset_plots(index, config)


def write_dataset_plots(index: pd.DataFrame, config: dict[str, Any]) -> None:
    fig_dir = figures_dir(config)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dataset, part in index.groupby("dataset"):
        plt.figure(figsize=(6, 4))
        plt.hist(part["weight_kg"], bins=min(20, max(4, len(part) // 2)), color="#2c7a7b", edgecolor="white")
        plt.xlabel("Weight (kg)")
        plt.ylabel("Images")
        plt.title(f"{dataset} weight distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / f"weight_distribution_{dataset.lower()}.png")
        plt.close()
    plt.figure(figsize=(6, 4))
    plt.scatter(index["width"], index["height"], c=np.arange(len(index)), cmap="viridis", s=18)
    plt.xlabel("Image width (px)")
    plt.ylabel("Image height (px)")
    plt.title("Image dimension distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "image_dimension_distribution.png")
    plt.close()
