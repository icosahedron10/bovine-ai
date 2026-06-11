from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cattle_weight.config import apply_cli_overrides, load_config
from cattle_weight.cv.keypoints import run_keypoints
from cattle_weight.cv.pointcloud import run_pointcloud
from cattle_weight.cv.segmentation import (
    run_mask_freeze,
    run_mask_review,
    run_mask_sanity_checks,
    run_segmentation,
)
from cattle_weight.cv.sticker import run_sticker_normalization
from cattle_weight.data.indexing import build_dataset_index
from cattle_weight.data.splits import create_splits
from cattle_weight.depth.providers import run_depth, run_depth_gpu_check
from cattle_weight.features.extraction import extract_features
from cattle_weight.io import ensure_base_dirs
from cattle_weight.maintenance import clean_generated
from cattle_weight.models.training import evaluate_models, train_models
from cattle_weight.pipeline import run_reproduction_full, run_reproduction_smoke
from cattle_weight.progress import configure_logging, log_stage_end, log_stage_start
from cattle_weight.reports.depth import write_depth_completion_summary
from cattle_weight.validation import validate_headline


app = typer.Typer(help="Paper-guided cattle weight reproduction CLI.")
data_app = typer.Typer(help="Dataset indexing commands.")
split_app = typer.Typer(help="Train/test split commands.")
cv_app = typer.Typer(help="Computer-vision artifact commands.")
features_app = typer.Typer(help="Feature extraction commands.")
run_app = typer.Typer(help="End-to-end reproduction commands.")
maintenance_app = typer.Typer(help="Maintenance commands.")
validate_app = typer.Typer(help="Validation commands.")

app.add_typer(data_app, name="data")
app.add_typer(split_app, name="split")
app.add_typer(cv_app, name="cv")
app.add_typer(features_app, name="features")
app.add_typer(run_app, name="run")
app.add_typer(maintenance_app, name="maintenance")
app.add_typer(validate_app, name="validate")


def prepare_config(
    config: Path,
    dataset_root: Optional[str],
    kaggle_ref: Optional[str],
    output_dir: Optional[str],
    seed: Optional[int],
) -> dict:
    cfg = load_config(config)
    cfg = apply_cli_overrides(cfg, dataset_root=dataset_root, kaggle_ref=kaggle_ref, output_dir=output_dir, seed=seed)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    ensure_base_dirs(cfg)
    return cfg


@data_app.command("index")
def data_index(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("data index")
    build_dataset_index(cfg, dataset_root=dataset_root)
    log_stage_end("data index", started)


@split_app.command("create")
def split_create(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("split create")
    create_splits(cfg)
    log_stage_end("split create", started)


@cv_app.command("sticker-normalize")
def cv_sticker_normalize(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("sticker normalize")
    run_sticker_normalization(cfg)
    log_stage_end("sticker normalize", started)


@cv_app.command("segment")
def cv_segment(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("segmentation")
    run_segmentation(cfg)
    log_stage_end("segmentation", started)


@cv_app.command("mask-sanity")
def cv_mask_sanity(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("mask sanity")
    run_mask_sanity_checks(cfg)
    log_stage_end("mask sanity", started)


@cv_app.command("mask-review")
def cv_mask_review(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("mask review")
    run_mask_review(cfg)
    log_stage_end("mask review", started)


@cv_app.command("mask-freeze")
def cv_mask_freeze(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    accepted_by: str = typer.Option("human_review", "--accepted-by"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("mask freeze")
    run_mask_freeze(cfg, accepted_by=accepted_by)
    log_stage_end("mask freeze", started)


@cv_app.command("keypoints")
def cv_keypoints(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("keypoints")
    run_keypoints(cfg)
    log_stage_end("keypoints", started)


@cv_app.command("depth")
def cv_depth(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("depth")
    run_depth(cfg)
    log_stage_end("depth", started)


@cv_app.command("depth-gpu-check")
def cv_depth_gpu_check(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    image_index: int = typer.Option(0, "--image-index"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("depth gpu check")
    payload = run_depth_gpu_check(cfg, image_index=image_index)
    log_stage_end("depth gpu check", started)
    typer.echo(f"Depth GPU check passed for image_id={payload['image_id']}")


@cv_app.command("depth-summary")
def cv_depth_summary(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("depth summary")
    write_depth_completion_summary(cfg)
    log_stage_end("depth summary", started)


@cv_app.command("pointcloud")
def cv_pointcloud(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("pointcloud")
    run_pointcloud(cfg)
    log_stage_end("pointcloud", started)


@features_app.command("extract")
def features_extract(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("feature extraction")
    extract_features(cfg)
    log_stage_end("feature extraction", started)


@app.command("train")
def train(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("train models")
    train_models(cfg)
    log_stage_end("train models", started)


@app.command("evaluate")
def evaluate(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("evaluate models")
    evaluate_models(cfg)
    log_stage_end("evaluate models", started)


@maintenance_app.command("clean-generated")
def maintenance_clean_generated(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    include_kaggle_cache: bool = typer.Option(False, "--include-kaggle-cache"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("clean generated")
    clean_generated(cfg, include_kaggle_cache=include_kaggle_cache, dry_run=dry_run)
    log_stage_end("clean generated", started)


@validate_app.command("headline")
def validate_headline_command(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    started = log_stage_start("validate headline")
    payload = validate_headline(cfg, write_status=True)
    log_stage_end("validate headline", started)
    typer.echo(f"headline status={payload['status']} metrics_allowed={payload['metrics_allowed']}")
    if not payload["metrics_allowed"]:
        raise typer.Exit(code=1)


@run_app.command("reproduction-smoke")
def reproduction_smoke(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    run_reproduction_smoke(cfg)


@run_app.command("reproduction-full")
def reproduction_full(
    config: Path = typer.Option(Path("configs/reproduction_b3_b4.yaml"), "--config", "-c"),
    dataset_root: Optional[str] = typer.Option(None, "--dataset-root"),
    kaggle_ref: Optional[str] = typer.Option(None, "--kaggle-ref"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    cfg = prepare_config(config, dataset_root, kaggle_ref, output_dir, seed)
    run_reproduction_full(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
