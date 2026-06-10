from __future__ import annotations

from typing import Any

from cattle_weight.cv.keypoints import run_keypoints
from cattle_weight.cv.pointcloud import run_pointcloud
from cattle_weight.cv.segmentation import run_mask_review, run_mask_sanity_checks, run_segmentation
from cattle_weight.cv.sticker import run_sticker_normalization
from cattle_weight.data.indexing import build_dataset_index
from cattle_weight.data.splits import create_splits
from cattle_weight.depth.providers import run_depth, run_depth_gpu_check
from cattle_weight.features.extraction import extract_features
from cattle_weight.io import ensure_base_dirs
from cattle_weight.models.training import evaluate_models, train_models
from cattle_weight.progress import configure_logging, run_logged_stage
from cattle_weight.validation import validate_headline


def run_reproduction_smoke(config: dict[str, Any]) -> None:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    ensure_base_dirs(config)
    run_logged_stage("data index", lambda: build_dataset_index(config))
    run_logged_stage("split create", lambda: create_splits(config))
    run_logged_stage("sticker normalize", lambda: run_sticker_normalization(config))
    run_logged_stage("segmentation", lambda: run_segmentation(config))
    run_logged_stage("mask sanity", lambda: run_mask_sanity_checks(config))
    run_logged_stage("keypoints", lambda: run_keypoints(config))
    run_logged_stage("depth", lambda: run_depth(config))
    run_logged_stage("pointcloud", lambda: run_pointcloud(config))
    run_logged_stage("feature extraction", lambda: extract_features(config))
    run_logged_stage("train models", lambda: train_models(config))
    run_logged_stage("evaluate models", lambda: evaluate_models(config))


def run_reproduction_full(config: dict[str, Any]) -> None:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    ensure_base_dirs(config)
    run_logged_stage("data index", lambda: build_dataset_index(config))
    run_logged_stage("split create", lambda: create_splits(config))
    run_logged_stage("sticker normalize", lambda: run_sticker_normalization(config))
    run_logged_stage("segmentation", lambda: run_segmentation(config))
    run_logged_stage("mask sanity", lambda: run_mask_sanity_checks(config))
    run_logged_stage("mask review", lambda: run_mask_review(config))
    run_logged_stage("keypoints", lambda: run_keypoints(config))
    run_logged_stage("depth gpu check", lambda: run_depth_gpu_check(config, image_index=0))
    run_logged_stage("depth", lambda: run_depth(config))
    run_logged_stage("pointcloud", lambda: run_pointcloud(config))
    run_logged_stage("feature extraction", lambda: extract_features(config))
    run_logged_stage("train models", lambda: train_models(config))
    run_logged_stage("evaluate models", lambda: evaluate_models(config))
    run_logged_stage("validate headline", lambda: validate_headline(config, write_status=True))
