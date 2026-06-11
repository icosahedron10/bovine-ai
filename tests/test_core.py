from __future__ import annotations
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest

from cattle_weight.config import apply_cli_overrides, load_config
from cattle_weight.cv.animalpose_direct import require_pose_torch
from cattle_weight.cv.keypoints import DIRECT_ANIMALPOSE_PROVIDER, assert_pvtv2_keypoint_inputs, write_keypoint_review_overlays
from cattle_weight.cv.pointcloud import (
    assert_pvtv2_pointcloud_inputs,
    cleanup_point_cloud,
    depth_to_point_cloud,
    is_trusted_pointcloud_resume_row,
)
from cattle_weight.cv.pvtv2 import PVT_SEGMENTATION_SOURCE, mask_from_saliency, postprocess_saliency_mask
from cattle_weight.cv.segmentation import bbox_from_mask, contour_from_mask, run_mask_freeze
from cattle_weight.cv.segmentation import run_segmentation, select_mask_review_rows, validate_segmentation_masks
from cattle_weight.cv.sticker import detect_sticker, normalize_image_and_mask
from cattle_weight.data.indexing import (
    apply_dataset_sampling,
    build_dataset_index,
    extracted_dataset_dirs_exist,
    extract_requested_dataset_members,
    parse_weight_from_filename,
    resolve_dataset_root,
)
from cattle_weight.data.splits import create_splits
from cattle_weight.depth.providers import is_trusted_depth_resume_row
from cattle_weight.eval.metrics import mae, mape
from cattle_weight.features.extraction import (
    assert_feature_inputs_ready,
    beam_angles,
    body_measurement_features,
    chain_codes,
    compute_all_features,
    hu_moment_features,
    keypoint_distance_features,
    mask_convex_hull_features,
    prefix_stats,
)
from cattle_weight.features.schema import PAPER_FEATURES
from cattle_weight.io import read_json, registry_dir, reports_dir, write_json, write_parquet
from cattle_weight.maintenance import clean_generated
from cattle_weight.models.training import assert_headline_reproduction_ready, fit_nmi_selector, write_coverage_funnel, write_evaluation_headline_status
from cattle_weight.pipeline import run_reproduction_smoke
from cattle_weight.reports.depth import write_depth_completion_summary
from cattle_weight.validation import validate_headline


def smoke_config(tmp_path: Path, image_count: int = 8) -> dict:
    cfg = load_config("configs/smoke.yaml")
    cfg = apply_cli_overrides(cfg, output_dir=str(tmp_path), seed=42)
    cfg["data"]["local_root"] = str(tmp_path / "smoke_dataset")
    cfg["data"]["smoke_root"] = str(tmp_path / "smoke_dataset")
    cfg["data"]["smoke_images_per_dataset"] = image_count
    cfg["run"]["output_dir"] = str(tmp_path)
    return cfg


def strict_validation_config(tmp_path: Path) -> dict:
    cfg = load_config("configs/reproduction_b3_b4.yaml")
    cfg = apply_cli_overrides(cfg, output_dir=str(tmp_path), seed=42)
    cfg["run"]["output_dir"] = str(tmp_path)
    return cfg


def write_minimal_headline_artifacts(
    cfg: dict,
    tmp_path: Path,
    *,
    segmentation_source: str = PVT_SEGMENTATION_SOURCE,
    keypoint_provider: str = DIRECT_ANIMALPOSE_PROVIDER,
    depth_provider: str = "depth_pro",
    config_hash: str | None = None,
    missing_mask: bool = False,
) -> None:
    artifact_hash = config_hash if config_hash is not None else cfg["_config_hash"]
    index_rows = []
    split_rows = []
    cv_rows = []
    keypoint_rows = []
    depth_rows = []
    pointcloud_rows = []
    feature_rows = []
    for dataset in ["B3", "B4"]:
        for split in ["train", "test"]:
            image_id = f"{dataset}:{split}.jpg"
            mask_path = tmp_path / "artifacts" / "segmentation" / dataset / f"{split}_mask.png"
            keypoint_path = tmp_path / "artifacts" / "keypoints" / dataset / f"{split}.json"
            depth_path = tmp_path / "artifacts" / "depth" / dataset / f"{split}_depth.npy"
            pointcloud_path = tmp_path / "artifacts" / "pointclouds" / dataset / f"{split}_pointcloud.npy"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            keypoint_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            pointcloud_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (12, 10), 255).save(mask_path)
            keypoints = np.column_stack([np.arange(20), np.arange(20), np.ones(20)])
            write_json(
                {
                    "image_id": image_id,
                    "provider": keypoint_provider,
                    "provider_status": "test",
                    "keypoint_names": [f"kp_{idx}" for idx in range(20)],
                    "keypoints": keypoints.tolist(),
                },
                keypoint_path,
            )
            np.save(depth_path, np.ones((10, 12), dtype=np.float32))
            np.save(pointcloud_path, np.ones((10, 3), dtype=np.float32))
            index_rows.append({"image_id": image_id, "dataset": dataset, "weight_kg": 100.0, "data_version": "data-v1"})
            split_rows.append({"image_id": image_id, "dataset": dataset, "split": split})
            cv_rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "segmentation_status": "success",
                    "segmentation_source": segmentation_source,
                    "config_hash": artifact_hash,
                    "mask_path": str((tmp_path / "missing_mask.png") if missing_mask else mask_path),
                }
            )
            keypoint_rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "keypoint_status": "success",
                    "provider": keypoint_provider,
                    "config_hash": artifact_hash,
                    "keypoint_path": str(keypoint_path),
                }
            )
            depth_rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "depth_status": "success",
                    "depth_provider": depth_provider,
                    "config_hash": artifact_hash,
                    "depth_map_path": str(depth_path),
                }
            )
            pointcloud_rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "pointcloud_status": "success",
                    "config_hash": artifact_hash,
                    "pointcloud_path": str(pointcloud_path),
                }
            )
            feature_rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "weight_kg": 100.0,
                    "feature_status": "success",
                    "config_hash": artifact_hash,
                    **{name: 1.0 for name in PAPER_FEATURES},
                }
            )
    write_parquet(pd.DataFrame(index_rows), registry_dir(cfg) / "dataset_index.parquet")
    write_parquet(pd.DataFrame(split_rows), registry_dir(cfg) / "splits.parquet")
    write_parquet(pd.DataFrame(cv_rows), registry_dir(cfg) / "cv_artifacts.parquet")
    write_parquet(pd.DataFrame(keypoint_rows), registry_dir(cfg) / "keypoint_metadata.parquet")
    write_parquet(pd.DataFrame(depth_rows), registry_dir(cfg) / "depth_metadata.parquet")
    write_parquet(pd.DataFrame(pointcloud_rows), registry_dir(cfg) / "pointcloud_metadata.parquet")
    features = pd.DataFrame(feature_rows)
    for dataset in ["B3", "B4"]:
        write_parquet(features[features["dataset"] == dataset], registry_dir(cfg) / f"features_{dataset.lower()}.parquet")


def test_config_loading() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg["run"]["mode"] == "smoke"
    assert cfg["_config_hash"]


def test_smoke_and_strict_outputs_are_isolated() -> None:
    smoke = load_config("configs/smoke.yaml")
    strict = load_config("configs/reproduction_b3_b4.yaml")
    assert smoke["run"]["output_dir"] == "runs/smoke"
    assert strict["run"]["output_dir"] == "runs/reproduction_b3_b4_strict"


def test_clean_generated_preserves_kaggle_cache(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["run"]["run_id"] = "strict"
    for path in [
        tmp_path / "artifacts" / "old.npy",
        tmp_path / "reports" / "old.csv",
        tmp_path / "models" / "strict" / "model.joblib",
        tmp_path / "data_registry" / "cv_artifacts.parquet",
        tmp_path / "data_registry" / "kaggle_download" / "keep.txt",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    clean_generated(cfg)
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "models" / "strict").exists()
    assert not (tmp_path / "data_registry" / "cv_artifacts.parquet").exists()
    assert (tmp_path / "data_registry" / "kaggle_download" / "keep.txt").exists()


def test_kaggle_extraction_keeps_only_requested_dataset_dirs(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as handle:
        handle.writestr("root/Pixel/B3/images/1_s_100_M.jpg", "b3")
        handle.writestr("root/Pixel/B3/annotations/1_s_100_M.jpg___fuse.png", "mask")
        handle.writestr("root/Pixel/B4/Side/1_s_120_F.jpg", "b4")
        handle.writestr("root/Pixel/B2/Side/1_s_90_F.jpg", "b2")
        handle.writestr("root/Vector/B3/Side/data/1.json", "{}")
    with zipfile.ZipFile(archive_path) as handle:
        extract_requested_dataset_members(handle, tmp_path, ["B3", "B4"])
    assert extracted_dataset_dirs_exist(tmp_path, ["B3", "B4"])
    assert (tmp_path / "root" / "Pixel" / "B3" / "images" / "1_s_100_M.jpg").exists()
    assert (tmp_path / "root" / "Pixel" / "B4" / "Side" / "1_s_120_F.jpg").exists()
    assert not (tmp_path / "root" / "Pixel" / "B2" / "Side" / "1_s_90_F.jpg").exists()
    assert not (tmp_path / "root" / "Vector" / "B3" / "Side" / "data" / "1.json").exists()


def test_kaggle_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    destination = tmp_path / "extract"
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as handle:
        handle.writestr("root/Pixel/B3/../../../../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as handle:
        with pytest.raises(RuntimeError, match="outside destination"):
            extract_requested_dataset_members(handle, destination, ["B3"])
    assert not (tmp_path / "escape.txt").exists()


def test_dataset_root_resolution_prefers_pixel_over_vector(tmp_path: Path) -> None:
    pixel = tmp_path / "dataset" / "Pixel"
    vector = tmp_path / "dataset" / "Vector"
    for root in [pixel, vector]:
        for dataset in ["B3", "B4"]:
            data_dir = root / dataset / "Side" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / f"{dataset}.txt").write_text(dataset, encoding="utf-8")

    assert resolve_dataset_root(tmp_path, ["B3", "B4"]) == pixel


def test_headline_validator_marks_smoke_sized_strict_rows_partial(tmp_path: Path) -> None:
    cfg = strict_validation_config(tmp_path)
    write_minimal_headline_artifacts(cfg, tmp_path)
    payload = validate_headline(cfg)
    assert payload["status"] == "partial"
    assert payload["metrics_allowed"] is False
    assert "indexed count" in payload["reason"]


def test_evaluation_status_refuses_metrics_for_partial_strict_outputs(tmp_path: Path) -> None:
    cfg = strict_validation_config(tmp_path)
    write_minimal_headline_artifacts(cfg, tmp_path)
    write_evaluation_headline_status(cfg)
    status = read_json(reports_dir(cfg) / "headline_reproduction_status.json")
    assert status["status"] == "partial"
    assert status["metrics_allowed"] is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"segmentation_source": "existing_mask"}, "successful segmentation rows"),
        ({"segmentation_source": "provided_mask_ablation"}, "successful segmentation rows"),
        ({"keypoint_provider": "fallback_bbox_geometry"}, "successful keypoint rows"),
        ({"depth_provider": "fallback_dummy_relative_depth"}, "successful depth rows"),
        ({"config_hash": "stale"}, "stale config_hash"),
        ({"missing_mask": True}, "missing artifact files"),
    ],
)
def test_headline_validator_rejects_untrusted_artifacts(tmp_path: Path, kwargs: dict, expected: str) -> None:
    cfg = strict_validation_config(tmp_path)
    write_minimal_headline_artifacts(cfg, tmp_path, **kwargs)
    payload = validate_headline(cfg)
    assert payload["status"] == "blocked"
    assert expected in payload["reason"]


def test_depth_resume_accepts_only_trusted_depth_pro_rows(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((2, 2), dtype=np.float32))
    row = {
        "image_id": "B3:cow.jpg",
        "depth_status": "success",
        "depth_provider": "depth_pro",
        "config_hash": cfg["_config_hash"],
        "depth_map_path": str(depth_path),
    }
    assert is_trusted_depth_resume_row(row, cfg, {"B3:cow.jpg"}, "depth_pro")
    assert not is_trusted_depth_resume_row({**row, "depth_provider": "fallback_dummy_relative_depth"}, cfg, {"B3:cow.jpg"}, "depth_pro")
    assert not is_trusted_depth_resume_row({**row, "config_hash": "stale"}, cfg, {"B3:cow.jpg"}, "depth_pro")


def test_depth_completion_summary_ignores_nan_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nan").write_text("not an artifact", encoding="utf-8")
    write_parquet(
        pd.DataFrame(
            [
                {
                    "image_id": "B3:cow.jpg",
                    "dataset": "B3",
                    "depth_provider": "depth_pro",
                    "depth_status": "success",
                    "depth_map_path": np.nan,
                }
            ]
        ),
        registry_dir(cfg) / "depth_metadata.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "image_id": "B3:cow.jpg",
                    "segmentation_source": PVT_SEGMENTATION_SOURCE,
                    "segmentation_status": "success",
                    "mask_path": np.nan,
                    "mask_area_px": 1,
                }
            ]
        ),
        registry_dir(cfg) / "cv_artifacts.parquet",
    )
    summary = write_depth_completion_summary(cfg)
    assert int(summary["depth_map_exists"].iloc[0]) == 0
    assert int(summary["mask_exists"].iloc[0]) == 0


def test_pointcloud_resume_requires_trusted_upstream_and_file(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    pointcloud_path = tmp_path / "pointcloud.npy"
    np.save(pointcloud_path, np.ones((2, 3), dtype=np.float32))
    pointcloud_row = {
        "image_id": "B3:cow.jpg",
        "pointcloud_status": "success",
        "config_hash": cfg["_config_hash"],
        "pointcloud_path": str(pointcloud_path),
    }
    current_row = {"segmentation_source": PVT_SEGMENTATION_SOURCE, "depth_provider": "depth_pro"}
    assert is_trusted_pointcloud_resume_row(pointcloud_row, current_row, cfg, require_pvtv2=True, require_depth_pro=True)
    assert not is_trusted_pointcloud_resume_row(
        pointcloud_row,
        {**current_row, "depth_provider": "existing_depth"},
        cfg,
        require_pvtv2=True,
        require_depth_pro=True,
    )
    assert not is_trusted_pointcloud_resume_row(
        {**pointcloud_row, "pointcloud_path": str(tmp_path / "missing.npy")},
        current_row,
        cfg,
        require_pvtv2=True,
        require_depth_pro=True,
    )


def test_direct_animalpose_schema_and_overlay_generation(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["keypoints"]["review"] = {"random_count": 1, "area_anomaly_count": 0, "complexity_count": 0, "low_confidence_count": 0}
    image_path = tmp_path / "cow.jpg"
    keypoint_path = tmp_path / "cow_keypoints.json"
    Image.new("RGB", (80, 60), (230, 230, 230)).save(image_path)
    keypoints = np.column_stack([np.linspace(10, 70, 20), np.linspace(8, 52, 20), np.ones(20)])
    payload = {
        "image_id": "B3:cow.jpg",
        "provider": DIRECT_ANIMALPOSE_PROVIDER,
        "provider_status": "mmpose_api_bypassed_by_config",
        "keypoint_names": [f"kp_{idx}" for idx in range(20)],
        "keypoints": keypoints.tolist(),
    }
    write_json(payload, keypoint_path)
    artifacts = pd.DataFrame(
        [
            {
                "image_id": "B3:cow.jpg",
                "dataset": "B3",
                "segmentation_status": "success",
                "normalized_image_path": str(image_path),
                "bbox_x0": 5,
                "bbox_y0": 5,
                "bbox_x1": 75,
                "bbox_y1": 55,
                "mask_area_px": 1000,
                "contour_point_count": 100,
            }
        ]
    )
    keypoint_metadata = pd.DataFrame(
        [
            {
                "image_id": "B3:cow.jpg",
                "dataset": "B3",
                "keypoint_status": "success",
                "provider": DIRECT_ANIMALPOSE_PROVIDER,
                "keypoint_path": str(keypoint_path),
            }
        ]
    )
    manifest = write_keypoint_review_overlays(cfg, artifacts, keypoint_metadata)
    assert payload["provider"] == DIRECT_ANIMALPOSE_PROVIDER
    assert len(payload["keypoints"]) == 20
    assert not manifest.empty
    assert Path(manifest["overlay_path"].iloc[0]).exists()


def test_require_pose_torch_reports_importerror_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch import failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="torch import failed"):
        require_pose_torch()


def test_filename_weight_parsing() -> None:
    assert parse_weight_from_filename("B3_weight_109_img_001.png") == 109
    assert parse_weight_from_filename("cow-274kg-side.jpg") == 274


def test_dataset_indexing_fixture_and_splits(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path, image_count=6)
    index = build_dataset_index(cfg)
    assert set(index["dataset"]) == {"B3", "B4"}
    assert index["weight_kg"].notna().all()
    splits_a = create_splits(cfg)
    splits_b = create_splits(cfg)
    pd.testing.assert_frame_equal(splits_a, splits_b)


def test_dataset_sampling_is_deterministic_per_dataset(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["data"]["sample_per_dataset"] = 3
    index = pd.DataFrame(
        [
            {
                "image_id": f"{dataset}:cow_{idx}.jpg",
                "dataset": dataset,
                "weight_kg": 100 + idx,
                "image_path": str(tmp_path / f"{dataset}_{idx}.jpg"),
                "mask_path": str(tmp_path / f"{dataset}_{idx}_mask.png"),
                "width": 20,
                "height": 10,
                "source_root": str(tmp_path),
                "status": "indexed",
            }
            for dataset in ["B3", "B4"]
            for idx in range(10)
        ]
    )
    sampled_a = apply_dataset_sampling(index, cfg)
    sampled_b = apply_dataset_sampling(index, cfg)
    assert sampled_a["dataset"].value_counts().to_dict() == {"B3": 3, "B4": 3}
    pd.testing.assert_frame_equal(sampled_a, sampled_b)


def test_frozen_saliency_postprocess_fills_holes_and_keeps_largest_component() -> None:
    mask = np.zeros((48, 48), dtype=bool)
    mask[6:34, 8:38] = True
    mask[16:22, 18:26] = False
    mask[40:44, 40:44] = True
    out = postprocess_saliency_mask(mask, {"fill_holes": True, "largest_component": True, "min_area_px": 1})
    assert out[18, 20]
    assert not out[41, 41]
    assert out.sum() == 28 * 30


def test_frozen_saliency_rejects_flat_maps() -> None:
    with pytest.raises(RuntimeError, match="empty or flat"):
        mask_from_saliency(np.ones((12, 12), dtype=np.float32), {"segmentation": {"frozen_pvtv2": {"threshold_method": "otsu"}}})


def test_mask_sanity_rejects_non_pvtv2_source(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((20, 20, 80, 80), fill=255)
    mask.save(mask_path)
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["allow_fallback"] = False
    cfg["segmentation"]["require_pvtv2_inference"] = True
    artifacts = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "dataset_version": "data-v1",
                "transform_version": "transform-v1",
                "config_hash": "config-v1",
                "segmentation_source": "provided_mask_ablation",
                "segmentation_status": "success",
                "mask_path": str(mask_path),
                "mask_area_px": 3721,
                "bbox_x0": 20,
                "bbox_y0": 20,
                "bbox_x1": 80,
                "bbox_y1": 80,
                "contour_point_count": 240,
            }
        ]
    )
    sticker = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "detection_status": "success",
                "normalized_width": 100,
                "normalized_height": 100,
            }
        ]
    )
    summary = validate_segmentation_masks(cfg, artifacts, sticker)
    assert summary["sanity_status"].iloc[0] == "failed"
    assert "segmentation_source=provided_mask_ablation" in summary["failure_reason"].iloc[0]


def test_provided_mask_baseline_copies_normalized_masks(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["provider"] = "provided_mask_baseline"
    cfg["segmentation"]["require_pvtv2_inference"] = False
    cfg["segmentation"]["overwrite_existing"] = True
    image_path = tmp_path / "normalized.jpg"
    mask_path = tmp_path / "normalized_mask.png"
    Image.new("RGB", (32, 24), (220, 220, 220)).save(image_path)
    mask = Image.new("L", (32, 24), 0)
    ImageDraw.Draw(mask).rectangle((4, 5, 20, 18), fill=255)
    mask.save(mask_path)
    sticker = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "data_version": "data-v1",
                "detection_status": "success",
                "normalized_width": 32,
                "normalized_height": 24,
                "normalized_image_path": str(image_path),
                "normalized_mask_path": str(mask_path),
                "transform_version": cfg["_config_hash"],
            }
        ]
    )
    write_parquet(sticker, tmp_path / "data_registry" / "sticker_metadata.parquet")
    artifacts = run_segmentation(cfg)
    assert artifacts["segmentation_source"].iloc[0] == "provided_mask_baseline"
    copied = Path(artifacts["mask_path"].iloc[0])
    assert copied.exists()
    assert np.array_equal(np.asarray(Image.open(mask_path).convert("L")), np.asarray(Image.open(copied).convert("L")))


def test_frozen_pvtv2_segmentation_writes_source_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["allow_fallback"] = False
    cfg["segmentation"]["use_provided_masks_as_fallback"] = False
    cfg["segmentation"]["require_pvtv2_inference"] = True
    cfg["segmentation"]["overwrite_existing"] = True
    image_path = tmp_path / "normalized.jpg"
    Image.new("RGB", (32, 24), (220, 220, 220)).save(image_path)
    sticker = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "data_version": "data-v1",
                "detection_status": "success",
                "normalized_width": 32,
                "normalized_height": 24,
                "normalized_image_path": str(image_path),
                "normalized_mask_path": np.nan,
                "transform_version": cfg["_config_hash"],
            }
        ]
    )
    write_parquet(sticker, tmp_path / "data_registry" / "sticker_metadata.parquet")

    def fake_predict(image_path: str | Path, config: dict) -> np.ndarray:
        mask = np.zeros((24, 32), dtype=bool)
        mask[4:20, 5:25] = True
        return mask

    monkeypatch.setattr("cattle_weight.cv.segmentation.predict_pvtv2_b2_mask", fake_predict)
    artifacts = run_segmentation(cfg)
    assert artifacts["segmentation_status"].iloc[0] == "success"
    assert artifacts["segmentation_source"].iloc[0] == PVT_SEGMENTATION_SOURCE
    assert Path(artifacts["mask_path"].iloc[0]).exists()


def test_feature_gate_allows_baseline_but_keeps_downstream_checks(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["provider"] = "provided_mask_baseline"
    cfg["segmentation"]["require_pvtv2_inference"] = False
    current_hash = cfg["_config_hash"]
    cv = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "segmentation_source": "provided_mask_baseline",
                "segmentation_status": "success",
                "config_hash": current_hash,
            }
        ]
    )
    keypoints = pd.DataFrame([{"image_id": "B3:cow_001.jpg", "keypoint_status": "success", "config_hash": current_hash}])
    depth = pd.DataFrame(
        [{"image_id": "B3:cow_001.jpg", "depth_status": "success", "depth_provider": "depth_pro", "config_hash": current_hash}]
    )
    pointcloud = pd.DataFrame([{"image_id": "B3:cow_001.jpg", "pointcloud_status": "success", "config_hash": current_hash}])
    assert_feature_inputs_ready(cfg, cv, keypoints, depth, pointcloud)
    stale_depth = depth.copy()
    stale_depth.loc[0, "config_hash"] = "stale"
    with pytest.raises(RuntimeError, match="current-config depth artifacts"):
        assert_feature_inputs_ready(cfg, cv, keypoints, stale_depth, pointcloud)
    for artifact_name in ["cv", "keypoints", "depth", "pointcloud"]:
        frames = {
            "cv": cv.copy(),
            "keypoints": keypoints.copy(),
            "depth": depth.copy(),
            "pointcloud": pointcloud.copy(),
        }
        frames[artifact_name].loc[0, "config_hash"] = "stale"
        frames[artifact_name].loc[0, "reusable_by_stable_input_config"] = True
        frames[artifact_name].loc[0, "stable_input_config_hash"] = "stable-input-v1"
        assert_feature_inputs_ready(cfg, frames["cv"], frames["keypoints"], frames["depth"], frames["pointcloud"])


def test_feature_gate_requires_frozen_pvtv2_for_headline(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["allow_fallback"] = False
    cfg["segmentation"]["require_pvtv2_inference"] = True
    current_hash = cfg["_config_hash"]
    cv = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "segmentation_source": PVT_SEGMENTATION_SOURCE,
                "segmentation_status": "success",
                "config_hash": current_hash,
            }
        ]
    )
    keypoints = pd.DataFrame([{"image_id": "B3:cow_001.jpg", "keypoint_status": "success", "config_hash": current_hash}])
    depth = pd.DataFrame(
        [{"image_id": "B3:cow_001.jpg", "depth_status": "success", "depth_provider": "depth_pro", "config_hash": current_hash}]
    )
    pointcloud = pd.DataFrame([{"image_id": "B3:cow_001.jpg", "pointcloud_status": "success", "config_hash": current_hash}])
    assert_feature_inputs_ready(cfg, cv, keypoints, depth, pointcloud)
    invalid = cv.copy()
    invalid.loc[0, "segmentation_source"] = "provided_mask_ablation"
    with pytest.raises(RuntimeError, match="frozen PVTv2-B2 saliency"):
        assert_feature_inputs_ready(cfg, invalid, keypoints, depth, pointcloud)


def test_pointcloud_gate_requires_frozen_pvtv2_source() -> None:
    valid = pd.DataFrame([{"segmentation_status": "success", "segmentation_source": PVT_SEGMENTATION_SOURCE}])
    assert_pvtv2_pointcloud_inputs(valid)
    invalid = pd.DataFrame([{"segmentation_status": "success", "segmentation_source": "fallback_image_threshold"}])
    with pytest.raises(RuntimeError, match="frozen PVTv2-B2 saliency"):
        assert_pvtv2_pointcloud_inputs(invalid)


def test_keypoint_gate_passes_config_to_expected_source(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = pd.DataFrame([{"segmentation_status": "success", "segmentation_source": "configured_source"}])

    def fake_required_source(config: dict | None = None) -> str:
        if config is None:
            raise AssertionError("config was not passed")
        return "configured_source"

    monkeypatch.setattr("cattle_weight.cv.keypoints.required_pvtv2_segmentation_source", fake_required_source)
    assert_pvtv2_keypoint_inputs(artifacts, {"segmentation": {}})


def test_headline_readiness_accepts_frozen_pvtv2_without_training_report(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["allow_fallback"] = False
    cfg["segmentation"]["require_pvtv2_inference"] = True
    cv = pd.DataFrame(
        [
            {
                "image_id": "B3:cow_001.jpg",
                "dataset": "B3",
                "segmentation_source": PVT_SEGMENTATION_SOURCE,
                "segmentation_status": "success",
            }
        ]
    )
    depth = pd.DataFrame([{"image_id": "B3:cow_001.jpg", "dataset": "B3", "depth_status": "success", "depth_provider": "depth_pro"}])
    write_parquet(cv, registry_dir(cfg) / "cv_artifacts.parquet")
    write_parquet(depth, registry_dir(cfg) / "depth_metadata.parquet")
    assert_headline_reproduction_ready(cfg)
    cv.loc[0, "segmentation_source"] = "provided_mask_ablation"
    write_parquet(cv, registry_dir(cfg) / "cv_artifacts.parquet")
    with pytest.raises(RuntimeError, match="frozen PVTv2-B2 saliency"):
        assert_headline_reproduction_ready(cfg)


def test_coverage_funnel_counts_frozen_pvtv2_masks(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    image_id = "B3:cow_001.jpg"
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "weight_kg": 100, "data_version": "data-v1"}]),
        registry_dir(cfg) / "dataset_index.parquet",
    )
    write_parquet(pd.DataFrame([{"image_id": image_id, "dataset": "B3", "split": "train"}]), registry_dir(cfg) / "splits.parquet")
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "detection_status": "success"}]),
        registry_dir(cfg) / "sticker_metadata.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [{"image_id": image_id, "dataset": "B3", "segmentation_status": "success", "segmentation_source": PVT_SEGMENTATION_SOURCE}]
        ),
        registry_dir(cfg) / "cv_artifacts.parquet",
    )
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "keypoint_status": "success"}]),
        registry_dir(cfg) / "keypoint_metadata.parquet",
    )
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "depth_status": "success", "depth_provider": "depth_pro"}]),
        registry_dir(cfg) / "depth_metadata.parquet",
    )
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "pointcloud_status": "success"}]),
        registry_dir(cfg) / "pointcloud_metadata.parquet",
    )
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "feature_status": "success"}]),
        registry_dir(cfg) / "features_b3.parquet",
    )
    funnel = write_coverage_funnel(cfg)
    b3 = funnel[funnel["dataset"] == "B3"].iloc[0]
    assert b3["pvtv2_mask"] == 1
    assert b3["features"] == 1


def test_coverage_funnel_reports_missing_sticker_metadata(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    image_id = "B3:cow_001.jpg"
    write_parquet(
        pd.DataFrame([{"image_id": image_id, "dataset": "B3", "weight_kg": 100, "data_version": "data-v1"}]),
        registry_dir(cfg) / "dataset_index.parquet",
    )
    write_parquet(pd.DataFrame([{"image_id": image_id, "dataset": "B3", "split": "train"}]), registry_dir(cfg) / "splits.parquet")
    with pytest.raises(RuntimeError, match="Coverage funnel requires sticker metadata"):
        write_coverage_funnel(cfg)


def test_mask_review_selection_is_deterministic(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    cfg["segmentation"]["review"] = {
        "random_count": 3,
        "area_anomaly_count": 3,
        "complexity_count": 3,
        "complexity_max_side": 64,
    }
    artifact_rows = []
    sticker_rows = []
    for idx in range(12):
        image_id = f"B3:cow_{idx:02d}.jpg"
        image_path = tmp_path / f"cow_{idx:02d}.jpg"
        mask_path = tmp_path / f"cow_{idx:02d}_mask.png"
        Image.new("RGB", (60, 40), (230, 230, 230)).save(image_path)
        mask = Image.new("L", (60, 40), 0)
        draw = ImageDraw.Draw(mask)
        if idx % 4 == 0:
            draw.polygon([(5, 5), (45, 6), (55, 20), (45, 35), (8, 34), (15, 22)], fill=255)
        else:
            draw.rectangle((5, 8, 20 + idx, 28), fill=255)
        mask.save(mask_path)
        mask_area = int((np.asarray(mask) > 0).sum())
        artifact_rows.append(
            {
                "image_id": image_id,
                "dataset": "B3",
                "segmentation_source": "provided_mask_baseline",
                "segmentation_status": "success",
                "mask_path": str(mask_path),
                "mask_area_px": mask_area,
            }
        )
        sticker_rows.append(
            {
                "image_id": image_id,
                "dataset": "B3",
                "detection_status": "success",
                "normalized_image_path": str(image_path),
                "normalized_width": 60,
                "normalized_height": 40,
            }
        )
    artifacts = pd.DataFrame(artifact_rows)
    sticker = pd.DataFrame(sticker_rows)
    selected_a = select_mask_review_rows(cfg, artifacts, sticker)
    selected_b = select_mask_review_rows(cfg, artifacts, sticker)
    assert selected_a["review_category"].value_counts().to_dict() == {
        "random": 3,
        "area_anomaly": 3,
        "high_complexity": 3,
    }
    pd.testing.assert_frame_equal(
        selected_a[["review_category", "selection_rank", "image_id"]],
        selected_b[["review_category", "selection_rank", "image_id"]],
    )


def test_run_mask_freeze_writes_manifest(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path)
    mask_path = tmp_path / "mask.png"
    Image.new("L", (8, 6), 255).save(mask_path)
    write_parquet(
        pd.DataFrame(
            [
                {
                    "image_id": "B3:cow.jpg",
                    "dataset": "B3",
                    "segmentation_status": "success",
                    "segmentation_source": PVT_SEGMENTATION_SOURCE,
                    "mask_path": str(mask_path),
                }
            ]
        ),
        registry_dir(cfg) / "cv_artifacts.parquet",
    )
    review_dir = reports_dir(cfg) / "mask_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"image_id": "B3:cow.jpg"}]).to_csv(review_dir / "mask_review_manifest.csv", index=False)
    manifest = run_mask_freeze(cfg)
    assert manifest["status"] == "frozen"
    assert manifest["mask_count"] == 1
    assert (review_dir / "frozen_mask_manifest.json").exists()


def test_sticker_detection_and_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "cow_weight_100.png"
    mask_path = tmp_path / "cow_weight_100_mask.png"
    image = Image.new("RGB", (100, 80), (230, 230, 230))
    mask = Image.new("L", (100, 80), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 80, 60), fill=(90, 70, 60))
    draw.ellipse((45, 30, 55, 40), fill=(220, 20, 20))
    ImageDraw.Draw(mask).ellipse((20, 20, 80, 60), fill=255)
    image.save(image_path)
    mask.save(mask_path)
    sticker = detect_sticker(image_path)
    assert 4 <= sticker["radius_px"] <= 8
    meta = normalize_image_and_mask(
        image_path,
        mask_path,
        sticker,
        tmp_path / "normalized.png",
        tmp_path / "normalized_mask.png",
        100.0,
    )
    assert abs(meta["normalized_sticker_radius_px"] - 100.0) < 1e-6
    assert Path(meta["normalized_mask_path"]).exists()


def test_mask_contour_hu_and_geometry_features() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True
    contour = contour_from_mask(mask)
    bbox = bbox_from_mask(mask)
    assert bbox == (8.0, 5.0, 21.0, 14.0)
    assert len(contour) > 0
    hu = hu_moment_features(mask)
    assert len(hu) == 7
    assert np.isfinite(list(hu.values())).all()
    hull = mask_convex_hull_features(mask, contour)
    assert hull["mask_hull_area_px2"] > 0
    stats = prefix_stats("mask_centroid_dist", np.array([1.0, 2.0, 3.0]))
    assert stats["mask_centroid_dist_median"] == 2.0
    assert len(beam_angles(contour)) > 0
    assert len(chain_codes(contour)) > 0


def test_body_and_keypoint_distance_features() -> None:
    keypoints = np.zeros((20, 3), dtype=float)
    for idx in range(20):
        keypoints[idx] = [idx * 3.0, idx * 2.0, 0.9]
    body = body_measurement_features(keypoints)
    assert body["bm_body_length_px"] > 0
    distances = keypoint_distance_features(keypoints)
    assert len(distances) == 20
    assert distances["kpd_l_eye_r_eye_px"] > 0


def test_pointcloud_projection_and_cleanup() -> None:
    depth = np.ones((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    points = depth_to_point_cloud(depth, mask, 10.0)
    assert points.shape[1] == 3
    cleaned, counts = cleanup_point_cloud(points, 0.2, 1)
    assert len(cleaned) > 0
    assert counts["point_count_after_mask"] == 36


def test_compute_exact_84_features() -> None:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 12:48] = True
    contour = contour_from_mask(mask)
    keypoints = np.zeros((20, 3), dtype=float)
    for idx in range(20):
        keypoints[idx] = [15 + idx, 10 + idx * 0.8, 0.9]
    yy, xx = np.where(mask)
    pointcloud = np.column_stack([xx[:200] / 50.0, yy[:200] / 50.0, np.linspace(0, 1, len(xx[:200]))])
    features = compute_all_features(mask, contour, keypoints, pointcloud)
    assert list(features.keys()) == PAPER_FEATURES
    assert len(features) == 84


def test_nmi_fit_uses_train_only() -> None:
    train = pd.DataFrame(
        {
            "weight_kg": [100, 120, 140, 160, 180, 200],
            **{name: [1, 2, 3, 4, 5, 6] for name in PAPER_FEATURES},
        }
    )
    train[PAPER_FEATURES[1]] = 1
    selected, summary = fit_nmi_selector(train, 0.0)
    assert PAPER_FEATURES[0] in selected
    assert summary["selected"].any()


def test_metrics() -> None:
    y = np.array([100.0, 200.0])
    pred = np.array([90.0, 210.0])
    assert mae(y, pred) == 10.0
    assert abs(mape(y, pred) - 7.5) < 1e-9


def test_reproduction_smoke_end_to_end(tmp_path: Path) -> None:
    cfg = smoke_config(tmp_path, image_count=8)
    run_reproduction_smoke(cfg)
    assert (tmp_path / "reports" / "reproduction_metrics.csv").exists()
    assert (tmp_path / "reports" / "depth_completion_summary.csv").exists()
    assert (tmp_path / "data_registry" / "features_b3.parquet").exists()
