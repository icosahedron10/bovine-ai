# Bovine AI Paper Reproduction

This repository implements a paper-guided reproduction of sticker-based B3/B4 cattle weight estimation from 2D side-view images. It follows `REPRODUCE_PAPER.md` and the root PDF, with explicit deviations: PVTv2-B2 segmentation instead of CAFE-Net, HGM plus LASSO only, MAE/MAPE only, and directional rather than exact numeric parity.

## Install

```bash
python -m pip install -e .[test]
```

Optional full CV integrations require extra packages and model setup:

```bash
python -m pip install -e .[cv,test]
```

The smoke pipeline does not require Kaggle credentials, Depth Pro, OpenCV, or PyTorch.

## Data Setup

For local B3/B4 data, set `data.local_root` in `configs/reproduction_b3_b4.yaml` or pass:

```bash
python -m cattle_weight.cli data index --config configs/reproduction_b3_b4.yaml --dataset-root /path/to/dataset
```

The dataset root should contain B3/B4 folders, image files with weights encoded in filenames, and optional masks. For Kaggle, configure credentials with `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`, then set `data.kaggle_dataset`.

Headline segmentation uses frozen public PVTv2-B2 weights through `timm`. The repo does not train, fine-tune, import, or validate a cattle-specific PVTv2 checkpoint.

## Smoke Test

```bash
python -m cattle_weight.cli run reproduction-smoke --config configs/smoke.yaml
```

This creates a tiny deterministic synthetic B3/B4 dataset, normalizes stickers to radius 100 px, uses labeled smoke fallbacks for segmentation/keypoints/depth, extracts 84 features, trains HGM/LASSO, and writes metrics.

## Full Reproduction

```bash
python -m cattle_weight.cli run reproduction-full --config configs/reproduction_b3_b4.yaml --dataset-root /path/to/B3B4
```

Full mode attempts the configured frozen PVTv2-B2/direct AnimalPose checkpoint/Depth Pro path. If public PVTv2 weights, providers, or checkpoints for the non-PVT components are missing, failed artifacts are recorded instead of silently ignored.

The main path does not reuse provided B3/B4 masks as PVTv2 outputs. It derives masks from frozen PVTv2-B2 feature saliency, labels them `pvtv2_b2_frozen_saliency`, loads the AnimalPose checkpoint directly with provider `mmpose_checkpoint_direct_simplebaseline`, overwrites segmentation/depth/point-cloud/model artifacts, runs automated mask sanity checks, and only allows headline metrics from successful frozen PVTv2 masks and `depth_provider == "depth_pro"` rows.

## Expected Outputs

- `data_registry/dataset_index.parquet`
- `data_registry/sticker_metadata.parquet`
- `data_registry/cv_artifacts.parquet`
- `data_registry/keypoint_metadata.parquet`
- `data_registry/depth_metadata.parquet`
- `data_registry/pointcloud_metadata.parquet`
- `data_registry/features_b3.parquet`
- `data_registry/features_b4.parquet`
- `data_registry/feature_schema.yaml`
- `reports/reproduction_metrics.csv`
- `reports/reproduction_metrics.md`
- `reports/reproduction_discrepancies.md`
- `reports/depth_completion_summary.csv`
- `reports/headline_reproduction_status.json`
- `reports/headline_reproduction_status.md`
- `reports/coverage_funnel.csv`
- `reports/paper_metric_gaps.csv`
- `models/<run_id>/`

Interpret discrepancies against `docs/reproduction_method.md`, `docs/known_limitations.md`, and the root PDF. Exact paper parity depends on dataset version, split seed, preprocessing, mask source, keypoint model, depth model, point-cloud cleanup, and LASSO hyperparameters.
