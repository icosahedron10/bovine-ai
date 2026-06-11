# Data Setup

Use `configs/reproduction_b3_b4.yaml` for full B3/B4 indexing.

Local datasets:

```bash
python -m cattle_weight.cli data index --config configs/reproduction_b3_b4.yaml --dataset-root /path/to/root
```

The root may contain `B3` and `B4` folders in any case. Images are discovered recursively. Masks are discovered recursively when a sibling or descendant file contains both the image stem and a configured mask keyword such as `mask`, `annotation`, `segmentation`, or `label`.

Kaggle datasets:

```bash
KAGGLE_USERNAME=... KAGGLE_KEY=... python -m cattle_weight.cli data index --config configs/reproduction_b3_b4.yaml
```

If Kaggle credentials are missing, the CLI fails with an explicit message. Tests and smoke reproduction do not require Kaggle.

## Frozen PVTv2 Segmentation

Full headline reproduction uses public pretrained PVTv2-B2 weights through `timm`. There is no cattle-specific PVTv2 training dataset, Colab handoff, local fine-tuning stage, or checkpoint import step.

The segmentation command derives binary cattle masks from frozen multi-scale PVTv2 feature saliency:

```bash
python -m cattle_weight.cli cv segment --config configs/reproduction_b3_b4.yaml
python -m cattle_weight.cli cv mask-sanity --config configs/reproduction_b3_b4.yaml
python -m cattle_weight.cli cv depth --config configs/reproduction_b3_b4.yaml
```

Headline masks are accepted only when `segmentation_source == "pvtv2_b2_frozen_saliency"`. Provided masks remain available for smoke tests and explicit ablation configs, but they are not accepted as headline PVTv2 outputs.
