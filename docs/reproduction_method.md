# Reproduction Method

The implemented pipeline is:

1. Index B3/B4 images and parse weights from filenames or metadata-like names.
2. Detect the colored circular sticker and resize each image so `sticker_radius_px = 100`.
3. Run frozen PVTv2-B2 feature saliency segmentation. Smoke mode uses a labeled fallback or provided masks.
4. Produce cattle masks, direct AnimalPose checkpoint keypoints, Depth Pro depth maps, and point clouds.
5. Project depth to points with `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`, `Z = depth`.
6. Invert and normalize point-cloud `z`, deterministically downsample to 2%, and apply Otsu cleanup.
7. Extract the exact 84 named paper-style features.
8. Fit NMI feature filtering, imputation, scaling, and LASSO on training rows only.
9. Evaluate HGM and LASSO with MAE and MAPE on B3 test, B4 test, B3 to B4, and B4 to B3.

The HGM baseline uses the paper formula:

`kg = ((heart_girth_in^2 * body_length_in) / 300) * 0.45359237`

Pixel measurements are converted to inches using the configured sticker reference radius.

## AnimalPose Checkpoint Path

Headline keypoints use the configured AnimalPose SimpleBaseline ResNet-50 checkpoint directly. The MMPose API path is bypassed for strict reproduction because the checkpoint can be loaded into the local model definition without depending on MMPose runtime extensions. Successful strict keypoint rows are labeled `mmpose_checkpoint_direct_simplebaseline`, and QA overlays are written under `reports/keypoint_review/`.
