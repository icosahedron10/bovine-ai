# Known Limitations

Exact parity is not required; directional reproduction depends on matching dataset version, split seed, preprocessing details, masks, keypoint model, depth model, and hyperparameters.

If frozen PVTv2-B2 segmentation, the direct AnimalPose checkpoint path, or Depth Pro are unavailable, the run is a partial or blocked reproduction.

Provided masks or fallback keypoints/depth do not match the main CV stack and are labeled as partial or ablation runs.

The strict keypoint path intentionally bypasses the MMPose API and loads the AnimalPose checkpoint directly. MMPose can still be installed for experiments, but headline rows must come from the direct checkpoint provider.

The headline reproduction is intentionally blocked unless masks are produced by frozen PVTv2-B2 saliency and labeled `pvtv2_b2_frozen_saliency`. Provided B3/B4 masks may be used for smoke tests or explicit ablations, but they are not accepted as headline PVTv2 masks.

PVTv2-B2 is not trained or fine-tuned in this repository. Because segmentation is unsupervised frozen-feature saliency, the project does not report PVTv2 Dice/IoU training metrics.

Paper parity must be evaluated against `docs/paper.md` if present, otherwise against the root PDF.

The paper does not fully specify every engineered feature formula. This implementation uses deterministic proxies where needed and documents those assumptions in `reports/reproduction_discrepancies.md`.
