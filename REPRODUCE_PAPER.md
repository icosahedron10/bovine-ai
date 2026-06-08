/goal

You are implementing a paper-faithful reproduction repository for a cattle weight estimation paper. I will provide a transcribed copy of the paper in the repository, likely at docs/paper.md. The B3/B4 datasets are available through Kaggle or through a local dataset path.

The only goal is to recreate the paper’s reported results as faithfully as possible.

Do not design, scaffold, discuss, or implement any product adaptation path. Do not consider no-sticker inference, sale-barn deployment, measurement-supervised alternatives, predicted measurements, runtime apps, video workflows, uncertainty/rejection systems, teacher-student models, or any future deployment architecture. This repository is strictly for reproducing the paper’s sticker-based B3/B4 experimental pipeline.

Primary objective:
Reproduce the paper’s sticker-based B3/B4 pipeline closely enough to determine whether this implementation matches the paper’s reported results.

Core reproduction target:
image
  -> detect sticker radius
  -> resize so sticker_radius_px = 100
  -> segmentation
  -> MMPose-style keypoints
  -> Depth Pro depth and focal-length estimation
  -> pinhole point-cloud generation
  -> paper-style engineered features
  -> NMI feature filtering
  -> HGM / OLS / LASSO / stepwise p-value / stepwise AIC / RF / ANN
  -> MAE, MAPE, RMSE
  -> B3 within-dataset, B4 within-dataset, B3→B4, B4→B3 evaluation

Hard scope boundaries:
- Reproduce the paper.
- Preserve sticker-based scale normalization.
- Do not replace the sticker with another scale strategy.
- Do not remove/inpaint/ablate the sticker.
- Do not create deployment-safe schemas.
- Do not build product inference code.
- Do not build sale-barn adaptation code.
- Do not introduce new modeling tracks unless they are required to match the paper.
- If the paper transcript conflicts with this prompt, trust docs/paper.md and document the discrepancy.

Expected paper facts to verify from docs/paper.md:
- B3 contains approximately 2,588 images with weights from about 36–621 kg.
- B4 contains approximately 1,934 images with weights from about 69–307 kg.
- Weights are encoded in filenames or available through dataset metadata.
- Images are normalized by detecting the circular sticker and resizing so sticker_radius_px = 100.
- The paper reports 84 engineered features across:
  - body measurement features,
  - basic contour features,
  - Hu moments,
  - statistics of contour-point distances from centroid,
  - convex hull features,
  - beam angle statistics,
  - chain-code statistics,
  - MMPose keypoint distances,
  - point-cloud analog features.
- The paper uses NMI feature filtering.
- The paper evaluates:
  - HGM,
  - OLS,
  - LASSO,
  - stepwise p-value,
  - stepwise AIC,
  - Random Forest,
  - ANN.
- The paper uses 80/20 B3 and B4 splits.
- The paper reports within-dataset and cross-dataset evaluation:
  - B3 train/test,
  - B4 train/test,
  - B3→B4,
  - B4→B3.
- The key LASSO reference to target is approximately:
  - B3-trained LASSO on B3 test: 25.45 kg MAE / 16.81% MAPE.
  - B3-trained LASSO on B4 cross-dataset: 24.56 kg MAE / 14.23% MAPE.
- The HGM baseline should reproduce directionally as a weaker baseline, approximately:
  - B3: 40.69 kg MAE / 24.63% MAPE.
  - B4: 53.65 kg MAE / 31.36% MAPE.

If docs/paper.md reports different values, use the paper values and record the discrepancy in reports/reproduction_discrepancies.md.

Repository layout to create:

data_registry/
configs/
src/cattle_weight/
  __init__.py
  cli.py
  data/
  cv/
  depth/
  features/
  models/
  eval/
notebooks/
reports/
tests/
docs/

Use Python. Prefer simple, maintainable, auditable code over complex research-code abstractions. Use standard tools such as pandas, numpy, scikit-learn, scipy, scikit-image, opencv-python, matplotlib, pyyaml, joblib, tqdm, pytest, and typer or click for the CLI. Use PyTorch only where required for ANN or optional CV integrations.

The repository must be runnable without full heavy CV dependencies by using smoke-test fallbacks. However, full reproduction mode should attempt the paper-faithful stack.

Required CLI:
Implement a command-line interface as either `cw` or `python -m cattle_weight.cli`.

Required commands:
cw data index
cw split create
cw cv sticker-normalize
cw cv segment
cw cv keypoints
cw cv depth
cw cv pointcloud
cw features extract
cw train
cw evaluate
cw run reproduction-smoke
cw run reproduction-full

All CLI commands must accept:
- config path,
- dataset root or Kaggle dataset reference,
- output directory,
- random seed where relevant.

Minimum configs:
configs/default.yaml
configs/reproduction_b3_b4.yaml
configs/smoke.yaml

Phase 0 — Repository foundation

Implement:
1. Project scaffolding.
2. Config loading.
3. Dataset registry.
4. Deterministic split generation.
5. Experiment output directories.
6. Basic artifact versioning.
7. Smoke-test dataset generation.

Acceptance criteria for Phase 0:
- `python -m cattle_weight.cli --help` works.
- All required CLI commands exist.
- `pytest` passes.
- `cw run reproduction-smoke --config configs/smoke.yaml` completes on synthetic or tiny sample data.
- Train/test splits are materialized and versioned.
- Model artifacts store:
  - data version,
  - split version,
  - feature schema version,
  - model parameters,
  - random seed,
  - code commit if available.
- A new engineer can run the smoke test without Kaggle credentials.

Phase 1 — B3/B4 ingestion

Implement:
1. B3/B4 dataset indexing.
2. Kaggle download support or local dataset path support.
3. Filename or metadata-based weight parsing.
4. Image discovery.
5. Mask discovery.
6. Dataset summary reporting.

Dataset access requirements:
- Do not hardcode Kaggle credentials.
- Support either:
  - Kaggle API credentials from the user’s environment,
  - or a configured local dataset root.
- If Kaggle access is unavailable, fail with a clear message explaining which environment variable or config value is missing.
- Do not require the full dataset for tests.

Required outputs:
data_registry/dataset_index.parquet
reports/b3_b4_dataset_summary.csv
reports/figures/weight_distribution_b3.png
reports/figures/weight_distribution_b4.png
reports/figures/image_dimension_distribution.png

Acceptance criteria for Phase 1:
- `cw data index --config configs/reproduction_b3_b4.yaml` creates a dataset index table.
- B3/B4 image counts match the paper or discrepancies are documented.
- B3/B4 weight ranges match the paper or discrepancies are documented.
- Weight parsing is covered by tests.
- Missing or malformed files are recorded instead of silently ignored.
- Dataset summary includes:
  - dataset name,
  - image count,
  - mask count if available,
  - min weight,
  - max weight,
  - mean weight,
  - median weight,
  - image dimension summary.

Phase 2 — Sticker detection and normalization

Implement:
1. Sticker detection.
2. Sticker radius calculation.
3. Image resizing so sticker_radius_px = 100.
4. Mask resizing using the same transform.
5. Raw and normalized metadata storage.
6. QA report generation.

Sticker detection requirements:
- Use annotations or masks if the dataset provides sticker location.
- Add fallback image-based circular detection using OpenCV Hough circles or contour detection.
- Store detection method.
- Store raw sticker radius.
- Store normalized sticker radius.
- Store detection success/failure.
- Store normalized image path.
- Store normalized mask path if masks are available.

Required outputs:
data_registry/sticker_metadata.parquet
reports/sticker_radius_before_after.csv
reports/sticker_detection_failures.csv
reports/figures/sticker_radius_before_after.png

Acceptance criteria for Phase 2:
- `cw cv sticker-normalize --config configs/reproduction_b3_b4.yaml` produces normalized images or normalized metadata.
- Normalized sticker radius distribution is tightly centered at 100 px for successful detections.
- Target automatic sticker detection success is ≥ 98%.
- If success is below 98%, the report must explain why and list failed cases.
- Failed detections are recorded, not silently dropped.
- Tests cover:
  - synthetic circular sticker detection,
  - radius computation,
  - image scaling,
  - mask scaling,
  - metadata output.

Phase 3 — Segmentation, keypoints, depth, and point clouds

Implement preprocessing artifact generation.

Segmentation:
- Fastest faithful path: use provided masks where available.
- If provided masks are used, clearly label them as provided masks in artifact metadata.
- Optional paper-faithful model path: CAFE-Net-like or PVTv2-B2 segmentation.
- Produce:
  - mask path,
  - bbox,
  - contour,
  - mask area,
  - segmentation source,
  - segmentation status.

Keypoints:
- Implement a provider interface.
- Preferred provider: MMPose AnimalPose-style keypoints if installed/configured.
- Smoke-test fallback: geometric pseudo-keypoints derived from mask/bbox.
- The fallback must be labeled as fallback and must not be represented as paper-equivalent.

Depth:
- Implement a provider interface.
- Preferred provider: Depth Pro if installed/configured.
- Smoke-test fallback: deterministic dummy relative-depth map.
- The fallback must be labeled as fallback and must not be represented as paper-equivalent.
- Store:
  - depth map path,
  - estimated focal length in pixels if available,
  - depth provider,
  - depth status.

Point cloud:
Generate point clouds using pinhole projection:

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth

Point-cloud cleanup:
- Apply cattle mask.
- Normalize z-axis.
- Downsample.
- Remove obvious z-noise using thresholding or Otsu-like cleanup.
- Store pointcloud_path and summary metadata.

Required outputs:
data_registry/cv_artifacts.parquet
data_registry/pointcloud_metadata.parquet

Acceptance criteria for Phase 3:
- `cw cv segment --config configs/reproduction_b3_b4.yaml` produces segmentation artifacts or records failure.
- `cw cv keypoints --config configs/reproduction_b3_b4.yaml` produces keypoint artifacts or records failure.
- `cw cv depth --config configs/reproduction_b3_b4.yaml` produces depth artifacts or records failure.
- `cw cv pointcloud --config configs/reproduction_b3_b4.yaml` produces point-cloud artifacts or records failure.
- Full B3/B4 mode attempts artifact production for all indexed images.
- Smoke mode completes without heavy CV dependencies.
- Point-cloud generation produces non-empty cattle-specific point clouds for accepted images.
- Every artifact traces back to:
  - image_id,
  - dataset version,
  - transform version,
  - config hash,
  - provider.
- Tests cover:
  - mask-to-contour extraction,
  - bbox extraction,
  - keypoint schema validation,
  - depth-to-point-cloud projection,
  - point-cloud cleanup,
  - artifact lineage.

Phase 4 — Paper-style feature extraction

Implement the paper-style feature families.

Required feature families:
1. Body-measurement proxies from keypoints and/or mask:
   - heart girth proxy,
   - withers height,
   - hip height,
   - tail-to-neck distance,
   - body length,
   - cylinder-style volume proxy.
2. Basic contour features:
   - bbox width,
   - bbox height,
   - perimeter,
   - area,
   - aspect ratio,
   - extent.
3. Hu moments:
   - hu_1 through hu_7.
4. Statistics of contour-point distances from centroid:
   - mean,
   - std,
   - min,
   - max,
   - percentiles.
5. Convex hull features:
   - hull area,
   - hull perimeter,
   - solidity,
   - convexity or defect summaries if feasible.
6. Beam angle statistics.
7. Chain-code statistics.
8. MMPose keypoint distances.
9. Point-cloud analog features:
   - pc_bbox dimensions,
   - pc_depth_mean,
   - pc_depth_std,
   - pc_volume_proxy,
   - pc_hull_area,
   - pc_point_count.

Feature extraction requirements:
- Record coordinate frame for every feature:
  - raw image,
  - sticker-normalized image,
  - model-input frame,
  - crop frame,
  - normalized cattle frame,
  - point-cloud frame.
- Feature extraction must be deterministic.
- Missing features must be explicit.
- Do not silently create large NaN-filled tables.
- Feature schema must record:
  - feature name,
  - family,
  - units,
  - coordinate frame,
  - source artifact,
  - whether paper-faithful,
  - missingness rule.

Required outputs:
data_registry/features_b3.parquet
data_registry/features_b4.parquet
data_registry/feature_schema.yaml
reports/feature_missingness.csv
reports/feature_summary.csv

Acceptance criteria for Phase 4:
- `cw features extract --config configs/reproduction_b3_b4.yaml` writes B3 and B4 feature tables.
- Feature table contains all paper feature families.
- Target feature count is approximately 84 paper-style features unless docs/paper.md establishes a different exact list.
- Any missing paper feature is tagged as:
  - unavailable,
  - replaced,
  - not reproducible,
  - or deferred.
- Feature extraction is deterministic: rerunning on the same artifact versions produces identical feature values within floating tolerance.
- Tests cover:
  - body-measurement proxy features,
  - contour features,
  - Hu moments,
  - centroid-distance statistics,
  - convex hull features,
  - beam-angle features,
  - chain-code features,
  - keypoint-distance features,
  - point-cloud features.

Phase 5 — Statistical filtering and model reproduction

Implement:
1. NMI feature filtering fitted on training data only.
2. Imputation fitted on training data only.
3. Scaling fitted on training data only.
4. HGM baseline.
5. OLS.
6. LASSO.
7. Stepwise p-value regression.
8. Stepwise AIC regression.
9. Random Forest.
10. ANN.

Model requirements:
- Use scikit-learn where possible.
- OLS: LinearRegression or statsmodels.
- LASSO: LassoCV or configured alpha grid.
- Stepwise p-value and AIC: statsmodels-based implementation is acceptable.
- RF: RandomForestRegressor.
- ANN: MLPRegressor or a small PyTorch MLP with deterministic seed.
- HGM: implement the formula from docs/paper.md. If the transcript does not provide the exact formula, document this in reports/reproduction_discrepancies.md.

Evaluation requirements:
- B3 80/20 train/test.
- B4 80/20 train/test.
- B3-trained model evaluated on B4.
- B4-trained model evaluated on B3.
- Metrics:
  - MAE,
  - MAPE,
  - RMSE,
  - optionally R².
- Plots:
  - predicted vs. actual,
  - residuals,
  - feature-selection summaries.

Splitting requirements:
- Reproduce the paper’s 80/20 split setup.
- Use deterministic random seed from config.
- Save exact split membership.
- Fit NMI, imputers, scalers, feature selectors, and hyperparameter selection only on training data.

Required outputs:
reports/reproduction_metrics.csv
reports/reproduction_metrics.md
reports/feature_selection_summary.csv
reports/reproduction_discrepancies.md
reports/figures/pred_vs_actual_<dataset>_<model>.png
reports/figures/residuals_<dataset>_<model>.png
models/<run_id>/

Acceptance criteria for Phase 5:
- `cw split create --config configs/reproduction_b3_b4.yaml` creates deterministic B3/B4 train/test splits.
- `cw train --config configs/reproduction_b3_b4.yaml` trains all configured models.
- `cw evaluate --config configs/reproduction_b3_b4.yaml` reports:
  - B3 test results,
  - B4 test results,
  - B3→B4 results,
  - B4→B3 results.
- LASSO reproduction target:
  - B3 test MAPE within ±3 points of the paper,
  - B3→B4 MAPE within ±3 points of the paper,
  - or a detailed discrepancy analysis is completed.
- HGM baseline is directionally weaker than learned feature models, or discrepancy analysis is completed.
- All preprocessing transforms, NMI filters, imputers, scalers, feature selectors, and hyperparameter searches are fitted only on training folds.
- Tests verify leakage prevention for train/test preprocessing transforms.

Testing requirements:
Use pytest.

Required unit tests:
- config loading,
- dataset indexing fixtures,
- filename weight parsing,
- sticker detection on synthetic images,
- sticker radius normalization,
- mask contour extraction,
- Hu moments,
- centroid-distance statistics,
- convex hull features,
- beam-angle features,
- chain-code features,
- keypoint-distance features,
- pinhole point-cloud projection,
- NMI fitted on train only,
- deterministic split generation,
- metrics computation,
- model training smoke test.

Required integration smoke test:
- Create a tiny synthetic dataset with simple cattle-like masks, stickers, and fake weights.
- Run:
  cw run reproduction-smoke --config configs/smoke.yaml
- The smoke test must complete end-to-end and produce a metrics file.

Documentation requirements:
Create or update:
- README.md
- docs/reproduction_method.md
- docs/data_setup.md
- docs/feature_schema.md
- docs/known_limitations.md
- reports/reproduction_discrepancies.md

README.md must include:
- installation instructions,
- Kaggle/local dataset setup,
- smoke test command,
- full reproduction command,
- expected outputs,
- optional heavy dependency notes,
- how to interpret discrepancies.

Known limitations must explicitly state:
- Exact parity depends on matching dataset version, split seed, preprocessing details, masks, keypoint model, depth model, and hyperparameters.
- If CAFE-Net, MMPose, or Depth Pro are unavailable, the run is a partial reproduction.
- Provided masks or fallback keypoints/depth may not match the paper’s exact CV stack.
- Paper parity must be evaluated against docs/paper.md.

Engineering constraints:
- Prefer deterministic, config-driven code.
- Avoid notebooks as the primary execution path.
- Do not hardcode absolute local paths.
- Do not hardcode Kaggle credentials.
- Fail loudly on missing required files.
- Record all skipped images and failed artifacts.
- Use typed functions and clear dataclasses or pydantic models where useful.
- Keep heavy CV integrations optional.
- Make the repository useful before full CV dependencies are installed by supporting smoke-test fallbacks.
- Do not ask for clarification unless absolutely blocked. Make reasonable assumptions, document them, and continue.

Final deliverables:
1. Working repository patch.
2. Passing tests.
3. Runnable CLI.
4. Smoke-test output.
5. B3/B4 dataset indexing.
6. Sticker normalization pipeline.
7. Segmentation/keypoint/depth/point-cloud artifact pipeline.
8. Paper-style feature extraction.
9. HGM, OLS, LASSO, stepwise p-value, stepwise AIC, RF, and ANN models.
10. B3/B4 within-dataset and cross-dataset metrics.
11. Reproduction report.
12. Discrepancy report.

Definition of done:
The task is complete only when:
- `pytest` passes.
- `python -m cattle_weight.cli --help` works.
- `cw run reproduction-smoke --config configs/smoke.yaml` completes.
- The repository can index B3/B4 from a configured local or Kaggle path.
- The repository can normalize images to sticker_radius_px = 100.
- The repository can produce or ingest segmentation artifacts.
- The repository can produce or ingest keypoint artifacts.
- The repository can produce or ingest depth artifacts.
- The repository can generate point clouds.
- The repository can extract paper-style features.
- The repository can train and evaluate HGM, OLS, LASSO, stepwise p-value, stepwise AIC, RF, and ANN.
- The repository reports B3 test, B4 test, B3→B4, and B4→B3 metrics.
- The repository stores enough metadata to reproduce a run.
- If exact paper numbers are not reached, reports/reproduction_discrepancies.md explains whether the likely cause is:
  - dataset mismatch,
  - split mismatch,
  - sticker detection mismatch,
  - mask source mismatch,
  - keypoint provider mismatch,
  - depth provider mismatch,
  - point-cloud mismatch,
  - feature mismatch,
  - hyperparameter mismatch,
  - or incomplete paper details.

When uncertain:
- Read docs/paper.md.
- Prefer the paper over this prompt.
- Make a reasonable implementation choice.
- Document the assumption in reports/reproduction_discrepancies.md.
- Continue building rather than stopping for clarification.
