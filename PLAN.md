## Executive implementation stance

The fastest defensible path is a **two-track build**:

1. **Paper-faithful reproduction track**: reproduce the B3/B4 sticker-based pipeline closely enough to know whether our implementation matches the paper.
2. **No-sticker product track**: replace sticker-based scale normalization with **training-only measurement supervision**, predicted physical measurements, predicted-HGM, residual correction, uncertainty, and rejection logic.

Do **not** start with a large end-to-end deep model as the primary path. The paper’s strongest reported model was LASSO, and the overall method is feature-driven: segmentation, MMPose keypoints, Depth Pro, point clouds, 84 engineered features, NMI filtering, and OLS/LASSO/stepwise/RF/ANN evaluation.   The product constraint is also explicit: no stickers or manual measurements at inference; sale-barn measurements are privileged training labels only.  

The practical MVP should be:

```text
image or short side-view video
  -> QA / best-frame selection
  -> segmentation
  -> keypoints
  -> Depth Pro
  -> point cloud
  -> paper-style + normalized + depth/device features
  -> predicted body measurements
  -> predicted-HGM base estimate
  -> residual correction model
  -> uncertainty + accept/reject decision
```

---

# Phase 0 — Product boundary, architecture, and repo foundation

**Timing:** Week 0–1
**Priority:** Mandatory. This phase prevents fast work from becoming unusable.

## Goal 0.1 — Lock the inference contract

### Tasks

* Define the production inference payload:

  * Required: image or short video.
  * Optional: sex, breed/type, age class, device metadata, EXIF, environment profile.
  * Forbidden: true body measurements, sale weight, lot average weight, sticker/marker/ruler fields.
* Define output contract:

  * `weight_estimate_kg`
  * `prediction_interval_kg`
  * `quality_status`
  * `retake_reason`
  * `pred_body_length_cm`
  * `pred_heart_girth_cm`
  * `pred_hgm_weight_kg`
  * `residual_correction_kg`
  * `model_version`
* Create feature role taxonomy:

  * `runtime_feature`
  * `model_output_feature`
  * `training_label`
  * `split_key`
  * `audit_only`
  * `forbidden_runtime`

### Acceptance criteria

* Inference schema accepts **no true physical measurement fields**.
* Every column in the data warehouse has a declared role.
* A CI test fails if any forbidden runtime column appears in a production feature matrix.
* Product can explain that sale-barn measurements are training labels, not runtime inputs.

---

## Goal 0.2 — Establish reproducible project scaffolding

### Tasks

* Create repo layout:

```text
data_registry/
configs/
src/cattle_weight/
  data/
  cv/
  depth/
  features/
  models/
  eval/
  leakage/
  serving/
notebooks/
reports/
```

* Add experiment tracking.
* Add dataset versioning.
* Add deterministic split generation.
* Add baseline CLI:

```bash
cw data index
cw split create
cw cv segment
cw cv keypoints
cw cv depth
cw cv pointcloud
cw features extract
cw train
cw evaluate
cw audit leakage
```

### Acceptance criteria

* Any experiment can be rerun from a config file.
* Train/test/calibration splits are materialized and versioned.
* Model artifacts store:

  * training data version,
  * feature schema version,
  * split version,
  * code commit,
  * model parameters.
* A new engineer can run a smoke test end-to-end on a small image subset.

---

# Phase 1 — Paper-faithful reproduction

**Timing:** Week 1–3
**Priority:** Mandatory. This is the scientific anchor.

The paper used B3 and B4 side-view datasets with sticker annotations; images were resized so the detected sticker radius was fixed at 100 pixels to reduce distance-driven scale variation. B3 contains 2,588 images from 36–621 kg; B4 contains 1,934 images from 69–307 kg. 

## Goal 1.1 — B3/B4 ingestion and sticker normalization

### Tasks

* Ingest B3/B4 images and masks.
* Parse weight labels from filenames.
* Detect sticker in each image.
* Compute sticker radius.
* Resize image and mask so:

```text
sticker_radius_px = 100
```

* Store both raw and normalized image metadata.
* Generate QA plots:

  * sticker radius before/after,
  * failed detections,
  * weight distribution,
  * image dimension distribution.

### Acceptance criteria

* B3/B4 image counts and weight ranges match the paper or discrepancies are documented.
* ≥ 98% of sticker detections succeed automatically, or failed cases are handled by fallback annotation/manual QA.
* Normalized sticker radius distribution is tightly centered at 100 px.
* No-sticker product code cannot import sticker-derived fields.

---

## Goal 1.2 — Reproduce preprocessing pipeline

### Tasks

* Run cattle segmentation:

  * fastest path: use provided masks where available for feature-pipeline validation;
  * defensible path: train/use CAFE-Net-like or PVTv2-B2 segmentation model.
* Run MMPose AnimalPose-style keypoint extraction.
* Run Depth Pro on masked cattle images.
* Estimate focal length from Depth Pro.
* Generate point clouds using pinhole projection.
* Apply point-cloud cleanup:

  * mask cattle region,
  * normalize z-axis,
  * downsample,
  * remove z-noise by thresholding/Otsu-like procedure.

The paper uses MMPose, segmentation, Depth Pro, estimated focal length, and pinhole projection to generate point clouds from 2D side-view images.  

### Acceptance criteria

* Segmentation, keypoint, depth, and point-cloud artifacts are produced for ≥ 95% of B3/B4 images.
* Segmentation performance is either:

  * within practical range of the paper’s reported CAFE-Net results, or
  * explicitly marked as using ground-truth/provided masks for reproduction speed.
* Point-cloud generation produces non-empty cattle-specific point clouds for ≥ 95% of accepted images.
* All preprocessing artifacts are traceable to the original image and transform history.

---

## Goal 1.3 — Reproduce feature extraction

### Tasks

Implement paper-style feature families:

* Body measurements from keypoints:

  * heart girth proxy,
  * withers height,
  * hip height,
  * tail-to-neck distance,
  * body length,
  * cylinder-style volume proxy.
* Basic contour features:

  * bbox width,
  * bbox height,
  * perimeter,
  * area.
* Hu moments.
* Statistics of contour-point distances from centroid.
* Convex hull features.
* Beam angle statistics.
* Chain-code statistics.
* MMPose keypoint distances.
* Point-cloud analogs where applicable.

The paper reports 84 features across these feature families, using keypoints, masks, and point clouds.  

### Acceptance criteria

* Feature table contains all paper feature families.
* Missing features are explicitly tagged as:

  * unavailable,
  * replaced,
  * not reproducible,
  * or deferred.
* Feature units and coordinate frames are recorded:

  * raw image,
  * sticker-normalized image,
  * model-input frame,
  * crop frame,
  * normalized cattle frame,
  * point-cloud frame.
* Feature extraction is deterministic for the same artifact version.

---

## Goal 1.4 — Reproduce statistical and model baselines

### Tasks

* Implement NMI filtering fitted on training data only.
* Train:

  * HGM baseline,
  * OLS,
  * LASSO,
  * stepwise p-value,
  * stepwise AIC,
  * Random Forest,
  * ANN.
* Evaluate:

  * B3 train/test,
  * B4 train/test,
  * B3 → B4,
  * B4 → B3.
* Report:

  * MAE,
  * MAPE,
  * RMSE,
  * residual plots,
  * predicted vs. actual plots.

The paper used MAE and MAPE, 80/20 B3/B4 splits, and cross-dataset evaluation. Its LASSO model trained on B3 reported 25.45 kg MAE / 16.81% MAPE on B3 test and 24.56 kg MAE / 14.23% MAPE on B4 cross-dataset test. 

### Acceptance criteria

* LASSO reproduction is within a predefined tolerance, initially:

  * ±3 MAPE points on B3 test,
  * ±3 MAPE points on B4 cross-dataset,
  * or discrepancy analysis is completed.
* HGM baseline is reproduced directionally as a weaker baseline.
* B3/B4 cross-dataset results are included.
* All transforms, feature filters, imputers, and scalers are fitted only on training folds.

---

# Phase 2 — No-sticker public-dataset ablation

**Timing:** Week 2–4
**Priority:** Mandatory. This quantifies how much the paper depends on the sticker.

## Goal 2.1 — Remove sticker signal from B3/B4

### Tasks

* Inpaint sticker from RGB image.
* Fill sticker hole in mask.
* Remove all sticker-related fields:

  * radius,
  * area,
  * location,
  * sticker-present flag,
  * detection confidence.
* Train a simple sticker-residual audit model:

  * Can it infer sticker presence/location from processed images?
  * Can it infer original sticker radius from remaining artifacts?

### Acceptance criteria

* Product feature matrix contains zero sticker-derived columns.
* Processed masks do not expose a circular sticker hole.
* Processed images do not retain obvious visual sticker signal.
* Sticker-residual audit is documented; if residual signal remains, images are reprocessed before no-sticker ablations are trusted.

---

## Goal 2.2 — Build no-sticker baselines

### Tasks

Train and evaluate:

1. **No-sticker raw-pixel feature baseline**

   * Keep raw pixel dimensions/features.
   * Shows how bad scale leakage/device dependence can be.

2. **No-sticker normalized-shape baseline**

   * Ratios only:

     * body length / bbox width,
     * withers height / bbox height,
     * area / bbox area,
     * hull area / mask area,
     * keypoint distances / bbox diagonal.

3. **No-sticker depth baseline**

   * Depth Pro features.
   * Focal-length estimate.
   * Point-cloud descriptors.
   * No sticker, no true scale.

4. **No-sticker hybrid baseline**

   * normalized shape + depth + keypoints + QA features.

### Acceptance criteria

* Gap report includes:

  * sticker-based reproduction,
  * no-sticker raw baseline,
  * no-sticker normalized-shape baseline,
  * no-sticker depth baseline,
  * no-sticker hybrid baseline.
* The team has a quantified estimate of performance loss caused by removing sticker normalization.
* Any raw-pixel feature that creates excessive resolution/device sensitivity is flagged as unsafe for production.

---

## Goal 2.3 — Resolution and device robustness harness

### Tasks

* Generate synthetic image variants:

  * 1.0×,
  * 0.75×,
  * 0.50×,
  * 0.33×,
  * JPEG compression,
  * mild blur,
  * letterbox/pad variants.
* Re-run full inference pipeline on variants.
* Measure:

  * prediction drift,
  * keypoint drift,
  * mask drift,
  * depth-feature drift,
  * predicted-measurement drift.
* Add leave-one-resolution and leave-one-device tests once real data is available.

### Acceptance criteria

* Every model candidate has a resolution stress report.
* Prediction drift across synthetic resolution variants is below a defined internal threshold or model is marked non-production.
* All resize/crop/letterbox transforms are stored and invertible.
* Aspect ratio is never stretched in production preprocessing.

---

# Phase 3 — Sale-barn data ingestion and governance

**Timing:** Week 3–6
**Priority:** Mandatory for product adaptation.

## Goal 3.1 — Build sale-barn data model

### Tasks

Create core tables:

```text
animal
sale_event
capture_session
frame
training_labels
cv_artifacts
features
predicted_measurements
predictions
```

Minimum fields:

```text
animal_id
sale_event_id
ranch_id
barn_id
lot_id
capture_datetime
image_path / video_path
scale_weight_kg
body_length_cm_true
heart_girth_cm_true
withers_height_cm_true
hip_height_cm_true
chest_depth_cm_true
sex
breed_type
age_class
device_model
image_width_px
image_height_px
EXIF fields
```

### Acceptance criteria

* ≥ 95% of images/videos are linked to animal/session/event identifiers, or missing linkage is explicitly flagged.
* Units are standardized:

  * kg,
  * cm,
  * px,
  * timestamp timezone.
* Measurement labels have source and timestamp fields.
* Runtime features and training-only labels are physically separated in storage and configs.

---

## Goal 3.2 — Implement split strategy and duplicate controls

### Tasks

* Primary split:

  * animal-level split.
* Secondary grouping:

  * session-level,
  * video-level,
  * event-level.
* Final validation:

  * sale event, ranch, barn, or date holdout.
* Duplicate detection:

  * perceptual hash,
  * image embedding similarity,
  * mask/keypoint geometry similarity,
  * video frame origin.

### Acceptance criteria

* No animal appears in both train and test.
* No video/session appears in both train and test.
* Near-duplicate frames are assigned to the same split.
* Final holdout is event/ranch/barn/date-based where data permits.
* Split report is generated before any model training.

---

## Goal 3.3 — Label quality and outlier audit

### Tasks

* Audit weight labels:

  * impossible weights,
  * extreme z-scores,
  * mismatched animal class,
  * duplicate images with conflicting weights.
* Audit physical measurements:

  * impossible dimensions,
  * unit errors,
  * inconsistent repeated measurements.
* Compare:

  * scale weight vs. HGM-style measurement estimate,
  * body length vs. visible body geometry,
  * height vs. age/sex/breed class.
* Create flags:

  * `label_suspect`,
  * `measurement_suspect`,
  * `image_suspect`,
  * `exclude_from_training`,
  * `include_for_robust_eval_only`.

The paper itself identified annotation inconsistencies, angular variation, and MMPose keypoint errors as contributors to prediction dispersion; its filtered B3 experiment improved to 11.62% MAPE on B3 and 10.02% MAPE on B4, reinforcing that label quality materially affects results. 

### Acceptance criteria

* Every excluded label has a documented reason.
* Training sets can be generated in:

  * raw mode,
  * conservative-clean mode,
  * robust-loss mode.
* Label audit report is reviewed before sale-barn model training.
* Outlier removal is never applied to final holdout metrics without reporting both raw and filtered results.

---

# Phase 4 — First deployable MVP model

**Timing:** Week 5–9
**Priority:** Highest product value.

This is the core replacement for sticker normalization.

## Goal 4.1 — Generate sale-barn runtime feature table

### Tasks

Run the full CV pipeline on sale-barn images/videos:

* QA:

  * single target cow,
  * full body visible,
  * broadside angle,
  * blur,
  * truncation,
  * occlusion,
  * cow size in frame.
* Segmentation.
* Keypoints.
* Depth Pro.
* Point cloud.
* Feature extraction:

  * paper-style features,
  * normalized-shape features,
  * depth features,
  * device/resolution features,
  * QA features.

### Acceptance criteria

* ≥ 90% of usable sale-barn images produce a complete feature row.
* Failed images have structured failure reasons.
* Feature matrix contains no true measurements, no sale weight, no lot average weight, and no sticker fields.
* Runtime feature schema is frozen for MVP training.

---

## Goal 4.2 — Train cross-fitted measurement predictors

### Tasks

Train image-derived predictors for:

```text
pred_body_length_cm
pred_heart_girth_cm
pred_withers_height_cm
pred_hip_height_cm
pred_chest_depth_cm
pred_scale_factor
```

Candidate models:

* ElasticNet / LASSO,
* Random Forest,
* Gradient boosting if available,
* small MLP only if tabular models plateau.

Cross-fitting procedure:

```text
for each fold:
  train measurement model on K-1 animal groups
  predict measurements for held-out animal group
concatenate OOF predicted measurements
```

### Acceptance criteria

* All downstream training uses **out-of-fold predicted measurements**, not in-sample predictions.
* Measurement MAPE is reported for every predicted measurement.
* Measurement predictors are evaluated by:

  * weight range,
  * sex,
  * breed/type,
  * age class,
  * capture condition,
  * device/resolution.
* Measurement predictions are calibrated enough to improve weight prediction over no-sticker naive baselines, or the failure is documented before proceeding.

---

## Goal 4.3 — Build predicted-HGM and residual correction model

### Tasks

* Compute predicted-HGM:

```text
pred_hgm_weight_kg = HGM(pred_heart_girth_cm, pred_body_length_cm)
```

* Train residual model:

```text
residual_kg = scale_weight_kg - pred_hgm_weight_kg
```

* Candidate residual models:

  * LASSO,
  * ElasticNet,
  * Huber regression,
  * Random Forest,
  * quantile regression model.

* Inputs:

  * predicted measurements,
  * predicted-HGM,
  * normalized shape features,
  * depth/point-cloud features,
  * QA features,
  * optional categorical metadata.

### Acceptance criteria

* Model beats no-sticker naive baseline on animal-level validation.
* Model beats predicted-HGM alone.
* Model reports:

  * MAE,
  * MAPE,
  * RMSE,
  * residual bias by subgroup,
  * residual bias by device/resolution,
  * residual bias by weight range.
* Feature importance or coefficients are explainable enough for an internal model review.
* Serialized production model contains no training-only labels.

---

## Goal 4.4 — Add uncertainty and rejection rules

### Tasks

Implement:

* prediction intervals:

  * ensemble variance,
  * quantile regression,
  * or conformal calibration;
* rejection rules:

  * cow too small in frame,
  * cow truncated,
  * low mask confidence,
  * low keypoint confidence,
  * bad broadside angle,
  * severe occlusion,
  * out-of-distribution device/capture condition,
  * excessive prediction interval width.

### Acceptance criteria

* API returns either:

  * accepted prediction with interval, or
  * rejected prediction with retake reason.
* 80% and 90% prediction interval coverage are reported on calibration and validation splits.
* Rejection improves accepted-sample MAPE without hiding the rejection rate.
* Rejection rate is reported overall and by capture condition.

---

## Goal 4.5 — Ship internal MVP inference service

### Tasks

* Package model pipeline.
* Create batch prediction CLI:

```bash
cw predict --input image_or_video --model mvp_v0
```

* Create service endpoint:

```text
POST /predict-cattle-weight
```

* Log:

  * input metadata,
  * QA flags,
  * feature schema version,
  * model version,
  * prediction,
  * interval,
  * rejection reason.

### Acceptance criteria

* Internal users can submit image or video and receive a prediction/retake decision.
* End-to-end latency is acceptable for internal testing.
* Every prediction is reproducible from stored model and artifact versions.
* MVP model card exists with limitations, metrics, data coverage, and known failure modes.

---

# Phase 5 — Realistic validation and product hardening

**Timing:** Week 8–12
**Priority:** Mandatory before external pilot.

## Goal 5.1 — Final event/ranch/barn/date holdout

### Tasks

* Freeze model candidates before final evaluation.
* Evaluate on held-out sale event, ranch, barn, or date.
* Report:

  * MAE,
  * MAPE,
  * RMSE,
  * median absolute error,
  * 80/90% interval coverage,
  * accepted vs. rejected metrics,
  * interval width,
  * subgroup metrics.

### Acceptance criteria

* Final holdout was not used for feature selection, tuning, threshold setting, or calibration.
* Report includes the required gap table:

| Model                             | Runtime deployable? | MAE | MAPE | RMSE | 90% PI coverage |
| --------------------------------- | ------------------: | --: | ---: | ---: | --------------: |
| Sticker reproduction              |                  No |     |      |      |                 |
| No-sticker naive                  |                 Yes |     |      |      |                 |
| Oracle measurement-assisted       |                  No |     |      |      |                 |
| Measurement-supervised image-only |                 Yes |     |      |      |                 |
| Teacher-student image-only        |    Yes, if complete |     |      |      |                 |

* The team can quantify how much of the oracle gap is closed.

---

## Goal 5.2 — Device, resolution, and capture robustness validation

### Tasks

* Collect same-animal test sets:

  * multiple devices,
  * multiple resolutions,
  * near/medium/far distance,
  * 1080p video,
  * 4K video,
  * different framing.
* Measure prediction drift for same animal.
* Test app capture protocol:

  * landscape,
  * full body,
  * broadside,
  * mid-body camera height,
  * cow fills target fraction of frame,
  * no digital zoom unless controlled.

### Acceptance criteria

* Same-animal, different-device prediction drift is within an acceptable operational threshold or triggers wider uncertainty.
* Resolution downsampling does not materially change prediction for accepted images.
* Capture protocol reduces rejection rate and prediction variance.
* Device metadata is either retained because it improves robustness or removed because it causes device-specific overfit.

---

## Goal 5.3 — Subgroup and failure-mode review

### Tasks

Evaluate by:

* weight range,
* sex,
* breed/type,
* age class,
* barn/ranch/event,
* device,
* image resolution,
* cow size in frame,
* side angle,
* lighting,
* occlusion,
* coat color,
* image/video source.

### Acceptance criteria

* Worst major subgroup MAPE is either within a defined tolerance or explicitly labeled out-of-scope.
* Model card includes known weak zones.
* Rejection rules catch the major visually obvious failure modes.
* External pilot criteria are explicitly approved.

---

# Phase 6 — Performance upgrades after MVP

**Timing:** Week 10–16+
**Priority:** Conditional. Do these only after the MVP baseline is measured.

## Goal 6.1 — Video aggregation

### Tasks

* Add frame sampler.
* Score frames for broadside quality.
* Predict per frame.
* Aggregate:

  * median prediction,
  * confidence-weighted mean,
  * interval from frame dispersion.
* Reject videos with no acceptable broadside frame.

### Acceptance criteria

* Video aggregation improves MAE/MAPE or uncertainty calibration over best single frame.
* Frame-level variance correlates with prediction error.
* Retake guidance is generated automatically.

---

## Goal 6.2 — Multi-task auxiliary measurement model

### Tasks

* Train shared representation using:

  * image/crop embeddings,
  * mask features,
  * keypoint features,
  * depth features,
  * tabular metadata.
* Predict:

  * weight,
  * body length,
  * heart girth,
  * withers height,
  * hip height,
  * chest depth.
* Compare against Phase 4 tabular residual model.

### Acceptance criteria

* Multi-task model beats Phase 4 on final-style validation, not just random folds.
* Measurement heads improve biological plausibility.
* Model remains auditable through predicted measurements and residual decomposition.
* If not better, retain Phase 4 model and archive this as research.

---

## Goal 6.3 — Teacher-student distillation

### Tasks

* Train privileged teacher using:

  * true sale-barn measurements,
  * true metadata,
  * image-derived features.
* Train student using:

  * image/video-derived features only.
* Distill:

  * teacher weight prediction,
  * teacher residual,
  * teacher uncertainty,
  * latent body-size embedding.

### Acceptance criteria

* Teacher materially beats image-only model, proving privileged measurements contain useful signal.
* Student closes additional oracle gap over Phase 4 model.
* Deployed student has no privileged input fields.
* Distillation improves final holdout or is deferred.

---

## Goal 6.4 — Metric-depth calibration and environment profiles

### Tasks

* Learn depth scale correction from:

  * Depth Pro focal estimate,
  * depth map statistics,
  * mask geometry,
  * keypoint spread,
  * EXIF/device metadata,
  * optional fixed environment profile.
* For controlled barns/chutes, test background geometry:

  * rail height,
  * chute width,
  * alley width,
  * fixed camera position,
  * ground plane.

### Acceptance criteria

* Depth calibration improves measurement-prediction MAPE and/or weight MAPE on held-out barns/events.
* Environment profile does not leak weight labels or lot information.
* Consumer mode still works without environment profile.
* Environment mode is treated as optional enhancement, not base requirement.

---

# Phase 7 — Production pilot and monitoring

**Timing:** Week 12–18
**Priority:** Required for real deployment.

## Goal 7.1 — Production-ready service

### Tasks

* Containerize inference pipeline.
* Add GPU/CPU execution modes.
* Add model registry.
* Add feature schema validation at serving time.
* Add request/response audit logs.
* Add privacy and retention controls for images/videos.

### Acceptance criteria

* Production service rejects malformed or forbidden payloads.
* Model version is pinned per prediction.
* Reproducibility test passes for historical predictions.
* Feature schema mismatch blocks serving rather than silently degrading.

---

## Goal 7.2 — Pilot monitoring dashboard

### Tasks

Track:

* prediction distribution,
* rejection rate,
* retake reasons,
* device/resolution distribution,
* capture quality,
* prediction interval width,
* drift against training distribution,
* error once true weights are available.

### Acceptance criteria

* Dashboard shows daily/weekly model health.
* Alerts fire for:

  * high rejection rate,
  * device/domain shift,
  * abnormal prediction distribution,
  * interval widening,
  * high error on newly labeled pilot data.
* Pilot data can be promoted into training only through reviewed ingestion.

---

## Goal 7.3 — Retraining and model governance

### Tasks

* Define retraining cadence.
* Add active-learning queue:

  * high uncertainty,
  * rejected images,
  * new device types,
  * rare breeds/classes,
  * large residuals after labels arrive.
* Maintain model comparison suite:

  * current production,
  * candidate replacement,
  * baseline LASSO,
  * no-sticker naive,
  * oracle measurement upper bound.

### Acceptance criteria

* No model is promoted without final-style holdout evaluation.
* New model must beat current production or improve a specific approved weakness.
* Regression tests prevent forbidden feature leakage.
* Model card is updated with every release.

---

# Critical path for speed

The minimum credible path to a working internal system is:

```text
Phase 0
  -> Phase 1 reproduction baseline
  -> Phase 2 no-sticker ablation
  -> Phase 3 sale-barn ingestion/splits
  -> Phase 4 measurement-supervised residual MVP
  -> Phase 5 final-style validation
```

The fastest internal MVP is likely **8–9 weeks** if sale-barn labels and images are available early. External pilot readiness is more likely **12+ weeks**, because validation, leakage controls, device robustness, and rejection behavior are not optional.

---

# What to do immediately in the first sprint

**Sprint length:** 2 weeks.

## Sprint Goal 1 — Get B3/B4 paper reproduction scaffold running

### Tasks

* Ingest B3/B4.
* Parse weights.
* Detect sticker.
* Normalize sticker radius to 100 px.
* Generate segmentation/keypoint/depth artifacts for a subset.
* Extract first version of paper-style features.
* Train LASSO and HGM on a subset.

### Acceptance criteria

* End-to-end run completes on a representative subset.
* Weight parsing and sticker normalization are verified.
* Feature table contains at least:

  * keypoint distances,
  * bbox/contour features,
  * Hu moments,
  * centroid-distance stats,
  * convex hull features.
* Early LASSO/HGM results are directionally plausible.

---

## Sprint Goal 2 — Stand up the no-sticker ablation harness

### Tasks

* Inpaint stickers.
* Fill mask holes.
* Remove sticker fields from feature configs.
* Build raw-pixel and normalized-shape baselines.
* Add resolution stress variants.

### Acceptance criteria

* Same image can be evaluated in:

  * sticker-normalized mode,
  * no-sticker raw mode,
  * no-sticker normalized-shape mode.
* Leakage audit blocks sticker fields.
* Initial report quantifies prediction drift under downsampling.

---

## Sprint Goal 3 — Define sale-barn schema and leakage audit before data lands

### Tasks

* Create schema files.
* Define forbidden columns.
* Implement feature role validation.
* Implement split config:

  * animal-level,
  * session-level,
  * event-level final holdout.
* Create duplicate-detection plan.

### Acceptance criteria

* Sale-barn data can be ingested without redesigning schemas.
* Any feature matrix containing true measurements or sale weight as input fails CI.
* Split generation cannot place the same animal/session/video in train and test.

---

# Time worth defending

Do not cut these:

1. **Leakage controls.** Without them, the model can silently learn sale weight, lot averages, true measurements, or sticker remnants.
2. **Animal/session/event splits.** Image-level random splits will overstate performance.
3. **No-sticker ablation.** Without it, we will not know how much the paper depended on sticker scale normalization.
4. **Cross-fitted predicted measurements.** In-sample predicted measurements will make the residual model look better than it is.
5. **Uncertainty and rejection.** A no-sticker single-image system has irreducible scale ambiguity; the product must know when not to provide a confident estimate.
6. **Device/resolution stress testing.** Different devices and resolutions can create false scale cues, especially if raw pixel features leak into the model.

Cut or defer these initially:

* morphable cattle body model,
* full custom pose-labeling campaign,
* heavy end-to-end ViT training,
* environment-specific calibration profiles,
* teacher-student distillation,
* extensive ANN hyperparameter searches.

Those are useful later, but they are not required to get a defensible MVP working quickly.
