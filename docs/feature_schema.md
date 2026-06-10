# Feature Schema

The canonical feature schema is written to `data_registry/feature_schema.yaml` during feature extraction. It contains 84 paper-style feature names:

- 6 body measurements.
- 3 basic contour features.
- 7 Hu moments.
- 7 mask centroid-distance statistics.
- 7 point-cloud centroid-distance statistics.
- 3 mask convex-hull features.
- 3 point-cloud convex-hull features.
- 7 mask beam-angle statistics.
- 7 point-cloud beam-angle statistics.
- 7 mask chain-code statistics.
- 7 point-cloud chain-code statistics.
- 20 AnimalPose keypoint-link distances.

Each schema row records family, units, coordinate frame, source artifact, paper-derived status, and missingness rule.
