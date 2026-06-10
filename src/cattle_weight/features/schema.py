from __future__ import annotations

from dataclasses import dataclass


STATS = ["mean", "std", "p25", "p75", "min", "max", "median"]

BODY_MEASUREMENT_FEATURES = [
    "bm_heart_girth_px",
    "bm_withers_height_px",
    "bm_hip_height_px",
    "bm_tail_distance_to_neck_px",
    "bm_body_length_px",
    "bm_cylinder_volume_px3",
]

BASIC_CONTOUR_FEATURES = [
    "bc_bbox_width_px",
    "bc_bbox_height_px",
    "bc_contour_perimeter_px",
]

HU_FEATURES = [f"hm_{idx}" for idx in range(1, 8)]

MASK_CENTROID_DISTANCE_FEATURES = [f"mask_centroid_dist_{name}" for name in STATS]
PC_CENTROID_DISTANCE_FEATURES = [f"pc_centroid_dist_{name}" for name in STATS]

MASK_HULL_FEATURES = [
    "mask_hull_area_px2",
    "mask_hull_perimeter_px",
    "mask_solidity",
]

PC_HULL_FEATURES = [
    "pc_hull_area",
    "pc_hull_perimeter",
    "pc_solidity",
]

MASK_BEAM_ANGLE_FEATURES = [f"mask_beam_angle_{name}" for name in STATS]
PC_BEAM_ANGLE_FEATURES = [f"pc_beam_angle_{name}" for name in STATS]

MASK_CHAIN_CODE_FEATURES = [f"mask_chain_code_{name}" for name in STATS]
PC_CHAIN_CODE_FEATURES = [f"pc_chain_code_{name}" for name in STATS]

KEYPOINT_DISTANCE_FEATURES = [
    "kpd_l_eye_r_eye_px",
    "kpd_l_eye_l_earbase_px",
    "kpd_r_eye_r_earbase_px",
    "kpd_l_eye_nose_px",
    "kpd_r_eye_nose_px",
    "kpd_nose_throat_px",
    "kpd_throat_withers_px",
    "kpd_tailbase_withers_px",
    "kpd_throat_l_f_elbow_px",
    "kpd_l_f_elbow_l_f_knee_px",
    "kpd_l_f_knee_l_f_paw_px",
    "kpd_throat_r_f_elbow_px",
    "kpd_r_f_elbow_r_f_knee_px",
    "kpd_r_f_knee_r_f_paw_px",
    "kpd_tailbase_l_b_elbow_px",
    "kpd_l_b_elbow_l_b_knee_px",
    "kpd_l_b_knee_l_b_paw_px",
    "kpd_tailbase_r_b_elbow_px",
    "kpd_r_b_elbow_r_b_knee_px",
    "kpd_r_b_knee_r_b_paw_px",
]

PAPER_FEATURES = (
    BODY_MEASUREMENT_FEATURES
    + BASIC_CONTOUR_FEATURES
    + HU_FEATURES
    + MASK_CENTROID_DISTANCE_FEATURES
    + PC_CENTROID_DISTANCE_FEATURES
    + MASK_HULL_FEATURES
    + PC_HULL_FEATURES
    + MASK_BEAM_ANGLE_FEATURES
    + PC_BEAM_ANGLE_FEATURES
    + MASK_CHAIN_CODE_FEATURES
    + PC_CHAIN_CODE_FEATURES
    + KEYPOINT_DISTANCE_FEATURES
)

ANIMALPOSE_KEYPOINTS = [
    "L_Eye",
    "R_Eye",
    "L_EarBase",
    "R_EarBase",
    "Nose",
    "Throat",
    "TailBase",
    "Withers",
    "L_F_Elbow",
    "R_F_Elbow",
    "L_B_Elbow",
    "R_B_Elbow",
    "L_F_Knee",
    "R_F_Knee",
    "L_B_Knee",
    "R_B_Knee",
    "L_F_Paw",
    "R_F_Paw",
    "L_B_Paw",
    "R_B_Paw",
]

KEYPOINT_LINKS = [
    ("kpd_l_eye_r_eye_px", 0, 1),
    ("kpd_l_eye_l_earbase_px", 0, 2),
    ("kpd_r_eye_r_earbase_px", 1, 3),
    ("kpd_l_eye_nose_px", 0, 4),
    ("kpd_r_eye_nose_px", 1, 4),
    ("kpd_nose_throat_px", 4, 5),
    ("kpd_throat_withers_px", 5, 7),
    ("kpd_tailbase_withers_px", 6, 7),
    ("kpd_throat_l_f_elbow_px", 5, 8),
    ("kpd_l_f_elbow_l_f_knee_px", 8, 12),
    ("kpd_l_f_knee_l_f_paw_px", 12, 16),
    ("kpd_throat_r_f_elbow_px", 5, 9),
    ("kpd_r_f_elbow_r_f_knee_px", 9, 13),
    ("kpd_r_f_knee_r_f_paw_px", 13, 17),
    ("kpd_tailbase_l_b_elbow_px", 6, 10),
    ("kpd_l_b_elbow_l_b_knee_px", 10, 14),
    ("kpd_l_b_knee_l_b_paw_px", 14, 18),
    ("kpd_tailbase_r_b_elbow_px", 6, 11),
    ("kpd_r_b_elbow_r_b_knee_px", 11, 15),
    ("kpd_r_b_knee_r_b_paw_px", 15, 19),
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    units: str
    coordinate_frame: str
    source_artifact: str
    paper_derived: bool
    missingness_rule: str


def build_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for name in BODY_MEASUREMENT_FEATURES:
        specs.append(FeatureSpec(name, "body_measurement", "px_or_px3", "sticker-normalized image", "keypoints", True, "NaN if required keypoint missing"))
    for name in BASIC_CONTOUR_FEATURES + HU_FEATURES:
        specs.append(FeatureSpec(name, "mask_shape", "px", "sticker-normalized image", "segmentation_mask", True, "NaN if mask missing or empty"))
    for name in MASK_CENTROID_DISTANCE_FEATURES + MASK_HULL_FEATURES + MASK_BEAM_ANGLE_FEATURES + MASK_CHAIN_CODE_FEATURES:
        specs.append(FeatureSpec(name, "mask_geometry", "px", "sticker-normalized image", "segmentation_mask", True, "NaN if mask contour cannot be computed"))
    for name in PC_CENTROID_DISTANCE_FEATURES + PC_HULL_FEATURES + PC_BEAM_ANGLE_FEATURES + PC_CHAIN_CODE_FEATURES:
        specs.append(FeatureSpec(name, "pointcloud_geometry", "depth units", "point-cloud frame", "pointcloud", True, "NaN if point cloud missing or degenerate"))
    for name in KEYPOINT_DISTANCE_FEATURES:
        specs.append(FeatureSpec(name, "keypoint_distance", "px", "sticker-normalized image", "keypoints", True, "NaN if either keypoint missing"))
    if len(specs) != 84:
        raise AssertionError(f"Expected 84 feature specs, got {len(specs)}")
    return specs
