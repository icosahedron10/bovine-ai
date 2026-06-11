from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cattle_weight.cv.segmentation import required_pvtv2_segmentation_source, should_require_pvtv2_inference
from cattle_weight.eval.metrics import mae, mape
from cattle_weight.features.schema import PAPER_FEATURES
from cattle_weight.io import artifact_metadata, figures_dir, models_dir, read_parquet, registry_dir, reports_dir, write_json
from cattle_weight.progress import logger
from cattle_weight.reports.depth import write_depth_completion_summary
from cattle_weight.reports.status import write_headline_reproduction_status


def load_feature_frame(config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for dataset in config.get("data", {}).get("datasets", ["B3", "B4"]):
        path = registry_dir(config) / f"features_{dataset.lower()}.parquet"
        if path.exists():
            frames.append(read_parquet(path))
    if not frames:
        raise FileNotFoundError("No feature tables found. Run `cw features extract` first.")
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["feature_status"] == "success"].reset_index(drop=True)


def load_training_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_headline_reproduction_ready(config)
    features = load_feature_frame(config)
    splits = read_parquet(registry_dir(config) / "splits.parquet")
    merged = features.merge(splits[["image_id", "split", "split_version"]], on="image_id", how="inner")
    if merged.empty:
        raise RuntimeError("No features matched split membership.")
    return merged, splits


def fit_nmi_selector(train: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame]:
    scores = []
    y = train["weight_kg"].to_numpy(dtype=float)
    y_disc = discretize_for_nmi(y)
    for feature in PAPER_FEATURES:
        x = train[feature].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3 or np.nanstd(x[valid]) == 0:
            score = 0.0
        else:
            score = float(normalized_mutual_info_score(discretize_for_nmi(x[valid]), y_disc[valid]))
        scores.append({"feature": feature, "nmi": score, "selected": score > threshold})
    summary = pd.DataFrame(scores)
    selected = summary.loc[summary["selected"], "feature"].tolist()
    if not selected:
        selected = summary.sort_values("nmi", ascending=False)["feature"].head(1).tolist()
        summary.loc[summary["feature"].isin(selected), "selected"] = True
    return selected, summary


def training_split_version(train: pd.DataFrame, dataset: str) -> str | None:
    if "split_version" not in train.columns:
        return None
    versions = sorted(str(value) for value in train["split_version"].dropna().unique())
    if len(versions) > 1:
        raise RuntimeError(f"Training rows for {dataset} contain multiple split_version values: {versions}")
    return versions[0] if versions else None


def discretize_for_nmi(values: np.ndarray, bins: int = 10) -> np.ndarray:
    series = pd.Series(values)
    unique = series.nunique(dropna=True)
    if unique <= 1:
        return np.zeros(len(series), dtype=int)
    q = min(bins, int(unique))
    try:
        return pd.qcut(series, q=q, labels=False, duplicates="drop").fillna(-1).astype(int).to_numpy()
    except ValueError:
        return pd.cut(series, bins=q, labels=False, duplicates="drop").fillna(-1).astype(int).to_numpy()


def train_models(config: dict[str, Any]) -> dict[str, Any]:
    frame, splits = load_training_tables(config)
    run_id = config.get("run", {}).get("run_id", "run")
    out_dir = models_dir(config) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("run", {}).get("random_seed", 0))
    threshold = float(config.get("features", {}).get("nmi_threshold", 0.0))
    all_summaries = []
    bundles = {}
    logger().info(
        "Training setup | feature_rows=%d | split_rows=%d | run_id=%s | nmi_threshold=%s | output_dir=%s",
        len(frame),
        len(splits),
        run_id,
        threshold,
        out_dir,
    )
    for dataset in config.get("data", {}).get("datasets", ["B3", "B4"]):
        train = frame[(frame["dataset"] == dataset) & (frame["split"] == "train")].copy()
        if len(train) < 2:
            raise RuntimeError(f"Need at least 2 training rows for {dataset}; got {len(train)}")
        logger().info("Training LASSO | dataset=%s | train_rows=%d", dataset, len(train))
        selected, nmi_summary = fit_nmi_selector(train, threshold)
        logger().info("NMI selection | dataset=%s | selected_features=%d", dataset, len(selected))
        nmi_summary.insert(0, "train_dataset", dataset)
        all_summaries.append(nmi_summary)
        X_train = train[selected].to_numpy(dtype=float)
        y_train = train["weight_kg"].to_numpy(dtype=float)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X_train)
        X_scaled = scaler.fit_transform(X_imp)
        model = fit_lasso(config, X_scaled, y_train, seed)
        split_version = training_split_version(train, dataset)
        bundle = {
            "train_dataset": dataset,
            "selected_features": selected,
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "metadata": artifact_metadata(
                config,
                data_version=train["data_version"].iloc[0] if "data_version" in train else None,
                split_version=split_version,
                feature_schema_version=config.get("features", {}).get("schema_version"),
                model_parameters={
                    "model": "LassoCV",
                    "alpha": float(getattr(model, "alpha_", getattr(model, "alpha", np.nan))),
                    "nmi_threshold": threshold,
                    "selected_feature_count": len(selected),
                    "selected_features": selected,
                },
            ),
        }
        path = out_dir / f"lasso_{dataset}.joblib"
        joblib.dump(bundle, path)
        bundles[dataset] = str(path.resolve())
        logger().info(
            "LASSO bundle written | dataset=%s | alpha=%s | selected_features=%d | path=%s",
            dataset,
            getattr(model, "alpha_", getattr(model, "alpha", "unknown")),
            len(selected),
            path,
        )
    selection = pd.concat(all_summaries, ignore_index=True)
    selection.to_csv(reports_dir(config) / "feature_selection_summary.csv", index=False)
    logger().info("Feature selection summary written | rows=%d | path=%s", len(selection), reports_dir(config) / "feature_selection_summary.csv")
    hgm = {
        "model": "Heart Girth Method",
        "formula": "kg = ((heart_girth_in^2 * body_length_in) / 300) * 0.45359237",
        "paper_formula": "(Heart Girth^2 * Body Length) / 300, inches to pounds; converted to kg",
        "metadata": artifact_metadata(config, model_parameters={"model": "HGM"}),
    }
    write_json(hgm, out_dir / "hgm_metadata.json")
    write_json({"lasso_bundles": bundles, "hgm": hgm}, out_dir / "training_manifest.json")
    logger().info("Training complete | lasso_bundles=%d | manifest=%s", len(bundles), out_dir / "training_manifest.json")
    return {"lasso_bundles": bundles, "hgm": hgm}


def fit_lasso(config: dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    alphas = np.asarray(config.get("model", {}).get("lasso_alphas", [0.01, 0.1, 1.0]), dtype=float)
    cv_folds = min(int(config.get("model", {}).get("cv_folds", 5)), len(y))
    if cv_folds >= 2:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        model = LassoCV(alphas=alphas, cv=cv, random_state=seed, max_iter=10000)
        return model.fit(X, y)
    model = Lasso(alpha=float(alphas[0]), max_iter=10000, random_state=seed)
    return model.fit(X, y)


def predict_lasso(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    selected = bundle["selected_features"]
    X = frame[selected].to_numpy(dtype=float)
    X_imp = bundle["imputer"].transform(X)
    X_scaled = bundle["scaler"].transform(X_imp)
    return bundle["model"].predict(X_scaled)


def predict_hgm(config: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    target_radius_px = float(config.get("sticker", {}).get("target_radius_px", 100.0))
    radius_inches = float(config.get("sticker", {}).get("reference_radius_inches", 1.0))
    if radius_inches <= 0:
        raise ValueError("sticker.reference_radius_inches must be positive for HGM.")
    px_per_inch = target_radius_px / radius_inches
    heart_girth_in = frame["bm_heart_girth_px"].to_numpy(dtype=float) / px_per_inch
    body_length_in = frame["bm_body_length_px"].to_numpy(dtype=float) / px_per_inch
    pounds = (heart_girth_in**2 * body_length_in) / 300.0
    return pounds * 0.45359237


def evaluate_models(config: dict[str, Any]) -> pd.DataFrame:
    frame, _ = load_training_tables(config)
    run_id = config.get("run", {}).get("run_id", "run")
    out_dir = models_dir(config) / run_id
    if not out_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {out_dir}. Run `cw train` first.")
    datasets = config.get("data", {}).get("datasets", ["B3", "B4"])
    logger().info("Evaluation setup | feature_rows=%d | datasets=%s | model_dir=%s", len(frame), datasets, out_dir)
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for dataset in datasets:
        eval_frame = frame[(frame["dataset"] == dataset) & (frame["split"] == "test")].copy()
        if eval_frame.empty:
            continue
        pred = predict_hgm(config, eval_frame)
        row = metric_row("HGM", "none", dataset, "test", eval_frame["weight_kg"].to_numpy(), pred)
        rows.append(row)
        logger().info("Evaluation metric | model=HGM | train_dataset=none | eval_dataset=%s | n=%d | mae=%.4f | mape=%.4f", dataset, row["n"], row["mae"], row["mape"])
        prediction_rows.append(prediction_frame("HGM", "none", dataset, eval_frame, pred))
    for train_dataset in datasets:
        bundle_path = out_dir / f"lasso_{train_dataset}.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing trained LASSO bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)
        for eval_dataset in datasets:
            eval_frame = frame[(frame["dataset"] == eval_dataset) & (frame["split"] == "test")].copy()
            if eval_frame.empty:
                continue
            pred = predict_lasso(bundle, eval_frame)
            label = "test" if train_dataset == eval_dataset else "cross_dataset_test"
            row = metric_row("LASSO", train_dataset, eval_dataset, label, eval_frame["weight_kg"].to_numpy(), pred)
            rows.append(row)
            logger().info(
                "Evaluation metric | model=LASSO | train_dataset=%s | eval_dataset=%s | split=%s | n=%d | mae=%.4f | mape=%.4f",
                train_dataset,
                eval_dataset,
                label,
                row["n"],
                row["mae"],
                row["mape"],
            )
            prediction_rows.append(prediction_frame("LASSO", train_dataset, eval_dataset, eval_frame, pred))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports_dir(config) / "reproduction_metrics.csv", index=False)
    write_depth_completion_summary(config)
    write_coverage_funnel(config)
    gaps = write_metric_gaps(metrics, config)
    write_metrics_markdown(metrics, reports_dir(config) / "reproduction_metrics.md")
    logger().info("Metrics written | rows=%d | path=%s", len(metrics), reports_dir(config) / "reproduction_metrics.csv")
    if prediction_rows:
        predictions = pd.concat(prediction_rows, ignore_index=True)
        predictions.to_csv(reports_dir(config) / "predictions.csv", index=False)
        write_evaluation_plots(predictions, config)
        logger().info("Predictions and plots written | prediction_rows=%d | figures_dir=%s", len(predictions), figures_dir(config))
    write_discrepancy_report(metrics, gaps, config)
    write_evaluation_headline_status(config)
    logger().info("Evaluation complete | discrepancy_report=%s", reports_dir(config) / "reproduction_discrepancies.md")
    return metrics


def write_evaluation_headline_status(config: dict[str, Any]) -> None:
    if config.get("run", {}).get("mode") != "full":
        write_headline_reproduction_status(
            config,
            "partial",
            "evaluation",
            "Smoke or non-full runs are not headline reproduction runs.",
            metrics_allowed=False,
        )
        return
    try:
        from cattle_weight.validation import validate_headline

        validate_headline(config, write_status=True)
    except Exception as exc:
        write_headline_reproduction_status(config, "blocked", "evaluation", str(exc), metrics_allowed=False)


def assert_headline_reproduction_ready(config: dict[str, Any]) -> None:
    if not should_require_pvtv2_inference(config):
        return

    cv = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    cv_success = cv[cv["segmentation_status"] == "success"]
    if cv_success.empty:
        raise RuntimeError("Headline metrics require successful PVTv2 segmentation rows.")
    expected_source = required_pvtv2_segmentation_source(config)
    invalid_sources = sorted(set(cv_success.loc[cv_success["segmentation_source"] != expected_source, "segmentation_source"].astype(str)))
    if invalid_sources:
        raise RuntimeError(f"Headline metrics require frozen PVTv2-B2 saliency masks; found sources={invalid_sources}.")

    depth = read_parquet(registry_dir(config) / "depth_metadata.parquet")
    depth_success = depth[depth["depth_status"] == "success"]
    if depth_success.empty:
        raise RuntimeError("Headline metrics require successful Depth Pro rows.")
    invalid_depth = sorted(set(depth_success.loc[depth_success["depth_provider"] != "depth_pro", "depth_provider"].astype(str)))
    if invalid_depth:
        raise RuntimeError(f"Headline metrics require depth_provider=depth_pro; found providers={invalid_depth}.")


def write_coverage_funnel(config: dict[str, Any]) -> pd.DataFrame:
    index = read_parquet(registry_dir(config) / "dataset_index.parquet")
    splits = read_parquet(registry_dir(config) / "splits.parquet")
    sticker_path = registry_dir(config) / "sticker_metadata.parquet"
    if not sticker_path.exists():
        raise RuntimeError(f"Coverage funnel requires sticker metadata at {sticker_path}. Run `cw stickers normalize` before `cw evaluate`.")
    sticker = read_parquet(sticker_path)
    cv = read_parquet(registry_dir(config) / "cv_artifacts.parquet")
    keypoints = read_parquet(registry_dir(config) / "keypoint_metadata.parquet")
    depth = read_parquet(registry_dir(config) / "depth_metadata.parquet")
    pointcloud = read_parquet(registry_dir(config) / "pointcloud_metadata.parquet")
    features = load_feature_frame(config)

    rows: list[dict[str, Any]] = []
    for dataset in config.get("data", {}).get("datasets", ["B3", "B4"]):
        ids = set(index.loc[index["dataset"] == dataset, "image_id"].astype(str))
        sticker_ids = set(sticker.loc[(sticker["dataset"] == dataset) & (sticker["detection_status"] == "success"), "image_id"].astype(str))
        mask_ids = set(
            cv.loc[
                (cv["dataset"] == dataset)
                & (cv["segmentation_status"] == "success")
                & (cv["segmentation_source"] == required_pvtv2_segmentation_source(config)),
                "image_id",
            ].astype(str)
        )
        keypoint_ids = set(keypoints.loc[(keypoints["dataset"] == dataset) & (keypoints["keypoint_status"] == "success"), "image_id"].astype(str))
        depth_ids = set(
            depth.loc[
                (depth["dataset"] == dataset) & (depth["depth_status"] == "success") & (depth["depth_provider"] == "depth_pro"),
                "image_id",
            ].astype(str)
        )
        pointcloud_ids = set(pointcloud.loc[(pointcloud["dataset"] == dataset) & (pointcloud["pointcloud_status"] == "success"), "image_id"].astype(str))
        feature_ids = set(features.loc[features["dataset"] == dataset, "image_id"].astype(str))
        usable = feature_ids & set(splits.loc[splits["dataset"] == dataset, "image_id"].astype(str))
        rows.append(
            {
                "dataset": dataset,
                "indexed": len(ids),
                "sticker": len(ids & sticker_ids),
                "pvtv2_mask": len(ids & sticker_ids & mask_ids),
                "keypoints": len(ids & sticker_ids & mask_ids & keypoint_ids),
                "depth_pro": len(ids & sticker_ids & mask_ids & keypoint_ids & depth_ids),
                "pointcloud": len(ids & sticker_ids & mask_ids & keypoint_ids & depth_ids & pointcloud_ids),
                "features": len(usable),
                "usable_train": int(splits.loc[splits["image_id"].astype(str).isin(usable) & (splits["split"] == "train"), "image_id"].nunique()),
                "usable_test": int(splits.loc[splits["image_id"].astype(str).isin(usable) & (splits["split"] == "test"), "image_id"].nunique()),
            }
        )
    funnel = pd.DataFrame(rows)
    reports_dir(config).mkdir(parents=True, exist_ok=True)
    funnel.to_csv(reports_dir(config) / "coverage_funnel.csv", index=False)
    return funnel


def write_metric_gaps(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    targets = {
        ("HGM", "none", "B3", "test"): {"target_mae": 40.69, "target_mape": 24.63},
        ("HGM", "none", "B4", "test"): {"target_mae": 53.65, "target_mape": 31.36},
        ("LASSO", "B3", "B3", "test"): {"target_mae": 25.45, "target_mape": 16.81},
        ("LASSO", "B3", "B4", "cross_dataset_test"): {"target_mae": 24.56, "target_mape": 14.23},
    }
    rows = []
    for _, row in metrics.iterrows():
        key = (row["model"], row["train_dataset"], row["eval_dataset"], row["eval_split"])
        target = targets.get(key)
        if not target:
            continue
        mae_gap = float(row["mae"] - target["target_mae"])
        mape_gap = float(row["mape"] - target["target_mape"])
        rows.append(
            {
                "model": row["model"],
                "train_dataset": row["train_dataset"],
                "eval_dataset": row["eval_dataset"],
                "eval_split": row["eval_split"],
                "n": row["n"],
                "mae": row["mae"],
                "target_mae": target["target_mae"],
                "mae_gap_abs": mae_gap,
                "mae_gap_pct": mae_gap / target["target_mae"] * 100.0,
                "mape": row["mape"],
                "target_mape": target["target_mape"],
                "mape_gap_abs": mape_gap,
                "mape_gap_pct": mape_gap / target["target_mape"] * 100.0,
            }
        )
    gaps = pd.DataFrame(rows)
    gaps.to_csv(reports_dir(config) / "paper_metric_gaps.csv", index=False)
    return gaps


def metric_row(model: str, train_dataset: str, eval_dataset: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "eval_split": split,
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def prediction_frame(model: str, train_dataset: str, eval_dataset: str, frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": frame["image_id"].to_numpy(),
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "actual_kg": frame["weight_kg"].to_numpy(dtype=float),
            "predicted_kg": pred,
            "residual_kg": frame["weight_kg"].to_numpy(dtype=float) - pred,
        }
    )


def write_metrics_markdown(metrics: pd.DataFrame, path: Path) -> None:
    lines = ["# Reproduction Metrics", ""]
    if metrics.empty:
        lines.append("No metrics were produced.")
    else:
        lines.append(markdown_table(metrics))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_plots(predictions: pd.DataFrame, config: dict[str, Any]) -> None:
    fig_dir = figures_dir(config)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (model, train_dataset, eval_dataset), part in predictions.groupby(["model", "train_dataset", "eval_dataset"]):
        safe = f"{eval_dataset.lower()}_{model.lower()}_{train_dataset.lower()}".replace("none", "baseline")
        plt.figure(figsize=(5, 5))
        plt.scatter(part["actual_kg"], part["predicted_kg"], s=20, color="#315c72")
        low = min(part["actual_kg"].min(), part["predicted_kg"].min())
        high = max(part["actual_kg"].max(), part["predicted_kg"].max())
        plt.plot([low, high], [low, high], color="black", linewidth=1)
        plt.xlabel("Actual weight (kg)")
        plt.ylabel("Predicted weight (kg)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"pred_vs_actual_{safe}.png")
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.axhline(0, color="black", linewidth=1)
        plt.scatter(part["actual_kg"], part["residual_kg"], s=20, color="#7a4b2c")
        plt.xlabel("Actual weight (kg)")
        plt.ylabel("Residual (kg)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"residuals_{safe}.png")
        plt.close()


def write_discrepancy_report(metrics: pd.DataFrame, gaps: pd.DataFrame, config: dict[str, Any]) -> None:
    lines = [
        "# Reproduction Discrepancies",
        "",
        "This repository targets directional reproduction, not exact numeric parity.",
        "",
        "Implemented plan deviations from the paper:",
        "- PVTv2-B2 segmentation path replaces CAFE-Net.",
        "- Learned modeling is limited to LASSO.",
        "- Evaluation reports MAE and MAPE only.",
        "- Smoke fallbacks for segmentation, keypoints, and depth are labeled partial reproduction.",
        "",
        "Paper reference values extracted from the root PDF:",
        "- B3 count 2,588, weight range 36-621 kg.",
        "- B4 count 1,934, weight range 69-307 kg.",
        "- HGM B3: 40.69 kg MAE / 24.63% MAPE.",
        "- HGM B4: 53.65 kg MAE / 31.36% MAPE.",
        "- B3-trained LASSO on B3 test: 25.45 kg MAE / 16.81% MAPE.",
        "- B3-trained LASSO on B4 test: 24.56 kg MAE / 14.23% MAPE.",
        "",
        "Assumptions and known mismatch sources:",
        "- HGM uses the paper formula in inches converted to kilograms; sticker real-world radius comes from config.",
        "- If PVTv2-B2, MMPose, or Depth Pro are unavailable, outputs are partial and not paper-equivalent.",
        "- Point-cloud hull solidity is implemented as a deterministic projected-point proxy because the paper does not specify the exact 3D solidity formula.",
        "- Feature values depend on dataset version, split seed, sticker detection, mask source, keypoint provider, depth provider, point-cloud cleanup, and alpha grid.",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were available for comparison.")
    else:
        lines.extend(["Produced metrics:", "", markdown_table(metrics), ""])
    if gaps.empty:
        lines.append("No paper target gaps were available for the produced metric rows.")
    else:
        lines.extend(["Paper target gaps:", "", markdown_table(gaps), ""])
    path = reports_dir(config) / "reproduction_discrepancies.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
