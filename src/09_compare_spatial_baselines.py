from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hk_aadt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
FOLD_PATH = PROCESSED_DIR / "atc_spatial_validation_folds.csv"

PREDICTION_PATH = PROCESSED_DIR / "atc_step9_oof_predictions.csv"
FOLD_METRIC_PATH = TABLE_DIR / "step9_spatial_holdout_metrics_by_fold.csv"
SUMMARY_PATH = TABLE_DIR / "step9_spatial_holdout_summary.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step9_model_decision_audit.csv"
CALIBRATION_PATH = TABLE_DIR / "step9_quartile_calibration.csv"
MAE_FIGURE_PATH = FIGURE_DIR / "step9_model_comparison_mae.png"
SCATTER_FIGURE_PATH = FIGURE_DIR / "step9_observed_vs_predicted.png"
CALIBRATION_FIGURE_PATH = FIGURE_DIR / "step9_quartile_calibration.png"

YEARS = (2011, 2016, 2021)
FOLDS = (1, 2, 3, 4, 5)
KNN_NEIGHBORS = 10

NUMERIC_MODEL_FEATURES = [
    "centroid_longitude",
    "centroid_latitude",
    "computed_length_m",
    "elevation",
    "travel_direction",
    "route_number_present",
    "endpoint_degree_mean",
    "street_code_segment_count",
]
CONSTANT_FEATURES_EXCLUDED = ["named_street"]
MODEL_ORDER = [
    "training_median",
    "spatial_knn_k10",
    "hist_gradient_boosting",
]
MODEL_LABELS = {
    "training_median": "Training median",
    "spatial_knn_k10": "Spatial KNN (k=10)",
    "hist_gradient_boosting": "HistGradientBoosting",
}
MODEL_COLORS = {
    "training_median": "#7F8C8D",
    "spatial_knn_k10": "#2E86AB",
    "hist_gradient_boosting": "#D35400",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAINING_PATH.exists() or not FOLD_PATH.exists():
        raise FileNotFoundError("Missing Step 8 outputs. Complete Step 8 first.")
    training = pd.read_csv(TRAINING_PATH)
    folds = pd.read_csv(FOLD_PATH)
    return training, folds


def support_route_split_count(folds: pd.DataFrame) -> int:
    route_folds: dict[str, set[int]] = defaultdict(set)
    for row in folds.itertuples(index=False):
        for route_id in str(row.label_support_route_ids).split(";"):
            if route_id:
                route_folds[route_id].add(int(row.spatial_fold))
    return sum(len(values) > 1 for values in route_folds.values())


def validate_inputs(training: pd.DataFrame, folds: pd.DataFrame) -> None:
    required_training = {
        "station_id",
        "spatial_fold",
        "support_group_id",
        *(f"aadt_{year}" for year in YEARS),
        *NUMERIC_MODEL_FEATURES,
    }
    missing_columns = sorted(required_training - set(training.columns))
    if missing_columns:
        raise ValueError(f"Missing training columns: {missing_columns}")
    if training["station_id"].duplicated().any():
        raise ValueError("Training station IDs are not unique.")
    if folds["station_id"].duplicated().any():
        raise ValueError("Fold station IDs are not unique.")
    if set(training["station_id"]) != set(folds["station_id"]):
        raise ValueError("Training and fold tables contain different stations.")
    if set(training["spatial_fold"].astype(int)) != set(FOLDS):
        raise ValueError("The training table does not contain folds 1 to 5.")
    if support_route_split_count(folds) != 0:
        raise ValueError("A label-support route is split across validation folds.")
    checked_columns = [
        *(f"aadt_{year}" for year in YEARS),
        *NUMERIC_MODEL_FEATURES,
    ]
    if training[checked_columns].isna().any().any():
        raise ValueError("Targets or initial model features contain missing values.")


def coordinates_radians(frame: pd.DataFrame) -> np.ndarray:
    return np.radians(
        frame[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    )


def tabular_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[NUMERIC_MODEL_FEATURES].copy()
    values["route_number_present"] = values["route_number_present"].astype(int)
    return values.to_numpy(dtype=float)


def fixed_nonlinear_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def metric_record(
    year: int,
    fold: int | str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    return {
        "year": year,
        "spatial_fold": fold,
        "model": model,
        "n": len(observed),
        "mae": round(mean_absolute_error(observed, predicted), 4),
        "rmse": round(math.sqrt(mean_squared_error(observed, predicted)), 4),
        "median_absolute_error": round(
            median_absolute_error(observed, predicted), 4
        ),
        "r2": round(r2_score(observed, predicted), 6),
    }


def run_spatial_holdouts(
    training: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []

    for year in YEARS:
        target_column = f"aadt_{year}"
        for fold in FOLDS:
            train_frame = training[training["spatial_fold"] != fold].copy()
            test_frame = training[training["spatial_fold"] == fold].copy()
            y_train = train_frame[target_column].to_numpy(dtype=float)
            y_test = test_frame[target_column].to_numpy(dtype=float)

            median_predictions = np.full(len(test_frame), np.median(y_train))

            knn = KNeighborsRegressor(
                n_neighbors=KNN_NEIGHBORS,
                weights="distance",
                algorithm="ball_tree",
                metric="haversine",
            )
            knn.fit(coordinates_radians(train_frame), y_train)
            knn_predictions = knn.predict(coordinates_radians(test_frame))

            nonlinear = fixed_nonlinear_model()
            nonlinear.fit(tabular_features(train_frame), y_train)
            nonlinear_predictions = nonlinear.predict(tabular_features(test_frame))

            predictions_by_model = {
                "training_median": median_predictions,
                "spatial_knn_k10": knn_predictions,
                "hist_gradient_boosting": nonlinear_predictions,
            }
            for model, predictions in predictions_by_model.items():
                fold_metric_rows.append(
                    metric_record(year, fold, model, y_test, predictions)
                )
                for position, (_, row) in enumerate(test_frame.iterrows()):
                    prediction_rows.append(
                        {
                            "station_id": int(row["station_id"]),
                            "year": year,
                            "spatial_fold": fold,
                            "support_group_id": row["support_group_id"],
                            "model": model,
                            "observed_aadt": round(float(y_test[position]), 4),
                            "predicted_aadt": round(float(predictions[position]), 4),
                            "error": round(
                                float(predictions[position] - y_test[position]), 4
                            ),
                            "absolute_error": round(
                                abs(float(predictions[position] - y_test[position])), 4
                            ),
                        }
                    )
    return prediction_rows, fold_metric_rows


def build_summary(
    prediction_rows: list[dict[str, object]],
    fold_metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    summary_rows: list[dict[str, object]] = []

    for year in YEARS:
        median_pool = predictions[
            (predictions["year"] == year)
            & (predictions["model"] == "training_median")
        ]
        median_mae = mean_absolute_error(
            median_pool["observed_aadt"], median_pool["predicted_aadt"]
        )
        median_rmse = math.sqrt(
            mean_squared_error(
                median_pool["observed_aadt"], median_pool["predicted_aadt"]
            )
        )
        knn_fold_mae = fold_metrics[
            (fold_metrics["year"] == year)
            & (fold_metrics["model"] == "spatial_knn_k10")
        ].set_index("spatial_fold")["mae"]

        for model in MODEL_ORDER:
            pool = predictions[
                (predictions["year"] == year)
                & (predictions["model"] == model)
            ]
            model_folds = fold_metrics[
                (fold_metrics["year"] == year)
                & (fold_metrics["model"] == model)
            ].set_index("spatial_fold")
            pooled = metric_record(
                year,
                "pooled_oof",
                model,
                pool["observed_aadt"].to_numpy(dtype=float),
                pool["predicted_aadt"].to_numpy(dtype=float),
            )
            pooled_mae = float(pooled["mae"])
            pooled_rmse = float(pooled["rmse"])

            folds_beating_knn: int | str = ""
            if model == "hist_gradient_boosting":
                folds_beating_knn = int((model_folds["mae"] < knn_fold_mae).sum())

            summary_rows.append(
                {
                    "year": year,
                    "model": model,
                    "n_oof": pooled["n"],
                    "pooled_mae": pooled["mae"],
                    "pooled_rmse": pooled["rmse"],
                    "pooled_median_absolute_error": pooled[
                        "median_absolute_error"
                    ],
                    "pooled_r2": pooled["r2"],
                    "mean_fold_mae": round(model_folds["mae"].mean(), 4),
                    "sd_fold_mae": round(model_folds["mae"].std(ddof=1), 4),
                    "mae_improvement_vs_median_pct": round(
                        100 * (median_mae - pooled_mae) / median_mae,
                        3,
                    ),
                    "rmse_improvement_vs_median_pct": round(
                        100 * (median_rmse - pooled_rmse) / median_rmse,
                        3,
                    ),
                    "folds_beating_knn_mae": folds_beating_knn,
                }
            )
    return summary_rows


def build_quartile_calibration(
    prediction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    predictions = predictions[
        predictions["model"] == "hist_gradient_boosting"
    ].copy()
    rows: list[dict[str, object]] = []
    quartile_labels = ["Q1_low", "Q2", "Q3", "Q4_high"]

    for year in YEARS:
        year_rows = predictions[predictions["year"] == year].copy()
        year_rows["observed_quartile"] = pd.qcut(
            year_rows["observed_aadt"],
            4,
            labels=quartile_labels,
        )
        for quartile in quartile_labels:
            group = year_rows[year_rows["observed_quartile"] == quartile]
            observed_mean = float(group["observed_aadt"].mean())
            predicted_mean = float(group["predicted_aadt"].mean())
            mean_error = predicted_mean - observed_mean
            rows.append(
                {
                    "year": year,
                    "observed_quartile": quartile,
                    "n": len(group),
                    "observed_min": round(float(group["observed_aadt"].min()), 4),
                    "observed_max": round(float(group["observed_aadt"].max()), 4),
                    "observed_mean": round(observed_mean, 4),
                    "predicted_mean": round(predicted_mean, 4),
                    "mean_error": round(mean_error, 4),
                    "mean_bias_pct_of_observed": round(
                        100 * mean_error / observed_mean,
                        3,
                    ),
                    "mae": round(float(group["absolute_error"].mean()), 4),
                }
            )
    return rows


def decision_signal(
    summary_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
) -> tuple[str, int, int]:
    summary = pd.DataFrame(summary_rows)
    promising_years = 0
    for year in YEARS:
        rows = summary[summary["year"] == year].set_index("model")
        nonlinear = rows.loc["hist_gradient_boosting"]
        if (
            nonlinear["pooled_mae"] < rows.loc["training_median", "pooled_mae"]
            and nonlinear["pooled_mae"] < rows.loc["spatial_knn_k10", "pooled_mae"]
            and int(nonlinear["folds_beating_knn_mae"]) >= 3
        ):
            promising_years += 1
    calibration = pd.DataFrame(calibration_rows)
    observed_bin_pattern_years = 0
    for year in YEARS:
        rows = calibration[calibration["year"] == year].set_index(
            "observed_quartile"
        )
        if (
            rows.loc["Q1_low", "mean_bias_pct_of_observed"] > 50
            and rows.loc["Q4_high", "mean_bias_pct_of_observed"] < -20
        ):
            observed_bin_pattern_years += 1

    if promising_years >= 2:
        signal = "nonlinear_gain_requires_honest_hierarchy_baseline_and_prediction_binned_reliability_check"
    else:
        signal = "no_consistent_nonlinear_gain_keep_model_simple_and_improve_features"
    return signal, promising_years, observed_bin_pattern_years


def build_decision_audit(
    training: pd.DataFrame,
    folds: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    fold_counts = Counter(training["spatial_fold"].astype(int))
    signal, promising_years, observed_bin_pattern_years = decision_signal(
        summary_rows,
        calibration_rows,
    )
    return [
        {"metric": "training_station_count", "count": len(training), "value": "", "decision": "high_confidence_links_only"},
        {"metric": "duplicate_training_station_ids", "count": int(training["station_id"].duplicated().sum()), "value": "", "decision": "must_equal_zero"},
        {"metric": "missing_target_or_feature_cells", "count": int(training[[*(f"aadt_{year}" for year in YEARS), *NUMERIC_MODEL_FEATURES]].isna().sum().sum()), "value": "", "decision": "must_equal_zero"},
        {"metric": "support_routes_split_across_folds", "count": support_route_split_count(folds), "value": "", "decision": "must_equal_zero"},
        {"metric": "minimum_fold_station_count", "count": min(fold_counts.values()), "value": "", "decision": "report_regional_imbalance"},
        {"metric": "maximum_fold_station_count", "count": max(fold_counts.values()), "value": "", "decision": "report_regional_imbalance"},
        {"metric": "knn_neighbors", "count": KNN_NEIGHBORS, "value": "", "decision": "fixed_before_outer_fold_evaluation"},
        {"metric": "outer_fold_hyperparameter_tuning", "count": 0, "value": "", "decision": "outer_holdouts_are_not_tuning_sets"},
        {"metric": "constant_training_feature_excluded", "count": len(CONSTANT_FEATURES_EXCLUDED), "value": ";".join(CONSTANT_FEATURES_EXCLUDED), "decision": "zero_training_variance"},
        {"metric": "years_with_consistent_nonlinear_gain", "count": promising_years, "value": "", "decision": "need_at_least_two_of_three"},
        {"metric": "years_with_observed_binned_regression_to_mean_pattern", "count": observed_bin_pattern_years, "value": "", "decision": "diagnostic_only_because_bins_condition_on_the_observed_outcome"},
        {"metric": "step9_decision_signal", "count": "", "value": signal, "decision": "fixed_reproducible_baseline_rule"},
        {"metric": "2021_interpretation", "count": "", "value": "calendar_year_surface", "decision": "do_not_assume_networkwide_pandemic_suppression_without_external_aggregate_validation"},
    ]


def plot_mae(summary_rows: list[dict[str, object]]) -> None:
    summary = pd.DataFrame(summary_rows)
    positions = np.arange(len(YEARS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for model_index, model in enumerate(MODEL_ORDER):
        values = [
            summary[(summary["year"] == year) & (summary["model"] == model)][
                "pooled_mae"
            ].iloc[0]
            for year in YEARS
        ]
        offset = (model_index - 1) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    axis.set_xticks(positions, [str(year) for year in YEARS])
    axis.set_ylabel("Pooled out-of-fold MAE (AADT vehicles/day)")
    axis.set_title("Hong Kong AADT reconstruction: regional spatial holdout")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {MAE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_observed_vs_predicted(prediction_rows: list[dict[str, object]]) -> None:
    predictions = pd.DataFrame(prediction_rows)
    predictions = predictions[
        predictions["model"] == "hist_gradient_boosting"
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True, sharey=True)
    limit = max(predictions["observed_aadt"].max(), predictions["predicted_aadt"].max())
    for axis, year in zip(axes, YEARS):
        rows = predictions[predictions["year"] == year]
        axis.scatter(
            rows["observed_aadt"],
            rows["predicted_aadt"],
            color=MODEL_COLORS["hist_gradient_boosting"],
            s=14,
            alpha=0.55,
            linewidths=0,
        )
        axis.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1)
        axis.set_title(str(year))
        axis.set_xlabel("Observed AADT")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("OOF predicted AADT")
    fig.suptitle("HistGradientBoosting: observed vs regional out-of-fold prediction")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCATTER_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SCATTER_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_quartile_calibration(
    calibration_rows: list[dict[str, object]],
) -> None:
    calibration = pd.DataFrame(calibration_rows)
    quartiles = ["Q1_low", "Q2", "Q3", "Q4_high"]
    positions = np.arange(len(quartiles))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for axis, year in zip(axes, YEARS):
        rows = calibration[calibration["year"] == year].set_index(
            "observed_quartile"
        ).loc[quartiles]
        axis.plot(
            positions,
            rows["observed_mean"],
            marker="o",
            label="Observed mean",
            color="#202124",
        )
        axis.plot(
            positions,
            rows["predicted_mean"],
            marker="o",
            label="OOF predicted mean",
            color=MODEL_COLORS["hist_gradient_boosting"],
        )
        axis.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
        axis.set_title(str(year))
        axis.set_xlabel("Observed-AADT quartile")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean AADT (vehicles/day)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Observed-AADT-bin residual diagnostic (not a reliability calibration)"
    )
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CALIBRATION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CALIBRATION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    training, folds = read_inputs()
    validate_inputs(training, folds)

    print("Running fixed five-fold regional comparisons for 2011, 2016, and 2021...")
    prediction_rows, fold_metric_rows = run_spatial_holdouts(training)
    summary_rows = build_summary(prediction_rows, fold_metric_rows)
    calibration_rows = build_quartile_calibration(prediction_rows)
    audit_rows = build_decision_audit(
        training,
        folds,
        summary_rows,
        calibration_rows,
    )

    write_csv(PREDICTION_PATH, prediction_rows)
    write_csv(FOLD_METRIC_PATH, fold_metric_rows)
    write_csv(SUMMARY_PATH, summary_rows)
    write_csv(CALIBRATION_PATH, calibration_rows)
    write_csv(DECISION_AUDIT_PATH, audit_rows)
    plot_mae(summary_rows)
    plot_observed_vs_predicted(prediction_rows)
    plot_quartile_calibration(calibration_rows)

    signal, promising_years, observed_bin_pattern_years = decision_signal(
        summary_rows,
        calibration_rows,
    )
    print("\nStep 9 spatial comparison is complete.")
    print(f"Out-of-fold prediction rows: {len(prediction_rows)}")
    print(f"Years with consistent nonlinear gain: {promising_years} of 3")
    print(
        "Years with the observed-binned regression-to-mean pattern: "
        f"{observed_bin_pattern_years} of 3"
    )
    print(f"Decision signal: {signal}")
    print(
        "Interpretation rule: observed-value bins are diagnostic only. Do not infer "
        "tail compression or network-wide pandemic suppression without the later "
        "prediction-binned and official aggregate checks."
    )


if __name__ == "__main__":
    main()
