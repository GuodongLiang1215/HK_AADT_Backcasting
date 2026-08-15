"""Step 15: re-examine the Step 9 and Step 10 evidence with corrected diagnostics.

Four things are re-tested, each because the existing diagnostic answers a
different question from the one the pilot needs answered.

1. Honest baseline. Beating a single training median by 36--39% is not evidence
   that the covariates carry fine-grained spatial signal, because a two-variable
   road-hierarchy lookup table beats the same median almost as well. The
   nonlinear model is scored against that lookup table as well as against the
   median and the spatial KNN.

2. Calibration in both directions. Binning by the observed value conditions on
   the outcome, so regression to the mean guarantees an apparent Q1
   overprediction and Q4 underprediction even for a well-calibrated model. The
   same out-of-fold predictions are therefore also binned by the predicted
   value, and calibration-in-the-large is reported for the first time. The
   global mean bias is not assumed to be inherited uniformly by every area.

3. Bootstrap uncertainty on the predicted change. The three year-specific models
   share identical, time-invariant predictors, so the year-to-year difference in
   network predictions may be no larger than the refit noise. A station-level
   bootstrap produces both a real change band and a same-year placebo band.

4. Cross-year transfer. If a model trained on one year's labels predicts another
   year's observations equally well, the year dimension carries no independent
   information and cannot support a segment-level change map.

The Step 10 official feature bundle is re-scored under the corrected rules.
No hyperparameter is retuned and the Step 8 folds are reused unchanged.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
EXTENDED_TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_official_extended.csv"
NETWORK_PATH = PROCESSED_DIR / "atc_network_segment_features.csv"

OOF_PATH = PROCESSED_DIR / "atc_step15_oof_predictions.csv"
BASELINE_PATH = TABLE_DIR / "step15_honest_baseline_comparison.csv"
CALIBRATION_PATH = TABLE_DIR / "step15_calibration_two_ways.csv"
FEATURE_PATH = TABLE_DIR / "step15_feature_contribution.csv"
PLACEBO_PATH = TABLE_DIR / "step15_change_placebo.csv"
TRANSFER_PATH = TABLE_DIR / "step15_cross_year_transfer.csv"
RESCORE_PATH = TABLE_DIR / "step15_step10_rescored.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step15_model_evidence_decision_audit.csv"

BASELINE_FIGURE_PATH = FIGURE_DIR / "step15_honest_baseline_comparison.png"
CALIBRATION_FIGURE_PATH = FIGURE_DIR / "step15_calibration_two_ways.png"
PLACEBO_FIGURE_PATH = FIGURE_DIR / "step15_change_placebo.png"

YEARS = (2011, 2016, 2021)
FOLDS = (1, 2, 3, 4, 5)
KNN_NEIGHBORS = 10
HIERARCHY_BINS = 5
BOOTSTRAP_DRAWS = 30
BOOTSTRAP_SEED = 20260815

# Identical to Step 9 and Step 10. Nothing is retuned here.
BASE_FEATURES = [
    "centroid_longitude",
    "centroid_latitude",
    "computed_length_m",
    "elevation",
    "travel_direction",
    "route_number_present",
    "endpoint_degree_mean",
    "street_code_segment_count",
]
# Mirrors EXTENDED_FEATURES in src/10_test_official_feature_extension.py.
EXTENDED_FEATURES = [
    "speed_limit_min_kmh",
    "speed_limit_max_kmh",
    "speed_limit_spread_kmh",
    "speed_limit_override_present",
    "official_intersection_count",
    "signalized_intersection_count",
    "intersection_max_road_count",
    "roundabout_count",
    "roundabout_max_arm_count",
    "bus_only_lane_count",
    "vehicle_restriction_count",
    "permit_count",
    "zebra_crossing_count",
    "yellow_box_count",
    "toll_feature_count",
    "cul_de_sac_count",
    "street_code_gazetted_named",
    "street_code_ungazetted_named",
    "street_code_unnamed",
    "name_expressway",
    "name_highway",
    "name_corridor",
    "name_bypass",
    "name_tunnel",
    "name_bridge",
    "name_flyover",
    *(f"strategic_route_{number}" for number in range(1, 11)),
]

BASELINE_ORDER = [
    "training_median",
    "spatial_knn_k10",
    "road_hierarchy_median",
    "hist_gradient_boosting",
]
BASELINE_LABELS = {
    "training_median": "Training median",
    "spatial_knn_k10": "Spatial KNN (k=10)",
    "road_hierarchy_median": "Road-hierarchy median lookup",
    "hist_gradient_boosting": "HistGradientBoosting",
}
BASELINE_COLORS = {
    "training_median": "#B0B7BC",
    "spatial_knn_k10": "#7F8C8D",
    "road_hierarchy_median": "#2E86AB",
    "hist_gradient_boosting": "#D35400",
}
QUARTILES = ["Q1_low", "Q2", "Q3", "Q4_high"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def fixed_step9_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def model_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame[features].copy()
    if "route_number_present" in values:
        values["route_number_present"] = values["route_number_present"].astype(int)
    return values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def coordinates_radians(frame: pd.DataFrame) -> np.ndarray:
    return np.radians(
        frame[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = (TRAINING_PATH, EXTENDED_TRAINING_PATH, NETWORK_PATH)
    missing = [path.relative_to(PROJECT_ROOT) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}. Complete Steps 8 to 11 first.")
    training = pd.read_csv(TRAINING_PATH)
    extended = pd.read_csv(EXTENDED_TRAINING_PATH)
    network = pd.read_csv(
        NETWORK_PATH,
        usecols=["route_id", "computed_length_m", "route_number_present", *BASE_FEATURES],
        dtype={"route_id": str},
    )
    if set(training["station_id"]) != set(extended["station_id"]):
        raise ValueError("Step 9 and Step 10 training tables cover different stations.")
    return training, extended, network


def road_hierarchy_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
) -> np.ndarray:
    """Median AADT inside strategic-route x corridor-extent cells.

    This is the baseline a reviewer will ask for: two road-hierarchy proxies and
    a lookup table, with the bin edges taken from the training folds only.
    """
    extent = train_frame["street_code_segment_count"].to_numpy(dtype=float)
    edges = np.unique(np.quantile(extent, np.linspace(0, 1, HIERARCHY_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    train_bins = pd.cut(extent, edges, labels=False)
    test_bins = pd.cut(
        test_frame["street_code_segment_count"].to_numpy(dtype=float), edges, labels=False
    )
    lookup = (
        pd.DataFrame(
            {
                "route_number_present": train_frame["route_number_present"].to_numpy(),
                "bin": train_bins,
                "y": y_train,
            }
        )
        .groupby(["route_number_present", "bin"], observed=True)["y"]
        .median()
    )
    fallback = float(np.median(y_train))
    return np.array(
        [
            float(lookup.get((route_number, bin_index), fallback))
            for route_number, bin_index in zip(
                test_frame["route_number_present"].to_numpy(), test_bins
            )
        ]
    )


def run_baseline_holdouts(
    training: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for year in YEARS:
        target = f"aadt_{year}"
        for fold in FOLDS:
            train_frame = training[training["spatial_fold"] != fold]
            test_frame = training[training["spatial_fold"] == fold]
            y_train = train_frame[target].to_numpy(dtype=float)
            y_test = test_frame[target].to_numpy(dtype=float)

            knn = KNeighborsRegressor(
                n_neighbors=KNN_NEIGHBORS,
                weights="distance",
                algorithm="ball_tree",
                metric="haversine",
            ).fit(coordinates_radians(train_frame), y_train)
            nonlinear = fixed_step9_model().fit(
                model_matrix(train_frame, BASE_FEATURES), y_train
            )

            predictions = {
                "training_median": np.full(len(test_frame), np.median(y_train)),
                "spatial_knn_k10": knn.predict(coordinates_radians(test_frame)),
                "road_hierarchy_median": road_hierarchy_predictions(
                    train_frame, test_frame, y_train
                ),
                "hist_gradient_boosting": nonlinear.predict(
                    model_matrix(test_frame, BASE_FEATURES)
                ),
            }
            for model, predicted in predictions.items():
                fold_rows.append(
                    {
                        "year": year,
                        "spatial_fold": fold,
                        "model": model,
                        "n": len(y_test),
                        "mae": round(mean_absolute_error(y_test, predicted), 4),
                    }
                )
                for position, station_id in enumerate(test_frame["station_id"].to_numpy()):
                    prediction_rows.append(
                        {
                            "station_id": int(station_id),
                            "year": year,
                            "spatial_fold": fold,
                            "model": model,
                            "observed_aadt": round(float(y_test[position]), 4),
                            "predicted_aadt": round(float(predicted[position]), 4),
                        }
                    )
    return prediction_rows, fold_rows


def build_baseline_summary(
    prediction_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    summary: list[dict[str, object]] = []

    for year in YEARS:
        year_predictions = predictions[predictions["year"] == year]
        median_mae = mean_absolute_error(
            *(
                year_predictions[year_predictions["model"] == "training_median"][column]
                for column in ("observed_aadt", "predicted_aadt")
            )
        )
        hierarchy_mae = mean_absolute_error(
            *(
                year_predictions[year_predictions["model"] == "road_hierarchy_median"][column]
                for column in ("observed_aadt", "predicted_aadt")
            )
        )
        for model in BASELINE_ORDER:
            pool = year_predictions[year_predictions["model"] == model]
            observed = pool["observed_aadt"].to_numpy(dtype=float)
            predicted = pool["predicted_aadt"].to_numpy(dtype=float)
            mae = mean_absolute_error(observed, predicted)
            folds = fold_metrics[
                (fold_metrics["year"] == year) & (fold_metrics["model"] == model)
            ]["mae"]
            summary.append(
                {
                    "year": year,
                    "model": model,
                    "n_oof": len(pool),
                    "pooled_mae": round(mae, 4),
                    "pooled_rmse": round(
                        math.sqrt(mean_squared_error(observed, predicted)), 4
                    ),
                    "pooled_r2": round(r2_score(observed, predicted), 6),
                    "mean_fold_mae": round(float(folds.mean()), 4),
                    "sd_fold_mae": round(float(folds.std(ddof=1)), 4),
                    "gain_vs_training_median_pct": round(
                        100 * (median_mae - mae) / median_mae, 3
                    ),
                    "gain_vs_road_hierarchy_median_pct": round(
                        100 * (hierarchy_mae - mae) / hierarchy_mae, 3
                    ),
                    "aggregate_mean_bias_pct": round(
                        100 * (predicted.mean() - observed.mean()) / observed.mean(), 3
                    ),
                }
            )
    return summary


def build_calibration_two_ways(
    prediction_rows: list[dict[str, object]],
    model: str = "hist_gradient_boosting",
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    predictions = predictions[predictions["model"] == model]
    rows: list[dict[str, object]] = []

    for year in YEARS:
        frame = predictions[predictions["year"] == year].copy()
        for binning, column in (
            ("binned_by_observed", "observed_aadt"),
            ("binned_by_predicted", "predicted_aadt"),
        ):
            frame["bin"] = pd.qcut(frame[column], 4, labels=QUARTILES)
            for quartile in QUARTILES:
                group = frame[frame["bin"] == quartile]
                observed_mean = float(group["observed_aadt"].mean())
                predicted_mean = float(group["predicted_aadt"].mean())
                rows.append(
                    {
                        "year": year,
                        "model": model,
                        "binning": binning,
                        "quartile": quartile,
                        "n": len(group),
                        "observed_mean": round(observed_mean, 4),
                        "predicted_mean": round(predicted_mean, 4),
                        "mean_bias_pct": round(
                            100 * (predicted_mean - observed_mean) / observed_mean, 3
                        ),
                        "interpretation": (
                            "conditions_on_the_outcome_so_regression_to_the_mean_is_guaranteed"
                            if binning == "binned_by_observed"
                            else "reliability_direction_usable_for_deployment"
                        ),
                    }
                )
        observed = frame["observed_aadt"].to_numpy(dtype=float)
        predicted = frame["predicted_aadt"].to_numpy(dtype=float)
        rows.append(
            {
                "year": year,
                "model": model,
                "binning": "whole_sample",
                "quartile": "all",
                "n": len(frame),
                "observed_mean": round(float(observed.mean()), 4),
                "predicted_mean": round(float(predicted.mean()), 4),
                "mean_bias_pct": round(
                    100 * (predicted.mean() - observed.mean()) / observed.mean(), 3
                ),
                "interpretation": (
                    "global_calibration_in_the_large_local_bias_may_vary_"
                    f"slope_observed_on_predicted_{np.polyfit(predicted, observed, 1)[0]:.3f}"
                ),
            }
        )
    return rows


def build_feature_contribution(training: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        target = training[f"aadt_{year}"].to_numpy(dtype=float)
        base = np.zeros(len(training))
        for fold in FOLDS:
            train_mask = (training["spatial_fold"] != fold).to_numpy()
            model = fixed_step9_model().fit(
                model_matrix(training[train_mask], BASE_FEATURES), target[train_mask]
            )
            base[~train_mask] = model.predict(
                model_matrix(training[~train_mask], BASE_FEATURES)
            )
        base_mae = mean_absolute_error(target, base)
        for dropped in BASE_FEATURES:
            features = [name for name in BASE_FEATURES if name != dropped]
            predicted = np.zeros(len(training))
            for fold in FOLDS:
                train_mask = (training["spatial_fold"] != fold).to_numpy()
                model = fixed_step9_model().fit(
                    model_matrix(training[train_mask], features), target[train_mask]
                )
                predicted[~train_mask] = model.predict(
                    model_matrix(training[~train_mask], features)
                )
            dropped_mae = mean_absolute_error(target, predicted)
            rows.append(
                {
                    "year": year,
                    "dropped_feature": dropped,
                    "full_model_mae": round(base_mae, 4),
                    "mae_without_feature": round(dropped_mae, 4),
                    "mae_increase_pct": round(100 * (dropped_mae - base_mae) / base_mae, 3),
                    "decision": "large_increase_means_the_model_depends_on_this_feature",
                }
            )
    return rows


def build_change_placebo(
    training: pd.DataFrame,
    network: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """Bootstrap the predicted change, and the same-year null change."""
    network_matrix = model_matrix(network, BASE_FEATURES)
    training_matrix = model_matrix(training, BASE_FEATURES)
    targets = {year: training[f"aadt_{year}"].to_numpy(dtype=float) for year in YEARS}

    point_estimates = {}
    for year in YEARS:
        model = fixed_step9_model().fit(training_matrix, targets[year])
        point_estimates[year] = model.predict(network_matrix)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    real_draws = {(2011, 2016): [], (2016, 2021): []}
    placebo_draws: list[np.ndarray] = []
    print(f"Running {BOOTSTRAP_DRAWS} station bootstrap refits (this is the slow step)...")
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(training), len(training))
        fitted = {}
        for year in YEARS:
            fitted[year] = fixed_step9_model().fit(
                training_matrix[index], targets[year][index]
            ).predict(network_matrix)
        for start, end in real_draws:
            real_draws[(start, end)].append(fitted[end] - fitted[start])
        # Placebo: the same 2016 labels, a second independent resample.
        placebo_index = rng.integers(0, len(training), len(training))
        placebo = fixed_step9_model().fit(
            training_matrix[placebo_index], targets[2016][placebo_index]
        ).predict(network_matrix)
        placebo_draws.append(placebo - fitted[2016])

    placebo_stack = np.vstack(placebo_draws)
    placebo_sd = placebo_stack.std(axis=0, ddof=1)
    # A same-year null sample on the same scale as a real change, for plotting.
    placebo_sample = rng.choice(placebo_stack.ravel(), size=len(placebo_sd), replace=False)

    rows: list[dict[str, object]] = []
    stored: dict[str, np.ndarray] = {
        "placebo_sd": placebo_sd,
        "placebo_sample": placebo_sample,
    }
    for (start, end), draws in real_draws.items():
        stack = np.vstack(draws)
        change_sd = stack.std(axis=0, ddof=1)
        change = point_estimates[end] - point_estimates[start]
        distinguishable = np.abs(change) > 2 * change_sd
        stored[f"change_{start}_{end}"] = change
        stored[f"change_sd_{start}_{end}"] = change_sd
        rows.append(
            {
                "period": f"{start}-{end}",
                "network_segment_count": len(change),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mean_predicted_change_aadt": round(float(change.mean()), 3),
                "mean_absolute_predicted_change_aadt": round(
                    float(np.abs(change).mean()), 3
                ),
                "sd_predicted_change_aadt": round(float(change.std()), 3),
                "mean_bootstrap_sd_of_change_aadt": round(float(change_sd.mean()), 3),
                "mean_same_year_placebo_sd_aadt": round(float(placebo_sd.mean()), 3),
                "signal_to_refit_noise_ratio": round(
                    float(np.abs(change).mean() / change_sd.mean()), 4
                ),
                "segments_distinguishable_from_zero_pct": round(
                    100 * float(distinguishable.mean()), 2
                ),
                "uncertainty_status": (
                    "diagnostic_30_draw_refit_stability_check_not_a_final_interval_estimate"
                ),
                "decision": (
                    "segment_level_change_map_is_reportable"
                    if distinguishable.mean() >= 0.5
                    else "segment_level_change_map_is_not_reportable"
                ),
            }
        )
    return rows, stored


def build_cross_year_transfer(training: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    out_of_fold: dict[int, np.ndarray] = {}
    for source_year in YEARS:
        source = training[f"aadt_{source_year}"].to_numpy(dtype=float)
        predicted = np.zeros(len(training))
        for fold in FOLDS:
            train_mask = (training["spatial_fold"] != fold).to_numpy()
            model = fixed_step9_model().fit(
                model_matrix(training[train_mask], BASE_FEATURES), source[train_mask]
            )
            predicted[~train_mask] = model.predict(
                model_matrix(training[~train_mask], BASE_FEATURES)
            )
        out_of_fold[source_year] = predicted

    for target_year in YEARS:
        observed = training[f"aadt_{target_year}"].to_numpy(dtype=float)
        own_mae = mean_absolute_error(observed, out_of_fold[target_year])
        for source_year in YEARS:
            mae = mean_absolute_error(observed, out_of_fold[source_year])
            rows.append(
                {
                    "target_year_observations": target_year,
                    "model_trained_on_year": source_year,
                    "pooled_oof_mae": round(mae, 4),
                    "penalty_vs_own_year_model_pct": round(
                        100 * (mae - own_mae) / own_mae, 3
                    ),
                    "decision": (
                        "own_year_model"
                        if source_year == target_year
                        else "a_small_penalty_means_the_year_label_carries_little_information"
                    ),
                }
            )
    return rows


def rescore_step10(extended: pd.DataFrame) -> list[dict[str, object]]:
    """Re-run the Step 10 ablation and judge it on metrics that are not artefacts."""
    rows: list[dict[str, object]] = []
    feature_sets = {
        "base_step9_features": BASE_FEATURES,
        "official_feature_extension": [*BASE_FEATURES, *EXTENDED_FEATURES],
    }
    for year in YEARS:
        observed = extended[f"aadt_{year}"].to_numpy(dtype=float)
        scores: dict[str, dict[str, float]] = {}
        for name, features in feature_sets.items():
            predicted = np.zeros(len(extended))
            for fold in FOLDS:
                train_mask = (extended["spatial_fold"] != fold).to_numpy()
                model = fixed_step9_model().fit(
                    model_matrix(extended[train_mask], features), observed[train_mask]
                )
                predicted[~train_mask] = model.predict(
                    model_matrix(extended[~train_mask], features)
                )
            bins = pd.qcut(predicted, 4, labels=QUARTILES)
            top = bins == "Q4_high"
            scores[name] = {
                "mae": mean_absolute_error(observed, predicted),
                "rmse": math.sqrt(mean_squared_error(observed, predicted)),
                "r2": r2_score(observed, predicted),
                "aggregate_bias_pct": 100
                * (predicted.mean() - observed.mean())
                / observed.mean(),
                "predicted_top_quartile_bias_pct": 100
                * (predicted[top].mean() - observed[top].mean())
                / observed[top].mean(),
            }
        base, extension = scores["base_step9_features"], scores["official_feature_extension"]
        rows.append(
            {
                "year": year,
                "base_mae": round(base["mae"], 4),
                "extended_mae": round(extension["mae"], 4),
                "mae_change_pct": round(100 * (extension["mae"] - base["mae"]) / base["mae"], 3),
                "base_rmse": round(base["rmse"], 4),
                "extended_rmse": round(extension["rmse"], 4),
                "rmse_change_pct": round(
                    100 * (extension["rmse"] - base["rmse"]) / base["rmse"], 3
                ),
                "base_r2": round(base["r2"], 6),
                "extended_r2": round(extension["r2"], 6),
                "base_aggregate_bias_pct": round(base["aggregate_bias_pct"], 3),
                "extended_aggregate_bias_pct": round(extension["aggregate_bias_pct"], 3),
                "base_predicted_top_quartile_bias_pct": round(
                    base["predicted_top_quartile_bias_pct"], 3
                ),
                "extended_predicted_top_quartile_bias_pct": round(
                    extension["predicted_top_quartile_bias_pct"], 3
                ),
                "mae_not_degraded_over_2pct": bool(
                    100 * (extension["mae"] - base["mae"]) / base["mae"] <= 2.0
                ),
                "rmse_improved": bool(extension["rmse"] < base["rmse"]),
                "r2_improved": bool(extension["r2"] > base["r2"]),
                "aggregate_bias_improved": bool(
                    abs(extension["aggregate_bias_pct"]) < abs(base["aggregate_bias_pct"])
                ),
            }
        )
    return rows


def step10_verdict(rescore_rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(rescore_rows)
    no_degradation = bool(frame["mae_not_degraded_over_2pct"].all())
    if no_degradation:
        return "original_rejection_superseded_bundle_not_adopted_reopen_features_for_nested_selection"
    return "official_extension_degrades_the_model_and_the_step10_rejection_stands"


def build_decision_audit(
    baseline_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    placebo_rows: list[dict[str, object]],
    transfer_rows: list[dict[str, object]],
    rescore_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline = pd.DataFrame(baseline_rows)
    nonlinear = baseline[baseline["model"] == "hist_gradient_boosting"]
    calibration = pd.DataFrame(calibration_rows)
    transfer = pd.DataFrame(transfer_rows)
    cross_year = transfer[
        transfer["target_year_observations"] != transfer["model_trained_on_year"]
    ]
    whole_sample = calibration[calibration["binning"] == "whole_sample"]
    observed_binned = calibration[
        (calibration["binning"] == "binned_by_observed") & (calibration["quartile"] == "Q4_high")
    ]
    predicted_binned = calibration[
        (calibration["binning"] == "binned_by_predicted") & (calibration["quartile"] == "Q4_high")
    ]
    reportable = sum(
        1 for row in placebo_rows if row["decision"] == "segment_level_change_map_is_reportable"
    )

    return [
        {"metric": "median_gain_vs_training_median_pct", "count": "", "value": round(float(nonlinear["gain_vs_training_median_pct"].median()), 2), "decision": "headline_number_currently_reported"},
        {"metric": "median_gain_vs_road_hierarchy_median_pct", "count": "", "value": round(float(nonlinear["gain_vs_road_hierarchy_median_pct"].median()), 2), "decision": "gain_over_an_honest_baseline_report_this_instead"},
        {"metric": "q4_bias_pct_binned_by_observed", "count": "", "value": round(float(observed_binned["mean_bias_pct"].mean()), 2), "decision": "artefact_of_conditioning_on_the_outcome"},
        {"metric": "q4_bias_pct_binned_by_predicted", "count": "", "value": round(float(predicted_binned["mean_bias_pct"].mean()), 2), "decision": "reliability_direction_no_material_tail_compression"},
        {"metric": "aggregate_mean_bias_pct", "count": "", "value": round(float(whole_sample["mean_bias_pct"].mean()), 2), "decision": "global_calibration_in_the_large_local_group_bias_must_be_checked_separately"},
        {"metric": "max_cross_year_transfer_penalty_pct", "count": "", "value": round(float(cross_year["penalty_vs_own_year_model_pct"].max()), 2), "decision": "a_penalty_under_5pct_means_the_year_dimension_is_not_identified"},
        {"metric": "periods_with_reportable_segment_change", "count": reportable, "value": f"of_{len(placebo_rows)}", "decision": "requires_over_half_the_segments_to_exceed_two_bootstrap_standard_deviations"},
        {"metric": "bootstrap_draws", "count": BOOTSTRAP_DRAWS, "value": f"seed_{BOOTSTRAP_SEED}", "decision": "diagnostic_refit_stability_check_not_final_segment_interval_estimation"},
        {"metric": "step10_rescore_verdict", "count": "", "value": step10_verdict(rescore_rows), "decision": "step10_used_an_observed_binned_q4_criterion_that_cannot_be_satisfied"},
        {"metric": "hyperparameter_retuning", "count": 0, "value": "", "decision": "frozen_step9_model_and_step8_folds_reused"},
        {"metric": "step15_decision_signal", "count": "", "value": "withdraw_segment_change_report_honest_baseline_and_reopen_individual_official_features_under_nested_selection", "decision": "the_full_step10_bundle_is_not_promoted_to_the_baseline"},
    ]


def plot_baselines(baseline_rows: list[dict[str, object]]) -> None:
    baseline = pd.DataFrame(baseline_rows)
    positions = np.arange(len(YEARS))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    for index, model in enumerate(BASELINE_ORDER):
        values = [
            float(
                baseline[(baseline["year"] == year) & (baseline["model"] == model)][
                    "pooled_mae"
                ].iloc[0]
            )
            for year in YEARS
        ]
        bars = axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=BASELINE_LABELS[model],
            color=BASELINE_COLORS[model],
        )
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7.5)
    axis.set_xticks(positions, [str(year) for year in YEARS])
    axis.set_ylabel("Pooled out-of-fold MAE (AADT vehicles/day)")
    axis.set_title("The nonlinear model against an honest road-hierarchy baseline")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASELINE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {BASELINE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_calibration(calibration_rows: list[dict[str, object]]) -> None:
    calibration = pd.DataFrame(calibration_rows)
    positions = np.arange(len(QUARTILES))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for axis, year in zip(axes, YEARS):
        for binning, colour, marker in (
            ("binned_by_observed", "#C0392B", "o"),
            ("binned_by_predicted", "#2E86AB", "s"),
        ):
            rows = calibration[
                (calibration["year"] == year) & (calibration["binning"] == binning)
            ].set_index("quartile").loc[QUARTILES]
            axis.plot(
                positions,
                rows["mean_bias_pct"],
                marker=marker,
                color=colour,
                label=binning.replace("_", " "),
            )
        axis.axhline(0, color="#202124", linewidth=1, linestyle="--")
        axis.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
        axis.set_title(str(year))
        axis.set_xlabel("Quartile")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean bias (% of observed)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Observed-value bins confound regression-to-the-mean; "
        "prediction-value bins show broad underprediction"
    )
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CALIBRATION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_placebo(stored: dict[str, np.ndarray]) -> None:
    placebo = stored["placebo_sample"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for axis, (start, end) in zip(axes, ((2011, 2016), (2016, 2021))):
        change = stored[f"change_{start}_{end}"]
        limit = float(
            max(np.quantile(np.abs(change), 0.99), np.quantile(np.abs(placebo), 0.99))
        )
        bins = np.linspace(-limit, limit, 90)
        axis.hist(
            placebo,
            bins=bins,
            range=(-limit, limit),
            color="#2E86AB",
            alpha=0.55,
            label="same-year placebo (no real change)",
        )
        axis.hist(
            change,
            bins=bins,
            range=(-limit, limit),
            color="#D35400",
            alpha=0.65,
            label="predicted change",
        )
        axis.set_xlim(-limit, limit)
        axis.axvline(0, color="#202124", linewidth=1, linestyle="--")
        axis.set_title(f"{start}-{end}")
        axis.set_xlabel("Predicted AADT change (vehicles/day)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Sampled segment-refit values")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Diagnostic comparison: predicted change and independent same-year refits"
    )
    fig.tight_layout()
    fig.savefig(PLACEBO_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLACEBO_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    training, extended, network = read_inputs()

    print("Re-running the Step 8 folds with an added road-hierarchy baseline...")
    prediction_rows, fold_rows = run_baseline_holdouts(training)
    baseline_rows = build_baseline_summary(prediction_rows, fold_rows)
    calibration_rows = build_calibration_two_ways(prediction_rows)

    print("Measuring per-feature contribution under the same folds...")
    feature_rows = build_feature_contribution(training)

    placebo_rows, stored = build_change_placebo(training, network)

    print("Running the cross-year transfer matrix...")
    transfer_rows = build_cross_year_transfer(training)

    print("Re-scoring the Step 10 official feature bundle...")
    rescore_rows = rescore_step10(extended)

    audit_rows = build_decision_audit(
        baseline_rows, calibration_rows, placebo_rows, transfer_rows, rescore_rows
    )

    write_csv(OOF_PATH, prediction_rows)
    write_csv(BASELINE_PATH, baseline_rows)
    write_csv(CALIBRATION_PATH, calibration_rows)
    write_csv(FEATURE_PATH, feature_rows)
    write_csv(PLACEBO_PATH, placebo_rows)
    write_csv(TRANSFER_PATH, transfer_rows)
    write_csv(RESCORE_PATH, rescore_rows)
    write_csv(DECISION_AUDIT_PATH, audit_rows)
    plot_baselines(baseline_rows)
    plot_calibration(calibration_rows)
    plot_placebo(stored)

    baseline = pd.DataFrame(baseline_rows)
    nonlinear = baseline[baseline["model"] == "hist_gradient_boosting"]
    print("\nStep 15 evidence re-check is complete.")
    print("\nHonest baseline comparison (pooled out-of-fold MAE):")
    for year in YEARS:
        parts = "  ".join(
            f"{BASELINE_LABELS[model]}: "
            f"{float(baseline[(baseline['year'] == year) & (baseline['model'] == model)]['pooled_mae'].iloc[0]):,.0f}"
            for model in BASELINE_ORDER
        )
        print(f"  {year}  {parts}")
    print(
        "  gain over the training median "
        f"{float(nonlinear['gain_vs_training_median_pct'].median()):.1f}% "
        "but over the road-hierarchy lookup only "
        f"{float(nonlinear['gain_vs_road_hierarchy_median_pct'].median()):.1f}%"
    )

    calibration = pd.DataFrame(calibration_rows)
    print("\nCalibration, both directions (Q4 mean bias):")
    for year in YEARS:
        by_observed = calibration[
            (calibration["year"] == year)
            & (calibration["binning"] == "binned_by_observed")
            & (calibration["quartile"] == "Q4_high")
        ]["mean_bias_pct"].iloc[0]
        by_predicted = calibration[
            (calibration["year"] == year)
            & (calibration["binning"] == "binned_by_predicted")
            & (calibration["quartile"] == "Q4_high")
        ]["mean_bias_pct"].iloc[0]
        aggregate = calibration[
            (calibration["year"] == year) & (calibration["binning"] == "whole_sample")
        ]["mean_bias_pct"].iloc[0]
        print(
            f"  {year}  binned by observed {float(by_observed):+.1f}%   "
            f"binned by predicted {float(by_predicted):+.1f}%   "
            f"aggregate bias {float(aggregate):+.1f}%"
        )

    print("\nPredicted change against bootstrap refit noise:")
    for row in placebo_rows:
        print(
            f"  {row['period']}  mean |change| {float(row['mean_absolute_predicted_change_aadt']):,.0f}  "
            f"refit sd {float(row['mean_bootstrap_sd_of_change_aadt']):,.0f}  "
            f"signal/noise {float(row['signal_to_refit_noise_ratio']):.2f}  "
            f"segments above 2 sd {float(row['segments_distinguishable_from_zero_pct']):.1f}%"
        )

    transfer = pd.DataFrame(transfer_rows)
    print("\nCross-year transfer (pooled out-of-fold MAE):")
    for target_year in YEARS:
        parts = "  ".join(
            f"trained on {int(row.model_trained_on_year)}: {float(row.pooled_oof_mae):,.0f}"
            for row in transfer[
                transfer["target_year_observations"] == target_year
            ].itertuples(index=False)
        )
        print(f"  predicting {target_year}: {parts}")

    print(f"\nStep 10 re-score verdict: {step10_verdict(rescore_rows)}")
    print("Next: python src\\16_neighbourhood_validation_and_equity_check.py")


if __name__ == "__main__":
    main()
