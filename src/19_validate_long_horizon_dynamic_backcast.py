"""Step 19: long-horizon temporal-identification gate for AADT backcasting.

This step asks a narrower question than fitting another full-network map:
does public, time-varying information identify 2011--2016 and 2016--2021
change at held-out road locations?

The validation sample uses every pair that Step 4 classified as a high-
confidence, same-station-type, same-road-type, directly measured physical
match.  Current official coordinates and major/minor metadata are used only to
assign the frozen spatial folds and official strata; they are not claimed to be
historical geometries.

Two validation tasks are kept separate.

1. ``known_baseline_temporal`` gives each model the observed AADT in the first
   year.  This is a necessary temporal gate: if an official growth factor or a
   spatial residual model cannot improve on no change even with the true base
   level, it cannot rescue an unseen-road backcast.
2. ``unseen_location_backcast`` predicts the first-year level from the other
   spatial folds and then predicts the target-year level.  This approximates
   deployment at a never-counted location.

Official vehicle-kilometrage growth and official implied mean AADT growth are
both tested.  They are not interchangeable: VKT also changes when official
road-network length changes.  The decision gate is based on the two adjacent
five-year transitions; 2011--2021 is retained as a sensitivity analysis.
"""
from __future__ import annotations

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
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PAIRWISE_PATH = PROCESSED_DIR / "atc_station_crosswalk_pairwise.csv"
CURRENT_METADATA_PATH = PROCESSED_DIR / "atc_step18_all_station_annual_panel.csv"
OFFICIAL_BENCHMARK_PATH = TABLE_DIR / "step13_official_vkt_benchmark.csv"

PAIR_PATH = PROCESSED_DIR / "atc_step19_long_horizon_pairs.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step19_long_horizon_predictions.csv"
PAIR_AUDIT_PATH = TABLE_DIR / "step19_pair_audit.csv"
FACTOR_PATH = TABLE_DIR / "step19_official_dynamic_factors.csv"
FOLD_METRIC_PATH = TABLE_DIR / "step19_metrics_by_fold.csv"
METRIC_PATH = TABLE_DIR / "step19_metrics_by_transition.csv"
DECISION_PATH = TABLE_DIR / "step19_decision_audit.csv"

COMPARISON_FIGURE_PATH = FIGURE_DIR / "step19_dynamic_model_comparison.png"
IDENTIFICATION_FIGURE_PATH = FIGURE_DIR / "step19_change_identification.png"

PRIMARY_TRANSITIONS = ((2011, 2016), (2016, 2021))
SENSITIVITY_TRANSITIONS = ((2011, 2021),)
ALL_TRANSITIONS = PRIMARY_TRANSITIONS + SENSITIVITY_TRANSITIONS
FOLDS = (1, 2, 3, 4, 5)
RANDOM_SEED = 20260815
BOOTSTRAP_ITERATIONS = 1000

REGION_TO_PREFIX = {
    "Hong Kong Island": "hong_kong_island",
    "Kowloon": "kowloon",
    "New Territories": "new_territories",
}

CONDITIONAL_MODELS = (
    "zero_change",
    "territory_total_vkt_factor",
    "territory_total_intensity_factor",
    "territory_network_intensity_factor",
    "region_network_vkt_factor",
    "region_network_intensity_factor",
    "fold_stratum_median_change",
    "official_plus_residual_hgb",
)

UNSEEN_MODELS = (
    "static_hierarchy_same_level",
    "independent_hierarchy_levels",
    "static_hgb_same_level",
    "independent_hgb_levels",
    "official_intensity_scaled_hgb",
    "official_plus_residual_scaled_hgb",
)

MODEL_LABELS = {
    "zero_change": "No change",
    "territory_total_vkt_factor": "Territory VKT factor",
    "territory_total_intensity_factor": "Territory intensity factor",
    "territory_network_intensity_factor": "Territory network intensity",
    "region_network_vkt_factor": "Region x network VKT",
    "region_network_intensity_factor": "Region x network intensity",
    "fold_stratum_median_change": "Fold-trained stratum median",
    "official_plus_residual_hgb": "Official intensity + residual HGB",
    "static_hierarchy_same_level": "Static hierarchy level",
    "independent_hierarchy_levels": "Independent hierarchy levels",
    "static_hgb_same_level": "Static HGB level",
    "independent_hgb_levels": "Independent HGB levels",
    "official_intensity_scaled_hgb": "Official intensity-scaled HGB",
    "official_plus_residual_scaled_hgb": "Official + residual HGB",
}

MODEL_COLORS = {
    "zero_change": "#1F1F1F",
    "territory_total_vkt_factor": "#B0B7BC",
    "territory_total_intensity_factor": "#7F8C8D",
    "territory_network_intensity_factor": "#4C78A8",
    "region_network_vkt_factor": "#72B7B2",
    "region_network_intensity_factor": "#2E86AB",
    "fold_stratum_median_change": "#F58518",
    "official_plus_residual_hgb": "#D35400",
    "static_hierarchy_same_level": "#A0A0A0",
    "independent_hierarchy_levels": "#4C78A8",
    "static_hgb_same_level": "#8C8C8C",
    "independent_hgb_levels": "#54A24B",
    "official_intensity_scaled_hgb": "#E45756",
    "official_plus_residual_scaled_hgb": "#B279A2",
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty result: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def read_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().eq("true")


def fixed_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = pd.DataFrame(index=frame.index)
    values["longitude"] = pd.to_numeric(frame["longitude"], errors="raise")
    values["latitude"] = pd.to_numeric(frame["latitude"], errors="raise")
    values["major_network"] = frame["road_network"].eq("MAJOR").astype(float)
    for region in REGION_TO_PREFIX:
        values[f"region_{region}"] = frame["region"].eq(region).astype(float)
    for road_type in sorted(frame.attrs["road_type_categories"]):
        values[f"road_type_{road_type}"] = frame["road_type_from"].eq(road_type).astype(float)
    for station_type in sorted(frame.attrs["station_type_categories"]):
        values[f"station_type_{station_type}"] = (
            frame["station_type_from"].eq(station_type).astype(float)
        )
    return values.to_numpy(dtype=float)


def attach_feature_categories(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    frame.attrs["road_type_categories"] = tuple(
        sorted(source["road_type_from"].dropna().astype(str).unique())
    )
    frame.attrs["station_type_categories"] = tuple(
        sorted(source["station_type_from"].dropna().astype(str).unique())
    )
    return frame


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (PAIRWISE_PATH, CURRENT_METADATA_PATH, OFFICIAL_BENCHMARK_PATH)
    missing = [path.relative_to(PROJECT_ROOT) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing inputs: {missing}. Complete Steps 4, 13, and 18 first."
        )

    raw = pd.read_csv(PAIRWISE_PATH)
    raw["recommended_observed_pair"] = read_bool(raw["recommended_observed_pair"])
    raw["both_labels_measured"] = read_bool(raw["both_labels_measured"])
    raw["station_type_same"] = read_bool(raw["station_type_same"])
    raw["road_type_same"] = read_bool(raw["road_type_same"])

    metadata_source = pd.read_csv(CURRENT_METADATA_PATH)
    metadata = (
        metadata_source.sort_values(["station_id", "year"])
        .drop_duplicates("station_id", keep="last")
        [[
            "station_id",
            "region",
            "road_network",
            "longitude",
            "latitude",
            "spatial_fold",
            "geometry_reference",
            "historical_geometry_status",
        ]]
    )

    candidate = raw[
        raw.apply(
            lambda row: (int(row["year_from"]), int(row["year_to"]))
            in ALL_TRANSITIONS,
            axis=1,
        )
    ].copy()
    candidate = candidate.merge(metadata, on="station_id", how="left", validate="many_to_one")
    candidate["transition"] = (
        candidate["year_from"].astype(int).astype(str)
        + "-"
        + candidate["year_to"].astype(int).astype(str)
    )
    candidate["pair_selection_status"] = np.select(
        [
            ~candidate["recommended_observed_pair"],
            candidate["longitude"].isna() | candidate["latitude"].isna(),
            candidate["spatial_fold"].isna(),
            ~candidate["region"].isin(REGION_TO_PREFIX),
            ~candidate["road_network"].isin(["MAJOR", "MINOR"]),
        ],
        [
            "excluded_not_recommended_high_confidence_pair",
            "excluded_no_current_coordinate",
            "excluded_no_frozen_spatial_fold",
            "excluded_unknown_region",
            "excluded_unknown_official_road_network",
        ],
        default="included",
    )

    audit_rows: list[dict[str, object]] = []
    for year_from, year_to in ALL_TRANSITIONS:
        transition = f"{year_from}-{year_to}"
        transition_raw = candidate[candidate["transition"] == transition]
        status_counts = transition_raw["pair_selection_status"].value_counts().to_dict()
        included = transition_raw[transition_raw["pair_selection_status"] == "included"]
        audit_rows.append(
            {
                "transition": transition,
                "analysis_role": (
                    "primary_adjacent_five_year"
                    if (year_from, year_to) in PRIMARY_TRANSITIONS
                    else "ten_year_sensitivity"
                ),
                "same_id_rows": len(transition_raw),
                "both_labels_measured": int(transition_raw["both_labels_measured"].sum()),
                "recommended_high_confidence_pairs": int(
                    transition_raw["recommended_observed_pair"].sum()
                ),
                "included_pairs": len(included),
                "excluded_not_recommended": status_counts.get(
                    "excluded_not_recommended_high_confidence_pair", 0
                ),
                "excluded_no_coordinate": status_counts.get(
                    "excluded_no_current_coordinate", 0
                ),
                "excluded_no_fold": status_counts.get(
                    "excluded_no_frozen_spatial_fold", 0
                ),
                "excluded_unknown_region": status_counts.get(
                    "excluded_unknown_region", 0
                ),
                "excluded_unknown_road_network": status_counts.get(
                    "excluded_unknown_official_road_network", 0
                ),
                "major_pairs": int(included["road_network"].eq("MAJOR").sum()),
                "minor_pairs": int(included["road_network"].eq("MINOR").sum()),
                "current_coordinate_caveat": (
                    "latest_official_coordinates_assign_spatial_fold_not_historical_geometry"
                ),
            }
        )

    included = candidate[candidate["pair_selection_status"] == "included"].copy()
    included["year_from"] = included["year_from"].astype(int)
    included["year_to"] = included["year_to"].astype(int)
    included["spatial_fold"] = included["spatial_fold"].astype(int)
    included["aadt_from"] = pd.to_numeric(included["aadt_from"], errors="raise")
    included["aadt_to"] = pd.to_numeric(included["aadt_to"], errors="raise")
    included["observed_change"] = included["aadt_to"] - included["aadt_from"]
    included["observed_log_ratio"] = np.log(included["aadt_to"] / included["aadt_from"])
    included = included.sort_values(["year_from", "year_to", "station_id"]).reset_index(drop=True)
    return included, pd.DataFrame(audit_rows)


def official_value(
    benchmark: pd.DataFrame,
    year: int,
    region: str | None,
    road_network: str | None,
    quantity: str,
) -> float:
    row = benchmark.loc[benchmark["census_year"] == year].iloc[0]
    if region is None and road_network is None:
        prefix = "territory_total"
    elif region is None and road_network is not None:
        prefix = f"territory_{road_network.casefold()}"
    elif region is not None and road_network is not None:
        prefix = f"{REGION_TO_PREFIX[region]}_{road_network.casefold()}"
    else:
        raise ValueError("Region-only official quantity is not used in Step 19.")
    suffix = (
        "daily_vehicle_km" if quantity == "vkt" else "implied_mean_aadt"
    )
    return float(row[f"{prefix}_{suffix}"])


def build_official_factors(pairs: pd.DataFrame) -> pd.DataFrame:
    benchmark = pd.read_csv(OFFICIAL_BENCHMARK_PATH)
    rows: list[dict[str, object]] = []
    for year_from, year_to in ALL_TRANSITIONS:
        for region in REGION_TO_PREFIX:
            for road_network in ("MAJOR", "MINOR"):
                territory_vkt = official_value(
                    benchmark, year_to, None, None, "vkt"
                ) / official_value(benchmark, year_from, None, None, "vkt")
                territory_intensity = official_value(
                    benchmark, year_to, None, None, "intensity"
                ) / official_value(benchmark, year_from, None, None, "intensity")
                territory_network_intensity = official_value(
                    benchmark, year_to, None, road_network, "intensity"
                ) / official_value(
                    benchmark, year_from, None, road_network, "intensity"
                )
                region_network_vkt = official_value(
                    benchmark, year_to, region, road_network, "vkt"
                ) / official_value(
                    benchmark, year_from, region, road_network, "vkt"
                )
                region_network_intensity = official_value(
                    benchmark, year_to, region, road_network, "intensity"
                ) / official_value(
                    benchmark, year_from, region, road_network, "intensity"
                )
                rows.append(
                    {
                        "year_from": year_from,
                        "year_to": year_to,
                        "transition": f"{year_from}-{year_to}",
                        "region": region,
                        "road_network": road_network,
                        "territory_total_vkt_factor": round(territory_vkt, 8),
                        "territory_total_intensity_factor": round(
                            territory_intensity, 8
                        ),
                        "territory_network_intensity_factor": round(
                            territory_network_intensity, 8
                        ),
                        "region_network_vkt_factor": round(region_network_vkt, 8),
                        "region_network_intensity_factor": round(
                            region_network_intensity, 8
                        ),
                        "vkt_minus_intensity_growth_percentage_points": round(
                            (region_network_vkt - region_network_intensity) * 100, 6
                        ),
                        "station_pairs_in_stratum": int(
                            (
                                (pairs["year_from"] == year_from)
                                & (pairs["year_to"] == year_to)
                                & (pairs["region"] == region)
                                & (pairs["road_network"] == road_network)
                            ).sum()
                        ),
                    }
                )
    factors = pd.DataFrame(rows)
    duplicate_keys = factors.duplicated(
        ["year_from", "year_to", "region", "road_network"]
    )
    if duplicate_keys.any():
        raise ValueError("Official factor table contains duplicate strata.")
    return factors


def add_official_factors(pairs: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    return pairs.merge(
        factors,
        on=["year_from", "year_to", "transition", "region", "road_network"],
        how="left",
        validate="many_to_one",
    )


def hierarchy_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    group_keys = ["region", "road_network", "road_type_from"]
    full_lookup = train.groupby(group_keys, observed=True)[target].median()
    network_type_lookup = train.groupby(
        ["road_network", "road_type_from"], observed=True
    )[target].median()
    network_lookup = train.groupby("road_network", observed=True)[target].median()
    fallback = float(train[target].median())
    predictions: list[float] = []
    for row in test.itertuples(index=False):
        value = full_lookup.get((row.region, row.road_network, row.road_type_from), np.nan)
        if pd.isna(value):
            value = network_type_lookup.get(
                (row.road_network, row.road_type_from), np.nan
            )
        if pd.isna(value):
            value = network_lookup.get(row.road_network, fallback)
        predictions.append(float(value))
    return np.asarray(predictions, dtype=float)


def stratum_median_log_change(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    target = "observed_log_ratio"
    full_lookup = train.groupby(
        ["region", "road_network", "road_type_from"], observed=True
    )[target].median()
    region_network_lookup = train.groupby(
        ["region", "road_network"], observed=True
    )[target].median()
    network_lookup = train.groupby("road_network", observed=True)[target].median()
    fallback = float(train[target].median())
    values: list[float] = []
    for row in test.itertuples(index=False):
        value = full_lookup.get((row.region, row.road_network, row.road_type_from), np.nan)
        if pd.isna(value):
            value = region_network_lookup.get((row.region, row.road_network), np.nan)
        if pd.isna(value):
            value = network_lookup.get(row.road_network, fallback)
        values.append(float(value))
    return np.asarray(values, dtype=float)


def fit_predict_hgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: np.ndarray,
) -> np.ndarray:
    train = attach_feature_categories(train, train)
    test = attach_feature_categories(test, train)
    model = fixed_hgb().fit(feature_matrix(train), target)
    return model.predict(feature_matrix(test))


def append_predictions(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    task: str,
    model: str,
    predicted_from: np.ndarray,
    predicted_to: np.ndarray,
) -> None:
    for position, source in enumerate(frame.itertuples(index=False)):
        rows.append(
            {
                "task": task,
                "model": model,
                "year_from": int(source.year_from),
                "year_to": int(source.year_to),
                "transition": source.transition,
                "analysis_role": (
                    "primary_adjacent_five_year"
                    if (int(source.year_from), int(source.year_to))
                    in PRIMARY_TRANSITIONS
                    else "ten_year_sensitivity"
                ),
                "station_id": int(source.station_id),
                "spatial_fold": int(source.spatial_fold),
                "region": source.region,
                "road_network": source.road_network,
                "road_type": source.road_type_from,
                "station_type": source.station_type_from,
                "observed_aadt_from": float(source.aadt_from),
                "observed_aadt_to": float(source.aadt_to),
                "predicted_aadt_from": float(predicted_from[position]),
                "predicted_aadt_to": float(predicted_to[position]),
                "observed_change": float(source.observed_change),
                "predicted_change": float(
                    predicted_to[position] - predicted_from[position]
                ),
                "region_network_vkt_factor": float(
                    source.region_network_vkt_factor
                ),
                "region_network_intensity_factor": float(
                    source.region_network_intensity_factor
                ),
            }
        )


def run_spatial_validation(pairs: pd.DataFrame) -> pd.DataFrame:
    prediction_rows: list[dict[str, object]] = []
    for year_from, year_to in ALL_TRANSITIONS:
        transition_frame = pairs[
            (pairs["year_from"] == year_from) & (pairs["year_to"] == year_to)
        ].copy()
        transition_frame = attach_feature_categories(transition_frame, transition_frame)
        for fold in FOLDS:
            train = transition_frame[transition_frame["spatial_fold"] != fold].copy()
            test = transition_frame[transition_frame["spatial_fold"] == fold].copy()
            if train.empty or test.empty:
                raise ValueError(
                    f"Transition {year_from}-{year_to} has an empty train/test fold {fold}."
                )
            train = attach_feature_categories(train, transition_frame)
            test = attach_feature_categories(test, transition_frame)

            observed_base = test["aadt_from"].to_numpy(dtype=float)
            zeros = np.ones(len(test), dtype=float)
            factor_columns = {
                "territory_total_vkt_factor": "territory_total_vkt_factor",
                "territory_total_intensity_factor": "territory_total_intensity_factor",
                "territory_network_intensity_factor": "territory_network_intensity_factor",
                "region_network_vkt_factor": "region_network_vkt_factor",
                "region_network_intensity_factor": "region_network_intensity_factor",
            }
            append_predictions(
                prediction_rows,
                test,
                "known_baseline_temporal",
                "zero_change",
                observed_base,
                observed_base * zeros,
            )
            for model, column in factor_columns.items():
                factor = test[column].to_numpy(dtype=float)
                append_predictions(
                    prediction_rows,
                    test,
                    "known_baseline_temporal",
                    model,
                    observed_base,
                    observed_base * factor,
                )

            median_log_change = stratum_median_log_change(train, test)
            append_predictions(
                prediction_rows,
                test,
                "known_baseline_temporal",
                "fold_stratum_median_change",
                observed_base,
                observed_base * np.exp(median_log_change),
            )

            train_official_log = np.log(
                train["region_network_intensity_factor"].to_numpy(dtype=float)
            )
            residual_target = (
                train["observed_log_ratio"].to_numpy(dtype=float)
                - train_official_log
            )
            residual_prediction = fit_predict_hgb(
                train, test, residual_target
            )
            lower, upper = np.quantile(residual_target, [0.01, 0.99])
            residual_prediction = np.clip(residual_prediction, lower, upper)
            official_factor = test[
                "region_network_intensity_factor"
            ].to_numpy(dtype=float)
            append_predictions(
                prediction_rows,
                test,
                "known_baseline_temporal",
                "official_plus_residual_hgb",
                observed_base,
                observed_base * official_factor * np.exp(residual_prediction),
            )

            hierarchy_from = hierarchy_predict(train, test, "aadt_from")
            hierarchy_to = hierarchy_predict(train, test, "aadt_to")
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "static_hierarchy_same_level",
                hierarchy_from,
                hierarchy_from,
            )
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "independent_hierarchy_levels",
                hierarchy_from,
                hierarchy_to,
            )

            hgb_from = fit_predict_hgb(
                train, test, train["aadt_from"].to_numpy(dtype=float)
            )
            hgb_to = fit_predict_hgb(
                train, test, train["aadt_to"].to_numpy(dtype=float)
            )
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "static_hgb_same_level",
                hgb_from,
                hgb_from,
            )
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "independent_hgb_levels",
                hgb_from,
                hgb_to,
            )
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "official_intensity_scaled_hgb",
                hgb_from,
                hgb_from * official_factor,
            )
            append_predictions(
                prediction_rows,
                test,
                "unseen_location_backcast",
                "official_plus_residual_scaled_hgb",
                hgb_from,
                hgb_from * official_factor * np.exp(residual_prediction),
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions["absolute_change_error"] = (
        predictions["predicted_change"] - predictions["observed_change"]
    ).abs()
    predictions["zero_change_absolute_error"] = predictions["observed_change"].abs()
    predictions["loss_difference_vs_zero"] = (
        predictions["absolute_change_error"]
        - predictions["zero_change_absolute_error"]
    )
    return predictions.sort_values(
        ["task", "year_from", "year_to", "model", "station_id"]
    ).reset_index(drop=True)


def loss_interval(
    frame: pd.DataFrame,
    seed: int,
) -> tuple[float, float]:
    station_groups = [
        group["loss_difference_vs_zero"].to_numpy(dtype=float)
        for _, group in frame.groupby("station_id")
    ]
    if len(station_groups) < 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled = generator.integers(0, len(station_groups), size=len(station_groups))
        estimates[iteration] = float(
            np.concatenate([station_groups[index] for index in sampled]).mean()
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def metric_row(
    frame: pd.DataFrame,
    task: str,
    model: str,
    scope: str,
    fold: str | int,
    seed: int,
) -> dict[str, object]:
    observed_from = frame["observed_aadt_from"].to_numpy(dtype=float)
    observed_to = frame["observed_aadt_to"].to_numpy(dtype=float)
    observed_change = frame["observed_change"].to_numpy(dtype=float)
    predicted_from = frame["predicted_aadt_from"].to_numpy(dtype=float)
    predicted_to = frame["predicted_aadt_to"].to_numpy(dtype=float)
    predicted_change = frame["predicted_change"].to_numpy(dtype=float)
    change_mae = mean_absolute_error(observed_change, predicted_change)
    zero_mae = float(np.mean(np.abs(observed_change)))
    low, high = (
        loss_interval(frame, seed)
        if fold == "pooled" and model not in {
            "zero_change",
            "static_hierarchy_same_level",
            "static_hgb_same_level",
        }
        else (float("nan"), float("nan"))
    )
    nonzero = observed_change != 0
    return {
        "task": task,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "transition_scope": scope,
        "spatial_fold": fold,
        "n": len(frame),
        "observed_mean_aadt_from": round(float(np.mean(observed_from)), 4),
        "observed_mean_aadt_to": round(float(np.mean(observed_to)), 4),
        "predicted_mean_aadt_from": round(float(np.mean(predicted_from)), 4),
        "predicted_mean_aadt_to": round(float(np.mean(predicted_to)), 4),
        "observed_mean_change": round(float(np.mean(observed_change)), 4),
        "predicted_mean_change": round(float(np.mean(predicted_change)), 4),
        "aadt_from_mae": round(mean_absolute_error(observed_from, predicted_from), 4),
        "aadt_to_mae": round(mean_absolute_error(observed_to, predicted_to), 4),
        "change_mae": round(change_mae, 4),
        "zero_change_mae": round(zero_mae, 4),
        "improvement_pct_vs_zero_change": round(
            (zero_mae - change_mae) / zero_mae * 100 if zero_mae else 0.0,
            4,
        ),
        "change_rmse": round(
            math.sqrt(mean_squared_error(observed_change, predicted_change)), 4
        ),
        "change_correlation": round(
            correlation(observed_change, predicted_change), 6
        ),
        "direction_accuracy_nonzero": round(
            float(
                np.mean(
                    np.sign(observed_change[nonzero])
                    == np.sign(predicted_change[nonzero])
                )
            )
            if nonzero.any()
            else float("nan"),
            6,
        ),
        "mean_loss_difference_vs_zero": round(change_mae - zero_mae, 4),
        "bootstrap_95pct_loss_difference_low": round(low, 4),
        "bootstrap_95pct_loss_difference_high": round(high, 4),
    }


def build_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    seed_counter = 0
    scopes = [
        ("2011-2016", ["2011-2016"]),
        ("2016-2021", ["2016-2021"]),
        ("primary_adjacent_transitions", ["2011-2016", "2016-2021"]),
        ("2011-2021_sensitivity", ["2011-2021"]),
    ]
    for task in predictions["task"].drop_duplicates():
        task_frame = predictions[predictions["task"] == task]
        for model in task_frame["model"].drop_duplicates():
            model_frame = task_frame[task_frame["model"] == model]
            for scope, transitions in scopes:
                frame = model_frame[model_frame["transition"].isin(transitions)]
                if frame.empty:
                    continue
                pooled_rows.append(
                    metric_row(
                        frame,
                        task,
                        model,
                        scope,
                        "pooled",
                        RANDOM_SEED + seed_counter,
                    )
                )
                seed_counter += 1
                if scope in {"2011-2016", "2016-2021", "2011-2021_sensitivity"}:
                    for fold in FOLDS:
                        fold_frame = frame[frame["spatial_fold"] == fold]
                        if fold_frame.empty:
                            continue
                        fold_rows.append(
                            metric_row(
                                fold_frame,
                                task,
                                model,
                                scope,
                                fold,
                                RANDOM_SEED + seed_counter,
                            )
                        )
                        seed_counter += 1
    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def model_passes_transition_gate(row: pd.Series) -> bool:
    return bool(
        row["improvement_pct_vs_zero_change"] > 0
        and row["change_correlation"] > 0
        and row["bootstrap_95pct_loss_difference_high"] < 0
    )


def gate_task(metrics: pd.DataFrame, task: str) -> tuple[str, bool, str]:
    task_metrics = metrics[
        (metrics["task"] == task)
        & (metrics["transition_scope"].isin(["2011-2016", "2016-2021"]))
    ].copy()
    baseline_models = {
        "zero_change",
        "static_hierarchy_same_level",
        "static_hgb_same_level",
    }
    candidates = [
        model
        for model in task_metrics["model"].unique()
        if model not in baseline_models
    ]
    summaries: list[tuple[str, bool, float]] = []
    for model in candidates:
        rows = task_metrics[task_metrics["model"] == model]
        passes = len(rows) == 2 and all(
            model_passes_transition_gate(row) for _, row in rows.iterrows()
        )
        pooled_improvement = float(rows["improvement_pct_vs_zero_change"].mean())
        summaries.append((model, passes, pooled_improvement))
    summaries.sort(key=lambda item: item[2], reverse=True)
    passing = [item for item in summaries if item[1]]
    selected = passing[0] if passing else summaries[0]
    reason = (
        "passes_both_five_year_transitions_with_positive_correlation_and_loss_interval_below_zero"
        if selected[1]
        else "no_single_model_passes_both_five_year_transitions_under_the_frozen_gate"
    )
    return selected[0], selected[1], reason


def build_decision_audit(
    pairs: pd.DataFrame,
    factors: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    conditional_model, conditional_pass, conditional_reason = gate_task(
        metrics, "known_baseline_temporal"
    )
    unseen_model, unseen_pass, unseen_reason = gate_task(
        metrics, "unseen_location_backcast"
    )
    official_rows = metrics[
        (metrics["task"] == "known_baseline_temporal")
        & (metrics["model"] == "region_network_intensity_factor")
        & (metrics["transition_scope"].isin(["2011-2016", "2016-2021"]))
    ]
    official_pass = len(official_rows) == 2 and all(
        model_passes_transition_gate(row) for _, row in official_rows.iterrows()
    )

    if conditional_pass and unseen_pass:
        final_decision = "segment_level_long_horizon_backcast_passes_step19_gate"
    elif conditional_pass:
        final_decision = (
            "temporal_signal_requires_a_known_base_level_not_a_validated_unseen_location_backcast"
        )
    else:
        final_decision = (
            "official_aggregate_time_trend_only_no_validated_segment_level_downscaling"
        )

    primary_pair_count = int(
        pairs["transition"].isin(["2011-2016", "2016-2021"]).sum()
    )
    max_length_component = float(
        factors[
            factors["transition"].isin(["2011-2016", "2016-2021"])
        ]["vkt_minus_intensity_growth_percentage_points"].abs().max()
    )
    rows = [
        {
            "question": "did_pairwise_long_horizon_sample_expand_beyond_the_three_year_intersection",
            "evidence": (
                f"{primary_pair_count} adjacent-transition pair records pass the high-confidence "
                "pair and current-metadata gate; each transition is evaluated separately."
            ),
            "decision": "yes_use_pairwise_samples_not_only_the_679_three_year_intersection",
        },
        {
            "question": "can_official_vkt_growth_be_treated_as_existing_segment_aadt_growth",
            "evidence": (
                f"Across primary region-network strata, the absolute VKT-growth minus intensity-"
                f"growth difference reaches {max_length_component:.2f} percentage points because "
                "official road length also changes."
            ),
            "decision": "no_report_vkt_and_vkt_per_official_km_as_separate_factors",
        },
        {
            "question": "does_public_dynamic_information_identify_change_with_the_true_base_level_known",
            "evidence": f"selected_model={conditional_model}; {conditional_reason}",
            "decision": str(conditional_pass),
        },
        {
            "question": "does_the_unseen_location_backcast_pass_both_five_year_transitions",
            "evidence": f"selected_model={unseen_model}; {unseen_reason}",
            "decision": str(unseen_pass),
        },
        {
            "question": "does_the_official_region_network_intensity_factor_alone_pass_the_segment_gate",
            "evidence": (
                "The same externally published factor must improve on no change with positive "
                "change correlation and a loss-difference interval below zero in both transitions."
            ),
            "decision": str(official_pass),
        },
        {
            "question": "can_step19_validate_never_counted_local_roads",
            "evidence": (
                "Validation labels remain ATC stations and current coordinates are only a fold/"
                "stratum anchor; no independent historical local-road census is introduced."
            ),
            "decision": "no",
        },
        {
            "question": "step19_final_decision",
            "evidence": (
                "A segment claim requires both the known-base temporal gate and unseen-location "
                "gate to pass the two adjacent five-year transitions. Official aggregates remain "
                "valid external temporal constraints even when within-stratum downscaling fails."
            ),
            "decision": final_decision,
        },
    ]
    return pd.DataFrame(rows)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    tasks = ["known_baseline_temporal", "unseen_location_backcast"]
    transitions = ["2011-2016", "2016-2021"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=False)
    for row_index, task in enumerate(tasks):
        task_models = CONDITIONAL_MODELS if task == tasks[0] else UNSEEN_MODELS
        display_models = [model for model in task_models if model in set(metrics["model"])]
        positions = np.arange(len(display_models))
        width = 0.36
        for transition_index, transition in enumerate(transitions):
            subset = metrics[
                (metrics["task"] == task)
                & (metrics["transition_scope"] == transition)
            ].set_index("model")
            improvements = [
                float(subset.loc[model, "improvement_pct_vs_zero_change"])
                for model in display_models
            ]
            correlations = [
                float(subset.loc[model, "change_correlation"])
                for model in display_models
            ]
            offset = (transition_index - 0.5) * width
            axes[row_index, 0].bar(
                positions + offset,
                improvements,
                width=width,
                label=transition,
                color="#2E86AB" if transition_index == 0 else "#D35400",
                alpha=0.85,
            )
            axes[row_index, 1].bar(
                positions + offset,
                correlations,
                width=width,
                label=transition,
                color="#2E86AB" if transition_index == 0 else "#D35400",
                alpha=0.85,
            )
        axes[row_index, 0].axhline(0, color="#333333", linewidth=1)
        axes[row_index, 1].axhline(0, color="#333333", linewidth=1)
        axes[row_index, 0].set_ylabel(
            "Known base" if task == tasks[0] else "Unseen location"
        )
        axes[row_index, 0].set_xticks(positions)
        axes[row_index, 1].set_xticks(positions)
        labels = [MODEL_LABELS[model] for model in display_models]
        axes[row_index, 0].set_xticklabels(labels, rotation=35, ha="right")
        axes[row_index, 1].set_xticklabels(labels, rotation=35, ha="right")
        axes[row_index, 0].grid(axis="y", alpha=0.2)
        axes[row_index, 1].grid(axis="y", alpha=0.2)
        axes[row_index, 0].tick_params(axis="x", labelbottom=True)
        axes[row_index, 1].tick_params(axis="x", labelbottom=True)
    axes[0, 0].set_title("Change-MAE improvement over no change (%)")
    axes[0, 1].set_title("Observed-predicted change correlation")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Step 19: do public dynamic constraints identify five-year segment change?",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {COMPARISON_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_change_identification(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    task = "known_baseline_temporal"
    primary = metrics[
        (metrics["task"] == task)
        & (metrics["transition_scope"].isin(["2011-2016", "2016-2021"]))
        & (~metrics["model"].eq("zero_change"))
    ]
    pooled_by_model = primary.groupby("model", observed=True)["change_mae"].mean()
    best_model = str(pooled_by_model.idxmin())
    figure_data = predictions[
        (predictions["task"] == task)
        & (predictions["model"] == best_model)
        & (predictions["transition"].isin(["2011-2016", "2016-2021"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    limits = figure_data[["observed_change", "predicted_change"]].to_numpy(dtype=float)
    low, high = np.quantile(limits, [0.01, 0.99])
    span = max(abs(low), abs(high))
    for axis, transition in zip(axes, ["2011-2016", "2016-2021"]):
        frame = figure_data[figure_data["transition"] == transition]
        axis.scatter(
            frame["observed_change"],
            frame["predicted_change"],
            s=18,
            alpha=0.45,
            color=MODEL_COLORS.get(best_model, "#2E86AB"),
            edgecolors="none",
        )
        axis.plot([-span, span], [-span, span], linestyle="--", color="#333333")
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.axvline(0, color="#999999", linewidth=0.8)
        axis.set_xlim(-span, span)
        axis.set_ylim(-span, span)
        axis.set_title(transition)
        axis.set_xlabel("Observed five-year AADT change")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Predicted five-year AADT change")
    fig.suptitle(
        f"Best candidate (does not beat no change): {MODEL_LABELS[best_model]}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IDENTIFICATION_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {IDENTIFICATION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def print_summary(decision: pd.DataFrame, metrics: pd.DataFrame) -> None:
    final = decision.loc[
        decision["question"] == "step19_final_decision", "decision"
    ].iloc[0]
    print("\nStep 19 long-horizon temporal-identification gate is complete.")
    for transition in ("2011-2016", "2016-2021"):
        rows = metrics[
            (metrics["task"] == "known_baseline_temporal")
            & (metrics["transition_scope"] == transition)
            & (~metrics["model"].eq("zero_change"))
        ]
        best = rows.loc[rows["change_mae"].idxmin()]
        print(
            f"  {transition}: best known-base model={best['model']}; "
            f"MAE={best['change_mae']:,.0f} versus no-change "
            f"{best['zero_change_mae']:,.0f}; correlation={best['change_correlation']:.3f}."
        )
    print(f"  Decision: {final}")


def main() -> None:
    pairs, pair_audit = load_pairs()
    factors = build_official_factors(pairs)
    pairs = add_official_factors(pairs, factors)
    predictions = run_spatial_validation(pairs)
    metrics, fold_metrics = build_metrics(predictions)
    decision = build_decision_audit(pairs, factors, metrics)

    save_csv(pairs, PAIR_PATH)
    save_csv(predictions, PREDICTION_PATH)
    save_csv(pair_audit, PAIR_AUDIT_PATH)
    save_csv(factors, FACTOR_PATH)
    save_csv(fold_metrics, FOLD_METRIC_PATH)
    save_csv(metrics, METRIC_PATH)
    save_csv(decision, DECISION_PATH)
    plot_model_comparison(metrics)
    plot_change_identification(predictions, metrics)
    print_summary(decision, metrics)


if __name__ == "__main__":
    main()
