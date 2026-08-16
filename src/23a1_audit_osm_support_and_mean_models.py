"""Step 23A.1: audit OSM support, rematch motor roads, and test mean models.

This step is deliberately split into two phases.

1. ``--phase prepare`` freezes the motor-road eligibility rule, rebuilds the
   crosswalk from the already cached 2023 OSM response, writes label-support
   tables, and draws a blinded 100-segment adjudication sample.  It does not
   fit a model.
2. After the reviewer fills ``reviewer_verdict`` in the sample CSV,
   ``--phase evaluate`` checks that every record was adjudicated, reports
   stratified Wilson intervals, and runs the predeclared absolute-error,
   Poisson, and squared-error comparisons on the unchanged five folds.

The ATC functional class and station-to-road match diagnostics remain outcomes
or diagnostics only.  They never enter a deployable model.  The Poisson loss is
used as a positive conditional-mean loss; it is not a distributional claim
about AADT.  No threshold, fold, match-acceptance rule, category recoding, or
model hyperparameter is selected after viewing the results.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

BASE_SCRIPT = PROJECT_ROOT / "src" / "23a_test_2023_osm_road_class.py"
STEP22_NETWORK_PATH = PROCESSED_DIR / "atc_step22_2023_network_feature_table.csv"
STEP22_STATION_PATH = PROCESSED_DIR / "atc_step22_2023_feature_table.csv"
STEP22_ROAD_MATCH_PATH = PROCESSED_DIR / "atc_step22_2023_road_matches.csv"
ORIGINAL_CROSSWALK_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_network_crosswalk.csv"

WAY_PATH = PROCESSED_DIR / "atc_step23a1_osm_2023_motor_way_table.csv"
CROSSWALK_PATH = PROCESSED_DIR / "atc_step23a1_osm_2023_motor_network_crosswalk.csv"
NETWORK_PATH = PROCESSED_DIR / "atc_step23a1_osm_2023_network_feature_table.csv"
STATION_PATH = PROCESSED_DIR / "atc_step23a1_osm_2023_station_feature_table.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step23a1_osm_2023_oof_predictions.csv"

TAG_AUDIT_PATH = TABLE_DIR / "step23a1_osm_raw_tag_audit.csv"
MATCH_AUDIT_PATH = TABLE_DIR / "step23a1_motor_match_audit.csv"
SUPPORT_PATH = TABLE_DIR / "step23a1_osm_label_support.csv"
MANUAL_SAMPLE_PATH = TABLE_DIR / "step23a1_blind_match_review.csv"
MANUAL_PROTOCOL_PATH = TABLE_DIR / "step23a1_blind_match_review_protocol.csv"
MANUAL_RESULT_PATH = TABLE_DIR / "step23a1_blind_match_review_results.csv"
METRICS_PATH = TABLE_DIR / "step23a1_metrics_by_fold.csv"
SUMMARY_PATH = TABLE_DIR / "step23a1_model_summary.csv"
COMPARISON_PATH = TABLE_DIR / "step23a1_paired_model_comparison.csv"
SUBGROUP_PATH = TABLE_DIR / "step23a1_subgroup_bias.csv"
CALIBRATION_PATH = TABLE_DIR / "step23a1_predicted_bin_calibration.csv"
SERVICE_PATH = TABLE_DIR / "step23a1_service_identifiability.csv"
DECISION_PATH = TABLE_DIR / "step23a1_decision_audit.csv"

SUPPORT_FIGURE_PATH = FIGURE_DIR / "step23a1_osm_label_support.png"
MODEL_FIGURE_PATH = FIGURE_DIR / "step23a1_mean_model_comparison.png"
BIAS_FIGURE_PATH = FIGURE_DIR / "step23a1_mean_model_bias.png"

FOLDS = (1, 2, 3, 4, 5)
RANDOM_SEED = 42
MIN_SAMPLES_LEAF = 20
MATERIAL_NETWORK_LENGTH_SHARE = 0.01
NETWORK_LENGTH_COVERAGE_THRESHOLD = 0.80
STATION_COVERAGE_THRESHOLD = 0.90
MINOR_STATION_COVERAGE_THRESHOLD = 0.80
SKILL_VS_HIERARCHY_THRESHOLD_PCT = 5.0
INCREMENT_VS_CONTEXT_THRESHOLD_PCT = 2.0
MINOR_INCREMENT_THRESHOLD_PCT = 2.0
AGGREGATE_BIAS_THRESHOLD_PCT = 10.0
SUBGROUP_BIAS_THRESHOLD_PCT = 15.0
PREDICTED_BIN_BIAS_THRESHOLD_PCT = 15.0
POISSON_RMSE_NONINFERIORITY_RATIO = 1.05

# Frozen before the corrected rematch.  Link suffixes inherit their base class.
MOTOR_HIGHWAY_BASES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "road",
    "service",
    "track",
}
NONMOTOR_HIGHWAYS = {
    "bridleway",
    "construction",
    "corridor",
    "cycleway",
    "elevator",
    "footway",
    "path",
    "pedestrian",
    "platform",
    "proposed",
    "raceway",
    "steps",
}
VERDICTS = (
    "correct",
    "parallel_carriageway_mismatch",
    "grade_level_mismatch",
    "nonmotorized",
    "wrong_road",
    "indeterminate",
)

MODEL_ORDER = (
    "hierarchy_lookup",
    "structure_gtfs_absolute",
    "osm_absolute",
    "osm_poisson",
    "osm_squared_error",
)
MODEL_LABELS = {
    "hierarchy_lookup": "10-cell lookup",
    "structure_gtfs_absolute": "Structure + GTFS",
    "osm_absolute": "OSM / absolute error",
    "osm_poisson": "OSM / Poisson mean",
    "osm_squared_error": "OSM / squared-error mean",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def first_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def raw_tag_table(features: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for feature in features:
        if feature.get("type") != "way" or feature.get("id") is None:
            continue
        osm_id = f"way/{feature['id']}"
        if osm_id in seen:
            continue
        seen.add(osm_id)
        tags = feature.get("tags", {}) or {}
        highway = first_value(tags.get("highway")).strip().lower()
        rows.append(
            {
                "osm_id": osm_id,
                "osm_highway": highway,
                "osm_highway_base": highway.removesuffix("_link"),
                "osm_service": first_value(tags.get("service")).strip().lower(),
                "osm_access_raw": first_value(tags.get("access")).strip().lower(),
                "osm_lanes_raw": first_value(tags.get("lanes")).strip().lower(),
                "osm_maxspeed_raw": first_value(tags.get("maxspeed")).strip().lower(),
                "osm_oneway_raw": first_value(tags.get("oneway")).strip().lower(),
            }
        )
    return pd.DataFrame(rows)


def prepare_crosswalk():
    base = load_module("hk_aadt_step23a_base", BASE_SCRIPT)
    base.OSM_WAY_TABLE_PATH = WAY_PATH
    base.OSM_CROSSWALK_PATH = CROSSWALK_PATH
    base.NETWORK_FEATURE_PATH = NETWORK_PATH
    base.STATION_FEATURE_PATH = STATION_PATH

    network = pd.read_csv(STEP22_NETWORK_PATH)
    stations = pd.read_csv(STEP22_STATION_PATH)
    centerline, road_geometries = base.read_centerline(base.find_road_geodatabase())
    if len(centerline) != len(network):
        raise ValueError("Step 22 network and historical road geodatabase differ")
    if set(stations["spatial_fold"].astype(int)) != set(FOLDS):
        raise ValueError("The frozen five spatial folds are not available")

    print("Reading the cached 2023 OSM tiles; no 2023 re-download is planned...")
    raw_features, _ = base.obtain_osm_tiles(centerline)
    tags = raw_tag_table(raw_features)
    ways, osm_geometries, _ = base.parse_osm_ways(raw_features)
    ways = ways.merge(
        tags.drop(columns=["osm_highway"], errors="ignore"),
        on="osm_id",
        how="left",
        validate="one_to_one",
    )
    eligible = ways["osm_highway"].fillna("").map(
        lambda value: str(value).removesuffix("_link") in MOTOR_HIGHWAY_BASES
    )
    ways = ways.loc[eligible].reset_index(drop=True)
    osm_geometries = osm_geometries[np.asarray(eligible, dtype=bool)]
    save_csv(ways, WAY_PATH)

    print("Rematching with the frozen motor-road candidate set...")
    crosswalk = base.match_centerline_to_osm(
        centerline, road_geometries, ways, osm_geometries
    )
    crosswalk = crosswalk.merge(
        ways[["osm_id", "osm_service", "osm_access_raw"]],
        on="osm_id",
        how="left",
        validate="many_to_one",
    )
    save_csv(crosswalk, CROSSWALK_PATH)
    network, stations, gtfs_features = base.build_feature_tables(
        network, stations, crosswalk
    )
    return base, centerline, network, stations, crosswalk, tags, gtfs_features


def write_tag_audit(tags: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    selected = original["osm_highway"].fillna("").astype(str).str.lower()
    frame = (
        selected.value_counts(dropna=False)
        .rename_axis("osm_highway")
        .reset_index(name="official_segment_count_original_match")
    )
    frame["frozen_motor_candidate"] = frame["osm_highway"].map(
        lambda value: str(value).removesuffix("_link") in MOTOR_HIGHWAY_BASES
    )
    source_counts = tags["osm_highway"].value_counts().rename("source_way_count")
    frame = frame.merge(source_counts, on="osm_highway", how="left")
    save_csv(frame, TAG_AUDIT_PATH)
    return frame


def write_match_audit(crosswalk: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    accepted = crosswalk["osm_match_status"].isin(["high", "moderate"])
    station_accepted = stations["osm_match_status"].isin(["high", "moderate"])
    minor = stations["road_network"].astype(str).eq("MINOR")
    lengths = crosswalk["road_segment_length_m"].to_numpy(dtype=float)
    rows = [
        ("network_accepted_length_share", lengths[accepted].sum() / lengths.sum(), NETWORK_LENGTH_COVERAGE_THRESHOLD),
        ("station_accepted_share", station_accepted.mean(), STATION_COVERAGE_THRESHOLD),
        ("minor_station_accepted_share", station_accepted[minor].mean(), MINOR_STATION_COVERAGE_THRESHOLD),
        ("network_unmatched_segment_share", crosswalk["osm_highway_group"].eq("unmatched").mean(), np.nan),
        ("median_accepted_distance_m", crosswalk.loc[accepted, "osm_match_distance_m"].median(), np.nan),
        ("median_accepted_overlap_share", crosswalk.loc[accepted, "osm_overlap_share"].median(), np.nan),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "threshold"])
    frame["pass"] = frame.apply(
        lambda row: True if pd.isna(row["threshold"]) else row["value"] >= row["threshold"],
        axis=1,
    )
    frame["matching_rule"] = (
        "high: distance<=15m and (overlap>=0.50 or name>=0.50); moderate: "
        "distance<=30m and (overlap>=0.20 or name>=0.25), or distance<=8m"
    )
    save_csv(frame, MATCH_AUDIT_PATH)
    return frame


def label_support(
    crosswalk: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    groups = list(crosswalk["osm_highway_group"].dropna().unique())
    rows: list[dict[str, object]] = []
    total_length = crosswalk["road_segment_length_m"].sum()
    for group_name in sorted(groups):
        road = crosswalk[crosswalk["osm_highway_group"] == group_name]
        labelled = stations[stations["osm_highway_group"] == group_name]
        length_km = road["road_segment_length_m"].sum() / 1000.0
        row: dict[str, object] = {
            "osm_highway_group": group_name,
            "network_segment_count": len(road),
            "network_length_km": length_km,
            "network_length_share": road["road_segment_length_m"].sum() / total_length,
            "station_count": len(labelled),
            "stations_per_100km": 100.0 * len(labelled) / length_km if length_km else np.nan,
            "station_region_count": labelled["region"].nunique(dropna=True),
            "observed_aadt_mean": labelled["aadt"].mean(),
            "observed_aadt_median": labelled["aadt"].median(),
            "observed_aadt_q25": labelled["aadt"].quantile(0.25),
            "observed_aadt_q75": labelled["aadt"].quantile(0.75),
        }
        identifiable_folds = 0
        test_supported_folds = 0
        for fold in FOLDS:
            train_count = int((labelled["spatial_fold"].astype(int) != fold).sum())
            test_count = int((labelled["spatial_fold"].astype(int) == fold).sum())
            row[f"fold_{fold}_training_station_count"] = train_count
            row[f"fold_{fold}_training_meets_min_leaf"] = train_count >= MIN_SAMPLES_LEAF
            row[f"fold_{fold}_test_station_count"] = test_count
            identifiable_folds += int(train_count >= MIN_SAMPLES_LEAF)
            test_supported_folds += int(test_count > 0)
        row["training_folds_identifiable"] = identifiable_folds
        row["test_folds_with_station_support"] = test_supported_folds
        row["material_network_group"] = row["network_length_share"] >= MATERIAL_NETWORK_LENGTH_SHARE
        row["full_fold_label_support"] = identifiable_folds == len(FOLDS)
        row["interpretation"] = (
            "OOF remains computable when a test fold has zero labels, but fold-specific "
            "transportability for this class is not assessable"
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("network_length_km", ascending=False)
    save_csv(frame, SUPPORT_PATH)
    return frame


def manual_protocol() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "reviewer_verdict": VERDICTS,
            "definition": (
                "same drivable road and carriageway",
                "wrong parallel or opposite carriageway",
                "wrong elevated, tunnel, or ground-level road",
                "matched candidate is not a motor-vehicle road",
                "different road or unrelated geometry",
                "imagery or map evidence is insufficient",
            ),
            "counts_as_match_error": (False, True, True, True, True, np.nan),
        }
    )
    save_csv(frame, MANUAL_PROTOCOL_PATH)
    return frame


def draw_manual_sample(
    centerline: pd.DataFrame,
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    if MANUAL_SAMPLE_PATH.exists():
        existing = pd.read_csv(MANUAL_SAMPLE_PATH)
        if len(existing) == 100:
            print("Preserved the previously frozen 100-record blind review sample.")
            return existing
    official = centerline[
        [
            "road_2023_segment_index",
            "STREET_ENAME",
            "ALIAS_ENAME",
            "road_longitude",
            "road_latitude",
        ]
    ].rename(columns={"STREET_ENAME": "official_name", "ALIAS_ENAME": "official_alias"})
    old = original[
        ["road_2023_segment_index", "osm_id", "osm_highway", "osm_highway_group"]
    ].rename(
        columns={
            "osm_id": "original_osm_id",
            "osm_highway": "original_osm_highway",
            "osm_highway_group": "original_osm_group",
        }
    )
    frame = corrected.merge(old, on="road_2023_segment_index", how="left").merge(
        official, on="road_2023_segment_index", how="left"
    )
    station_meta = stations[
        ["road_2023_segment_index", "road_network", "region"]
    ].drop_duplicates("road_2023_segment_index")
    frame = frame.merge(station_meta, on="road_2023_segment_index", how="left")
    nonmotor = frame["original_osm_highway"].fillna("").astype(str).str.removesuffix("_link").isin(NONMOTOR_HIGHWAYS)
    strata = (
        ("service", frame["osm_highway_group"].eq("service")),
        ("originally_nonmotorized", nonmotor),
        ("minor_station_link", frame["road_network"].eq("MINOR")),
        ("hong_kong_island_station_link", frame["region"].eq("Hong Kong Island")),
    )
    rng = np.random.default_rng(RANDOM_SEED)
    selected_rows: list[pd.DataFrame] = []
    used: set[int] = set()
    for stratum, mask in strata:
        candidates = frame.loc[mask & ~frame["road_2023_segment_index"].isin(used)].copy()
        if len(candidates) < 25:
            raise ValueError(f"Blind-review stratum {stratum} has only {len(candidates)} unique segments")
        positions = rng.choice(candidates.index.to_numpy(), size=25, replace=False)
        chosen = candidates.loc[positions].copy()
        chosen["review_stratum"] = stratum
        selected_rows.append(chosen)
        used.update(chosen["road_2023_segment_index"].astype(int))
    sample = pd.concat(selected_rows, ignore_index=True)
    sample.insert(0, "audit_id", [f"A{i:03d}" for i in range(1, len(sample) + 1)])
    sample["reviewer_verdict"] = ""
    sample["reviewer_note"] = ""
    keep = [
        "audit_id",
        "review_stratum",
        "road_2023_segment_index",
        "official_name",
        "official_alias",
        "road_longitude",
        "road_latitude",
        "original_osm_id",
        "original_osm_highway",
        "original_osm_group",
        "osm_id",
        "osm_highway",
        "osm_service",
        "osm_highway_group",
        "osm_match_status",
        "osm_match_distance_m",
        "osm_overlap_share",
        "osm_name_similarity",
        "reviewer_verdict",
        "reviewer_note",
    ]
    save_csv(sample[keep], MANUAL_SAMPLE_PATH)
    return sample[keep]


def wilson_interval(errors: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = errors / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def evaluate_manual_review() -> pd.DataFrame:
    review = pd.read_csv(MANUAL_SAMPLE_PATH, keep_default_na=False)
    invalid = sorted(set(review["reviewer_verdict"]) - set(VERDICTS))
    if invalid or (review["reviewer_verdict"] == "").any():
        raise ValueError(
            "Complete all 100 reviewer_verdict cells before --phase evaluate. "
            f"Allowed values: {', '.join(VERDICTS)}"
        )
    rows: list[dict[str, object]] = []
    for stratum, group in [("all", review), *review.groupby("review_stratum")]:
        determinate = group[group["reviewer_verdict"] != "indeterminate"]
        errors = int((determinate["reviewer_verdict"] != "correct").sum())
        low, high = wilson_interval(errors, len(determinate))
        rows.append(
            {
                "review_stratum": stratum,
                "reviewed_n": len(group),
                "determinate_n": len(determinate),
                "indeterminate_n": int((group["reviewer_verdict"] == "indeterminate").sum()),
                "match_error_n": errors,
                "match_error_rate": errors / len(determinate) if len(determinate) else np.nan,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, MANUAL_RESULT_PATH)
    return frame


def fixed_model(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].copy()
    for column in values:
        if values[column].dtype == bool:
            values[column] = values[column].astype(int)
    return values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def metric_row(fold: int | str, model: str, observed, predicted) -> dict[str, object]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rho = (
        spearmanr(observed, predicted).statistic
        if len(observed) > 2 and np.std(observed) > 0 and np.std(predicted) > 0
        else np.nan
    )
    positive = np.maximum(predicted, 1e-6)
    return {
        "spatial_fold": fold,
        "model": model,
        "n": len(observed),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": math.sqrt(mean_squared_error(observed, predicted)),
        "r2": r2_score(observed, predicted),
        "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
        "spearman": rho,
        "mean_poisson_deviance": mean_poisson_deviance(observed, positive),
        "nonpositive_prediction_count": int((predicted <= 0).sum()),
    }


def run_models(base, stations: pd.DataFrame, gtfs_features: list[str]):
    structure_gtfs = [*base.STRUCTURAL_FEATURES, *gtfs_features]
    osm_features = [*structure_gtfs, *base.OSM_CORE_FEATURES]
    rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for fold in FOLDS:
        train = stations[stations["spatial_fold"].astype(int) != fold].copy()
        test = stations[stations["spatial_fold"].astype(int) == fold].copy()
        y_train = train["aadt"].to_numpy(dtype=float)
        y_test = test["aadt"].to_numpy(dtype=float)
        pred: dict[str, np.ndarray] = {
            "hierarchy_lookup": base.hierarchy_lookup_predict(train, test),
        }
        specifications = (
            ("structure_gtfs_absolute", "absolute_error", structure_gtfs),
            ("osm_absolute", "absolute_error", osm_features),
            ("osm_poisson", "poisson", osm_features),
            ("osm_squared_error", "squared_error", osm_features),
        )
        for name, loss, columns in specifications:
            model = fixed_model(loss)
            model.fit(matrix(train, columns), y_train)
            pred[name] = model.predict(matrix(test, columns))
        for name in MODEL_ORDER:
            metrics.append(metric_row(fold, name, y_test, pred[name]))
            for position, (_, station) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "station_id": int(station["station_id"]),
                        "spatial_fold": fold,
                        "region": station["region"],
                        "road_network": station["road_network"],
                        "road_type": station["road_type"],
                        "osm_highway_group": station["osm_highway_group"],
                        "model": name,
                        "observed_aadt": y_test[position],
                        "predicted_aadt": pred[name][position],
                        "absolute_error": abs(pred[name][position] - y_test[position]),
                    }
                )
        print(f"Completed frozen spatial fold {fold}: train={len(train)}, test={len(test)}")
    predictions = pd.DataFrame(rows)
    fold_metrics = pd.DataFrame(metrics)
    save_csv(predictions, PREDICTION_PATH)
    save_csv(fold_metrics, METRICS_PATH)
    summary = pd.DataFrame(
        [
            metric_row(
                "pooled",
                name,
                predictions.loc[predictions["model"] == name, "observed_aadt"],
                predictions.loc[predictions["model"] == name, "predicted_aadt"],
            )
            for name in MODEL_ORDER
        ]
    )
    save_csv(summary, SUMMARY_PATH)
    return predictions, fold_metrics, summary, osm_features


def bootstrap_interval(candidate: pd.DataFrame, reference: pd.DataFrame, draws=4000):
    paired = candidate[["station_id", "spatial_fold", "absolute_error"]].merge(
        reference[["station_id", "absolute_error"]],
        on="station_id",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    paired["difference"] = paired["absolute_error_candidate"] - paired["absolute_error_reference"]
    values = {
        int(fold): group["difference"].to_numpy(dtype=float)
        for fold, group in paired.groupby("spatial_fold")
    }
    rng = np.random.default_rng(RANDOM_SEED)
    fold_ids = np.asarray(sorted(values))
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        estimates[draw] = np.concatenate([values[int(fold)] for fold in sampled]).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("osm_absolute", "hierarchy_lookup", "all"),
        ("osm_absolute", "structure_gtfs_absolute", "all"),
        ("osm_absolute", "structure_gtfs_absolute", "minor"),
    )
    rows = []
    for candidate_name, reference_name, subset in specs:
        candidate = predictions[predictions["model"] == candidate_name]
        reference = predictions[predictions["model"] == reference_name]
        if subset == "minor":
            candidate = candidate[candidate["road_network"] == "MINOR"]
            reference = reference[reference["road_network"] == "MINOR"]
        candidate_mae = candidate["absolute_error"].mean()
        reference_mae = reference["absolute_error"].mean()
        low, high = bootstrap_interval(candidate, reference)
        candidate_fold = candidate.groupby("spatial_fold")["absolute_error"].mean()
        reference_fold = reference.groupby("spatial_fold")["absolute_error"].mean()
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                "evaluation_subset": subset,
                "n": len(candidate),
                "candidate_mae": candidate_mae,
                "reference_mae": reference_mae,
                "mae_improvement_pct": 100.0 * (reference_mae - candidate_mae) / reference_mae,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "improved_fold_count": int((candidate_fold < reference_fold).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, COMPARISON_PATH)
    return frame


def subgroup_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model in predictions.groupby("model"):
        strata = [("all", model)]
        for column in ("region", "road_network", "road_type", "osm_highway_group"):
            strata.extend((f"{column}:{value}", group) for value, group in model.groupby(column, dropna=False))
        for stratum, group in strata:
            observed = group["observed_aadt"].to_numpy(dtype=float)
            predicted = group["predicted_aadt"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "stratum": stratum,
                    "n": len(group),
                    "observed_mean": observed.mean(),
                    "predicted_mean": predicted.mean(),
                    "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, SUBGROUP_PATH)
    return frame


def predicted_bin_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model in predictions.groupby("model"):
        ranked = model["predicted_aadt"].rank(method="first")
        bins = pd.qcut(ranked, q=5, labels=False) + 1
        for quintile, group in model.assign(predicted_quintile=bins).groupby("predicted_quintile"):
            observed = group["observed_aadt"].sum()
            predicted = group["predicted_aadt"].sum()
            rows.append(
                {
                    "model": model_name,
                    "predicted_quintile": int(quintile),
                    "n": len(group),
                    "observed_mean": group["observed_aadt"].mean(),
                    "predicted_mean": group["predicted_aadt"].mean(),
                    "aggregate_bias_pct": 100.0 * (predicted - observed) / observed,
                    "binning_direction": "condition_on_prediction_not_observation",
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, CALIBRATION_PATH)
    return frame


def service_identifiability(
    stations: pd.DataFrame,
    network: pd.DataFrame,
    osm_features: list[str],
) -> pd.DataFrame:
    no_service = [column for column in osm_features if column != "osm_highway_group_service"]
    rows: list[dict[str, object]] = []
    fitted_predictions: dict[str, np.ndarray] = {}
    for label, columns in (("with_service_indicator", osm_features), ("without_service_indicator", no_service)):
        model = fixed_model("absolute_error")
        model.fit(matrix(stations, columns), stations["aadt"].to_numpy(dtype=float))
        fitted_predictions[label] = model.predict(matrix(network, columns))
    working = network[["road_2023_segment_index", "osm_highway_group"]].copy()
    for label, values in fitted_predictions.items():
        working[label] = values
    working["absolute_prediction_difference"] = (
        working["with_service_indicator"] - working["without_service_indicator"]
    ).abs()
    for group_name, group in working.groupby("osm_highway_group", dropna=False):
        observed = stations[stations["osm_highway_group"] == group_name]["aadt"]
        rows.append(
            {
                "osm_highway_group": group_name,
                "network_segment_count": len(group),
                "training_station_count": len(observed),
                "network_prediction_mean": group["with_service_indicator"].mean(),
                "network_prediction_median": group["with_service_indicator"].median(),
                "network_prediction_q10": group["with_service_indicator"].quantile(0.10),
                "network_prediction_q90": group["with_service_indicator"].quantile(0.90),
                "observed_station_mean": observed.mean(),
                "observed_station_median": observed.median(),
                "mean_abs_change_when_indicator_removed": group["absolute_prediction_difference"].mean(),
                "max_abs_change_when_indicator_removed": group["absolute_prediction_difference"].max(),
                "interpretation": "full-sample diagnostic only; network predictions are not validation truth",
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, SERVICE_PATH)
    return frame


def feature_gate(row: pd.Series, threshold: float) -> tuple[bool, str]:
    effect = row["mae_improvement_pct"] >= threshold
    interval = row["cluster_bootstrap_high"] < 0
    folds = row["improved_fold_count"] >= 3
    failures = []
    if not effect:
        failures.append("effect_below_threshold")
    if not interval:
        failures.append("interval_includes_zero")
    if not folds:
        failures.append("fewer_than_three_folds_improve")
    return effect and interval and folds, ";".join(failures) if failures else "none"


def decision_audit(
    match: pd.DataFrame,
    support: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    subgroups: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    match_lookup = match.set_index("metric")
    comp = comparison.set_index(["candidate", "reference", "evaluation_subset"])
    vs_lookup = comp.loc[("osm_absolute", "hierarchy_lookup", "all")]
    vs_context = comp.loc[("osm_absolute", "structure_gtfs_absolute", "all")]
    minor = comp.loc[("osm_absolute", "structure_gtfs_absolute", "minor")]
    lookup_pass, lookup_failure = feature_gate(vs_lookup, SKILL_VS_HIERARCHY_THRESHOLD_PCT)
    context_pass, context_failure = feature_gate(vs_context, INCREMENT_VS_CONTEXT_THRESHOLD_PCT)
    minor_pass, minor_failure = feature_gate(minor, MINOR_INCREMENT_THRESHOLD_PCT)
    coverage_pass = bool(
        match_lookup.loc["network_accepted_length_share", "pass"]
        and match_lookup.loc["station_accepted_share", "pass"]
        and match_lookup.loc["minor_station_accepted_share", "pass"]
    )
    material = support[support["material_network_group"].astype(bool)]
    support_pass = bool(material["full_fold_label_support"].astype(bool).all())
    unsupported = ",".join(material.loc[~material["full_fold_label_support"].astype(bool), "osm_highway_group"])

    summary_lookup = summary.set_index("model")
    poisson = summary_lookup.loc["osm_poisson"]
    absolute = summary_lookup.loc["osm_absolute"]
    mean_bias = abs(poisson["aggregate_bias_pct"]) <= AGGREGATE_BIAS_THRESHOLD_PCT
    mean_groups = subgroups[
        (subgroups["model"] == "osm_poisson")
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ]
    max_group_bias = mean_groups["aggregate_bias_pct"].abs().max()
    group_bias = max_group_bias <= SUBGROUP_BIAS_THRESHOLD_PCT
    max_bin_bias = calibration.loc[
        calibration["model"] == "osm_poisson", "aggregate_bias_pct"
    ].abs().max()
    bin_bias = max_bin_bias <= PREDICTED_BIN_BIAS_THRESHOLD_PCT
    rmse_ratio = poisson["rmse"] / absolute["rmse"]
    rmse_gate = rmse_ratio <= POISSON_RMSE_NONINFERIORITY_RATIO
    poisson_gate = bool(mean_bias and group_bias and bin_bias and rmse_gate)
    full_gate = bool(
        coverage_pass
        and support_pass
        and lookup_pass
        and context_pass
        and minor_pass
        and poisson_gate
    )

    def comp_evidence(row):
        return (
            f"improvement={row['mae_improvement_pct']:.2f}%; interval="
            f"[{row['cluster_bootstrap_low']:.1f},{row['cluster_bootstrap_high']:.1f}]; "
            f"improved_folds={int(row['improved_fold_count'])}/5"
        )

    rows = [
        {
            "decision": "corrected_motor_crosswalk_coverage_passes",
            "pass": coverage_pass,
            "evidence": "network, station, and MINOR thresholds are evaluated separately",
            "failed_criterion": "none" if coverage_pass else "coverage_threshold_failed",
        },
        {
            "decision": "material_osm_groups_have_foldwise_label_support",
            "pass": support_pass,
            "evidence": f"material_group_threshold={MATERIAL_NETWORK_LENGTH_SHARE:.2%}; unsupported={unsupported or 'none'}",
            "failed_criterion": "none" if support_pass else "min_training_count_below_min_samples_leaf",
        },
        {
            "decision": "absolute_osm_feature_block_beats_hierarchy_lookup",
            "pass": lookup_pass,
            "evidence": comp_evidence(vs_lookup),
            "failed_criterion": lookup_failure,
        },
        {
            "decision": "absolute_osm_feature_block_adds_skill_beyond_context",
            "pass": context_pass,
            "evidence": comp_evidence(vs_context),
            "failed_criterion": context_failure,
        },
        {
            "decision": "absolute_osm_feature_block_improves_minor_roads",
            "pass": minor_pass,
            "evidence": comp_evidence(minor),
            "failed_criterion": minor_failure,
        },
        {
            "decision": "poisson_mean_model_overall_bias_passes",
            "pass": mean_bias,
            "evidence": f"bias={poisson['aggregate_bias_pct']:+.2f}%; threshold=±{AGGREGATE_BIAS_THRESHOLD_PCT:.0f}%",
            "failed_criterion": "none" if mean_bias else "overall_bias_exceeds_10pct",
        },
        {
            "decision": "poisson_mean_model_subgroup_bias_passes",
            "pass": group_bias,
            "evidence": f"maximum_region_or_network_abs_bias={max_group_bias:.2f}%; threshold={SUBGROUP_BIAS_THRESHOLD_PCT:.0f}%",
            "failed_criterion": "none" if group_bias else "region_or_network_bias_exceeds_15pct",
        },
        {
            "decision": "poisson_mean_model_prediction_bin_calibration_passes",
            "pass": bin_bias,
            "evidence": f"maximum_prediction_quintile_abs_bias={max_bin_bias:.2f}%; threshold={PREDICTED_BIN_BIAS_THRESHOLD_PCT:.0f}%",
            "failed_criterion": "none" if bin_bias else "prediction_bin_bias_exceeds_15pct",
        },
        {
            "decision": "poisson_mean_model_rmse_is_noninferior",
            "pass": rmse_gate,
            "evidence": f"poisson_to_absolute_rmse_ratio={rmse_ratio:.4f}; ceiling={POISSON_RMSE_NONINFERIORITY_RATIO:.2f}",
            "failed_criterion": "none" if rmse_gate else "rmse_deterioration_exceeds_5pct",
        },
        {
            "decision": "step23a1_full_network_2023_gate",
            "pass": full_gate,
            "evidence": (
                f"coverage={coverage_pass}; label_support={support_pass}; feature_skill="
                f"{lookup_pass and context_pass and minor_pass}; poisson_mean={poisson_gate}"
            ),
            "failed_criterion": "none" if full_gate else "one_or_more_predeclared_full_network_gates_failed",
        },
        {
            "decision": "step23b_data_audit_may_run_independently",
            "pass": True,
            "evidence": "historical availability and edit-churn auditing does not use or change the 2023 model gate",
            "failed_criterion": "none",
        },
        {
            "decision": "step23b_historical_modelling_is_authorised",
            "pass": False,
            "evidence": "Step 23B-data is descriptive; modelling requires both the 2023 gate and a separate temporal-signal decision",
            "failed_criterion": "data_audit_does_not_establish_temporal_identification",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, DECISION_PATH)
    return frame


def plots(support: pd.DataFrame, summary: pd.DataFrame, subgroups: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    ordered = support.sort_values("network_length_km")
    colors = ["#C44E52" if not value else "#4C78A8" for value in ordered["full_fold_label_support"]]
    minimum_training_count = ordered[
        [f"fold_{fold}_training_station_count" for fold in FOLDS]
    ].min(axis=1)
    axis.barh(ordered["osm_highway_group"], minimum_training_count, color=colors)
    axis.axvline(MIN_SAMPLES_LEAF, color="#333333", linestyle="--", label="min_samples_leaf")
    axis.set_xlabel("Minimum training-station count across the five folds")
    axis.set_title("OSM class support must be assessed inside every training fold")
    axis.legend()
    figure.tight_layout()
    figure.savefig(SUPPORT_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    selected = summary.set_index("model").loc[list(MODEL_ORDER)]
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(range(len(selected)), selected["mae"], color="#4C78A8")
    axis.set_xticks(range(len(selected)))
    axis.set_xticklabels([MODEL_LABELS[value] for value in selected.index], rotation=22, ha="right")
    axis.set_ylabel("Spatial OOF MAE (vehicles/day)")
    axis.set_title("Feature skill and mean calibration are separate decisions")
    figure.tight_layout()
    figure.savefig(MODEL_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    selected_bias = subgroups[
        subgroups["model"].isin(["osm_absolute", "osm_poisson", "osm_squared_error"])
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ]
    pivot = selected_bias.pivot(index="stratum", columns="model", values="aggregate_bias_pct")
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot.rename(columns=MODEL_LABELS).plot.bar(ax=axis)
    axis.axhline(15, color="#C44E52", linestyle="--")
    axis.axhline(-15, color="#C44E52", linestyle="--")
    axis.set_ylabel("Aggregate bias (%)")
    axis.set_title("Mean-target models are judged by calibration, not the MAE feature gate")
    figure.tight_layout()
    figure.savefig(BIAS_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def update_manifest(paths: list[Path]) -> None:
    existing = (
        pd.read_csv(REPORT_MANIFEST_PATH)
        if REPORT_MANIFEST_PATH.exists()
        else pd.DataFrame(columns=["artifact", "status", "reason"])
    )
    names = {str(path.relative_to(PROJECT_ROOT)) for path in paths}
    existing = existing[~existing["artifact"].isin(names)]
    added = pd.DataFrame(
        {
            "artifact": sorted(names),
            "status": "step23a1_audit_or_result",
            "reason": "motor_only_crosswalk_label_support_blind_review_and_mean_target_gates",
        }
    )
    save_csv(pd.concat([existing, added], ignore_index=True), REPORT_MANIFEST_PATH)


def validate_inputs() -> None:
    required = [BASE_SCRIPT, STEP22_NETWORK_PATH, STEP22_STATION_PATH, STEP22_ROAD_MATCH_PATH, ORIGINAL_CROSSWALK_PATH]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the corrected Steps 22 and 23A first; missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "evaluate"), default="prepare")
    args = parser.parse_args()
    validate_inputs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base, centerline, network, stations, crosswalk, tags, gtfs_features = prepare_crosswalk()
    original = pd.read_csv(ORIGINAL_CROSSWALK_PATH)
    write_tag_audit(tags, original)
    match = write_match_audit(crosswalk, stations)
    support = label_support(crosswalk, stations)
    manual_protocol()
    draw_manual_sample(centerline, original, crosswalk, stations)

    if args.phase == "prepare":
        plot_paths = [TAG_AUDIT_PATH, MATCH_AUDIT_PATH, SUPPORT_PATH, MANUAL_SAMPLE_PATH, MANUAL_PROTOCOL_PATH, CROSSWALK_PATH, NETWORK_PATH, STATION_PATH]
        update_manifest(plot_paths)
        print("\nStep 23A.1 preparation is complete.")
        print("  The frozen 100-record blind sample contains no AADT, predictions, or residuals.")
        print(f"  Fill reviewer_verdict in {MANUAL_SAMPLE_PATH.relative_to(PROJECT_ROOT)}.")
        print("  For an automatic local geometry atlas, run src/23a1b_build_blind_review_atlas.py.")
        print("  Then rerun this script with --phase evaluate. No model has been fit in this phase.")
        return

    evaluate_manual_review()
    predictions, _, summary, osm_features = run_models(base, stations, gtfs_features)
    comparison = comparisons(predictions)
    subgroups = subgroup_bias(predictions)
    calibration = predicted_bin_calibration(predictions)
    service_identifiability(stations, network, osm_features)
    decisions = decision_audit(match, support, summary, comparison, subgroups, calibration)
    plots(support, summary, subgroups)
    all_paths = [
        TAG_AUDIT_PATH, MATCH_AUDIT_PATH, SUPPORT_PATH, MANUAL_SAMPLE_PATH,
        MANUAL_PROTOCOL_PATH, MANUAL_RESULT_PATH, METRICS_PATH, SUMMARY_PATH,
        COMPARISON_PATH, SUBGROUP_PATH, CALIBRATION_PATH, SERVICE_PATH,
        DECISION_PATH, SUPPORT_FIGURE_PATH, MODEL_FIGURE_PATH, BIAS_FIGURE_PATH,
        CROSSWALK_PATH, NETWORK_PATH, STATION_PATH, PREDICTION_PATH,
    ]
    update_manifest(all_paths)
    full = bool(decisions.loc[decisions["decision"] == "step23a1_full_network_2023_gate", "pass"].iloc[0])
    print("\nStep 23A.1 evaluation is complete.")
    for _, row in summary.iterrows():
        print(f"  {MODEL_LABELS[row['model']]}: MAE {row['mae']:,.0f}; RMSE {row['rmse']:,.0f}; bias {row['aggregate_bias_pct']:+.1f}%.")
    print("  Decision: " + ("the bounded 2023 gate passes." if full else "the predeclared bounded 2023 full-network gate does not pass."))


if __name__ == "__main__":
    main()
