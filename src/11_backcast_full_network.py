from __future__ import annotations

import csv
import gzip
import html
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hk_aadt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SPATIAL_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "spatial"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

CENTERLINE_PATH = RAW_SPATIAL_DIR / "CENTERLINE.kmz"
NETWORK_PATH = PROCESSED_DIR / "atc_network_segment_features.csv"
TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
OOF_PATH = PROCESSED_DIR / "atc_step9_oof_predictions.csv"
STEP9_SUMMARY_PATH = TABLE_DIR / "step9_spatial_holdout_summary.csv"
STEP9_CALIBRATION_PATH = TABLE_DIR / "step9_quartile_calibration.csv"

BACKCAST_PATH = PROCESSED_DIR / "atc_step11_full_network_backcast.csv"
BACKCAST_GEOJSON_PATH = PROCESSED_DIR / "atc_step11_full_network_backcast.geojson.gz"
SUMMARY_PATH = TABLE_DIR / "step11_backcast_summary.csv"
CHANGE_SUMMARY_PATH = TABLE_DIR / "step11_change_summary.csv"
SUPPORT_AUDIT_PATH = TABLE_DIR / "step11_spatial_support_audit.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step11_backcast_decision_audit.csv"
BACKCAST_MAP_PATH = FIGURE_DIR / "step11_full_network_backcast_maps.png"
CHANGE_MAP_PATH = FIGURE_DIR / "step11_backcast_change_maps.png"
SUPPORT_MAP_PATH = FIGURE_DIR / "step11_spatial_support_map.png"

CENTERLINE_URL = "https://static.data.gov.hk/td/traffic-flow-census/CENTERLINE.kmz"
YEARS = (2011, 2016, 2021)
PERIODS = ((2011, 2016), (2016, 2021), (2011, 2021))
EARTH_RADIUS_KM = 6371.0088

MODEL_FEATURES = [
    "centroid_longitude",
    "centroid_latitude",
    "computed_length_m",
    "elevation",
    "travel_direction",
    "route_number_present",
    "endpoint_degree_mean",
    "street_code_segment_count",
]
STRUCTURAL_RANGE_FEATURES = [
    "computed_length_m",
    "elevation",
    "travel_direction",
    "route_number_present",
    "endpoint_degree_mean",
    "street_code_segment_count",
]

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
PLACEMARK_TAG = f"{{{KML_NAMESPACE}}}Placemark"
DESCRIPTION_TAG = f"{{{KML_NAMESPACE}}}description"
COORDINATES_TAG = f"{{{KML_NAMESPACE}}}coordinates"


def normalize_route_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


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


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[MODEL_FEATURES].copy()
    values["route_number_present"] = values["route_number_present"].astype(int)
    return values.to_numpy(dtype=float)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = (
        NETWORK_PATH,
        TRAINING_PATH,
        OOF_PATH,
        STEP9_SUMMARY_PATH,
        STEP9_CALIBRATION_PATH,
    )
    missing = [path.relative_to(PROJECT_ROOT) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 9 inputs: {missing}")

    network = pd.read_csv(NETWORK_PATH, dtype={"route_id": str})
    training = pd.read_csv(TRAINING_PATH, dtype={"selected_route_id": str})
    oof = pd.read_csv(OOF_PATH)
    step9_summary = pd.read_csv(STEP9_SUMMARY_PATH)
    calibration = pd.read_csv(STEP9_CALIBRATION_PATH)
    network["route_id"] = network["route_id"].map(normalize_route_id)
    training["selected_route_id"] = training["selected_route_id"].map(
        normalize_route_id
    )
    return network, training, oof, step9_summary, calibration


def validate_inputs(
    network: pd.DataFrame,
    training: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    required_network = {"route_id", *MODEL_FEATURES}
    required_training = {
        "station_id",
        "selected_route_id",
        *(f"aadt_{year}" for year in YEARS),
        *MODEL_FEATURES,
    }
    if missing := sorted(required_network - set(network.columns)):
        raise ValueError(f"Missing network columns: {missing}")
    if missing := sorted(required_training - set(training.columns)):
        raise ValueError(f"Missing training columns: {missing}")
    if network["route_id"].duplicated().any():
        raise ValueError("Network route IDs are not unique.")
    if training["station_id"].duplicated().any():
        raise ValueError("Training station IDs are not unique.")
    if not set(training["selected_route_id"]).issubset(set(network["route_id"])):
        raise ValueError("A training route is absent from the prediction network.")
    checked_network = network[MODEL_FEATURES]
    checked_training = training[[*MODEL_FEATURES, *(f"aadt_{year}" for year in YEARS)]]
    if checked_network.isna().any().any() or checked_training.isna().any().any():
        raise ValueError("Model inputs contain missing values.")
    hgb_oof = oof[oof["model"] == "hist_gradient_boosting"]
    if len(hgb_oof) != len(training) * len(YEARS):
        raise ValueError("Step 9 nonlinear OOF predictions are incomplete.")


def nearest_training_distance_km(
    network: pd.DataFrame,
    training: pd.DataFrame,
) -> np.ndarray:
    training_coordinates = np.radians(
        training[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    )
    network_coordinates = np.radians(
        network[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    )
    neighbours = NearestNeighbors(
        n_neighbors=1,
        algorithm="ball_tree",
        metric="haversine",
    )
    neighbours.fit(training_coordinates)
    distances, _ = neighbours.kneighbors(network_coordinates)
    return distances[:, 0] * EARTH_RADIUS_KM


def structural_range_exceedance_count(
    network: pd.DataFrame,
    training: pd.DataFrame,
) -> np.ndarray:
    counts = np.zeros(len(network), dtype=int)
    for feature in STRUCTURAL_RANGE_FEATURES:
        network_values = pd.to_numeric(network[feature], errors="raise").to_numpy()
        training_values = pd.to_numeric(training[feature], errors="raise")
        counts += (
            (network_values < training_values.min())
            | (network_values > training_values.max())
        ).astype(int)
    return counts


def support_tier(
    direct_station_count: int,
    nearest_distance_km: float,
    range_exceedance_count: int,
) -> str:
    if direct_station_count > 0:
        return "observed_anchor_route"
    if nearest_distance_km <= 1.0 and range_exceedance_count == 0:
        return "within_1km_in_range"
    if nearest_distance_km <= 3.0 and range_exceedance_count <= 1:
        return "within_3km_limited_exceedance"
    return "sparse_or_extrapolative_support"


def add_support_fields(
    network: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    result = network.copy()
    station_count_by_route = Counter(training["selected_route_id"])
    result["direct_observed_station_count"] = result["route_id"].map(
        station_count_by_route
    ).fillna(0).astype(int)
    result["nearest_training_route_centroid_km"] = np.round(
        nearest_training_distance_km(result, training),
        4,
    )
    result["structural_feature_range_exceedance_count"] = (
        structural_range_exceedance_count(result, training)
    )
    result["spatial_support_tier"] = [
        support_tier(station_count, distance, exceedance)
        for station_count, distance, exceedance in zip(
            result["direct_observed_station_count"],
            result["nearest_training_route_centroid_km"],
            result["structural_feature_range_exceedance_count"],
        )
    ]
    result["support_tier_status"] = "descriptive_screen_not_prediction_interval"
    return result


def add_observed_anchor_fields(
    network: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    result = network.copy()
    for year in YEARS:
        anchor = (
            training.groupby("selected_route_id", as_index=True)[f"aadt_{year}"]
            .median()
            .to_dict()
        )
        result[f"observed_anchor_aadt_{year}"] = result["route_id"].map(anchor)
    result["observed_anchor_aggregation"] = np.where(
        result["direct_observed_station_count"] > 1,
        "median_of_multiple_linked_stations",
        np.where(
            result["direct_observed_station_count"] == 1,
            "single_linked_station",
            "no_observed_anchor",
        ),
    )
    return result


def run_full_network_backcast(
    network: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    result = network.copy()
    network_matrix = model_matrix(result)
    training_matrix = model_matrix(training)
    print("Fitting the frozen Step 9 model on all 679 stations...")
    for year in YEARS:
        model = fixed_step9_model()
        model.fit(training_matrix, training[f"aadt_{year}"].to_numpy(dtype=float))
        raw_predictions = model.predict(network_matrix)
        result[f"raw_predicted_aadt_{year}"] = np.round(raw_predictions, 4)
        result[f"prediction_floor_applied_{year}"] = raw_predictions < 1.0
        result[f"predicted_aadt_{year}"] = np.round(
            np.maximum(raw_predictions, 1.0),
            4,
        )
        print(
            f"Predicted current-network AADT support for {year}: "
            f"{len(raw_predictions)} segments; "
            f"physical floor applied to {(raw_predictions < 1.0).sum()}"
        )

    for start_year, end_year in PERIODS:
        start = result[f"predicted_aadt_{start_year}"].to_numpy(dtype=float)
        end = result[f"predicted_aadt_{end_year}"].to_numpy(dtype=float)
        result[f"predicted_change_{start_year}_{end_year}"] = np.round(end - start, 4)
        result[f"predicted_pct_change_{start_year}_{end_year}"] = np.round(
            100 * (end - start) / start,
            4,
        )
    result["prediction_model"] = "frozen_step9_hist_gradient_boosting_full_fit"
    result["prediction_status"] = (
        "preliminary_backcast_with_physical_floor_raw_preserved"
    )
    result["physical_prediction_floor_vehicles_per_day"] = 1.0
    result["tail_calibration_applied"] = False
    result["network_time_support"] = "current_centerline_not_historical_topology_proof"
    result["year_2021_interpretation"] = (
        "calendar_year_surface_no_networkwide_pandemic_assumption"
    )
    return result


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def build_backcast_summary(
    backcast: pd.DataFrame,
    training: pd.DataFrame,
    oof: pd.DataFrame,
    step9_summary: pd.DataFrame,
    calibration: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hgb_oof = oof[oof["model"] == "hist_gradient_boosting"]
    hgb_summary = step9_summary[step9_summary["model"] == "hist_gradient_boosting"]
    for year in YEARS:
        values = backcast[f"predicted_aadt_{year}"]
        raw_values = backcast[f"raw_predicted_aadt_{year}"]
        targets = training[f"aadt_{year}"]
        errors = hgb_oof[hgb_oof["year"] == year]["absolute_error"]
        model_summary = hgb_summary[hgb_summary["year"] == year].iloc[0]
        year_calibration = calibration[calibration["year"] == year].set_index(
            "observed_quartile"
        )
        rows.append(
            {
                "year": year,
                "network_segment_count": len(backcast),
                "raw_nonpositive_prediction_count": int((raw_values <= 0).sum()),
                "raw_prediction_min_aadt": round(float(raw_values.min()), 4),
                "physical_floor_applied_count": int(
                    backcast[f"prediction_floor_applied_{year}"].sum()
                ),
                "predicted_mean_aadt": round(float(values.mean()), 4),
                "predicted_length_weighted_mean_aadt": round(
                    weighted_mean(values, backcast["computed_length_m"]), 4
                ),
                "predicted_min_aadt": round(float(values.min()), 4),
                "predicted_p10_aadt": round(float(values.quantile(0.10)), 4),
                "predicted_p25_aadt": round(float(values.quantile(0.25)), 4),
                "predicted_median_aadt": round(float(values.median()), 4),
                "predicted_p75_aadt": round(float(values.quantile(0.75)), 4),
                "predicted_p90_aadt": round(float(values.quantile(0.90)), 4),
                "predicted_p95_aadt": round(float(values.quantile(0.95)), 4),
                "predicted_p99_aadt": round(float(values.quantile(0.99)), 4),
                "predicted_max_aadt": round(float(values.max()), 4),
                "training_observed_median_aadt": round(float(targets.median()), 4),
                "training_observed_p90_aadt": round(float(targets.quantile(0.90)), 4),
                "training_observed_p99_aadt": round(float(targets.quantile(0.99)), 4),
                "training_observed_max_aadt": round(float(targets.max()), 4),
                "step9_oof_mae": round(float(model_summary["pooled_mae"]), 4),
                "step9_oof_rmse": round(float(model_summary["pooled_rmse"]), 4),
                "step9_oof_r2": round(float(model_summary["pooled_r2"]), 6),
                "step9_oof_absolute_error_p80": round(float(errors.quantile(0.80)), 4),
                "step9_oof_absolute_error_p90": round(float(errors.quantile(0.90)), 4),
                "step9_q1_mean_bias_pct": round(
                    float(year_calibration.loc["Q1_low", "mean_bias_pct_of_observed"]),
                    3,
                ),
                "step9_q4_mean_bias_pct": round(
                    float(year_calibration.loc["Q4_high", "mean_bias_pct_of_observed"]),
                    3,
                ),
                "interpretation": "preliminary_raw_cross_section_not_temporally_identified",
            }
        )
    return rows


def build_change_summary(backcast: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start_year, end_year in PERIODS:
        start_values = backcast[f"predicted_aadt_{start_year}"]
        end_values = backcast[f"predicted_aadt_{end_year}"]
        changes = backcast[f"predicted_change_{start_year}_{end_year}"]
        percentages = backcast[f"predicted_pct_change_{start_year}_{end_year}"]
        start_weighted_mean = weighted_mean(
            start_values,
            backcast["computed_length_m"],
        )
        end_weighted_mean = weighted_mean(
            end_values,
            backcast["computed_length_m"],
        )
        rows.append(
            {
                "period": f"{start_year}-{end_year}",
                "network_segment_count": len(backcast),
                "mean_change_aadt": round(float(changes.mean()), 4),
                "length_weighted_mean_change_aadt": round(
                    weighted_mean(changes, backcast["computed_length_m"]), 4
                ),
                "median_change_aadt": round(float(changes.median()), 4),
                "p10_change_aadt": round(float(changes.quantile(0.10)), 4),
                "p90_change_aadt": round(float(changes.quantile(0.90)), 4),
                "network_mean_pct_change": round(
                    100 * (float(end_values.mean()) - float(start_values.mean()))
                    / float(start_values.mean()),
                    4,
                ),
                "length_weighted_mean_pct_change": round(
                    100 * (end_weighted_mean - start_weighted_mean)
                    / start_weighted_mean,
                    4,
                ),
                "median_pct_change": round(float(percentages.median()), 4),
                "p10_pct_change": round(float(percentages.quantile(0.10)), 4),
                "p90_pct_change": round(float(percentages.quantile(0.90)), 4),
                "segments_predicted_increase": int((changes > 0).sum()),
                "segments_predicted_decrease": int((changes < 0).sum()),
                "segments_predicted_no_change": int((changes == 0).sum()),
                "predicted_increase_pct": round(100 * float((changes > 0).mean()), 3),
                "predicted_decrease_pct": round(100 * float((changes < 0).mean()), 3),
                "interpretation": "exploratory_change_not_reportable_without_refit_and_transfer_diagnostics",
            }
        )
    return rows


def audit_row(
    record_type: str,
    category: str,
    count: int,
    total: int,
    decision: str,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "category": category,
        "count": count,
        "pct_of_network": round(100 * count / total, 3),
        "decision": decision,
    }


def build_support_audit(backcast: pd.DataFrame) -> list[dict[str, object]]:
    total = len(backcast)
    rows: list[dict[str, object]] = []
    for category, count in backcast["spatial_support_tier"].value_counts().items():
        rows.append(
            audit_row(
                "spatial_support_tier",
                str(category),
                int(count),
                total,
                "descriptive_screen_not_statistical_confidence",
            )
        )

    distances = backcast["nearest_training_route_centroid_km"]
    distance_masks = {
        "within_1km": distances <= 1,
        "1_to_3km": (distances > 1) & (distances <= 3),
        "3_to_5km": (distances > 3) & (distances <= 5),
        "over_5km": distances > 5,
    }
    for category, mask in distance_masks.items():
        rows.append(
            audit_row(
                "nearest_training_distance_band",
                category,
                int(mask.sum()),
                total,
                "report_spatial_support_coverage",
            )
        )

    exceedance = backcast["structural_feature_range_exceedance_count"]
    for value, count in exceedance.value_counts().sort_index().items():
        rows.append(
            audit_row(
                "structural_range_exceedance_count",
                str(int(value)),
                int(count),
                total,
                "outside_training_range_is_not_automatic_prediction_failure",
            )
        )
    return rows


def build_decision_audit(
    backcast: pd.DataFrame,
    training: pd.DataFrame,
) -> list[dict[str, object]]:
    prediction_columns = [f"predicted_aadt_{year}" for year in YEARS]
    raw_prediction_columns = [f"raw_predicted_aadt_{year}" for year in YEARS]
    return [
        {"metric": "network_segment_count", "count": len(backcast), "value": "", "decision": "one_current_centerline_feature_per_row"},
        {"metric": "unique_route_id_count", "count": backcast["route_id"].nunique(), "value": "", "decision": "must_equal_network_segment_count"},
        {"metric": "training_station_count", "count": len(training), "value": "", "decision": "high_confidence_observed_panel"},
        {"metric": "direct_observed_anchor_route_count", "count": int((backcast["direct_observed_station_count"] > 0).sum()), "value": "", "decision": "observations_stored_separately_from_predictions"},
        {"metric": "missing_prediction_cells", "count": int(backcast[prediction_columns].isna().sum().sum()), "value": "", "decision": "must_equal_zero"},
        {"metric": "nonpositive_prediction_cells", "count": int((backcast[prediction_columns] <= 0).sum().sum()), "value": "", "decision": "must_equal_zero"},
        {"metric": "raw_nonpositive_prediction_cells", "count": int((backcast[raw_prediction_columns] <= 0).sum().sum()), "value": "", "decision": "retain_raw_values_for_audit"},
        {"metric": "physical_floor_adjusted_prediction_cells", "count": int(backcast[[f"prediction_floor_applied_{year}" for year in YEARS]].sum().sum()), "value": "floor_at_1_vehicle_per_day", "decision": "physical_output_only_raw_values_preserved"},
        {"metric": "model_hyperparameter_retuning", "count": 0, "value": "", "decision": "frozen_step9_model"},
        {"metric": "tail_calibration_applied", "count": 0, "value": "", "decision": "preserve_raw_prediction_and_report_bias"},
        {"metric": "full_fit_predictions_are_oof", "count": 0, "value": "", "decision": "validation_metrics_come_from_step9_oof_only"},
        {"metric": "historical_topology_proven", "count": 0, "value": "", "decision": "current_centerline_is_harmonised_support"},
        {"metric": "direct_sum_ready_for_equity_exposure", "count": 0, "value": "", "decision": "requires_official_network_support_crosswalk_and_directed_or_parallel_carriageway_rule"},
        {"metric": "year_2021_status", "count": "", "value": "calendar_year_surface", "decision": "do_not_assume_networkwide_pandemic_suppression_without_official_aggregate_check"},
        {"metric": "step11_decision_signal", "count": "", "value": "proceed_to_temporal_identifiability_and_network_support_audits_before_equity_use", "decision": "full_fit_surfaces_are_not_validation_evidence"},
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def download_centerline() -> None:
    if CENTERLINE_PATH.exists() and CENTERLINE_PATH.stat().st_size > 0:
        print(f"Already available: {CENTERLINE_PATH.relative_to(PROJECT_ROOT)}")
        return
    CENTERLINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading: CENTERLINE.kmz (about 120 MB)")
    request = Request(
        CENTERLINE_URL,
        headers={"User-Agent": "HK-AADT-research-pilot/1.0"},
    )
    with urlopen(request, timeout=180) as response, CENTERLINE_PATH.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    print(
        f"Saved: {CENTERLINE_PATH.relative_to(PROJECT_ROOT)} "
        f"({CENTERLINE_PATH.stat().st_size / (1024 * 1024):.1f} MB)"
    )


def clean_cell(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = " ".join(html.unescape(value).split())
    return "" if value.casefold() in {"<null>", "null", "-99"} else value


def description_fields(description: str) -> dict[str, str]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", description, re.IGNORECASE | re.DOTALL)
    fields: dict[str, str] = {}
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
        if len(cells) >= 2:
            fields[clean_cell(cells[0]).upper()] = clean_cell(cells[1])
    return fields


def parse_coordinate_text(value: str) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for coordinate in value.split():
        parts = coordinate.split(",")
        if len(parts) >= 2:
            coordinates.append([round(float(parts[0]), 6), round(float(parts[1]), 6)])
    return coordinates


def kml_document_name(archive: ZipFile) -> str:
    names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
    if not names:
        raise ValueError("No KML document in CENTERLINE.kmz")
    return names[0]


def stream_centerline_placemarks():
    with ZipFile(CENTERLINE_PATH) as archive:
        with archive.open(kml_document_name(archive)) as kml_file:
            for _, element in ET.iterparse(kml_file, events=("end",)):
                if element.tag == PLACEMARK_TAG:
                    fields = description_fields(
                        element.findtext(DESCRIPTION_TAG, default="")
                    )
                    parts = [
                        parse_coordinate_text(coordinate.text or "")
                        for coordinate in element.iter(COORDINATES_TAG)
                    ]
                    yield fields, [part for part in parts if len(part) >= 2]
                    element.clear()


def geometry_payload(parts: list[list[list[float]]]) -> dict[str, object]:
    if len(parts) == 1:
        return {"type": "LineString", "coordinates": parts[0]}
    return {"type": "MultiLineString", "coordinates": parts}


def write_backcast_geojson_and_collect_lines(
    backcast: pd.DataFrame,
) -> tuple[list[np.ndarray], list[str]]:
    download_centerline()
    prediction_lookup = backcast.set_index("route_id").to_dict(orient="index")
    BACKCAST_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    line_parts: list[np.ndarray] = []
    part_route_ids: list[str] = []
    feature_count = 0
    first_feature = True

    print("Streaming the current official centerline for GeoJSON and maps...")
    with gzip.open(BACKCAST_GEOJSON_PATH, "wt", encoding="utf-8") as output:
        output.write(
            '{"type":"FeatureCollection","name":"atc_step11_full_network_backcast",'
            '"crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},'
            '"features":['
        )
        for fields, parts in stream_centerline_placemarks():
            route_id = normalize_route_id(fields.get("ROUTE_ID", ""))
            if not route_id or route_id not in prediction_lookup or not parts:
                continue
            row = prediction_lookup[route_id]
            properties = {
                "route_id": route_id,
                "street_ename": row.get("street_ename", ""),
                "street_cname": row.get("street_cname", ""),
                "predicted_aadt_2011": row["predicted_aadt_2011"],
                "predicted_aadt_2016": row["predicted_aadt_2016"],
                "predicted_aadt_2021": row["predicted_aadt_2021"],
                "prediction_floor_applied_any": bool(
                    row["prediction_floor_applied_2011"]
                    or row["prediction_floor_applied_2016"]
                    or row["prediction_floor_applied_2021"]
                ),
                "predicted_change_2011_2016": row["predicted_change_2011_2016"],
                "predicted_change_2016_2021": row["predicted_change_2016_2021"],
                "direct_observed_station_count": row["direct_observed_station_count"],
                "nearest_training_route_centroid_km": row[
                    "nearest_training_route_centroid_km"
                ],
                "spatial_support_tier": row["spatial_support_tier"],
                "prediction_status": row["prediction_status"],
            }
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry_payload(parts),
            }
            if not first_feature:
                output.write(",")
            json.dump(feature, output, ensure_ascii=False, separators=(",", ":"))
            first_feature = False
            feature_count += 1
            for part in parts:
                line_parts.append(np.asarray(part, dtype=float))
                part_route_ids.append(route_id)
        output.write("]}")

    if feature_count != len(backcast):
        raise ValueError(
            f"GeoJSON contains {feature_count} features, expected {len(backcast)}."
        )
    print(f"Saved: {BACKCAST_GEOJSON_PATH.relative_to(PROJECT_ROOT)}")
    return line_parts, part_route_ids


def common_map_limits(backcast: pd.DataFrame) -> tuple[float, float, float, float]:
    padding = 0.01
    return (
        float(backcast["min_longitude"].min()) - padding,
        float(backcast["max_longitude"].max()) + padding,
        float(backcast["min_latitude"].min()) - padding,
        float(backcast["max_latitude"].max()) + padding,
    )


def style_map_axis(axis: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def plot_backcast_maps(
    backcast: pd.DataFrame,
    line_parts: list[np.ndarray],
    part_route_ids: list[str],
) -> None:
    lookup = backcast.set_index("route_id")
    all_values = np.concatenate(
        [backcast[f"predicted_aadt_{year}"].to_numpy(dtype=float) for year in YEARS]
    )
    lower = max(float(np.quantile(all_values, 0.02)), 1.0)
    upper = float(np.quantile(all_values, 0.98))
    norm = LogNorm(vmin=lower, vmax=upper)
    limits = common_map_limits(backcast)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    collection = None
    for axis, year in zip(axes, YEARS):
        values = np.array(
            [lookup.loc[route_id, f"predicted_aadt_{year}"] for route_id in part_route_ids],
            dtype=float,
        )
        collection = LineCollection(
            line_parts,
            array=values,
            cmap="viridis",
            norm=norm,
            linewidths=0.42,
            alpha=0.9,
        )
        axis.add_collection(collection)
        style_map_axis(axis, limits)
        axis.set_title(str(year), fontsize=13)
    if collection is not None:
        colorbar = fig.colorbar(collection, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Predicted AADT (vehicles/day; log scale)")
    fig.suptitle("Preliminary Hong Kong road-segment AADT backcast", fontsize=16)
    fig.text(
        0.5,
        0.04,
        "Frozen Step 9 model fitted to all 679 stations; common colour scale clipped to network p2–p98. "
        "Do not interpret differences between panels as identified temporal change.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.10, top=0.88, wspace=0.02)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BACKCAST_MAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {BACKCAST_MAP_PATH.relative_to(PROJECT_ROOT)}")


def plot_change_maps(
    backcast: pd.DataFrame,
    line_parts: list[np.ndarray],
    part_route_ids: list[str],
) -> None:
    periods = ((2011, 2016), (2016, 2021))
    lookup = backcast.set_index("route_id")
    all_changes = np.concatenate(
        [
            backcast[f"predicted_change_{start}_{end}"].to_numpy(dtype=float)
            for start, end in periods
        ]
    )
    limit = max(float(np.quantile(np.abs(all_changes), 0.98)), 1.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    limits = common_map_limits(backcast)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    collection = None
    for axis, (start, end) in zip(axes, periods):
        values = np.array(
            [
                lookup.loc[route_id, f"predicted_change_{start}_{end}"]
                for route_id in part_route_ids
            ],
            dtype=float,
        )
        collection = LineCollection(
            line_parts,
            array=values,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.45,
            alpha=0.9,
        )
        axis.add_collection(collection)
        style_map_axis(axis, limits)
        axis.set_title(f"{start}–{end}", fontsize=13)
    if collection is not None:
        colorbar = fig.colorbar(collection, ax=axes, fraction=0.03, pad=0.02)
        colorbar.set_label("Predicted AADT change (vehicles/day)")
    fig.suptitle(
        "Exploratory road-segment change surface — not reportable as trend",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.04,
        "Red indicates increase and blue indicates decrease; common scale clipped to absolute p98. "
        "Step 15 refit and cross-year diagnostics are required before interpretation.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.89, bottom=0.10, top=0.88, wspace=0.02)
    fig.savefig(CHANGE_MAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CHANGE_MAP_PATH.relative_to(PROJECT_ROOT)}")


def plot_spatial_support_map(
    backcast: pd.DataFrame,
    training: pd.DataFrame,
    line_parts: list[np.ndarray],
    part_route_ids: list[str],
) -> None:
    lookup = backcast.set_index("route_id")
    values = np.array(
        [
            lookup.loc[route_id, "nearest_training_route_centroid_km"]
            for route_id in part_route_ids
        ],
        dtype=float,
    )
    upper = max(float(backcast["nearest_training_route_centroid_km"].quantile(0.95)), 0.1)
    norm = Normalize(vmin=0, vmax=upper)
    limits = common_map_limits(backcast)
    fig, axis = plt.subplots(figsize=(8.2, 6.2))
    collection = LineCollection(
        line_parts,
        array=values,
        cmap="magma_r",
        norm=norm,
        linewidths=0.5,
        alpha=0.9,
    )
    axis.add_collection(collection)
    axis.scatter(
        training["centroid_longitude"],
        training["centroid_latitude"],
        s=3,
        color="#00B8D9",
        alpha=0.75,
        linewidths=0,
        label="679 training stations",
    )
    style_map_axis(axis, limits)
    axis.legend(frameon=False, loc="lower left", fontsize=8)
    colorbar = fig.colorbar(collection, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Distance to nearest training-route centroid (km)")
    axis.set_title("Spatial support for full-network backcasting", fontsize=15)
    fig.text(
        0.5,
        0.04,
        "Distance scale clipped at network p95; this is a descriptive support screen, not a prediction interval.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.09, top=0.92)
    fig.savefig(SUPPORT_MAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SUPPORT_MAP_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    network, training, oof, step9_summary, calibration = read_inputs()
    validate_inputs(network, training, oof)

    backcast = add_support_fields(network, training)
    backcast = add_observed_anchor_fields(backcast, training)
    backcast = run_full_network_backcast(backcast, training)

    prediction_columns = [f"predicted_aadt_{year}" for year in YEARS]
    if backcast[prediction_columns].isna().any().any():
        raise ValueError("Full-network backcast contains missing predictions.")
    if (backcast[prediction_columns] <= 0).any().any():
        raise ValueError("Full-network backcast contains nonpositive predictions.")

    summary_rows = build_backcast_summary(
        backcast,
        training,
        oof,
        step9_summary,
        calibration,
    )
    change_rows = build_change_summary(backcast)
    support_rows = build_support_audit(backcast)
    decision_rows = build_decision_audit(backcast, training)

    BACKCAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    backcast.to_csv(BACKCAST_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {BACKCAST_PATH.relative_to(PROJECT_ROOT)}")
    write_csv(SUMMARY_PATH, summary_rows)
    write_csv(CHANGE_SUMMARY_PATH, change_rows)
    write_csv(SUPPORT_AUDIT_PATH, support_rows)
    write_csv(DECISION_AUDIT_PATH, decision_rows)

    line_parts, part_route_ids = write_backcast_geojson_and_collect_lines(backcast)
    plot_backcast_maps(backcast, line_parts, part_route_ids)
    plot_change_maps(backcast, line_parts, part_route_ids)
    plot_spatial_support_map(backcast, training, line_parts, part_route_ids)

    support_counts = Counter(backcast["spatial_support_tier"])
    print("\nStep 11 full-network backcasting is complete.")
    print(f"Current-network segments predicted: {len(backcast)}")
    print(f"Observed training stations: {len(training)}")
    print(f"Direct observed anchor routes: {(backcast['direct_observed_station_count'] > 0).sum()}")
    print(f"Spatial support tiers: {dict(support_counts)}")
    print(
        "Decision signal: proceed_to_temporal_identifiability_and_network_support_audits_before_equity_use"
    )
    print(
        "Interpretation rule: predictions are raw full-fit cross-sectional surfaces; "
        "Step 9 OOF metrics remain the validation evidence, and panel differences are "
        "not identified temporal change."
    )


if __name__ == "__main__":
    main()
