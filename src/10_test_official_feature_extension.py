from __future__ import annotations

import csv
import html
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "road_network_v2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

NETWORK_PATH = PROCESSED_DIR / "atc_network_segment_features.csv"
TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
FOLD_PATH = PROCESSED_DIR / "atc_spatial_validation_folds.csv"

ENRICHED_NETWORK_PATH = (
    PROCESSED_DIR / "atc_network_segment_features_official_extended.csv"
)
ENRICHED_TRAINING_PATH = (
    PROCESSED_DIR / "atc_high_confidence_training_official_extended.csv"
)
COVERAGE_PATH = TABLE_DIR / "step10_official_feature_coverage.csv"
METRIC_PATH = TABLE_DIR / "step10_feature_extension_metrics_by_fold.csv"
SUMMARY_PATH = TABLE_DIR / "step10_feature_extension_summary.csv"
CALIBRATION_PATH = TABLE_DIR / "step10_feature_extension_calibration.csv"
DECISION_PATH = TABLE_DIR / "step10_feature_extension_decision_audit.csv"
MAE_FIGURE_PATH = FIGURE_DIR / "step10_feature_extension_mae.png"
BIAS_FIGURE_PATH = FIGURE_DIR / "step10_tail_bias_comparison.png"

ROAD_NETWORK_BASE_URL = "https://static.data.gov.hk/td/road-network-v2"
LAYER_NAMES = (
    "SPEED_LIMIT",
    "INTERSECTION",
    "ROUNDABOUT",
    "BUS_ONLY_LANE",
    "TRAFFIC_FEATURES",
    "VEHICLE_RESTRICTION",
    "PERMIT",
)
LAYER_URLS = {
    name: f"{ROAD_NETWORK_BASE_URL}/{name}.kmz" for name in LAYER_NAMES
}

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
PLACEMARK_TAG = f"{{{KML_NAMESPACE}}}Placemark"
DESCRIPTION_TAG = f"{{{KML_NAMESPACE}}}description"

YEARS = (2011, 2016, 2021)
FOLDS = (1, 2, 3, 4, 5)
MODEL_ORDER = ("base_step9_features", "official_feature_extension")
MODEL_LABELS = {
    "base_step9_features": "Step 9 features",
    "official_feature_extension": "Official feature extension",
}
MODEL_COLORS = {
    "base_step9_features": "#7F8C8D",
    "official_feature_extension": "#D35400",
}

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

SOURCE_BY_FEATURE = {
    "speed_limit_min_kmh": "SPEED_LIMIT.kmz_or_default_50",
    "speed_limit_max_kmh": "SPEED_LIMIT.kmz_or_default_50",
    "speed_limit_spread_kmh": "SPEED_LIMIT.kmz",
    "speed_limit_override_present": "SPEED_LIMIT.kmz",
    "official_intersection_count": "INTERSECTION.kmz",
    "signalized_intersection_count": "INTERSECTION.kmz",
    "intersection_max_road_count": "INTERSECTION.kmz",
    "roundabout_count": "ROUNDABOUT.kmz",
    "roundabout_max_arm_count": "ROUNDABOUT.kmz",
    "bus_only_lane_count": "BUS_ONLY_LANE.kmz",
    "vehicle_restriction_count": "VEHICLE_RESTRICTION.kmz",
    "permit_count": "PERMIT.kmz",
    "zebra_crossing_count": "TRAFFIC_FEATURES.kmz",
    "yellow_box_count": "TRAFFIC_FEATURES.kmz",
    "toll_feature_count": "TRAFFIC_FEATURES.kmz",
    "cul_de_sac_count": "TRAFFIC_FEATURES.kmz",
    "street_code_gazetted_named": "CENTERLINE_ST_CODE_data_dictionary",
    "street_code_ungazetted_named": "CENTERLINE_ST_CODE_data_dictionary",
    "street_code_unnamed": "CENTERLINE_ST_CODE_data_dictionary",
    "name_expressway": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_highway": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_corridor": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_bypass": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_tunnel": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_bridge": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    "name_flyover": "CENTERLINE_STREET_ENAME_and_ALIAS_ENAME",
    **{
        f"strategic_route_{number}": "CENTERLINE_ROUTE_NUM"
        for number in range(1, 11)
    },
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


def download_layer(name: str) -> Path:
    destination = RAW_DIR / f"{name}.kmz"
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already available: {destination.relative_to(PROJECT_ROOT)}")
        return destination
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {name}.kmz")
    request = Request(
        LAYER_URLS[name],
        headers={"User-Agent": "HK-AADT-research-pilot/1.0"},
    )
    with urlopen(request, timeout=180) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    print(f"Saved: {destination.relative_to(PROJECT_ROOT)}")
    return destination


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


def kml_document_name(archive: ZipFile) -> str:
    names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
    if not names:
        raise ValueError("No KML document in archive.")
    return names[0]


def stream_layer_fields(path: Path):
    with ZipFile(path) as archive:
        with archive.open(kml_document_name(archive)) as kml_file:
            for _, element in ET.iterparse(kml_file, events=("end",)):
                if element.tag == PLACEMARK_TAG:
                    yield description_fields(
                        element.findtext(DESCRIPTION_TAG, default="")
                    )
                    element.clear()


def normalize_route_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def numeric_value(value: object, default: float = 0.0) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def referenced_routes(fields: dict[str, str]) -> set[str]:
    return {
        normalize_route_id(fields.get(f"RD_ID_{index}", ""))
        for index in range(1, 11)
        if normalize_route_id(fields.get(f"RD_ID_{index}", ""))
    }


def collect_official_layer_features(
    valid_route_ids: set[str],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    speed_by_route: dict[str, list[int]] = defaultdict(list)
    intersection_count: Counter[str] = Counter()
    signalized_count: Counter[str] = Counter()
    intersection_max_roads: Counter[str] = Counter()
    roundabout_count: Counter[str] = Counter()
    roundabout_max_arms: Counter[str] = Counter()
    bus_lane_count: Counter[str] = Counter()
    vehicle_restriction_count: Counter[str] = Counter()
    permit_count: Counter[str] = Counter()
    traffic_counts: dict[str, Counter[str]] = {
        "1": Counter(),
        "2": Counter(),
        "3": Counter(),
        "4": Counter(),
    }
    layer_audit: list[dict[str, object]] = []

    for layer_name in LAYER_NAMES:
        path = download_layer(layer_name)
        row_count = 0
        referenced_ids: set[str] = set()

        for fields in stream_layer_fields(path):
            row_count += 1
            if layer_name == "SPEED_LIMIT":
                route_id = normalize_route_id(fields.get("ROAD_ROUTE_ID", ""))
                match = re.search(r"\d+", fields.get("SPEED_LIMIT", ""))
                if route_id and match:
                    referenced_ids.add(route_id)
                    speed_by_route[route_id].append(int(match.group()))
            elif layer_name in {
                "BUS_ONLY_LANE",
                "VEHICLE_RESTRICTION",
                "PERMIT",
            }:
                route_id = normalize_route_id(fields.get("ROAD_ROUTE_ID", ""))
                if not route_id:
                    continue
                referenced_ids.add(route_id)
                target_counter = {
                    "BUS_ONLY_LANE": bus_lane_count,
                    "VEHICLE_RESTRICTION": vehicle_restriction_count,
                    "PERMIT": permit_count,
                }[layer_name]
                target_counter[route_id] += 1
            else:
                route_ids = referenced_routes(fields)
                referenced_ids.update(route_ids)
                if layer_name == "INTERSECTION":
                    road_count = len(route_ids)
                    signalized = fields.get("INT_TYPE", "") == "1"
                    for route_id in route_ids:
                        intersection_count[route_id] += 1
                        intersection_max_roads[route_id] = max(
                            intersection_max_roads[route_id],
                            road_count,
                        )
                        if signalized:
                            signalized_count[route_id] += 1
                elif layer_name == "ROUNDABOUT":
                    arm_count = int(numeric_value(fields.get("NO_OF_ARM", ""), 0))
                    for route_id in route_ids:
                        roundabout_count[route_id] += 1
                        roundabout_max_arms[route_id] = max(
                            roundabout_max_arms[route_id],
                            arm_count,
                        )
                elif layer_name == "TRAFFIC_FEATURES":
                    feature_type = fields.get("FEATURE_TYPE", "") or fields.get(
                        "FEATURE TYPE", ""
                    )
                    if feature_type in traffic_counts:
                        for route_id in route_ids:
                            traffic_counts[feature_type][route_id] += 1

        overlap_count = len(referenced_ids & valid_route_ids)
        layer_audit.append(
            {
                "layer": layer_name,
                "feature_rows": row_count,
                "unique_referenced_route_ids": len(referenced_ids),
                "referenced_route_ids_on_tfc_support": overlap_count,
                "route_id_overlap_pct": round(
                    100 * overlap_count / len(referenced_ids), 3
                )
                if referenced_ids
                else "",
                "source_url": LAYER_URLS[layer_name],
                "interpretation": "current_official_feature_not_historical_capacity_proof",
            }
        )

    enrichment: dict[str, dict[str, object]] = {}
    for route_id in valid_route_ids:
        speeds = speed_by_route.get(route_id, [])
        speed_min = min(speeds) if speeds else 50
        speed_max = max(speeds) if speeds else 50
        enrichment[route_id] = {
            "speed_limit_min_kmh": speed_min,
            "speed_limit_max_kmh": speed_max,
            "speed_limit_spread_kmh": speed_max - speed_min,
            "speed_limit_override_present": bool(speeds),
            "official_intersection_count": intersection_count[route_id],
            "signalized_intersection_count": signalized_count[route_id],
            "intersection_max_road_count": intersection_max_roads[route_id],
            "roundabout_count": roundabout_count[route_id],
            "roundabout_max_arm_count": roundabout_max_arms[route_id],
            "bus_only_lane_count": bus_lane_count[route_id],
            "vehicle_restriction_count": vehicle_restriction_count[route_id],
            "permit_count": permit_count[route_id],
            "zebra_crossing_count": traffic_counts["1"][route_id],
            "yellow_box_count": traffic_counts["2"][route_id],
            "toll_feature_count": traffic_counts["3"][route_id],
            "cul_de_sac_count": traffic_counts["4"][route_id],
        }
    return enrichment, layer_audit


def street_code_indicators(value: object) -> dict[str, bool]:
    code = int(numeric_value(value, 0))
    return {
        "street_code_gazetted_named": 10001 <= code <= 29999,
        "street_code_ungazetted_named": 30001 <= code <= 39999,
        "street_code_unnamed": 40001 <= code <= 59999,
    }


def controlled_name_indicators(row: dict[str, object]) -> dict[str, bool]:
    text = " ".join(
        str(row.get(field, ""))
        for field in ("street_ename", "alias_ename")
        if str(row.get(field, "")).casefold() != "nan"
    ).upper()
    return {
        "name_expressway": "EXPRESSWAY" in text,
        "name_highway": "HIGHWAY" in text,
        "name_corridor": "CORRIDOR" in text,
        "name_bypass": "BYPASS" in text,
        "name_tunnel": "TUNNEL" in text,
        "name_bridge": "BRIDGE" in text,
        "name_flyover": "FLYOVER" in text,
    }


def strategic_route_indicators(value: object) -> dict[str, bool]:
    route_number = int(numeric_value(value, 0))
    return {
        f"strategic_route_{number}": route_number == number
        for number in range(1, 11)
    }


def build_enriched_tables(
    network: pd.DataFrame,
    training: pd.DataFrame,
    enrichment: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_rows: list[dict[str, object]] = []
    for source_row in network.to_dict(orient="records"):
        route_id = normalize_route_id(source_row["route_id"])
        row = {
            **source_row,
            **enrichment[route_id],
            **street_code_indicators(source_row.get("street_code", "")),
            **controlled_name_indicators(source_row),
            **strategic_route_indicators(source_row.get("route_number", "")),
            "official_feature_snapshot": "current_monthly_road_network_v2",
            "historical_capacity_status": "not_proven",
        }
        new_rows.append(row)
    enriched_network = pd.DataFrame(new_rows)

    extension = enriched_network[["route_id", *EXTENDED_FEATURES]].copy()
    extension["route_id"] = extension["route_id"].map(normalize_route_id)
    training = training.copy()
    training_row_count = len(training)
    training["selected_route_id"] = training["selected_route_id"].map(
        normalize_route_id
    )
    enriched_training = training.merge(
        extension,
        how="left",
        left_on="selected_route_id",
        right_on="route_id",
        suffixes=("", "_official_extension"),
        validate="many_to_one",
    )
    if len(enriched_training) != training_row_count:
        raise ValueError("Official feature join changed the number of training rows.")
    enriched_training["official_feature_snapshot"] = (
        "current_monthly_road_network_v2"
    )
    enriched_training["historical_capacity_status"] = "not_proven"
    return enriched_network, enriched_training


def feature_coverage_rows(
    enriched_network: pd.DataFrame,
    enriched_training: pd.DataFrame,
    layer_audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = [
        {
            "record_type": "layer_route_id_audit",
            "feature_or_layer": row["layer"],
            "network_nondefault_count": row["referenced_route_ids_on_tfc_support"],
            "network_nondefault_pct": row["route_id_overlap_pct"],
            "training_nondefault_count": "",
            "training_nondefault_pct": "",
            "source": row["source_url"],
            "decision": row["interpretation"],
        }
        for row in layer_audit
    ]
    for feature in EXTENDED_FEATURES:
        network_values = pd.to_numeric(
            enriched_network[feature], errors="coerce"
        ).fillna(0)
        training_values = pd.to_numeric(
            enriched_training[feature], errors="coerce"
        ).fillna(0)
        if feature in {"speed_limit_min_kmh", "speed_limit_max_kmh"}:
            network_mask = network_values != 50
            training_mask = training_values != 50
        else:
            network_mask = network_values != 0
            training_mask = training_values != 0
        rows.append(
            {
                "record_type": "feature_coverage",
                "feature_or_layer": feature,
                "network_nondefault_count": int(network_mask.sum()),
                "network_nondefault_pct": round(100 * network_mask.mean(), 3),
                "training_nondefault_count": int(training_mask.sum()),
                "training_nondefault_pct": round(100 * training_mask.mean(), 3),
                "source": SOURCE_BY_FEATURE[feature],
                "decision": "current_feature_for_diagnostic_extension",
            }
        )
    return rows


def model_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame[features].copy()
    for column in values:
        if values[column].dtype == bool:
            values[column] = values[column].astype(int)
    return values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def fixed_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def metric_row(
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
        "r2": round(r2_score(observed, predicted), 6),
    }


def run_extension_comparison(
    training: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    features_by_model = {
        "base_step9_features": BASE_FEATURES,
        "official_feature_extension": [*BASE_FEATURES, *EXTENDED_FEATURES],
    }

    for year in YEARS:
        target = f"aadt_{year}"
        for fold in FOLDS:
            train = training[training["spatial_fold"] != fold]
            test = training[training["spatial_fold"] == fold]
            y_train = train[target].to_numpy(dtype=float)
            y_test = test[target].to_numpy(dtype=float)
            for model_name, features in features_by_model.items():
                model = fixed_model()
                model.fit(model_matrix(train, features), y_train)
                predicted = model.predict(model_matrix(test, features))
                metric_rows.append(
                    metric_row(year, fold, model_name, y_test, predicted)
                )
                for index, station_id in enumerate(test["station_id"]):
                    prediction_rows.append(
                        {
                            "station_id": int(station_id),
                            "year": year,
                            "spatial_fold": fold,
                            "model": model_name,
                            "observed_aadt": round(float(y_test[index]), 4),
                            "predicted_aadt": round(float(predicted[index]), 4),
                            "error": round(float(predicted[index] - y_test[index]), 4),
                            "absolute_error": round(
                                abs(float(predicted[index] - y_test[index])), 4
                            ),
                        }
                    )
    return prediction_rows, metric_rows


def build_summary(
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    rows: list[dict[str, object]] = []
    for year in YEARS:
        base = predictions[
            (predictions["year"] == year)
            & (predictions["model"] == "base_step9_features")
        ]
        base_mae = mean_absolute_error(base["observed_aadt"], base["predicted_aadt"])
        base_rmse = math.sqrt(
            mean_squared_error(base["observed_aadt"], base["predicted_aadt"])
        )
        for model_name in MODEL_ORDER:
            pool = predictions[
                (predictions["year"] == year)
                & (predictions["model"] == model_name)
            ]
            fold_metrics = metrics[
                (metrics["year"] == year) & (metrics["model"] == model_name)
            ]
            mae = mean_absolute_error(pool["observed_aadt"], pool["predicted_aadt"])
            rmse = math.sqrt(
                mean_squared_error(pool["observed_aadt"], pool["predicted_aadt"])
            )
            rows.append(
                {
                    "year": year,
                    "model": model_name,
                    "n_oof": len(pool),
                    "pooled_mae": round(mae, 4),
                    "pooled_rmse": round(rmse, 4),
                    "pooled_r2": round(
                        r2_score(pool["observed_aadt"], pool["predicted_aadt"]), 6
                    ),
                    "mean_fold_mae": round(fold_metrics["mae"].mean(), 4),
                    "sd_fold_mae": round(fold_metrics["mae"].std(ddof=1), 4),
                    "mae_improvement_vs_step9_pct": round(
                        100 * (base_mae - mae) / base_mae, 3
                    ),
                    "rmse_improvement_vs_step9_pct": round(
                        100 * (base_rmse - rmse) / base_rmse, 3
                    ),
                }
            )
    return rows


def build_calibration(
    prediction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    labels = ["Q1_low", "Q2", "Q3", "Q4_high"]
    rows: list[dict[str, object]] = []
    for year in YEARS:
        for model_name in MODEL_ORDER:
            group = predictions[
                (predictions["year"] == year)
                & (predictions["model"] == model_name)
            ].copy()
            group["observed_quartile"] = pd.qcut(
                group["observed_aadt"], 4, labels=labels
            )
            for label in labels:
                quartile = group[group["observed_quartile"] == label]
                observed_mean = float(quartile["observed_aadt"].mean())
                predicted_mean = float(quartile["predicted_aadt"].mean())
                bias = predicted_mean - observed_mean
                rows.append(
                    {
                        "year": year,
                        "model": model_name,
                        "observed_quartile": label,
                        "n": len(quartile),
                        "observed_mean": round(observed_mean, 4),
                        "predicted_mean": round(predicted_mean, 4),
                        "mean_bias": round(bias, 4),
                        "mean_bias_pct_of_observed": round(
                            100 * bias / observed_mean, 3
                        ),
                        "mae": round(float(quartile["absolute_error"].mean()), 4),
                    }
                )
    return rows


def build_decision_audit(
    summary_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary = pd.DataFrame(summary_rows)
    calibration = pd.DataFrame(calibration_rows)
    mae_improved_years = 0
    rmse_improved_years = 0
    r2_improved_years = 0
    q4_improved_years = 0
    q1_improved_years = 0
    no_material_mae_degradation = True

    for year in YEARS:
        year_summary = summary[summary["year"] == year].set_index("model")
        base_mae = year_summary.loc["base_step9_features", "pooled_mae"]
        extended_mae = year_summary.loc[
            "official_feature_extension", "pooled_mae"
        ]
        if extended_mae < base_mae:
            mae_improved_years += 1
        if (
            year_summary.loc["official_feature_extension", "pooled_rmse"]
            < year_summary.loc["base_step9_features", "pooled_rmse"]
        ):
            rmse_improved_years += 1
        if (
            year_summary.loc["official_feature_extension", "pooled_r2"]
            > year_summary.loc["base_step9_features", "pooled_r2"]
        ):
            r2_improved_years += 1
        if extended_mae > base_mae * 1.02:
            no_material_mae_degradation = False

        year_calibration = calibration[calibration["year"] == year]
        base = year_calibration[
            year_calibration["model"] == "base_step9_features"
        ].set_index("observed_quartile")
        extended = year_calibration[
            year_calibration["model"] == "official_feature_extension"
        ].set_index("observed_quartile")
        if (
            abs(extended.loc["Q4_high", "mean_bias_pct_of_observed"])
            <= abs(base.loc["Q4_high", "mean_bias_pct_of_observed"]) - 5
        ):
            q4_improved_years += 1
        if (
            abs(extended.loc["Q1_low", "mean_bias_pct_of_observed"])
            <= abs(base.loc["Q1_low", "mean_bias_pct_of_observed"]) - 5
        ):
            q1_improved_years += 1

    if (
        no_material_mae_degradation
        and rmse_improved_years >= 2
        and r2_improved_years >= 2
    ):
        signal = "mixed_bundle_signal_reopen_individual_features_under_nested_selection"
    else:
        signal = "full_bundle_not_promoted_reassess_individual_features_under_nested_selection"

    layer_rows = [row for row in coverage_rows if row["record_type"] == "layer_route_id_audit"]
    minimum_overlap = min(float(row["network_nondefault_pct"]) for row in layer_rows)
    return [
        {"metric": "development_comparison_not_final_test", "count": 1, "value": "same_regional_folds_as_step9", "decision": "freeze_alternative_validation_after_feature_choice"},
        {"metric": "outer_fold_hyperparameter_tuning", "count": 0, "value": "", "decision": "same_model_parameters_as_step9"},
        {"metric": "official_extension_feature_count", "count": len(EXTENDED_FEATURES), "value": "", "decision": "fixed_bundle"},
        {"metric": "minimum_layer_route_id_overlap_pct", "count": "", "value": round(minimum_overlap, 3), "decision": "audit_cross_dataset_route_id_compatibility"},
        {"metric": "years_with_lower_pooled_mae", "count": mae_improved_years, "value": "", "decision": "diagnostic"},
        {"metric": "years_with_lower_pooled_rmse", "count": rmse_improved_years, "value": "", "decision": "diagnostic"},
        {"metric": "years_with_higher_pooled_r2", "count": r2_improved_years, "value": "", "decision": "diagnostic"},
        {"metric": "years_with_q1_observed_bin_bias_improved_at_least_5pp", "count": q1_improved_years, "value": "", "decision": "diagnostic_only_conditioned_on_observed_outcome"},
        {"metric": "years_with_q4_observed_bin_bias_improved_at_least_5pp", "count": q4_improved_years, "value": "", "decision": "diagnostic_only_not_an_acceptance_criterion"},
        {"metric": "no_year_mae_degraded_more_than_2pct", "count": int(no_material_mae_degradation), "value": "", "decision": "guardrail"},
        {"metric": "step10_decision_signal", "count": "", "value": signal, "decision": "fixed_reproducible_development_rule"},
        {"metric": "historical_capacity_interpretation", "count": "", "value": "current_features_only", "decision": "not_proof_of_historical_road_configuration"},
    ]


def plot_mae(summary_rows: list[dict[str, object]]) -> None:
    summary = pd.DataFrame(summary_rows)
    positions = np.arange(len(YEARS))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9.2, 5.3))
    for model_index, model_name in enumerate(MODEL_ORDER):
        values = [
            summary[
                (summary["year"] == year) & (summary["model"] == model_name)
            ]["pooled_mae"].iloc[0]
            for year in YEARS
        ]
        offset = (model_index - 0.5) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axis.set_xticks(positions, [str(year) for year in YEARS])
    axis.set_ylabel("Pooled regional OOF MAE (vehicles/day)")
    axis.set_title("Official road-feature extension: development comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {MAE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_observed_bin_residuals(calibration_rows: list[dict[str, object]]) -> None:
    calibration = pd.DataFrame(calibration_rows)
    positions = np.arange(len(YEARS))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for axis, quartile, title in zip(
        axes,
        ("Q1_low", "Q4_high"),
        ("Lowest observed-AADT quartile", "Highest observed-AADT quartile"),
    ):
        for model_index, model_name in enumerate(MODEL_ORDER):
            values = [
                calibration[
                    (calibration["year"] == year)
                    & (calibration["model"] == model_name)
                    & (calibration["observed_quartile"] == quartile)
                ]["mean_bias_pct_of_observed"].iloc[0]
                for year in YEARS
            ]
            offset = (model_index - 0.5) * width
            bars = axis.bar(
                positions + offset,
                values,
                width,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(positions, [str(year) for year in YEARS])
        axis.set_title(title)
        axis.set_ylabel("Mean bias as % of observed mean")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Observed-value-bin residual pattern (diagnostic only)",
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BIAS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {BIAS_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    for required in (NETWORK_PATH, TRAINING_PATH, FOLD_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required.relative_to(PROJECT_ROOT)}")

    network = pd.read_csv(NETWORK_PATH, dtype={"route_id": str})
    training = pd.read_csv(TRAINING_PATH, dtype={"selected_route_id": str})
    valid_route_ids = {normalize_route_id(value) for value in network["route_id"]}
    enrichment, layer_audit = collect_official_layer_features(valid_route_ids)
    enriched_network, enriched_training = build_enriched_tables(
        network,
        training,
        enrichment,
    )
    if enriched_training[EXTENDED_FEATURES].isna().any().any():
        raise ValueError("Official feature join left missing training values.")

    coverage_rows = feature_coverage_rows(
        enriched_network,
        enriched_training,
        layer_audit,
    )
    print("Running fixed Step 9 model with and without the official feature bundle...")
    prediction_rows, metric_rows = run_extension_comparison(enriched_training)
    summary_rows = build_summary(prediction_rows, metric_rows)
    calibration_rows = build_calibration(prediction_rows)
    decision_rows = build_decision_audit(
        summary_rows,
        calibration_rows,
        coverage_rows,
    )

    ENRICHED_NETWORK_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched_network.to_csv(ENRICHED_NETWORK_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {ENRICHED_NETWORK_PATH.relative_to(PROJECT_ROOT)}")
    enriched_training.to_csv(
        ENRICHED_TRAINING_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved: {ENRICHED_TRAINING_PATH.relative_to(PROJECT_ROOT)}")
    write_csv(COVERAGE_PATH, coverage_rows)
    write_csv(METRIC_PATH, metric_rows)
    write_csv(SUMMARY_PATH, summary_rows)
    write_csv(CALIBRATION_PATH, calibration_rows)
    write_csv(DECISION_PATH, decision_rows)
    plot_mae(summary_rows)
    plot_observed_bin_residuals(calibration_rows)

    decision = next(
        row["value"] for row in decision_rows if row["metric"] == "step10_decision_signal"
    )
    print("\nStep 10 official feature-extension experiment is complete.")
    print(f"Current network segments enriched: {len(enriched_network)}")
    print(f"High-confidence training stations enriched: {len(enriched_training)}")
    print(f"Decision signal: {decision}")
    print(
        "Interpretation rule: this reuses the Step 9 regional folds as development "
        "evidence and is not a new independent final test. The full bundle is neither "
        "promoted nor rejected from observed-value-bin bias; individual features need "
        "fold-internal selection."
    )


if __name__ == "__main__":
    main()
