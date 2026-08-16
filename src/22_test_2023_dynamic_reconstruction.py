"""Step 22: test a deployable, bounded 2023 dynamic AADT reconstruction.

This experiment separates two deployment questions that must not share one
score:

1. sensor-assisted interpolation, where nearby 2023 traffic detectors are
   available at prediction time; and
2. sensor-free spatial extrapolation, where detectors assigned to the held-out
   region and detectors within one kilometre of held-out stations are removed.

The 2023 ATC station AADT values are outcomes only.  ATC-only road-network and
road-type labels, and station-to-road matching diagnostics, are excluded from
every deployable model.  They enter one explicitly labelled oracle diagnostic so
that their apparent contribution cannot be confused with full-network skill.
Niu-derived traffic and emission fields are not read by this script.

All deployable structural, GTFS and detector features are first generated on the
complete official 2023 centreline and only then joined to measured stations by
the matched centreline-segment identifier.  This makes the feature-availability
claim testable on all prediction segments instead of assuming that a
station-centred feature can be transferred to a road without a station.

The annual detector summaries use a predeclared bounded sample: the second
Tuesday and second Saturday of each month at 08:00, 13:00 and 18:00 for the
strategic detector feed, plus the corresponding daily vehicle-class files.  The
sample is a predictor experiment, not an estimate of official annual detector
AADT.
"""
from __future__ import annotations

import calendar
import io
import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from difflib import SequenceMatcher
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
from pyogrio.raw import read as read_ogr
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "step22_2023"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

MEASURED_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
STEP21_SUPPORT_PATH = PROCESSED_DIR / "atc_step21_2023_station_sensor_support.csv"
STEP21_DECISION_PATH = TABLE_DIR / "step21_decision_audit.csv"
STEP15_TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
STEP15_CORE_LONG_PATH = PROCESSED_DIR / "atc_core_panel_long.csv"
STEP15_SUMMARY_PATH = TABLE_DIR / "step15_honest_baseline_comparison.csv"

FEATURE_TABLE_PATH = PROCESSED_DIR / "atc_step22_2023_feature_table.csv"
NETWORK_FEATURE_TABLE_PATH = PROCESSED_DIR / "atc_step22_2023_network_feature_table.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step22_2023_oof_predictions.csv"
STRATEGIC_FEATURE_PATH = PROCESSED_DIR / "atc_step22_strategic_detector_features.csv"
CLASS_FEATURE_PATH = PROCESSED_DIR / "atc_step22_vehicle_class_features.csv"
ROAD_MATCH_PATH = PROCESSED_DIR / "atc_step22_2023_road_matches.csv"

SAMPLING_AUDIT_PATH = TABLE_DIR / "step22_temporal_sampling_audit.csv"
ROAD_AUDIT_PATH = TABLE_DIR / "step22_road_network_match_audit.csv"
GTFS_AUDIT_PATH = TABLE_DIR / "step22_gtfs_feature_audit.csv"
SENSOR_MASK_PATH = TABLE_DIR / "step22_sensor_mask_audit.csv"
METRICS_BY_FOLD_PATH = TABLE_DIR / "step22_metrics_by_fold.csv"
SUMMARY_PATH = TABLE_DIR / "step22_model_summary.csv"
SUBGROUP_PATH = TABLE_DIR / "step22_subgroup_bias.csv"
CALIBRATION_PATH = TABLE_DIR / "step22_predicted_bin_calibration.csv"
COMPARISON_PATH = TABLE_DIR / "step22_paired_model_comparison.csv"
FEATURE_MANIFEST_PATH = TABLE_DIR / "step22_feature_deployability_manifest.csv"
DEPLOYABILITY_PATH = TABLE_DIR / "step22_deployability_ablation.csv"
COMPARABILITY_PATH = TABLE_DIR / "step22_step15_comparability_audit.csv"
DECISION_PATH = TABLE_DIR / "step22_decision_audit.csv"

MODEL_FIGURE_PATH = FIGURE_DIR / "step22_2023_model_comparison.png"
SCATTER_FIGURE_PATH = FIGURE_DIR / "step22_2023_observed_vs_predicted.png"
FOLD_FIGURE_PATH = FIGURE_DIR / "step22_2023_dynamic_gain_by_fold.png"
BIAS_FIGURE_PATH = FIGURE_DIR / "step22_2023_subgroup_bias.png"

ARCHIVE_LIST_ROOT = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
ARCHIVE_GET_ROOT = "https://app.data.gov.hk/v1/historical-archive/get-file"
STRATEGIC_RAW_URL = (
    "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
)
STRATEGIC_LOCATION_URL = (
    "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/"
    "traffic_speed_volume_occ_info.csv"
)
VEHICLE_CLASS_URL = (
    "https://resource.data.one.gov.hk/td/traffic-detectors/volByVClass-all.xml"
)
VEHICLE_CLASS_LOCATION_URL = (
    "https://static.data.gov.hk/td/traffic-atc-veh-class/info/"
    "traffic_prop_vehicle_class_info.csv"
)
GTFS_URL = "https://static.data.gov.hk/td/pt-headway-en/gtfs.zip"
ROAD_NETWORK_URL = "https://static.data.gov.hk/td/road-network-v2/RdNet_IRNP.gdb.zip"

YEAR = 2023
FOLDS = (1, 2, 3, 4, 5)
STRATEGIC_SAMPLE_HOURS = (800, 1300, 1800)
MASK_BUFFER_M = 1000.0
PROJECTED_CRS = "EPSG:2326"
GTFS_WEEKDAY = date(2023, 6, 19)
GTFS_SATURDAY = date(2023, 6, 24)

MODEL_ORDER = (
    "training_median",
    "hierarchy_lookup",
    "atc_class_oracle_hgb",
    "deployable_structural_hgb",
    "deployable_structural_gtfs_hgb",
    "deployable_sensor_assisted_hgb",
    "deployable_sensor_free_hgb",
)
MODEL_LABELS = {
    "training_median": "Training median",
    "hierarchy_lookup": "10-cell hierarchy lookup",
    "atc_class_oracle_hgb": "ATC-class oracle (not deployable)",
    "deployable_structural_hgb": "Deployable road structure",
    "deployable_structural_gtfs_hgb": "Deployable structure + GTFS",
    "deployable_sensor_assisted_hgb": "Deployable sensor assisted",
    "deployable_sensor_free_hgb": "Deployable sensor free (masked)",
}
MODEL_COLORS = {
    "training_median": "#9E9E9E",
    "hierarchy_lookup": "#6C757D",
    "atc_class_oracle_hgb": "#9C755F",
    "deployable_structural_hgb": "#4C78A8",
    "deployable_structural_gtfs_hgb": "#72A0C1",
    "deployable_sensor_assisted_hgb": "#E07A1F",
    "deployable_sensor_free_hgb": "#1B9E77",
}

GENERIC_ROAD_TOKENS = {
    "ROAD",
    "STREET",
    "AVENUE",
    "HIGHWAY",
    "FLYOVER",
    "BRIDGE",
    "TUNNEL",
    "BYPASS",
    "DRIVE",
    "LANE",
    "PATH",
    "WAY",
    "NEAR",
    "EASTBOUND",
    "WESTBOUND",
    "NORTHBOUND",
    "SOUTHBOUND",
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty result: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def update_report_manifest() -> None:
    replacement_rows = [
        (MODEL_FIGURE_PATH, "reportable_deployability_comparison", "separates_the_atc_class_oracle_from_models_using_predictors_generated_on_every_centreline_segment"),
        (SCATTER_FIGURE_PATH, "reportable_calibration_diagnostic", "compares_only_deployable_structure_sensor_assisted_and_sensor_free_tasks"),
        (FOLD_FIGURE_PATH, "reportable_transportability_diagnostic", "tests_whether_detector_gain_repeats_across_spatial_folds_after_masking"),
        (BIAS_FIGURE_PATH, "reportable_bias_gate", "reports_region_and_major_minor_bias_for_deployable_models"),
        (FEATURE_MANIFEST_PATH, "reportable_feature_lineage_audit", "records_full_network_availability_and_prohibits_atc_class_and_station_match_fields_from_deployable_models"),
        (DEPLOYABILITY_PATH, "reportable_deployability_ablation", "contrasts_the_non_deployable_atc_class_oracle_with_full_network_feature_sets"),
        (COMPARABILITY_PATH, "reportable_cross_step_scope_audit", "explains_why_step15_and_step22_skill_estimates_are_not_directly_interchangeable"),
        (SENSOR_MASK_PATH, "reportable_leakage_audit", "states_that_retained_share_counts_sensors_and_simulates_detector_sparse_extrapolation_around_held_out_atc_locations"),
        (METRICS_BY_FOLD_PATH, "reportable_spatial_validation", "contains_identical_fold_metrics_for_all_seven_model_variants"),
        (SUMMARY_PATH, "reportable_model_summary", "uses_only_full_network_predictors_for_deployment_claims"),
        (SUBGROUP_PATH, "reportable_bias_audit", "quantifies_region_major_minor_and_sensor_support_strata"),
        (CALIBRATION_PATH, "reportable_calibration_audit", "uses_prediction_conditioned_bins"),
        (COMPARISON_PATH, "reportable_increment_test", "separates_oracle_structure_gtfs_supported_sensor_and_masked_sensor_contributions"),
        (DECISION_PATH, "reportable_decision", "separates_effect_size_interval_and_fold_consistency_failures_and_distinguishes_structure_only_from_structure_plus_gtfs"),
        (NETWORK_FEATURE_TABLE_PATH, "analysis_input", "contains_structural_gtfs_and_unmasked_detector_features_for_every_2023_centreline_segment"),
        (FEATURE_TABLE_PATH, "analysis_input", "joins_station_outcomes_to_predictors_precomputed_on_matched_centreline_segments"),
        (PREDICTION_PATH, "analysis_input", "stores_seven_spatial_oof_prediction_sets_including_a_non_deployable_oracle"),
    ]
    existing = (
        pd.read_csv(REPORT_MANIFEST_PATH)
        if REPORT_MANIFEST_PATH.exists()
        else pd.DataFrame(columns=["artifact", "status", "reason"])
    )
    replacement_artifacts = {
        str(path.relative_to(PROJECT_ROOT)) for path, _, _ in replacement_rows
    }
    existing = existing[~existing["artifact"].isin(replacement_artifacts)]
    additions = pd.DataFrame(
        [
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "status": status,
                "reason": reason,
            }
            for path, status, reason in replacement_rows
        ]
    )
    save_csv(pd.concat([existing, additions], ignore_index=True), REPORT_MANIFEST_PATH)


def fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def archive_versions_for_day(resource_url: str, day: str) -> dict[str, object]:
    query = urllib.parse.urlencode({"url": resource_url, "start": day, "end": day})
    return fetch_json(f"{ARCHIVE_LIST_ROOT}?{query}")


def download_url(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return path
    with urllib.request.urlopen(url, timeout=240) as response, path.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    return path


def download_historical(resource_url: str, timestamp: str, path: Path) -> Path:
    query = urllib.parse.urlencode({"url": resource_url, "time": timestamp})
    return download_url(f"{ARCHIVE_GET_ROOT}?{query}", path)


def closest_timestamp(timestamps: list[str], target_day: str, target_hhmm: int) -> str:
    matching = [value for value in timestamps if value.startswith(target_day)]
    if not matching:
        raise RuntimeError(f"No archived version found for {target_day}")

    def distance(value: str) -> int:
        hhmm = int(value[-4:])
        value_minutes = (hhmm // 100) * 60 + hhmm % 100
        target_minutes = (target_hhmm // 100) * 60 + target_hhmm % 100
        return abs(value_minutes - target_minutes)

    return min(matching, key=distance)


def nth_weekday(year: int, month: int, weekday: int, occurrence: int = 2) -> date:
    days = [
        day
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if date(year, month, day).weekday() == weekday
    ]
    return date(year, month, days[occurrence - 1])


def sampling_days() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        rows.append(
            {
                "sample_date": nth_weekday(YEAR, month, calendar.TUESDAY),
                "day_type": "weekday",
            }
        )
        rows.append(
            {
                "sample_date": nth_weekday(YEAR, month, calendar.SATURDAY),
                "day_type": "weekend",
            }
        )
    return rows


def obtain_strategic_day(specification: dict[str, object]) -> list[dict[str, object]]:
    sample_date = specification["sample_date"]
    day_text = sample_date.strftime("%Y%m%d")
    payload = archive_versions_for_day(STRATEGIC_RAW_URL, day_text)
    rows: list[dict[str, object]] = []
    for requested_hour in STRATEGIC_SAMPLE_HOURS:
        timestamp = closest_timestamp(payload.get("timestamps", []), day_text, requested_hour)
        path = download_historical(
            STRATEGIC_RAW_URL,
            timestamp,
            RAW_DIR / "strategic" / f"strategic_{timestamp}.xml",
        )
        rows.append(
            {
                "source": "strategic_detector",
                "sample_date": sample_date.isoformat(),
                "day_type": specification["day_type"],
                "requested_hour": requested_hour,
                "archive_timestamp": timestamp,
                "xml_source_date": ET.parse(path).getroot().findtext("date"),
                "local_path": str(path.relative_to(PROJECT_ROOT)),
                "download_size_mb": path.stat().st_size / 1e6,
            }
        )
    return rows


def obtain_vehicle_class_day(specification: dict[str, object]) -> list[dict[str, object]]:
    sample_date = specification["sample_date"]
    day_text = sample_date.strftime("%Y%m%d")
    payload = archive_versions_for_day(VEHICLE_CLASS_URL, day_text)
    timestamp = closest_timestamp(payload.get("timestamps", []), day_text, 1000)
    path = download_historical(
        VEHICLE_CLASS_URL,
        timestamp,
        RAW_DIR / "vehicle_class" / f"vehicle_class_{timestamp}.xml",
    )
    return [
        {
            "source": "vehicle_class_detector",
            "sample_date": sample_date.isoformat(),
            "day_type": specification["day_type"],
            "requested_hour": "full_day_profile",
            "archive_timestamp": timestamp,
            "xml_source_date": ET.parse(path).getroot().findtext("date"),
            "local_path": str(path.relative_to(PROJECT_ROOT)),
            "download_size_mb": path.stat().st_size / 1e6,
        }
    ]


def obtain_temporal_samples() -> pd.DataFrame:
    jobs: list[tuple[str, dict[str, object]]] = []
    for specification in sampling_days():
        jobs.append(("strategic", specification))
        jobs.append(("vehicle_class", specification))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                obtain_strategic_day if source == "strategic" else obtain_vehicle_class_day,
                specification,
            ): (source, specification)
            for source, specification in jobs
        }
        completed = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            completed += 1
            if completed % 8 == 0 or completed == len(futures):
                print(f"Downloaded/cached {completed} of {len(futures)} sampling-day jobs.")
    audit = pd.DataFrame(rows).sort_values(
        ["sample_date", "source", "requested_hour"]
    )
    return audit


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_strategic_samples(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = audit[audit["source"] == "strategic_detector"]
    for sample in samples.itertuples(index=False):
        root = ET.parse(PROJECT_ROOT / sample.local_path).getroot()
        per_detector: dict[str, list[dict[str, float]]] = defaultdict(list)
        for period in root.findall(".//period"):
            for detector in period.findall("./detectors/detector"):
                sensor_id = detector.findtext("detector_id", default="").strip()
                lane_values: list[tuple[float, float, float]] = []
                for lane in detector.findall("./lanes/lane"):
                    if lane.findtext("valid") != "Y":
                        continue
                    volume = safe_float(lane.findtext("volume"))
                    speed = safe_float(lane.findtext("speed"))
                    occupancy = safe_float(lane.findtext("occupancy"))
                    if np.isfinite(volume):
                        lane_values.append((volume, speed, occupancy))
                if not lane_values:
                    continue
                array = np.asarray(lane_values, dtype=float)
                weights = np.maximum(array[:, 0], 1.0)
                per_detector[sensor_id].append(
                    {
                        "minute_volume": float(np.nansum(array[:, 0])),
                        "speed": float(np.average(array[:, 1], weights=weights)),
                        "occupancy": float(np.nanmean(array[:, 2])),
                    }
                )
        for sensor_id, values in per_detector.items():
            frame = pd.DataFrame(values)
            requested_hour = int(sample.requested_hour)
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "sample_date": sample.sample_date,
                    "day_type": sample.day_type,
                    "time_block": (
                        "am" if requested_hour < 1000 else "midday" if requested_hour < 1600 else "pm"
                    ),
                    "sampled_minute_volume": frame["minute_volume"].mean(),
                    "sampled_speed": frame["speed"].mean(),
                    "sampled_occupancy": frame["occupancy"].mean(),
                }
            )
    long = pd.DataFrame(rows)
    expected_samples = len(samples)
    records: list[dict[str, object]] = []
    for sensor_id, group in long.groupby("sensor_id"):
        row: dict[str, object] = {
            "sensor_id": sensor_id,
            "strategic_sample_count": len(group),
            "strategic_sample_coverage": len(group) / expected_samples,
            "strategic_mean_minute_volume": group["sampled_minute_volume"].mean(),
            "strategic_sd_minute_volume": group["sampled_minute_volume"].std(),
            "strategic_p10_minute_volume": group["sampled_minute_volume"].quantile(0.10),
            "strategic_p90_minute_volume": group["sampled_minute_volume"].quantile(0.90),
            "strategic_mean_speed": group["sampled_speed"].mean(),
            "strategic_mean_occupancy": group["sampled_occupancy"].mean(),
        }
        for day_type in ("weekday", "weekend"):
            selected = group[group["day_type"] == day_type]
            row[f"strategic_{day_type}_mean_volume"] = selected[
                "sampled_minute_volume"
            ].mean()
        for time_block in ("am", "midday", "pm"):
            selected = group[group["time_block"] == time_block]
            row[f"strategic_{time_block}_mean_volume"] = selected[
                "sampled_minute_volume"
            ].mean()
        records.append(row)
    features = pd.DataFrame(records)
    save_csv(features, STRATEGIC_FEATURE_PATH)
    return features


def class_slug(value: str) -> str:
    value = value.lower().replace("s.d.", "sd").replace("d.d.", "dd")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def parse_vehicle_class_samples(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = audit[audit["source"] == "vehicle_class_detector"]
    for sample in samples.itertuples(index=False):
        root = ET.parse(PROJECT_ROOT / sample.local_path).getroot()
        full_day = None
        for period in root.findall(".//period"):
            if (
                period.findtext("period_from") == "00:00:00"
                and period.findtext("period_to") == "24:00:00"
            ):
                full_day = period
                break
        if full_day is None:
            raise RuntimeError(f"No full-day vehicle-class period in {sample.local_path}")
        for detector in full_day.findall("./detectors/detector"):
            if detector.findtext("valid") != "Y":
                continue
            sensor_id = detector.findtext("detector_id", default="").strip()
            row: dict[str, object] = {
                "sensor_id": sensor_id,
                "sample_date": sample.sample_date,
                "day_type": sample.day_type,
            }
            for entry in detector.findall("./vehicle_class/class"):
                name = entry.findtext("class_name", default="").strip()
                row[f"class_{class_slug(name)}_pct"] = safe_float(
                    entry.findtext("proportion")
                )
            rows.append(row)
    long = pd.DataFrame(rows)
    class_columns = sorted(column for column in long if column.startswith("class_"))
    features = long.groupby("sensor_id", as_index=False)[class_columns].mean()
    counts = long.groupby("sensor_id").size().rename("class_sample_count")
    features = features.merge(counts, on="sensor_id", how="left")
    features["class_sample_coverage"] = features["class_sample_count"] / len(samples)
    save_csv(features, CLASS_FEATURE_PATH)
    return features


def normalise_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return " ".join(
        token for token in text.split() if token not in GENERIC_ROAD_TOKENS
    )


def name_similarity(left: object, right: object) -> float:
    left_name = normalise_name(left)
    right_name = normalise_name(right)
    if not left_name or not right_name:
        return 0.0
    left_tokens = set(left_name.split())
    right_tokens = set(right_name.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence_score = 0.5 * SequenceMatcher(None, left_name, right_name).ratio()
    containment = float(left_name in right_name or right_name in left_name)
    return float(max(token_score, sequence_score, containment))


def obtain_road_network() -> Path:
    payload = archive_versions_for_day(ROAD_NETWORK_URL, "20230630")
    timestamp = closest_timestamp(payload.get("timestamps", []), "20230630", 1000)
    archive_path = download_historical(
        ROAD_NETWORK_URL,
        timestamp,
        RAW_DIR / f"road_network_{timestamp}.zip",
    )
    extract_root = RAW_DIR / f"road_network_{timestamp}"
    geodatabase = extract_root / "RdNet_IRNP.gdb"
    if not geodatabase.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
    return geodatabase


def read_centerline(geodatabase: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata, _, geometry_wkb, field_arrays = read_ogr(
        geodatabase,
        layer="CENTERLINE",
    )
    frame = pd.DataFrame(
        {
            field: values
            for field, values in zip(metadata["fields"], field_arrays)
        }
    )
    geometries = shapely.from_wkb(geometry_wkb)
    frame["road_2023_segment_index"] = np.arange(len(frame), dtype=int)
    frame["computed_length_m"] = shapely.length(geometries)
    street_codes = pd.to_numeric(frame["ST_CODE"], errors="coerce")
    counts = street_codes.dropna().astype(int).value_counts()
    frame["street_code_segment_count"] = street_codes.map(counts).fillna(0)

    endpoint_keys: list[tuple[tuple[float, float], tuple[float, float]]] = []
    degree: Counter[tuple[float, float]] = Counter()
    for geometry in geometries:
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        first = parts[0].coords[0]
        last = parts[-1].coords[-1]
        start_key = (round(first[0], 1), round(first[1], 1))
        end_key = (round(last[0], 1), round(last[1], 1))
        endpoint_keys.append((start_key, end_key))
        degree[start_key] += 1
        degree[end_key] += 1
    frame["endpoint_degree_mean"] = [
        (degree[start] + degree[end]) / 2.0 for start, end in endpoint_keys
    ]

    representative_points = [
        geometry.interpolate(0.5, normalized=True) for geometry in geometries
    ]
    frame["road_segment_x"] = [point.x for point in representative_points]
    frame["road_segment_y"] = [point.y for point in representative_points]
    transformer = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        frame["road_segment_x"].to_numpy(dtype=float),
        frame["road_segment_y"].to_numpy(dtype=float),
    )
    frame["road_longitude"] = longitude
    frame["road_latitude"] = latitude
    return frame, geometries


def build_network_base_features(centerline: pd.DataFrame) -> pd.DataFrame:
    """Create predictors known for every official 2023 centreline segment."""
    direction = centerline["TRAVEL_DIRECTION"].fillna("MISSING").astype(str)
    direction_levels = {
        value: index for index, value in enumerate(sorted(direction.unique()))
    }
    frame = pd.DataFrame(
        {
            "road_2023_segment_index": centerline["road_2023_segment_index"].astype(int),
            "road_2023_route_id": pd.to_numeric(
                centerline["ROUTE_ID"], errors="coerce"
            ),
            "road_2023_street_code": pd.to_numeric(
                centerline["ST_CODE"], errors="coerce"
            ),
            "road_name": centerline["STREET_ENAME"].fillna("").astype(str),
            "road_longitude": centerline["road_longitude"],
            "road_latitude": centerline["road_latitude"],
            "road_elevation": pd.to_numeric(
                centerline["ELEVATION"], errors="coerce"
            ),
            "road_travel_direction": direction.map(direction_levels).astype(float),
            "road_route_number_present": centerline["ROUTE_NUM"].notna().astype(int),
            "road_named_street": centerline["STREET_ENAME"].map(
                lambda value: int(bool(normalise_name(value)))
            ),
            "road_segment_length_m": centerline["computed_length_m"],
            "road_endpoint_degree_mean": centerline["endpoint_degree_mean"],
            "road_street_code_segment_count": centerline[
                "street_code_segment_count"
            ],
        }
    )
    if frame["road_2023_segment_index"].duplicated().any():
        raise ValueError("Centreline segment identifiers are not unique")
    if frame[["road_longitude", "road_latitude"]].isna().any(axis=None):
        raise ValueError("A centreline segment has no representative coordinate")
    return frame


def measured_2023_stations() -> pd.DataFrame:
    panel = pd.read_csv(MEASURED_PANEL_PATH)
    stations = (
        panel[panel["year"].astype(int) == YEAR]
        .sort_values("station_id")
        .drop_duplicates("station_id")
        .copy()
        .reset_index(drop=True)
    )
    if len(stations) != 880:
        raise ValueError(f"Expected 880 directly measured 2023 stations, found {len(stations)}")
    missing_coordinates = stations[["longitude", "latitude"]].isna().any(axis=1)
    if missing_coordinates.any():
        excluded_ids = ",".join(
            stations.loc[missing_coordinates, "station_id"].astype(str)
        )
        print(
            f"Excluding {int(missing_coordinates.sum())} measured station(s) without "
            f"coordinates from spatial validation: {excluded_ids}."
        )
        stations = stations.loc[~missing_coordinates].copy().reset_index(drop=True)
    fold_centroids = stations.dropna(subset=["spatial_fold"]).groupby("spatial_fold")[[
        "longitude",
        "latitude",
    ]].mean()
    missing = stations["spatial_fold"].isna()
    for index in stations.index[missing]:
        distances = (
            (fold_centroids["longitude"] - stations.at[index, "longitude"]) ** 2
            + (fold_centroids["latitude"] - stations.at[index, "latitude"]) ** 2
        )
        stations.at[index, "spatial_fold"] = int(distances.idxmin())
    stations["spatial_fold"] = stations["spatial_fold"].astype(int)
    return stations


def match_stations_to_road_network(
    stations: pd.DataFrame,
    centerline: pd.DataFrame,
    geometries: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    station_x, station_y = transformer.transform(
        stations["longitude"].to_numpy(dtype=float),
        stations["latitude"].to_numpy(dtype=float),
    )
    tree = STRtree(geometries)
    rows: list[dict[str, object]] = []
    for station, x_value, y_value in zip(
        stations.itertuples(index=False), station_x, station_y
    ):
        point = Point(x_value, y_value)
        candidates = tree.query(point, predicate="dwithin", distance=75.0)
        if len(candidates) == 0:
            candidates = np.atleast_1d(tree.query_nearest(point))
        scored: list[tuple[float, float, int, float]] = []
        for candidate in candidates:
            road = centerline.iloc[int(candidate)]
            similarity = max(
                name_similarity(station.road_name, road.get("STREET_ENAME")),
                name_similarity(station.road_name, road.get("ALIAS_ENAME")),
            )
            distance_m = float(point.distance(geometries[int(candidate)]))
            score = 2.0 * similarity - min(distance_m, 75.0) / 75.0
            scored.append((score, -distance_m, int(candidate), similarity))
        _, negative_distance, selected, similarity = max(scored)
        road = centerline.iloc[selected]
        distance_m = -negative_distance
        status = (
            "high"
            if (distance_m <= 20 and similarity >= 0.5) or distance_m <= 5
            else "moderate"
            if distance_m <= 50 and similarity >= 0.25
            else "low"
        )
        rows.append(
            {
                "station_id": int(station.station_id),
                "road_2023_segment_index": int(selected),
                "road_2023_route_id": int(road["ROUTE_ID"]),
                "road_2023_street_code": road["ST_CODE"],
                "road_2023_name": road["STREET_ENAME"],
                "road_2023_alias": road["ALIAS_ENAME"],
                "road_match_distance_m": distance_m,
                "road_name_similarity": similarity,
                "road_match_status": status,
            }
        )
    matches = pd.DataFrame(rows)
    audit_rows = [
        {
            "metric": "spatially_eligible_station_count",
            "value": len(matches),
            "interpretation": "directly measured 2023 stations with coordinates",
        },
        {
            "metric": "excluded_missing_coordinate_count",
            "value": 880 - len(matches),
            "interpretation": "cannot enter a spatial holdout without a location",
        },
        {
            "metric": "high_or_moderate_match_share",
            "value": matches["road_match_status"].isin(["high", "moderate"]).mean(),
            "interpretation": "2023 road snapshot spatial/name compatibility",
        },
        {
            "metric": "median_match_distance_m",
            "value": matches["road_match_distance_m"].median(),
            "interpretation": "distance from current ATC point to 2023 centreline",
        },
        {
            "metric": "p95_match_distance_m",
            "value": matches["road_match_distance_m"].quantile(0.95),
            "interpretation": "tail of station-to-road matching uncertainty",
        },
        {
            "metric": "2023_centerline_segment_count",
            "value": len(centerline),
            "interpretation": "historical Road Network 2nd Generation snapshot",
        },
    ]
    audit = pd.DataFrame(audit_rows)
    save_csv(matches, ROAD_MATCH_PATH)
    save_csv(audit, ROAD_AUDIT_PATH)
    return matches, audit


def obtain_gtfs() -> tuple[Path, str]:
    payload = archive_versions_for_day(GTFS_URL, "20230617")
    timestamp = closest_timestamp(payload.get("timestamps", []), "20230617", 1000)
    path = download_historical(
        GTFS_URL,
        timestamp,
        RAW_DIR / f"gtfs_{timestamp}.zip",
    )
    return path, timestamp


def seconds_from_gtfs_time(value: object) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(":")
    if len(parts) != 3:
        return np.nan
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def active_services(
    calendar_frame: pd.DataFrame,
    calendar_dates: pd.DataFrame,
    service_date: date,
) -> set[str]:
    date_number = int(service_date.strftime("%Y%m%d"))
    weekday = service_date.strftime("%A").lower()
    calendar_frame = calendar_frame.copy()
    calendar_frame["service_id"] = calendar_frame["service_id"].astype(str)
    mask = (
        (pd.to_numeric(calendar_frame["start_date"]) <= date_number)
        & (pd.to_numeric(calendar_frame["end_date"]) >= date_number)
        & (pd.to_numeric(calendar_frame[weekday]) == 1)
    )
    active = set(calendar_frame.loc[mask, "service_id"])
    exceptions = calendar_dates[
        pd.to_numeric(calendar_dates["date"], errors="coerce") == date_number
    ].copy()
    exceptions["service_id"] = exceptions["service_id"].astype(str)
    for row in exceptions.itertuples(index=False):
        if int(row.exception_type) == 1:
            active.add(row.service_id)
        elif int(row.exception_type) == 2:
            active.discard(row.service_id)
    return active


def gtfs_stop_features(
    gtfs_path: Path,
    targets: pd.DataFrame,
    timestamp: str,
    target_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(gtfs_path) as archive:
        def read_member(name: str, **kwargs: object) -> pd.DataFrame:
            with archive.open(name) as source:
                return pd.read_csv(source, low_memory=False, **kwargs)

        stops = read_member("stops.txt", dtype={"stop_id": str})
        trips = read_member(
            "trips.txt",
            dtype={"trip_id": str, "route_id": str, "service_id": str},
        )
        stop_times = read_member(
            "stop_times.txt",
            dtype={"trip_id": str, "stop_id": str},
            usecols=["trip_id", "stop_id"],
        )
        frequencies = read_member(
            "frequencies.txt",
            dtype={"trip_id": str},
        )
        calendar_frame = read_member("calendar.txt", dtype={"service_id": str})
        calendar_dates = read_member(
            "calendar_dates.txt",
            dtype={"service_id": str},
        )

    frequencies["start_seconds"] = frequencies["start_time"].map(seconds_from_gtfs_time)
    frequencies["end_seconds"] = frequencies["end_time"].map(seconds_from_gtfs_time)
    frequencies["run_count"] = (
        (frequencies["end_seconds"] - frequencies["start_seconds"])
        / pd.to_numeric(frequencies["headway_secs"], errors="coerce")
    ).clip(lower=1)
    frequency_weight = frequencies.groupby("trip_id")["run_count"].sum()

    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    stop_x, stop_y = transformer.transform(
        stops["stop_lon"].to_numpy(dtype=float),
        stops["stop_lat"].to_numpy(dtype=float),
    )
    target_x, target_y = transformer.transform(
        targets["road_longitude"].to_numpy(dtype=float),
        targets["road_latitude"].to_numpy(dtype=float),
    )
    tree = cKDTree(np.column_stack([stop_x, stop_y]))
    result = pd.DataFrame({target_id: targets[target_id].astype(int)})
    audit_rows: list[dict[str, object]] = []

    for label, service_date in (("weekday", GTFS_WEEKDAY), ("saturday", GTFS_SATURDAY)):
        services = active_services(calendar_frame, calendar_dates, service_date)
        active_trips = trips[trips["service_id"].isin(services)].copy()
        active_trips["trip_weight"] = active_trips["trip_id"].map(frequency_weight).fillna(1.0)
        events = stop_times.merge(
            active_trips[["trip_id", "route_id", "trip_weight"]],
            on="trip_id",
            how="inner",
        )
        stop_weight = events.groupby("stop_id")["trip_weight"].sum().to_dict()
        stop_routes = events.groupby("stop_id")["route_id"].agg(lambda values: set(values))
        route_lookup = stop_routes.to_dict()
        for radius in (250, 500):
            neighbourhoods = tree.query_ball_point(
                np.column_stack([target_x, target_y]),
                r=radius,
            )
            stop_counts: list[int] = []
            service_counts: list[float] = []
            route_counts: list[int] = []
            for indices in neighbourhoods:
                ids = stops.iloc[indices]["stop_id"].tolist()
                stop_counts.append(len(ids))
                service_counts.append(sum(float(stop_weight.get(value, 0.0)) for value in ids))
                routes: set[str] = set()
                for value in ids:
                    routes.update(route_lookup.get(value, set()))
                route_counts.append(len(routes))
            result[f"gtfs_{label}_stop_count_{radius}m"] = stop_counts
            result[f"gtfs_{label}_weighted_service_{radius}m"] = service_counts
            result[f"gtfs_{label}_route_count_{radius}m"] = route_counts
        audit_rows.append(
            {
                "gtfs_timestamp": timestamp,
                "service_date": service_date.isoformat(),
                "day_type": label,
                "active_service_ids": len(services),
                "active_trips": len(active_trips),
                "active_stop_events": len(events),
                "median_segment_stops_500m": result[f"gtfs_{label}_stop_count_500m"].median(),
                "segment_share_with_service_500m": (
                    result[f"gtfs_{label}_weighted_service_500m"] > 0
                ).mean(),
                "target_segment_count": len(targets),
                "interpretation": "scheduled_service_context_not_realised_traffic",
            }
        )
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, GTFS_AUDIT_PATH)
    return result, audit


def prepare_sensor_locations(
    strategic_features: pd.DataFrame,
    class_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategic_path = download_url(
        STRATEGIC_LOCATION_URL,
        RAW_DIR / "strategic_detector_locations.csv",
    )
    class_path = download_url(
        VEHICLE_CLASS_LOCATION_URL,
        RAW_DIR / "vehicle_class_detector_locations.csv",
    )
    strategic = pd.read_csv(strategic_path).rename(columns={"AID_ID_Number": "sensor_id"})
    vehicle_class = pd.read_csv(class_path).rename(columns={"Device_ID": "sensor_id"})
    strategic["sensor_id"] = strategic["sensor_id"].astype(str)
    vehicle_class["sensor_id"] = vehicle_class["sensor_id"].astype(str)
    strategic_features["sensor_id"] = strategic_features["sensor_id"].astype(str)
    class_features["sensor_id"] = class_features["sensor_id"].astype(str)
    strategic = strategic.merge(strategic_features, on="sensor_id", how="inner")
    vehicle_class = vehicle_class.merge(class_features, on="sensor_id", how="inner")
    return strategic, vehicle_class


def projected_coordinates(frame: pd.DataFrame, lon: str, lat: str) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x_values, y_values = transformer.transform(
        frame[lon].to_numpy(dtype=float),
        frame[lat].to_numpy(dtype=float),
    )
    return np.column_stack([x_values, y_values])


def assign_sensor_folds(
    sensors: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    output = sensors.copy()
    station_xy = projected_coordinates(stations, "longitude", "latitude")
    station_with_xy = stations[["spatial_fold"]].copy()
    station_with_xy["x"] = station_xy[:, 0]
    station_with_xy["y"] = station_xy[:, 1]
    centroids = station_with_xy.groupby("spatial_fold")[["x", "y"]].mean()
    sensor_xy = projected_coordinates(output, "Longitude", "Latitude")
    centroid_tree = cKDTree(centroids[["x", "y"]].to_numpy())
    _, indices = centroid_tree.query(sensor_xy, k=1)
    output["sensor_fold"] = centroids.index.to_numpy()[indices].astype(int)
    output["projected_x"] = sensor_xy[:, 0]
    output["projected_y"] = sensor_xy[:, 1]
    return output


def nearest_sensor_features(
    targets: pd.DataFrame,
    sensors: pd.DataFrame,
    prefix: str,
    value_columns: list[str],
) -> pd.DataFrame:
    target_xy = projected_coordinates(targets, "road_longitude", "road_latitude")
    sensor_xy = sensors[["projected_x", "projected_y"]].to_numpy(dtype=float)
    distances, indices = cKDTree(sensor_xy).query(target_xy, k=1)
    nearest = sensors.iloc[indices].reset_index(drop=True)
    output = pd.DataFrame(index=targets.index)
    output[f"{prefix}_nearest_distance_m"] = distances
    output[f"{prefix}_nearest_log_distance"] = np.log1p(distances)
    output[f"{prefix}_nearest_name_similarity"] = [
        name_similarity(station_name, sensor_name)
        for station_name, sensor_name in zip(targets["road_name"], nearest["Road_EN"])
    ]
    for column in value_columns:
        output[f"{prefix}_{column}"] = pd.to_numeric(
            nearest[column], errors="coerce"
        ).to_numpy()
    return output


def masked_sensor_sets(
    sensors: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    test_xy = projected_coordinates(test, "station_longitude", "station_latitude")
    sensor_xy = sensors[["projected_x", "projected_y"]].to_numpy(dtype=float)
    nearest_test_distance, _ = cKDTree(test_xy).query(sensor_xy, k=1)
    allowed = (sensors["sensor_fold"].to_numpy(dtype=int) != fold) & (
        nearest_test_distance > MASK_BUFFER_M
    )
    retained = sensors.loc[allowed].copy()
    return retained, {
        "spatial_fold": fold,
        "source_sensor_count": len(sensors),
        "retained_sensor_count": len(retained),
        "masked_sensor_count": int((~allowed).sum()),
        "retained_share": allowed.mean(),
        "retained_share_denominator": "source_sensor_count",
        "mask_buffer_m": MASK_BUFFER_M,
        "mask_target": "held_out_ATC_station_locations",
        "mask_rule": "remove_test_fold_sensors_and_any_sensor_within_1km_of_a_held_out_ATC_station",
        "deployment_scenario": "detector_sparse_spatial_extrapolation_around_a_labelled_test_road_segment",
        "does_not_validate": "roads_outside_ATC_label_support",
        "interpretation": "retained_share_is_the_fraction_of_source_sensors_available_to_generate_segment_features_after_masking",
    }


def build_network_feature_table(
    network_base: pd.DataFrame,
    gtfs_features: pd.DataFrame,
    strategic: pd.DataFrame,
    vehicle_class: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    """Generate every deployable predictor on all centreline segments."""
    frame = network_base.merge(
        gtfs_features,
        on="road_2023_segment_index",
        how="left",
        validate="one_to_one",
    )
    strategic_value_columns = [
        column
        for column in strategic.columns
        if column.startswith("strategic_")
        and column not in {"strategic_sample_count"}
    ]
    class_value_columns = [
        column
        for column in vehicle_class.columns
        if column.startswith("class_")
        and column not in {"class_sample_count"}
    ]
    segment_strategic = nearest_sensor_features(
        frame,
        strategic,
        "strategic",
        strategic_value_columns,
    )
    segment_class = nearest_sensor_features(
        frame,
        vehicle_class,
        "vehicle_class",
        class_value_columns,
    )
    frame = pd.concat(
        [frame.reset_index(drop=True), segment_strategic, segment_class],
        axis=1,
    )
    deployable_structural_features = [
        "road_longitude",
        "road_latitude",
        "road_elevation",
        "road_travel_direction",
        "road_route_number_present",
        "road_named_street",
        "road_segment_length_m",
        "road_endpoint_degree_mean",
        "road_street_code_segment_count",
    ]
    gtfs_columns = sorted(column for column in frame if column.startswith("gtfs_"))
    dynamic_columns = [
        column
        for column in frame
        if column.startswith("strategic_") or column.startswith("vehicle_class_")
    ]
    dynamic_value_columns_strategic = [
        column.replace("strategic_", "", 1)
        for column in dynamic_columns
        if column.startswith("strategic_strategic_")
    ]
    dynamic_value_columns_class = [
        column.replace("vehicle_class_", "", 1)
        for column in dynamic_columns
        if column.startswith("vehicle_class_class_")
    ]
    return (
        frame,
        deployable_structural_features,
        gtfs_columns,
        dynamic_columns,
        dynamic_value_columns_strategic,
        dynamic_value_columns_class,
    )


def build_station_feature_table(
    stations: pd.DataFrame,
    road_matches: pd.DataFrame,
    network_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach full-network predictors to outcomes and construct an oracle only."""
    outcomes = stations[
        [
            "station_id",
            "spatial_fold",
            "region",
            "road_network",
            "road_type",
            "road_name",
            "longitude",
            "latitude",
            "aadt",
        ]
    ].rename(
        columns={
            "road_name": "station_road_name",
            "longitude": "station_longitude",
            "latitude": "station_latitude",
        }
    )
    match_columns = [
        "station_id",
        "road_2023_segment_index",
        "road_match_distance_m",
        "road_name_similarity",
        "road_match_status",
    ]
    frame = outcomes.merge(
        road_matches[match_columns],
        on="station_id",
        how="left",
        validate="one_to_one",
    ).merge(
        network_features,
        on="road_2023_segment_index",
        how="left",
        validate="many_to_one",
    )

    for value in sorted(frame["road_type"].dropna().unique()):
        column = "road_type_" + re.sub(
            r"[^a-z0-9]+", "_", value.lower()
        ).strip("_")
        frame[column] = (frame["road_type"] == value).astype(int)
    frame["road_network_major"] = (frame["road_network"] == "MAJOR").astype(int)
    road_type_columns = sorted(
        column for column in frame if column.startswith("road_type_")
    )
    oracle_features = [
        "road_longitude",
        "road_latitude",
        "road_network_major",
        "road_elevation",
        "road_travel_direction",
        "road_route_number_present",
        "road_named_street",
        "road_segment_length_m",
        "road_endpoint_degree_mean",
        "road_street_code_segment_count",
        "road_match_distance_m",
        "road_name_similarity",
        *road_type_columns,
    ]
    return frame, oracle_features


def feature_deployability_manifest(
    network_features: pd.DataFrame,
    deployable_structural: list[str],
    gtfs_features: list[str],
    dynamic_features: list[str],
    oracle_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        (deployable_structural, "official_road_network", "deployable_structural", True),
        (gtfs_features, "historical_gtfs", "deployable_context", True),
        (dynamic_features, "public_detector_archive", "deployable_sensor_proxy", True),
    )
    for columns, source, role, allowed in groups:
        for column in columns:
            rows.append(
                {
                    "feature": column,
                    "source": source,
                    "role": role,
                    "available_segment_count": int(network_features[column].notna().sum()),
                    "total_segment_count": len(network_features),
                    "available_segment_share": network_features[column].notna().mean(),
                    "allowed_in_deployable_model": allowed,
                }
            )
    prohibited = sorted(
        set(oracle_features)
        - set(deployable_structural)
    )
    for column in prohibited:
        source = (
            "station_to_road_matching"
            if column in {"road_match_distance_m", "road_name_similarity"}
            else "atc_station_metadata"
        )
        rows.append(
            {
                "feature": column,
                "source": source,
                "role": "oracle_diagnostic_only",
                "available_segment_count": 0,
                "total_segment_count": len(network_features),
                "available_segment_share": 0.0,
                "allowed_in_deployable_model": False,
            }
        )
    manifest = pd.DataFrame(rows)
    save_csv(manifest, FEATURE_MANIFEST_PATH)
    return manifest


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].copy()
    for column in values:
        if values[column].dtype == bool:
            values[column] = values[column].astype(int)
    return values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


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


def hierarchy_lookup_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    variable = "road_street_code_segment_count"
    quantiles = np.quantile(
        pd.to_numeric(train[variable], errors="coerce"),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    edges = np.unique(quantiles)
    if len(edges) < 3:
        raise ValueError("Street-code segment count cannot form a useful hierarchy lookup")

    def assign(values: pd.Series) -> np.ndarray:
        return np.clip(
            np.digitize(pd.to_numeric(values).to_numpy(), edges[1:-1], right=True),
            0,
            len(edges) - 2,
        )

    train_bins = assign(train[variable])
    test_bins = assign(test[variable])
    train_route = train["road_route_number_present"].astype(int).to_numpy()
    test_route = test["road_route_number_present"].astype(int).to_numpy()
    target = train["aadt"].to_numpy(dtype=float)
    cell_medians = {
        (route, level): float(np.median(target[(train_route == route) & (train_bins == level)]))
        for route in (0, 1)
        for level in np.unique(train_bins)
        if np.any((train_route == route) & (train_bins == level))
    }
    route_medians = {
        route: float(np.median(target[train_route == route]))
        for route in (0, 1)
        if np.any(train_route == route)
    }
    global_median = float(np.median(target))
    return np.asarray(
        [
            cell_medians.get(
                (route, level),
                route_medians.get(route, global_median),
            )
            for route, level in zip(test_route, test_bins)
        ]
    )


def metric_row(
    fold: int | str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    correlation = (
        spearmanr(observed, predicted).statistic
        if len(observed) > 2
        and np.nanstd(observed) > 0
        and np.nanstd(predicted) > 0
        else np.nan
    )
    return {
        "spatial_fold": fold,
        "model": model,
        "n": len(observed),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": math.sqrt(mean_squared_error(observed, predicted)),
        "r2": r2_score(observed, predicted),
        "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
        "spearman": correlation,
    }


def run_spatial_experiment(
    features: pd.DataFrame,
    deployable_structural_features: list[str],
    oracle_features: list[str],
    gtfs_features: list[str],
    dynamic_features: list[str],
    strategic: pd.DataFrame,
    vehicle_class: pd.DataFrame,
    strategic_value_columns: list[str],
    class_value_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    deployable_structural_gtfs = [*deployable_structural_features, *gtfs_features]
    deployable_full_features = [*deployable_structural_gtfs, *dynamic_features]

    for fold in FOLDS:
        train = features[features["spatial_fold"] != fold].copy().reset_index(drop=True)
        test = features[features["spatial_fold"] == fold].copy().reset_index(drop=True)
        y_train = train["aadt"].to_numpy(dtype=float)
        y_test = test["aadt"].to_numpy(dtype=float)

        predictions: dict[str, np.ndarray] = {
            "training_median": np.full(len(test), np.median(y_train)),
            "hierarchy_lookup": hierarchy_lookup_predict(train, test),
        }
        for model_name, columns in (
            ("atc_class_oracle_hgb", oracle_features),
            ("deployable_structural_hgb", deployable_structural_features),
            ("deployable_structural_gtfs_hgb", deployable_structural_gtfs),
            ("deployable_sensor_assisted_hgb", deployable_full_features),
        ):
            model = fixed_model()
            model.fit(matrix(train, columns), y_train)
            predictions[model_name] = model.predict(matrix(test, columns))

        allowed_strategic, strategic_mask = masked_sensor_sets(strategic, test, fold)
        allowed_class, class_mask = masked_sensor_sets(vehicle_class, test, fold)
        strategic_mask["sensor_source"] = "strategic"
        class_mask["sensor_source"] = "vehicle_class"
        mask_rows.extend([strategic_mask, class_mask])

        masked_train = pd.concat(
            [
                train.drop(columns=dynamic_features).reset_index(drop=True),
                nearest_sensor_features(
                    train,
                    allowed_strategic,
                    "strategic",
                    strategic_value_columns,
                ),
                nearest_sensor_features(
                    train,
                    allowed_class,
                    "vehicle_class",
                    class_value_columns,
                ),
            ],
            axis=1,
        )
        masked_test = pd.concat(
            [
                test.drop(columns=dynamic_features).reset_index(drop=True),
                nearest_sensor_features(
                    test,
                    allowed_strategic,
                    "strategic",
                    strategic_value_columns,
                ),
                nearest_sensor_features(
                    test,
                    allowed_class,
                    "vehicle_class",
                    class_value_columns,
                ),
            ],
            axis=1,
        )
        free_model = fixed_model()
        free_model.fit(matrix(masked_train, deployable_full_features), y_train)
        predictions["deployable_sensor_free_hgb"] = free_model.predict(
            matrix(masked_test, deployable_full_features)
        )

        for model_name in MODEL_ORDER:
            predicted = predictions[model_name]
            metric_rows.append(metric_row(fold, model_name, y_test, predicted))
            for row_index, station in test.iterrows():
                prediction_rows.append(
                    {
                        "station_id": int(station["station_id"]),
                        "spatial_fold": fold,
                        "region": station["region"],
                        "road_network": station["road_network"],
                        "road_type": station["road_type"],
                        "model": model_name,
                        "observed_aadt": y_test[row_index],
                        "predicted_aadt": predicted[row_index],
                        "error": predicted[row_index] - y_test[row_index],
                        "absolute_error": abs(predicted[row_index] - y_test[row_index]),
                        "step21_credible_strategic_support": bool(
                            station.get("strategic_credible_nearby_sensor", False)
                        ),
                        "assisted_nearest_strategic_distance_m": station[
                            "strategic_nearest_distance_m"
                        ],
                        "masked_nearest_strategic_distance_m": masked_test.loc[
                            row_index, "strategic_nearest_distance_m"
                        ],
                    }
                )
        print(f"Completed spatial fold {fold}: train={len(train)}, test={len(test)}")
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    masks = pd.DataFrame(mask_rows)
    save_csv(predictions, PREDICTION_PATH)
    save_csv(metrics, METRICS_BY_FOLD_PATH)
    save_csv(masks, SENSOR_MASK_PATH)
    return predictions, metrics, masks


def summarise_models(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        group = predictions[predictions["model"] == model_name]
        row = metric_row(
            "pooled",
            model_name,
            group["observed_aadt"].to_numpy(dtype=float),
            group["predicted_aadt"].to_numpy(dtype=float),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    lookup_mae = summary.set_index("model").loc["hierarchy_lookup", "mae"]
    structural_mae = summary.set_index("model").loc[
        "deployable_structural_gtfs_hgb", "mae"
    ]
    summary["mae_improvement_vs_hierarchy_pct"] = 100.0 * (
        lookup_mae - summary["mae"]
    ) / lookup_mae
    summary["mae_improvement_vs_structural_gtfs_pct"] = 100.0 * (
        structural_mae - summary["mae"]
    ) / structural_mae
    save_csv(summary, SUMMARY_PATH)
    return summary


def subgroup_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata: list[tuple[str, str, pd.DataFrame]] = []
    for model_name, model_group in predictions.groupby("model"):
        strata.append((model_name, "all", model_group))
        for value, group in model_group.groupby("region"):
            strata.append((model_name, f"region:{value}", group))
        for value, group in model_group.groupby("road_network"):
            strata.append((model_name, f"road_network:{value}", group))
        for value, group in model_group.groupby("step21_credible_strategic_support"):
            strata.append((model_name, f"credible_sensor_support:{bool(value)}", group))
    for model_name, stratum, group in strata:
        observed = group["observed_aadt"].to_numpy(dtype=float)
        predicted = group["predicted_aadt"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "stratum": stratum,
                "n": len(group),
                "mae": mean_absolute_error(observed, predicted),
                "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
                "observed_mean": observed.mean(),
                "predicted_mean": predicted.mean(),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, SUBGROUP_PATH)
    return frame


def predicted_bin_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("model"):
        group = group.copy()
        group["predicted_quintile"] = pd.qcut(
            group["predicted_aadt"],
            5,
            labels=False,
            duplicates="drop",
        ) + 1
        for quintile, selected in group.groupby("predicted_quintile"):
            observed_mean = selected["observed_aadt"].mean()
            predicted_mean = selected["predicted_aadt"].mean()
            rows.append(
                {
                    "model": model_name,
                    "predicted_quintile": int(quintile),
                    "n": len(selected),
                    "observed_mean": observed_mean,
                    "predicted_mean": predicted_mean,
                    "relative_bias_pct": 100.0 * (predicted_mean - observed_mean) / observed_mean,
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, CALIBRATION_PATH)
    return frame


def cluster_bootstrap_loss_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    draws: int = 4000,
) -> tuple[float, float]:
    paired = candidate[["station_id", "spatial_fold", "absolute_error"]].merge(
        reference[["station_id", "absolute_error"]],
        on="station_id",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    paired["loss_difference"] = (
        paired["absolute_error_candidate"] - paired["absolute_error_reference"]
    )
    fold_values = {
        fold: group["loss_difference"].to_numpy(dtype=float)
        for fold, group in paired.groupby("spatial_fold")
    }
    rng = np.random.default_rng(42)
    fold_ids = np.asarray(sorted(fold_values))
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled_folds = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        values = np.concatenate([fold_values[int(fold)] for fold in sampled_folds])
        estimates[draw] = values.mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_comparisons(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_name, reference_name, subset_name, support_value in (
        ("atc_class_oracle_hgb", "hierarchy_lookup", "all", None),
        ("deployable_structural_hgb", "hierarchy_lookup", "all", None),
        (
            "deployable_structural_gtfs_hgb",
            "deployable_structural_hgb",
            "all",
            None,
        ),
        ("deployable_structural_gtfs_hgb", "hierarchy_lookup", "all", None),
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "all",
            None,
        ),
        (
            "deployable_sensor_free_hgb",
            "deployable_structural_gtfs_hgb",
            "all",
            None,
        ),
        ("deployable_sensor_free_hgb", "hierarchy_lookup", "all", None),
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "credible_strategic_support",
            True,
        ),
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "without_credible_strategic_support",
            False,
        ),
    ):
        candidate = predictions[predictions["model"] == candidate_name]
        reference = predictions[predictions["model"] == reference_name]
        if support_value is not None:
            candidate = candidate[
                candidate["step21_credible_strategic_support"] == support_value
            ]
            reference = reference[
                reference["step21_credible_strategic_support"] == support_value
            ]
        candidate_mae = candidate["absolute_error"].mean()
        reference_mae = reference["absolute_error"].mean()
        low, high = cluster_bootstrap_loss_difference(candidate, reference)
        candidate_fold = candidate.groupby("spatial_fold")["absolute_error"].mean()
        reference_fold = reference.groupby("spatial_fold")["absolute_error"].mean()
        improved_fold_count = int((candidate_fold < reference_fold).sum())
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                "evaluation_subset": subset_name,
                "n": len(candidate),
                "candidate_mae": candidate_mae,
                "reference_mae": reference_mae,
                "mae_improvement_pct": 100.0 * (reference_mae - candidate_mae) / reference_mae,
                "mean_absolute_loss_difference": candidate_mae - reference_mae,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "improved_fold_count": improved_fold_count,
                "interpretation": "negative_loss_difference_favours_candidate",
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, COMPARISON_PATH)
    return frame


def deployability_ablation(
    summary: pd.DataFrame,
    subgroups: pd.DataFrame,
    feature_counts: dict[str, int],
) -> pd.DataFrame:
    summary_lookup = summary.set_index("model")
    subgroup_lookup = subgroups.set_index(["model", "stratum"])
    rows: list[dict[str, object]] = []
    for model_name, deployable in (
        ("hierarchy_lookup", True),
        ("atc_class_oracle_hgb", False),
        ("deployable_structural_hgb", True),
        ("deployable_structural_gtfs_hgb", True),
        ("deployable_sensor_assisted_hgb", True),
        ("deployable_sensor_free_hgb", True),
    ):
        row = summary_lookup.loc[model_name]
        major = subgroup_lookup.loc[(model_name, "road_network:MAJOR")]
        minor = subgroup_lookup.loc[(model_name, "road_network:MINOR")]
        rows.append(
            {
                "model": model_name,
                "deployable_on_all_centreline_segments": deployable,
                "feature_count": feature_counts[model_name],
                "pooled_mae": row["mae"],
                "mae_improvement_vs_hierarchy_pct": row[
                    "mae_improvement_vs_hierarchy_pct"
                ],
                "major_mae": major["mae"],
                "minor_mae": minor["mae"],
                "minor_aggregate_bias_pct": minor["aggregate_bias_pct"],
                "interpretation": (
                    "oracle_diagnostic_not_deployment_evidence"
                    if not deployable
                    else "eligible_for_deployment_gate"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, DEPLOYABILITY_PATH)
    return frame


def step15_step22_comparability_audit(
    step22_features: pd.DataFrame,
    step22_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Describe, rather than erase, the support difference between Steps 15 and 22."""
    step15_training = pd.read_csv(STEP15_TRAINING_PATH)
    step15_core = pd.read_csv(STEP15_CORE_LONG_PATH)
    step15_summary = pd.read_csv(STEP15_SUMMARY_PATH)

    step15_ids = set(step15_training["station_id"].astype(int))
    step22_ids = set(step22_features["station_id"].astype(int))
    overlap_ids = step15_ids & step22_ids
    reference_year = int(step15_core["year"].max())
    step15_types = step15_core[
        (step15_core["year"].astype(int) == reference_year)
        & step15_core["station_id"].astype(int).isin(step15_ids)
    ].drop_duplicates("station_id")
    step22_unique = step22_features.drop_duplicates("station_id")

    step15_hgb = step15_summary[
        step15_summary["model"] == "hist_gradient_boosting"
    ]["gain_vs_road_hierarchy_median_pct"]
    step22_lookup = step22_summary.set_index("model")
    structure_gain = step22_lookup.loc[
        "deployable_structural_hgb", "mae_improvement_vs_hierarchy_pct"
    ]
    structure_gtfs_gain = step22_lookup.loc[
        "deployable_structural_gtfs_hgb", "mae_improvement_vs_hierarchy_pct"
    ]

    rows = [
        {
            "comparison_dimension": "label_years",
            "step15_value": "2011,2016,2021",
            "step22_value": "2023",
            "decision_implication": "performance_difference_mixes_calendar_time_with_sample_and_feature_support",
        },
        {
            "comparison_dimension": "unique_station_count",
            "step15_value": str(len(step15_ids)),
            "step22_value": str(len(step22_ids)),
            "decision_implication": "the_two_scores_use_different_label_support",
        },
        {
            "comparison_dimension": "exact_station_id_overlap",
            "step15_value": f"{len(overlap_ids)} of {len(step15_ids)}",
            "step22_value": f"{len(overlap_ids)} of {len(step22_ids)}",
            "decision_implication": "a_679_station_2023_matched_reproduction_is_not_available",
        },
        {
            "comparison_dimension": "local_distributor_count",
            "step15_value": str(int((step15_types["road_type"] == "LD").sum())),
            "step22_value": str(
                int((step22_unique["road_type"] == "LOCAL DISTRIBUTOR").sum())
            ),
            "decision_implication": "the_step15_panel_is_not_devoid_of_local_distributors",
        },
        {
            "comparison_dimension": "minor_network_count",
            "step15_value": "not_defined_in_step15_training_table",
            "step22_value": str(int((step22_unique["road_network"] == "MINOR").sum())),
            "decision_implication": "minor_network_and_local_distributor_are_not_interchangeable_categories",
        },
        {
            "comparison_dimension": "minor_count_in_exact_overlap",
            "step15_value": "not_defined_in_step15_training_table",
            "step22_value": str(
                int(
                    (
                        step22_unique["station_id"].astype(int).isin(overlap_ids)
                        & (step22_unique["road_network"] == "MINOR")
                    ).sum()
                )
            ),
            "decision_implication": "the_exact_overlap_is_not_a_minor_road_free_comparison_set",
        },
        {
            "comparison_dimension": "deployable_feature_anchor",
            "step15_value": "earlier_network_segment_centroids",
            "step22_value": "matched_2023_centreline_representative_points",
            "decision_implication": "coordinate_and_network_snapshot_semantics_changed",
        },
        {
            "comparison_dimension": "hgb_gain_vs_fold_internal_hierarchy_lookup_pct",
            "step15_value": (
                f"range={step15_hgb.min():.2f} to {step15_hgb.max():.2f}; "
                f"median={step15_hgb.median():.2f}"
            ),
            "step22_value": (
                f"structure_only={structure_gain:.2f}; "
                f"structure_plus_gtfs={structure_gtfs_gain:.2f}"
            ),
            "decision_implication": "report_non_reproduction_and_task_difference_not_one_pooled_skill_number",
        },
        {
            "comparison_dimension": "causal_attribution_of_score_difference",
            "step15_value": "not_identified",
            "step22_value": "not_identified",
            "decision_implication": "do_not_attribute_the_gap_to_minor_road_sample_composition_alone",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, COMPARABILITY_PATH)
    return frame


def failed_criteria(
    improvement_pct: float,
    threshold_pct: float,
    interval_high: float,
    improved_fold_count: int | None = None,
    minimum_improved_folds: int | None = None,
) -> dict[str, object]:
    effect_pass = improvement_pct >= threshold_pct
    interval_pass = interval_high < 0
    fold_pass = (
        True
        if minimum_improved_folds is None
        else int(improved_fold_count) >= minimum_improved_folds
    )
    failures: list[str] = []
    if not effect_pass:
        failures.append("effect_below_threshold")
    if not interval_pass:
        failures.append("interval_includes_zero")
    if not fold_pass:
        failures.append("insufficient_fold_consistency")
    return {
        "effect_threshold_pct": threshold_pct,
        "effect_threshold_pass": effect_pass,
        "interval_excludes_zero": interval_pass,
        "minimum_improved_folds": minimum_improved_folds,
        "fold_consistency_pass": fold_pass,
        "failed_criterion": ";".join(failures) if failures else "none",
    }


def decision_audit(
    summary: pd.DataFrame,
    subgroups: pd.DataFrame,
    comparisons: pd.DataFrame,
    road_audit: pd.DataFrame,
    strategic_features: pd.DataFrame,
    class_features: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    network_features: pd.DataFrame,
) -> pd.DataFrame:
    model_summary = summary.set_index("model")
    comparison = comparisons.set_index(["candidate", "reference", "evaluation_subset"])
    road_match_share = road_audit.set_index("metric").loc[
        "high_or_moderate_match_share", "value"
    ]
    oracle_vs_lookup = comparison.loc[
        ("atc_class_oracle_hgb", "hierarchy_lookup", "all")
    ]
    structure_only_vs_lookup = comparison.loc[
        ("deployable_structural_hgb", "hierarchy_lookup", "all")
    ]
    structure_gtfs_vs_lookup = comparison.loc[
        ("deployable_structural_gtfs_hgb", "hierarchy_lookup", "all")
    ]
    free_vs_structural = comparison.loc[
        (
            "deployable_sensor_free_hgb",
            "deployable_structural_gtfs_hgb",
            "all",
        )
    ]
    free_vs_lookup = comparison.loc[
        ("deployable_sensor_free_hgb", "hierarchy_lookup", "all")
    ]
    gtfs_vs_road = comparison.loc[
        (
            "deployable_structural_gtfs_hgb",
            "deployable_structural_hgb",
            "all",
        )
    ]
    assisted_vs_structural = comparison.loc[
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "all",
        )
    ]
    assisted_supported = comparison.loc[
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "credible_strategic_support",
        )
    ]
    assisted_unsupported = comparison.loc[
        (
            "deployable_sensor_assisted_hgb",
            "deployable_structural_gtfs_hgb",
            "without_credible_strategic_support",
        )
    ]
    free_summary = model_summary.loc["deployable_sensor_free_hgb"]
    free_groups = subgroups[
        (subgroups["model"] == "deployable_sensor_free_hgb")
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ]
    max_subgroup_bias = free_groups["aggregate_bias_pct"].abs().max()
    sampling_gate = (
        len(strategic_features) >= 500
        and strategic_features["strategic_sample_coverage"].median() >= 0.75
        and len(class_features) >= 50
        and class_features["class_sample_coverage"].median() >= 0.75
    )
    expected_segment_count = int(
        road_audit.set_index("metric").loc["2023_centerline_segment_count", "value"]
    )
    network_generation_gate = (
        len(network_features) == expected_segment_count
        and not network_features["road_2023_segment_index"].duplicated().any()
    )
    lineage_gate = not feature_manifest.loc[
        feature_manifest["allowed_in_deployable_model"], "source"
    ].isin(["atc_station_metadata", "station_to_road_matching"]).any()

    structure_only_diagnostic = failed_criteria(
        structure_only_vs_lookup["mae_improvement_pct"],
        5.0,
        structure_only_vs_lookup["cluster_bootstrap_high"],
    )
    structure_gtfs_diagnostic = failed_criteria(
        structure_gtfs_vs_lookup["mae_improvement_pct"],
        5.0,
        structure_gtfs_vs_lookup["cluster_bootstrap_high"],
    )
    gtfs_diagnostic = failed_criteria(
        gtfs_vs_road["mae_improvement_pct"],
        2.0,
        gtfs_vs_road["cluster_bootstrap_high"],
        int(gtfs_vs_road["improved_fold_count"]),
        3,
    )
    assisted_diagnostic = failed_criteria(
        assisted_vs_structural["mae_improvement_pct"],
        2.0,
        assisted_vs_structural["cluster_bootstrap_high"],
        int(assisted_vs_structural["improved_fold_count"]),
        3,
    )
    assisted_supported_diagnostic = failed_criteria(
        assisted_supported["mae_improvement_pct"],
        5.0,
        assisted_supported["cluster_bootstrap_high"],
        int(assisted_supported["improved_fold_count"]),
        3,
    )
    sensor_free_diagnostic = failed_criteria(
        free_vs_structural["mae_improvement_pct"],
        2.0,
        free_vs_structural["cluster_bootstrap_high"],
        int(free_vs_structural["improved_fold_count"]),
        3,
    )
    useful_skill_diagnostic = failed_criteria(
        free_vs_lookup["mae_improvement_pct"],
        5.0,
        free_vs_lookup["cluster_bootstrap_high"],
    )

    def diagnostic_pass(values: dict[str, object]) -> bool:
        return bool(
            values["effect_threshold_pass"]
            and values["interval_excludes_zero"]
            and values["fold_consistency_pass"]
        )

    assisted_added_value = diagnostic_pass(assisted_diagnostic)
    gtfs_added_value = diagnostic_pass(gtfs_diagnostic)
    assisted_supported_added_value = diagnostic_pass(
        assisted_supported_diagnostic
    )
    sensor_free_added_value = diagnostic_pass(sensor_free_diagnostic)
    useful_skill = diagnostic_pass(useful_skill_diagnostic)
    structure_only_skill = diagnostic_pass(structure_only_diagnostic)
    structure_gtfs_skill = diagnostic_pass(structure_gtfs_diagnostic)
    aggregate_gate = abs(free_summary["aggregate_bias_pct"]) <= 10.0
    subgroup_gate = max_subgroup_bias <= 15.0
    full_gate = (
        sampling_gate
        and network_generation_gate
        and lineage_gate
        and road_match_share >= 0.90
        and sensor_free_added_value
        and useful_skill
        and aggregate_gate
        and subgroup_gate
    )
    full_failures = [
        name
        for name, passed in (
            ("detector_sample_coverage_gate", sampling_gate),
            ("full_network_feature_generation_gate", network_generation_gate),
            ("deployable_predictor_lineage_gate", lineage_gate),
            ("road_match_support_gate", road_match_share >= 0.90),
            ("sensor_free_dynamic_increment_gate", sensor_free_added_value),
            ("sensor_free_skill_vs_hierarchy_gate", useful_skill),
            ("aggregate_bias_gate", aggregate_gate),
            ("subgroup_bias_gate", subgroup_gate),
        )
        if not passed
    ]

    rows = [
        {
            "decision": "original_step22_deployable_feature_gate",
            "pass": False,
            "evidence": (
                "ATC road_network/road_type and station match diagnostics were used "
                f"in the old structural score; oracle gain={oracle_vs_lookup['mae_improvement_pct']:.2f}%"
            ),
            "failed_criterion": "non_deployable_predictor_lineage",
            "action": "supersede the old 9.82% skill claim and retain the old model only as an oracle diagnostic",
        },
        {
            "decision": "corrected_full_network_feature_generation_gate",
            "pass": network_generation_gate,
            "evidence": (
                f"generated_segments={len(network_features)}; "
                f"expected_segments={expected_segment_count}; unique_segment_ids="
                f"{not network_features['road_2023_segment_index'].duplicated().any()}"
            ),
            "failed_criterion": "none" if network_generation_gate else "full_network_feature_generation_incomplete",
            "action": "require structural, GTFS and unmasked detector features to exist on the full centreline before station validation",
        },
        {
            "decision": "deployable_predictor_lineage_excludes_atc_class_and_match_diagnostics",
            "pass": lineage_gate,
            "evidence": "ATC class and station-match fields appear only in atc_class_oracle_hgb",
            "failed_criterion": "none" if lineage_gate else "prohibited_predictor_lineage",
            "action": "never use the oracle score as evidence for an unmeasured-road prediction surface",
        },
        {
            "decision": "predeclared_dynamic_sample_has_adequate_detector_coverage",
            "pass": sampling_gate,
            "evidence": (
                f"strategic_sensors={len(strategic_features)}; "
                f"strategic_median_coverage={strategic_features['strategic_sample_coverage'].median():.3f}; "
                f"class_sensors={len(class_features)}; "
                f"class_median_coverage={class_features['class_sample_coverage'].median():.3f}"
            ),
            "failed_criterion": "none" if sampling_gate else "detector_sample_coverage_below_threshold",
            "action": "retain the bounded annual sample as a predictor experiment, not detector AADT",
        },
        {
            "decision": "2023_road_snapshot_matches_measured_station_support",
            "pass": road_match_share >= 0.90,
            "evidence": f"high_or_moderate_match_share={road_match_share:.3f}",
            "failed_criterion": "none" if road_match_share >= 0.90 else "road_match_share_below_threshold",
            "action": "retain 2023 road attributes; keep low matches in sensitivity reporting",
        },
        {
            "decision": "deployable_structure_only_materially_beats_honest_hierarchy_lookup",
            "pass": structure_only_skill,
            "evidence": (
                f"improvement={structure_only_vs_lookup['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={structure_only_vs_lookup['cluster_bootstrap_high']:.1f}; "
                f"improved_folds={int(structure_only_vs_lookup['improved_fold_count'])}/5"
            ),
            **structure_only_diagnostic,
            "action": "require at least 5% improvement using only predictors generated for all centreline segments",
        },
        {
            "decision": "deployable_structure_plus_gtfs_materially_beats_honest_hierarchy_lookup",
            "pass": structure_gtfs_skill,
            "evidence": (
                f"improvement={structure_gtfs_vs_lookup['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={structure_gtfs_vs_lookup['cluster_bootstrap_high']:.1f}; "
                f"improved_folds={int(structure_gtfs_vs_lookup['improved_fold_count'])}/5"
            ),
            **structure_gtfs_diagnostic,
            "action": "require at least 5% improvement for the complete sensor-free structural context",
        },
        {
            "decision": "gtfs_context_adds_material_skill_beyond_road_structure",
            "pass": gtfs_added_value,
            "evidence": (
                f"improvement={gtfs_vs_road['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={gtfs_vs_road['cluster_bootstrap_high']:.1f}; "
                f"improved_folds={int(gtfs_vs_road['improved_fold_count'])}/5"
            ),
            **gtfs_diagnostic,
            "action": "retain GTFS only as a small structural context block unless this gate passes",
        },
        {
            "decision": "sensor_assisted_dynamic_block_adds_material_skill",
            "pass": assisted_added_value,
            "evidence": (
                f"improvement={assisted_vs_structural['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={assisted_vs_structural['cluster_bootstrap_high']:.1f}; "
                f"improved_folds={int(assisted_vs_structural['improved_fold_count'])}/5"
            ),
            **assisted_diagnostic,
            "action": "report only as a sensor-supported deployment result",
        },
        {
            "decision": "sensor_assisted_block_helps_credibly_supported_stations",
            "pass": assisted_supported_added_value,
            "evidence": (
                f"supported_n={int(assisted_supported['n'])}; "
                f"supported_improvement={assisted_supported['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={assisted_supported['cluster_bootstrap_high']:.1f}; "
                f"unsupported_improvement={assisted_unsupported['mae_improvement_pct']:.2f}%"
            ),
            **assisted_supported_diagnostic,
            "action": "separate the supported-road use case from unsupported-road reconstruction",
        },
        {
            "decision": "sensor_free_dynamic_block_adds_material_skill",
            "pass": sensor_free_added_value,
            "evidence": (
                f"improvement={free_vs_structural['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={free_vs_structural['cluster_bootstrap_high']:.1f}; "
                f"improved_folds={int(free_vs_structural['improved_fold_count'])}/5"
            ),
            **sensor_free_diagnostic,
            "action": "retain dynamic block for unsupported-road reconstruction only if true",
        },
        {
            "decision": "sensor_free_model_materially_beats_honest_hierarchy_lookup",
            "pass": useful_skill,
            "evidence": (
                f"improvement={free_vs_lookup['mae_improvement_pct']:.2f}%; "
                f"upper_cluster_interval={free_vs_lookup['cluster_bootstrap_high']:.1f}"
            ),
            **useful_skill_diagnostic,
            "action": "require at least 5% improvement beyond the earlier 3-4% gain",
        },
        {
            "decision": "sensor_free_aggregate_bias_is_acceptable",
            "pass": aggregate_gate,
            "evidence": f"aggregate_bias={free_summary['aggregate_bias_pct']:.2f}%",
            "failed_criterion": "none" if aggregate_gate else "aggregate_bias_exceeds_10pct",
            "action": "require absolute pooled bias no greater than 10%",
        },
        {
            "decision": "sensor_free_region_and_road_class_bias_is_acceptable",
            "pass": subgroup_gate,
            "evidence": f"maximum_absolute_subgroup_bias={max_subgroup_bias:.2f}%",
            "failed_criterion": "none" if subgroup_gate else "subgroup_bias_exceeds_15pct",
            "action": "require every region and major/minor aggregate bias within 15%",
        },
        {
            "decision": "2023_full_network_reconstruction_gate",
            "pass": full_gate,
            "evidence": (
                f"data_gate={sampling_gate}; dynamic_increment={sensor_free_added_value}; useful_skill={useful_skill}; "
                f"network_feature_gate={network_generation_gate}; lineage_gate={lineage_gate}; "
                f"aggregate_bias_gate={aggregate_gate}; subgroup_bias_gate={subgroup_gate}"
            ),
            "failed_criterion": ";".join(full_failures) if full_failures else "none",
            "action": (
                "proceed to a separately validated 2023 equity proof of concept"
                if full_gate
                else "do not claim full-network reconstruction; report which component gate failed"
            ),
        },
        {
            "decision": "step22_establishes_multiyear_backcasting",
            "pass": False,
            "evidence": "Step 22 is a 2023 cross-sectional reconstruction experiment",
            "failed_criterion": "no_multiyear_segment_level_validation",
            "action": "historical extension remains conditional on equivalent year-specific inputs and temporal tests",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, DECISION_PATH)
    return frame


def plot_model_comparison(summary: pd.DataFrame) -> None:
    ordered = summary.set_index("model").loc[list(MODEL_ORDER)].reset_index()
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    bars = axis.bar(
        np.arange(len(ordered)),
        ordered["mae"],
        color=[MODEL_COLORS[value] for value in ordered["model"]],
    )
    axis.set_xticks(
        np.arange(len(ordered)),
        [MODEL_LABELS[value] for value in ordered["model"]],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel("Spatial OOF MAE (vehicles/day)")
    axis.set_title("Step 22: 2023 reconstruction models are compared on identical held-out stations")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, ordered["mae"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(MODEL_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {MODEL_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_observed_predicted(predictions: pd.DataFrame) -> None:
    models = (
        "deployable_structural_gtfs_hgb",
        "deployable_sensor_assisted_hgb",
        "deployable_sensor_free_hgb",
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    minimum = max(100.0, predictions["observed_aadt"].min() * 0.8)
    maximum = max(predictions["observed_aadt"].max(), predictions["predicted_aadt"].max()) * 1.1
    for axis, model_name in zip(axes, models):
        group = predictions[predictions["model"] == model_name]
        axis.scatter(
            group["observed_aadt"],
            group["predicted_aadt"],
            s=11,
            alpha=0.48,
            color=MODEL_COLORS[model_name],
        )
        axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#333333", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Observed ATC 2023 AADT")
        axis.grid(alpha=0.16, which="both")
    axes[0].set_ylabel("Spatial OOF predicted AADT")
    figure.suptitle("Local detectors help only if improvement survives the masked task")
    figure.tight_layout()
    figure.savefig(SCATTER_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {SCATTER_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_fold_gain(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(index="spatial_fold", columns="model", values="mae")
    assisted = 100.0 * (
        pivot["deployable_structural_gtfs_hgb"]
        - pivot["deployable_sensor_assisted_hgb"]
    ) / pivot["deployable_structural_gtfs_hgb"]
    free = 100.0 * (
        pivot["deployable_structural_gtfs_hgb"]
        - pivot["deployable_sensor_free_hgb"]
    ) / pivot["deployable_structural_gtfs_hgb"]
    x_values = np.arange(len(pivot))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(x_values - width / 2, assisted, width, color=MODEL_COLORS["deployable_sensor_assisted_hgb"], label="sensor assisted")
    axis.bar(x_values + width / 2, free, width, color=MODEL_COLORS["deployable_sensor_free_hgb"], label="sensor free")
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x_values, [f"Fold {value}" for value in pivot.index])
    axis.set_ylabel("MAE improvement over structure + GTFS (%)")
    axis.set_title("Dynamic-feature gain must be spatially repeatable")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FOLD_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {FOLD_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_subgroup_bias(subgroups: pd.DataFrame) -> None:
    selected = subgroups[
        subgroups["model"].isin(
            [
                "deployable_structural_gtfs_hgb",
                "deployable_sensor_assisted_hgb",
                "deployable_sensor_free_hgb",
            ]
        )
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ].copy()
    strata = list(dict.fromkeys(selected["stratum"]))
    x_values = np.arange(len(strata))
    width = 0.25
    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    for index, model_name in enumerate(
        (
            "deployable_structural_gtfs_hgb",
            "deployable_sensor_assisted_hgb",
            "deployable_sensor_free_hgb",
        )
    ):
        values = selected[selected["model"] == model_name].set_index("stratum").reindex(strata)["aggregate_bias_pct"]
        axis.bar(
            x_values + (index - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.axhline(15, color="#A33", linewidth=0.8, linestyle="--")
    axis.axhline(-15, color="#A33", linewidth=0.8, linestyle="--")
    axis.set_xticks(x_values, [value.replace("region:", "").replace("road_network:", "") for value in strata], rotation=20, ha="right")
    axis.set_ylabel("Aggregate bias (%)")
    axis.set_title("A pooled gain is insufficient if regional or road-class bias remains large")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(BIAS_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {BIAS_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def validate_inputs() -> None:
    for path in (
        MEASURED_PANEL_PATH,
        STEP21_SUPPORT_PATH,
        STEP21_DECISION_PATH,
        STEP15_TRAINING_PATH,
        STEP15_CORE_LONG_PATH,
        STEP15_SUMMARY_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 22 input: {path.relative_to(PROJECT_ROOT)}")
    decisions = pd.read_csv(STEP21_DECISION_PATH)
    authorised = decisions.loc[
        decisions["decision"] == "step22_2023_reconstruction_experiment_is_authorised",
        "pass",
    ]
    if authorised.empty or not bool(authorised.iloc[0]):
        raise RuntimeError("Step 21 did not authorise the 2023 reconstruction experiment")


def main() -> None:
    validate_inputs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Building the predeclared 2023 temporal sample...")
    sampling_audit = obtain_temporal_samples()
    save_csv(sampling_audit, SAMPLING_AUDIT_PATH)
    strategic_features = parse_strategic_samples(sampling_audit)
    class_features = parse_vehicle_class_samples(sampling_audit)

    print("Extracting the mid-2023 official road-network snapshot...")
    geodatabase = obtain_road_network()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Measured .* geometry types are not supported.*")
        centerline, geometries = read_centerline(geodatabase)
    stations = measured_2023_stations()
    road_matches, road_audit = match_stations_to_road_network(
        stations,
        centerline,
        geometries,
    )
    network_base = build_network_base_features(centerline)

    print("Building full-centreline weekday and Saturday GTFS context...")
    gtfs_path, gtfs_timestamp = obtain_gtfs()
    gtfs_features, _ = gtfs_stop_features(
        gtfs_path,
        network_base,
        gtfs_timestamp,
        "road_2023_segment_index",
    )

    strategic, vehicle_class = prepare_sensor_locations(
        strategic_features,
        class_features,
    )
    strategic = assign_sensor_folds(strategic, stations)
    vehicle_class = assign_sensor_folds(vehicle_class, stations)

    (
        network_feature_table,
        deployable_structural_features,
        gtfs_columns,
        dynamic_columns,
        strategic_value_columns,
        class_value_columns,
    ) = build_network_feature_table(
        network_base,
        gtfs_features,
        strategic,
        vehicle_class,
    )
    save_csv(network_feature_table, NETWORK_FEATURE_TABLE_PATH)
    feature_table, oracle_features = build_station_feature_table(
        stations,
        road_matches,
        network_feature_table,
    )
    support = pd.read_csv(STEP21_SUPPORT_PATH)[
        ["station_id", "strategic_credible_nearby_sensor"]
    ]
    feature_table = feature_table.merge(support, on="station_id", how="left")
    save_csv(feature_table, FEATURE_TABLE_PATH)
    feature_manifest = feature_deployability_manifest(
        network_feature_table,
        deployable_structural_features,
        gtfs_columns,
        dynamic_columns,
        oracle_features,
    )

    print("Running the frozen five-fold deployable 2023 reconstruction comparison...")
    predictions, metrics, _ = run_spatial_experiment(
        feature_table,
        deployable_structural_features,
        oracle_features,
        gtfs_columns,
        dynamic_columns,
        strategic,
        vehicle_class,
        strategic_value_columns,
        class_value_columns,
    )
    summary = summarise_models(predictions)
    subgroups = subgroup_bias(predictions)
    predicted_bin_calibration(predictions)
    comparisons = paired_comparisons(predictions)
    feature_counts = {
        "hierarchy_lookup": 2,
        "atc_class_oracle_hgb": len(oracle_features),
        "deployable_structural_hgb": len(deployable_structural_features),
        "deployable_structural_gtfs_hgb": len(deployable_structural_features)
        + len(gtfs_columns),
        "deployable_sensor_assisted_hgb": len(deployable_structural_features)
        + len(gtfs_columns)
        + len(dynamic_columns),
        "deployable_sensor_free_hgb": len(deployable_structural_features)
        + len(gtfs_columns)
        + len(dynamic_columns),
    }
    deployability_ablation(summary, subgroups, feature_counts)
    step15_step22_comparability_audit(feature_table, summary)
    decisions = decision_audit(
        summary,
        subgroups,
        comparisons,
        road_audit,
        strategic_features,
        class_features,
        feature_manifest,
        network_feature_table,
    )

    plot_model_comparison(summary)
    plot_observed_predicted(predictions)
    plot_fold_gain(metrics)
    plot_subgroup_bias(subgroups)
    update_report_manifest()

    model_summary = summary.set_index("model")
    full_gate = decisions.loc[
        decisions["decision"] == "2023_full_network_reconstruction_gate",
        "pass",
    ].iloc[0]
    print("\nStep 22 2023 dynamic reconstruction experiment is complete.")
    for model_name in (
        "hierarchy_lookup",
        "atc_class_oracle_hgb",
        "deployable_structural_gtfs_hgb",
        "deployable_sensor_assisted_hgb",
        "deployable_sensor_free_hgb",
    ):
        row = model_summary.loc[model_name]
        print(
            f"  {MODEL_LABELS[model_name]}: MAE {row['mae']:,.0f}; "
            f"R2 {row['r2']:.3f}; aggregate bias {row['aggregate_bias_pct']:+.1f}%."
        )
    print(
        "  Decision: "
        + (
            "the bounded 2023 full-network gate passes"
            if full_gate
            else "the bounded 2023 full-network gate does not pass"
        )
        + "."
    )
    print(
        "  The ATC-class oracle is diagnostic only and is excluded from the "
        "full-network decision."
    )
    print("  This result does not establish multi-year segment backcasting.")


if __name__ == "__main__":
    main()
