"""Step 21: 2023 reconstruction-data and leakage gate.

This step does not fit an AADT model.  It asks whether a defensible 2023 model
experiment can be assembled from public data without recycling the ATC target
through Niu-derived products.

The script:

1. queries official DATA.GOV.HK metadata and 2023 historical-file archives;
2. downloads three representative strategic-detector and vehicle-class files;
3. audits a mid-year 2023 GTFS snapshot;
4. measures proximity between all measured 2023 ATC stations and public traffic
   sensors;
5. freezes the role of every candidate source before Step 22.

The representative files validate schema and spatial support only.  They are not
annual summaries.  Step 22 must use a predeclared temporal sample and must report
sensor-assisted interpolation separately from sensor-free extrapolation.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hk_aadt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "step21_2023"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

MEASURED_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
STEP20_STATION_PATH = PROCESSED_DIR / "atc_step20_niu_station_crosswalk.csv"
STEP20_SOURCE_PATH = TABLE_DIR / "step20_niu_source_audit.csv"

SOURCE_INVENTORY_PATH = TABLE_DIR / "step21_2023_source_inventory.csv"
ARCHIVE_AUDIT_PATH = TABLE_DIR / "step21_2023_archive_audit.csv"
DETECTOR_SAMPLE_PATH = TABLE_DIR / "step21_strategic_detector_sample_audit.csv"
VEHICLE_CLASS_SAMPLE_PATH = TABLE_DIR / "step21_vehicle_class_sample_audit.csv"
GTFS_AUDIT_PATH = TABLE_DIR / "step21_gtfs_2023_snapshot_audit.csv"
SENSOR_SUPPORT_SUMMARY_PATH = TABLE_DIR / "step21_sensor_support_summary.csv"
LEAKAGE_MATRIX_PATH = TABLE_DIR / "step21_source_leakage_matrix.csv"
DECISION_PATH = TABLE_DIR / "step21_decision_audit.csv"
STATION_SENSOR_PATH = PROCESSED_DIR / "atc_step21_2023_station_sensor_support.csv"

ROLE_FIGURE_PATH = FIGURE_DIR / "step21_2023_source_role_matrix.png"
SENSOR_FIGURE_PATH = FIGURE_DIR / "step21_2023_sensor_support.png"
ARCHIVE_FIGURE_PATH = FIGURE_DIR / "step21_2023_archive_scale.png"

CKAN_ROOT = "https://data.gov.hk/en-data/api/3/action/package_show"
ARCHIVE_LIST_ROOT = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
ARCHIVE_GET_ROOT = "https://app.data.gov.hk/v1/historical-archive/get-file"

STRATEGIC_DATASET_ID = "hk-td-sm_4-traffic-data-strategic-major-roads"
VEHICLE_CLASS_DATASET_ID = "hk-td-sm_5-annual-traffic-census-survey-data"
ROAD_NETWORK_DATASET_ID = "hk-td-tis_15-road-network-v2"
GTFS_DATASET_ID = "hk-td-tis_11-pt-headway-en"
AI_CCTV_DATASET_ID = "hk-td-tis_32-traffic-data-aivas"
DCCA_DATASET_ID = "hk-censtatd-census_geo-2021-population-census-by-dcca"
SSG_DATASET_ID = "hk-censtatd-census_geo-2021-population-census-by-ssg"
JOURNEY_TIME_DATASET_ID = "hk-td-tis_34-car-journey-time-data"

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

SAMPLE_DATES = ("20230115", "20230615", "20231215")
TARGET_SAMPLE_TIME = 1200
PROJECTED_CRS = "EPSG:2326"


ROLE_COLUMNS = (
    "outcome_label",
    "structural_predictor",
    "sensor_assisted_predictor",
    "external_diagnostic",
    "equity_input",
)


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


def fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def ckan_package(dataset_id: str) -> dict[str, object]:
    url = CKAN_ROOT + "?" + urllib.parse.urlencode({"id": dataset_id})
    payload = fetch_json(url)
    if not payload.get("success"):
        raise RuntimeError(f"DATA.GOV.HK metadata request failed for {dataset_id}")
    return payload["result"]


def archive_versions(resource_url: str) -> dict[str, object]:
    query = urllib.parse.urlencode(
        {"url": resource_url, "start": "20230101", "end": "20231231"}
    )
    return fetch_json(f"{ARCHIVE_LIST_ROOT}?{query}")


def archive_versions_for_day(resource_url: str, date: str) -> dict[str, object]:
    query = urllib.parse.urlencode({"url": resource_url, "start": date, "end": date})
    return fetch_json(f"{ARCHIVE_LIST_ROOT}?{query}")


def closest_timestamp(timestamps: list[str], date: str) -> str:
    if not timestamps:
        raise RuntimeError(f"No archived version is available for {date}")

    def distance(value: str) -> tuple[int, str]:
        time_match = re.search(r"-(\d{4})$", value)
        if not time_match:
            return (24 * 60, value)
        hhmm = int(time_match.group(1))
        minutes = (hhmm // 100) * 60 + hhmm % 100
        target_minutes = (TARGET_SAMPLE_TIME // 100) * 60 + TARGET_SAMPLE_TIME % 100
        return (abs(minutes - target_minutes), value)

    matching_date = [value for value in timestamps if value.startswith(date)]
    if not matching_date:
        matching_date = timestamps
    return min(matching_date, key=distance)


def download_url(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return path
    with urllib.request.urlopen(url, timeout=180) as response, path.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    return path


def download_historical(resource_url: str, timestamp: str, path: Path) -> Path:
    query = urllib.parse.urlencode({"url": resource_url, "time": timestamp})
    return download_url(f"{ARCHIVE_GET_ROOT}?{query}", path)


def historical_summary(
    source: str,
    dataset_id: str,
    resource_url: str,
    payload: dict[str, object],
) -> dict[str, object]:
    data_files = payload.get("data-files", [])
    return {
        "source": source,
        "dataset_id": dataset_id,
        "resource_url": resource_url,
        "archive_version_count_2023": int(payload.get("version-count", 0)),
        "archive_monthly_package_count_2023": int(payload.get("version-count-zip", 0)),
        "raw_archive_size_gb_2023": float(payload.get("total-size", 0)) / 1e9,
        "monthly_packages_size_gb_2023": float(payload.get("total-size-zip", 0)) / 1e9,
        "first_timestamp_2023": (
            payload.get("timestamps", [np.nan])[0]
            if payload.get("timestamps")
            else np.nan
        ),
        "last_monthly_package_2023": (
            data_files[-1].get("timestamp", np.nan) if data_files else np.nan
        ),
        "archive_status": (
            "verified_2023_versions" if int(payload.get("version-count", 0)) > 0 else "no_2023_versions"
        ),
    }


def parse_strategic_snapshot(path: Path, timestamp: str) -> tuple[dict[str, object], set[str]]:
    root = ET.parse(path).getroot()
    detectors = root.findall(".//detector")
    lanes = root.findall(".//lane")
    detector_ids = {
        element.findtext("detector_id", default="").strip()
        for element in detectors
        if element.findtext("detector_id")
    }
    valid_lanes = [lane for lane in lanes if lane.findtext("valid") == "Y"]
    numeric_volume = [
        float(lane.findtext("volume"))
        for lane in valid_lanes
        if lane.findtext("volume") not in (None, "")
    ]
    numeric_speed = [
        float(lane.findtext("speed"))
        for lane in valid_lanes
        if lane.findtext("speed") not in (None, "")
    ]
    return (
        {
            "sample_timestamp": timestamp,
            "source_date": root.findtext("date"),
            "period_count": len(root.findall(".//period")),
            "unique_detector_count": len(detector_ids),
            "detector_period_records": len(detectors),
            "lane_records": len(lanes),
            "valid_lane_share": len(valid_lanes) / len(lanes),
            "sample_volume_sum": np.sum(numeric_volume),
            "median_valid_lane_speed": np.median(numeric_speed),
            "interpretation": "schema_and_support_sample_not_annual_summary",
        },
        detector_ids,
    )


def parse_vehicle_class_snapshot(
    path: Path,
    timestamp: str,
) -> tuple[dict[str, object], set[str]]:
    root = ET.parse(path).getroot()
    detectors = root.findall(".//detector")
    detector_ids = {
        element.findtext("detector_id", default="").strip()
        for element in detectors
        if element.findtext("detector_id")
    }
    valid_detectors = [detector for detector in detectors if detector.findtext("valid") == "Y"]
    class_names = sorted(
        {
            element.findtext("class_name", default="").strip()
            for element in root.findall(".//class")
            if element.findtext("class_name")
        }
    )
    period_count = len(root.findall(".//period"))
    return (
        {
            "sample_timestamp": timestamp,
            "source_date": root.findtext("date"),
            "period_count": period_count,
            "unique_detector_count": len(detector_ids),
            "detector_period_records": len(detectors),
            "valid_detector_record_share": len(valid_detectors) / len(detectors),
            "class_field_count": len(class_names),
            "class_fields": "|".join(class_names),
            "interpretation": "hourly_class_proportions_schema_sample_not_annual_summary",
        },
        detector_ids,
    )


def audit_gtfs(path: Path, timestamp: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for filename, id_column in (
            ("agency.txt", "agency_id"),
            ("routes.txt", "route_id"),
            ("trips.txt", "trip_id"),
            ("stops.txt", "stop_id"),
            ("stop_times.txt", "trip_id"),
            ("frequencies.txt", "trip_id"),
            ("calendar.txt", "service_id"),
            ("calendar_dates.txt", "service_id"),
        ):
            if filename not in members:
                rows.append(
                    {
                        "sample_timestamp": timestamp,
                        "gtfs_member": filename,
                        "row_count": 0,
                        "unique_primary_ids": 0,
                        "status": "missing",
                    }
                )
                continue
            with archive.open(filename) as source:
                frame = pd.read_csv(source, low_memory=False)
            rows.append(
                {
                    "sample_timestamp": timestamp,
                    "gtfs_member": filename,
                    "row_count": len(frame),
                    "unique_primary_ids": (
                        frame[id_column].nunique() if id_column in frame.columns else np.nan
                    ),
                    "status": "available",
                }
            )
    return pd.DataFrame(rows)


def normalise_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in GENERIC_ROAD_TOKENS]
    return " ".join(tokens)


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


def prepare_sensor_locations() -> tuple[pd.DataFrame, pd.DataFrame]:
    strategic_path = download_url(
        STRATEGIC_LOCATION_URL,
        RAW_DIR / "strategic_detector_locations.csv",
    )
    class_path = download_url(
        VEHICLE_CLASS_LOCATION_URL,
        RAW_DIR / "vehicle_class_detector_locations.csv",
    )
    strategic = pd.read_csv(strategic_path).rename(
        columns={"AID_ID_Number": "sensor_id"}
    )
    vehicle_class = pd.read_csv(class_path).rename(columns={"Device_ID": "sensor_id"})
    strategic["sensor_source"] = "strategic_detector"
    vehicle_class["sensor_source"] = "vehicle_class_detector"
    return strategic, vehicle_class


def nearest_sensor_fields(
    stations: pd.DataFrame,
    sensors: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    sensor_x, sensor_y = transformer.transform(
        sensors["Longitude"].to_numpy(dtype=float),
        sensors["Latitude"].to_numpy(dtype=float),
    )
    tree = cKDTree(np.column_stack([sensor_x, sensor_y]))

    valid = stations["longitude"].notna() & stations["latitude"].notna()
    station_x, station_y = transformer.transform(
        stations.loc[valid, "longitude"].to_numpy(dtype=float),
        stations.loc[valid, "latitude"].to_numpy(dtype=float),
    )
    distances, indices = tree.query(np.column_stack([station_x, station_y]), k=1)
    result = pd.DataFrame(index=stations.index)
    result[f"nearest_{prefix}_distance_m"] = np.nan
    result[f"nearest_{prefix}_sensor_id"] = pd.Series(
        pd.NA,
        index=stations.index,
        dtype="string",
    )
    result[f"nearest_{prefix}_road_name"] = pd.Series(
        pd.NA,
        index=stations.index,
        dtype="string",
    )
    result[f"nearest_{prefix}_name_similarity"] = np.nan
    result.loc[valid, f"nearest_{prefix}_distance_m"] = distances
    result.loc[valid, f"nearest_{prefix}_sensor_id"] = sensors.iloc[indices][
        "sensor_id"
    ].astype(str).to_numpy()
    result.loc[valid, f"nearest_{prefix}_road_name"] = sensors.iloc[indices][
        "Road_EN"
    ].to_numpy()
    result.loc[valid, f"nearest_{prefix}_name_similarity"] = [
        name_similarity(station_name, sensor_name)
        for station_name, sensor_name in zip(
            stations.loc[valid, "road_name"],
            sensors.iloc[indices]["Road_EN"],
        )
    ]
    return result


def build_station_sensor_support(
    strategic: pd.DataFrame,
    vehicle_class: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(MEASURED_PANEL_PATH)
    stations = (
        panel[panel["year"].astype(int) == 2023]
        .drop_duplicates("station_id")
        .copy()
        .reset_index(drop=True)
    )
    strategic_fields = nearest_sensor_fields(stations, strategic, "strategic")
    class_fields = nearest_sensor_fields(stations, vehicle_class, "vehicle_class")
    support = pd.concat(
        [
            stations[
                [
                    "station_id",
                    "region",
                    "road_network",
                    "road_type",
                    "road_name",
                    "longitude",
                    "latitude",
                    "aadt",
                ]
            ],
            strategic_fields,
            class_fields,
        ],
        axis=1,
    )
    for prefix in ("strategic", "vehicle_class"):
        distance = support[f"nearest_{prefix}_distance_m"]
        similarity = support[f"nearest_{prefix}_name_similarity"]
        support[f"{prefix}_credible_nearby_sensor"] = (
            ((distance <= 100.0) & (similarity >= 0.15)) | (distance <= 25.0)
        )
    save_csv(support, STATION_SENSOR_PATH)

    rows: list[dict[str, object]] = []
    strata = [("all", support)]
    strata.extend((f"region:{name}", group) for name, group in support.groupby("region"))
    strata.extend(
        (f"road_network:{name}", group)
        for name, group in support.groupby("road_network")
    )
    for source in ("strategic", "vehicle_class"):
        distance_column = f"nearest_{source}_distance_m"
        for stratum, group in strata:
            for threshold in (25, 50, 100, 250):
                rows.append(
                    {
                        "sensor_source": source,
                        "stratum": stratum,
                        "distance_threshold_m": threshold,
                        "station_count": len(group),
                        "station_share_within_threshold": np.mean(
                            group[distance_column] <= threshold
                        ),
                        "median_nearest_sensor_distance_m": group[
                            distance_column
                        ].median(),
                        "credible_nearby_sensor_share": group[
                            f"{source}_credible_nearby_sensor"
                        ].mean(),
                    }
                )
    summary = pd.DataFrame(rows)
    save_csv(summary, SENSOR_SUPPORT_SUMMARY_PATH)
    return support, summary


def source_inventory(
    archive_table: pd.DataFrame,
    detector_audit: pd.DataFrame,
    class_audit: pd.DataFrame,
    gtfs_audit: pd.DataFrame,
    metadata: dict[str, dict[str, object]],
) -> pd.DataFrame:
    strategic_detectors = int(detector_audit["unique_detector_count"].median())
    class_detectors = int(class_audit["unique_detector_count"].median())
    strategic_location_id_share = detector_audit[
        "share_sample_detector_ids_in_current_location_list"
    ].iloc[0]
    class_location_id_share = class_audit[
        "share_sample_detector_ids_in_current_location_list"
    ].iloc[0]
    gtfs_routes = int(
        gtfs_audit.loc[gtfs_audit["gtfs_member"] == "routes.txt", "row_count"].iloc[0]
    )
    journey_2023_count = sum(
        "2023" in str(resource.get("name", ""))
        for resource in metadata[JOURNEY_TIME_DATASET_ID].get("resources", [])
    )
    dcca_resource_count = len(metadata[DCCA_DATASET_ID].get("resources", []))
    ssg_resource_count = len(metadata[SSG_DATASET_ID].get("resources", []))

    rows = [
        {
            "source": "ATC 2023 directly measured AADT",
            "provider": "Transport Department",
            "verified_2023_support": "880 directly measured stations in Step 18",
            "temporal_resolution": "annual AADT",
            "role": "outcome_label",
            "contains_atc_target_information": True,
            "independent_of_atc_2023": False,
            "step22_use": "target_only",
            "status": "ready",
            "limitation": "sample is road-hierarchy selective; local-road performance must be separate",
        },
        {
            "source": "Strategic/major-road detector volume speed occupancy",
            "provider": "Transport Department",
            "verified_2023_support": (
                f"2023 archive and {strategic_detectors} detectors; "
                f"{strategic_location_id_share:.1%} of sample IDs occur in current location list"
            ),
            "temporal_resolution": "30-second/minute archive",
            "role": "sensor_assisted_predictor",
            "contains_atc_target_information": True,
            "independent_of_atc_2023": True,
            "step22_use": "sensor_assisted_task_and_masked_sensor_free_task",
            "status": "ready_after_temporal_sampling",
            "limitation": "direct traffic proxy, major-road selective, full archive is large, and current coordinates are provisional",
        },
        {
            "source": "ATC detector hourly vehicle-class proportions",
            "provider": "Transport Department",
            "verified_2023_support": (
                f"daily archive and {class_detectors} detectors; "
                f"{class_location_id_share:.1%} of sample IDs occur in current location list"
            ),
            "temporal_resolution": "hourly profiles in daily files",
            "role": "sensor_assisted_predictor",
            "contains_atc_target_information": True,
            "independent_of_atc_2023": True,
            "step22_use": "vehicle_mix_feature_with_fold_location_audit",
            "status": "ready_after_temporal_sampling",
            "limitation": "direct traffic proxy; small detector network can reveal nearby targets; current coordinates are provisional",
        },
        {
            "source": "2023 public-transport GTFS",
            "provider": "Transport Department",
            "verified_2023_support": f"27 archived versions; mid-year snapshot has {gtfs_routes} route rows",
            "temporal_resolution": "biweekly schedule snapshots",
            "role": "structural_predictor",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "bus_route_and_frequency_features",
            "status": "ready",
            "limitation": "scheduled service is not realised vehicle movement",
        },
        {
            "source": "2023 Road Network 2nd Generation",
            "provider": "Transport Department",
            "verified_2023_support": "16 archived FGDB versions",
            "temporal_resolution": "monthly/as-updated snapshots",
            "role": "structural_predictor",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "2023_geometry_direction_speed_limit_and_restrictions",
            "status": "ready_after_2023_snapshot_extraction",
            "limitation": "must replace current geometry for a year-specific experiment",
        },
        {
            "source": "Niu 2023 link AADT vehicle classes and emissions",
            "provider": "Niu et al. public release",
            "verified_2023_support": "17,706 directed link rows in Step 20",
            "temporal_resolution": "2023 hourly activity profile",
            "role": "equity_input",
            "contains_atc_target_information": True,
            "independent_of_atc_2023": False,
            "step22_use": "prohibited_as_predictor_or_independent_validation",
            "status": "equity_and_descriptive_benchmark_only",
            "limitation": "shared ATC lineage; exact AADT values at most matched stations",
        },
        {
            "source": "2021 Census DCCA and SSG",
            "provider": "Census and Statistics Department",
            "verified_2023_support": f"DCCA resources={dcca_resource_count}; SSG resources={ssg_resource_count}",
            "temporal_resolution": "2021 census proxy for 2023",
            "role": "equity_input",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "population_and_income_context_not_aadt_target",
            "status": "ready_with_date_label",
            "limitation": "2021 to 2023 temporal mismatch and cross-scale income/population allocation",
        },
        {
            "source": "EPD hourly NOx NO2 and PM2.5 monitoring",
            "provider": "Environmental Protection Department",
            "verified_2023_support": "official download system includes 2023 hourly data",
            "temporal_resolution": "hourly monitoring stations",
            "role": "external_diagnostic",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "emissions_plausibility_only",
            "status": "ready_for_exposure_stage",
            "limitation": "concentrations are affected by meteorology dispersion and regional background",
        },
        {
            "source": "2023 car journey time tables",
            "provider": "Transport Department",
            "verified_2023_support": f"{journey_2023_count} named 2023 resources",
            "temporal_resolution": "AM/PM route summaries",
            "role": "external_diagnostic",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "congestion_pattern_diagnostic_only",
            "status": "ready",
            "limitation": "journey time is not an AADT label",
        },
        {
            "source": "AI video analytics traffic volume and speed",
            "provider": "Transport Department",
            "verified_2023_support": "CSDI API resource exists from August 2023",
            "temporal_resolution": "15-minute current/API product",
            "role": "external_diagnostic",
            "contains_atc_target_information": False,
            "independent_of_atc_2023": True,
            "step22_use": "reserve_until_2023_archive_and_site_history_are_verified",
            "status": "blocked_archive_not_exposed_as_file_resource",
            "limitation": "partial-year coverage and historical observations not yet retrieved",
        },
    ]
    inventory = pd.DataFrame(rows)
    save_csv(inventory, SOURCE_INVENTORY_PATH)
    return inventory


def leakage_matrix() -> pd.DataFrame:
    rows = [
        {
            "data_family": "ATC 2023 AADT",
            "contains_target_or_direct_proxy": True,
            "permitted_model_role": "outcome_only",
            "independent_validation_role": "none",
            "fold_rule": "held-out stations never enter training summaries or feature construction",
            "reason": "this is the prediction target",
        },
        {
            "data_family": "Niu link AADT",
            "contains_target_or_direct_proxy": True,
            "permitted_model_role": "none",
            "independent_validation_role": "none",
            "fold_rule": "exclude from every Step 22 feature and score",
            "reason": "public product uses ATC 2023 and reproduces most matched station values exactly",
        },
        {
            "data_family": "Niu hourly class volumes and NOx/PM2.5",
            "contains_target_or_direct_proxy": True,
            "permitted_model_role": "post-model equity benchmark only",
            "independent_validation_role": "none",
            "fold_rule": "do not derive AADT features from these fields",
            "reason": "traffic and emissions are downstream of the same AADT surface",
        },
        {
            "data_family": "Strategic detector volume speed occupancy",
            "contains_target_or_direct_proxy": True,
            "permitted_model_role": "sensor-assisted predictor",
            "independent_validation_role": "only at sites never used as predictors",
            "fold_rule": "report keep-sensors and mask-sensors-within-held-out-region tasks separately",
            "reason": "a nearby live volume sensor can nearly reveal the held-out road target",
        },
        {
            "data_family": "ATC detector vehicle-class proportions",
            "contains_target_or_direct_proxy": True,
            "permitted_model_role": "sensor-assisted vehicle-mix predictor",
            "independent_validation_role": "none at colocated stations",
            "fold_rule": "audit location overlap and mask within held-out-region task",
            "reason": "traffic-derived proportions may identify sensor-supported roads",
        },
        {
            "data_family": "GTFS routes and headways",
            "contains_target_or_direct_proxy": False,
            "permitted_model_role": "structural predictor",
            "independent_validation_role": "not a traffic truth set",
            "fold_rule": "derive features without station labels inside every fold",
            "reason": "scheduled service is public context rather than measured AADT",
        },
        {
            "data_family": "2023 Road Network attributes",
            "contains_target_or_direct_proxy": False,
            "permitted_model_role": "structural predictor",
            "independent_validation_role": "not a traffic truth set",
            "fold_rule": "same 2023 snapshot for all folds",
            "reason": "geometry and restrictions do not encode measured AADT",
        },
        {
            "data_family": "Census population and income",
            "contains_target_or_direct_proxy": False,
            "permitted_model_role": "equity weighting and sensitivity",
            "independent_validation_role": "equity estimand only",
            "fold_rule": "do not tune traffic model to maximise an income contrast",
            "reason": "socioeconomic outcomes must remain downstream of traffic validation",
        },
        {
            "data_family": "EPD pollutant concentration",
            "contains_target_or_direct_proxy": False,
            "permitted_model_role": "external exposure plausibility diagnostic",
            "independent_validation_role": "emissions pattern only",
            "fold_rule": "do not score concentration as if it were AADT",
            "reason": "meteorology dispersion chemistry and background separate concentration from traffic",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, LEAKAGE_MATRIX_PATH)
    return frame


def plot_role_matrix(inventory: pd.DataFrame) -> None:
    role_values = np.zeros((len(inventory), len(ROLE_COLUMNS)), dtype=int)
    role_to_column = {role: index for index, role in enumerate(ROLE_COLUMNS)}
    for row, role in enumerate(inventory["role"]):
        if role in role_to_column:
            role_values[row, role_to_column[role]] = 1

    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    axis.imshow(role_values, cmap=ListedColormap(["#F3F5F7", "#176B87"]), vmin=0, vmax=1)
    axis.set_xticks(
        range(len(ROLE_COLUMNS)),
        [value.replace("_", " ") for value in ROLE_COLUMNS],
        rotation=28,
        ha="right",
    )
    axis.set_yticks(range(len(inventory)), inventory["source"])
    for row in range(role_values.shape[0]):
        for column in range(role_values.shape[1]):
            axis.text(
                column,
                row,
                "use" if role_values[row, column] else "",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
    axis.set_title("Step 21: one primary role per 2023 source prevents leakage")
    figure.tight_layout()
    figure.savefig(ROLE_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {ROLE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_sensor_support(
    support: pd.DataFrame,
    strategic: pd.DataFrame,
    vehicle_class: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].scatter(
        support["longitude"],
        support["latitude"],
        s=7,
        color="#999999",
        alpha=0.45,
        label="ATC 2023 station",
    )
    axes[0].scatter(
        strategic["Longitude"],
        strategic["Latitude"],
        s=8,
        color="#1B9E77",
        alpha=0.65,
        label="strategic detector",
    )
    axes[0].scatter(
        vehicle_class["Longitude"],
        vehicle_class["Latitude"],
        s=18,
        facecolors="none",
        edgecolors="#D95F02",
        linewidths=0.8,
        label="vehicle-class detector",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Measured AADT and public dynamic-sensor support")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.15)

    all_summary = summary[
        (summary["stratum"] == "all")
        & (summary["distance_threshold_m"].isin([25, 50, 100, 250]))
    ].copy()
    thresholds = [25, 50, 100, 250]
    x_values = np.arange(len(thresholds))
    width = 0.36
    for offset, source, color in (
        (-width / 2, "strategic", "#1B9E77"),
        (width / 2, "vehicle_class", "#D95F02"),
    ):
        values = [
            100.0
            * all_summary.loc[
                (all_summary["sensor_source"] == source)
                & (all_summary["distance_threshold_m"] == threshold),
                "station_share_within_threshold",
            ].iloc[0]
            for threshold in thresholds
        ]
        axes[1].bar(x_values + offset, values, width, label=source.replace("_", " "), color=color)
    axes[1].set_xticks(x_values, [f"{value} m" for value in thresholds])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Share of measured 2023 ATC stations (%)")
    axes[1].set_title("Proximity is support, not independent validation")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(SENSOR_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {SENSOR_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_archive_scale(archive_table: pd.DataFrame) -> None:
    ordered = archive_table.sort_values("raw_archive_size_gb_2023", ascending=True)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.barh(
        ordered["source"],
        ordered["raw_archive_size_gb_2023"],
        color="#4C78A8",
    )
    axis.set_xscale("log")
    axis.set_xlabel("2023 historical archive size (GB, logarithmic scale)")
    axis.set_title("Archive existence does not imply pilot-scale ingestibility")
    axis.grid(axis="x", alpha=0.2, which="both")
    for bar, value in zip(bars, ordered["raw_archive_size_gb_2023"]):
        axis.text(
            max(value * 1.08, 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g} GB",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(ARCHIVE_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {ARCHIVE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def decision_audit(
    archive_table: pd.DataFrame,
    detector_audit: pd.DataFrame,
    class_audit: pd.DataFrame,
    gtfs_audit: pd.DataFrame,
    sensor_summary: pd.DataFrame,
) -> pd.DataFrame:
    archive = archive_table.set_index("source")
    strategic_count = int(archive.loc["strategic_detector", "archive_version_count_2023"])
    class_count = int(archive.loc["vehicle_class_detector", "archive_version_count_2023"])
    gtfs_count = int(archive.loc["gtfs", "archive_version_count_2023"])
    road_count = int(archive.loc["road_network", "archive_version_count_2023"])
    strategic_sensor_count = int(detector_audit["unique_detector_count"].median())
    class_sensor_count = int(class_audit["unique_detector_count"].median())
    strategic_location_id_share = detector_audit[
        "share_sample_detector_ids_in_current_location_list"
    ].iloc[0]
    class_location_id_share = class_audit[
        "share_sample_detector_ids_in_current_location_list"
    ].iloc[0]
    gtfs_complete = (gtfs_audit["status"] == "available").all()
    support_100 = sensor_summary[
        (sensor_summary["sensor_source"] == "strategic")
        & (sensor_summary["stratum"] == "all")
        & (sensor_summary["distance_threshold_m"] == 100)
    ]["station_share_within_threshold"].iloc[0]

    data_gate = (
        strategic_count > 300_000
        and class_count >= 300
        and gtfs_count >= 20
        and road_count >= 10
        and strategic_sensor_count >= 500
        and class_sensor_count >= 50
        and strategic_location_id_share >= 0.95
        and class_location_id_share >= 0.95
        and gtfs_complete
    )
    rows = [
        {
            "decision": "verified_2023_dynamic_public_data_exist_for_a_model_experiment",
            "pass": data_gate,
            "evidence": (
                f"strategic_versions={strategic_count}; class_days={class_count}; "
                f"gtfs_versions={gtfs_count}; road_snapshots={road_count}"
            ),
            "action": "proceed to a bounded temporal-sampling design for Step 22",
        },
        {
            "decision": "representative_step21_files_are_annual_predictors",
            "pass": False,
            "evidence": "three detector dates and one GTFS snapshot validate schema and support only",
            "action": "build fold-safe annual summaries from a predeclared weekday/weekend monthly sample",
        },
        {
            "decision": "current_static_detector_coordinates_are_verified_2023_positions",
            "pass": False,
            "evidence": (
                f"current list contains {strategic_location_id_share:.1%} of sampled strategic IDs and "
                f"{class_location_id_share:.1%} of sampled class-detector IDs, but no archived 2023 location file was verified"
            ),
            "action": "treat coordinates as provisional and report 25/50/100/250 m plus road-name sensitivity",
        },
        {
            "decision": "download_every_minute_of_the_strategic_archive_for_this_pilot",
            "pass": False,
            "evidence": (
                f"raw_2023_archive_gb={archive.loc['strategic_detector', 'raw_archive_size_gb_2023']:.1f}; "
                f"monthly_packages_gb={archive.loc['strategic_detector', 'monthly_packages_size_gb_2023']:.1f}"
            ),
            "action": "sample fixed dates/hours across all months and retain completeness weights",
        },
        {
            "decision": "one_2023_model_score_can_represent_all_deployment_tasks",
            "pass": False,
            "evidence": f"share_of_atc_stations_within_100m_of_strategic_sensor={support_100:.3f}",
            "action": "report sensor-assisted interpolation and masked sensor-free extrapolation separately",
        },
        {
            "decision": "niu_2023_can_be_used_as_a_step22_predictor_or_independent_test",
            "pass": False,
            "evidence": "Step 20 shows shared ATC lineage and exact AADT carry-through at most matched stations",
            "action": "reserve Niu for descriptive comparison and downstream equity only",
        },
        {
            "decision": "2021_census_can_support_a_2023_equity_proof_of_concept",
            "pass": True,
            "evidence": "public DCCA and SSG resources provide finer population and socioeconomic support",
            "action": "label it as a 2021 socioeconomic proxy and run geography/weighting sensitivity",
        },
        {
            "decision": "proceed_to_equity_before_the_2023_aadt_gate_passes",
            "pass": False,
            "evidence": "Step 21 validates inputs rather than out-of-fold AADT performance",
            "action": "run Step 22 first; only a passing traffic surface proceeds to independent equity estimates",
        },
        {
            "decision": "step22_2023_reconstruction_experiment_is_authorised",
            "pass": data_gate,
            "evidence": "outcome, dynamic sensors, vehicle mix, GTFS and year-specific road snapshots are verified",
            "action": "freeze baselines, spatial folds, temporal sample and leakage rules before fitting",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, DECISION_PATH)
    return frame


def validate_inputs() -> None:
    missing = [
        path.relative_to(PROJECT_ROOT)
        for path in (MEASURED_PANEL_PATH, STEP20_STATION_PATH, STEP20_SOURCE_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Step 21 inputs: {missing}. Complete Steps 18 and 20 first.")


def main() -> None:
    validate_inputs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata_ids = (
        STRATEGIC_DATASET_ID,
        VEHICLE_CLASS_DATASET_ID,
        ROAD_NETWORK_DATASET_ID,
        GTFS_DATASET_ID,
        AI_CCTV_DATASET_ID,
        DCCA_DATASET_ID,
        SSG_DATASET_ID,
        JOURNEY_TIME_DATASET_ID,
    )
    metadata = {dataset_id: ckan_package(dataset_id) for dataset_id in metadata_ids}

    archive_payloads = {
        "strategic_detector": archive_versions(STRATEGIC_RAW_URL),
        "vehicle_class_detector": archive_versions(VEHICLE_CLASS_URL),
        "gtfs": archive_versions(GTFS_URL),
        "road_network": archive_versions(ROAD_NETWORK_URL),
    }
    archive_table = pd.DataFrame(
        [
            historical_summary(
                "strategic_detector",
                STRATEGIC_DATASET_ID,
                STRATEGIC_RAW_URL,
                archive_payloads["strategic_detector"],
            ),
            historical_summary(
                "vehicle_class_detector",
                VEHICLE_CLASS_DATASET_ID,
                VEHICLE_CLASS_URL,
                archive_payloads["vehicle_class_detector"],
            ),
            historical_summary(
                "gtfs",
                GTFS_DATASET_ID,
                GTFS_URL,
                archive_payloads["gtfs"],
            ),
            historical_summary(
                "road_network",
                ROAD_NETWORK_DATASET_ID,
                ROAD_NETWORK_URL,
                archive_payloads["road_network"],
            ),
        ]
    )
    save_csv(archive_table, ARCHIVE_AUDIT_PATH)

    detector_rows: list[dict[str, object]] = []
    detector_sample_ids: set[str] = set()
    class_rows: list[dict[str, object]] = []
    class_sample_ids: set[str] = set()
    for date in SAMPLE_DATES:
        strategic_day = archive_versions_for_day(STRATEGIC_RAW_URL, date)
        strategic_timestamp = closest_timestamp(strategic_day.get("timestamps", []), date)
        strategic_path = download_historical(
            STRATEGIC_RAW_URL,
            strategic_timestamp,
            RAW_DIR / f"strategic_{strategic_timestamp}.xml",
        )
        row, ids = parse_strategic_snapshot(strategic_path, strategic_timestamp)
        detector_rows.append(row)
        detector_sample_ids.update(ids)

        class_day = archive_versions_for_day(VEHICLE_CLASS_URL, date)
        class_timestamp = closest_timestamp(class_day.get("timestamps", []), date)
        class_path = download_historical(
            VEHICLE_CLASS_URL,
            class_timestamp,
            RAW_DIR / f"vehicle_class_{class_timestamp}.xml",
        )
        row, ids = parse_vehicle_class_snapshot(class_path, class_timestamp)
        class_rows.append(row)
        class_sample_ids.update(ids)

    detector_audit = pd.DataFrame(detector_rows)
    class_audit = pd.DataFrame(class_rows)
    strategic_locations, class_locations = prepare_sensor_locations()
    detector_audit["share_sample_detector_ids_in_current_location_list"] = [
        len(detector_sample_ids & set(strategic_locations["sensor_id"].astype(str)))
        / len(detector_sample_ids)
    ] * len(detector_audit)
    class_audit["share_sample_detector_ids_in_current_location_list"] = [
        len(class_sample_ids & set(class_locations["sensor_id"].astype(str)))
        / len(class_sample_ids)
    ] * len(class_audit)
    save_csv(detector_audit, DETECTOR_SAMPLE_PATH)
    save_csv(class_audit, VEHICLE_CLASS_SAMPLE_PATH)

    gtfs_timestamps = archive_payloads["gtfs"].get("timestamps", [])
    gtfs_timestamp = min(
        gtfs_timestamps,
        key=lambda value: abs(
            (
                datetime.strptime(value, "%Y%m%d-%H%M")
                - datetime(2023, 6, 15, 12, 0)
            ).total_seconds()
        ),
    )
    gtfs_path = download_historical(
        GTFS_URL,
        gtfs_timestamp,
        RAW_DIR / f"gtfs_{gtfs_timestamp}.zip",
    )
    gtfs_audit = audit_gtfs(gtfs_path, gtfs_timestamp)
    save_csv(gtfs_audit, GTFS_AUDIT_PATH)

    support, sensor_summary = build_station_sensor_support(
        strategic_locations,
        class_locations,
    )
    inventory = source_inventory(
        archive_table,
        detector_audit,
        class_audit,
        gtfs_audit,
        metadata,
    )
    leakage_matrix()
    decisions = decision_audit(
        archive_table,
        detector_audit,
        class_audit,
        gtfs_audit,
        sensor_summary,
    )

    plot_role_matrix(inventory)
    plot_sensor_support(
        support,
        strategic_locations,
        class_locations,
        sensor_summary,
    )
    plot_archive_scale(archive_table)

    all_sensor = sensor_summary[
        (sensor_summary["sensor_source"] == "strategic")
        & (sensor_summary["stratum"] == "all")
        & (sensor_summary["distance_threshold_m"] == 100)
    ].iloc[0]
    gate = decisions.loc[
        decisions["decision"] == "step22_2023_reconstruction_experiment_is_authorised",
        "pass",
    ].iloc[0]
    print("\nStep 21 2023 reconstruction-data and leakage gate is complete.")
    print(
        "  Strategic detector archive: "
        f"{int(archive_table.loc[archive_table['source'] == 'strategic_detector', 'archive_version_count_2023'].iloc[0]):,} "
        "versions in 2023."
    )
    print(
        "  Vehicle-class archive: "
        f"{int(archive_table.loc[archive_table['source'] == 'vehicle_class_detector', 'archive_version_count_2023'].iloc[0]):,} daily files; "
        f"GTFS: {int(archive_table.loc[archive_table['source'] == 'gtfs', 'archive_version_count_2023'].iloc[0])} versions."
    )
    print(
        f"  ATC stations within 100 m of a strategic detector: {all_sensor['station_share_within_threshold']:.1%}."
    )
    print("  Niu-derived traffic and emissions are frozen out of every Step 22 feature and score.")
    print(
        "  Decision: "
        + ("proceed to Step 22 with two validation tasks" if gate else "repair the 2023 source stack before fitting")
        + "."
    )


if __name__ == "__main__":
    main()
