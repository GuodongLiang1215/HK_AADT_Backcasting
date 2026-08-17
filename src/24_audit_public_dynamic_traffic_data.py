"""Step 24: audit public dynamic traffic data before fitting another model.

This step asks three bounded questions.

1. Do the currently public detector systems add independent traffic-volume
   observations on local/service roads that are poorly represented by ATC?
2. Is there enough colocation with 2023 ATC stations to calibrate a detector
   proxy without treating a one-minute snapshot as annual AADT?
3. Has a sufficiently continuous multi-year archive actually been obtained
   locally to support a later forward temporal experiment?

The script downloads only the current location tables and one current feed
snapshot for the strategic-road, Smart Lamppost, and vehicle-class products.
It inventories existing Step 21/local archives but does not download a large
historical archive.  AIVAS is audited from the official specification: it is a
16-camera pilot in which each camera contributes a 130-second rotating view.

Passing Step 24 authorises only a pre-registered Step 25 experiment.  It never
establishes annual AADT, full-network reconstruction, or a historical trend.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "step24_public_dynamic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

LOCATION_PATH = PROCESSED_DIR / "atc_step24_public_dynamic_detector_locations.csv"
AIVAS_PATH = PROCESSED_DIR / "atc_step24_aivas_official_segment_manifest.csv"
INVENTORY_PATH = TABLE_DIR / "step24_public_dynamic_source_inventory.csv"
FEED_AUDIT_PATH = TABLE_DIR / "step24_current_feed_audit.csv"
ROAD_SUPPORT_PATH = TABLE_DIR / "step24_detector_road_support.csv"
NETWORK_COVERAGE_PATH = TABLE_DIR / "step24_network_sensor_coverage.csv"
ATC_OVERLAP_PATH = TABLE_DIR / "step24_atc_overlap_audit.csv"
ARCHIVE_PATH = TABLE_DIR / "step24_historical_archive_audit.csv"
UPSTREAM_PATH = TABLE_DIR / "step24_upstream_evidence_inventory.csv"
DECISION_PATH = TABLE_DIR / "step24_decision_audit.csv"
MAP_PATH = FIGURE_DIR / "step24_public_detector_coverage.png"
SUPPORT_FIGURE_PATH = FIGURE_DIR / "step24_public_detector_road_support.png"

NETWORK_CANDIDATES = (
    PROCESSED_DIR / "atc_step23a1_osm_2023_network_feature_table.csv",
    PROCESSED_DIR / "atc_step23a_osm_2023_network_feature_table.csv",
    PROCESSED_DIR / "atc_step22_2023_network_feature_table.csv",
)
STATION_CANDIDATES = (
    PROCESSED_DIR / "atc_step23a1_osm_2023_station_feature_table.csv",
    PROCESSED_DIR / "atc_step23a_osm_2023_station_feature_table.csv",
    PROCESSED_DIR / "atc_step22_2023_feature_table.csv",
)
STEP23A_BASE_SCRIPT = PROJECT_ROOT / "src" / "23a_test_2023_osm_road_class.py"

ROAD_MATCH_DISTANCE_M = 100.0
MIN_LOCAL_DETECTORS = 20
MIN_LOCAL_SPATIAL_FOLDS = 4
MIN_ATC_COLOCATIONS = 20
MIN_CONSECUTIVE_ARCHIVE_YEARS = 3
MIN_DISTINCT_DAYS_PER_YEAR = 300
LOCAL_GROUPS = {"local", "service"}
STEP24_REVISION = "2026-08-17.3"

SOURCES = {
    "strategic_detector": {
        "label": "Strategic / major-road detector",
        "location_url": "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv",
        "feed_url": "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml",
        "page_url": "https://data.gov.hk/en-data/dataset/hk-td-sm_4-traffic-data-strategic-major-roads",
        "frequency": "1 minute raw; 2 minutes processed",
        "measurement": "volume, speed, occupancy",
        "annual_aadt_label": False,
        "independent_dynamic_count": True,
        "nominal_sampling": "fixed detectors on selected strategic/major roads",
        "limit": "strong time signal but deliberately concentrated on major roads",
    },
    "smart_lamppost": {
        "label": "Smart Lamppost detector",
        "location_url": "https://static.data.gov.hk/td/traffic-data-slp/info/traffic_speed_volume_occ_info-slp.csv",
        "feed_url": "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol_SLP-all.xml",
        "page_url": "https://data.gov.hk/en-data/dataset/hk-td-tis_33-traffic-data-traffic-detectors-installed-at-smart-lampposts",
        "frequency": "1 minute",
        "measurement": "volume, speed, occupancy",
        "annual_aadt_label": False,
        "independent_dynamic_count": True,
        "nominal_sampling": "fixed detectors at selected Smart Lamppost sites",
        "limit": "coverage and historical continuity must be measured, not assumed",
    },
    "vehicle_class": {
        "label": "Selected-route vehicle-class detector",
        "location_url": "https://static.data.gov.hk/td/traffic-atc-veh-class/info/traffic_prop_vehicle_class_info.csv",
        "feed_url": "https://resource.data.one.gov.hk/td/traffic-detectors/volByVClass-all.xml",
        "page_url": "https://data.gov.hk/en-data/dataset/hk-td-sm_5-annual-traffic-census-survey-data",
        "frequency": "daily",
        "measurement": "hourly traffic volume proportions for 10 vehicle classes",
        "annual_aadt_label": False,
        "independent_dynamic_count": True,
        "nominal_sampling": "14 detectors on selected routes in the current release",
        "limit": "useful for fleet composition; too selected to identify local-road AADT alone",
    },
    "aivas": {
        "label": "AI Video Analytics System of CCTVs",
        "location_url": "",
        "feed_url": "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1671693527354_28926/FeatureServer",
        "page_url": "https://data.gov.hk/en-data/dataset/hk-td-tis_32-traffic-data-aivas",
        "frequency": "rotating 130-second observations; dataset updated on major change",
        "measurement": "130-second segment flow and space-mean speed",
        "annual_aadt_label": False,
        "independent_dynamic_count": True,
        "nominal_sampling": "16 selected CCTV cameras, mostly near signalised junctions",
        "limit": "not continuous; view-limited; requires scaling and averaging; selected sites",
    },
    "gtfs": {
        "label": "Public transport GTFS",
        "location_url": "",
        "feed_url": "",
        "page_url": "https://data.gov.hk/en-data/dataset/hk-td-tis_21-etakmb",
        "frequency": "versioned service schedules",
        "measurement": "scheduled public-transport service context",
        "annual_aadt_label": False,
        "independent_dynamic_count": False,
        "nominal_sampling": "routes/stops rather than road traffic counters",
        "limit": "context feature only; cannot validate road traffic volume",
    },
    "fine_census": {
        "label": "Fine census population and socioeconomic geography",
        "location_url": "",
        "feed_url": "",
        "page_url": "https://www.census2021.gov.hk/en/building-boundaries.html",
        "frequency": "census / by-census years",
        "measurement": "population weights and socioeconomic attributes",
        "annual_aadt_label": False,
        "independent_dynamic_count": False,
        "nominal_sampling": "small-area population geography",
        "limit": "supports a redesigned equity estimand, not traffic reconstruction or annual change",
    },
}

AIVAS_SERVICE_BASES = (
    "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1671693527354_28926/FeatureServer",
    "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1671693527354_28926/MapServer",
)

# Official AIVAS specification, last updated 15 August 2023.  These are
# ROAD_ID references in the TD intelligent road network, not annual counts.
AIVAS_CAMERAS = {
    "K121F": ("Canton Road / Peking Road", [("105277", "Canton Road")]),
    "K102F": (
        "Chatham Road South / Salisbury Road",
        [("104944", "Salisbury Road"), ("104938", "Salisbury Road"),
         ("104942", "Salisbury Road"), ("104939", "Chatham Road South")],
    ),
    "K106F": (
        "Chatham Road South / Austin Road",
        [("104230", "Chatham Road South"), ("104233", "Chatham Road South"),
         ("104205", "Chatham Road South"), ("277244", "Chatham Road South"),
         ("104234", "Cheong Wan Road"), ("104227", "Cheong Wan Road")],
    ),
    "K108F": (
        "Jordan Road / Ferry Street",
        [("106875", "Jordan Road"), ("106867", "Jordan Road"),
         ("106866", "Jordan Road"), ("105294", "Canton Road")],
    ),
    "K202F": ("Yim Po Fong Street / Argyle Street", [("107299", "Argyle Street"), ("107259", "Argyle Street")]),
    "K205F": ("Nathan Road / Lai Chi Kok Road", [("105866", "Lai Chi Kok Road"), ("105876", "Nathan Road"), ("105867", "Nathan Road")]),
    "K209F": ("Tai Hang Tung Road / Boundary Street", [("107993", "Boundary Street"), ("107994", "Boundary Street")]),
    "K305": ("Cheung Sha Wan Road / Kwong Cheung Street", [("106604", "Cheung Sha Wan Road"), ("9343", "Cheung Sha Wan Road")]),
    "H108F": ("Queen's Road Central near Ice House Street", [("401", "Queen's Road Central")]),
    "H201F": ("Hennessy Road near Arsenal Street", [("5316", "Hennessy Road"), ("4202", "Hennessy Road")]),
    "H203F": ("Yee Wo Street near Hennessy Road", [("463", "Yee Wo Street"), ("483", "Yee Wo Street")]),
    "H305F": ("Causeway Road near Hing Fat Street", [("2197", "Causeway Road"), ("3603", "Causeway Road")]),
    "H307F": ("King's Road near Tin Chong Street", [("3526", "King's Road"), ("5575", "King's Road")]),
    "TW108F": ("Kwai Chung Road / Kwai On Road", [("60042", "Kwai Chung Road"), ("58203", "Kwai Chung Road")]),
    "YL106F": ("Castle Peak Road - Yuen Long / Fung Cheung Road", [("95503", "Castle Peak Road - Yuen Long"), ("97123", "Castle Peak Road - Yuen Long")]),
    "K602F": ("Cha Kwo Ling Road / Lei Yue Mun Road", [("109497", "Lei Yue Mun Road")]),
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    names = ", ".join(str(value.relative_to(PROJECT_ROOT)) for value in paths)
    raise FileNotFoundError(f"None of the required upstream tables exists: {names}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    direct = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in direct:
            return direct[candidate.lower()]
    compact = {re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column) for column in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in compact:
            return compact[key]
    if required:
        raise KeyError(f"Could not find any of {candidates}; available={list(frame.columns)}")
    return None


def numeric_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    column = find_column(frame, candidates)
    return pd.to_numeric(frame[column], errors="coerce")


def normalise_identifier(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return re.sub(r"\s+", "", text)


def download_current(url: str, path: Path, refresh: bool) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 20 and not refresh:
        return True, "reused_cached_file"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HK-AADT-public-data-audit/24"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        if len(payload) <= 20:
            return False, "download_returned_empty_payload"
        path.write_bytes(payload)
        return True, "downloaded_current_resource"
    except Exception as exc:  # Network state is evidence, not a reason to erase the audit.
        return False, f"download_failed:{type(exc).__name__}:{exc}"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def normalise_location_table(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    # The strategic-detector location resource names its official identifier
    # ``AID_ID_Number``.  Preserve that identifier because the archived raw
    # speed/volume XML uses the same AID values.  Falling back to row-order
    # pseudo IDs would make historical-to-current linkage impossible.
    id_col = find_column(
        frame,
        (
            "device_id",
            "detector_id",
            "station_id",
            "aid_id_number",
            "id",
        ),
        required=False,
    )
    lat_col = find_column(frame, ("latitude", "lat", "wgs84_latitude"), required=False)
    lon_col = find_column(frame, ("longitude", "long", "lon", "lng", "wgs84_longitude"), required=False)
    road_col = find_column(frame, ("road_en", "road_name_en", "road_name", "road", "street_ename"), required=False)
    district_col = find_column(frame, ("district", "district_en"), required=False)
    direction_col = find_column(frame, ("direction", "detector_direction", "traffic_direction", "rotation"), required=False)

    result = pd.DataFrame(index=frame.index)
    result["source"] = source
    if id_col is None:
        result["device_id"] = [f"{source}_{index + 1}" for index in range(len(frame))]
    else:
        result["device_id"] = frame[id_col].map(normalise_identifier)
    result["road_name"] = frame[road_col].fillna("").astype(str).str.strip() if road_col else ""
    result["district"] = frame[district_col].fillna("").astype(str).str.strip() if district_col else ""
    result["direction"] = frame[direction_col].fillna("").astype(str).str.strip() if direction_col else ""
    result["latitude"] = pd.to_numeric(frame[lat_col], errors="coerce") if lat_col else np.nan
    result["longitude"] = pd.to_numeric(frame[lon_col], errors="coerce") if lon_col else np.nan
    result["source_row_number"] = np.arange(1, len(frame) + 1)
    result["location_coordinate_available"] = result[["latitude", "longitude"]].notna().all(axis=1)
    result["detector_key"] = (
        result["source"].astype(str) + ":" + result["device_id"].astype(str) + ":" + result["direction"].astype(str)
    )
    return result.reset_index(drop=True)


def strip_tag(value: str) -> str:
    return value.rsplit("}", 1)[-1].strip().lower()


def parse_current_feed(source: str, path: Path, available: bool, download_note: str) -> dict[str, object]:
    base = {
        "source": source,
        "feed_available": available,
        "download_note": download_note,
        "record_count": 0,
        "unique_device_count": 0,
        "records_with_volume_or_flow": 0,
        "records_with_valid_flag": 0,
        "valid_record_count": 0,
        "timestamp_min": "",
        "timestamp_max": "",
        "interpretation": "one current snapshot is not an annual AADT label",
    }
    if not available or not path.exists():
        return base
    try:
        root = ElementTree.fromstring(path.read_bytes())
    except ElementTree.ParseError as exc:
        base["download_note"] = f"xml_parse_failed:{exc}"
        return base

    records: list[dict[str, str]] = []
    id_names = {"device_id", "detector_id", "station_id", "cctv_id", "id"}
    for element in root.iter():
        leaves: dict[str, str] = {}
        for child in element.iter():
            if child is element or len(list(child)):
                continue
            text = (child.text or "").strip()
            if text:
                leaves[strip_tag(child.tag)] = text
        if leaves and any(name in leaves for name in id_names):
            records.append(leaves)

    # Parent and child nodes can describe the same detector.  Keep one record
    # for each exact tuple of leaf fields.
    unique_records: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for record in records:
        signature = tuple(sorted(record.items()))
        if signature not in seen:
            seen.add(signature)
            unique_records.append(record)
    records = unique_records

    devices: set[str] = set()
    timestamps: list[pd.Timestamp] = []
    volume_records = 0
    validity_records = 0
    valid_records = 0
    for record in records:
        for name in id_names:
            if name in record:
                devices.add(record[name])
                break
        if any(("vol" in key or "flow" in key) and re.search(r"\d", value) for key, value in record.items()):
            volume_records += 1
        validity_values = [value for key, value in record.items() if "valid" in key.lower()]
        if validity_values:
            validity_records += 1
            if any(str(value).strip().lower() in {"y", "yes", "true", "1", "valid", "online"} for value in validity_values):
                valid_records += 1
        for key, value in record.items():
            if "time" in key or "date" in key:
                timestamp = pd.to_datetime(value, errors="coerce", utc=True)
                if not pd.isna(timestamp):
                    timestamps.append(timestamp)

    base.update(
        {
            "record_count": len(records),
            "unique_device_count": len(devices),
            "records_with_volume_or_flow": volume_records,
            "records_with_valid_flag": validity_records,
            "valid_record_count": valid_records,
            "timestamp_min": min(timestamps).isoformat() if timestamps else "",
            "timestamp_max": max(timestamps).isoformat() if timestamps else "",
        }
    )
    return base


def nested_coordinates(value: object) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            points.append((float(value[0]), float(value[1])))
        else:
            for item in value:
                points.extend(nested_coordinates(item))
    return points


def first_property(properties: dict[str, object], candidates: tuple[str, ...]) -> object:
    lookup = {str(key).lower(): value for key, value in properties.items()}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return ""


def obtain_aivas_csdi(refresh: bool) -> tuple[pd.DataFrame, dict[str, object], tuple[bool, str]]:
    attempts: list[str] = []
    for service_index, base_url in enumerate(AIVAS_SERVICE_BASES, start=1):
        service_path = RAW_DIR / f"aivas_service_{service_index}.json"
        available, note = download_current(f"{base_url}?f=json", service_path, refresh)
        attempts.append(f"{base_url.rsplit('/', 1)[-1]}:{note}")
        if not available:
            continue
        try:
            metadata = json.loads(service_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            attempts.append(f"{base_url.rsplit('/', 1)[-1]}:invalid_json_metadata")
            continue
        layers = metadata.get("layers", []) or []
        if not layers and "fields" in metadata:
            layers = [{"id": 0, "name": metadata.get("name", "layer_0")}]
        all_features: list[dict[str, object]] = []
        used_layers: list[str] = []
        for layer in layers:
            layer_id = layer.get("id")
            if layer_id is None:
                continue
            query_url = (
                f"{base_url}/{layer_id}/query?f=geojson&where=1%3D1"
                "&outFields=*&returnGeometry=true&outSR=4326&resultRecordCount=10000"
            )
            layer_path = RAW_DIR / f"aivas_current_{base_url.rsplit('/', 1)[-1].lower()}_{layer_id}.geojson"
            layer_available, layer_note = download_current(query_url, layer_path, refresh)
            attempts.append(f"layer_{layer_id}:{layer_note}")
            if not layer_available:
                continue
            try:
                payload = json.loads(layer_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            features = payload.get("features", []) or []
            if features:
                all_features.extend(features)
                used_layers.append(str(layer_id))
        if not all_features:
            continue

        rows: list[dict[str, object]] = []
        device_ids: set[str] = set()
        timestamps: list[pd.Timestamp] = []
        volume_count = validity_count = valid_count = 0
        for index, feature in enumerate(all_features, start=1):
            properties = feature.get("properties", {}) or {}
            camera_id = normalise_identifier(
                first_property(properties, ("cctv_id", "camera_id", "device_id", "detector_id", "id"))
            )
            if not camera_id:
                camera_id = f"aivas_feature_{index}"
            device_ids.add(camera_id)
            geometry = feature.get("geometry", {}) or {}
            points = nested_coordinates(geometry.get("coordinates", []))
            longitude = float(np.mean([point[0] for point in points])) if points else np.nan
            latitude = float(np.mean([point[1] for point in points])) if points else np.nan
            flow_value = first_property(properties, ("flow", "traffic_flow", "volume", "traffic_volume"))
            if str(flow_value).strip() and re.search(r"\d", str(flow_value)):
                volume_count += 1
            validity = str(first_property(properties, ("is_valid", "valid", "validity", "status"))).strip()
            if validity:
                validity_count += 1
                if validity.lower() in {"y", "yes", "true", "1", "valid", "online"}:
                    valid_count += 1
            timestamp_value = first_property(properties, ("generated_timestamp", "timestamp", "date_time", "datetime"))
            timestamp = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
            if not pd.isna(timestamp):
                timestamps.append(timestamp)
            road_name = str(first_property(properties, ("road_name", "road_en", "street_name", "location"))).strip()
            route_id = normalise_identifier(first_property(properties, ("segment", "route_id", "road_segment_id")))
            rows.append(
                {
                    "source": "aivas",
                    "device_id": camera_id,
                    "road_name": road_name,
                    "district": "",
                    "direction": "",
                    "latitude": latitude,
                    "longitude": longitude,
                    "source_row_number": index,
                    "location_coordinate_available": bool(np.isfinite(latitude) and np.isfinite(longitude)),
                    "detector_key": f"aivas:{camera_id}:{route_id or index}",
                    "official_route_id": route_id,
                }
            )
        frame = pd.DataFrame(rows)
        audit = {
            "source": "aivas",
            "feed_available": True,
            "download_note": f"queried {base_url}; layers={','.join(used_layers)}",
            "record_count": len(all_features),
            "unique_device_count": len(device_ids),
            "records_with_volume_or_flow": volume_count,
            "records_with_valid_flag": validity_count,
            "valid_record_count": valid_count,
            "timestamp_min": min(timestamps).isoformat() if timestamps else "",
            "timestamp_max": max(timestamps).isoformat() if timestamps else "",
            "interpretation": "current 130-second rotating observations; not annual AADT",
        }
        return frame, audit, (True, f"CSDI current feature service obtained from {base_url}")

    audit = {
        "source": "aivas",
        "feed_available": False,
        "download_note": " | ".join(attempts) if attempts else "CSDI service not queried",
        "record_count": 0,
        "unique_device_count": 16,
        "records_with_volume_or_flow": 0,
        "records_with_valid_flag": 0,
        "valid_record_count": 0,
        "timestamp_min": "",
        "timestamp_max": "",
        "interpretation": "fixed official specification still establishes 16 selected 130-second rotating camera sites",
    }
    return pd.DataFrame(), audit, (False, "CSDI current feature query did not return records; official segment manifest retained")


def aivas_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for camera_id, (camera_location, segments) in AIVAS_CAMERAS.items():
        for route_id, road_name in segments:
            rows.append(
                {
                    "source": "aivas",
                    "camera_id": camera_id,
                    "camera_location": camera_location,
                    "route_id": route_id,
                    "road_name": road_name,
                    "observation_duration_seconds": 130,
                    "continuous_monitoring": False,
                    "camera_note": "pan-tilt-zoom view can move; no record when displaced" if camera_id == "K305" else "",
                    "source_specification": "https://www.td.gov.hk/datagovhk_td/traffic-data-aivas/resources/traffic_data_from_ai_video_analytics_system_of_cctvs_dataspec.pdf",
                }
            )
    return pd.DataFrame(rows)


def projected_xy(longitude: pd.Series, latitude: pd.Series, reference_latitude: float) -> np.ndarray:
    lon = pd.to_numeric(longitude, errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(latitude, errors="coerce").to_numpy(dtype=float)
    x = lon * (111_320.0 * math.cos(math.radians(reference_latitude)))
    y = lat * 110_540.0
    return np.column_stack([x, y])


def normalise_osm_group(frame: pd.DataFrame) -> pd.Series:
    group_col = find_column(frame, ("osm_highway_group", "osm_group"), required=False)
    highway_col = find_column(frame, ("osm_highway", "highway"), required=False)
    if group_col:
        values = frame[group_col].fillna("unmatched").astype(str).str.lower().str.strip()
    elif highway_col:
        values = frame[highway_col].fillna("unmatched").astype(str).str.lower().str.strip()
    else:
        return pd.Series("unknown", index=frame.index)

    def recode(value: str) -> str:
        base = value.removesuffix("_link")
        if base in {"motorway"}:
            return "motorway"
        if base in {"trunk"}:
            return "trunk"
        if base in {"primary"}:
            return "primary"
        if base in {"secondary"}:
            return "secondary"
        if base in {"tertiary"}:
            return "tertiary"
        if base in {"residential", "living_street", "unclassified", "road", "track"}:
            return "local"
        if base == "service":
            return "service"
        if base in {"", "nan", "none", "unmatched"}:
            return "unmatched"
        return "other"

    return values.map(recode)


def add_official_route_id_if_available(network: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Recover ROUTE_ID in frozen centreline row order for the AIVAS audit.

    Step 22 correctly excluded ROUTE_ID as a general model feature, but AIVAS
    publishes its 130-second observations by that identifier.  Here it is used
    only to locate the 16-camera pilot on the already frozen network.
    """
    route_col = find_column(network, ("route_id", "routeid"), required=False)
    if route_col:
        return network, f"route_id_already_present:{route_col}"
    if not STEP23A_BASE_SCRIPT.exists():
        return network, "route_id_not_available_base_script_missing"
    try:
        base = load_module("hk_aadt_step24_route_reader", STEP23A_BASE_SCRIPT)
        centerline, _ = base.read_centerline(base.find_road_geodatabase())
        route_col = find_column(centerline, ("route_id", "routeid"), required=False)
        if route_col is None:
            return network, "route_id_not_present_in_official_centerline_reader"
        if len(centerline) != len(network):
            return network, f"route_id_row_alignment_failed:centerline={len(centerline)};network={len(network)}"
        result = network.copy()
        result["route_id"] = centerline[route_col].to_numpy()
        return result, "route_id_attached_from_frozen_official_centerline_row_order"
    except Exception as exc:
        return network, f"route_id_attachment_failed:{type(exc).__name__}:{exc}"


def attach_nearest_network(locations: pd.DataFrame, network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lon_col = find_column(network, ("centroid_longitude", "road_longitude", "longitude", "lon"))
    lat_col = find_column(network, ("centroid_latitude", "road_latitude", "latitude", "lat"))
    length_col = find_column(network, ("computed_length_m", "length_m", "shape_length"), required=False)
    fold_col = find_column(network, ("spatial_fold", "fold"), required=False)
    segment_col = find_column(network, ("road_2023_segment_index", "segment_index", "road_segment_index"), required=False)
    route_col = find_column(network, ("route_id", "routeid"), required=False)

    work = network.copy()
    work["_longitude"] = pd.to_numeric(work[lon_col], errors="coerce")
    work["_latitude"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["_osm_group"] = normalise_osm_group(work)
    work["_length_m"] = pd.to_numeric(work[length_col], errors="coerce").fillna(0.0) if length_col else 0.0
    work["_fold"] = pd.to_numeric(work[fold_col], errors="coerce") if fold_col else np.nan
    work["_segment"] = work[segment_col] if segment_col else work.index
    valid_network = work[["_longitude", "_latitude"]].notna().all(axis=1)
    net = work.loc[valid_network].copy()
    if net.empty:
        raise ValueError("Network feature table has no valid longitude/latitude rows")

    reference_latitude = float(net["_latitude"].median())
    net_xy = projected_xy(net["_longitude"], net["_latitude"], reference_latitude)
    net_tree = cKDTree(net_xy)

    result = locations.copy()
    result["nearest_segment_index"] = np.nan
    result["nearest_road_distance_m"] = np.nan
    result["nearest_osm_highway_group"] = ""
    result["nearest_spatial_fold"] = np.nan
    valid_location = result[["longitude", "latitude"]].notna().all(axis=1)
    if valid_location.any():
        query_xy = projected_xy(result.loc[valid_location, "longitude"], result.loc[valid_location, "latitude"], reference_latitude)
        distance, index = net_tree.query(query_xy, k=1)
        matched = net.iloc[index]
        result.loc[valid_location, "nearest_segment_index"] = matched["_segment"].to_numpy()
        result.loc[valid_location, "nearest_road_distance_m"] = distance
        result.loc[valid_location, "nearest_osm_highway_group"] = matched["_osm_group"].to_numpy()
        result.loc[valid_location, "nearest_spatial_fold"] = matched["_fold"].to_numpy()

    # AIVAS has official route references instead of detector coordinates.
    if route_col and "official_route_id" in result:
        route_lookup = defaultdict(list)
        for index, value in work[route_col].items():
            route_lookup[normalise_identifier(value)].append(index)
        # An official AIVAS ROUTE_ID is stronger evidence than proximity to a
        # camera point near a multi-arm junction, so it takes precedence.
        aivas_rows = result["official_route_id"].fillna("").astype(str).ne("")
        for row_index in result.index[aivas_rows]:
            candidates = route_lookup.get(normalise_identifier(result.at[row_index, "official_route_id"]), [])
            if not candidates:
                continue
            selected = candidates[0]
            result.at[row_index, "nearest_segment_index"] = work.at[selected, "_segment"]
            result.at[row_index, "nearest_road_distance_m"] = 0.0
            result.at[row_index, "nearest_osm_highway_group"] = work.at[selected, "_osm_group"]
            result.at[row_index, "nearest_spatial_fold"] = work.at[selected, "_fold"]

    # Reverse coverage: which official centreline centroids are within 100 m
    # of a public detector?  This is a support screen, not road-length truth.
    coverage_rows: list[dict[str, object]] = []
    for source, subset in result.groupby("source", dropna=False):
        points = subset.loc[subset[["longitude", "latitude"]].notna().all(axis=1)]
        if not points.empty:
            detector_xy = projected_xy(points["longitude"], points["latitude"], reference_latitude)
            distance, _ = cKDTree(detector_xy).query(net_xy, k=1)
            support = distance <= ROAD_MATCH_DISTANCE_M
            coverage_method = "network centroid within 100 m of detector coordinate"
        else:
            route_segments = set(
                subset["nearest_segment_index"].dropna().map(normalise_identifier)
            )
            if not route_segments:
                continue
            support = net["_segment"].map(normalise_identifier).isin(route_segments).to_numpy()
            coverage_method = "exact official ROUTE_ID link to frozen centreline"
        for group in ["all", "local", "service", "minor_proxy"]:
            if group == "all":
                mask = np.ones(len(net), dtype=bool)
            elif group == "minor_proxy":
                mask = net["_osm_group"].isin(LOCAL_GROUPS).to_numpy()
            else:
                mask = net["_osm_group"].eq(group).to_numpy()
            denominator_n = int(mask.sum())
            denominator_length = float(net.loc[mask, "_length_m"].sum())
            supported_n = int((mask & support).sum())
            supported_length = float(net.loc[mask & support, "_length_m"].sum())
            coverage_rows.append(
                {
                    "source": source,
                    "road_domain": group,
                    "network_segment_count": denominator_n,
                    "supported_segment_count": supported_n,
                    "segment_support_share": supported_n / denominator_n if denominator_n else np.nan,
                    "network_length_km": denominator_length / 1000.0,
                    "supported_length_km": supported_length / 1000.0,
                    "length_support_share": supported_length / denominator_length if denominator_length else np.nan,
                    "support_radius_m": ROAD_MATCH_DISTANCE_M,
                    "coverage_method": coverage_method,
                    "interpretation": "public detector support screen; not validation of annual AADT",
                }
            )
    return result, pd.DataFrame(coverage_rows)


def assign_frozen_spatial_folds(
    locations: pd.DataFrame,
    stations: pd.DataFrame,
    network: pd.DataFrame,
) -> pd.DataFrame:
    """Recover the frozen five spatial regions for detector locations.

    The full-network feature table does not carry station fold labels.  Fold
    labels are therefore recovered from the already frozen labelled-station
    table: the geographic centre of each frozen fold is calculated once, and
    each detector is assigned to its nearest frozen centre.  Detectors without
    coordinates, such as exact ROUTE_ID-linked AIVAS rows, use the coordinates
    of their already matched official centreline segment.

    This repairs an interface field only.  It does not create a new split,
    alter any detector-to-road match, or use an AADT outcome.
    """
    fold_col = find_column(stations, ("spatial_fold", "fold"))
    station_lon = find_column(stations, ("centroid_longitude", "road_longitude", "longitude", "lon"))
    station_lat = find_column(stations, ("centroid_latitude", "road_latitude", "latitude", "lat"))
    network_lon = find_column(network, ("centroid_longitude", "road_longitude", "longitude", "lon"))
    network_lat = find_column(network, ("centroid_latitude", "road_latitude", "latitude", "lat"))
    network_segment = find_column(
        network,
        ("road_2023_segment_index", "segment_index", "road_segment_index"),
        required=False,
    )

    labelled = stations.copy()
    labelled["_fold"] = pd.to_numeric(labelled[fold_col], errors="coerce")
    labelled["_lon"] = pd.to_numeric(labelled[station_lon], errors="coerce")
    labelled["_lat"] = pd.to_numeric(labelled[station_lat], errors="coerce")
    labelled = labelled.dropna(subset=["_fold", "_lon", "_lat"])
    centres = labelled.groupby("_fold", as_index=False)[["_lon", "_lat"]].mean()
    if centres.empty:
        raise ValueError("Frozen station folds have no valid geographic centres")

    result = locations.copy()
    result["_fold_lon"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["_fold_lat"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["spatial_fold_assignment_method"] = np.where(
        result[["_fold_lon", "_fold_lat"]].notna().all(axis=1),
        "detector_coordinate_to_nearest_frozen_fold_centre",
        "",
    )

    segment_values = network[network_segment] if network_segment else pd.Series(network.index, index=network.index)
    segment_locations = {
        normalise_identifier(segment): (
            pd.to_numeric(pd.Series([network.at[index, network_lon]]), errors="coerce").iloc[0],
            pd.to_numeric(pd.Series([network.at[index, network_lat]]), errors="coerce").iloc[0],
        )
        for index, segment in segment_values.items()
    }
    missing_coordinates = ~result[["_fold_lon", "_fold_lat"]].notna().all(axis=1)
    for row_index in result.index[missing_coordinates & result["nearest_segment_index"].notna()]:
        point = segment_locations.get(normalise_identifier(result.at[row_index, "nearest_segment_index"]))
        if point is None or not all(np.isfinite(value) for value in point):
            continue
        result.at[row_index, "_fold_lon"] = point[0]
        result.at[row_index, "_fold_lat"] = point[1]
        result.at[row_index, "spatial_fold_assignment_method"] = (
            "route_linked_segment_to_nearest_frozen_fold_centre"
        )

    reference_latitude = float(labelled["_lat"].median())
    centre_xy = projected_xy(centres["_lon"], centres["_lat"], reference_latitude)
    valid = result[["_fold_lon", "_fold_lat"]].notna().all(axis=1)
    result["nearest_spatial_fold"] = np.nan
    result["distance_to_frozen_fold_centre_m"] = np.nan
    if valid.any():
        query_xy = projected_xy(
            result.loc[valid, "_fold_lon"],
            result.loc[valid, "_fold_lat"],
            reference_latitude,
        )
        distance, centre_index = cKDTree(centre_xy).query(query_xy, k=1)
        result.loc[valid, "nearest_spatial_fold"] = centres.iloc[centre_index]["_fold"].to_numpy()
        result.loc[valid, "distance_to_frozen_fold_centre_m"] = distance
    return result.drop(columns=["_fold_lon", "_fold_lat"])


def attach_aivas_rows(locations: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        rows.append(
            {
                "source": "aivas",
                "device_id": row["camera_id"],
                "road_name": row["road_name"],
                "district": "",
                "direction": "",
                "latitude": np.nan,
                "longitude": np.nan,
                "source_row_number": len(rows) + 1,
                "location_coordinate_available": False,
                "detector_key": f"aivas:{row['camera_id']}:{row['route_id']}",
                "official_route_id": row["route_id"],
            }
        )
    base = locations.copy()
    if "official_route_id" not in base:
        base["official_route_id"] = ""
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False)


def road_support_table(locations: pd.DataFrame) -> pd.DataFrame:
    work = locations.copy()
    work["within_match_radius"] = pd.to_numeric(work["nearest_road_distance_m"], errors="coerce") <= ROAD_MATCH_DISTANCE_M
    rows: list[dict[str, object]] = []
    for source, source_frame in work.groupby("source", dropna=False):
        device_total = source_frame["device_id"].astype(str).nunique()
        for group, group_frame in source_frame.groupby("nearest_osm_highway_group", dropna=False):
            accepted = group_frame[group_frame["within_match_radius"]]
            rows.append(
                {
                    "source": source,
                    "osm_highway_group": str(group) if str(group) else "unmatched",
                    "source_device_count": device_total,
                    "location_or_route_record_count": len(group_frame),
                    "accepted_record_count": len(accepted),
                    "accepted_unique_device_count": accepted["device_id"].astype(str).nunique(),
                    "median_match_distance_m": pd.to_numeric(accepted["nearest_road_distance_m"], errors="coerce").median(),
                    "spatial_fold_count": pd.to_numeric(accepted["nearest_spatial_fold"], errors="coerce").dropna().astype(int).nunique(),
                    "local_road_proxy": str(group) in LOCAL_GROUPS,
                }
            )
    return pd.DataFrame(rows)


def atc_overlap(locations: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    lon_col = find_column(stations, ("centroid_longitude", "road_longitude", "longitude", "lon"))
    lat_col = find_column(stations, ("centroid_latitude", "road_latitude", "latitude", "lat"))
    network_col = find_column(stations, ("road_network", "station_road_network"), required=False)
    road_type_col = find_column(stations, ("road_type", "station_road_type"), required=False)
    segment_col = find_column(stations, ("road_2023_segment_index", "segment_index", "road_segment_index"), required=False)
    valid_station = pd.to_numeric(stations[lon_col], errors="coerce").notna() & pd.to_numeric(stations[lat_col], errors="coerce").notna()
    work = stations.loc[valid_station].copy()
    reference_latitude = float(pd.to_numeric(work[lat_col], errors="coerce").median())
    station_xy = projected_xy(work[lon_col], work[lat_col], reference_latitude)

    rows: list[dict[str, object]] = []
    for source, subset in locations.groupby("source", dropna=False):
        points = subset.loc[subset[["longitude", "latitude"]].notna().all(axis=1)].drop_duplicates("detector_key")
        if not points.empty:
            detector_xy = projected_xy(points["longitude"], points["latitude"], reference_latitude)
            distance, _ = cKDTree(detector_xy).query(station_xy, k=1)
            overlap = distance <= ROAD_MATCH_DISTANCE_M
            overlap_method = "ATC centroid within 100 m of detector coordinate"
        elif segment_col and subset["nearest_segment_index"].notna().any():
            route_segments = set(subset["nearest_segment_index"].dropna().map(normalise_identifier))
            station_segments = work[segment_col].map(normalise_identifier)
            overlap = station_segments.isin(route_segments).to_numpy()
            distance = np.where(overlap, 0.0, np.nan)
            overlap_method = "same frozen official centreline segment by ROUTE_ID"
        else:
            overlap = np.zeros(len(work), dtype=bool)
            distance = np.full(len(work), np.nan)
            overlap_method = "not_evaluable_without_coordinates_or_route_id_alignment"
        domains: dict[str, np.ndarray] = {"all": np.ones(len(work), dtype=bool)}
        if network_col:
            values = work[network_col].fillna("unknown").astype(str).str.upper()
            for value in sorted(values.unique()):
                domains[f"road_network={value}"] = values.eq(value).to_numpy()
        if road_type_col:
            values = work[road_type_col].fillna("unknown").astype(str).str.upper()
            if values.str.contains("LOCAL").any():
                domains["road_type_contains_LOCAL"] = values.str.contains("LOCAL").to_numpy()
        for domain, mask in domains.items():
            denominator = int(mask.sum())
            rows.append(
                {
                    "source": source,
                    "atc_domain": domain,
                    "atc_station_count": denominator,
                    "atc_with_detector_within_100m": int((mask & overlap).sum()),
                    "atc_overlap_share": float((mask & overlap).sum() / denominator) if denominator else np.nan,
                    "median_nearest_detector_distance_m": float(np.nanmedian(distance[mask])) if denominator and np.isfinite(distance[mask]).any() else np.nan,
                    "overlap_method": overlap_method,
                }
            )
    return pd.DataFrame(rows)


def classify_archive_source(name: str) -> str | None:
    value = name.lower()
    if "rawspeedvol_slp" in value or ("smart" in value and "lamppost" in value):
        return "smart_lamppost"
    if "volbyvclass" in value or "vehicle_class" in value or "veh-class" in value:
        return "vehicle_class"
    if "aivas" in value or "traffic-data-aivas" in value:
        return "aivas"
    if "rawspeedvol" in value or "strategic_detector" in value:
        return "strategic_detector"
    return None


DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[-_]([01]\d)(?!\d)"),
)


def extract_date(name: str) -> tuple[int | None, int | None, int | None]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        values = [int(value) for value in match.groups()]
        year, month = values[0], values[1]
        day = values[2] if len(values) > 2 else None
        if 2000 <= year <= 2100 and 1 <= month <= 12 and (day is None or 1 <= day <= 31):
            return year, month, day
    return None, None, None


def archive_inventory() -> pd.DataFrame:
    roots = [PROJECT_ROOT / "data" / "raw"]
    evidence: dict[str, dict[str, object]] = {
        source: {"files": 0, "dates": set(), "months": set(), "years": set(), "undated": 0}
        for source in ("strategic_detector", "smart_lamppost", "vehicle_class", "aivas")
    }

    def observe(display_name: str) -> None:
        source = classify_archive_source(display_name)
        if source is None:
            return
        record = evidence[source]
        record["files"] = int(record["files"]) + 1
        year, month, day = extract_date(display_name)
        if year is None:
            record["undated"] = int(record["undated"]) + 1
            return
        record["years"].add(year)
        if month is not None:
            record["months"].add((year, month))
        if day is not None:
            try:
                record["dates"].add(datetime(year, month, day).date().isoformat())
            except ValueError:
                pass

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            observe(str(path.relative_to(PROJECT_ROOT)))
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        for name in archive.namelist():
                            observe(f"{path.name}/{name}")
                except zipfile.BadZipFile:
                    continue

    rows: list[dict[str, object]] = []
    for source, record in evidence.items():
        years = sorted(record["years"])
        dates = sorted(record["dates"])
        counts_by_year = Counter(int(value[:4]) for value in dates)
        months_by_year = Counter(year for year, _ in record["months"])
        for year in years or [None]:
            rows.append(
                {
                    "source": source,
                    "year": year,
                    "observed_file_or_member_count_all_years": record["files"],
                    "distinct_dates": counts_by_year.get(year, 0) if year else 0,
                    "distinct_months": months_by_year.get(year, 0) if year else 0,
                    "undated_file_or_member_count": record["undated"],
                    "first_observed_date": dates[0] if dates else "",
                    "last_observed_date": dates[-1] if dates else "",
                    "meets_300_day_year_rule": counts_by_year.get(year, 0) >= MIN_DISTINCT_DAYS_PER_YEAR if year else False,
                    "interpretation": "local archive evidence only; absence here does not prove the official archive is unavailable",
                }
            )
    return pd.DataFrame(rows)


def consecutive_qualifying_years(archive: pd.DataFrame, source: str) -> int:
    subset = archive[(archive["source"] == source) & archive["meets_300_day_year_rule"].astype(bool)]
    years = sorted(pd.to_numeric(subset["year"], errors="coerce").dropna().astype(int).unique())
    longest = current = 0
    previous = None
    for year in years:
        current = current + 1 if previous is not None and year == previous + 1 else 1
        longest = max(longest, current)
        previous = year
    return longest


def upstream_inventory() -> pd.DataFrame:
    candidates = sorted(TABLE_DIR.glob("step21_*.csv")) + sorted(TABLE_DIR.glob("step22_*.csv"))
    rows: list[dict[str, object]] = []
    for path in candidates:
        try:
            frame = pd.read_csv(path)
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "row_count": len(frame),
                    "column_count": len(frame.columns),
                    "columns": "|".join(map(str, frame.columns)),
                    "use_in_step24": "provenance only; Step 24 recomputes current location and support metrics",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "row_count": np.nan,
                    "column_count": np.nan,
                    "columns": "",
                    "use_in_step24": f"could_not_read:{type(exc).__name__}",
                }
            )
    return pd.DataFrame(rows)


def source_inventory(
    location_status: dict[str, tuple[bool, str]],
    feed_audit: pd.DataFrame,
    locations: pd.DataFrame,
    archive: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feed_lookup = feed_audit.set_index("source").to_dict("index") if not feed_audit.empty else {}
    for source, metadata in SOURCES.items():
        location_available, location_note = location_status.get(source, (False, "not_applicable"))
        source_locations = locations[locations["source"] == source] if not locations.empty else pd.DataFrame()
        qualifying_years = consecutive_qualifying_years(archive, source) if source in set(archive["source"]) else 0
        feed = feed_lookup.get(source, {})
        rows.append(
            {
                "source": source,
                "source_label": metadata["label"],
                "measurement": metadata["measurement"],
                "nominal_update_frequency": metadata["frequency"],
                "nominal_sampling": metadata["nominal_sampling"],
                "official_page": metadata["page_url"],
                "official_location_resource": metadata["location_url"],
                "official_live_resource": metadata["feed_url"],
                "current_location_resource_obtained": location_available,
                "location_download_note": location_note,
                "current_location_or_route_record_count": len(source_locations),
                "current_unique_device_count": source_locations["device_id"].astype(str).nunique() if not source_locations.empty else (16 if source == "aivas" else 0),
                "current_feed_obtained": bool(feed.get("feed_available", False)),
                "current_feed_unique_device_count": int(feed.get("unique_device_count", 0) or 0),
                "current_feed_has_volume_or_flow": int(feed.get("records_with_volume_or_flow", 0) or 0) > 0,
                "annual_aadt_label": metadata["annual_aadt_label"],
                "independent_dynamic_count_candidate": metadata["independent_dynamic_count"],
                "locally_observed_consecutive_years_with_300_days": qualifying_years,
                "primary_limit": metadata["limit"],
            }
        )
    return pd.DataFrame(rows)


def make_decisions(locations: pd.DataFrame, overlap: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    dynamic_sources = ["strategic_detector", "smart_lamppost", "vehicle_class", "aivas"]
    accepted_locations = locations[
        locations["source"].isin(dynamic_sources)
        & locations["nearest_osm_highway_group"].isin(LOCAL_GROUPS)
        & (pd.to_numeric(locations["nearest_road_distance_m"], errors="coerce") <= ROAD_MATCH_DISTANCE_M)
    ].copy()
    local_devices_by_source = accepted_locations.groupby("source")["device_id"].nunique().to_dict()
    local_folds_by_source = (
        accepted_locations.assign(_fold=pd.to_numeric(accepted_locations["nearest_spatial_fold"], errors="coerce"))
        .dropna(subset=["_fold"])
        .groupby("source")["_fold"]
        .nunique()
        .to_dict()
    )
    overlap_all = overlap[overlap["atc_domain"].eq("all")].set_index("source") if not overlap.empty else pd.DataFrame()

    rows: list[dict[str, object]] = []
    eligible_sources: list[str] = []
    for source in dynamic_sources:
        local_n = int(local_devices_by_source.get(source, 0))
        folds = int(local_folds_by_source.get(source, 0))
        passed = local_n >= MIN_LOCAL_DETECTORS and folds >= MIN_LOCAL_SPATIAL_FOLDS
        if passed:
            eligible_sources.append(source)
        rows.append(
            {
                "decision": f"{source}_adds_minimum_local_road_label_support",
                "pass": passed,
                "evidence": f"local/service unique devices={local_n}; spatial folds={folds}; required devices>={MIN_LOCAL_DETECTORS} and folds>={MIN_LOCAL_SPATIAL_FOLDS}",
                "failed_criterion": "" if passed else ("local_device_count_below_threshold" if local_n < MIN_LOCAL_DETECTORS else "spatial_fold_coverage_below_threshold"),
                "action": "eligible for a bounded local-road proxy test" if passed else "do not treat this source as representative local-road labels",
            }
        )

    colocated_by_source: dict[str, int] = {}
    for source in dynamic_sources:
        if isinstance(overlap_all, pd.DataFrame) and source in overlap_all.index:
            value = overlap_all.loc[source, "atc_with_detector_within_100m"]
            if isinstance(value, pd.Series):
                value = value.max()
            colocated_by_source[source] = int(value)
        else:
            colocated_by_source[source] = 0
    calibration_sources = [source for source, value in colocated_by_source.items() if value >= MIN_ATC_COLOCATIONS]
    rows.append(
        {
            "decision": "at_least_one_public_dynamic_source_has_minimum_atc_colocation",
            "pass": bool(calibration_sources),
            "evidence": "; ".join(f"{source}={value}" for source, value in colocated_by_source.items()) + f"; threshold={MIN_ATC_COLOCATIONS}",
            "failed_criterion": "" if calibration_sources else "no_source_has_20_atc_colocations",
            "action": "pre-register temporal aggregation and calibrate proxy against held-out ATC" if calibration_sources else "current detector systems cannot yet be calibrated as an AADT proxy",
        }
    )

    archive_years = {source: consecutive_qualifying_years(archive, source) for source in dynamic_sources}
    temporal_sources = [source for source, years in archive_years.items() if years >= MIN_CONSECUTIVE_ARCHIVE_YEARS]
    rows.append(
        {
            "decision": "local_archive_supports_a_multiyear_forward_experiment",
            "pass": bool(temporal_sources),
            "evidence": "; ".join(f"{source}={years} qualifying consecutive years" for source, years in archive_years.items()) + f"; required>={MIN_CONSECUTIVE_ARCHIVE_YEARS} years with >={MIN_DISTINCT_DAYS_PER_YEAR} distinct dates/year",
            "failed_criterion": "" if temporal_sources else "historical_archive_not_yet_materialised_locally",
            "action": "freeze a forward prediction design before annualising the feeds" if temporal_sources else "retrieve date-bounded official archives; do not infer that advertised historical data are absent",
        }
    )

    aivas_pass = False
    rows.append(
        {
            "decision": "aivas_alone_can_represent_annual_local_road_aadt",
            "pass": aivas_pass,
            "evidence": "16 selected CCTV cameras; rotating 130-second observations; most sites near signalised junctions; official specification requires scaling and recommends >1-hour averaging",
            "failed_criterion": "selected_noncontinuous_sample_not_annual_or_representative",
            "action": "use AIVAS only as supplementary point evidence after temporal aggregation; never as full-network annual truth",
        }
    )

    cross_sectional_authorised = bool(eligible_sources) and bool(calibration_sources)
    rows.append(
        {
            "decision": "step25_bounded_cross_sectional_dynamic_proxy_experiment_authorised",
            "pass": cross_sectional_authorised,
            "evidence": f"sources meeting local support={eligible_sources or 'none'}; sources meeting ATC colocation={calibration_sources or 'none'}",
            "failed_criterion": "" if cross_sectional_authorised else "local_support_or_atc_calibration_gate_failed",
            "action": "test source-specific proxies; keep detector-present and detector-absent domains separate" if cross_sectional_authorised else "stop model fitting and report the measured coverage boundary",
        }
    )

    temporal_authorised = bool(temporal_sources) and bool(calibration_sources)
    rows.append(
        {
            "decision": "step25_multiyear_forward_dynamic_experiment_authorised",
            "pass": temporal_authorised,
            "evidence": f"sources meeting archive continuity={temporal_sources or 'none'}; sources meeting ATC colocation={calibration_sources or 'none'}",
            "failed_criterion": "" if temporal_authorised else "continuous_multiyear_archive_or_calibration_gate_failed",
            "action": "compare forward change predictions with no-change after freezing aggregation" if temporal_authorised else "do not claim annual-change identification from current snapshots",
        }
    )

    rows.append(
        {
            "decision": "public_data_equity_redesign_is_separately_feasible",
            "pass": True,
            "evidence": "fine census geography can supply population weights, while Step 16 showed Large TPU Group mean AADT is the wrong estimand",
            "failed_criterion": "",
            "action": "develop a population-weighted 100 m / 200 m near-road activity proof of concept, explicitly conditional on the validated traffic domain",
        }
    )
    rows.append(
        {
            "decision": "public_data_boundary_has_been_fully_reached",
            "pass": False,
            "evidence": "Step 24 audits current public dynamic sources and local archive materialisation; a failed archive gate means not-yet-obtained, not unavailable",
            "failed_criterion": "not_a_valid_positive_claim",
            "action": "replace the broad boundary claim with source-specific coverage and continuity findings",
        }
    )
    return pd.DataFrame(rows)


def write_figures(network: pd.DataFrame, locations: pd.DataFrame, support: pd.DataFrame) -> None:
    lon_col = find_column(network, ("centroid_longitude", "road_longitude", "longitude", "lon"))
    lat_col = find_column(network, ("centroid_latitude", "road_latitude", "latitude", "lat"))
    valid = pd.to_numeric(network[lon_col], errors="coerce").notna() & pd.to_numeric(network[lat_col], errors="coerce").notna()
    network_plot = network.loc[valid]
    colors = {
        "strategic_detector": "#d73027",
        "smart_lamppost": "#1a9850",
        "vehicle_class": "#4575b4",
        "aivas": "#984ea3",
    }
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.scatter(
        pd.to_numeric(network_plot[lon_col], errors="coerce"),
        pd.to_numeric(network_plot[lat_col], errors="coerce"),
        s=0.35,
        color="#c9c9c9",
        alpha=0.35,
        linewidths=0,
        label="TD centreline centroids",
    )
    for source, subset in locations.groupby("source"):
        points = subset.loc[subset[["longitude", "latitude"]].notna().all(axis=1)].drop_duplicates("detector_key")
        if points.empty:
            continue
        ax.scatter(points["longitude"], points["latitude"], s=22, alpha=0.85, color=colors.get(source, "black"), label=source)
    ax.set_title("Step 24 public dynamic detector locations\nCurrent location tables; AIVAS lacks coordinates in the audited specification table")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAP_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved: {MAP_PATH.relative_to(PROJECT_ROOT)}")

    plot = support[support["accepted_record_count"] > 0].copy()
    if not plot.empty:
        pivot = plot.pivot_table(index="osm_highway_group", columns="source", values="accepted_unique_device_count", aggfunc="sum", fill_value=0)
        preferred = [value for value in ["motorway", "trunk", "primary", "secondary", "tertiary", "local", "service", "other", "unmatched"] if value in pivot.index]
        pivot = pivot.reindex(preferred + [value for value in pivot.index if value not in preferred])
        fig, ax = plt.subplots(figsize=(11.0, 6.2))
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.axvspan(max(-0.5, len(preferred) - 4.5), len(preferred) - 0.5, color="#f0f7ec", alpha=0.15)
        ax.set_title("Step 24 detector support by nearest deployable OSM road class")
        ax.set_xlabel("Nearest OSM class within 100 m")
        ax.set_ylabel("Unique public detector devices")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(SUPPORT_FIGURE_PATH, dpi=220)
        plt.close(fig)
        print(f"Saved: {SUPPORT_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def update_report_manifest(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        rows.append(
            {
                "step": "24",
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "status": "current",
                "reporting_rule": "public dynamic data suitability audit; not annual AADT or historical reconstruction",
            }
        )
    new = pd.DataFrame(rows)
    if REPORT_MANIFEST_PATH.exists():
        old = pd.read_csv(REPORT_MANIFEST_PATH)
        for column in new.columns:
            if column not in old:
                old[column] = ""
        for column in old.columns:
            if column not in new:
                new[column] = ""
        old = old[~old.get("step", pd.Series(index=old.index, dtype=str)).astype(str).eq("24")]
        new = pd.concat([old, new[old.columns]], ignore_index=True)
    save_csv(new, REPORT_MANIFEST_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-current", action="store_true", help="redownload current small location/feed resources")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step 24 revision: {STEP24_REVISION}")

    network_path = first_existing(NETWORK_CANDIDATES)
    station_path = first_existing(STATION_CANDIDATES)
    print(f"Using network features: {network_path.relative_to(PROJECT_ROOT)}")
    print(f"Using 2023 station features: {station_path.relative_to(PROJECT_ROOT)}")
    network = pd.read_csv(network_path)
    network, route_id_note = add_official_route_id_if_available(network)
    print(f"AIVAS route-location audit: {route_id_note}")
    stations = pd.read_csv(station_path)

    location_status: dict[str, tuple[bool, str]] = {}
    feed_rows: list[dict[str, object]] = []
    location_frames: list[pd.DataFrame] = []
    for source in ("strategic_detector", "smart_lamppost", "vehicle_class"):
        metadata = SOURCES[source]
        location_file = RAW_DIR / f"{source}_locations.csv"
        feed_file = RAW_DIR / f"{source}_current.xml"
        print(f"Obtaining current {source} location table...")
        location_available, location_note = download_current(metadata["location_url"], location_file, args.refresh_current)
        location_status[source] = (location_available, location_note)
        if location_available:
            frame = read_csv_flexible(location_file)
            location_frames.append(normalise_location_table(source, frame))
        print(f"Obtaining one current {source} feed snapshot...")
        feed_available, feed_note = download_current(metadata["feed_url"], feed_file, args.refresh_current)
        feed_rows.append(parse_current_feed(source, feed_file, feed_available, feed_note))

    manifest = aivas_manifest()
    manifest["route_id_location_audit"] = route_id_note
    save_csv(manifest, AIVAS_PATH)
    print("Obtaining one current AIVAS CSDI feature snapshot...")
    aivas_locations, aivas_feed_row, aivas_location_status = obtain_aivas_csdi(args.refresh_current)
    location_status["aivas"] = aivas_location_status
    if not aivas_locations.empty:
        location_frames.append(aivas_locations)
    feed_rows.append(aivas_feed_row)
    feed_audit = pd.DataFrame(feed_rows)
    save_csv(feed_audit, FEED_AUDIT_PATH)

    locations = pd.concat(location_frames, ignore_index=True, sort=False) if location_frames else pd.DataFrame(
        columns=["source", "device_id", "road_name", "district", "direction", "latitude", "longitude", "source_row_number", "location_coordinate_available", "detector_key"]
    )
    locations = attach_aivas_rows(locations, manifest)
    locations, network_coverage = attach_nearest_network(locations, network)
    locations = assign_frozen_spatial_folds(locations, stations, network)
    save_csv(locations, LOCATION_PATH)
    save_csv(network_coverage, NETWORK_COVERAGE_PATH)

    support = road_support_table(locations)
    save_csv(support, ROAD_SUPPORT_PATH)
    overlap = atc_overlap(locations, stations)
    save_csv(overlap, ATC_OVERLAP_PATH)

    archive = archive_inventory()
    save_csv(archive, ARCHIVE_PATH)
    upstream = upstream_inventory()
    save_csv(upstream, UPSTREAM_PATH)
    inventory = source_inventory(location_status, feed_audit, locations, archive)
    save_csv(inventory, INVENTORY_PATH)
    decisions = make_decisions(locations, overlap, archive)
    save_csv(decisions, DECISION_PATH)
    write_figures(network, locations, support)

    update_report_manifest(
        [
            LOCATION_PATH,
            AIVAS_PATH,
            INVENTORY_PATH,
            FEED_AUDIT_PATH,
            ROAD_SUPPORT_PATH,
            NETWORK_COVERAGE_PATH,
            ATC_OVERLAP_PATH,
            ARCHIVE_PATH,
            UPSTREAM_PATH,
            DECISION_PATH,
            MAP_PATH,
            SUPPORT_FIGURE_PATH,
        ]
    )

    local_summary = locations[
        locations["nearest_osm_highway_group"].isin(LOCAL_GROUPS)
        & locations["source"].isin(["strategic_detector", "smart_lamppost", "vehicle_class", "aivas"])
        & (pd.to_numeric(locations["nearest_road_distance_m"], errors="coerce") <= ROAD_MATCH_DISTANCE_M)
    ].groupby("source")["device_id"].nunique()
    print("\nStep 24 public dynamic traffic-data audit is complete.")
    if len(local_summary):
        print("  Local/service detector devices: " + "; ".join(f"{source}={int(value)}" for source, value in local_summary.items()))
    for _, row in decisions.iterrows():
        if row["decision"] in {
            "step25_bounded_cross_sectional_dynamic_proxy_experiment_authorised",
            "step25_multiyear_forward_dynamic_experiment_authorised",
            "public_data_equity_redesign_is_separately_feasible",
        }:
            print(f"  {row['decision']}: {row['pass']}")
    print("  Interpret a failed archive gate as not-yet-materialised locally, not as proof that public history is unavailable.")


if __name__ == "__main__":
    main()
