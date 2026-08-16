"""Step 23A: test whether historical OSM road class adds deployable 2023 skill.

This is a 2023-only gate.  It does not backcast earlier years.  The script:

1. obtains the OpenStreetMap state at 2023-06-30 from the Overpass API;
2. matches OSM highway ways to every segment of the official 2023 centreline;
3. audits full-network, measured-station and MINOR-road support;
4. adds OSM ``highway=*`` to the frozen Step 22 deployable context; and
5. repeats the same five spatial folds against the honest hierarchy lookup.

ATC ``road_network`` and ``road_type`` are outcomes/evaluation strata only.
They enter one explicitly labelled oracle model and no deployable model.  OSM
matching diagnostics are also excluded from model features: they are used only
to decide whether the external class is sufficiently supported.

The primary hypothesis is the OSM highway-class block.  Lanes, maxspeed,
oneway, bridge, tunnel, junction and access tags form a secondary, separately
tested block.  Historical OSM extraction is authorised only if the primary
2023 gate passes.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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
from scipy.stats import spearmanr
import shapely
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "step23a_osm_2023"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

STEP22_STATION_PATH = PROCESSED_DIR / "atc_step22_2023_feature_table.csv"
STEP22_NETWORK_PATH = PROCESSED_DIR / "atc_step22_2023_network_feature_table.csv"
STEP22_ROAD_MATCH_PATH = PROCESSED_DIR / "atc_step22_2023_road_matches.csv"
STEP22_DECISION_PATH = TABLE_DIR / "step22_decision_audit.csv"

OSM_WAY_TABLE_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_way_table.csv"
OSM_CROSSWALK_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_network_crosswalk.csv"
NETWORK_FEATURE_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_network_feature_table.csv"
STATION_FEATURE_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_station_feature_table.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step23a_osm_2023_oof_predictions.csv"

SOURCE_AUDIT_PATH = TABLE_DIR / "step23a_osm_source_audit.csv"
MATCH_AUDIT_PATH = TABLE_DIR / "step23a_osm_match_audit.csv"
CLASS_AGREEMENT_PATH = TABLE_DIR / "step23a_osm_atc_class_agreement.csv"
FEATURE_MANIFEST_PATH = TABLE_DIR / "step23a_feature_manifest.csv"
METRICS_BY_FOLD_PATH = TABLE_DIR / "step23a_metrics_by_fold.csv"
SUMMARY_PATH = TABLE_DIR / "step23a_model_summary.csv"
COMPARISON_PATH = TABLE_DIR / "step23a_paired_model_comparison.csv"
SUBGROUP_PATH = TABLE_DIR / "step23a_subgroup_bias.csv"
DECISION_PATH = TABLE_DIR / "step23a_decision_audit.csv"

COVERAGE_FIGURE_PATH = FIGURE_DIR / "step23a_osm_coverage_and_class.png"
MODEL_FIGURE_PATH = FIGURE_DIR / "step23a_2023_model_comparison.png"
BIAS_FIGURE_PATH = FIGURE_DIR / "step23a_subgroup_bias.png"
AGREEMENT_FIGURE_PATH = FIGURE_DIR / "step23a_osm_atc_class_agreement.png"

OVERPASS_ENDPOINTS = (
    (
        "vk_maps",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ),
    (
        "openstreetmap_main",
        "https://overpass-api.de/api/interpreter",
    ),
)
OVERPASS_DOC = "https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL#Date"
OSM_TIMESTAMP = "2023-06-30T00:00:00Z"
OSM_FILTER = 'way["highway"]'
OVERPASS_USER_AGENT = (
    "HK-AADT-Step23A/1.1 "
    "(+https://github.com/GuodongLiang1215/HK_AADT_Backcasting)"
)
OVERPASS_429_WAIT_SECONDS = 30
OVERPASS_SUCCESS_PAUSE_SECONDS = 2
MAX_TILE_SUBDIVISION_DEPTH = 2
PROJECTED_CRS = "EPSG:2326"
TILE_COUNT_PER_AXIS = 4
MATCH_SEARCH_M = 60.0
MATCH_BUFFER_M = 12.0
FOLDS = (1, 2, 3, 4, 5)

NETWORK_LENGTH_COVERAGE_THRESHOLD = 0.80
STATION_COVERAGE_THRESHOLD = 0.90
MINOR_STATION_COVERAGE_THRESHOLD = 0.80
SKILL_VS_HIERARCHY_THRESHOLD_PCT = 5.0
INCREMENT_VS_CONTEXT_THRESHOLD_PCT = 2.0
MINOR_INCREMENT_THRESHOLD_PCT = 2.0
AGGREGATE_BIAS_THRESHOLD_PCT = 10.0
SUBGROUP_BIAS_THRESHOLD_PCT = 15.0

STRUCTURAL_FEATURES = [
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

OSM_GROUPS = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "local",
    "service",
    "track",
    "other",
    "unmatched",
)
OSM_CORE_FEATURES = [f"osm_highway_group_{value}" for value in OSM_GROUPS]
OSM_EXTENDED_FEATURES = [
    "osm_lanes",
    "osm_maxspeed_kmh",
    "osm_oneway",
    "osm_link",
    "osm_bridge",
    "osm_tunnel",
    "osm_roundabout",
    "osm_access_restricted",
    "osm_name_present",
]

MODEL_ORDER = (
    "training_median",
    "hierarchy_lookup",
    "deployable_structural_hgb",
    "deployable_structural_gtfs_hgb",
    "deployable_osm_highway_hgb",
    "deployable_osm_extended_hgb",
    "atc_class_oracle_hgb",
)
MODEL_LABELS = {
    "training_median": "Training median",
    "hierarchy_lookup": "10-cell hierarchy lookup",
    "deployable_structural_hgb": "Deployable road structure",
    "deployable_structural_gtfs_hgb": "Deployable structure + GTFS",
    "deployable_osm_highway_hgb": "Deployable context + OSM highway",
    "deployable_osm_extended_hgb": "Deployable context + extended OSM",
    "atc_class_oracle_hgb": "ATC-class oracle (not deployable)",
}
MODEL_COLORS = {
    "training_median": "#A0A0A0",
    "hierarchy_lookup": "#6C757D",
    "deployable_structural_hgb": "#4C78A8",
    "deployable_structural_gtfs_hgb": "#72A0C1",
    "deployable_osm_highway_hgb": "#1B9E77",
    "deployable_osm_extended_hgb": "#59A14F",
    "atc_class_oracle_hgb": "#9C755F",
}

GENERIC_ROAD_TOKENS = {
    "ROAD", "STREET", "AVENUE", "HIGHWAY", "FLYOVER", "BRIDGE",
    "TUNNEL", "BYPASS", "DRIVE", "LANE", "PATH", "WAY", "NEAR",
    "EASTBOUND", "WESTBOUND", "NORTHBOUND", "SOUTHBOUND",
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty result: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


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


def find_road_geodatabase() -> Path:
    candidates = sorted(
        PROJECT_ROOT.glob("data/raw/step22_2023/road_network_*/RdNet_IRNP.gdb")
    )
    if not candidates:
        raise FileNotFoundError(
            "The Step 22 historical road geodatabase is missing; run the corrected "
            "Step 22 script before Step 23A."
        )
    return candidates[-1]


def read_centerline(geodatabase: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata, _, geometry_wkb, field_arrays = read_ogr(
        geodatabase,
        layer="CENTERLINE",
    )
    frame = pd.DataFrame(
        {field: values for field, values in zip(metadata["fields"], field_arrays)}
    )
    geometries = shapely.from_wkb(geometry_wkb)
    frame["road_2023_segment_index"] = np.arange(len(frame), dtype=int)
    frame["road_segment_length_m"] = shapely.length(geometries)
    transformer = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)
    representative = [geometry.interpolate(0.5, normalized=True) for geometry in geometries]
    longitude, latitude = transformer.transform(
        np.asarray([point.x for point in representative]),
        np.asarray([point.y for point in representative]),
    )
    frame["road_longitude"] = longitude
    frame["road_latitude"] = latitude
    return frame, geometries


def bbox_tiles(centerline: pd.DataFrame) -> list[dict[str, object]]:
    margin = 0.03
    west = float(centerline["road_longitude"].min() - margin)
    east = float(centerline["road_longitude"].max() + margin)
    south = float(centerline["road_latitude"].min() - margin)
    north = float(centerline["road_latitude"].max() + margin)
    xs = np.linspace(west, east, TILE_COUNT_PER_AXIS + 1)
    ys = np.linspace(south, north, TILE_COUNT_PER_AXIS + 1)
    return [
        {
            "row": row,
            "column": column,
            "west": float(xs[column]),
            "south": float(ys[row]),
            "east": float(xs[column + 1]),
            "north": float(ys[row + 1]),
            "bbox": f"{xs[column]:.7f},{ys[row]:.7f},{xs[column + 1]:.7f},{ys[row + 1]:.7f}",
        }
        for row in range(TILE_COUNT_PER_AXIS)
        for column in range(TILE_COUNT_PER_AXIS)
    ]


class OverpassTileDownloadError(RuntimeError):
    """A bounded endpoint failure, carrying whether a smaller bbox may help."""

    def __init__(
        self,
        tile_label: str,
        failures: list[str],
        subdivision_flags: list[bool],
    ) -> None:
        self.can_subdivide = bool(subdivision_flags) and all(subdivision_flags)
        super().__init__(
            f"No Overpass endpoint delivered {tile_label}. "
            f"Attempts: {'; '.join(failures)}."
        )


def download_overpass_tile(query: str, path: Path, tile_label: str) -> tuple[str, str]:
    """Download one historical tile without repeatedly hitting one public server."""
    request_data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    failures: list[str] = []
    subdivision_flags: list[bool] = []
    for endpoint_name, endpoint_url in OVERPASS_ENDPOINTS:
        print(f"Downloading {tile_label} via {endpoint_name}...")
        request = urllib.request.Request(
            endpoint_url,
            data=request_data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": OVERPASS_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                response_bytes = response.read()
            payload = json.loads(response_bytes)
            if "elements" not in payload:
                raise ValueError("response JSON has no elements field")
            path.write_bytes(response_bytes)
            time.sleep(OVERPASS_SUCCESS_PAUSE_SECONDS)
            return endpoint_name, endpoint_url
        except urllib.error.HTTPError as exc:
            failures.append(f"{endpoint_name}: HTTP {exc.code}")
            subdivision_flags.append(exc.code == 504)
            if exc.code == 429:
                print(
                    f"  {endpoint_name} returned HTTP 429; waiting "
                    f"{OVERPASS_429_WAIT_SECONDS} seconds before switching endpoint."
                )
                time.sleep(OVERPASS_429_WAIT_SECONDS)
            else:
                print(f"  {endpoint_name} returned HTTP {exc.code}; switching endpoint.")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"{endpoint_name}: {type(exc).__name__}: {exc}")
            timeout_reason = getattr(exc, "reason", None)
            subdivision_flags.append(
                isinstance(exc, TimeoutError) or isinstance(timeout_reason, TimeoutError)
            )
            print(f"  {endpoint_name} did not deliver a valid tile; switching endpoint.")
    raise OverpassTileDownloadError(tile_label, failures, subdivision_flags)


def split_bbox(tile: dict[str, object]) -> list[dict[str, object]]:
    """Split one failed query bbox into four smaller child bboxes."""
    west = float(tile["west"])
    south = float(tile["south"])
    east = float(tile["east"])
    north = float(tile["north"])
    middle_x = (west + east) / 2.0
    middle_y = (south + north) / 2.0
    bounds = (
        (west, south, middle_x, middle_y),
        (middle_x, south, east, middle_y),
        (west, middle_y, middle_x, north),
        (middle_x, middle_y, east, north),
    )
    return [
        {
            **tile,
            "west": child_west,
            "south": child_south,
            "east": child_east,
            "north": child_north,
            "bbox": (
                f"{child_west:.7f},{child_south:.7f},"
                f"{child_east:.7f},{child_north:.7f}"
            ),
        }
        for child_west, child_south, child_east, child_north in bounds
    ]


def obtain_osm_tile_parts(
    tile: dict[str, object],
    path: Path,
    top_tile_label: str,
    subdivision_depth: int = 0,
    subdivision_id: str = "root",
) -> list[dict[str, object]]:
    """Obtain one bbox, subdividing only after all endpoints time out."""
    child_tiles = split_bbox(tile)
    child_paths = [
        path.with_name(f"{path.stem}_q{child_index}{path.suffix}")
        for child_index in range(4)
    ]
    path_is_cached = path.exists() and path.stat().st_size >= 100

    if path_is_cached:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "elements" not in payload:
            raise ValueError(f"Unexpected Overpass response in {path.name}")
        return [
            {
                "payload": payload,
                "source": "cached",
                "endpoint_name": "cached_previous_download",
                "endpoint": "not_requeried",
                "bbox": tile["bbox"],
                "subdivision_depth": subdivision_depth,
                "subdivision_id": subdivision_id,
            }
        ]

    reusable_child_exists = any(
        child_path.exists() and child_path.stat().st_size >= 100
        for child_path in child_paths
    )
    if reusable_child_exists and subdivision_depth < MAX_TILE_SUBDIVISION_DEPTH:
        print(f"Resuming cached subtiles for {top_tile_label}, part {subdivision_id}...")
        parts: list[dict[str, object]] = []
        for child_index, (child_tile, child_path) in enumerate(
            zip(child_tiles, child_paths)
        ):
            parts.extend(
                obtain_osm_tile_parts(
                    child_tile,
                    child_path,
                    top_tile_label,
                    subdivision_depth + 1,
                    f"{subdivision_id}.{child_index}",
                )
            )
        return parts

    query = (
        f'[out:json][timeout:300][date:"{OSM_TIMESTAMP}"];'
        f'way["highway"]('
        f'{tile["south"]:.7f},{tile["west"]:.7f},'
        f'{tile["north"]:.7f},{tile["east"]:.7f});'
        "out tags geom;"
    )
    part_label = f"{top_tile_label}, part {subdivision_id}"
    try:
        endpoint_name, endpoint_url = download_overpass_tile(query, path, part_label)
    except OverpassTileDownloadError as exc:
        if not exc.can_subdivide or subdivision_depth >= MAX_TILE_SUBDIVISION_DEPTH:
            raise RuntimeError(
                f"{exc} Existing cached tiles remain usable; rerun later to resume "
                "from the first missing subtile."
            ) from exc
        print(
            f"Both endpoints timed out for {part_label}; splitting only this "
            "bbox into four smaller subtiles."
        )
        parts = []
        for child_index, (child_tile, child_path) in enumerate(
            zip(child_tiles, child_paths)
        ):
            parts.extend(
                obtain_osm_tile_parts(
                    child_tile,
                    child_path,
                    top_tile_label,
                    subdivision_depth + 1,
                    f"{subdivision_id}.{child_index}",
                )
            )
        return parts

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "payload": payload,
            "source": "downloaded",
            "endpoint_name": endpoint_name,
            "endpoint": endpoint_url,
            "bbox": tile["bbox"],
            "subdivision_depth": subdivision_depth,
            "subdivision_id": subdivision_id,
        }
    ]


def obtain_osm_tiles(centerline: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    unique_features: dict[str, dict[str, object]] = {}
    duplicate_tile_features = 0
    audit_rows: list[dict[str, object]] = []
    tiles = bbox_tiles(centerline)
    for tile_number, tile in enumerate(tiles, start=1):
        path = RAW_DIR / f"osm_highway_20230630_r{tile['row']}_c{tile['column']}.json"
        top_tile_label = (
            f"OSM tile {tile_number}/{len(tiles)} "
            f"(row {tile['row']}, column {tile['column']})"
        )
        for part in obtain_osm_tile_parts(tile, path, top_tile_label):
            payload = part["payload"]
            features = payload.get("elements", [])
            for feature_position, feature in enumerate(features):
                osm_id = (
                    f"way/{feature.get('id')}"
                    if feature.get("type") == "way" and feature.get("id") is not None
                    else ""
                )
                key = (
                    osm_id
                    or f"tile_{tile['row']}_{tile['column']}_"
                    f"{part['subdivision_id']}_feature_{feature_position}"
                )
                if key in unique_features:
                    duplicate_tile_features += 1
                else:
                    unique_features[key] = feature
            audit_rows.append(
                {
                    "scope": "tile_or_subtile",
                    "tile_row": tile["row"],
                    "tile_column": tile["column"],
                    "subdivision_depth": part["subdivision_depth"],
                    "subdivision_id": part["subdivision_id"],
                    "bbox": part["bbox"],
                    "feature_count": len(features),
                    "source": part["source"],
                    "timestamp": OSM_TIMESTAMP,
                    "filter": OSM_FILTER,
                    "endpoint_name": part["endpoint_name"],
                    "endpoint": part["endpoint"],
                    "documentation": OVERPASS_DOC,
                    "attribution": payload.get("osm3s", {}).get("copyright", ""),
                    "server_generator": payload.get("generator", ""),
                }
            )
    audit_rows.append(
        {
            "scope": "combined_unique_features",
            "tile_row": np.nan,
            "tile_column": np.nan,
            "bbox": f"union_of_{TILE_COUNT_PER_AXIS ** 2}_tiles",
            "feature_count": len(unique_features),
            "source": "deduplicated_by_osm_id",
            "timestamp": OSM_TIMESTAMP,
            "filter": OSM_FILTER,
            "endpoint_name": "multiple_or_cached",
            "endpoint": ";".join(url for _, url in OVERPASS_ENDPOINTS),
            "documentation": OVERPASS_DOC,
            "attribution": "© OpenStreetMap contributors",
            "duplicate_tile_features_removed": duplicate_tile_features,
        }
    )
    return list(unique_features.values()), pd.DataFrame(audit_rows)


def first_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def parse_number(value: object) -> float:
    text = first_value(value).strip().lower()
    if not text:
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def parse_speed(value: object) -> float:
    speed = parse_number(value)
    text = first_value(value).lower()
    if np.isnan(speed):
        return np.nan
    return speed * 1.609344 if "mph" in text else speed


def flag(value: object, true_values: set[str]) -> int:
    return int(first_value(value).strip().lower() in true_values)


def highway_group(value: object) -> str:
    highway = first_value(value).strip().lower()
    base = highway.removesuffix("_link")
    if base in {"motorway", "trunk", "primary", "secondary", "tertiary"}:
        return base
    if base in {"unclassified", "residential", "living_street", "road"}:
        return "local"
    if base == "service":
        return "service"
    if base == "track":
        return "track"
    return "other"


def parse_osm_ways(
    features: list[dict[str, object]],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    rows: list[dict[str, object]] = []
    geometries: list[object] = []
    seen_ids: set[str] = set()
    duplicate_count = 0
    unsupported_geometry_count = 0
    for feature in features:
        properties = feature.get("tags", {}) or {}
        osm_id = (
            f"way/{feature.get('id')}"
            if feature.get("type") == "way" and feature.get("id") is not None
            else ""
        )
        if not osm_id or osm_id in seen_ids:
            duplicate_count += int(bool(osm_id))
            continue
        geometry_nodes = feature.get("geometry", [])
        if len(geometry_nodes) < 2:
            unsupported_geometry_count += 1
            continue
        geometry_wgs84 = shape(
            {
                "type": "LineString",
                "coordinates": [
                    [float(node["lon"]), float(node["lat"])]
                    for node in geometry_nodes
                ],
            }
        )
        geometry = transform_geometry(transformer.transform, geometry_wgs84)
        if geometry.is_empty or geometry.length <= 0:
            unsupported_geometry_count += 1
            continue
        seen_ids.add(osm_id)
        highway = first_value(properties.get("highway")).strip().lower()
        rows.append(
            {
                "osm_id": osm_id,
                "osm_version": feature.get("version"),
                "osm_valid_from": OSM_TIMESTAMP,
                "osm_highway": highway,
                "osm_highway_group": highway_group(highway),
                "osm_name": first_value(
                    properties.get("name:en", properties.get("name", ""))
                ),
                "osm_lanes": parse_number(properties.get("lanes")),
                "osm_maxspeed_kmh": parse_speed(properties.get("maxspeed")),
                "osm_oneway": flag(properties.get("oneway"), {"yes", "true", "1", "-1"}),
                "osm_link": int(highway.endswith("_link")),
                "osm_bridge": flag(properties.get("bridge"), {"yes", "true", "1"}),
                "osm_tunnel": flag(properties.get("tunnel"), {"yes", "true", "1"}),
                "osm_roundabout": flag(properties.get("junction"), {"roundabout", "circular"}),
                "osm_access_restricted": flag(
                    properties.get("access"),
                    {"no", "private", "destination", "customers", "permit"},
                ),
                "osm_name_present": int(
                    bool(first_value(properties.get("name:en", properties.get("name", ""))))
                ),
                "osm_way_length_m": float(geometry.length),
            }
        )
        geometries.append(geometry)
    ways = pd.DataFrame(rows)
    source_totals = pd.DataFrame(
        [
            {
                "scope": "combined_unique_ways",
                "tile_row": np.nan,
                "tile_column": np.nan,
                "bbox": f"union_of_{TILE_COUNT_PER_AXIS ** 2}_tiles",
                "feature_count": len(ways),
                "source": "deduplicated_by_osm_id",
                "timestamp": OSM_TIMESTAMP,
                "filter": OSM_FILTER,
                "endpoint_name": "multiple_or_cached",
                "endpoint": ";".join(url for _, url in OVERPASS_ENDPOINTS),
                "documentation": OVERPASS_DOC,
                "attribution": "© OpenStreetMap contributors",
                "duplicate_tile_features_removed": duplicate_count,
                "unsupported_or_empty_geometries": unsupported_geometry_count,
            }
        ]
    )
    save_csv(ways, OSM_WAY_TABLE_PATH)
    return ways, np.asarray(geometries, dtype=object), source_totals


def match_centerline_to_osm(
    centerline: pd.DataFrame,
    road_geometries: np.ndarray,
    ways: pd.DataFrame,
    osm_geometries: np.ndarray,
) -> pd.DataFrame:
    tree = STRtree(osm_geometries)
    rows: list[dict[str, object]] = []
    for position, road in enumerate(centerline.itertuples(index=False)):
        road_geometry = road_geometries[position]
        candidates = tree.query(
            road_geometry,
            predicate="dwithin",
            distance=MATCH_SEARCH_M,
        )
        selected: int | None = None
        selected_values: tuple[float, float, float, float] | None = None
        official_names = [
            getattr(road, "STREET_ENAME", ""),
            getattr(road, "ALIAS_ENAME", ""),
        ]
        for candidate in candidates:
            candidate = int(candidate)
            osm_geometry = osm_geometries[candidate]
            distance_m = float(road_geometry.distance(osm_geometry))
            overlap_share = float(
                min(
                    1.0,
                    road_geometry.intersection(osm_geometry.buffer(MATCH_BUFFER_M)).length
                    / max(float(road_geometry.length), 1.0),
                )
            )
            similarity = max(
                name_similarity(name, ways.iloc[candidate]["osm_name"])
                for name in official_names
            )
            score = (
                2.5 * overlap_share
                + 0.75 * similarity
                - 0.30 * min(distance_m, MATCH_SEARCH_M) / MATCH_SEARCH_M
            )
            values = (score, overlap_share, similarity, -distance_m)
            if selected_values is None or values > selected_values:
                selected = candidate
                selected_values = values

        base = {
            "road_2023_segment_index": int(road.road_2023_segment_index),
            "road_segment_length_m": float(road.road_segment_length_m),
        }
        if selected is None or selected_values is None:
            rows.append(
                {
                    **base,
                    "osm_id": "",
                    "osm_match_distance_m": np.nan,
                    "osm_overlap_share": 0.0,
                    "osm_name_similarity": 0.0,
                    "osm_match_status": "unmatched",
                    "osm_highway": "",
                    "osm_highway_group": "unmatched",
                    **{column: np.nan for column in OSM_EXTENDED_FEATURES},
                }
            )
            continue

        _, overlap_share, similarity, negative_distance = selected_values
        distance_m = -negative_distance
        status = (
            "high"
            if distance_m <= 15.0 and (overlap_share >= 0.50 or similarity >= 0.50)
            else "moderate"
            if (
                distance_m <= 30.0
                and (overlap_share >= 0.20 or similarity >= 0.25)
            ) or distance_m <= 8.0
            else "low"
        )
        osm = ways.iloc[selected]
        accepted = status in {"high", "moderate"}
        rows.append(
            {
                **base,
                "osm_id": osm["osm_id"],
                "osm_match_distance_m": distance_m,
                "osm_overlap_share": overlap_share,
                "osm_name_similarity": similarity,
                "osm_match_status": status,
                "osm_highway": osm["osm_highway"] if accepted else "",
                "osm_highway_group": osm["osm_highway_group"] if accepted else "unmatched",
                **{
                    column: osm[column] if accepted else np.nan
                    for column in OSM_EXTENDED_FEATURES
                },
            }
        )
    crosswalk = pd.DataFrame(rows)
    for group in OSM_GROUPS:
        crosswalk[f"osm_highway_group_{group}"] = (
            crosswalk["osm_highway_group"] == group
        ).astype(int)
    if crosswalk["road_2023_segment_index"].duplicated().any():
        raise ValueError("OSM crosswalk contains duplicate official segment identifiers")
    save_csv(crosswalk, OSM_CROSSWALK_PATH)
    return crosswalk


def attach_segment_identifier(stations: pd.DataFrame) -> pd.DataFrame:
    if "road_2023_segment_index" in stations.columns:
        return stations
    matches = pd.read_csv(STEP22_ROAD_MATCH_PATH)[
        ["station_id", "road_2023_segment_index"]
    ]
    return stations.merge(matches, on="station_id", how="left", validate="one_to_one")


def build_feature_tables(
    network: pd.DataFrame,
    stations: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    crosswalk_features = [
        "road_2023_segment_index",
        "osm_id",
        "osm_match_distance_m",
        "osm_overlap_share",
        "osm_name_similarity",
        "osm_match_status",
        "osm_highway",
        "osm_highway_group",
        *OSM_CORE_FEATURES,
        *OSM_EXTENDED_FEATURES,
    ]
    network = network.merge(
        crosswalk[crosswalk_features],
        on="road_2023_segment_index",
        how="left",
        validate="one_to_one",
    )
    stations = attach_segment_identifier(stations).merge(
        crosswalk[crosswalk_features],
        on="road_2023_segment_index",
        how="left",
        validate="many_to_one",
    )

    for column in ("aadt", "spatial_fold", "station_id"):
        if column not in stations:
            raise ValueError(f"Step 22 station table lacks required column: {column}")
    if stations["road_2023_segment_index"].isna().any():
        raise ValueError("A measured station has no matched official segment identifier")

    gtfs_features = sorted(
        column for column in network.columns if column.startswith("gtfs_")
    )
    if not gtfs_features:
        raise ValueError("The corrected Step 22 full-network GTFS features are missing")
    required_network = [*STRUCTURAL_FEATURES, *gtfs_features]
    missing = [column for column in required_network if column not in network]
    if missing:
        raise ValueError(f"Step 22 full-network table lacks: {', '.join(missing)}")
    missing_station = [column for column in required_network if column not in stations]
    if missing_station:
        stations = stations.drop(columns=missing_station, errors="ignore").merge(
            network[["road_2023_segment_index", *missing_station]],
            on="road_2023_segment_index",
            how="left",
            validate="many_to_one",
        )

    for value in sorted(stations["road_type"].dropna().astype(str).unique()):
        column = "oracle_road_type_" + re.sub(
            r"[^a-z0-9]+", "_", value.lower()
        ).strip("_")
        stations[column] = (stations["road_type"].astype(str) == value).astype(int)
    stations["oracle_road_network_major"] = (
        stations["road_network"].astype(str) == "MAJOR"
    ).astype(int)

    save_csv(network, NETWORK_FEATURE_PATH)
    save_csv(stations, STATION_FEATURE_PATH)
    return network, stations, gtfs_features


def coverage_audits(
    crosswalk: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    accepted = crosswalk["osm_match_status"].isin(["high", "moderate"])
    length = crosswalk["road_segment_length_m"].to_numpy(dtype=float)
    station_status = stations["osm_match_status"].isin(["high", "moderate"])
    minor = stations["road_network"].astype(str).eq("MINOR")
    rows = [
        {
            "metric": "official_segment_count",
            "value": len(crosswalk),
            "threshold": np.nan,
            "pass": True,
            "interpretation": "complete 2023 official centreline denominator",
        },
        {
            "metric": "all_network_high_or_moderate_segment_share",
            "value": accepted.mean(),
            "threshold": np.nan,
            "pass": True,
            "interpretation": "unweighted mapping coverage; descriptive only",
        },
        {
            "metric": "all_network_high_or_moderate_length_share",
            "value": float(length[accepted].sum() / length.sum()),
            "threshold": NETWORK_LENGTH_COVERAGE_THRESHOLD,
            "pass": float(length[accepted].sum() / length.sum())
            >= NETWORK_LENGTH_COVERAGE_THRESHOLD,
            "interpretation": "primary full-network OSM coverage gate",
        },
        {
            "metric": "measured_station_high_or_moderate_share",
            "value": station_status.mean(),
            "threshold": STATION_COVERAGE_THRESHOLD,
            "pass": station_status.mean() >= STATION_COVERAGE_THRESHOLD,
            "interpretation": "coverage on the 2023 validation sample",
        },
        {
            "metric": "minor_station_high_or_moderate_share",
            "value": station_status[minor].mean(),
            "threshold": MINOR_STATION_COVERAGE_THRESHOLD,
            "pass": station_status[minor].mean() >= MINOR_STATION_COVERAGE_THRESHOLD,
            "interpretation": "coverage on the weakly supported local-road target",
        },
        {
            "metric": "median_accepted_match_distance_m",
            "value": crosswalk.loc[accepted, "osm_match_distance_m"].median(),
            "threshold": np.nan,
            "pass": True,
            "interpretation": "geometry compatibility diagnostic",
        },
        {
            "metric": "median_accepted_overlap_share",
            "value": crosswalk.loc[accepted, "osm_overlap_share"].median(),
            "threshold": np.nan,
            "pass": True,
            "interpretation": "official-segment length captured by a 12m OSM buffer",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, MATCH_AUDIT_PATH)
    return frame


def class_agreement(stations: pd.DataFrame) -> pd.DataFrame:
    frame = (
        stations.groupby(["road_network", "road_type", "osm_highway_group"], dropna=False)
        .agg(
            station_count=("station_id", "size"),
            median_observed_aadt=("aadt", "median"),
        )
        .reset_index()
    )
    frame["interpretation"] = (
        "taxonomy_compatibility_diagnostic_only_not_a_model_feature_or_success_gate"
    )
    save_csv(frame, CLASS_AGREEMENT_PATH)
    return frame


def feature_manifest(
    network: pd.DataFrame,
    gtfs_features: list[str],
    oracle_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        (STRUCTURAL_FEATURES, "official_road_network", "deployable_structure", True),
        (gtfs_features, "historical_gtfs", "deployable_context", True),
        (OSM_CORE_FEATURES, "overpass_osm_2023_snapshot", "primary_osm_class", True),
        (OSM_EXTENDED_FEATURES, "overpass_osm_2023_snapshot", "secondary_osm_tags", True),
    )
    for features, source, role, allowed in groups:
        for feature in features:
            rows.append(
                {
                    "feature": feature,
                    "source": source,
                    "role": role,
                    "available_segment_count": int(network[feature].notna().sum()),
                    "total_segment_count": len(network),
                    "available_segment_share": network[feature].notna().mean(),
                    "allowed_in_deployable_model": allowed,
                }
            )
    for feature in oracle_features:
        rows.append(
            {
                "feature": feature,
                "source": "atc_station_metadata",
                "role": "oracle_only",
                "available_segment_count": 0,
                "total_segment_count": len(network),
                "available_segment_share": 0.0,
                "allowed_in_deployable_model": False,
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, FEATURE_MANIFEST_PATH)
    return frame


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
        raise ValueError("Street-code count cannot form the frozen hierarchy lookup")

    def assign(values: pd.Series) -> np.ndarray:
        return np.clip(
            np.digitize(pd.to_numeric(values), edges[1:-1], right=True),
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
            cell_medians.get((route, level), route_medians.get(route, global_median))
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
        if len(observed) > 2 and np.std(observed) > 0 and np.std(predicted) > 0
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
    stations: pd.DataFrame,
    gtfs_features: list[str],
    oracle_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure_gtfs = [*STRUCTURAL_FEATURES, *gtfs_features]
    osm_core = [*structure_gtfs, *OSM_CORE_FEATURES]
    osm_extended = [*osm_core, *OSM_EXTENDED_FEATURES]
    oracle = [*structure_gtfs, *oracle_features]
    model_features = {
        "deployable_structural_hgb": STRUCTURAL_FEATURES,
        "deployable_structural_gtfs_hgb": structure_gtfs,
        "deployable_osm_highway_hgb": osm_core,
        "deployable_osm_extended_hgb": osm_extended,
        "atc_class_oracle_hgb": oracle,
    }
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        train = stations[stations["spatial_fold"].astype(int) != fold].copy()
        test = stations[stations["spatial_fold"].astype(int) == fold].copy()
        y_train = train["aadt"].to_numpy(dtype=float)
        y_test = test["aadt"].to_numpy(dtype=float)
        predictions = {
            "training_median": np.full(len(test), np.median(y_train)),
            "hierarchy_lookup": hierarchy_lookup_predict(train, test),
        }
        for model_name, features in model_features.items():
            model = fixed_model()
            model.fit(matrix(train, features), y_train)
            predictions[model_name] = model.predict(matrix(test, features))
        for model_name in MODEL_ORDER:
            predicted = predictions[model_name]
            metric_rows.append(metric_row(fold, model_name, y_test, predicted))
            for row_position, (_, station) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "station_id": int(station["station_id"]),
                        "spatial_fold": fold,
                        "region": station["region"],
                        "road_network": station["road_network"],
                        "road_type": station["road_type"],
                        "osm_match_status": station["osm_match_status"],
                        "osm_highway_group": station["osm_highway_group"],
                        "model": model_name,
                        "observed_aadt": y_test[row_position],
                        "predicted_aadt": predicted[row_position],
                        "absolute_error": abs(predicted[row_position] - y_test[row_position]),
                    }
                )
        print(f"Completed spatial fold {fold}: train={len(train)}, test={len(test)}")
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    save_csv(predictions, PREDICTION_PATH)
    save_csv(metrics, METRICS_BY_FOLD_PATH)
    return predictions, metrics


def summarise_models(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        selected = predictions[predictions["model"] == model_name]
        row = metric_row(
            "pooled",
            model_name,
            selected["observed_aadt"].to_numpy(dtype=float),
            selected["predicted_aadt"].to_numpy(dtype=float),
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    lookup_mae = frame.loc[frame["model"] == "hierarchy_lookup", "mae"].iloc[0]
    context_mae = frame.loc[
        frame["model"] == "deployable_structural_gtfs_hgb", "mae"
    ].iloc[0]
    frame["mae_improvement_vs_hierarchy_pct"] = (
        100.0 * (lookup_mae - frame["mae"]) / lookup_mae
    )
    frame["mae_improvement_vs_structure_gtfs_pct"] = (
        100.0 * (context_mae - frame["mae"]) / context_mae
    )
    save_csv(frame, SUMMARY_PATH)
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
        int(fold): group["loss_difference"].to_numpy(dtype=float)
        for fold, group in paired.groupby("spatial_fold")
    }
    rng = np.random.default_rng(42)
    fold_ids = np.asarray(sorted(fold_values))
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled_folds = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        estimates[draw] = np.concatenate(
            [fold_values[int(fold)] for fold in sampled_folds]
        ).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    specifications = (
        ("deployable_structural_hgb", "hierarchy_lookup", "all"),
        ("deployable_structural_gtfs_hgb", "hierarchy_lookup", "all"),
        ("deployable_osm_highway_hgb", "hierarchy_lookup", "all"),
        ("deployable_osm_highway_hgb", "deployable_structural_gtfs_hgb", "all"),
        ("deployable_osm_extended_hgb", "deployable_osm_highway_hgb", "all"),
        ("atc_class_oracle_hgb", "hierarchy_lookup", "all"),
        ("deployable_osm_highway_hgb", "deployable_structural_gtfs_hgb", "minor"),
    )
    rows: list[dict[str, object]] = []
    for candidate_name, reference_name, subset in specifications:
        candidate = predictions[predictions["model"] == candidate_name]
        reference = predictions[predictions["model"] == reference_name]
        if subset == "minor":
            candidate = candidate[candidate["road_network"].astype(str) == "MINOR"]
            reference = reference[reference["road_network"].astype(str) == "MINOR"]
        candidate_mae = candidate["absolute_error"].mean()
        reference_mae = reference["absolute_error"].mean()
        low, high = cluster_bootstrap_loss_difference(candidate, reference)
        candidate_folds = candidate.groupby("spatial_fold")["absolute_error"].mean()
        reference_folds = reference.groupby("spatial_fold")["absolute_error"].mean()
        improved_folds = int((candidate_folds < reference_folds).sum())
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                "evaluation_subset": subset,
                "n": len(candidate),
                "candidate_mae": candidate_mae,
                "reference_mae": reference_mae,
                "mae_improvement_pct": 100.0 * (reference_mae - candidate_mae) / reference_mae,
                "mean_absolute_loss_difference": candidate_mae - reference_mae,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "improved_fold_count": improved_folds,
                "interpretation": "negative_loss_difference_favours_candidate",
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, COMPARISON_PATH)
    return frame


def subgroup_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        model = predictions[predictions["model"] == model_name]
        strata: list[tuple[str, pd.DataFrame]] = [("all", model)]
        for column in ("region", "road_network", "road_type", "osm_match_status"):
            strata.extend(
                (f"{column}:{value}", group)
                for value, group in model.groupby(column, dropna=False)
            )
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
                    "mae": mean_absolute_error(observed, predicted),
                    "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, SUBGROUP_PATH)
    return frame


def failed_criteria(
    improvement_pct: float,
    threshold_pct: float,
    interval_high: float,
    improved_fold_count: int,
    minimum_improved_folds: int = 3,
) -> dict[str, object]:
    effect_pass = improvement_pct >= threshold_pct
    interval_pass = interval_high < 0
    fold_pass = improved_fold_count >= minimum_improved_folds
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
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    subgroups: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    coverage_lookup = coverage.set_index("metric")
    comparison = comparisons.set_index(["candidate", "reference", "evaluation_subset"])
    summary_lookup = summary.set_index("model")
    vs_lookup = comparison.loc[("deployable_osm_highway_hgb", "hierarchy_lookup", "all")]
    vs_context = comparison.loc[
        ("deployable_osm_highway_hgb", "deployable_structural_gtfs_hgb", "all")
    ]
    extended_vs_core = comparison.loc[
        ("deployable_osm_extended_hgb", "deployable_osm_highway_hgb", "all")
    ]
    minor_vs_context = comparison.loc[
        ("deployable_osm_highway_hgb", "deployable_structural_gtfs_hgb", "minor")
    ]
    vs_lookup_diag = failed_criteria(
        vs_lookup["mae_improvement_pct"],
        SKILL_VS_HIERARCHY_THRESHOLD_PCT,
        vs_lookup["cluster_bootstrap_high"],
        int(vs_lookup["improved_fold_count"]),
    )
    vs_context_diag = failed_criteria(
        vs_context["mae_improvement_pct"],
        INCREMENT_VS_CONTEXT_THRESHOLD_PCT,
        vs_context["cluster_bootstrap_high"],
        int(vs_context["improved_fold_count"]),
    )
    extended_diag = failed_criteria(
        extended_vs_core["mae_improvement_pct"],
        INCREMENT_VS_CONTEXT_THRESHOLD_PCT,
        extended_vs_core["cluster_bootstrap_high"],
        int(extended_vs_core["improved_fold_count"]),
    )
    minor_diag = failed_criteria(
        minor_vs_context["mae_improvement_pct"],
        MINOR_INCREMENT_THRESHOLD_PCT,
        minor_vs_context["cluster_bootstrap_high"],
        int(minor_vs_context["improved_fold_count"]),
    )

    def diagnostic_pass(values: dict[str, object]) -> bool:
        return bool(
            values["effect_threshold_pass"]
            and values["interval_excludes_zero"]
            and values["fold_consistency_pass"]
        )

    network_coverage = bool(
        coverage_lookup.loc["all_network_high_or_moderate_length_share", "pass"]
    )
    station_coverage = bool(
        coverage_lookup.loc["measured_station_high_or_moderate_share", "pass"]
    )
    minor_coverage = bool(
        coverage_lookup.loc["minor_station_high_or_moderate_share", "pass"]
    )
    lineage_gate = not manifest.loc[
        manifest["allowed_in_deployable_model"], "source"
    ].eq("atc_station_metadata").any()
    core_summary = summary_lookup.loc["deployable_osm_highway_hgb"]
    aggregate_gate = abs(core_summary["aggregate_bias_pct"]) <= AGGREGATE_BIAS_THRESHOLD_PCT
    bias_groups = subgroups[
        (subgroups["model"] == "deployable_osm_highway_hgb")
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ]
    max_subgroup_bias = bias_groups["aggregate_bias_pct"].abs().max()
    subgroup_gate = max_subgroup_bias <= SUBGROUP_BIAS_THRESHOLD_PCT
    minor_bias = subgroups.loc[
        (subgroups["model"] == "deployable_osm_highway_hgb")
        & (subgroups["stratum"] == "road_network:MINOR"),
        "aggregate_bias_pct",
    ].iloc[0]
    minor_bias_gate = abs(minor_bias) <= SUBGROUP_BIAS_THRESHOLD_PCT
    skill_gate = diagnostic_pass(vs_lookup_diag)
    increment_gate = diagnostic_pass(vs_context_diag)
    minor_skill_gate = diagnostic_pass(minor_diag)
    extended_retention = diagnostic_pass(extended_diag)
    full_gate = all(
        [
            network_coverage,
            station_coverage,
            minor_coverage,
            lineage_gate,
            skill_gate,
            increment_gate,
            aggregate_gate,
            subgroup_gate,
            minor_skill_gate,
            minor_bias_gate,
        ]
    )
    full_failures = [
        name
        for name, passed in (
            ("full_network_osm_coverage_gate", network_coverage),
            ("station_osm_coverage_gate", station_coverage),
            ("minor_station_osm_coverage_gate", minor_coverage),
            ("deployable_predictor_lineage_gate", lineage_gate),
            ("skill_vs_hierarchy_gate", skill_gate),
            ("increment_vs_structure_gtfs_gate", increment_gate),
            ("aggregate_bias_gate", aggregate_gate),
            ("region_and_network_bias_gate", subgroup_gate),
            ("minor_road_increment_gate", minor_skill_gate),
            ("minor_road_bias_gate", minor_bias_gate),
        )
        if not passed
    ]

    def comparison_evidence(row: pd.Series) -> str:
        return (
            f"improvement={row['mae_improvement_pct']:.2f}%; "
            f"interval=[{row['cluster_bootstrap_low']:.1f}, "
            f"{row['cluster_bootstrap_high']:.1f}]; "
            f"improved_folds={int(row['improved_fold_count'])}/5"
        )

    rows = [
        {
            "decision": "osm_2023_full_network_length_coverage_is_adequate",
            "pass": network_coverage,
            "evidence": (
                f"length_share={coverage_lookup.loc['all_network_high_or_moderate_length_share', 'value']:.3f}; "
                f"threshold={NETWORK_LENGTH_COVERAGE_THRESHOLD:.2f}"
            ),
            "failed_criterion": "none" if network_coverage else "network_length_coverage_below_threshold",
            "action": "do not treat unmatched or low-confidence OSM classes as known road class",
        },
        {
            "decision": "osm_2023_station_and_minor_support_is_adequate",
            "pass": station_coverage and minor_coverage,
            "evidence": (
                f"all_stations={coverage_lookup.loc['measured_station_high_or_moderate_share', 'value']:.3f}; "
                f"minor_stations={coverage_lookup.loc['minor_station_high_or_moderate_share', 'value']:.3f}"
            ),
            "failed_criterion": "none" if station_coverage and minor_coverage else "station_or_minor_coverage_below_threshold",
            "action": "require adequate support both overall and on the local-road target",
        },
        {
            "decision": "osm_predictor_lineage_is_deployable",
            "pass": lineage_gate,
            "evidence": "all OSM features are generated before station outcomes are joined; ATC class is oracle-only",
            "failed_criterion": "none" if lineage_gate else "atc_station_metadata_in_deployable_features",
            "action": "exclude ATC road class and all station-match diagnostics from deployment claims",
        },
        {
            "decision": "osm_highway_materially_beats_honest_hierarchy_lookup",
            "pass": skill_gate,
            "evidence": comparison_evidence(vs_lookup),
            **vs_lookup_diag,
            "action": "require at least 5% pooled MAE gain with interval and fold consistency",
        },
        {
            "decision": "osm_highway_adds_skill_beyond_structure_and_gtfs",
            "pass": increment_gate,
            "evidence": comparison_evidence(vs_context),
            **vs_context_diag,
            "action": "attribute new skill to external road class only if its incremental 2% gate passes",
        },
        {
            "decision": "osm_secondary_tags_add_skill_beyond_highway_class",
            "pass": extended_retention,
            "evidence": comparison_evidence(extended_vs_core),
            **extended_diag,
            "action": "retain secondary OSM tags only if their separate 2% gate passes",
        },
        {
            "decision": "osm_highway_improves_minor_road_predictions",
            "pass": minor_skill_gate,
            "evidence": comparison_evidence(minor_vs_context),
            **minor_diag,
            "action": "require improvement on the weakly supported target, not only the pooled station sample",
        },
        {
            "decision": "osm_highway_aggregate_bias_is_acceptable",
            "pass": aggregate_gate,
            "evidence": f"aggregate_bias={core_summary['aggregate_bias_pct']:+.2f}%",
            "failed_criterion": "none" if aggregate_gate else "aggregate_bias_exceeds_10pct",
            "action": "require absolute pooled bias no greater than 10%",
        },
        {
            "decision": "osm_highway_subgroup_bias_is_acceptable",
            "pass": subgroup_gate and minor_bias_gate,
            "evidence": (
                f"maximum_region_or_network_bias={max_subgroup_bias:.2f}%; "
                f"minor_bias={minor_bias:+.2f}%"
            ),
            "failed_criterion": "none" if subgroup_gate and minor_bias_gate else "region_or_road_network_bias_exceeds_15pct",
            "action": "require region and MAJOR/MINOR absolute bias no greater than 15%",
        },
        {
            "decision": "step23a_2023_osm_full_network_gate",
            "pass": full_gate,
            "evidence": (
                f"coverage={network_coverage and station_coverage and minor_coverage}; "
                f"skill_vs_lookup={skill_gate}; increment_vs_context={increment_gate}; "
                f"minor_skill={minor_skill_gate}; aggregate_bias={aggregate_gate}; "
                f"subgroup_bias={subgroup_gate and minor_bias_gate}"
            ),
            "failed_criterion": ";".join(full_failures) if full_failures else "none",
            "action": (
                "authorise Step 23B historical OSM stability and availability audit"
                if full_gate
                else "do not extend OSM to multi-year modelling; diagnose the failed 2023 component first"
            ),
        },
        {
            "decision": "step23b_historical_osm_audit_is_authorised",
            "pass": full_gate,
            "evidence": "historical extraction is conditional on the complete 2023 gate, not only pooled MAE",
            "failed_criterion": "none" if full_gate else "step23a_full_gate_failed",
            "action": "test 2011/2016/2021 OSM coverage and tag stability only after the 2023 gate passes",
        },
        {
            "decision": "step23a_establishes_multiyear_segment_backcasting",
            "pass": False,
            "evidence": "Step 23A uses one 2023 OSM state and one 2023 outcome year",
            "failed_criterion": "no_multiyear_segment_level_validation",
            "action": "do not describe this experiment as historical backcasting",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, DECISION_PATH)
    return frame


def plot_coverage(crosswalk: pd.DataFrame) -> None:
    status_order = ["high", "moderate", "low", "unmatched"]
    status_share = crosswalk["osm_match_status"].value_counts(normalize=True).reindex(status_order).fillna(0)
    group_share = (
        crosswalk["osm_highway_group"].value_counts(normalize=True).reindex(OSM_GROUPS).fillna(0)
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(status_order, 100 * status_share, color=["#1B9E77", "#80CDC1", "#F2C14E", "#9E9E9E"])
    axes[0].set_ylabel("Share of official segments (%)")
    axes[0].set_title("OSM-to-centreline match status")
    axes[1].barh(list(OSM_GROUPS), 100 * group_share, color="#4C78A8")
    axes[1].set_xlabel("Share of official segments (%)")
    axes[1].set_title("Accepted OSM highway group")
    figure.suptitle("Step 23A: 2023 OSM coverage before modelling")
    figure.tight_layout()
    figure.savefig(COVERAGE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_models(summary: pd.DataFrame) -> None:
    selected = summary.set_index("model").loc[list(MODEL_ORDER)]
    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(selected))
    bars = axis.bar(
        positions,
        selected["mae"],
        color=[MODEL_COLORS[model] for model in selected.index],
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([MODEL_LABELS[model] for model in selected.index], rotation=25, ha="right")
    axis.set_ylabel("Spatial OOF MAE (vehicles/day)")
    axis.set_title("Does independent OSM road class add deployable 2023 skill?")
    for bar, value in zip(bars, selected["mae"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(MODEL_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {MODEL_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_bias(subgroups: pd.DataFrame) -> None:
    selected = subgroups[
        subgroups["model"].isin(["hierarchy_lookup", "deployable_structural_gtfs_hgb", "deployable_osm_highway_hgb"])
        & (
            subgroups["stratum"].str.startswith("region:")
            | subgroups["stratum"].str.startswith("road_network:")
        )
    ]
    pivot = selected.pivot(index="stratum", columns="model", values="aggregate_bias_pct")
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot.rename(columns=MODEL_LABELS).plot.bar(ax=axis, color=[MODEL_COLORS.get(column, "#777777") for column in pivot.columns])
    axis.axhline(0, color="#333333", linewidth=1)
    axis.axhline(15, color="#C44E52", linestyle="--", linewidth=1)
    axis.axhline(-15, color="#C44E52", linestyle="--", linewidth=1)
    axis.set_ylabel("Aggregate bias (%)")
    axis.set_title("OSM model bias must improve across regions and MAJOR/MINOR roads")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(BIAS_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {BIAS_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_agreement(agreement: pd.DataFrame) -> None:
    table = agreement.pivot_table(
        index="road_type",
        columns="osm_highway_group",
        values="station_count",
        aggfunc="sum",
        fill_value=0,
    )
    shares = table.div(table.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(shares.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(0.5, shares.to_numpy().max()))
    axis.set_xticks(np.arange(len(shares.columns)))
    axis.set_xticklabels(shares.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(shares.index)))
    axis.set_yticklabels(shares.index)
    axis.set_title("OSM highway group versus ATC road type (diagnostic only)")
    axis.set_xlabel("OSM highway group")
    axis.set_ylabel("ATC station road type")
    figure.colorbar(image, ax=axis, label="Row share")
    figure.tight_layout()
    figure.savefig(AGREEMENT_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {AGREEMENT_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def update_report_manifest() -> None:
    rows = [
        (SOURCE_AUDIT_PATH, "reportable_source_audit", "records_overpass_date_query_tiles_and_osm_attribution"),
        (MATCH_AUDIT_PATH, "reportable_coverage_gate", "separates_full_network_station_and_minor_road_osm_support"),
        (CLASS_AGREEMENT_PATH, "reportable_taxonomy_diagnostic", "compares_osm_and_atc_class_without_using_atc_class_in_deployable_models"),
        (FEATURE_MANIFEST_PATH, "reportable_feature_lineage", "separates_primary_osm_secondary_osm_and_atc_oracle_predictors"),
        (METRICS_BY_FOLD_PATH, "reportable_spatial_validation", "reuses_the_frozen_five_spatial_folds"),
        (SUMMARY_PATH, "reportable_model_summary", "reports_primary_osm_skill_separately_from_secondary_tags_and_oracle"),
        (COMPARISON_PATH, "reportable_increment_test", "tests_osm_against_both_hierarchy_and_structure_plus_gtfs"),
        (SUBGROUP_PATH, "reportable_bias_audit", "includes_region_major_minor_and_osm_match_strata"),
        (DECISION_PATH, "reportable_decision", "authorises_historical_osm_only_if_the_complete_2023_gate_passes"),
        (COVERAGE_FIGURE_PATH, "reportable_coverage_figure", "shows_matching_support_before_model_results"),
        (MODEL_FIGURE_PATH, "reportable_model_figure", "contrasts_deployable_osm_with_frozen_baselines_and_non_deployable_oracle"),
        (BIAS_FIGURE_PATH, "reportable_bias_figure", "tests_whether_pooled_skill_hides_regional_or_road_network_bias"),
        (AGREEMENT_FIGURE_PATH, "reportable_taxonomy_figure", "diagnostic_crosswalk_not_a_success_gate"),
        (OSM_CROSSWALK_PATH, "analysis_input", "stores_one_external_osm_class_assignment_per_official_segment"),
        (NETWORK_FEATURE_PATH, "analysis_input", "adds_osm_features_to_every_step22_network_segment"),
        (STATION_FEATURE_PATH, "analysis_input", "joins_station_outcomes_after_full_network_osm_feature_generation"),
        (PREDICTION_PATH, "analysis_input", "stores_spatial_oof_predictions_for_all_predeclared_models"),
    ]
    existing = (
        pd.read_csv(REPORT_MANIFEST_PATH)
        if REPORT_MANIFEST_PATH.exists()
        else pd.DataFrame(columns=["artifact", "status", "reason"])
    )
    artifacts = {str(path.relative_to(PROJECT_ROOT)) for path, _, _ in rows}
    existing = existing[~existing["artifact"].isin(artifacts)]
    additions = pd.DataFrame(
        [
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "status": status,
                "reason": reason,
            }
            for path, status, reason in rows
        ]
    )
    save_csv(pd.concat([existing, additions], ignore_index=True), REPORT_MANIFEST_PATH)


def validate_inputs() -> None:
    for path in (
        STEP22_STATION_PATH,
        STEP22_NETWORK_PATH,
        STEP22_ROAD_MATCH_PATH,
        STEP22_DECISION_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing corrected Step 22 input: {path.relative_to(PROJECT_ROOT)}"
            )
    decisions = pd.read_csv(STEP22_DECISION_PATH)
    lineage = decisions.loc[
        decisions["decision"]
        == "deployable_predictor_lineage_excludes_atc_class_and_match_diagnostics",
        "pass",
    ]
    if lineage.empty or str(lineage.iloc[0]).lower() not in {"true", "1"}:
        raise RuntimeError("The corrected Step 22 deployable-lineage gate is not present or did not pass")


def main() -> None:
    validate_inputs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    network = pd.read_csv(STEP22_NETWORK_PATH)
    stations = pd.read_csv(STEP22_STATION_PATH)
    if network["road_2023_segment_index"].duplicated().any():
        raise ValueError("Step 22 full-network table has duplicate segment identifiers")

    geodatabase = find_road_geodatabase()
    centerline, road_geometries = read_centerline(geodatabase)
    if len(centerline) != len(network):
        raise ValueError(
            f"Road snapshot mismatch: geodatabase={len(centerline)}, Step22 network={len(network)}"
        )
    expected_ids = set(centerline["road_2023_segment_index"].astype(int))
    network_ids = set(network["road_2023_segment_index"].astype(int))
    if network_ids != expected_ids:
        raise ValueError("Step 22 network identifiers do not match the historical geodatabase")
    if len(stations) != 879 or set(stations["spatial_fold"].astype(int)) != set(FOLDS):
        raise ValueError("Step 23A requires the corrected Step 22 sample of 879 stations in five folds")

    print("Obtaining the historical 2023-06-30 OSM highway state...")
    raw_features, tile_audit = obtain_osm_tiles(centerline)
    ways, osm_geometries, source_totals = parse_osm_ways(raw_features)
    source_audit = pd.concat([tile_audit, source_totals], ignore_index=True, sort=False)
    save_csv(source_audit, SOURCE_AUDIT_PATH)

    print("Matching OSM ways to every official 2023 centreline segment...")
    crosswalk = match_centerline_to_osm(
        centerline,
        road_geometries,
        ways,
        osm_geometries,
    )
    network, stations, gtfs_features = build_feature_tables(
        network,
        stations,
        crosswalk,
    )
    coverage = coverage_audits(crosswalk, stations)
    agreement = class_agreement(stations)
    oracle_features = [
        "oracle_road_network_major",
        *sorted(column for column in stations if column.startswith("oracle_road_type_")),
    ]
    manifest = feature_manifest(network, gtfs_features, oracle_features)

    print("Running the frozen five-fold Step 23A comparison...")
    predictions, _ = run_spatial_experiment(
        stations,
        gtfs_features,
        oracle_features,
    )
    summary = summarise_models(predictions)
    comparisons = paired_comparisons(predictions)
    subgroups = subgroup_bias(predictions)
    decisions = decision_audit(
        coverage,
        summary,
        comparisons,
        subgroups,
        manifest,
    )

    plot_coverage(crosswalk)
    plot_models(summary)
    plot_bias(subgroups)
    plot_agreement(agreement)
    update_report_manifest()

    summary_lookup = summary.set_index("model")
    full_gate = bool(
        decisions.loc[
            decisions["decision"] == "step23a_2023_osm_full_network_gate",
            "pass",
        ].iloc[0]
    )
    print("\nStep 23A 2023 OSM road-class gate is complete.")
    for model_name in (
        "hierarchy_lookup",
        "deployable_structural_gtfs_hgb",
        "deployable_osm_highway_hgb",
        "deployable_osm_extended_hgb",
        "atc_class_oracle_hgb",
    ):
        row = summary_lookup.loc[model_name]
        print(
            f"  {MODEL_LABELS[model_name]}: MAE {row['mae']:,.0f}; "
            f"R2 {row['r2']:.3f}; aggregate bias {row['aggregate_bias_pct']:+.1f}%."
        )
    print(
        "  Decision: "
        + (
            "the complete 2023 OSM gate passes; Step 23B is authorised."
            if full_gate
            else "the complete 2023 OSM gate does not pass; do not start historical OSM modelling."
        )
    )
    print("  Step 23A does not establish multi-year segment backcasting.")


if __name__ == "__main__":
    main()
