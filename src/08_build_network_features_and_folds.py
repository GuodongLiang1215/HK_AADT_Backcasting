from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SPATIAL_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "spatial"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CENTERLINE_PATH = RAW_SPATIAL_DIR / "CENTERLINE.kmz"
MATCH_PATH = PROCESSED_DIR / "atc_core_station_centerline_match.csv"

NETWORK_FEATURE_PATH = PROCESSED_DIR / "atc_network_segment_features.csv"
TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
FOLD_PATH = PROCESSED_DIR / "atc_spatial_validation_folds.csv"
FOLD_GEOJSON_PATH = PROCESSED_DIR / "atc_spatial_validation_folds.geojson"
MANIFEST_PATH = PROCESSED_DIR / "atc_model_feature_manifest.csv"
AUDIT_PATH = PROCESSED_DIR / "atc_network_feature_audit.csv"

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
PLACEMARK_TAG = f"{{{KML_NAMESPACE}}}Placemark"
DESCRIPTION_TAG = f"{{{KML_NAMESPACE}}}description"
COORDINATES_TAG = f"{{{KML_NAMESPACE}}}coordinates"

LATITUDE_ORIGIN = 22.35
LONGITUDE_ORIGIN = 114.15
X_SCALE = 111_320.0 * math.cos(math.radians(LATITUDE_ORIGIN))
Y_SCALE = 110_540.0
N_SPATIAL_FOLDS = 5
NETWORK_SUPPORT_REFERENCE = "latest_official_current_centerline"
HISTORICAL_TOPOLOGY_STATUS = "not_proven"


def read_bool(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Complete Step 7 first."
        )
    with path.open(encoding="utf-8-sig", newline="") as source_file:
        return list(csv.DictReader(source_file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def write_geojson(path: Path, features: list[dict[str, object]], name: str) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": name,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def clean_cell(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = " ".join(html.unescape(value).split())
    return "" if value.casefold() in {"<null>", "null", "-99"} else value


def description_fields(description: str) -> dict[str, str]:
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
    fields: dict[str, str] = {}
    for row_html in row_pattern.findall(description):
        cells = cell_pattern.findall(row_html)
        if len(cells) >= 2:
            fields[clean_cell(cells[0]).upper()] = clean_cell(cells[1])
    return fields


def parse_coordinate_text(value: str) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for coordinate in value.split():
        parts = coordinate.split(",")
        if len(parts) >= 2:
            coordinates.append([float(parts[0]), float(parts[1])])
    return coordinates


def placemark_parts(placemark: ET.Element) -> list[list[list[float]]]:
    parts = [
        parse_coordinate_text(element.text or "")
        for element in placemark.iter(COORDINATES_TAG)
    ]
    return [part for part in parts if len(part) >= 2]


def kml_document_name(archive: ZipFile) -> str:
    names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
    if not names:
        raise ValueError(f"No KML document in {CENTERLINE_PATH.name}")
    return names[0]


def stream_placemarks(kmz_path: Path):
    with ZipFile(kmz_path) as archive:
        with archive.open(kml_document_name(archive)) as kml_file:
            for _, element in ET.iterparse(kml_file, events=("end",)):
                if element.tag == PLACEMARK_TAG:
                    yield element
                    element.clear()


def to_xy(longitude: float, latitude: float) -> tuple[float, float]:
    return (
        (longitude - LONGITUDE_ORIGIN) * X_SCALE,
        (latitude - LATITUDE_ORIGIN) * Y_SCALE,
    )


def endpoint_key(coordinate: list[float]) -> tuple[float, float]:
    return round(coordinate[0], 6), round(coordinate[1], 6)


def numeric_or_blank(value: object) -> float | str:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return ""


def geometry_summary(parts: list[list[list[float]]]) -> dict[str, object]:
    total_length = 0.0
    weighted_longitude = 0.0
    weighted_latitude = 0.0
    all_coordinates = [coordinate for part in parts for coordinate in part]

    for part in parts:
        for start, end in zip(part, part[1:]):
            start_x, start_y = to_xy(start[0], start[1])
            end_x, end_y = to_xy(end[0], end[1])
            segment_length = math.hypot(end_x - start_x, end_y - start_y)
            total_length += segment_length
            weighted_longitude += segment_length * (start[0] + end[0]) / 2
            weighted_latitude += segment_length * (start[1] + end[1]) / 2

    if total_length == 0:
        centroid_longitude = sum(row[0] for row in all_coordinates) / len(all_coordinates)
        centroid_latitude = sum(row[1] for row in all_coordinates) / len(all_coordinates)
    else:
        centroid_longitude = weighted_longitude / total_length
        centroid_latitude = weighted_latitude / total_length

    endpoint_keys = [
        endpoint_key(coordinate)
        for part in parts
        for coordinate in (part[0], part[-1])
    ]
    return {
        "computed_length_m": round(total_length, 4),
        "centroid_longitude": round(centroid_longitude, 12),
        "centroid_latitude": round(centroid_latitude, 12),
        "start_longitude": round(parts[0][0][0], 12),
        "start_latitude": round(parts[0][0][1], 12),
        "end_longitude": round(parts[-1][-1][0], 12),
        "end_latitude": round(parts[-1][-1][1], 12),
        "min_longitude": round(min(row[0] for row in all_coordinates), 12),
        "min_latitude": round(min(row[1] for row in all_coordinates), 12),
        "max_longitude": round(max(row[0] for row in all_coordinates), 12),
        "max_latitude": round(max(row[1] for row in all_coordinates), 12),
        "part_count": len(parts),
        "vertex_count": len(all_coordinates),
        "_endpoint_keys": endpoint_keys,
    }


def extract_network_features() -> list[dict[str, object]]:
    if not CENTERLINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CENTERLINE_PATH.relative_to(PROJECT_ROOT)}. Run Step 7 first."
        )

    rows: list[dict[str, object]] = []
    endpoint_counts: Counter[tuple[float, float]] = Counter()
    street_code_counts: Counter[str] = Counter()
    route_ids: set[str] = set()

    print("Streaming current official centerline and extracting model features...")
    for placemark in stream_placemarks(CENTERLINE_PATH):
        fields = description_fields(
            placemark.findtext(DESCRIPTION_TAG, default="")
        )
        parts = placemark_parts(placemark)
        if not parts:
            continue
        route_id = fields.get("ROUTE_ID", "")
        if not route_id:
            raise ValueError("A centerline feature has no ROUTE_ID.")
        if route_id in route_ids:
            raise ValueError(f"Duplicate ROUTE_ID in current centerline: {route_id}")
        route_ids.add(route_id)

        summary = geometry_summary(parts)
        endpoint_counts.update(summary["_endpoint_keys"])
        street_code = fields.get("ST_CODE", "")
        if street_code:
            street_code_counts[street_code] += 1

        route_number = fields.get("ROUTE_NUM", "")
        street_ename = fields.get("STREET_ENAME", "")
        alias_ename = fields.get("ALIAS_ENAME", "")
        rows.append(
            {
                "route_id": route_id,
                "street_code": street_code,
                "street_ename": street_ename,
                "street_cname": fields.get("STREET_CNAME", ""),
                "alias_ename": alias_ename,
                "elevation": fields.get("ELEVATION", ""),
                "travel_direction": fields.get("TRAVEL_DIRECTION", ""),
                "route_number": route_number,
                "route_number_present": bool(route_number),
                "named_street": bool(street_ename or alias_ename),
                "official_shape_length_m": numeric_or_blank(
                    fields.get("SHAPE_LENGTH", "")
                ),
                **summary,
                "centerline_last_update": fields.get("LAST_UPD_DATE_V", ""),
                "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
            }
        )

    for row in rows:
        endpoint_degrees = [
            endpoint_counts[key] for key in row.pop("_endpoint_keys")
        ]
        row["endpoint_degree_min"] = min(endpoint_degrees)
        row["endpoint_degree_max"] = max(endpoint_degrees)
        row["endpoint_degree_mean"] = round(
            sum(endpoint_degrees) / len(endpoint_degrees), 4
        )
        street_code = str(row["street_code"])
        row["street_code_segment_count"] = (
            street_code_counts[street_code] if street_code else 0
        )
    return rows


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def station_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**12, value)


def support_routes(row: dict[str, str]) -> list[str]:
    return [value for value in row["label_support_route_ids"].split(";") if value]


def build_spatial_folds(
    high_matches: list[dict[str, str]],
) -> list[dict[str, object]]:
    station_ids = [row["station_id"] for row in high_matches]
    union_find = UnionFind(station_ids)
    stations_by_support_route: dict[str, list[str]] = defaultdict(list)
    for row in high_matches:
        for route_id in support_routes(row):
            stations_by_support_route[route_id].append(row["station_id"])
    for linked_stations in stations_by_support_route.values():
        for station_id in linked_stations[1:]:
            union_find.union(linked_stations[0], station_id)

    components: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in high_matches:
        components[union_find.find(row["station_id"])].append(row)
    ordered_components = sorted(
        components.values(),
        key=lambda rows: station_sort_key(min(row["station_id"] for row in rows)),
    )
    if len(ordered_components) < N_SPATIAL_FOLDS:
        raise ValueError("Too few independent support groups for five spatial folds.")

    component_coordinates = np.array(
        [
            [
                np.mean([float(row["station_longitude"]) for row in rows]),
                np.mean([float(row["station_latitude"]) for row in rows]),
            ]
            for rows in ordered_components
        ]
    )
    component_weights = np.array([len(rows) for rows in ordered_components])
    clustering = KMeans(
        n_clusters=N_SPATIAL_FOLDS,
        random_state=42,
        n_init=20,
    )
    cluster_ids = clustering.fit_predict(
        component_coordinates,
        sample_weight=component_weights,
    )
    west_to_east = sorted(
        range(N_SPATIAL_FOLDS),
        key=lambda cluster_id: clustering.cluster_centers_[cluster_id][0],
    )
    fold_by_cluster = {
        cluster_id: fold_number
        for fold_number, cluster_id in enumerate(west_to_east, start=1)
    }

    fold_rows: list[dict[str, object]] = []
    for component_number, (rows, cluster_id) in enumerate(
        zip(ordered_components, cluster_ids),
        start=1,
    ):
        fold_number = fold_by_cluster[int(cluster_id)]
        group_id = f"G{component_number:04d}"
        for row in rows:
            fold_rows.append(
                {
                    "station_id": row["station_id"],
                    "spatial_fold": fold_number,
                    "support_group_id": group_id,
                    "support_group_station_count": len(rows),
                    "station_longitude": row["station_longitude"],
                    "station_latitude": row["station_latitude"],
                    "selected_route_id": row["selected_route_id"],
                    "label_support_route_ids": row["label_support_route_ids"],
                    "label_support_route_count": len(support_routes(row)),
                    "aadt_2011": row["aadt_2011"],
                    "aadt_2016": row["aadt_2016"],
                    "aadt_2021": row["aadt_2021"],
                    "fold_rule": "regional_kmeans_support_groups_kept_together",
                }
            )
    return sorted(fold_rows, key=lambda row: station_sort_key(str(row["station_id"])))


def build_training_table(
    high_matches: list[dict[str, str]],
    network_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    network_by_route = {str(row["route_id"]): row for row in network_rows}
    fold_by_station = {str(row["station_id"]): row for row in fold_rows}
    training_rows: list[dict[str, object]] = []

    for match in high_matches:
        station_id = match["station_id"]
        selected_route_id = match["selected_route_id"]
        if selected_route_id not in network_by_route:
            raise ValueError(
                f"Selected route {selected_route_id} for station {station_id} was not found."
            )
        network = network_by_route[selected_route_id]
        fold = fold_by_station[station_id]
        training_rows.append(
            {
                "station_id": station_id,
                "aadt_2011": match["aadt_2011"],
                "aadt_2016": match["aadt_2016"],
                "aadt_2021": match["aadt_2021"],
                "spatial_fold": fold["spatial_fold"],
                "support_group_id": fold["support_group_id"],
                "selected_route_id": selected_route_id,
                "label_support_route_ids": match["label_support_route_ids"],
                "label_support_route_count": len(support_routes(match)),
                "station_segment_text": match["station_segment_text"],
                "station_longitude": match["station_longitude"],
                "station_latitude": match["station_latitude"],
                "link_distance_m": match["distance_m"],
                "road_name_similarity": match["road_name_similarity"],
                "match_confidence": match["match_confidence"],
                "route_id": network["route_id"],
                "street_code": network["street_code"],
                "street_ename": network["street_ename"],
                "elevation": network["elevation"],
                "travel_direction": network["travel_direction"],
                "route_number_present": network["route_number_present"],
                "named_street": network["named_street"],
                "computed_length_m": network["computed_length_m"],
                "centroid_longitude": network["centroid_longitude"],
                "centroid_latitude": network["centroid_latitude"],
                "part_count": network["part_count"],
                "vertex_count": network["vertex_count"],
                "endpoint_degree_min": network["endpoint_degree_min"],
                "endpoint_degree_max": network["endpoint_degree_max"],
                "endpoint_degree_mean": network["endpoint_degree_mean"],
                "street_code_segment_count": network["street_code_segment_count"],
                "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
            }
        )
    return training_rows


def build_fold_geojson(fold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "type": "Feature",
            "properties": {
                "station_id": row["station_id"],
                "spatial_fold": row["spatial_fold"],
                "support_group_id": row["support_group_id"],
                "selected_route_id": row["selected_route_id"],
                "aadt_2011": row["aadt_2011"],
                "aadt_2016": row["aadt_2016"],
                "aadt_2021": row["aadt_2021"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["station_longitude"]),
                    float(row["station_latitude"]),
                ],
            },
        }
        for row in fold_rows
    ]


def build_manifest() -> list[dict[str, object]]:
    return [
        {"field": "aadt_2011", "role": "target", "type": "numeric", "initial_model": True, "decision": "fit_and_validate_separately_by_year"},
        {"field": "aadt_2016", "role": "target", "type": "numeric", "initial_model": True, "decision": "fit_and_validate_separately_by_year"},
        {"field": "aadt_2021", "role": "target", "type": "numeric", "initial_model": True, "decision": "calendar_year_target_do_not_assume_networkwide_pandemic_suppression_without_external_benchmark"},
        {"field": "centroid_longitude", "role": "predictor", "type": "numeric", "initial_model": True, "decision": "spatial_location_feature"},
        {"field": "centroid_latitude", "role": "predictor", "type": "numeric", "initial_model": True, "decision": "spatial_location_feature"},
        {"field": "computed_length_m", "role": "predictor", "type": "numeric", "initial_model": True, "decision": "segment_geometry_feature"},
        {"field": "elevation", "role": "predictor", "type": "categorical", "initial_model": True, "decision": "grade_separation_indicator"},
        {"field": "travel_direction", "role": "predictor", "type": "categorical", "initial_model": True, "decision": "current_network_attribute"},
        {"field": "route_number_present", "role": "predictor", "type": "binary", "initial_model": True, "decision": "road_hierarchy_proxy"},
        {"field": "named_street", "role": "audit", "type": "binary", "initial_model": False, "decision": "exclude_after_zero_variance_in_high_confidence_training_panel"},
        {"field": "endpoint_degree_mean", "role": "predictor", "type": "numeric", "initial_model": True, "decision": "local_network_connectivity_proxy"},
        {"field": "street_code_segment_count", "role": "predictor", "type": "numeric", "initial_model": True, "decision": "corridor_extent_proxy"},
        {"field": "street_code", "role": "audit", "type": "identifier", "initial_model": False, "decision": "exclude_to_reduce_high_cardinality_memorisation"},
        {"field": "street_ename", "role": "audit", "type": "text", "initial_model": False, "decision": "exclude_to_reduce_name_memorisation"},
        {"field": "selected_route_id", "role": "identifier", "type": "identifier", "initial_model": False, "decision": "never_use_as_predictor"},
        {"field": "spatial_fold", "role": "validation", "type": "integer", "initial_model": False, "decision": "held_out_region_not_predictor"},
        {"field": "support_group_id", "role": "validation", "type": "identifier", "initial_model": False, "decision": "prevents_shared_support_leakage"},
        {"field": "station_road_type", "role": "excluded", "type": "categorical", "initial_model": False, "decision": "not_available_for_all_prediction_segments"},
    ]


def build_audit(
    network_rows: list[dict[str, object]],
    high_matches: list[dict[str, str]],
    fold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    fold_counts = Counter(int(row["spatial_fold"]) for row in fold_rows)
    support_route_folds: dict[str, set[int]] = defaultdict(set)
    for row in fold_rows:
        for route_id in str(row["label_support_route_ids"]).split(";"):
            if route_id:
                support_route_folds[route_id].add(int(row["spatial_fold"]))
    split_support_routes = sum(
        len(folds) > 1 for folds in support_route_folds.values()
    )
    support_group_count = len({row["support_group_id"] for row in fold_rows})
    rows = [
        {"metric": "current_centerline_segment_count", "count": len(network_rows), "value": "", "decision": "prediction_support"},
        {"metric": "unique_current_route_id_count", "count": len({row["route_id"] for row in network_rows}), "value": "", "decision": "must_equal_segment_count"},
        {"metric": "high_confidence_training_station_count", "count": len(high_matches), "value": "", "decision": "primary_labels_only"},
        {"metric": "support_group_count", "count": support_group_count, "value": "", "decision": "group_before_spatial_clustering"},
        {"metric": "support_routes_split_across_folds", "count": split_support_routes, "value": "", "decision": "must_equal_zero"},
        {"metric": "spatial_fold_count", "count": len(fold_counts), "value": "", "decision": "five_regional_holdouts"},
    ]
    rows.extend(
        {
            "metric": f"spatial_fold_{fold_number}_station_count",
            "count": fold_counts[fold_number],
            "value": "",
            "decision": "report_fold_balance_not_random_split",
        }
        for fold_number in range(1, N_SPATIAL_FOLDS + 1)
    )
    return rows


def main() -> None:
    matches = read_csv(MATCH_PATH)
    high_matches = [row for row in matches if row["match_confidence"] == "high"]
    if not high_matches:
        raise ValueError("No high-confidence Step 7 links were found.")

    network_rows = extract_network_features()
    fold_rows = build_spatial_folds(high_matches)
    training_rows = build_training_table(high_matches, network_rows, fold_rows)
    audit_rows = build_audit(network_rows, high_matches, fold_rows)

    split_metric = next(
        row for row in audit_rows if row["metric"] == "support_routes_split_across_folds"
    )
    if int(split_metric["count"]) != 0:
        raise ValueError("A shared label-support route was split across validation folds.")

    write_csv(NETWORK_FEATURE_PATH, network_rows)
    write_csv(TRAINING_PATH, training_rows)
    write_csv(FOLD_PATH, fold_rows)
    write_geojson(
        FOLD_GEOJSON_PATH,
        build_fold_geojson(fold_rows),
        "atc_spatial_validation_folds",
    )
    write_csv(MANIFEST_PATH, build_manifest())
    write_csv(AUDIT_PATH, audit_rows)

    fold_counts = Counter(int(row["spatial_fold"]) for row in fold_rows)
    print("\nNetwork features and validation design are ready.")
    print(f"Current centerline segments: {len(network_rows)}")
    print(f"High-confidence training stations: {len(training_rows)}")
    print(
        "Spatial fold station counts: "
        + ", ".join(
            f"F{fold}={fold_counts[fold]}" for fold in range(1, N_SPATIAL_FOLDS + 1)
        )
    )
    print("Shared label-support routes split across folds: 0")
    print(
        "Next decision: compare a median baseline, spatial KNN, and one nonlinear "
        "tabular model using these held-out regional folds before adding complexity."
    )


if __name__ == "__main__":
    main()
