from __future__ import annotations

import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SPATIAL_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "spatial"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CORE_SPATIAL_PATH = PROCESSED_DIR / "atc_core_spatial_model_panel.csv"
CENTERLINE_PATH = RAW_SPATIAL_DIR / "CENTERLINE.kmz"

MATCH_PATH = PROCESSED_DIR / "atc_core_station_centerline_match.csv"
CANDIDATE_PATH = PROCESSED_DIR / "atc_core_station_centerline_candidates.csv"
POINT_GEOJSON_PATH = PROCESSED_DIR / "atc_core_station_centerline_match_points.geojson"
SEGMENT_GEOJSON_PATH = PROCESSED_DIR / "atc_core_labelled_centerline_segments.geojson"
AUDIT_PATH = PROCESSED_DIR / "atc_core_station_centerline_audit.csv"

CENTERLINE_URL = "https://static.data.gov.hk/td/traffic-flow-census/CENTERLINE.kmz"
CENTERLINE_RESOURCE_PAGE = (
    "https://data.gov.hk/en-data/dataset/hk-td-tis_7-traffic-flow-census/"
    "resource/064619b4-d7ca-4847-bfe5-afca7b911599"
)

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
PLACEMARK_TAG = f"{{{KML_NAMESPACE}}}Placemark"
DESCRIPTION_TAG = f"{{{KML_NAMESPACE}}}description"
COORDINATES_TAG = f"{{{KML_NAMESPACE}}}coordinates"

SEARCH_RADIUS_METERS = 200.0
MAX_CANDIDATES = 8
LATITUDE_ORIGIN = 22.35
LONGITUDE_ORIGIN = 114.15
X_SCALE = 111_320.0 * math.cos(math.radians(LATITUDE_ORIGIN))
Y_SCALE = 110_540.0
NETWORK_SUPPORT_REFERENCE = "latest_official_current_centerline"
HISTORICAL_TOPOLOGY_STATUS = "not_proven"


def read_bool(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already available: {destination.relative_to(PROJECT_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {destination.name} (about 120 MB)")
    request = Request(url, headers={"User-Agent": "HK-AADT-research-pilot/1.0"})
    with urlopen(request, timeout=180) as response, destination.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Saved: {destination.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Complete Step 6 first."
        )
    with path.open(encoding="utf-8-sig", newline="") as source_file:
        return list(csv.DictReader(source_file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
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
    return "" if value.casefold() in {"<null>", "null"} else value


def description_fields(description: str) -> dict[str, str]:
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
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


def to_xy(longitude: float, latitude: float) -> tuple[float, float]:
    return (
        (longitude - LONGITUDE_ORIGIN) * X_SCALE,
        (latitude - LATITUDE_ORIGIN) * Y_SCALE,
    )


def to_lon_lat(x: float, y: float) -> tuple[float, float]:
    return (
        x / X_SCALE + LONGITUDE_ORIGIN,
        y / Y_SCALE + LATITUDE_ORIGIN,
    )


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - ax, py - ay), ax, ay
    position = ((px - ax) * dx + (py - ay) * dy) / denominator
    position = max(0.0, min(1.0, position))
    nearest_x = ax + position * dx
    nearest_y = ay + position * dy
    return math.hypot(px - nearest_x, py - nearest_y), nearest_x, nearest_y


def distance_to_parts(
    px: float,
    py: float,
    projected_parts: list[list[tuple[float, float]]],
) -> tuple[float, float, float]:
    best_distance = math.inf
    best_x = 0.0
    best_y = 0.0
    for part in projected_parts:
        for start, end in zip(part, part[1:]):
            distance, nearest_x, nearest_y = point_segment_distance(
                px,
                py,
                start[0],
                start[1],
                end[0],
                end[1],
            )
            if distance < best_distance:
                best_distance = distance
                best_x = nearest_x
                best_y = nearest_y
    return best_distance, best_x, best_y


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = normalized.replace("&", " and ")
    replacements = {
        r"\broad\b": "rd",
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bhighway\b": "hwy",
        r"\bjunction\b": "jct",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def text_similarity(left: str, right: str) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return 0.65 * sequence_score + 0.35 * token_score


def station_name_variants(segment_text: str) -> list[str]:
    road_phrase = segment_text.split("|")[0].strip()
    pieces = re.split(r"\s*(?:&|/|,|\band\b)\s*", road_phrase, flags=re.IGNORECASE)
    variants = [road_phrase]
    variants.extend(piece for piece in pieces if len(normalize_text(piece)) >= 4)
    return list(dict.fromkeys(variants))


def road_name_similarity(
    station_variants: list[str],
    street_name: str,
    alias_name: str,
) -> float:
    candidate_names = [name for name in (street_name, alias_name) if name and name != "-99"]
    if not candidate_names:
        return 0.0
    return max(
        text_similarity(station_name, candidate_name)
        for station_name in station_variants
        for candidate_name in candidate_names
    )


def candidate_score(distance_m: float, name_similarity: float) -> float:
    distance_component = max(0.0, 1.0 - distance_m / 60.0)
    return 0.55 * distance_component + 0.45 * name_similarity


def road_identity_key(fields: dict[str, str]) -> str:
    street_code = fields.get("ST_CODE", "")
    if street_code:
        return f"street_code:{street_code}"
    name = fields.get("STREET_ENAME", "") or fields.get("ALIAS_ENAME", "")
    return f"street_name:{normalize_text(name)}"


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


def build_station_records(core_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    stations: list[dict[str, object]] = []
    for row in core_rows:
        longitude = float(row["current_longitude"])
        latitude = float(row["current_latitude"])
        x, y = to_xy(longitude, latitude)
        stations.append(
            {
                "source": row,
                "station_id": row["station_id"],
                "longitude": longitude,
                "latitude": latitude,
                "x": x,
                "y": y,
                "name_variants": station_name_variants(row["segment_text_2021"]),
            }
        )
    return stations


def candidate_record(
    station: dict[str, object],
    fields: dict[str, str],
    distance_m: float,
    nearest_x: float,
    nearest_y: float,
) -> dict[str, object]:
    name_similarity = road_name_similarity(
        station["name_variants"],
        fields.get("STREET_ENAME", ""),
        fields.get("ALIAS_ENAME", ""),
    )
    nearest_longitude, nearest_latitude = to_lon_lat(nearest_x, nearest_y)
    return {
        "station_id": station["station_id"],
        "route_id": fields.get("ROUTE_ID", ""),
        "road_identity_key": road_identity_key(fields),
        "street_ename": fields.get("STREET_ENAME", ""),
        "street_cname": fields.get("STREET_CNAME", ""),
        "alias_ename": fields.get("ALIAS_ENAME", ""),
        "elevation": fields.get("ELEVATION", ""),
        "street_code": fields.get("ST_CODE", ""),
        "route_number": fields.get("ROUTE_NUM", ""),
        "travel_direction": fields.get("TRAVEL_DIRECTION", ""),
        "centerline_last_update": fields.get("LAST_UPD_DATE_V", ""),
        "distance_m": round(distance_m, 4),
        "road_name_similarity": round(name_similarity, 4),
        "selection_score": round(candidate_score(distance_m, name_similarity), 6),
        "nearest_longitude": round(nearest_longitude, 12),
        "nearest_latitude": round(nearest_latitude, 12),
    }


def retain_candidate(
    candidates: list[dict[str, object]],
    candidate: dict[str, object],
) -> None:
    candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            float(row["distance_m"]),
        )
    )
    del candidates[MAX_CANDIDATES:]


def find_candidates(
    stations: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], int]:
    candidates: dict[str, list[dict[str, object]]] = {
        str(station["station_id"]): [] for station in stations
    }
    feature_count = 0

    print("Streaming current official centerline and building candidates...")
    for placemark in stream_placemarks(CENTERLINE_PATH):
        feature_count += 1
        description = placemark.findtext(DESCRIPTION_TAG, default="")
        fields = description_fields(description)
        parts = placemark_parts(placemark)
        if not parts:
            continue
        projected_parts = [[to_xy(lon, lat) for lon, lat in part] for part in parts]
        min_x = min(x for part in projected_parts for x, _ in part)
        max_x = max(x for part in projected_parts for x, _ in part)
        min_y = min(y for part in projected_parts for _, y in part)
        max_y = max(y for part in projected_parts for _, y in part)

        for station in stations:
            station_x = float(station["x"])
            station_y = float(station["y"])
            if not (
                min_x - SEARCH_RADIUS_METERS <= station_x <= max_x + SEARCH_RADIUS_METERS
                and min_y - SEARCH_RADIUS_METERS <= station_y <= max_y + SEARCH_RADIUS_METERS
            ):
                continue
            distance_m, nearest_x, nearest_y = distance_to_parts(
                station_x,
                station_y,
                projected_parts,
            )
            if distance_m <= SEARCH_RADIUS_METERS:
                retain_candidate(
                    candidates[str(station["station_id"])],
                    candidate_record(station, fields, distance_m, nearest_x, nearest_y),
                )
    return candidates, feature_count


def classify_match(
    selected: dict[str, object],
    candidates: list[dict[str, object]],
) -> tuple[str, bool, bool, float, list[dict[str, object]]]:
    selected_score = float(selected["selection_score"])
    compatible = [
        candidate
        for candidate in candidates
        if candidate["road_identity_key"] == selected["road_identity_key"]
        and str(candidate["elevation"]) == str(selected["elevation"])
        and float(candidate["distance_m"]) <= 30.0
    ]
    cross_road_candidates = [
        candidate
        for candidate in candidates
        if candidate["road_identity_key"] != selected["road_identity_key"]
        or str(candidate["elevation"]) != str(selected["elevation"])
    ]
    second_score = (
        max(float(candidate["selection_score"]) for candidate in cross_road_candidates)
        if cross_road_candidates
        else 0.0
    )
    score_gap = selected_score - second_score
    ambiguous = bool(cross_road_candidates) and score_gap < 0.05
    distance_m = float(selected["distance_m"])
    name_similarity = float(selected["road_name_similarity"])

    if distance_m <= 15 and name_similarity >= 0.70 and not ambiguous:
        confidence = "high"
    elif distance_m <= 40 and name_similarity >= 0.45 and not ambiguous:
        confidence = "medium"
    else:
        confidence = "low"
    manual_review_required = confidence != "high"
    return confidence, manual_review_required, ambiguous, round(score_gap, 6), compatible


def build_match_outputs(
    stations: list[dict[str, object]],
    candidates: dict[str, list[dict[str, object]]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    matches: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    point_features: list[dict[str, object]] = []

    for station in stations:
        station_id = str(station["station_id"])
        source = station["source"]
        station_candidates = candidates[station_id]
        if not station_candidates:
            matches.append(
                {
                    "station_id": station_id,
                    "aadt_2011": source["aadt_2011"],
                    "aadt_2016": source["aadt_2016"],
                    "aadt_2021": source["aadt_2021"],
                    "station_segment_text": source["segment_text_2021"],
                    "station_longitude": station["longitude"],
                    "station_latitude": station["latitude"],
                    "selected_route_id": "",
                    "selected_street_ename": "",
                    "selected_alias_ename": "",
                    "selected_elevation": "",
                    "selected_travel_direction": "",
                    "distance_m": "",
                    "road_name_similarity": "",
                    "selection_score": "",
                    "score_gap_to_cross_road_candidate": "",
                    "candidate_count_retained": 0,
                    "compatible_route_count": 0,
                    "label_support_route_ids": "",
                    "cross_road_ambiguous": True,
                    "match_confidence": "unmatched",
                    "manual_review_required": True,
                    "nearest_longitude": "",
                    "nearest_latitude": "",
                    "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                    "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
                }
            )
            continue

        selected = station_candidates[0]
        confidence, manual_review, ambiguous, score_gap, compatible = classify_match(
            selected,
            station_candidates,
        )
        label_support_route_ids = ";".join(
            sorted(
                {str(candidate["route_id"]) for candidate in compatible},
                key=int,
            )
        )
        match = {
            "station_id": station_id,
            "aadt_2011": source["aadt_2011"],
            "aadt_2016": source["aadt_2016"],
            "aadt_2021": source["aadt_2021"],
            "station_segment_text": source["segment_text_2021"],
            "station_longitude": station["longitude"],
            "station_latitude": station["latitude"],
            "selected_route_id": selected["route_id"],
            "selected_street_ename": selected["street_ename"],
            "selected_alias_ename": selected["alias_ename"],
            "selected_elevation": selected["elevation"],
            "selected_travel_direction": selected["travel_direction"],
            "distance_m": selected["distance_m"],
            "road_name_similarity": selected["road_name_similarity"],
            "selection_score": selected["selection_score"],
            "score_gap_to_cross_road_candidate": score_gap,
            "candidate_count_retained": len(station_candidates),
            "compatible_route_count": len(compatible),
            "label_support_route_ids": label_support_route_ids,
            "cross_road_ambiguous": ambiguous,
            "match_confidence": confidence,
            "manual_review_required": manual_review,
            "nearest_longitude": selected["nearest_longitude"],
            "nearest_latitude": selected["nearest_latitude"],
            "network_support_reference": NETWORK_SUPPORT_REFERENCE,
            "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
        }
        matches.append(match)

        for rank, candidate in enumerate(station_candidates, start=1):
            candidate_rows.append(
                {
                    "station_id": station_id,
                    "candidate_rank": rank,
                    "selected": rank == 1,
                    "station_segment_text": source["segment_text_2021"],
                    **candidate,
                    "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                    "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
                }
            )

        point_features.append(
            {
                "type": "Feature",
                "properties": {
                    "station_id": station_id,
                    "selected_route_id": selected["route_id"],
                    "selected_street_ename": selected["street_ename"],
                    "distance_m": selected["distance_m"],
                    "road_name_similarity": selected["road_name_similarity"],
                    "match_confidence": confidence,
                    "manual_review_required": manual_review,
                    "compatible_route_count": len(compatible),
                    "label_support_route_ids": label_support_route_ids,
                    "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                    "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [station["longitude"], station["latitude"]],
                },
            }
        )
    return matches, candidate_rows, point_features


def collect_selected_segment_features(
    matches: list[dict[str, object]],
) -> list[dict[str, object]]:
    matches_by_route: dict[str, list[dict[str, object]]] = defaultdict(list)
    for match in matches:
        for route_id in str(match["label_support_route_ids"]).split(";"):
            if route_id:
                matches_by_route[route_id].append(match)

    features: list[dict[str, object]] = []
    print("Collecting geometries for selected road segments...")
    for placemark in stream_placemarks(CENTERLINE_PATH):
        description = placemark.findtext(DESCRIPTION_TAG, default="")
        fields = description_fields(description)
        route_id = fields.get("ROUTE_ID", "")
        if route_id not in matches_by_route:
            continue
        parts = placemark_parts(placemark)
        if not parts:
            continue
        geometry_type = "LineString" if len(parts) == 1 else "MultiLineString"
        geometry_coordinates: object = parts[0] if len(parts) == 1 else parts
        route_matches = matches_by_route[route_id]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "route_id": route_id,
                    "street_ename": fields.get("STREET_ENAME", ""),
                    "street_cname": fields.get("STREET_CNAME", ""),
                    "alias_ename": fields.get("ALIAS_ENAME", ""),
                    "elevation": fields.get("ELEVATION", ""),
                    "travel_direction": fields.get("TRAVEL_DIRECTION", ""),
                    "station_ids": ";".join(
                        sorted((str(match["station_id"]) for match in route_matches), key=int)
                    ),
                    "station_count": len(route_matches),
                    "representative_station_ids": ";".join(
                        sorted(
                            (
                                str(match["station_id"])
                                for match in route_matches
                                if str(match["selected_route_id"]) == route_id
                            ),
                            key=int,
                        )
                    ),
                    "all_links_high_confidence": all(
                        match["match_confidence"] == "high" for match in route_matches
                    ),
                    "network_support_reference": NETWORK_SUPPORT_REFERENCE,
                    "historical_topology_status": HISTORICAL_TOPOLOGY_STATUS,
                },
                "geometry": {
                    "type": geometry_type,
                    "coordinates": geometry_coordinates,
                },
            }
        )
    return features


def build_audit(
    matches: list[dict[str, object]],
    feature_count: int,
) -> list[dict[str, object]]:
    confidence_counts = Counter(str(match["match_confidence"]) for match in matches)
    matched = [match for match in matches if match["match_confidence"] != "unmatched"]
    distances = [float(match["distance_m"]) for match in matched]
    route_counts: Counter[str] = Counter()
    for match in matched:
        for route_id in str(match["label_support_route_ids"]).split(";"):
            if route_id:
                route_counts[route_id] += 1
    duplicate_route_links = sum(count > 1 for count in route_counts.values())
    audit = [
        {
            "metric": "centerline_feature_count",
            "count": feature_count,
            "value": "",
            "decision": "current_network_support_only",
        },
        {
            "metric": "core_spatial_station_count",
            "count": len(matches),
            "value": "",
            "decision": "input_station_labels",
        },
        {
            "metric": "matched_station_count",
            "count": len(matched),
            "value": "",
            "decision": "candidate_found_within_200m",
        },
        {
            "metric": "high_confidence_links",
            "count": confidence_counts["high"],
            "value": "",
            "decision": "eligible_for_first_network_baseline",
        },
        {
            "metric": "medium_confidence_links",
            "count": confidence_counts["medium"],
            "value": "",
            "decision": "sensitivity_or_manual_review",
        },
        {
            "metric": "low_confidence_links",
            "count": confidence_counts["low"],
            "value": "",
            "decision": "exclude_from_first_network_baseline",
        },
        {
            "metric": "unmatched_stations",
            "count": confidence_counts["unmatched"],
            "value": "",
            "decision": "manual_or_alternative_network_source",
        },
        {
            "metric": "median_selected_distance_m",
            "count": "",
            "value": round(statistics_median(distances), 4),
            "decision": "descriptive_not_accuracy_proof",
        },
        {
            "metric": "maximum_selected_distance_m",
            "count": "",
            "value": round(max(distances), 4),
            "decision": "inspect_large_distance_tail",
        },
        {
            "metric": "cross_road_ambiguous_links",
            "count": sum(bool(match["cross_road_ambiguous"]) for match in matched),
            "value": "",
            "decision": "different_road_or_elevation_manual_review_required",
        },
        {
            "metric": "support_routes_with_multiple_stations",
            "count": duplicate_route_links,
            "value": "",
            "decision": "do_not_average_labels_automatically",
        },
    ]
    return audit


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main() -> None:
    download_file(CENTERLINE_URL, CENTERLINE_PATH)
    core_rows = read_csv(CORE_SPATIAL_PATH)
    if len(core_rows) != 778:
        print(f"Note: current core spatial panel contains {len(core_rows)} stations.")

    stations = build_station_records(core_rows)
    candidates, feature_count = find_candidates(stations)
    matches, candidate_rows, point_features = build_match_outputs(stations, candidates)
    segment_features = collect_selected_segment_features(matches)
    audit_rows = build_audit(matches, feature_count)

    write_csv(MATCH_PATH, matches)
    write_csv(CANDIDATE_PATH, candidate_rows)
    write_geojson(
        POINT_GEOJSON_PATH,
        point_features,
        "atc_core_station_centerline_match_points",
    )
    write_geojson(
        SEGMENT_GEOJSON_PATH,
        segment_features,
        "atc_core_labelled_centerline_segments",
    )
    write_csv(AUDIT_PATH, audit_rows)

    confidence_counts = Counter(str(match["match_confidence"]) for match in matches)
    print("\nCurrent centerline linkage is ready.")
    print(f"Centerline features scanned: {feature_count}")
    print(
        "Station links: "
        f"{confidence_counts['high']} high, "
        f"{confidence_counts['medium']} medium, "
        f"{confidence_counts['low']} low, "
        f"{confidence_counts['unmatched']} unmatched."
    )
    print(
        "Primary rule: only high-confidence links enter the first road-network baseline; "
        "current centerline geometry does not prove historical topology."
    )


if __name__ == "__main__":
    main()
