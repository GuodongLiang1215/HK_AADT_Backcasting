from __future__ import annotations

import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = PROJECT_ROOT / "config" / "official_sources.json"
RAW_SPATIAL_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "spatial"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATION_YEAR_PATH = PROCESSED_DIR / "atc_appendix_b_station_year.csv"
THREE_YEAR_PATH = PROCESSED_DIR / "atc_station_crosswalk_three_year.csv"
REVIEW_PATH = PROCESSED_DIR / "atc_station_crosswalk_review.csv"

POINT_CSV_PATH = PROCESSED_DIR / "atc_current_station_points.csv"
LINE_CSV_PATH = PROCESSED_DIR / "atc_current_station_lines.csv"
POINT_GEOJSON_PATH = PROCESSED_DIR / "atc_current_station_points.geojson"
LINE_GEOJSON_PATH = PROCESSED_DIR / "atc_current_station_lines.geojson"
PANEL_ANCHOR_PATH = PROCESSED_DIR / "atc_three_year_panel_current_spatial_anchor.csv"
REVIEW_ANCHOR_PATH = PROCESSED_DIR / "atc_crosswalk_review_current_spatial_anchor.csv"
AUDIT_PATH = PROCESSED_DIR / "atc_current_spatial_audit.csv"

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
KML = {"k": KML_NAMESPACE}
GEOMETRY_REFERENCE = "latest_official_snapshot_at_download"
HISTORICAL_GEOMETRY_STATUS = "not_proven"


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already available: {destination.relative_to(PROJECT_ROOT)}")
        return

    print(f"Downloading: {destination.name}")
    request = Request(url, headers={"User-Agent": "HK-AADT-research-pilot/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Saved: {destination.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")


def spatial_sources() -> dict[str, dict[str, str]]:
    with SOURCE_CONFIG.open(encoding="utf-8") as source_file:
        config = json.load(source_file)
    return {
        source["key"]: source
        for source in config["current_spatial_sources"]
    }


def download_current_spatial_sources() -> dict[str, Path]:
    RAW_SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    sources = spatial_sources()
    downloaded: dict[str, Path] = {}
    for key in ("station_points", "station_lines", "data_specification"):
        source = sources[key]
        destination = RAW_SPATIAL_DIR / source["filename"]
        download_file(source["url"], destination)
        downloaded[key] = destination
    return downloaded


def clean_cell(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(value).split())


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


def point_wkt(longitude: float, latitude: float) -> str:
    return f"POINT ({longitude:.12f} {latitude:.12f})"


def line_wkt(parts: list[list[list[float]]]) -> str:
    rendered_parts = [
        ", ".join(f"{longitude:.12f} {latitude:.12f}" for longitude, latitude in part)
        for part in parts
    ]
    if len(rendered_parts) == 1:
        return f"LINESTRING ({rendered_parts[0]})"
    return "MULTILINESTRING (" + ", ".join(
        f"({rendered})" for rendered in rendered_parts
    ) + ")"


def read_kml_root(kmz_path: Path) -> ET.Element:
    with ZipFile(kmz_path) as archive:
        kml_names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError(f"No KML document in {kmz_path.name}")
        with archive.open(kml_names[0]) as kml_file:
            return ET.parse(kml_file).getroot()


def parse_points(kmz_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    root = read_kml_root(kmz_path)
    rows: list[dict[str, object]] = []
    available_fields: set[str] = set()

    for placemark in root.findall(".//k:Placemark", KML):
        description = placemark.findtext("k:description", default="", namespaces=KML)
        fields = description_fields(description)
        available_fields.update(fields)
        station_id = fields.get("ATC_STATION_NO", "")
        coordinate_text = placemark.findtext(
            ".//k:Point/k:coordinates",
            default="",
            namespaces=KML,
        )
        coordinates = parse_coordinate_text(coordinate_text)
        if not station_id or len(coordinates) != 1:
            raise ValueError("A station point is missing its station number or coordinate")
        longitude, latitude = coordinates[0]
        rows.append(
            {
                "station_id": station_id,
                "feature_id": fields.get("FEATUREID", ""),
                "longitude": round(longitude, 12),
                "latitude": round(latitude, 12),
                "geometry_wkt": point_wkt(longitude, latitude),
                "geometry_reference": GEOMETRY_REFERENCE,
                "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
            }
        )

    station_ids = [str(row["station_id"]) for row in rows]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Current station point file contains duplicate station numbers")
    return rows, available_fields


def parse_lines(
    kmz_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    root = read_kml_root(kmz_path)
    rows: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    available_fields: set[str] = set()

    for placemark in root.findall(".//k:Placemark", KML):
        description = placemark.findtext("k:description", default="", namespaces=KML)
        fields = description_fields(description)
        available_fields.update(fields)
        station_id = fields.get("ATC_STATION_NO", "")
        parts = [
            parse_coordinate_text(node.text or "")
            for node in placemark.findall(".//k:LineString/k:coordinates", KML)
        ]
        parts = [part for part in parts if len(part) >= 2]
        if not station_id or not parts:
            raise ValueError("A station line is missing its station number or geometry")
        geometry_type = "LineString" if len(parts) == 1 else "MultiLineString"
        geometry_coordinates: object = parts[0] if len(parts) == 1 else parts
        row = {
            "station_id": station_id,
            "feature_id": fields.get("FEATUREID", ""),
            "direction": fields.get("DIRECTION", ""),
            "part_count": len(parts),
            "vertex_count": sum(len(part) for part in parts),
            "geometry_wkt": line_wkt(parts),
            "geometry_reference": GEOMETRY_REFERENCE,
            "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
        }
        rows.append(row)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "station_id": station_id,
                    "feature_id": fields.get("FEATUREID", ""),
                    "direction": fields.get("DIRECTION", ""),
                    "geometry_reference": GEOMETRY_REFERENCE,
                    "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
                },
                "geometry": {
                    "type": geometry_type,
                    "coordinates": geometry_coordinates,
                },
            }
        )
    return rows, features, available_fields


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Complete Step 4 first."
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


def write_geojson(path: Path, features: list[dict[str, object]]) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def build_point_features(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "type": "Feature",
            "properties": {
                "station_id": row["station_id"],
                "feature_id": row["feature_id"],
                "geometry_reference": GEOMETRY_REFERENCE,
                "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
        }
        for row in point_rows
    ]


def line_summary(line_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in line_rows:
        grouped[str(row["station_id"])].append(row)
    return {
        station_id: {
            "current_line_present": True,
            "current_line_feature_count": len(rows),
            "current_line_directions": ";".join(
                sorted({str(row["direction"]) for row in rows if row["direction"]})
            ),
        }
        for station_id, rows in grouped.items()
    }


def add_spatial_anchor(
    source_rows: list[dict[str, str]],
    point_rows: list[dict[str, object]],
    line_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    point_lookup = {str(row["station_id"]): row for row in point_rows}
    line_lookup = line_summary(line_rows)
    anchored: list[dict[str, object]] = []

    for source_row in source_rows:
        station_id = source_row["station_id"]
        point = point_lookup.get(station_id, {})
        line = line_lookup.get(station_id, {})
        anchored.append(
            {
                **source_row,
                "current_point_present": bool(point),
                "current_feature_id": point.get("feature_id", ""),
                "current_longitude": point.get("longitude", ""),
                "current_latitude": point.get("latitude", ""),
                "current_point_wkt": point.get("geometry_wkt", ""),
                "current_line_present": bool(line),
                "current_line_feature_count": line.get("current_line_feature_count", 0),
                "current_line_directions": line.get("current_line_directions", ""),
                "geometry_reference": GEOMETRY_REFERENCE,
                "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
                "allowed_use": "mapping_and_manual_review_anchor",
                "automatic_historical_match_confirmation": False,
            }
        )
    return anchored


def year_field_present(fields: set[str]) -> bool:
    return any("YEAR" in field for field in fields)


def build_audit(
    station_year_rows: list[dict[str, str]],
    three_year_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    point_rows: list[dict[str, object]],
    line_rows: list[dict[str, object]],
    point_fields: set[str],
    line_fields: set[str],
) -> list[dict[str, object]]:
    point_ids = {str(row["station_id"]) for row in point_rows}
    line_ids = {str(row["station_id"]) for row in line_rows}
    point_has_year = year_field_present(point_fields)
    line_has_year = year_field_present(line_fields)
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    audit_rows: list[dict[str, object]] = [
        {
            "scope": "current_snapshot",
            "station_count": "",
            "point_anchor_count": len(point_ids),
            "point_anchor_missing": "",
            "point_anchor_coverage": "",
            "line_anchor_count": len(line_ids),
            "line_anchor_coverage": "",
            "point_feature_count": len(point_rows),
            "line_feature_count": len(line_rows),
            "point_year_field_present": point_has_year,
            "line_year_field_present": line_has_year,
            "downloaded_at_utc": downloaded_at,
            "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
            "decision": "current_anchor_only_not_historical_ground_truth",
        }
    ]

    for year in (2011, 2016, 2021):
        historical_ids = {
            row["station_id"]
            for row in station_year_rows
            if int(row["year"]) == year
        }
        point_match = len(historical_ids & point_ids)
        line_match = len(historical_ids & line_ids)
        audit_rows.append(
            {
                "scope": str(year),
                "station_count": len(historical_ids),
                "point_anchor_count": point_match,
                "point_anchor_missing": len(historical_ids - point_ids),
                "point_anchor_coverage": round(point_match / len(historical_ids), 4),
                "line_anchor_count": line_match,
                "line_anchor_coverage": round(line_match / len(historical_ids), 4),
                "point_feature_count": "",
                "line_feature_count": "",
                "point_year_field_present": point_has_year,
                "line_year_field_present": line_has_year,
                "downloaded_at_utc": downloaded_at,
                "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
                "decision": "coverage_only_not_location_stability_proof",
            }
        )

    all_three_ids = {
        row["station_id"]
        for row in three_year_rows
        if row["all_three_present"].casefold() == "true"
    }
    review_ids = {row["station_id"] for row in review_rows}
    for scope, station_ids in (
        ("all_three_present", all_three_ids),
        ("crosswalk_review_unique", review_ids),
    ):
        point_match = len(station_ids & point_ids)
        line_match = len(station_ids & line_ids)
        audit_rows.append(
            {
                "scope": scope,
                "station_count": len(station_ids),
                "point_anchor_count": point_match,
                "point_anchor_missing": len(station_ids - point_ids),
                "point_anchor_coverage": round(point_match / len(station_ids), 4),
                "line_anchor_count": line_match,
                "line_anchor_coverage": round(line_match / len(station_ids), 4),
                "point_feature_count": "",
                "line_feature_count": "",
                "point_year_field_present": point_has_year,
                "line_year_field_present": line_has_year,
                "downloaded_at_utc": downloaded_at,
                "historical_geometry_status": HISTORICAL_GEOMETRY_STATUS,
                "decision": "manual_review_anchor_only",
            }
        )
    return audit_rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = download_current_spatial_sources()

    point_rows, point_fields = parse_points(downloaded["station_points"])
    line_rows, line_features, line_fields = parse_lines(downloaded["station_lines"])
    station_year_rows = read_csv(STATION_YEAR_PATH)
    three_year_rows = read_csv(THREE_YEAR_PATH)
    review_rows = read_csv(REVIEW_PATH)

    write_csv(POINT_CSV_PATH, point_rows)
    write_csv(LINE_CSV_PATH, line_rows)
    write_geojson(POINT_GEOJSON_PATH, build_point_features(point_rows))
    write_geojson(LINE_GEOJSON_PATH, line_features)
    write_csv(
        PANEL_ANCHOR_PATH,
        add_spatial_anchor(three_year_rows, point_rows, line_rows),
    )
    write_csv(
        REVIEW_ANCHOR_PATH,
        add_spatial_anchor(review_rows, point_rows, line_rows),
    )
    write_csv(
        AUDIT_PATH,
        build_audit(
            station_year_rows,
            three_year_rows,
            review_rows,
            point_rows,
            line_rows,
            point_fields,
            line_fields,
        ),
    )

    print("\nCurrent official spatial anchor is ready.")
    print(f"Point features: {len(point_rows)}")
    print(f"Line features: {len(line_rows)}")
    print(
        "Decision rule: use these files for mapping and manual review only; "
        "do not treat them as proof of 2011/2016/2021 station location stability."
    )


if __name__ == "__main__":
    main()
