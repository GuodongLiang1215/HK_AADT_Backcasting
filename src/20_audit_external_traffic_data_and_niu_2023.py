"""Step 20: external-data selection and Niu et al. (2026) 2023 audit.

This is a data and estimand gate, not another backcasting model.  It asks:

1. Which recent traffic/AADT/exposure studies have data structures that are
   relevant to the Hong Kong pilot?
2. What exactly is contained in the public 2023 Hong Kong link-level product
   released with Niu et al. (2026)?
3. How well can those OSM links be crosswalked to the current Transport
   Department centreline and to directly measured 2023 ATC stations?
4. Which uses are scientifically defensible given that the Niu product and
   this project share 2023 ATC labels?

Agreement with 2023 ATC is explicitly labelled descriptive, not independent
validation.  The public release does not expose its training/test membership or
out-of-fold AADT predictions.  The release is also a 2023 product and therefore
does not identify 2011--2021 link-level change.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hk_aadt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from shapely.strtree import STRtree
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "niu_2023"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

NETWORK_PATH = PROCESSED_DIR / "atc_network_segment_features.csv"
MEASURED_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"

LITERATURE_PATH = TABLE_DIR / "step20_literature_data_comparison.csv"
CAPABILITY_PATH = TABLE_DIR / "step20_literature_capability_matrix.csv"
SOURCE_AUDIT_PATH = TABLE_DIR / "step20_niu_source_audit.csv"
DIRECTION_AUDIT_PATH = TABLE_DIR / "step20_niu_directionality_audit.csv"
NETWORK_SUMMARY_PATH = TABLE_DIR / "step20_niu_network_crosswalk_summary.csv"
STATION_METRIC_PATH = TABLE_DIR / "step20_niu_station_agreement_metrics.csv"
HISTORICAL_PATH = TABLE_DIR / "step20_historical_source_availability.csv"
DECISION_PATH = TABLE_DIR / "step20_decision_audit.csv"

NETWORK_CROSSWALK_PATH = PROCESSED_DIR / "atc_step20_niu_network_crosswalk.csv"
STATION_CROSSWALK_PATH = PROCESSED_DIR / "atc_step20_niu_station_crosswalk.csv"

CAPABILITY_FIGURE_PATH = FIGURE_DIR / "step20_literature_capability_matrix.png"
NETWORK_FIGURE_PATH = FIGURE_DIR / "step20_niu_network_crosswalk.png"
AGREEMENT_FIGURE_PATH = FIGURE_DIR / "step20_niu_station_agreement.png"

ZENODO_ROOT = "https://zenodo.org/records/19127840/files"
NIU_FILES = {
    "traffic_volume": "Traffic_Volume_by_Class_HongKong_2023.csv",
    "nox_emissions": "Traffic_NOx_Emissions_by_Class_HongKong_2023.csv",
    "pm25_emissions": "Traffic_PM2.5_Emissions_by_Class_HongKong_2023.csv",
}

LINK_KEYS = ["osmid", "u", "v"]
VEHICLE_CLASSES = ("MC", "PC", "TX", "LDB", "HDB", "LGV", "HGV", "FBDD")
EXPECTED_HOURS = tuple(f"{hour:02d}" for hour in range(7, 23))

PROJECTED_CRS = "EPSG:2326"
NETWORK_HIGH_DISTANCE_M = 10.0
NETWORK_MODERATE_DISTANCE_M = 40.0
NETWORK_MAX_DISTANCE_M = 100.0
STATION_HIGH_DISTANCE_M = 20.0
STATION_MODERATE_DISTANCE_M = 50.0
STATION_SEARCH_RADIUS_M = 100.0


LITERATURE_ROWS = (
    {
        "study_id": "ma_2026",
        "reference": "Ma et al. (2026), Journal of Transport Geography",
        "place": "England and Wales",
        "target_and_period": "link-level AADT and VKT; 2021",
        "traffic_labels": "19,560 public DfT major/minor-road count points",
        "other_inputs": "OS Open Roads; 905 demographic, employment, business, income, car-ownership and accessibility features; 144 spatial features",
        "validation": "sampling-intensity-weighted and spatial-block CV",
        "public_output": "2021 link estimates released on Zenodo",
        "main_transferable_lesson": "local-road labels, sampling-aware validation and spatial features matter more than adding a small static bundle",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1016/j.jtrangeo.2026.104715",
    },
    {
        "study_id": "bonnemaizon_2024",
        "reference": "Bonnemaizon et al. (2024), ERIS",
        "place": "Paris",
        "target_and_period": "hourly flow and occupancy; 2018-2022",
        "traffic_labels": "2,086 permanent magnetic road sensors",
        "other_inputs": "sensor histories, calendar variables and road attributes",
        "validation": "five-fold held-out sensor-record evaluation",
        "public_output": "6,846 road-segment hourly estimates",
        "main_transferable_lesson": "multi-year dynamics require repeated sensors; temporal density cannot be replaced by static land use",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1088/2634-4505/ad6bbf",
    },
    {
        "study_id": "bonnemaizon_2025",
        "reference": "Bonnemaizon et al. (2025), Scientific Data",
        "place": "36 European cities",
        "target_and_period": "harmonised measured AADT/AAWT; 2015-2024 depending on city",
        "traffic_labels": "about 35,000 local-authority sensors and 120,000 annual records",
        "other_inputs": "local open-data portals, OSM map matching and direction metadata",
        "validation": "comparison with independent floating-car data in 31 cities",
        "public_output": "harmonised annual station/segment files and processing code",
        "main_transferable_lesson": "direction, lane aggregation, metadata and road-reference matching must be audited before cross-year use",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1038/s41597-025-05698-y",
    },
    {
        "study_id": "zalzal_2022",
        "reference": "Zalzal and Hatzopoulou (2022), ES&T",
        "place": "Toronto",
        "target_and_period": "truck/LDV emissions and equity; 2006-2020",
        "traffic_labels": "annual turning-movement and automatic-recorder counts separated by vehicle type",
        "other_inputs": "roads, land use, three censuses, marginalisation indices and emission factors",
        "validation": "random train/test split; errors reported separately for arterial and local roads",
        "public_output": "supporting methods; some spatial inputs are commercial",
        "main_transferable_lesson": "annual vehicle-class counts enable trends, but local-road truck errors remain large",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1021/acs.est.2c04320",
    },
    {
        "study_id": "ganji_2024",
        "reference": "Ganji et al. (2024), Science of the Total Environment",
        "place": "Toronto",
        "target_and_period": "NO2 exposure backcast; 2006-2020",
        "traffic_labels": "historical AADT and on-road traffic-emission inventories",
        "other_inputs": "Urban Scanner mobile NO2, reference monitors and 2006/2016 travel surveys",
        "validation": "historical monitor trends and traffic-emission surfaces",
        "public_output": "article methods; no directly transferable Hong Kong product",
        "main_transferable_lesson": "backcasting is identified by historical traffic/emission and reference-station trends, not by year-specific static refits",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1016/j.scitotenv.2024.170075",
    },
    {
        "study_id": "wen_2024",
        "reference": "Wen et al. (2024), ES&T",
        "place": "Los Angeles",
        "target_and_period": "hourly air quality and environmental justice; 2019",
        "traffic_labels": "Caltrans PeMS hourly traffic used to simulate all-road truck/non-truck VKT",
        "other_inputs": "air monitors, ERA5, WorldPop, 10 m land cover, industrial POIs and demographics",
        "validation": "pollutant monitoring sites; dynamic-versus-static traffic comparison",
        "public_output": "supporting methods; derived full-network traffic is not a Hong Kong label source",
        "main_transferable_lesson": "real dynamic traffic can materially change pollution and equity estimates",
        "direct_use_in_hong_kong": False,
        "source_url": "https://doi.org/10.1021/acs.est.3c07545",
    },
    {
        "study_id": "niu_2026",
        "reference": "Niu et al. (2026), ES&T",
        "place": "Hong Kong",
        "target_and_period": "hourly vehicle-class traffic, NOx/PM2.5 emissions and equity; 2023",
        "traffic_labels": "ATC 2023, strategic/major-road detectors and traffic snapshot images",
        "other_inputs": "OSM, zoning, 2021 Census, public-transport routes/frequencies and EMFAC-HK",
        "validation": "hold-out model tests plus six-site NOx exposure-proxy comparison",
        "public_output": "three 2023 link-level CSV files on Zenodo",
        "main_transferable_lesson": "best direct Hong Kong cross-sectional benchmark and equity-estimand input; not a multi-year or independent ATC truth set",
        "direct_use_in_hong_kong": True,
        "source_url": "https://doi.org/10.1021/acs.est.5c14619",
    },
)


CAPABILITY_COLUMNS = (
    "hong_kong",
    "road_link_output",
    "local_road_labels",
    "hourly_dynamics",
    "multi_year_dynamics",
    "vehicle_classes",
    "exposure_equity",
    "public_derived_output",
)

CAPABILITY_ROWS = (
    ("Ma 2026", 0, 1, 1, 0, 0, 1, 0, 1),
    ("Bonnemaizon 2024", 0, 1, 0, 1, 1, 0, 0, 1),
    ("Bonnemaizon 2025", 0, 1, 0, 0, 1, 0, 0, 1),
    ("Zalzal 2022", 0, 1, 1, 0, 1, 1, 1, 0),
    ("Ganji 2024", 0, 1, 0, 0, 1, 0, 1, 0),
    ("Wen 2024", 0, 1, 0, 1, 0, 1, 1, 0),
    ("Niu 2026", 1, 1, 0, 1, 0, 1, 1, 1),
)


HISTORICAL_ROWS = (
    {
        "source": "ATC directly measured station AADT",
        "verified_years": "2011, 2016, 2018-2024",
        "spatial_support": "measured stations; predominantly classified network",
        "temporal_role": "primary annual traffic labels",
        "status": "verified_in_project",
        "limitation": "does not validate never-counted local roads",
    },
    {
        "source": "ATC official VKT and road length",
        "verified_years": "2011, 2016, 2021",
        "spatial_support": "territory/region and official major/minor aggregates",
        "temporal_role": "aggregate consistency constraint",
        "status": "verified_in_project",
        "limitation": "derived from ATC and cannot identify individual-link change",
    },
    {
        "source": "Niu public link traffic and emissions",
        "verified_years": "2023",
        "spatial_support": "OSM road links; eight vehicle classes; 07:00-23:00 activity window",
        "temporal_role": "recent-year cross-sectional and intraday benchmark",
        "status": "verified_by_step20",
        "limitation": "single year and shared ATC label lineage",
    },
    {
        "source": "Strategic/major-road detector volume and speed",
        "verified_years": "2023 use documented by Niu et al.",
        "spatial_support": "detector-supported strategic/major roads",
        "temporal_role": "hourly and potentially annual dynamic predictors",
        "status": "historical_archive_not_yet_verified",
        "limitation": "availability and consistent detector identities across target years unknown",
    },
    {
        "source": "Traffic snapshot images and vehicle classification",
        "verified_years": "2023 use documented by Niu et al.",
        "spatial_support": "traffic-camera locations",
        "temporal_role": "vehicle-class proportions",
        "status": "historical_archive_not_yet_verified",
        "limitation": "historically comparable image archive and camera stability unknown",
    },
    {
        "source": "Public-transport routes and service frequencies",
        "verified_years": "2023 use documented by Niu et al.",
        "spatial_support": "route-level bus service",
        "temporal_role": "bus-specific link activity",
        "status": "historical_archive_not_yet_verified",
        "limitation": "historical route/frequency snapshots for 2011/2016/2021 not established",
    },
    {
        "source": "Census population, employment, income and ethnicity",
        "verified_years": "2011, 2016, 2021",
        "spatial_support": "census small areas/LTPUG",
        "temporal_role": "population-weighted equity estimands",
        "status": "partly_verified_in_project",
        "limitation": "cross-year fine-area harmonisation and population-weighted crosswalk still required",
    },
    {
        "source": "Road-network topology and classifications",
        "verified_years": "latest official snapshot",
        "spatial_support": "current Transport Department centreline",
        "temporal_role": "spatial backbone",
        "status": "current_only",
        "limitation": "historical openings, closures and reclassification are not established",
    },
)


NAME_REPLACEMENTS = {
    "RD": "ROAD",
    "ST": "STREET",
    "AVE": "AVENUE",
    "HWY": "HIGHWAY",
    "EXPWY": "EXPRESSWAY",
    "CTR": "CENTRE",
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
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty result: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def download_file(filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Already available: {path.relative_to(PROJECT_ROOT)}")
        return path

    url = f"{ZENODO_ROOT}/{filename}?download=1"
    print(f"Downloading: {filename}")
    with urlopen(url, timeout=180) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")
    return path


def normalise_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = [NAME_REPLACEMENTS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def name_similarity(left: object, right: object) -> float:
    left_name = normalise_name(left)
    right_name = normalise_name(right)
    if not left_name or not right_name:
        return 0.0
    left_tokens = set(left_name.split()) - GENERIC_ROAD_TOKENS
    right_tokens = set(right_name.split()) - GENERIC_ROAD_TOKENS
    if not left_tokens or not right_tokens:
        return 0.0
    left_core = " ".join(token for token in left_name.split() if token in left_tokens)
    right_core = " ".join(token for token in right_name.split() if token in right_tokens)
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence_score = 0.5 * SequenceMatcher(None, left_core, right_core).ratio()
    containment = float(left_core in right_core or right_core in left_core)
    return float(max(token_score, sequence_score, containment))


def extract_schema(columns: list[str], product: str) -> tuple[set[str], set[str]]:
    classes: set[str] = set()
    hours: set[str] = set()
    if product == "traffic_volume":
        pattern = re.compile(r"^([A-Z]+)_Volume_(\d{2})$")
    elif product == "nox_emissions":
        pattern = re.compile(r"^NOX_([A-Z]+)_(\d{2})$")
    else:
        pattern = re.compile(r"^PM25_([A-Z]+)_(\d{2})$")
    for column in columns:
        match = pattern.match(column)
        if match:
            classes.add(match.group(1))
            hours.add(match.group(2))
    return classes, hours


def build_literature_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    literature = pd.DataFrame(LITERATURE_ROWS)
    save_csv(literature, LITERATURE_PATH)

    capability = pd.DataFrame(
        CAPABILITY_ROWS,
        columns=("study",) + CAPABILITY_COLUMNS,
    )
    save_csv(capability, CAPABILITY_PATH)

    values = capability[list(CAPABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = [column.replace("_", " ") for column in CAPABILITY_COLUMNS]
    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    axis.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(labels)), labels, rotation=32, ha="right")
    axis.set_yticks(range(len(capability)), capability["study"])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "yes" if values[row, column] else "no",
                ha="center",
                va="center",
                color="white" if values[row, column] else "#555555",
                fontsize=9,
                fontweight="bold" if values[row, column] else "normal",
            )
    axis.set_title("What recent studies actually have: data capability, not model complexity")
    axis.set_xlabel("Capability present in the study or released product")
    figure.tight_layout()
    CAPABILITY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CAPABILITY_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {CAPABILITY_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    return literature, capability


def load_niu_products() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    source_rows: list[dict[str, object]] = []
    key_frames: dict[str, pd.DataFrame] = {}
    traffic: pd.DataFrame | None = None

    for product, filename in NIU_FILES.items():
        path = download_file(filename)
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        classes, hours = extract_schema(columns, product)
        key_frame = pd.read_csv(path, usecols=LINK_KEYS)
        key_frames[product] = key_frame
        if product == "traffic_volume":
            traffic = pd.read_csv(path)
        source_rows.append(
            {
                "product": product,
                "filename": filename,
                "source_url": f"{ZENODO_ROOT}/{filename}",
                "file_size_mb": path.stat().st_size / (1024 * 1024),
                "row_count": len(key_frame),
                "column_count": len(columns),
                "unique_directed_link_keys": len(key_frame.drop_duplicates(LINK_KEYS)),
                "duplicate_directed_link_keys": int(key_frame.duplicated(LINK_KEYS).sum()),
                "vehicle_classes": "|".join(sorted(classes)),
                "hour_fields": "|".join(sorted(hours)),
                "has_expected_eight_classes": classes == set(VEHICLE_CLASSES),
                "has_expected_16_hours": hours == set(EXPECTED_HOURS),
                "year_scope": "2023_only",
                "label_lineage": "derived_product_using_atc_2023_among_inputs",
            }
        )

    if traffic is None:
        raise RuntimeError("Traffic-volume product was not loaded.")

    base_keys = pd.MultiIndex.from_frame(key_frames["traffic_volume"][LINK_KEYS])
    exact_key_alignment = True
    for product, keys in key_frames.items():
        product_keys = pd.MultiIndex.from_frame(keys[LINK_KEYS])
        equal = base_keys.equals(product_keys)
        exact_key_alignment &= equal
        for row in source_rows:
            if row["product"] == product:
                row["exact_row_order_and_key_alignment_with_traffic"] = equal
                break

    source_audit = pd.DataFrame(source_rows)
    save_csv(source_audit, SOURCE_AUDIT_PATH)
    return traffic, source_audit, exact_key_alignment


def build_directionality_audit(traffic: pd.DataFrame) -> pd.DataFrame:
    edges = traffic[LINK_KEYS + ["AADT", "length"]].copy()
    edges["u_text"] = edges["u"].astype(str)
    edges["v_text"] = edges["v"].astype(str)
    edges["node_low"] = edges[["u_text", "v_text"]].min(axis=1)
    edges["node_high"] = edges[["u_text", "v_text"]].max(axis=1)
    pair_columns = ["osmid", "node_low", "node_high"]
    pair_sizes = edges.groupby(pair_columns, dropna=False).size()
    reverse_pairs = pair_sizes[pair_sizes >= 2]
    edge_with_pair = edges.merge(
        pair_sizes.rename("directed_records_in_pair").reset_index(),
        on=pair_columns,
        how="left",
        validate="many_to_one",
    )

    paired = edge_with_pair[edge_with_pair["directed_records_in_pair"] >= 2]
    pair_aadt = (
        paired.groupby(pair_columns, dropna=False)["AADT"]
        .agg(["count", "min", "max", "mean"])
        .reset_index()
    )
    pair_aadt["relative_range"] = np.where(
        pair_aadt["mean"] > 0,
        (pair_aadt["max"] - pair_aadt["min"]) / pair_aadt["mean"],
        np.nan,
    )

    volume_columns = [
        column for column in traffic.columns if re.match(r"^[A-Z]+_Volume_\d{2}$", column)
    ]
    daytime_volume = traffic[volume_columns].sum(axis=1)
    aadt = pd.to_numeric(traffic["AADT"], errors="coerce")
    activity_ratio = daytime_volume / aadt.replace(0, np.nan)

    audit = pd.DataFrame(
        [
            {"metric": "directed_link_rows", "value": len(edges), "interpretation": "rows in public traffic product"},
            {"metric": "unique_osmid_u_v_keys", "value": len(edges.drop_duplicates(LINK_KEYS)), "interpretation": "directed link keys"},
            {"metric": "canonical_endpoint_pairs", "value": len(pair_sizes), "interpretation": "osmid plus unordered endpoints"},
            {"metric": "canonical_pairs_with_multiple_directed_records", "value": len(reverse_pairs), "interpretation": "potential reverse-direction representation"},
            {"metric": "share_of_rows_in_multiple_direction_pairs", "value": len(paired) / len(edges), "interpretation": "do not sum directed rows without a direction rule"},
            {"metric": "median_relative_aadt_range_within_reverse_pairs", "value": pair_aadt["relative_range"].median(), "interpretation": "zero indicates duplicated AADT across directions"},
            {"metric": "p95_relative_aadt_range_within_reverse_pairs", "value": pair_aadt["relative_range"].quantile(0.95), "interpretation": "directional AADT asymmetry diagnostic"},
            {"metric": "median_07_22_class_volume_sum_divided_by_aadt", "value": activity_ratio.median(), "interpretation": "the released hourly window is not automatically a full-day total"},
            {"metric": "p05_07_22_class_volume_sum_divided_by_aadt", "value": activity_ratio.quantile(0.05), "interpretation": "activity-window coverage diagnostic"},
            {"metric": "p95_07_22_class_volume_sum_divided_by_aadt", "value": activity_ratio.quantile(0.95), "interpretation": "activity-window coverage diagnostic"},
        ]
    )
    save_csv(audit, DIRECTION_AUDIT_PATH)
    return audit


def projected_niu_geometries(traffic: pd.DataFrame) -> tuple[list[object], STRtree]:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    geometries = [
        transform(transformer.transform, wkt.loads(value))
        for value in traffic["geometry"].astype(str)
    ]
    return geometries, STRtree(geometries)


def tree_indices(tree: STRtree, result: object, geometries: list[object]) -> list[int]:
    array = np.asarray(result)
    if array.size == 0:
        return []
    first = array.reshape(-1)[0]
    if isinstance(first, (int, np.integer)):
        return [int(value) for value in array.reshape(-1)]
    lookup = {id(geometry): index for index, geometry in enumerate(geometries)}
    return [lookup[id(value)] for value in array.reshape(-1)]


def nearest_index(tree: STRtree, point: Point, geometries: list[object]) -> int:
    result = tree.nearest(point)
    if isinstance(result, (int, np.integer)):
        return int(result)
    return {id(geometry): index for index, geometry in enumerate(geometries)}[id(result)]


def network_match_status(distance_m: float, similarity: float) -> str:
    if distance_m <= NETWORK_HIGH_DISTANCE_M or (
        distance_m <= 25.0 and similarity >= 0.55
    ):
        return "high"
    if distance_m <= NETWORK_MODERATE_DISTANCE_M or (
        distance_m <= 60.0 and similarity >= 0.55
    ):
        return "moderate"
    if distance_m <= NETWORK_MAX_DISTANCE_M:
        return "low"
    return "unmatched"


def build_network_crosswalk(
    traffic: pd.DataFrame,
    geometries: list[object],
    tree: STRtree,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    network = pd.read_csv(NETWORK_PATH)
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x_values, y_values = transformer.transform(
        network["centroid_longitude"].to_numpy(dtype=float),
        network["centroid_latitude"].to_numpy(dtype=float),
    )

    rows: list[dict[str, object]] = []
    for position, (_, segment) in enumerate(network.iterrows()):
        point = Point(float(x_values[position]), float(y_values[position]))
        index = nearest_index(tree, point, geometries)
        niu = traffic.iloc[index]
        distance = float(point.distance(geometries[index]))
        td_name = segment.get("street_ename", "")
        similarity = name_similarity(td_name, niu.get("name", ""))
        rows.append(
            {
                "route_id": segment["route_id"],
                "td_street_ename": td_name,
                "td_computed_length_m": segment["computed_length_m"],
                "td_centroid_longitude": segment["centroid_longitude"],
                "td_centroid_latitude": segment["centroid_latitude"],
                "niu_osmid": niu["osmid"],
                "niu_u": niu["u"],
                "niu_v": niu["v"],
                "niu_name": niu.get("name", np.nan),
                "niu_highway": niu.get("highway", np.nan),
                "niu_length_m": niu.get("length", np.nan),
                "niu_aadt_2023": niu.get("AADT", np.nan),
                "centroid_to_niu_link_distance_m": distance,
                "road_name_similarity": similarity,
                "crosswalk_status": network_match_status(distance, similarity),
                "geometry_caveat": "td_match_uses_segment_centroid_against_niu_full_linestring",
            }
        )

    crosswalk = pd.DataFrame(rows)
    save_csv(crosswalk, NETWORK_CROSSWALK_PATH)

    summary_rows: list[dict[str, object]] = []
    total_length = crosswalk["td_computed_length_m"].sum()
    for status in ("high", "moderate", "low", "unmatched"):
        subset = crosswalk[crosswalk["crosswalk_status"] == status]
        length = subset["td_computed_length_m"].sum()
        summary_rows.append(
            {
                "crosswalk_status": status,
                "td_segment_count": len(subset),
                "segment_share": len(subset) / len(crosswalk),
                "td_length_km": length / 1000.0,
                "length_share": length / total_length,
                "median_distance_m": subset["centroid_to_niu_link_distance_m"].median(),
                "median_name_similarity": subset["road_name_similarity"].median(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_csv(summary, NETWORK_SUMMARY_PATH)
    return crosswalk, summary


def best_station_candidate(
    point: Point,
    road_name: object,
    traffic: pd.DataFrame,
    geometries: list[object],
    tree: STRtree,
) -> tuple[int, float, float]:
    candidates = tree_indices(tree, tree.query(point.buffer(STATION_SEARCH_RADIUS_M)), geometries)
    if not candidates:
        candidates = [nearest_index(tree, point, geometries)]

    best: tuple[float, int, float, float] | None = None
    for index in candidates:
        distance = float(point.distance(geometries[index]))
        similarity = name_similarity(road_name, traffic.iloc[index].get("name", ""))
        ranking_score = distance + 30.0 * (1.0 - similarity)
        candidate = (ranking_score, index, distance, similarity)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def station_match_status(distance_m: float, similarity: float) -> str:
    if distance_m <= STATION_HIGH_DISTANCE_M and similarity >= 0.45:
        return "high"
    if distance_m <= 8.0:
        return "high"
    if distance_m <= STATION_MODERATE_DISTANCE_M and similarity >= 0.25:
        return "moderate"
    if distance_m <= STATION_SEARCH_RADIUS_M:
        return "low"
    return "unmatched"


def build_station_crosswalk(
    traffic: pd.DataFrame,
    geometries: list[object],
    tree: STRtree,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(MEASURED_PANEL_PATH)
    stations = panel[panel["year"].astype(int) == 2023].copy()
    stations = stations.drop_duplicates("station_id")
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)

    rows: list[dict[str, object]] = []
    for _, station in stations.iterrows():
        if pd.isna(station["longitude"]) or pd.isna(station["latitude"]):
            rows.append(
                {
                    "station_id": station["station_id"],
                    "region": station["region"],
                    "road_network": station["road_network"],
                    "station_road_type": station["road_type"],
                    "station_road_name": station["road_name"],
                    "station_longitude": station["longitude"],
                    "station_latitude": station["latitude"],
                    "observed_aadt_2023": float(station["aadt"]),
                    "niu_osmid": np.nan,
                    "niu_u": np.nan,
                    "niu_v": np.nan,
                    "niu_name": np.nan,
                    "niu_highway": np.nan,
                    "niu_aadt_2023": np.nan,
                    "station_to_niu_link_distance_m": np.nan,
                    "road_name_similarity": np.nan,
                    "crosswalk_status": "unmatched",
                    "aadt_error": np.nan,
                    "absolute_aadt_error": np.nan,
                    "comparison_status": "unmatched_missing_station_coordinates",
                }
            )
            continue
        x_value, y_value = transformer.transform(
            float(station["longitude"]),
            float(station["latitude"]),
        )
        point = Point(x_value, y_value)
        index, distance, similarity = best_station_candidate(
            point,
            station.get("road_name", ""),
            traffic,
            geometries,
            tree,
        )
        niu = traffic.iloc[index]
        observed = float(station["aadt"])
        predicted = float(niu["AADT"])
        rows.append(
            {
                "station_id": station["station_id"],
                "region": station["region"],
                "road_network": station["road_network"],
                "station_road_type": station["road_type"],
                "station_road_name": station["road_name"],
                "station_longitude": station["longitude"],
                "station_latitude": station["latitude"],
                "observed_aadt_2023": observed,
                "niu_osmid": niu["osmid"],
                "niu_u": niu["u"],
                "niu_v": niu["v"],
                "niu_name": niu.get("name", np.nan),
                "niu_highway": niu.get("highway", np.nan),
                "niu_aadt_2023": predicted,
                "station_to_niu_link_distance_m": distance,
                "road_name_similarity": similarity,
                "crosswalk_status": station_match_status(distance, similarity),
                "aadt_error": predicted - observed,
                "absolute_aadt_error": abs(predicted - observed),
                "comparison_status": "descriptive_not_independent_shared_atc_2023_lineage",
            }
        )

    crosswalk = pd.DataFrame(rows)
    save_csv(crosswalk, STATION_CROSSWALK_PATH)

    matched = crosswalk[crosswalk["crosswalk_status"].isin(["high", "moderate"])].copy()
    metric_rows: list[dict[str, object]] = []
    strata = [("all_matched", matched)]
    strata.extend(
        (f"crosswalk_status:{name}", group)
        for name, group in matched.groupby("crosswalk_status")
    )
    strata.extend((f"region:{name}", group) for name, group in matched.groupby("region"))
    strata.extend(
        (f"road_network:{name}", group)
        for name, group in matched.groupby("road_network")
    )
    for stratum, group in strata:
        observed = group["observed_aadt_2023"].to_numpy(dtype=float)
        predicted = group["niu_aadt_2023"].to_numpy(dtype=float)
        if len(group) >= 3:
            pearson = float(np.corrcoef(observed, predicted)[0, 1])
            spearman = float(spearmanr(observed, predicted).statistic)
        else:
            pearson = float("nan")
            spearman = float("nan")
        metric_rows.append(
            {
                "stratum": stratum,
                "station_count": len(group),
                "share_of_all_2023_measured_stations": len(group) / len(crosswalk),
                "mae": mean_absolute_error(observed, predicted),
                "rmse": math.sqrt(mean_squared_error(observed, predicted)),
                "aggregate_bias_pct": 100.0 * np.sum(predicted - observed) / np.sum(observed),
                "pearson": pearson,
                "spearman": spearman,
                "median_match_distance_m": group["station_to_niu_link_distance_m"].median(),
                "exact_aadt_match_share": np.mean(observed == predicted),
                "within_one_percent_aadt_match_share": np.mean(
                    np.abs(predicted - observed) / observed <= 0.01
                ),
                "interpretation": "descriptive_agreement_not_independent_validation",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    save_csv(metrics, STATION_METRIC_PATH)
    return crosswalk, metrics


def line_coordinates(geometry: object) -> list[np.ndarray]:
    if geometry.geom_type == "LineString":
        return [np.asarray(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [np.asarray(part.coords) for part in geometry.geoms]
    return []


def plot_network_crosswalk(
    geometries: list[object],
    network_summary: pd.DataFrame,
    station_crosswalk: pd.DataFrame,
) -> None:
    segments: list[np.ndarray] = []
    for geometry in geometries:
        segments.extend(line_coordinates(geometry))

    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.45, 1]})
    axes[0].add_collection(LineCollection(segments, colors="#C7CED3", linewidths=0.35, alpha=0.8))
    status_colors = {
        "high": "#1B9E77",
        "moderate": "#E6AB02",
        "low": "#D95F02",
        "unmatched": "#7570B3",
    }
    for status in ("high", "moderate", "low", "unmatched"):
        group = station_crosswalk[station_crosswalk["crosswalk_status"] == status]
        if group.empty:
            continue
        x_values, y_values = transformer.transform(
            group["station_longitude"].to_numpy(dtype=float),
            group["station_latitude"].to_numpy(dtype=float),
        )
        axes[0].scatter(
            x_values,
            y_values,
            s=10,
            color=status_colors[status],
            label=f"ATC station: {status}",
            alpha=0.8,
            linewidths=0,
        )
    axes[0].autoscale()
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    axes[0].set_title("Niu 2023 OSM links and ATC station crosswalk")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)

    ordered = network_summary.set_index("crosswalk_status").loc[
        ["high", "moderate", "low", "unmatched"]
    ]
    bars = axes[1].bar(
        ordered.index,
        100.0 * ordered["length_share"],
        color=[status_colors[value] for value in ordered.index],
    )
    axes[1].set_ylabel("Share of current TD centreline length (%)")
    axes[1].set_title("Centroid-to-link crosswalk coverage")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, 100.0 * ordered["length_share"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("Step 20 crosswalk audit: geographic compatibility is necessary, not validation")
    figure.tight_layout()
    NETWORK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(NETWORK_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {NETWORK_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_station_agreement(crosswalk: pd.DataFrame) -> None:
    matched = crosswalk[crosswalk["crosswalk_status"].isin(["high", "moderate"])].copy()
    matched = matched[(matched["observed_aadt_2023"] > 0) & (matched["niu_aadt_2023"] > 0)]
    region_colors = {
        "Hong Kong Island": "#4C78A8",
        "Kowloon": "#F58518",
        "New Territories": "#54A24B",
    }
    figure, axis = plt.subplots(figsize=(7.4, 6.5))
    for region, group in matched.groupby("region"):
        axis.scatter(
            group["observed_aadt_2023"],
            group["niu_aadt_2023"],
            s=18,
            alpha=0.55,
            color=region_colors.get(region, "#777777"),
            label=region,
            linewidths=0,
        )
    minimum = max(100.0, min(matched["observed_aadt_2023"].min(), matched["niu_aadt_2023"].min()))
    maximum = max(matched["observed_aadt_2023"].max(), matched["niu_aadt_2023"].max())
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#333333", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(minimum * 0.8, maximum * 1.2)
    axis.set_ylim(minimum * 0.8, maximum * 1.2)
    axis.set_xlabel("Directly measured ATC 2023 AADT")
    axis.set_ylabel("Niu public link AADT estimate")
    exact_share = np.mean(
        matched["observed_aadt_2023"].to_numpy(dtype=float)
        == matched["niu_aadt_2023"].to_numpy(dtype=float)
    )
    axis.set_title(
        "Descriptive only: shared ATC 2023 lineage; "
        f"{exact_share:.1%} of matched AADT values are exact"
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2, which="both")
    figure.tight_layout()
    AGREEMENT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(AGREEMENT_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {AGREEMENT_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def build_decision_outputs(
    source_audit: pd.DataFrame,
    exact_key_alignment: bool,
    network_summary: pd.DataFrame,
    station_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    historical = pd.DataFrame(HISTORICAL_ROWS)
    save_csv(historical, HISTORICAL_PATH)

    schema_complete = bool(
        source_audit["has_expected_eight_classes"].all()
        and source_audit["has_expected_16_hours"].all()
    )
    compatible_length_share = float(
        network_summary.loc[
            network_summary["crosswalk_status"].isin(["high", "moderate"]),
            "length_share",
        ].sum()
    )
    station_match_share = float(
        station_crosswalk["crosswalk_status"].isin(["high", "moderate"]).mean()
    )
    crosswalk_usable = compatible_length_share >= 0.70 and station_match_share >= 0.70

    decisions = pd.DataFrame(
        [
            {
                "decision": "niu_schema_and_cross_file_keys_are_consistent",
                "pass": schema_complete and exact_key_alignment,
                "evidence": f"schema_complete={schema_complete}; exact_key_alignment={exact_key_alignment}",
                "action": "retain all three products as one 2023 release only if true",
            },
            {
                "decision": "transport_department_to_niu_crosswalk_is_usable_for_aggregate_coverage_analysis",
                "pass": crosswalk_usable,
                "evidence": f"high_or_moderate_td_length_share={compatible_length_share:.3f}; high_or_moderate_station_match_share={station_match_share:.3f}",
                "action": "use the crosswalk for coverage and estimand work; do not claim historical geometry",
            },
            {
                "decision": "use_niu_2023_as_independent_aadt_validation",
                "pass": False,
                "evidence": "Niu et al. use ATC 2023 and public train/test or OOF membership is not supplied",
                "action": "label station comparison descriptive; request OOF predictions or fold membership for independent evaluation",
            },
            {
                "decision": "use_niu_2023_as_cross_sectional_hong_kong_benchmark",
                "pass": schema_complete and exact_key_alignment and crosswalk_usable,
                "evidence": "Hong Kong road-link traffic, eight classes and hourly fields are available with an auditable crosswalk",
                "action": "proceed to a richer 2023 benchmark while preserving label-lineage warnings",
            },
            {
                "decision": "use_niu_2023_for_population_weighted_near_road_equity_proof_of_concept",
                "pass": schema_complete and exact_key_alignment,
                "evidence": "vehicle-class traffic and NOx/PM2.5 link products share keys for 2023",
                "action": "proceed on finer geography with buffer/weighting sensitivity; report 2023 only",
            },
            {
                "decision": "use_niu_2023_to_infer_2011_2016_2021_link_change",
                "pass": False,
                "evidence": "the public release is 2023 only",
                "action": "do not back-scale links with territory VKT factors; verify historical dynamic inputs first",
            },
            {
                "decision": "static_features_alone_are_sufficient_for_temporal_backcasting",
                "pass": False,
                "evidence": "Steps 17-19 fail annual and five-year change gates; Step 20 finds no additional historical labels in the 2023 product",
                "action": "retain static features as spatial backbone only; require year-varying predictors and held-out temporal validation",
            },
        ]
    )
    save_csv(decisions, DECISION_PATH)
    return decisions


def validate_inputs() -> None:
    missing = [
        path.relative_to(PROJECT_ROOT)
        for path in (NETWORK_PATH, MEASURED_PANEL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Step 20 inputs: {missing}. Complete Steps 8 and 18 first.")


def main() -> None:
    validate_inputs()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    build_literature_outputs()
    traffic, source_audit, exact_key_alignment = load_niu_products()
    direction_audit = build_directionality_audit(traffic)
    geometries, tree = projected_niu_geometries(traffic)
    _, network_summary = build_network_crosswalk(traffic, geometries, tree)
    station_crosswalk, station_metrics = build_station_crosswalk(traffic, geometries, tree)
    plot_network_crosswalk(geometries, network_summary, station_crosswalk)
    plot_station_agreement(station_crosswalk)
    decisions = build_decision_outputs(
        source_audit,
        exact_key_alignment,
        network_summary,
        station_crosswalk,
    )

    overall_metric = station_metrics[station_metrics["stratum"] == "all_matched"].iloc[0]
    compatible_length_share = network_summary.loc[
        network_summary["crosswalk_status"].isin(["high", "moderate"]),
        "length_share",
    ].sum()
    station_match_share = station_crosswalk["crosswalk_status"].isin(["high", "moderate"]).mean()
    reverse_share = direction_audit.loc[
        direction_audit["metric"] == "share_of_rows_in_multiple_direction_pairs",
        "value",
    ].iloc[0]

    print("\nStep 20 external-data and Niu 2023 gate is complete.")
    print(f"  Niu directed links: {len(traffic):,}; reverse-pair row share: {reverse_share:.1%}.")
    print(f"  TD high/moderate crosswalk length share: {compatible_length_share:.1%}.")
    print(f"  2023 measured-station high/moderate match share: {station_match_share:.1%}.")
    print(
        "  Descriptive station agreement: "
        f"MAE {overall_metric['mae']:,.0f}; aggregate bias {overall_metric['aggregate_bias_pct']:+.1f}%; "
        f"Spearman {overall_metric['spearman']:.2f}."
    )
    print("  This is not independent validation because both products use ATC 2023.")
    passed = decisions.loc[
        decisions["decision"] == "use_niu_2023_as_cross_sectional_hong_kong_benchmark",
        "pass",
    ].iloc[0]
    print(
        "  Decision: "
        + ("proceed to a 2023 cross-sectional/equity benchmark" if passed else "repair the crosswalk before Step 21")
        + "; historical link change remains a separate gate."
    )


if __name__ == "__main__":
    main()
