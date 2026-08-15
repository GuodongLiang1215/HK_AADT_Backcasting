"""Step 18: test whether Step 17's conclusion is a stable-station selection artefact.

Step 17 deliberately used the small, stable set of detailed Core and Coverage
(B) workbooks.  This step expands the validation label set to every station-year
whose current AADT is unstarred in the official 2018--2024 Appendix B tables.

The PDF supplies the measured-versus-growth-factor flag.  Where available, the
newer structured GISDB record supplies the latest official AADT and road-network
metadata; the detailed Step 17 workbook remains the highest-priority value for
its own subset.  Appendix C is independently checked as a station-ID inventory.

The central comparison keeps Step 17's models, hyperparameters and spatial folds
fixed.  New stations are assigned to the nearest Step 17 fold centroid.  The
script therefore changes the validation sample, not the modelling recipe.

This remains a measured-station transportability test.  It does not validate
never-counted local roads, restore segment-level change maps, or estimate equity.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
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
import pdfplumber
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "reports"
ANNUAL_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "annual"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

STEP3_SCRIPT = PROJECT_ROOT / "src" / "03_extract_appendix_b.py"
STEP17_SCRIPT = (
    PROJECT_ROOT / "src" / "17_build_annual_panel_and_test_temporal_identification.py"
)
STEP17_PANEL_PATH = PROCESSED_DIR / "atc_step17_directly_measured_annual_panel.csv"
STEP17_CHANGE_METRIC_PATH = (
    TABLE_DIR / "step17_spatiotemporal_change_metrics.csv"
)
CURRENT_POINTS_PATH = PROCESSED_DIR / "atc_current_station_points.csv"
PACKAGE_PATH = ANNUAL_DIR / "ATC_TRAFFIC_DATA.zip"

ALL_PANEL_PATH = PROCESSED_DIR / "atc_step18_all_station_annual_panel.csv"
MEASURED_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
PAIR_PATH = PROCESSED_DIR / "atc_step18_consecutive_measured_pairs.csv"
KNOWN_PREDICTION_PATH = PROCESSED_DIR / "atc_step18_known_station_predictions.csv"
SPATIOTEMPORAL_PREDICTION_PATH = (
    PROCESSED_DIR / "atc_step18_spatiotemporal_predictions.csv"
)
CHANGE_PAIR_PATH = PROCESSED_DIR / "atc_step18_spatiotemporal_change_pairs.csv"

EXTRACTION_AUDIT_PATH = TABLE_DIR / "step18_extraction_audit.csv"
REPRESENTATIVENESS_PATH = TABLE_DIR / "step18_sampling_representativeness.csv"
RETENTION_PATH = TABLE_DIR / "step18_station_retention.csv"
KNOWN_METRIC_PATH = TABLE_DIR / "step18_known_station_forward_metrics.csv"
FORWARD_METRIC_PATH = TABLE_DIR / "step18_spatiotemporal_forward_metrics.csv"
CHANGE_METRIC_PATH = TABLE_DIR / "step18_change_metrics.csv"
STEP17_COMPARISON_PATH = TABLE_DIR / "step18_step17_comparison.csv"
DECISION_PATH = TABLE_DIR / "step18_decision_audit.csv"

COVERAGE_FIGURE_PATH = FIGURE_DIR / "step18_measured_sample_coverage.png"
REPRESENTATIVENESS_FIGURE_PATH = (
    FIGURE_DIR / "step18_aadt_distribution_by_sample.png"
)
RETENTION_FIGURE_PATH = FIGURE_DIR / "step18_station_retention.png"
CHANGE_FIGURE_PATH = FIGURE_DIR / "step18_change_identification_comparison.png"

YEARS = tuple(range(2018, 2025))
TARGET_YEARS = tuple(range(2019, 2025))
CHANGE_YEARS = tuple(range(2020, 2025))
NONPANDEMIC_CHANGE_YEARS = (2022, 2023, 2024)
RANDOM_SEED = 42

APPENDIX_B_BANDS = {
    "station_id": (50.0, 90.0),
    "station_type": (90.0, 115.0),
    "road_type": (115.0, 140.0),
    "road_name": (140.0, 210.0),
    "road_from": (210.0, 285.0),
    "road_to": (285.0, 360.0),
    "aadt_previous": (360.0, 410.0),
    "aadt_current": (410.0, 465.0),
    "reported_change": (465.0, 525.0),
}

ROAD_TYPE_LABELS = {
    "EX": "EXPRESSWAY",
    "UT": "URBAN TRUNK ROAD",
    "PD": "PRIMARY DISTRIBUTOR",
    "DD": "DISTRICT DISTRIBUTOR",
    "LD": "LOCAL DISTRIBUTOR",
    "RT": "RURAL TRUNK ROAD",
    "RR": "RURAL ROAD",
}

MODEL_COLORS = {
    "training_median": "#B0B7BC",
    "latest_year_road_hierarchy": "#2E86AB",
    "pooled_static_hgb": "#7F8C8D",
    "latest_year_static_hgb": "#D35400",
    "road_hierarchy_linear_trend": "#6C5CE7",
    "zero_change": "#1F1F1F",
}


@dataclass(frozen=True)
class ReportSpec:
    year: int
    filename: str
    url: str
    appendix_b_first_page: int
    appendix_b_last_page: int
    appendix_c_first_page: int
    appendix_c_last_page: int
    official_station_total: int
    official_surveyed_total: int
    appendix_station_rows: int
    appendix_measured_labels: int
    surveyed_hong_kong_island: int
    surveyed_kowloon: int
    surveyed_new_territories: int


REPORT_SPECS = (
    ReportSpec(
        2018,
        "ATC_2018.pdf",
        "https://www.td.gov.hk/filemanager/en/content_4953/annual%20traffic%20census%202018.pdf",
        50,
        110,
        111,
        171,
        1663,
        861,
        1662,
        861,
        219,
        308,
        334,
    ),
    ReportSpec(
        2019,
        "ATC_2019.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5018/annual%20traffic%20census%202019.pdf",
        50,
        106,
        107,
        163,
        1669,
        871,
        1667,
        871,
        229,
        309,
        333,
    ),
    ReportSpec(
        2020,
        "ATC_2020.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5114/annual%20traffic%20census%202020.pdf",
        50,
        106,
        107,
        163,
        1669,
        869,
        1666,
        868,
        220,
        316,
        333,
    ),
    ReportSpec(
        2021,
        "ATC_2021.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5167/Annual%20Traffic%20Census%202021.pdf",
        50,
        106,
        107,
        163,
        1678,
        873,
        1677,
        873,
        221,
        312,
        340,
    ),
    ReportSpec(
        2022,
        "ATC_2022.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5200/Annual%20Traffic%20Census%202022.pdf",
        50,
        106,
        107,
        163,
        1686,
        886,
        1686,
        886,
        230,
        305,
        351,
    ),
    ReportSpec(
        2023,
        "ATC_2023.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5267/Annual%20Traffic%20Census%202023.pdf",
        50,
        107,
        108,
        165,
        1691,
        880,
        1691,
        880,
        222,
        308,
        350,
    ),
    ReportSpec(
        2024,
        "ATC_2024.pdf",
        "https://www.td.gov.hk/filemanager/en/content_5287/Annual%20Traffic%20Census%202024.pdf",
        425,
        482,
        483,
        540,
        1694,
        892,
        1694,
        892,
        230,
        314,
        348,
    ),
)


def load_script_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required earlier-step script: {path}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty result: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"Already available: {path.relative_to(PROJECT_ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading: {path.name}")
    with urlopen(url, timeout=180) as response, temporary_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary_path.replace(path)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def ensure_official_sources(step17) -> None:
    for specification in REPORT_SPECS:
        download_file(specification.url, REPORT_DIR / specification.filename)
    step17.download_package()


def extract_appendix_c_station_ids(
    report_path: Path,
    first_page: int,
    last_page: int,
) -> set[int]:
    station_ids: list[int] = []
    with pdfplumber.open(report_path) as report:
        for page_number in range(first_page, last_page + 1):
            page = report.pages[page_number - 1]
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            for word in words:
                text = str(word["text"])
                x0 = float(word["x0"])
                top = float(word["top"])
                if (
                    285.0 <= x0 < 312.0
                    and top < page.height - 60.0
                    and text.isdigit()
                    and 4 <= len(text) <= 5
                ):
                    station_ids.append(int(text))
    if len(station_ids) != len(set(station_ids)):
        raise ValueError(f"Duplicate station IDs found in Appendix C of {report_path}")
    return set(station_ids)


def extract_pdf_panel(step3) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    rows: list[dict[str, object]] = []
    appendix_c_ids: dict[int, set[int]] = {}
    for report in REPORT_SPECS:
        year_specification = step3.YearSpec(
            year=report.year,
            first_page=report.appendix_b_first_page,
            last_page=report.appendix_b_last_page,
            report_name=report.filename,
            reported_station_total=report.official_station_total,
            reported_surveyed_total=report.official_surveyed_total,
            bands=APPENDIX_B_BANDS,
        )
        rows.extend(step3.extract_report(year_specification))
        appendix_c_ids[report.year] = extract_appendix_c_station_ids(
            REPORT_DIR / report.filename,
            report.appendix_c_first_page,
            report.appendix_c_last_page,
        )
    panel = pd.DataFrame(rows)
    panel["station_id"] = panel["station_id"].astype(int)
    panel["year"] = panel["year"].astype(int)
    if panel.duplicated(["year", "station_id"]).any():
        raise ValueError("Duplicate year-station rows in Appendix B extraction.")
    return panel, appendix_c_ids


def parse_number(value: object) -> float:
    text = str(value or "").replace(",", "").replace("*", "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_structured_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ZipFile(PACKAGE_PATH) as outer_archive:
        for year in range(2019, 2025):
            nested_name = f"ATC_TRAFFIC_DATA/{year}.zip"
            with ZipFile(BytesIO(outer_archive.read(nested_name))) as annual_archive:
                xml_names = [
                    name
                    for name in annual_archive.namelist()
                    if name.lower().endswith(".xml")
                ]
                if len(xml_names) != 1:
                    raise ValueError(f"Expected one GISDB XML file for {year}.")
                xml_bytes = annual_archive.read(xml_names[0]).lstrip(b"\xef\xbb\xbf")
                root = ET.fromstring(xml_bytes)
                for element in root.findall(".//PStnList"):
                    station_id = element.findtext("StationNo")
                    if station_id is None:
                        continue
                    rows.append(
                        {
                            "year": year,
                            "station_id": int(station_id),
                            "structured_region": str(
                                element.findtext("Regional") or ""
                            ).strip(),
                            "structured_station_type": str(
                                element.findtext("StationType") or ""
                            ).strip(),
                            "structured_road_type": str(
                                element.findtext("RoadType") or ""
                            ).strip(),
                            "structured_road_network": str(
                                element.findtext("RoadNetwork") or ""
                            ).strip().upper(),
                            "structured_aadt": parse_number(
                                element.findtext("CurAADT")
                            ),
                            "structured_source_member": xml_names[0],
                        }
                    )
    panel = pd.DataFrame(rows)
    if panel.duplicated(["year", "station_id"]).any():
        raise ValueError("Duplicate year-station rows in structured GISDB data.")
    return panel


def station_region_from_id(station_id: int) -> str:
    if station_id < 3000:
        return "Hong Kong Island"
    if station_id < 5000:
        return "Kowloon"
    return "New Territories"


def extend_step17_spatial_folds(
    panel: pd.DataFrame,
    step17_panel: pd.DataFrame,
) -> pd.DataFrame:
    seed = (
        step17_panel[
            ["station_id", "longitude", "latitude", "spatial_fold"]
        ]
        .drop_duplicates("station_id")
        .dropna(subset=["longitude", "latitude", "spatial_fold"])
        .copy()
    )
    centroids = seed.groupby("spatial_fold")[["longitude", "latitude"]].mean()
    seed_folds = seed.set_index("station_id")["spatial_fold"].to_dict()

    def assigned_fold(row: pd.Series) -> float:
        station_id = int(row["station_id"])
        if station_id in seed_folds:
            return float(seed_folds[station_id])
        if pd.isna(row["longitude"]) or pd.isna(row["latitude"]):
            return float("nan")
        distance = (
            (centroids["longitude"] - float(row["longitude"])) ** 2
            + (centroids["latitude"] - float(row["latitude"])) ** 2
        )
        return float(distance.idxmin())

    output = panel.copy()
    output["spatial_fold"] = output.apply(assigned_fold, axis=1).astype("Int64")
    return output


def build_analysis_panels(
    pdf_panel: pd.DataFrame,
    structured_panel: pd.DataFrame,
    step17_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structured_fields = [
        "year",
        "station_id",
        "structured_region",
        "structured_station_type",
        "structured_road_type",
        "structured_road_network",
        "structured_aadt",
        "structured_source_member",
    ]
    panel = pdf_panel.merge(
        structured_panel[structured_fields],
        on=["year", "station_id"],
        how="left",
        validate="one_to_one",
    )

    step17_fields = [
        "year",
        "station_id",
        "aadt",
        "station_class",
        "road_network",
        "road_type",
        "link_description",
        "longitude",
        "latitude",
        "geometry_reference",
        "historical_geometry_status",
        "spatial_fold",
    ]
    step17_values = step17_panel[step17_fields].rename(
        columns={
            "aadt": "step17_aadt",
            "station_class": "step17_station_class",
            "road_network": "step17_road_network",
            "road_type": "step17_road_type",
            "link_description": "step17_link_description",
            "longitude": "step17_longitude",
            "latitude": "step17_latitude",
            "geometry_reference": "step17_geometry_reference",
            "historical_geometry_status": "step17_historical_geometry_status",
            "spatial_fold": "step17_spatial_fold",
        }
    )
    panel = panel.merge(
        step17_values,
        on=["year", "station_id"],
        how="left",
        validate="one_to_one",
    )
    panel["in_step17_subset"] = panel["step17_aadt"].notna()

    points = pd.read_csv(CURRENT_POINTS_PATH)[
        [
            "station_id",
            "longitude",
            "latitude",
            "geometry_reference",
            "historical_geometry_status",
        ]
    ]
    panel = panel.merge(points, on="station_id", how="left", validate="many_to_one")
    for field in [
        "longitude",
        "latitude",
        "geometry_reference",
        "historical_geometry_status",
    ]:
        panel[field] = panel[f"step17_{field}"].combine_first(panel[field])

    metadata = structured_panel.sort_values(["station_id", "year"]).copy()
    metadata = metadata.groupby("station_id", as_index=False).first()
    metadata = metadata[
        ["station_id", "structured_road_network", "structured_road_type"]
    ].rename(
        columns={
            "structured_road_network": "frozen_structured_road_network",
            "structured_road_type": "frozen_structured_road_type",
        }
    )
    panel = panel.merge(metadata, on="station_id", how="left", validate="many_to_one")

    panel["region"] = panel["station_id"].map(station_region_from_id)
    panel["station_class"] = panel["station_type"].map(
        {"A": "core", "B": "coverage_b", "C": "coverage_c"}
    )
    panel["road_network"] = panel["frozen_structured_road_network"].replace(
        "", np.nan
    ).fillna("UNKNOWN")
    panel["road_type"] = panel["road_type"].map(ROAD_TYPE_LABELS).fillna("UNKNOWN")
    panel["link_description"] = panel["road_segment_text"]

    subset_mask = panel["in_step17_subset"]
    panel.loc[subset_mask, "station_class"] = panel.loc[
        subset_mask, "step17_station_class"
    ]
    panel.loc[subset_mask, "road_network"] = panel.loc[
        subset_mask, "step17_road_network"
    ]
    panel.loc[subset_mask, "road_type"] = panel.loc[
        subset_mask, "step17_road_type"
    ]
    panel.loc[subset_mask, "link_description"] = panel.loc[
        subset_mask, "step17_link_description"
    ]

    panel["aadt_pdf"] = pd.to_numeric(panel["aadt_current"], errors="coerce")
    panel["aadt"] = panel["step17_aadt"].combine_first(
        panel["structured_aadt"]
    ).combine_first(panel["aadt_pdf"])
    panel["aadt_source"] = np.select(
        [panel["step17_aadt"].notna(), panel["structured_aadt"].notna()],
        ["detailed_workbook", "structured_gisdb"],
        default="appendix_b_pdf",
    )
    panel["pdf_vs_structured_absolute_difference"] = (
        panel["aadt_pdf"] - panel["structured_aadt"]
    ).abs()
    panel["pdf_vs_step17_absolute_difference"] = (
        panel["aadt_pdf"] - panel["step17_aadt"]
    ).abs()

    panel = extend_step17_spatial_folds(panel, step17_panel)
    panel = panel.sort_values(["year", "station_id"]).reset_index(drop=True)
    measured = panel[
        panel["primary_label_eligible"].astype(bool) & panel["aadt"].notna()
    ].copy()
    return panel, measured


def build_extraction_audit(
    all_panel: pd.DataFrame,
    appendix_c_ids: dict[int, set[int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for specification in REPORT_SPECS:
        group = all_panel[all_panel["year"] == specification.year]
        measured = group[group["primary_label_eligible"].astype(bool)]
        appendix_b_ids = set(group["station_id"].astype(int))
        c_ids = appendix_c_ids[specification.year]
        region_counts = measured.groupby("region").size().to_dict()
        expected_regions = {
            "Hong Kong Island": specification.surveyed_hong_kong_island,
            "Kowloon": specification.surveyed_kowloon,
            "New Territories": specification.surveyed_new_territories,
        }
        rows.append(
            {
                "year": specification.year,
                "appendix_b_pages": (
                    f"{specification.appendix_b_first_page}-"
                    f"{specification.appendix_b_last_page}"
                ),
                "official_station_inventory": specification.official_station_total,
                "appendix_b_rows": len(group),
                "inventory_minus_appendix_rows": (
                    specification.official_station_total - len(group)
                ),
                "expected_appendix_rows": specification.appendix_station_rows,
                "appendix_c_unique_station_ids": len(c_ids),
                "appendix_b_ids_missing_from_c": len(appendix_b_ids - c_ids),
                "appendix_c_ids_missing_from_b": len(c_ids - appendix_b_ids),
                "duplicate_year_station_rows": int(
                    group.duplicated(["year", "station_id"]).sum()
                ),
                "official_surveyed_stations": specification.official_surveyed_total,
                "eligible_measured_labels": len(measured),
                "official_surveyed_minus_labels": (
                    specification.official_surveyed_total - len(measured)
                ),
                "expected_appendix_measured_labels": (
                    specification.appendix_measured_labels
                ),
                "measured_hong_kong_island": region_counts.get(
                    "Hong Kong Island", 0
                ),
                "measured_kowloon": region_counts.get("Kowloon", 0),
                "measured_new_territories": region_counts.get(
                    "New Territories", 0
                ),
                "regional_surveyed_count_gap": sum(
                    abs(region_counts.get(region, 0) - expected)
                    for region, expected in expected_regions.items()
                ),
                "structured_rows_available": int(group["structured_aadt"].notna().sum()),
                "pdf_structured_aadt_disagreements": int(
                    (group["pdf_vs_structured_absolute_difference"].fillna(0) > 0).sum()
                ),
                "step17_subset_rows": int(group["in_step17_subset"].sum()),
                "pdf_step17_aadt_disagreements": int(
                    (group["pdf_vs_step17_absolute_difference"].fillna(0) > 0).sum()
                ),
                "status": (
                    "pass"
                    if len(group) == specification.appendix_station_rows
                    and len(measured) == specification.appendix_measured_labels
                    and appendix_b_ids == c_ids
                    and not group.duplicated(["year", "station_id"]).any()
                    else "fail"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    if not (audit["status"] == "pass").all():
        failed = audit.loc[audit["status"] != "pass", "year"].tolist()
        raise ValueError(f"Appendix extraction audit failed for: {failed}")
    return audit


def cohort_frames(measured_panel: pd.DataFrame):
    return {
        "all_measured": measured_panel.copy(),
        "step17_subset": measured_panel[measured_panel["in_step17_subset"]].copy(),
    }


def build_representativeness(measured_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = ["region", "station_class", "road_network"]
    for year in YEARS:
        year_panel = measured_panel[measured_panel["year"] == year]
        cohorts = {
            "all_measured": year_panel,
            "step17_subset": year_panel[year_panel["in_step17_subset"]],
        }
        for cohort, frame in cohorts.items():
            for dimension in dimensions:
                for category, group in frame.groupby(dimension, dropna=False):
                    rows.append(
                        {
                            "year": year,
                            "cohort": cohort,
                            "dimension": dimension,
                            "category": str(category),
                            "n": len(group),
                            "share_of_cohort": round(len(group) / len(frame), 6),
                            "mean_aadt": round(float(group["aadt"].mean()), 4),
                            "median_aadt": round(float(group["aadt"].median()), 4),
                            "p10_aadt": round(float(group["aadt"].quantile(0.1)), 4),
                            "p90_aadt": round(float(group["aadt"].quantile(0.9)), 4),
                            "current_coordinate_coverage": round(
                                float(group["longitude"].notna().mean()), 6
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def standardised_log_aadt_difference(
    all_measured: pd.DataFrame,
    subset: pd.DataFrame,
) -> float:
    left = np.log1p(all_measured["aadt"].to_numpy(dtype=float))
    right = np.log1p(subset["aadt"].to_numpy(dtype=float))
    pooled_sd = math.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2)
    return float((np.mean(right) - np.mean(left)) / pooled_sd) if pooled_sd else 0.0


def build_step17_sample_comparison(measured_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        all_measured = measured_panel[measured_panel["year"] == year]
        subset = all_measured[all_measured["in_step17_subset"]]
        rows.append(
            {
                "year": year,
                "all_measured_n": len(all_measured),
                "step17_subset_n": len(subset),
                "step17_share_of_all_measured": round(len(subset) / len(all_measured), 6),
                "all_measured_median_aadt": round(float(all_measured["aadt"].median()), 4),
                "step17_subset_median_aadt": round(float(subset["aadt"].median()), 4),
                "all_measured_mean_aadt": round(float(all_measured["aadt"].mean()), 4),
                "step17_subset_mean_aadt": round(float(subset["aadt"].mean()), 4),
                "standardised_log_aadt_difference_subset_minus_all": round(
                    standardised_log_aadt_difference(all_measured, subset), 6
                ),
                "coverage_c_share_all_measured": round(
                    float((all_measured["station_class"] == "coverage_c").mean()), 6
                ),
                "coverage_c_share_step17_subset": round(
                    float((subset["station_class"] == "coverage_c").mean()), 6
                ),
                "minor_share_all_measured": round(
                    float((all_measured["road_network"] == "MINOR").mean()), 6
                ),
                "minor_share_step17_subset": round(
                    float((subset["road_network"] == "MINOR").mean()), 6
                ),
                "all_measured_current_coordinate_coverage": round(
                    float(all_measured["longitude"].notna().mean()), 6
                ),
                "step17_current_coordinate_coverage": round(
                    float(subset["longitude"].notna().mean()), 6
                ),
            }
        )
    return pd.DataFrame(rows)


def build_consecutive_pairs(measured_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: list[pd.DataFrame] = []
    retention_rows: list[dict[str, object]] = []
    for cohort, panel in cohort_frames(measured_panel).items():
        for target_year in TARGET_YEARS:
            previous = panel[panel["year"] == target_year - 1][
                ["station_id", "aadt"]
            ].rename(columns={"aadt": "aadt_previous"})
            current = panel[panel["year"] == target_year][
                [
                    "station_id",
                    "aadt",
                    "station_class",
                    "road_network",
                    "region",
                    "spatial_fold",
                ]
            ].rename(columns={"aadt": "aadt_current"})
            paired = previous.merge(current, on="station_id", validate="one_to_one")
            paired["cohort"] = cohort
            paired["previous_year"] = target_year - 1
            paired["target_year"] = target_year
            paired["observed_change"] = paired["aadt_current"] - paired["aadt_previous"]
            pairs.append(paired)
            retention_rows.append(
                {
                    "cohort": cohort,
                    "previous_year": target_year - 1,
                    "target_year": target_year,
                    "previous_year_station_count": int(
                        (panel["year"] == target_year - 1).sum()
                    ),
                    "target_year_station_count": int((panel["year"] == target_year).sum()),
                    "consecutive_measured_pairs": len(paired),
                    "target_year_retained_from_previous": round(
                        len(paired) / max(int((panel["year"] == target_year).sum()), 1),
                        6,
                    ),
                    "level_correlation": round(
                        float(paired[["aadt_previous", "aadt_current"]].corr().iloc[0, 1]),
                        6,
                    ),
                    "mean_absolute_observed_change": round(
                        float(paired["observed_change"].abs().mean()), 4
                    ),
                }
            )
    return pd.concat(pairs, ignore_index=True), pd.DataFrame(retention_rows)


def add_cohort(rows: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    return [{"cohort": cohort, **row} for row in rows]


def run_forward_validations(step17, measured_panel: pd.DataFrame):
    known_predictions: list[dict[str, object]] = []
    known_metrics: list[dict[str, object]] = []
    spatial_predictions: list[dict[str, object]] = []
    spatial_metrics: list[dict[str, object]] = []

    for cohort, panel in cohort_frames(measured_panel).items():
        known_prediction_rows, known_year_rows = step17.run_known_station_forward(panel)
        known_prediction_frame = pd.DataFrame(known_prediction_rows)
        known_pooled = step17.pooled_metric_rows(
            known_prediction_frame,
            "known_station_forward",
            step17.KNOWN_MODEL_ORDER,
        )
        known_predictions.extend(add_cohort(known_prediction_rows, cohort))
        known_metrics.extend(add_cohort(known_year_rows, cohort))
        known_metrics.extend(
            add_cohort(
                [
                    row
                    for row in known_pooled
                    if str(row["target_year"]) == "all_target_years"
                ],
                cohort,
            )
        )

        spatial_prediction_rows, spatial_fold_rows = step17.run_spatiotemporal_forward(
            panel
        )
        spatial_prediction_frame = pd.DataFrame(spatial_prediction_rows)
        spatial_pooled = step17.pooled_metric_rows(
            spatial_prediction_frame,
            "future_year_and_held_out_spatial_fold",
            step17.SPATIOTEMPORAL_MODEL_ORDER,
        )
        spatial_predictions.extend(add_cohort(spatial_prediction_rows, cohort))
        spatial_metrics.extend(add_cohort(spatial_fold_rows, cohort))
        spatial_metrics.extend(add_cohort(spatial_pooled, cohort))

    return (
        pd.DataFrame(known_predictions),
        pd.DataFrame(known_metrics),
        pd.DataFrame(spatial_predictions),
        pd.DataFrame(spatial_metrics),
    )


def build_change_pairs(
    step17,
    spatial_predictions: pd.DataFrame,
    measured_panel: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    metadata = measured_panel[
        [
            "year",
            "station_id",
            "station_class",
            "road_network",
            "region",
        ]
    ].rename(columns={"year": "change_year"})
    for cohort in spatial_predictions["cohort"].unique():
        cohort_predictions = spatial_predictions[
            spatial_predictions["cohort"] == cohort
        ]
        for change_year in CHANGE_YEARS:
            for model in step17.SPATIOTEMPORAL_MODEL_ORDER:
                previous = cohort_predictions[
                    (cohort_predictions["target_year"] == change_year - 1)
                    & (cohort_predictions["model"] == model)
                ][["station_id", "observed_aadt", "predicted_aadt"]]
                current = cohort_predictions[
                    (cohort_predictions["target_year"] == change_year)
                    & (cohort_predictions["model"] == model)
                ][["station_id", "observed_aadt", "predicted_aadt"]]
                paired = previous.merge(
                    current,
                    on="station_id",
                    suffixes=("_previous", "_current"),
                    validate="one_to_one",
                )
                paired["cohort"] = cohort
                paired["change_year"] = change_year
                paired["model"] = model
                paired["observed_change"] = (
                    paired["observed_aadt_current"] - paired["observed_aadt_previous"]
                )
                paired["predicted_change"] = (
                    paired["predicted_aadt_current"]
                    - paired["predicted_aadt_previous"]
                )
                frames.append(paired)
    output = pd.concat(frames, ignore_index=True)
    output = output.merge(
        metadata,
        on=["change_year", "station_id"],
        how="left",
        validate="many_to_one",
    )
    output["absolute_error"] = (
        output["predicted_change"] - output["observed_change"]
    ).abs()
    output["zero_change_absolute_error"] = output["observed_change"].abs()
    output["absolute_error_difference_vs_zero"] = (
        output["absolute_error"] - output["zero_change_absolute_error"]
    )
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def clustered_loss_interval(
    frame: pd.DataFrame,
    seed: int,
    iterations: int = 1000,
) -> tuple[float, float]:
    station_groups = [
        group["absolute_error_difference_vs_zero"].to_numpy(dtype=float)
        for _, group in frame.groupby("station_id")
    ]
    if len(station_groups) < 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = generator.integers(0, len(station_groups), size=len(station_groups))
        values = np.concatenate([station_groups[index] for index in sampled])
        estimates[iteration] = float(values.mean())
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def change_metric_row(
    frame: pd.DataFrame,
    cohort: str,
    scope: str,
    dimension: str,
    category: str,
    model: str,
    seed: int,
) -> dict[str, object]:
    observed = frame["observed_change"].to_numpy(dtype=float)
    predicted = (
        np.zeros(len(frame), dtype=float)
        if model == "zero_change"
        else frame["predicted_change"].to_numpy(dtype=float)
    )
    zero_mae = float(np.mean(np.abs(observed)))
    model_mae = mean_absolute_error(observed, predicted)
    nonzero = np.sign(observed) != 0
    interval_low = float("nan")
    interval_high = float("nan")
    if model != "zero_change" and dimension == "overall":
        interval_low, interval_high = clustered_loss_interval(frame, seed)
    return {
        "cohort": cohort,
        "transition_scope": scope,
        "stratum_dimension": dimension,
        "stratum": category,
        "model": model,
        "n": len(frame),
        "observed_mean_change": round(float(np.mean(observed)), 4),
        "predicted_mean_change": round(float(np.mean(predicted)), 4),
        "change_mae": round(float(model_mae), 4),
        "zero_change_mae": round(zero_mae, 4),
        "improvement_pct_vs_zero_change": round(
            (zero_mae - model_mae) / zero_mae * 100 if zero_mae else 0.0,
            4,
        ),
        "change_rmse": round(
            math.sqrt(mean_squared_error(observed, predicted)), 4
        ),
        "change_correlation": round(correlation(observed, predicted), 6),
        "direction_accuracy_nonzero_observed": round(
            float(np.mean(np.sign(observed[nonzero]) == np.sign(predicted[nonzero])))
            if nonzero.any()
            else float("nan"),
            6,
        ),
        "mean_absolute_error_difference_vs_zero": round(
            float(model_mae - zero_mae), 4
        ),
        "clustered_95pct_loss_difference_low": round(interval_low, 4),
        "clustered_95pct_loss_difference_high": round(interval_high, 4),
    }


def build_change_metrics(step17, change_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = ["region", "station_class", "road_network"]
    scopes: list[tuple[str, list[int]]] = [
        *[(str(year), [year]) for year in CHANGE_YEARS],
        ("all_change_years", list(CHANGE_YEARS)),
        ("nonpandemic_change_years", list(NONPANDEMIC_CHANGE_YEARS)),
    ]
    models = [*step17.SPATIOTEMPORAL_MODEL_ORDER, "zero_change"]
    for cohort in change_pairs["cohort"].unique():
        cohort_pairs = change_pairs[change_pairs["cohort"] == cohort]
        for scope_index, (scope, years) in enumerate(scopes):
            scope_pairs = cohort_pairs[cohort_pairs["change_year"].isin(years)]
            for model_index, model in enumerate(models):
                if model == "zero_change":
                    model_pairs = scope_pairs[
                        scope_pairs["model"] == step17.SPATIOTEMPORAL_MODEL_ORDER[0]
                    ]
                else:
                    model_pairs = scope_pairs[scope_pairs["model"] == model]
                rows.append(
                    change_metric_row(
                        model_pairs,
                        cohort,
                        scope,
                        "overall",
                        "all",
                        model,
                        RANDOM_SEED + scope_index * 20 + model_index,
                    )
                )
                for dimension in dimensions:
                    for category, group in model_pairs.groupby(dimension, dropna=False):
                        rows.append(
                            change_metric_row(
                                group,
                                cohort,
                                scope,
                                dimension,
                                str(category),
                                model,
                                RANDOM_SEED + scope_index * 20 + model_index,
                            )
                        )
    return pd.DataFrame(rows)


def compare_step17_reproduction(
    step17,
    change_metrics: pd.DataFrame,
) -> pd.DataFrame:
    original = pd.read_csv(STEP17_CHANGE_METRIC_PATH)
    original = original[
        original["change_year"].astype(str) == "all_change_years"
    ][["model", "n", "change_mae", "change_correlation"]].rename(
        columns={
            "n": "step17_original_n",
            "change_mae": "step17_original_change_mae",
            "change_correlation": "step17_original_change_correlation",
        }
    )
    rerun = change_metrics[
        (change_metrics["cohort"] == "step17_subset")
        & (change_metrics["transition_scope"] == "all_change_years")
        & (change_metrics["stratum_dimension"] == "overall")
    ][["model", "n", "change_mae", "change_correlation"]].rename(
        columns={
            "n": "step18_rerun_n",
            "change_mae": "step18_rerun_change_mae",
            "change_correlation": "step18_rerun_change_correlation",
        }
    )
    comparison = original.merge(rerun, on="model", validate="one_to_one")
    comparison["change_mae_absolute_difference"] = (
        comparison["step17_original_change_mae"]
        - comparison["step18_rerun_change_mae"]
    ).abs()
    comparison["correlation_absolute_difference"] = (
        comparison["step17_original_change_correlation"]
        - comparison["step18_rerun_change_correlation"]
    ).abs()
    correlation_matches = np.isclose(
        comparison["step17_original_change_correlation"],
        comparison["step18_rerun_change_correlation"],
        atol=0.00001,
        equal_nan=True,
    )
    comparison["reproduction_status"] = np.where(
        (comparison["step17_original_n"] == comparison["step18_rerun_n"])
        & (comparison["change_mae_absolute_difference"] < 0.1)
        & correlation_matches,
        "exact_within_saved_rounding",
        "different",
    )
    return comparison


def cohort_temporal_gate(
    step17,
    change_metrics: pd.DataFrame,
    cohort: str,
) -> dict[str, object]:
    overall = change_metrics[
        (change_metrics["cohort"] == cohort)
        & (change_metrics["transition_scope"] == "all_change_years")
        & (change_metrics["stratum_dimension"] == "overall")
        & (change_metrics["model"].isin(step17.SPATIOTEMPORAL_MODEL_ORDER))
    ].set_index("model")
    best_model = str(overall["change_mae"].idxmin())
    best = overall.loc[best_model]
    yearly = change_metrics[
        (change_metrics["cohort"] == cohort)
        & (change_metrics["stratum_dimension"] == "overall")
        & (change_metrics["model"] == best_model)
        & (change_metrics["transition_scope"].isin([str(year) for year in CHANGE_YEARS]))
    ]
    nonpandemic = change_metrics[
        (change_metrics["cohort"] == cohort)
        & (change_metrics["transition_scope"] == "nonpandemic_change_years")
        & (change_metrics["stratum_dimension"] == "overall")
        & (change_metrics["model"] == best_model)
    ].iloc[0]
    yearly_wins = int((yearly["improvement_pct_vs_zero_change"] > 0).sum())
    supported = bool(
        best["improvement_pct_vs_zero_change"] > 0
        and best["clustered_95pct_loss_difference_high"] < 0
        and best["change_correlation"] > 0
        and yearly_wins >= 3
        and nonpandemic["improvement_pct_vs_zero_change"] > 0
    )
    return {
        "cohort": cohort,
        "best_model": best_model,
        "change_mae": float(best["change_mae"]),
        "zero_change_mae": float(best["zero_change_mae"]),
        "improvement_pct": float(best["improvement_pct_vs_zero_change"]),
        "change_correlation": float(best["change_correlation"]),
        "clustered_interval_high": float(
            best["clustered_95pct_loss_difference_high"]
        ),
        "yearly_wins": yearly_wins,
        "nonpandemic_improvement_pct": float(
            nonpandemic["improvement_pct_vs_zero_change"]
        ),
        "supported": supported,
    }


def build_decision_audit(
    step17,
    extraction_audit: pd.DataFrame,
    sample_comparison: pd.DataFrame,
    change_metrics: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> pd.DataFrame:
    full_gate = cohort_temporal_gate(step17, change_metrics, "all_measured")
    subset_gate = cohort_temporal_gate(step17, change_metrics, "step17_subset")
    if full_gate["supported"] and not subset_gate["supported"]:
        conclusion = "step17_selection_bias_is_material"
    elif full_gate["supported"] and subset_gate["supported"]:
        conclusion = "temporal_signal_supported_in_both_samples"
    else:
        conclusion = "step17_negative_temporal_result_generalises"

    average_subset_share = float(
        sample_comparison["step17_share_of_all_measured"].mean()
    )
    rows = [
        {
            "question": "did_the_full_appendix_extraction_pass",
            "evidence": (
                f"All seven Appendix B inventories match Appendix C; "
                f"eligible labels range from {extraction_audit['eligible_measured_labels'].min()} "
                f"to {extraction_audit['eligible_measured_labels'].max()} per year. "
                f"The appendices supply {int(extraction_audit['official_surveyed_minus_labels'].sum())} "
                "fewer usable labels than the summed official surveyed totals."
            ),
            "decision": "yes",
        },
        {
            "question": "is_step17_a_representative_sample_of_measured_stations",
            "evidence": (
                f"Step 17 covers on average {average_subset_share * 100:.1f}% "
                "of all measured station-years and excludes rotating Coverage C by design."
            ),
            "decision": "no_selection_bias_requires_direct_sensitivity_test",
        },
        {
            "question": "was_the_step17_pipeline_reproduced_before_changing_the_sample",
            "evidence": (
                f"{int((reproduction['reproduction_status'] == 'exact_within_saved_rounding').sum())}/"
                f"{len(reproduction)} saved model results reproduce within saved rounding."
            ),
            "decision": (
                "yes"
                if (reproduction["reproduction_status"] == "exact_within_saved_rounding").all()
                else "no_review_required"
            ),
        },
        {
            "question": "does_the_all_measured_sample_support_unseen_location_annual_change",
            "evidence": (
                f"best={full_gate['best_model']}; MAE={full_gate['change_mae']:.1f}; "
                f"zero-change={full_gate['zero_change_mae']:.1f}; "
                f"improvement={full_gate['improvement_pct']:.2f}%; "
                f"correlation={full_gate['change_correlation']:.3f}; "
                f"yearly wins={full_gate['yearly_wins']}/5; "
                f"non-pandemic improvement={full_gate['nonpandemic_improvement_pct']:.2f}%."
            ),
            "decision": "yes" if full_gate["supported"] else "no",
        },
        {
            "question": "is_step17_failure_a_stable_station_selection_artefact",
            "evidence": (
                f"all-measured supported={full_gate['supported']}; "
                f"Step-17-subset supported={subset_gate['supported']}."
            ),
            "decision": conclusion,
        },
        {
            "question": "can_step18_validate_full_network_or_equity_trends",
            "evidence": (
                "The expanded labels are still ATC counting stations with current "
                "coordinates, not annual ground truth for never-counted local roads."
            ),
            "decision": "no_measured_station_transportability_gate_only",
        },
    ]
    return pd.DataFrame(rows)


def plot_coverage(measured_panel: pd.DataFrame) -> None:
    counts = (
        measured_panel.groupby(["year", "station_class"]).size().unstack(fill_value=0)
    )
    subset = measured_panel[measured_panel["in_step17_subset"]].groupby("year").size()
    order = ["core", "coverage_b", "coverage_c"]
    colors = ["#2E86AB", "#F39C12", "#7F8C8D"]
    figure, axis = plt.subplots(figsize=(10, 5.6))
    bottom = np.zeros(len(counts))
    for station_class, color in zip(order, colors):
        values = counts.get(station_class, pd.Series(0, index=counts.index)).to_numpy()
        axis.bar(counts.index, values, bottom=bottom, label=station_class, color=color)
        bottom += values
    axis.plot(
        subset.index,
        subset.values,
        marker="o",
        color="#C0392B",
        linewidth=2.2,
        label="Step 17 subset",
    )
    axis.set_title("Step 18 expands the measured label set beyond stable stations")
    axis.set_xlabel("Year")
    axis.set_ylabel("Measured stations")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=4)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(COVERAGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_representativeness(measured_panel: pd.DataFrame) -> None:
    yearly: list[dict[str, object]] = []
    for year in YEARS:
        frame = measured_panel[measured_panel["year"] == year]
        for cohort, group in {
            "All measured": frame,
            "Step 17 subset": frame[frame["in_step17_subset"]],
        }.items():
            yearly.append(
                {
                    "year": year,
                    "cohort": cohort,
                    "median": group["aadt"].median(),
                    "p10": group["aadt"].quantile(0.1),
                    "p90": group["aadt"].quantile(0.9),
                }
            )
    summary = pd.DataFrame(yearly)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for cohort, color, marker in [
        ("All measured", "#2E86AB", "o"),
        ("Step 17 subset", "#D35400", "s"),
    ]:
        group = summary[summary["cohort"] == cohort]
        axis.plot(
            group["year"],
            group["median"],
            marker=marker,
            color=color,
            linewidth=2,
            label=f"{cohort} median",
        )
        axis.fill_between(
            group["year"],
            group["p10"],
            group["p90"],
            color=color,
            alpha=0.12,
            label=f"{cohort} P10-P90",
        )
    axis.set_title("AADT support differs between the stable and full measured samples")
    axis.set_xlabel("Year")
    axis.set_ylabel("AADT (vehicles/day)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPRESENTATIVENESS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {REPRESENTATIVENESS_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_retention(retention: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.4))
    for cohort, label, color, marker in [
        ("all_measured", "All measured", "#2E86AB", "o"),
        ("step17_subset", "Step 17 subset", "#D35400", "s"),
    ]:
        group = retention[retention["cohort"] == cohort]
        axis.plot(
            group["target_year"],
            group["consecutive_measured_pairs"],
            marker=marker,
            color=color,
            linewidth=2,
            label=label,
        )
    axis.set_title("Consecutive measured pairs: more labels do not create a balanced panel")
    axis.set_xlabel("Change ending in year")
    axis.set_ylabel("Stations measured in both adjacent years")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(RETENTION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {RETENTION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_change_comparison(step17, change_metrics: pd.DataFrame) -> None:
    overall = change_metrics[
        (change_metrics["transition_scope"] == "all_change_years")
        & (change_metrics["stratum_dimension"] == "overall")
    ]
    models = [*step17.SPATIOTEMPORAL_MODEL_ORDER, "zero_change"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    width = 0.38
    positions = np.arange(len(models))
    for offset, cohort, label, alpha in [
        (-width / 2, "step17_subset", "Step 17 subset", 0.65),
        (width / 2, "all_measured", "All measured", 1.0),
    ]:
        cohort_frame = overall[overall["cohort"] == cohort].set_index("model")
        values = [cohort_frame.loc[model, "change_mae"] for model in models]
        axes[0].bar(
            positions + offset,
            values,
            width,
            label=label,
            color=[MODEL_COLORS[model] for model in models],
            alpha=alpha,
            edgecolor="white",
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [step17.MODEL_LABELS[model] for model in models],
        rotation=30,
        ha="right",
    )
    axes[0].set_title("Pooled annual-change MAE")
    axes[0].set_ylabel("Vehicles/day")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for cohort, label, color, marker in [
        ("all_measured", "All measured", "#2E86AB", "o"),
        ("step17_subset", "Step 17 subset", "#D35400", "s"),
    ]:
        candidate = overall[
            (overall["cohort"] == cohort)
            & overall["model"].isin(step17.SPATIOTEMPORAL_MODEL_ORDER)
        ]
        best_model = str(candidate.loc[candidate["change_mae"].idxmin(), "model"])
        yearly = change_metrics[
            (change_metrics["cohort"] == cohort)
            & (change_metrics["stratum_dimension"] == "overall")
            & (change_metrics["model"] == best_model)
            & change_metrics["transition_scope"].isin([str(year) for year in CHANGE_YEARS])
        ].sort_values("transition_scope")
        axes[1].plot(
            yearly["transition_scope"].astype(int),
            yearly["improvement_pct_vs_zero_change"],
            marker=marker,
            color=color,
            linewidth=2,
            label=f"{label}: {step17.MODEL_LABELS[best_model]}",
        )
    axes[1].axhline(0, color="#1F1F1F", linewidth=1)
    axes[1].axvspan(2019.7, 2021.3, color="#B0B7BC", alpha=0.18, label="Pandemic transitions")
    axes[1].set_title("Best-model improvement over no change")
    axes[1].set_xlabel("Change ending in year")
    axes[1].set_ylabel("MAE improvement (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Is Step 17's negative result a stable-station selection artefact?")
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHANGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {CHANGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    step3 = load_script_module("hk_aadt_step3", STEP3_SCRIPT)
    step17 = load_script_module("hk_aadt_step17", STEP17_SCRIPT)

    for required_path in [
        STEP17_PANEL_PATH,
        STEP17_CHANGE_METRIC_PATH,
        CURRENT_POINTS_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing {required_path.relative_to(PROJECT_ROOT)}. Complete Step 17 first."
            )

    ensure_official_sources(step17)
    pdf_panel, appendix_c_ids = extract_pdf_panel(step3)
    structured_panel = parse_structured_panel()
    step17_panel = pd.read_csv(STEP17_PANEL_PATH)
    all_panel, measured_panel = build_analysis_panels(
        pdf_panel,
        structured_panel,
        step17_panel,
    )

    extraction_audit = build_extraction_audit(all_panel, appendix_c_ids)
    representativeness = build_representativeness(measured_panel)
    sample_comparison = build_step17_sample_comparison(measured_panel)
    consecutive_pairs, retention = build_consecutive_pairs(measured_panel)

    save_csv(all_panel, ALL_PANEL_PATH)
    save_csv(measured_panel, MEASURED_PANEL_PATH)
    save_csv(consecutive_pairs, PAIR_PATH)
    save_csv(extraction_audit, EXTRACTION_AUDIT_PATH)
    save_csv(representativeness, REPRESENTATIVENESS_PATH)
    save_csv(retention, RETENTION_PATH)

    (
        known_predictions,
        known_metrics,
        spatial_predictions,
        spatial_metrics,
    ) = run_forward_validations(step17, measured_panel)
    change_pairs = build_change_pairs(step17, spatial_predictions, measured_panel)
    change_metrics = build_change_metrics(step17, change_pairs)
    reproduction = compare_step17_reproduction(step17, change_metrics)
    decision = build_decision_audit(
        step17,
        extraction_audit,
        sample_comparison,
        change_metrics,
        reproduction,
    )

    save_csv(known_predictions, KNOWN_PREDICTION_PATH)
    save_csv(spatial_predictions, SPATIOTEMPORAL_PREDICTION_PATH)
    save_csv(change_pairs, CHANGE_PAIR_PATH)
    save_csv(known_metrics, KNOWN_METRIC_PATH)
    save_csv(spatial_metrics, FORWARD_METRIC_PATH)
    save_csv(change_metrics, CHANGE_METRIC_PATH)
    sample_comparison_output = sample_comparison.copy()
    sample_comparison_output.insert(0, "record_type", "sample_composition")
    reproduction_output = reproduction.copy()
    reproduction_output.insert(0, "record_type", "step17_metric_reproduction")
    save_csv(
        pd.concat(
            [sample_comparison_output, reproduction_output],
            ignore_index=True,
            sort=False,
        ),
        STEP17_COMPARISON_PATH,
    )
    save_csv(decision, DECISION_PATH)

    plot_coverage(measured_panel)
    plot_representativeness(measured_panel)
    plot_retention(retention)
    plot_change_comparison(step17, change_metrics)

    full_gate = cohort_temporal_gate(step17, change_metrics, "all_measured")
    subset_gate = cohort_temporal_gate(step17, change_metrics, "step17_subset")
    print("\nStep 18 full measured-station transportability gate is complete.")
    print(
        f"  Measured labels: {measured_panel.groupby('year').size().min()}-"
        f"{measured_panel.groupby('year').size().max()} per year; Step 17 contributes "
        f"{measured_panel[measured_panel['in_step17_subset']].groupby('year').size().min()}-"
        f"{measured_panel[measured_panel['in_step17_subset']].groupby('year').size().max()}."
    )
    print(
        f"  All-measured best change model: {full_gate['best_model']}; "
        f"MAE {full_gate['change_mae']:,.0f} versus no-change "
        f"{full_gate['zero_change_mae']:,.0f}; correlation "
        f"{full_gate['change_correlation']:.3f}."
    )
    print(
        f"  Formal gate: all-measured={full_gate['supported']}; "
        f"Step-17-subset={subset_gate['supported']}."
    )
    print(f"  Decision: {decision.iloc[4]['decision']}")


if __name__ == "__main__":
    main()
