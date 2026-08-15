from __future__ import annotations

import csv
import gzip
import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import truststore
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CENSUS_DIR = PROJECT_ROOT / "data" / "raw" / "census"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SOURCE_INVENTORY_PATH = TABLE_DIR / "step12_census_source_inventory.csv"
PAIR_SUMMARY_PATH = TABLE_DIR / "step12_boundary_pair_summary.csv"
MATCH_REVIEW_PATH = TABLE_DIR / "step12_boundary_match_review.csv"
VARIABLE_AUDIT_PATH = TABLE_DIR / "step12_variable_comparability.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step12_census_decision_audit.csv"
PANEL_PATH = PROCESSED_DIR / "census_ltpug_standardised_panel.csv"
CROSSWALK_PATH = PROCESSED_DIR / "census_ltpug_area_crosswalk.csv"
REFERENCE_BOUNDARY_PATH = (
    PROCESSED_DIR / "census_ltpug_2016_reference_boundaries.geojson.gz"
)
BOUNDARY_MAP_PATH = FIGURE_DIR / "step12_ltpug_boundary_stability.png"

REFERENCE_YEAR = 2016
STABLE_OVERLAP_THRESHOLD = 0.95
NEAR_STABLE_OVERLAP_THRESHOLD = 0.90
MIN_CROSSWALK_SHARE = 0.001


@dataclass(frozen=True)
class CensusSource:
    year: int
    title: str
    dataset_id: str
    expected_units: int
    data_dictionary_url: str
    data_dictionary_filename: str
    official_dataset_page: str

    @property
    def feature_service_url(self) -> str:
        return (
            "https://portal.csdi.gov.hk/server/rest/services/common/"
            f"{self.dataset_id}/FeatureServer/0"
        )

    @property
    def query_url(self) -> str:
        parameters = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        return f"{self.feature_service_url}/query?{urlencode(parameters)}"

    @property
    def raw_geojson_path(self) -> Path:
        return RAW_CENSUS_DIR / f"LTPUG_{self.year}.geojson"

    @property
    def dictionary_path(self) -> Path:
        return RAW_CENSUS_DIR / self.data_dictionary_filename


SOURCES = {
    2011: CensusSource(
        year=2011,
        title=(
            "2011 Population Census (Statistics and Boundaries of Large "
            "Tertiary Planning Unit Groups)"
        ),
        dataset_id="censtatd_rcd_1629267205229_857",
        expected_units=154,
        data_dictionary_url=(
            "https://www.censtatd.gov.hk/datagovhk/"
            "2011C_classification_idds_en.pdf"
        ),
        data_dictionary_filename="2011C_classification_idds_en.pdf",
        official_dataset_page=(
            "https://data.gov.hk/en-data/dataset/"
            "hk-censtatd-census_geo-2011-population-census-by-ltpu"
        ),
    ),
    2016: CensusSource(
        year=2016,
        title=(
            "2016 Population By-census (Statistics and Boundaries of Large "
            "Tertiary Planning Unit Groups)"
        ),
        dataset_id="censtatd_rcd_1629267205224_87566",
        expected_units=154,
        data_dictionary_url=(
            "https://www.censtatd.gov.hk/datagovhk/"
            "2016BC_classification_idds_en.pdf"
        ),
        data_dictionary_filename="2016BC_classification_idds_en.pdf",
        official_dataset_page=(
            "https://data.gov.hk/en-data/dataset/"
            "hk-censtatd-census_geo-2016-population-bycensus-by-ltpu"
        ),
    ),
    2021: CensusSource(
        year=2021,
        title=(
            "2021 Population Census (Statistics and Boundaries of Large "
            "Tertiary Planning Unit Groups)"
        ),
        dataset_id="censtatd_rcd_1635933102538_82889",
        expected_units=159,
        data_dictionary_url=(
            "https://www.census2021.gov.hk/doc/"
            "2021-statistics-classification-idds-en.pdf"
        ),
        data_dictionary_filename="2021_statistics_classification_idds_en.pdf",
        official_dataset_page=(
            "https://data.gov.hk/en-data/dataset/"
            "hk-censtatd-census_geo-2021-population-census-by-ltpu"
        ),
    ),
}


STANDARD_FIELDS = {
    "t_pop": "total_population",
    "dh": "domestic_households",
    "adhz": "average_domestic_household_size",
    "ma_hh": "median_monthly_domestic_household_income_hkd_nominal",
    "lfpr_t": "labour_force_participation_rate_pct",
    "t_wp": "working_population",
    "t_mmearn": "median_monthly_income_from_main_employment_hkd_nominal",
    "dmr_ir": "median_rent_to_income_ratio_pct",
}


VARIABLE_AUDIT_ROWS = [
    {
        "standard_variable": "total_population",
        "source_field": "t_pop",
        "unit": "persons",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "population denominator and neighbourhood context",
        "decision": "retain",
        "limitation": "Do not interpret changes before applying the boundary rule.",
    },
    {
        "standard_variable": "domestic_households",
        "source_field": "dh",
        "unit": "households",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "household denominator",
        "decision": "retain",
        "limitation": "Small values may be suppressed in published small-area data.",
    },
    {
        "standard_variable": "average_domestic_household_size",
        "source_field": "adhz",
        "unit": "persons_per_household",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "descriptive control",
        "decision": "retain_secondary",
        "limitation": "Area averages do not describe within-area household variation.",
    },
    {
        "standard_variable": "median_monthly_domestic_household_income_hkd_nominal",
        "source_field": "ma_hh",
        "unit": "HKD_per_month_nominal",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "comparable_definition_nominal_money",
        "primary_use": "within-year socioeconomic ranking and income quintiles",
        "decision": "retain_rank_primary",
        "limitation": (
            "Do not interpret nominal monetary change as real-income change until "
            "a CPI adjustment is specified."
        ),
    },
    {
        "standard_variable": "labour_force_participation_rate_pct",
        "source_field": "lfpr_t",
        "unit": "percent_population_aged_15_plus",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "economic participation context",
        "decision": "retain_secondary",
        "limitation": "Not a direct deprivation measure.",
    },
    {
        "standard_variable": "working_population",
        "source_field": "t_wp",
        "unit": "persons",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "economic population denominator",
        "decision": "retain_secondary",
        "limitation": "Count is sensitive to the selected boundary.",
    },
    {
        "standard_variable": "median_monthly_income_from_main_employment_hkd_nominal",
        "source_field": "t_mmearn",
        "unit": "HKD_per_month_nominal",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "comparable_definition_nominal_money",
        "primary_use": "sensitivity socioeconomic ranking",
        "decision": "retain_sensitivity",
        "limitation": (
            "Applies to the working population and requires CPI adjustment for "
            "monetary trend claims."
        ),
    },
    {
        "standard_variable": "median_rent_to_income_ratio_pct",
        "source_field": "dmr_ir",
        "unit": "percent",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "definition_comparable_boundary_harmonisation_required",
        "primary_use": "housing-cost pressure sensitivity",
        "decision": "retain_sensitivity",
        "limitation": "Defined only for relevant renter households.",
    },
    {
        "standard_variable": "post_secondary_education_share",
        "source_field": "derived_from_education_categories",
        "unit": "percent_population_aged_15_plus",
        "available_2011": True,
        "available_2016": True,
        "available_2021": True,
        "cross_year_status": "available_but_derivation_not_yet_frozen",
        "primary_use": "possible second socioeconomic axis",
        "decision": "defer_until_category_mapping_is_verified",
        "limitation": (
            "Do not assume that edu_deg alone represents all post-secondary "
            "education categories."
        ),
    },
]


WGS84_TO_HK1980 = Transformer.from_crs(
    "EPSG:4326", "EPSG:2326", always_xy=True
).transform
SYSTEM_SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already available: {destination.relative_to(PROJECT_ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading: {destination.name}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(
        request,
        timeout=180,
        context=SYSTEM_SSL_CONTEXT,
    ) as response, temporary_path.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    temporary_path.replace(destination)
    print(
        f"Saved: {destination.relative_to(PROJECT_ROOT)} "
        f"({destination.stat().st_size / (1024 * 1024):.1f} MB)"
    )


def load_geojson(source: CensusSource) -> dict:
    download_file(source.query_url, source.raw_geojson_path)
    try:
        with source.raw_geojson_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print(f"Incomplete GeoJSON cache detected; downloading {source.year} again.")
        source.raw_geojson_path.unlink()
        download_file(source.query_url, source.raw_geojson_path)
        with source.raw_geojson_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    features = data.get("features", [])
    if len(features) != source.expected_units:
        raise ValueError(
            f"{source.year} expected {source.expected_units} LTPUGs, found "
            f"{len(features)}."
        )
    ids = [str(feature["properties"].get("ltpug", "")).strip() for feature in features]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{source.year} LTPUG identifiers are missing or duplicated.")
    return data


def to_numeric(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def geometry_records(year: int, geojson: dict) -> list[dict]:
    records = []
    for feature in geojson["features"]:
        properties = feature["properties"]
        geometry_wgs84 = shape(feature["geometry"])
        if not geometry_wgs84.is_valid:
            geometry_wgs84 = make_valid(geometry_wgs84)
        geometry_projected = transform(WGS84_TO_HK1980, geometry_wgs84)
        records.append(
            {
                "year": year,
                "ltpug_id": str(properties["ltpug"]).strip(),
                "properties": properties,
                "geometry_wgs84": geometry_wgs84,
                "geometry_projected": geometry_projected,
                "area_km2": geometry_projected.area / 1_000_000.0,
            }
        )
    return records


def build_standardised_panel(all_records: dict[int, list[dict]]) -> pd.DataFrame:
    rows = []
    for year, records in all_records.items():
        for record in records:
            properties = record["properties"]
            row = {
                "year": year,
                "ltpug_id": record["ltpug_id"],
                "boundary_area_km2": round(record["area_km2"], 6),
                "source_dataset_id": SOURCES[year].dataset_id,
            }
            suppressed = 0
            for source_field, standard_field in STANDARD_FIELDS.items():
                value = to_numeric(properties.get(source_field))
                row[standard_field] = value
                suppressed += int(pd.isna(value))
            row["selected_field_missing_or_suppressed_count"] = suppressed
            row["income_price_basis"] = "nominal_HKD"
            row["geography_status"] = (
                "year_specific_large_tpu_group_boundary_not_yet_harmonised"
            )
            rows.append(row)
    panel = pd.DataFrame(rows).sort_values(["year", "ltpug_id"]).reset_index(drop=True)
    return panel


def crosswalk_pair(
    source_year: int,
    target_year: int,
    source_records: list[dict],
    target_records: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_geometries = [record["geometry_projected"] for record in target_records]
    target_tree = STRtree(target_geometries)
    rows = []
    for source_record in source_records:
        source_geometry = source_record["geometry_projected"]
        for target_index in target_tree.query(source_geometry, predicate="intersects"):
            target_record = target_records[int(target_index)]
            intersection_area = source_geometry.intersection(
                target_record["geometry_projected"]
            ).area
            if intersection_area <= 0:
                continue
            source_area = source_geometry.area
            target_area = target_record["geometry_projected"].area
            source_share = intersection_area / source_area
            target_share = intersection_area / target_area
            if max(source_share, target_share) < MIN_CROSSWALK_SHARE:
                continue
            union_area = source_area + target_area - intersection_area
            rows.append(
                {
                    "source_year": source_year,
                    "target_year": target_year,
                    "source_ltpug_id": source_record["ltpug_id"],
                    "target_ltpug_id": target_record["ltpug_id"],
                    "same_code": source_record["ltpug_id"] == target_record["ltpug_id"],
                    "intersection_area_km2": intersection_area / 1_000_000.0,
                    "share_of_source_area": source_share,
                    "share_of_target_area": target_share,
                    "intersection_over_union": intersection_area / union_area,
                }
            )
    crosswalk = pd.DataFrame(rows)
    crosswalk = crosswalk.sort_values(
        ["source_ltpug_id", "intersection_area_km2"], ascending=[True, False]
    ).reset_index(drop=True)
    crosswalk["source_overlap_rank"] = (
        crosswalk.groupby("source_ltpug_id").cumcount() + 1
    )

    best = crosswalk[crosswalk["source_overlap_rank"] == 1].copy()
    best["boundary_match_status"] = np.select(
        [
            best["same_code"]
            & (best["share_of_source_area"] >= STABLE_OVERLAP_THRESHOLD)
            & (best["share_of_target_area"] >= STABLE_OVERLAP_THRESHOLD),
            best["same_code"]
            & (best["share_of_source_area"] >= NEAR_STABLE_OVERLAP_THRESHOLD)
            & (best["share_of_target_area"] >= NEAR_STABLE_OVERLAP_THRESHOLD),
        ],
        ["stable_same_code", "near_stable_same_code"],
        default="changed_split_or_reassigned",
    )
    best["eligible_for_direct_panel_comparison"] = (
        best["boundary_match_status"] == "stable_same_code"
    )

    stable_count = int(best["eligible_for_direct_panel_comparison"].sum())
    source_count = len(source_records)
    same_code_rows = crosswalk[crosswalk["same_code"]]
    same_code_area = same_code_rows["intersection_area_km2"].sum()
    source_area = sum(record["area_km2"] for record in source_records)
    summary = {
        "source_year": source_year,
        "target_year": target_year,
        "source_unit_count": source_count,
        "target_unit_count": len(target_records),
        "common_identifier_count": len(
            {record["ltpug_id"] for record in source_records}
            & {record["ltpug_id"] for record in target_records}
        ),
        "best_match_same_code_count": int(best["same_code"].sum()),
        "stable_same_code_count": stable_count,
        "stable_same_code_pct": stable_count / source_count,
        "median_best_source_area_share": best["share_of_source_area"].median(),
        "median_best_target_area_share": best["share_of_target_area"].median(),
        "median_best_intersection_over_union": best[
            "intersection_over_union"
        ].median(),
        "same_code_intersection_share_of_source_area": same_code_area / source_area,
        "units_requiring_boundary_review": source_count - stable_count,
        "decision": (
            "use_stable_same_code_units_for_direct_panel_and_keep_all_units_for_"
            "repeated_cross_section"
        ),
    }
    return crosswalk, best, summary


def save_reference_boundary(geojson: dict) -> None:
    simplified = {
        "type": "FeatureCollection",
        "name": "2016 Large Tertiary Planning Unit Group reference boundaries",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "reference_year": REFERENCE_YEAR,
                    "ltpug_id": str(feature["properties"]["ltpug"]).strip(),
                },
                "geometry": feature["geometry"],
            }
            for feature in geojson["features"]
        ],
    }
    with gzip.open(REFERENCE_BOUNDARY_PATH, "wt", encoding="utf-8") as file:
        json.dump(simplified, file, ensure_ascii=False, separators=(",", ":"))


def add_polygon_patches(axis, records: list[dict], values: dict[str, str]) -> None:
    colours = {
        "stable_same_code": "#2A9D8F",
        "near_stable_same_code": "#E9C46A",
        "changed_split_or_reassigned": "#E76F51",
        "missing": "#B7B7B7",
    }
    patches = []
    face_colours = []
    for record in records:
        geometry = record["geometry_projected"]
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for polygon in polygons:
            patches.append(MplPolygon(np.asarray(polygon.exterior.coords), closed=True))
            face_colours.append(colours[values.get(record["ltpug_id"], "missing")])
    collection = PatchCollection(
        patches,
        facecolor=face_colours,
        edgecolor="#FFFFFF",
        linewidth=0.25,
    )
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.axis("off")


def make_boundary_map(
    reference_records: list[dict],
    best_matches: dict[tuple[int, int], pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for axis, source_year in zip(axes, (2011, 2021)):
        best = best_matches[(source_year, REFERENCE_YEAR)]
        status_by_reference = (
            best.sort_values("intersection_area_km2", ascending=False)
            .drop_duplicates("target_ltpug_id")
            .set_index("target_ltpug_id")["boundary_match_status"]
            .to_dict()
        )
        add_polygon_patches(axis, reference_records, status_by_reference)
        axis.set_title(f"{source_year} boundary matched to 2016 reference", fontsize=14)
    legend = [
        Patch(facecolor="#2A9D8F", label="Stable same code (>=95% both ways)"),
        Patch(facecolor="#E9C46A", label="Near stable same code (>=90%)"),
        Patch(facecolor="#E76F51", label="Changed, split, or reassigned"),
        Patch(facecolor="#B7B7B7", label="No retained match"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Large TPU Group boundary stability audit",
        fontsize=19,
        y=0.97,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Geometry overlap is a comparability screen, not a population-weighted "
            "crosswalk. 2021 remains a pandemic stress case."
        ),
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.17, wspace=0.04)
    fig.savefig(BOUNDARY_MAP_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def source_inventory_rows() -> list[dict]:
    rows = []
    for source in SOURCES.values():
        rows.append(
            {
                "year": source.year,
                "geography": "Large Tertiary Planning Unit Group",
                "expected_unit_count": source.expected_units,
                "dataset_id": source.dataset_id,
                "feature_service_url": source.feature_service_url,
                "official_dataset_page": source.official_dataset_page,
                "data_dictionary_url": source.data_dictionary_url,
                "raw_boundary_statistics_file": str(
                    source.raw_geojson_path.relative_to(PROJECT_ROOT)
                ),
                "role": (
                    "main_pre_covid_anchor"
                    if source.year == 2011
                    else (
                        "reference_geography_and_pre_covid_midpoint"
                        if source.year == 2016
                        else "2021_repeated_cross_section_no_networkwide_suppression_assumption"
                    )
                ),
            }
        )
    return rows


def save_tables(
    panel: pd.DataFrame,
    crosswalks: list[pd.DataFrame],
    best_matches: list[pd.DataFrame],
    pair_summaries: list[dict],
) -> None:
    panel.to_csv(PANEL_PATH, index=False)
    pd.concat(crosswalks, ignore_index=True).to_csv(CROSSWALK_PATH, index=False)
    pd.concat(best_matches, ignore_index=True).to_csv(MATCH_REVIEW_PATH, index=False)
    pd.DataFrame(pair_summaries).to_csv(PAIR_SUMMARY_PATH, index=False)
    pd.DataFrame(VARIABLE_AUDIT_ROWS).to_csv(VARIABLE_AUDIT_PATH, index=False)
    pd.DataFrame(source_inventory_rows()).to_csv(SOURCE_INVENTORY_PATH, index=False)

    summary_by_source = {row["source_year"]: row for row in pair_summaries}
    decision_rows = [
        {
            "metric": "selected_neighbourhood_geography",
            "count": np.nan,
            "value": "Large Tertiary Planning Unit Group",
            "decision": "best_first_balance_of_detail_and_three_year_variable_coverage",
        },
        {
            "metric": "reference_boundary_year",
            "count": REFERENCE_YEAR,
            "value": np.nan,
            "decision": "pre_covid_midpoint_and_main_2011_2016_comparison_anchor",
        },
        {
            "metric": "ltpug_count_2011",
            "count": SOURCES[2011].expected_units,
            "value": np.nan,
            "decision": "official_census_layer",
        },
        {
            "metric": "ltpug_count_2016",
            "count": SOURCES[2016].expected_units,
            "value": np.nan,
            "decision": "official_by_census_layer",
        },
        {
            "metric": "ltpug_count_2021",
            "count": SOURCES[2021].expected_units,
            "value": np.nan,
            "decision": "boundary_count_change_requires_harmonisation",
        },
        {
            "metric": "stable_direct_panel_units_2011_to_2016",
            "count": summary_by_source[2011]["stable_same_code_count"],
            "value": np.nan,
            "decision": "primary_same_neighbourhood_sensitivity_panel",
        },
        {
            "metric": "stable_direct_panel_units_2021_to_2016",
            "count": summary_by_source[2021]["stable_same_code_count"],
            "value": np.nan,
            "decision": "pandemic_stress_same_neighbourhood_sensitivity_panel",
        },
        {
            "metric": "direct_area_weighted_transfer_of_median_income_authorised",
            "count": 0,
            "value": np.nan,
            "decision": "medians_are_not_additive_and_area_weights_are_not_population_weights",
        },
        {
            "metric": "primary_equity_design",
            "count": np.nan,
            "value": "repeated_cross_section_plus_stable_same_code_panel_sensitivity",
            "decision": "avoids_forcing_changed_units_into_false_longitudinal_identity",
        },
        {
            "metric": "primary_socioeconomic_axis",
            "count": np.nan,
            "value": "within_year_household_income_rank_or_quintile",
            "decision": "avoids_unadjusted_nominal_income_change_claim",
        },
        {
            "metric": "post_secondary_education_axis_status",
            "count": 0,
            "value": "deferred",
            "decision": "category_mapping_must_be_verified_before_derivation",
        },
        {
            "metric": "raw_road_aadt_sum_ready_for_equity_burden",
            "count": 0,
            "value": np.nan,
            "decision": "road_direction_and_parallel_carriageway_rule_still_required",
        },
        {
            "metric": "year_2021_status",
            "count": np.nan,
            "value": "calendar_year_repeated_cross_section",
            "decision": "official_total_vkt_context_required_before_pandemic_interpretation",
        },
        {
            "metric": "step12_stage1_decision_signal",
            "count": np.nan,
            "value": "proceed_to_2016_reference_road_neighbourhood_aggregation",
            "decision": "retain_boundary_flags_and_do_not_area_weight_census_medians",
        },
    ]
    pd.DataFrame(decision_rows).to_csv(DECISION_AUDIT_PATH, index=False)


def main() -> None:
    for directory in (RAW_CENSUS_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    geojson_by_year = {}
    records_by_year = {}
    for year, source in SOURCES.items():
        download_file(source.data_dictionary_url, source.dictionary_path)
        geojson_by_year[year] = load_geojson(source)
        records_by_year[year] = geometry_records(year, geojson_by_year[year])
        print(f"Validated {len(records_by_year[year])} official LTPUGs for {year}.")

    panel = build_standardised_panel(records_by_year)
    crosswalks = []
    best_frames = []
    pair_summaries = []
    best_by_pair = {}
    for source_year in (2011, 2021):
        crosswalk, best, summary = crosswalk_pair(
            source_year,
            REFERENCE_YEAR,
            records_by_year[source_year],
            records_by_year[REFERENCE_YEAR],
        )
        crosswalks.append(crosswalk)
        best_frames.append(best)
        pair_summaries.append(summary)
        best_by_pair[(source_year, REFERENCE_YEAR)] = best

    save_tables(panel, crosswalks, best_frames, pair_summaries)
    save_reference_boundary(geojson_by_year[REFERENCE_YEAR])
    make_boundary_map(records_by_year[REFERENCE_YEAR], best_by_pair)

    for path in (
        PANEL_PATH,
        CROSSWALK_PATH,
        REFERENCE_BOUNDARY_PATH,
        SOURCE_INVENTORY_PATH,
        PAIR_SUMMARY_PATH,
        MATCH_REVIEW_PATH,
        VARIABLE_AUDIT_PATH,
        DECISION_AUDIT_PATH,
        BOUNDARY_MAP_PATH,
    ):
        print(f"Saved: {path.relative_to(PROJECT_ROOT)}")

    print("\nStep 12 Stage 1 Census and boundary audit is complete.")
    print(
        "Decision signal: proceed to 2016-reference road-neighbourhood "
        "aggregation; keep repeated cross-sections and a stable-unit panel "
        "as separate evidence."
    )


if __name__ == "__main__":
    main()
