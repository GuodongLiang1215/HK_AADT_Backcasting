"""Step 14: audit model and official road support before any calibration.

The original Step 14 compared predicted VKT on current centreline subsets with
official VKT on the Annual Traffic Census network. That comparison mixed two
quantities: traffic intensity and road-support length. It also treated
``route_number_present`` as the official major network even though Appendix H
defines the major network as the CTS simplified road network.

This corrected step therefore does not freeze a primary estimand and does not
rake any subset to an unmatched official total. It reports predicted and
official VKT, road-support length, and implied length-weighted mean AADT side by
side. Raw Step 11 predictions are preserved unchanged.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import tempfile
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

BACKCAST_PATH = PROCESSED_DIR / "atc_step11_full_network_backcast.csv"
TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
BENCHMARK_PATH = TABLE_DIR / "step13_official_vkt_benchmark.csv"
BOUNDARY_PATH = PROCESSED_DIR / "census_ltpug_2016_reference_boundaries.geojson.gz"
LEGACY_REGION_PATH = (
    PROCESSED_DIR / "atc_step14_network_estimand_and_calibrated_backcast.csv"
)

SUPPORT_PATH = PROCESSED_DIR / "atc_step14_network_support_audit.csv"
CONSISTENCY_PATH = TABLE_DIR / "step14_support_consistency.csv"
REGIONAL_PATH = TABLE_DIR / "step14_regional_support_diagnostics.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step14_support_decision_audit.csv"
FIGURE_PATH = FIGURE_DIR / "step14_support_consistency.png"

YEARS = (2011, 2016, 2021)
REGIONS = ("hong_kong_island", "kowloon", "new_territories")
REGION_BY_TPU_PREFIX = {"1": "hong_kong_island", "2": "kowloon"}
NEAREST_FALLBACK_LIMIT_M = 1000.0
LATITUDE_ORIGIN = 22.35
X_SCALE = 111_320.0 * math.cos(math.radians(LATITUDE_ORIGIN))
Y_SCALE = 110_540.0

CANDIDATE_SUPPORTS = (
    (
        "E0_full_centreline",
        "territory_total",
        "current centreline support; not the official census support",
        "support_not_matched_compare_length_and_mean_not_vkt_alone",
    ),
    (
        "E1_route_number_subset",
        "territory_major",
        "features carrying a route number; not the CTS major network definition",
        "proxy_subset_not_official_major_network",
    ),
    (
        "E2_station_street_subset",
        "territory_total",
        "route-number subset plus streets containing an ATC station",
        "subset_vs_total_is_not_an_accuracy_test",
    ),
    (
        "E3_supported_station_street_subset",
        "territory_total",
        "E2 restricted to observed-anchor or within-1km model support",
        "subset_vs_total_is_not_an_accuracy_test",
    ),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def normalise_code(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<null>"}:
        return ""
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = (BACKCAST_PATH, TRAINING_PATH, BENCHMARK_PATH, BOUNDARY_PATH)
    missing = [path.relative_to(PROJECT_ROOT) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing inputs: {missing}. Run Step 11, Step 12, and corrected Step 13 first."
        )
    columns = [
        "route_id",
        "street_code",
        "street_ename",
        "computed_length_m",
        "centroid_longitude",
        "centroid_latitude",
        "route_number_present",
        "spatial_support_tier",
        "direct_observed_station_count",
        *(f"predicted_aadt_{year}" for year in YEARS),
    ]
    backcast = pd.read_csv(BACKCAST_PATH, usecols=columns)
    backcast["route_id"] = backcast["route_id"].map(normalise_code)
    backcast["street_code"] = backcast["street_code"].map(normalise_code)
    backcast["route_number_present"] = (
        backcast["route_number_present"].astype(str).str.casefold() == "true"
    )
    training = pd.read_csv(TRAINING_PATH)
    training["street_code"] = training["street_code"].map(normalise_code)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    return backcast, training, benchmark


def load_reference_boundaries() -> tuple[list[object], list[str]]:
    from shapely.geometry import shape

    with gzip.open(BOUNDARY_PATH, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    geometries = [shape(feature["geometry"]) for feature in payload["features"]]
    identifiers = [str(feature["properties"]["ltpug_id"]) for feature in payload["features"]]
    return geometries, identifiers


def assign_region(backcast: pd.DataFrame) -> pd.DataFrame:
    if LEGACY_REGION_PATH.exists():
        prior = pd.read_csv(
            LEGACY_REGION_PATH,
            usecols=[
                "route_id",
                "reference_ltpug_id_2016",
                "reference_unit_assignment",
                "region",
            ],
            dtype={"route_id": str, "reference_ltpug_id_2016": str},
        )
        prior["route_id"] = prior["route_id"].map(normalise_code)
        if prior["route_id"].duplicated().any():
            raise ValueError("Saved Step 14 region assignments contain duplicate route_id values.")
        result = backcast.merge(prior, on="route_id", how="left", validate="one_to_one")
        if result["region"].isna().any():
            raise ValueError("Saved Step 14 region assignments do not cover every route_id.")
        result["reference_ltpug_id_2016"] = result[
            "reference_ltpug_id_2016"
        ].map(normalise_code)
        return result

    import shapely
    from shapely import STRtree

    geometries, identifiers = load_reference_boundaries()
    tree = STRtree(geometries)
    points = shapely.points(
        backcast["centroid_longitude"].to_numpy(dtype=float),
        backcast["centroid_latitude"].to_numpy(dtype=float),
    )
    assigned = np.full(len(backcast), "", dtype=object)
    method = np.full(len(backcast), "outside_all_reference_units", dtype=object)
    point_index, geometry_index = tree.query(points, predicate="intersects")
    seen: set[int] = set()
    for position, geometry_position in zip(point_index, geometry_index):
        if int(position) in seen:
            continue
        seen.add(int(position))
        assigned[position] = identifiers[geometry_position]
        method[position] = "within_reference_unit"
    unmatched = np.where(assigned == "")[0]
    if len(unmatched):
        nearest = tree.nearest(points[unmatched])
        for position, geometry_position in zip(unmatched, nearest):
            distance_m = geometries[geometry_position].distance(points[position]) * max(
                X_SCALE, Y_SCALE
            )
            if distance_m <= NEAREST_FALLBACK_LIMIT_M:
                assigned[position] = identifiers[geometry_position]
                method[position] = "nearest_reference_unit_within_1km"
    result = backcast.copy()
    result["reference_ltpug_id_2016"] = assigned
    result["reference_unit_assignment"] = method
    result["region"] = [
        REGION_BY_TPU_PREFIX.get(value[0], "new_territories") if value else "unassigned"
        for value in assigned
    ]
    return result


def add_candidate_support_flags(
    backcast: pd.DataFrame, training: pd.DataFrame
) -> pd.DataFrame:
    result = backcast.copy()
    station_street_codes = {code for code in training["street_code"] if code}
    result["on_atc_station_street"] = result["street_code"].isin(station_street_codes)
    supported = result["spatial_support_tier"].isin(
        ["observed_anchor_route", "within_1km_in_range"]
    )
    result["E0_full_centreline"] = True
    result["E1_route_number_subset"] = result["route_number_present"]
    result["E2_station_street_subset"] = (
        result["route_number_present"] | result["on_atc_station_street"]
    )
    result["E3_supported_station_street_subset"] = (
        result["E2_station_street_subset"] & supported
    )
    result["in_primary_estimand"] = False
    result["estimand_status"] = "not_frozen_official_network_mapping_required"
    return result


def daily_vehicle_km(frame: pd.DataFrame, year: int) -> float:
    return float(
        (frame[f"predicted_aadt_{year}"] * frame["computed_length_m"] / 1000).sum()
    )


def official_values(
    benchmark: pd.DataFrame, year: int, benchmark_name: str
) -> tuple[float, float, float]:
    row = benchmark[benchmark["census_year"] == year].iloc[0]
    prefix = "territory_total" if benchmark_name == "territory_total" else "territory_major"
    return (
        float(row[f"{prefix}_daily_vehicle_km"]),
        float(row[f"{prefix}_road_length_km"]),
        float(row[f"{prefix}_implied_mean_aadt"]),
    )


def build_consistency_table(
    backcast: pd.DataFrame, benchmark: pd.DataFrame
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for support, official_name, description, status in CANDIDATE_SUPPORTS:
        subset = backcast[backcast[support]]
        model_length = float(subset["computed_length_m"].sum()) / 1000
        for year in YEARS:
            predicted_vkt = daily_vehicle_km(subset, year)
            predicted_mean = predicted_vkt / model_length
            official_vkt, official_length, official_mean = official_values(
                benchmark, year, official_name
            )
            rows.append(
                {
                    "candidate_support": support,
                    "support_description": description,
                    "comparison_status": status,
                    "official_benchmark": official_name,
                    "year": year,
                    "segment_count": len(subset),
                    "model_support_length_km": round(model_length, 3),
                    "official_support_length_km": round(official_length, 3),
                    "model_over_official_length_ratio": round(model_length / official_length, 4),
                    "predicted_daily_vehicle_km": round(predicted_vkt, 1),
                    "official_daily_vehicle_km": round(official_vkt, 1),
                    "predicted_over_official_vkt_ratio": round(predicted_vkt / official_vkt, 4),
                    "predicted_length_weighted_mean_aadt": round(predicted_mean, 2),
                    "official_implied_mean_aadt": round(official_mean, 2),
                    "predicted_over_official_mean_aadt_ratio": round(predicted_mean / official_mean, 4),
                    "eligible_as_calibration_target": False,
                }
            )
    return rows


def build_regional_route_subset_diagnostics(
    backcast: pd.DataFrame, benchmark: pd.DataFrame
) -> list[dict[str, object]]:
    route_subset = backcast[backcast["E1_route_number_subset"]]
    rows: list[dict[str, object]] = []
    for year in YEARS:
        official_row = benchmark[benchmark["census_year"] == year].iloc[0]
        for region in REGIONS:
            subset = route_subset[route_subset["region"] == region]
            model_length = float(subset["computed_length_m"].sum()) / 1000
            predicted_vkt = daily_vehicle_km(subset, year)
            official_length = float(official_row[f"{region}_major_road_length_km"])
            official_vkt = float(official_row[f"{region}_major_daily_vehicle_km"])
            rows.append(
                {
                    "year": year,
                    "region": region,
                    "candidate_support": "E1_route_number_subset",
                    "official_network": "major_CTS_simplified_network",
                    "segment_count": len(subset),
                    "route_subset_length_km": round(model_length, 3),
                    "official_major_length_km": round(official_length, 3),
                    "route_subset_share_of_official_major_length_pct": round(100 * model_length / official_length, 2),
                    "route_subset_predicted_vkt": round(predicted_vkt, 1),
                    "official_major_vkt": round(official_vkt, 1),
                    "route_subset_predicted_mean_aadt": round(predicted_vkt / model_length, 2),
                    "official_major_implied_mean_aadt": round(official_vkt / official_length, 2),
                    "decision": "route_number_subset_is_not_the_official_major_network",
                }
            )
    return rows


def build_decision_audit(
    backcast: pd.DataFrame,
    consistency_rows: list[dict[str, object]],
    regional_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    consistency = pd.DataFrame(consistency_rows)
    full = consistency[consistency["candidate_support"] == "E0_full_centreline"]
    regional = pd.DataFrame(regional_rows)
    return [
        {"metric": "network_segment_count", "count": len(backcast), "value": "", "decision": "matches_step11_backcast"},
        {"metric": "full_centreline_vkt_ratio_range", "count": "", "value": f"{full['predicted_over_official_vkt_ratio'].min():.3f}_to_{full['predicted_over_official_vkt_ratio'].max():.3f}", "decision": "do_not_interpret_without_support_length"},
        {"metric": "full_centreline_length_ratio_range", "count": "", "value": f"{full['model_over_official_length_ratio'].min():.3f}_to_{full['model_over_official_length_ratio'].max():.3f}", "decision": "explains_most_of_the_raw_vkt_ratio"},
        {"metric": "full_centreline_mean_aadt_ratio_range", "count": "", "value": f"{full['predicted_over_official_mean_aadt_ratio'].min():.3f}_to_{full['predicted_over_official_mean_aadt_ratio'].max():.3f}", "decision": "traffic_intensity_check_after_dividing_by_each_support_length"},
        {"metric": "route_subset_official_major_length_share_range_pct", "count": "", "value": f"{regional['route_subset_share_of_official_major_length_pct'].min():.1f}_to_{regional['route_subset_share_of_official_major_length_pct'].max():.1f}", "decision": "route_number_subset_cannot_receive_the_full_official_major_vkt"},
        {"metric": "primary_estimand_frozen", "count": 0, "value": "official_CTS_major_minor_mapping_required", "decision": "supersedes_original_step14_E1_freeze"},
        {"metric": "regional_raking_performed", "count": 0, "value": "", "decision": "no_subset_is_raked_to_an_unmatched_official_total"},
        {"metric": "step14_decision_signal", "count": "", "value": "support_mismatch_explains_the_apparent_twofold_vkt_gap", "decision": "map_the_official_census_network_before_calibration_or_equity_aggregation"},
    ]


def plot_support_consistency(consistency_rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(consistency_rows)
    full = frame[frame["candidate_support"] == "E0_full_centreline"].set_index("year")
    metrics = (
        ("predicted_over_official_vkt_ratio", "VKT ratio", "#D35400"),
        ("model_over_official_length_ratio", "Support-length ratio", "#7F8C8D"),
        ("predicted_over_official_mean_aadt_ratio", "Mean-AADT ratio", "#2E86AB"),
    )
    positions = np.arange(len(YEARS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.8, 5.6))
    for index, (column, label, colour) in enumerate(metrics):
        values = [float(full.loc[year, column]) for year in YEARS]
        bars = axis.bar(positions + (index - 1) * width, values, width, label=label, color=colour)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.axhline(1.0, color="#202124", linestyle="--", linewidth=1.1)
    axis.set_xticks(positions, [str(year) for year in YEARS])
    axis.set_ylabel("Model support ÷ official census support")
    axis.set_title("The apparent twofold VKT gap follows the twofold road-support gap")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    backcast, training, benchmark = read_inputs()
    backcast = assign_region(backcast)
    backcast = add_candidate_support_flags(backcast, training)
    consistency_rows = build_consistency_table(backcast, benchmark)
    regional_rows = build_regional_route_subset_diagnostics(backcast, benchmark)
    audit_rows = build_decision_audit(backcast, consistency_rows, regional_rows)

    output_columns = [
        "route_id", "street_code", "street_ename", "computed_length_m",
        "centroid_longitude", "centroid_latitude", "reference_ltpug_id_2016",
        "reference_unit_assignment", "region", "route_number_present",
        "on_atc_station_street", "spatial_support_tier",
        "direct_observed_station_count",
        *(name for name, _, _, _ in CANDIDATE_SUPPORTS),
        "in_primary_estimand", "estimand_status",
        *(f"predicted_aadt_{year}" for year in YEARS),
    ]
    SUPPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    backcast[output_columns].to_csv(SUPPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {SUPPORT_PATH.relative_to(PROJECT_ROOT)}")
    write_csv(CONSISTENCY_PATH, consistency_rows)
    write_csv(REGIONAL_PATH, regional_rows)
    write_csv(DECISION_AUDIT_PATH, audit_rows)
    plot_support_consistency(consistency_rows)

    consistency = pd.DataFrame(consistency_rows)
    full = consistency[consistency["candidate_support"] == "E0_full_centreline"]
    print("\nStep 14 support-consistency audit is complete.")
    for row in full.itertuples(index=False):
        print(
            f"  {int(row.year)}  VKT ratio {float(row.predicted_over_official_vkt_ratio):.2f}  "
            f"length ratio {float(row.model_over_official_length_ratio):.2f}  "
            f"mean-AADT ratio {float(row.predicted_over_official_mean_aadt_ratio):.2f}"
        )
    print(
        "\nDecision: no primary estimand is frozen and no regional raking is performed. "
        "Map the official CTS major/minor network before calibration or equity aggregation."
    )
    print("Next: python src\\15_recheck_model_evidence.py")


if __name__ == "__main__":
    main()
