"""Step 23B-data: audit historical OSM availability, coverage, and edit churn.

This is a data audit, not a historical AADT model.  It downloads or resumes
dated OSM highway snapshots for 2018--2024, plus 2016 as a census-year bridge,
and maps them to one fixed 2023 official centreline solely for compatibility
diagnostics.  The existing Step 23A 2023 cache is reused.

The script also demonstrates the Overpass pre-attic failure mode by directly
comparing a small 2011 request with the documented first attic state on
2012-09-12.  Comparisons use IDs, tags, versions, timestamps, and coordinates
directly; no digest is generated.

Passing this step cannot authorise historical modelling.  OSM tag changes can
be map edits rather than real traffic or road changes, and a fixed 2023
centreline is not a historical road-network truth.  Step 23B modelling remains
locked until Step 23A.1 and a separate temporal-identification design pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "step23b_osm_history"
STEP23A_RAW = PROJECT_ROOT / "data" / "raw" / "step23a_osm_2023"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

BASE_SCRIPT = PROJECT_ROOT / "src" / "23a_test_2023_osm_road_class.py"
STEP18_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
STEP22_ROAD_MATCH_PATH = PROCESSED_DIR / "atc_step22_2023_road_matches.csv"

SEGMENT_YEAR_PATH = PROCESSED_DIR / "atc_step23b_osm_segment_year_panel.csv"
STATION_YEAR_PATH = PROCESSED_DIR / "atc_step23b_station_year_osm_panel.csv"
STATION_PAIR_PATH = PROCESSED_DIR / "atc_step23b_station_change_pairs.csv"

PRE2012_PATH = TABLE_DIR / "step23b_overpass_pre2012_audit.csv"
SOURCE_AUDIT_PATH = TABLE_DIR / "step23b_snapshot_source_audit.csv"
COVERAGE_PATH = TABLE_DIR / "step23b_snapshot_coverage.csv"
CHURN_PATH = TABLE_DIR / "step23b_osm_churn_by_transition.csv"
ALIGNMENT_PATH = TABLE_DIR / "step23b_aadt_change_alignment.csv"
DECISION_PATH = TABLE_DIR / "step23b_data_decision_audit.csv"

COVERAGE_FIGURE_PATH = FIGURE_DIR / "step23b_osm_snapshot_coverage.png"
CHURN_FIGURE_PATH = FIGURE_DIR / "step23b_osm_churn_and_aadt_change.png"

SNAPSHOT_YEARS = (2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
PRIMARY_YEARS = tuple(range(2018, 2025))
NETWORK_LENGTH_COVERAGE_THRESHOLD = 0.80
STATION_COVERAGE_THRESHOLD = 0.90
MINIMUM_TAG_CHANGE_SHARE = 0.05

OVERPASS_FIRST_ATTIC_STATE = "2012-09-12T06:55:00Z"
PRE_ATTIC_TIMESTAMP = "2011-06-30T00:00:00Z"
PROBE_BBOX = (22.2700, 114.1500, 22.2850, 114.1700)
OVERPASS_DOC = "https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL#Date"

MOTOR_HIGHWAY_BASES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "road",
    "service",
    "track",
}
TRANSPORT_TAGS = (
    "osm_highway",
    "osm_highway_group",
    "osm_service",
    "osm_lanes_raw",
    "osm_maxspeed_raw",
    "osm_oneway_raw",
    "osm_access_raw",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def first_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def probe_signature(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    signatures: dict[str, dict[str, object]] = {}
    for element in payload.get("elements", []):
        if element.get("type") != "way" or element.get("id") is None:
            continue
        tags = element.get("tags", {}) or {}
        signatures[f"way/{element['id']}"] = {
            "version": element.get("version"),
            "timestamp": element.get("timestamp"),
            "highway": first_value(tags.get("highway")),
            "service": first_value(tags.get("service")),
            "lanes": first_value(tags.get("lanes")),
            "maxspeed": first_value(tags.get("maxspeed")),
            "oneway": first_value(tags.get("oneway")),
            "access": first_value(tags.get("access")),
            "geometry": tuple(
                (float(node["lat"]), float(node["lon"]))
                for node in element.get("geometry", [])
            ),
        }
    return signatures


def obtain_probe(base, timestamp: str, label: str) -> dict[str, object]:
    south, west, north, east = PROBE_BBOX
    query = (
        f'[out:json][timeout:180][date:"{timestamp}"];'
        f'way["highway"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});'
        "out meta geom;"
    )
    path = RAW_ROOT / "pre2012_probe" / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size < 100:
        base.download_overpass_tile(query, path, f"pre-attic audit {label}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "elements" not in payload:
        raise ValueError(f"Unexpected Overpass probe response in {path.name}")
    return payload


def pre2012_audit(base) -> pd.DataFrame:
    before = obtain_probe(base, PRE_ATTIC_TIMESTAMP, "20110630")
    first = obtain_probe(base, OVERPASS_FIRST_ATTIC_STATE, "20120912_first_attic")
    left = probe_signature(before)
    right = probe_signature(first)
    left_ids = set(left)
    right_ids = set(right)
    shared = sorted(left_ids & right_ids)
    rows = [
        {
            "comparison": "2011_request_vs_first_available_attic_state",
            "requested_timestamp_a": PRE_ATTIC_TIMESTAMP,
            "requested_timestamp_b": OVERPASS_FIRST_ATTIC_STATE,
            "returned_osm_base_a": (before.get("osm3s", {}) or {}).get("timestamp_osm_base", ""),
            "returned_osm_base_b": (first.get("osm3s", {}) or {}).get("timestamp_osm_base", ""),
            "way_count_a": len(left),
            "way_count_b": len(right),
            "symmetric_way_id_difference_count": len(left_ids ^ right_ids),
            "version_difference_count": sum(left[value]["version"] != right[value]["version"] for value in shared),
            "timestamp_difference_count": sum(left[value]["timestamp"] != right[value]["timestamp"] for value in shared),
            "highway_difference_count": sum(left[value]["highway"] != right[value]["highway"] for value in shared),
            "service_difference_count": sum(left[value]["service"] != right[value]["service"] for value in shared),
            "geometry_difference_count": sum(left[value]["geometry"] != right[value]["geometry"] for value in shared),
            "complete_direct_equality": left == right,
            "official_interpretation": "Overpass has no attic before 2012-09-12; a 2011 date is not a valid 2011 OSM state",
            "documentation": OVERPASS_DOC,
        }
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, PRE2012_PATH)
    return frame


def raw_metadata(features: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for feature in features:
        if feature.get("type") != "way" or feature.get("id") is None:
            continue
        osm_id = f"way/{feature['id']}"
        if osm_id in seen:
            continue
        seen.add(osm_id)
        tags = feature.get("tags", {}) or {}
        highway = first_value(tags.get("highway")).strip().lower()
        rows.append(
            {
                "osm_id": osm_id,
                "feature_version": feature.get("version"),
                "feature_timestamp": feature.get("timestamp"),
                "osm_highway_raw": highway,
                "osm_service": first_value(tags.get("service")).strip().lower(),
                "osm_lanes_raw": first_value(tags.get("lanes")).strip().lower(),
                "osm_maxspeed_raw": first_value(tags.get("maxspeed")).strip().lower(),
                "osm_oneway_raw": first_value(tags.get("oneway")).strip().lower(),
                "osm_access_raw": first_value(tags.get("access")).strip().lower(),
            }
        )
    return pd.DataFrame(rows)


def snapshot_raw_dir(year: int) -> Path:
    return STEP23A_RAW if year == 2023 else RAW_ROOT / str(year)


def obtain_snapshot(base, year, centerline, road_geometries):
    timestamp = f"{year}-06-30T00:00:00Z"
    base.OSM_TIMESTAMP = timestamp
    base.RAW_DIR = snapshot_raw_dir(year)
    base.OSM_WAY_TABLE_PATH = PROCESSED_DIR / f"atc_step23b_osm_{year}_way_table.csv"
    base.OSM_CROSSWALK_PATH = PROCESSED_DIR / f"atc_step23b_osm_{year}_network_crosswalk.csv"
    print(f"\nObtaining OSM snapshot {timestamp}...")
    features, tile_audit = base.obtain_osm_tiles(centerline)
    metadata = raw_metadata(features)
    ways, geometries, source_totals = base.parse_osm_ways(features)
    ways = ways.merge(
        metadata.drop(columns=["osm_highway_raw"], errors="ignore"),
        on="osm_id",
        how="left",
        validate="one_to_one",
    )
    eligible = ways["osm_highway"].fillna("").map(
        lambda value: str(value).removesuffix("_link") in MOTOR_HIGHWAY_BASES
    )
    ways = ways.loc[eligible].reset_index(drop=True)
    geometries = geometries[np.asarray(eligible, dtype=bool)]
    save_csv(ways, base.OSM_WAY_TABLE_PATH)
    crosswalk = base.match_centerline_to_osm(
        centerline, road_geometries, ways, geometries
    )
    crosswalk = crosswalk.merge(
        ways[
            [
                "osm_id",
                "feature_version",
                "feature_timestamp",
                "osm_service",
                "osm_lanes_raw",
                "osm_maxspeed_raw",
                "osm_oneway_raw",
                "osm_access_raw",
            ]
        ],
        on="osm_id",
        how="left",
        validate="many_to_one",
    )
    crosswalk.insert(1, "year", year)
    crosswalk.insert(2, "snapshot_timestamp", timestamp)
    save_csv(crosswalk, base.OSM_CROSSWALK_PATH)
    source = pd.concat([tile_audit, source_totals], ignore_index=True, sort=False)
    source.insert(0, "year", year)
    source.insert(1, "snapshot_timestamp", timestamp)
    source["raw_cache_directory"] = str(snapshot_raw_dir(year).relative_to(PROJECT_ROOT))
    source["reuses_step23a_cache"] = year == 2023
    return crosswalk, source


def station_support_for_year(
    year: int,
    crosswalk: pd.DataFrame,
    measured: pd.DataFrame,
    station_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    labels = measured[measured["year"].astype(int) == year].copy()
    labels = labels.merge(
        station_matches,
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    mapped_share = labels["road_2023_segment_index"].notna().mean()
    labels = labels.merge(
        crosswalk.drop(columns=["year", "snapshot_timestamp"], errors="ignore"),
        on="road_2023_segment_index",
        how="left",
        validate="many_to_one",
    )
    labels.insert(1, "snapshot_year", year)
    accepted_share = labels["osm_match_status"].isin(["high", "moderate"]).mean()
    return labels, float(mapped_share), float(accepted_share)


def snapshot_coverage(
    crosswalk: pd.DataFrame,
    station_mapped_share: float,
    station_osm_share: float,
) -> dict[str, object]:
    accepted = crosswalk["osm_match_status"].isin(["high", "moderate"])
    length = crosswalk["road_segment_length_m"].to_numpy(dtype=float)
    return {
        "year": int(crosswalk["year"].iloc[0]),
        "official_segment_count": len(crosswalk),
        "accepted_segment_share": accepted.mean(),
        "accepted_length_share": length[accepted].sum() / length.sum(),
        "station_to_2023_segment_mapping_share": station_mapped_share,
        "station_accepted_osm_share": station_osm_share,
        "network_coverage_pass": length[accepted].sum() / length.sum() >= NETWORK_LENGTH_COVERAGE_THRESHOLD,
        "station_coverage_pass": station_osm_share >= STATION_COVERAGE_THRESHOLD,
        "fixed_support_warning": "2023 official centreline is a compatibility support, not historical network truth",
    }


def build_churn(segment_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = segment_year.sort_values(["road_2023_segment_index", "year"]).copy()
    pair_rows: list[pd.DataFrame] = []
    transitions = ((2016, 2018), *((year - 1, year) for year in range(2019, 2025)))
    for previous_year, current_year in transitions:
        left = panel[panel["year"] == previous_year].copy()
        right = panel[panel["year"] == current_year].copy()
        columns = [
            "road_2023_segment_index",
            "road_segment_length_m",
            "osm_id",
            "osm_match_status",
            *TRANSPORT_TAGS,
        ]
        paired = left[columns].merge(
            right[columns],
            on="road_2023_segment_index",
            suffixes=("_previous", "_current"),
            validate="one_to_one",
        )
        paired.insert(1, "previous_year", previous_year)
        paired.insert(2, "current_year", current_year)
        paired["transition_role"] = "primary_annual" if previous_year >= 2018 else "supplemental_2016_to_2018"
        paired["accepted_both"] = (
            paired["osm_match_status_previous"].isin(["high", "moderate"])
            & paired["osm_match_status_current"].isin(["high", "moderate"])
        )
        paired["osm_id_changed"] = paired["osm_id_previous"].fillna("") != paired["osm_id_current"].fillna("")
        for column in TRANSPORT_TAGS:
            paired[f"{column}_changed"] = (
                paired[f"{column}_previous"].fillna("").astype(str)
                != paired[f"{column}_current"].fillna("").astype(str)
            )
        paired["any_transport_tag_changed"] = paired[
            [f"{column}_changed" for column in TRANSPORT_TAGS]
        ].any(axis=1)
        pair_rows.append(paired)
    pairs = pd.concat(pair_rows, ignore_index=True)
    rows = []
    for (previous_year, current_year, role), group in pairs.groupby(
        ["previous_year", "current_year", "transition_role"]
    ):
        accepted = group[group["accepted_both"]].copy()
        weights = accepted["road_segment_length_m_current"].to_numpy(dtype=float)
        changed = accepted["any_transport_tag_changed"].to_numpy(dtype=bool)
        rows.append(
            {
                "previous_year": previous_year,
                "current_year": current_year,
                "transition_role": role,
                "segment_pairs": len(group),
                "accepted_both_count": len(accepted),
                "osm_id_change_share": accepted["osm_id_changed"].mean(),
                "highway_group_change_share": accepted["osm_highway_group_changed"].mean(),
                "any_transport_tag_change_share": changed.mean(),
                "length_weighted_any_tag_change_share": weights[changed].sum() / weights.sum() if weights.sum() else np.nan,
                "tag_change_share_with_osm_id_change": accepted.loc[accepted["any_transport_tag_changed"], "osm_id_changed"].mean(),
                "interpretation": "same fixed 2023 segment; changes may be rematching or map editing rather than road change",
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, CHURN_PATH)
    return pairs, frame


def build_station_change_alignment(station_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = station_year[station_year["year"].isin(PRIMARY_YEARS)].copy()
    rows: list[pd.DataFrame] = []
    for current_year in PRIMARY_YEARS[1:]:
        previous_year = current_year - 1
        left = primary[primary["year"] == previous_year][
            ["station_id", "aadt", "osm_id", "osm_match_status", *TRANSPORT_TAGS]
        ]
        right = primary[primary["year"] == current_year][
            ["station_id", "aadt", "osm_id", "osm_match_status", *TRANSPORT_TAGS]
        ]
        paired = left.merge(
            right,
            on="station_id",
            suffixes=("_previous", "_current"),
            validate="one_to_one",
        )
        paired.insert(1, "previous_year", previous_year)
        paired.insert(2, "current_year", current_year)
        paired = paired[
            paired["osm_match_status_previous"].isin(["high", "moderate"])
            & paired["osm_match_status_current"].isin(["high", "moderate"])
        ].copy()
        paired["aadt_change"] = paired["aadt_current"] - paired["aadt_previous"]
        paired["absolute_aadt_change"] = paired["aadt_change"].abs()
        paired["aadt_change_pct"] = 100.0 * paired["aadt_change"] / paired["aadt_previous"]
        paired["osm_id_changed"] = paired["osm_id_previous"].fillna("") != paired["osm_id_current"].fillna("")
        for column in TRANSPORT_TAGS:
            paired[f"{column}_changed"] = (
                paired[f"{column}_previous"].fillna("").astype(str)
                != paired[f"{column}_current"].fillna("").astype(str)
            )
        paired["any_transport_tag_changed"] = paired[
            [f"{column}_changed" for column in TRANSPORT_TAGS]
        ].any(axis=1)
        rows.append(paired)
    pairs = pd.concat(rows, ignore_index=True)
    save_csv(pairs, STATION_PAIR_PATH)
    summaries = []
    for label, group in [("all_2018_2024", pairs), *pairs.groupby(["previous_year", "current_year"])]:
        total_change = group["absolute_aadt_change"].sum()
        unchanged = ~group["any_transport_tag_changed"]
        rho = (
            spearmanr(group["any_transport_tag_changed"].astype(int), group["absolute_aadt_change"]).statistic
            if group["any_transport_tag_changed"].nunique() > 1
            else np.nan
        )
        summaries.append(
            {
                "transition": label if isinstance(label, str) else f"{label[0]}-{label[1]}",
                "station_pairs": len(group),
                "osm_unchanged_pair_share": unchanged.mean(),
                "share_total_absolute_aadt_change_on_osm_unchanged_pairs": group.loc[unchanged, "absolute_aadt_change"].sum() / total_change if total_change else np.nan,
                "median_abs_aadt_change_osm_unchanged": group.loc[unchanged, "absolute_aadt_change"].median(),
                "median_abs_aadt_change_osm_changed": group.loc[~unchanged, "absolute_aadt_change"].median(),
                "spearman_tag_change_indicator_vs_abs_aadt_change": rho,
                "interpretation": "association is descriptive and does not establish that an OSM edit is a real road change",
            }
        )
    frame = pd.DataFrame(summaries)
    save_csv(frame, ALIGNMENT_PATH)
    return pairs, frame


def decisions(pre2012, coverage, churn, alignment) -> pd.DataFrame:
    network_pass = bool(coverage["network_coverage_pass"].all())
    station_pass = bool(
        coverage.loc[coverage["year"].isin(PRIMARY_YEARS), "station_coverage_pass"].all()
    )
    primary_churn = churn[churn["transition_role"] == "primary_annual"]
    tag_change_share = (
        primary_churn["any_transport_tag_change_share"].mul(primary_churn["accepted_both_count"]).sum()
        / primary_churn["accepted_both_count"].sum()
    )
    variation_pass = tag_change_share >= MINIMUM_TAG_CHANGE_SHARE
    all_alignment = alignment[alignment["transition"] == "all_2018_2024"].iloc[0]
    rows = [
        {
            "decision": "overpass_can_supply_a_valid_2011_snapshot",
            "pass": False,
            "evidence": f"official first attic state={OVERPASS_FIRST_ATTIC_STATE}; direct equality={bool(pre2012['complete_direct_equality'].iloc[0])}",
            "failed_criterion": "pre_attic_date_is_not_a_2011_state",
            "action": "do not use an Overpass 2011 request as a historical 2011 feature layer",
        },
        {
            "decision": "2016_and_2018_2024_fixed_support_network_coverage_passes",
            "pass": network_pass,
            "evidence": f"minimum_length_coverage={coverage['accepted_length_share'].min():.3f}; threshold={NETWORK_LENGTH_COVERAGE_THRESHOLD:.2f}",
            "failed_criterion": "none" if network_pass else "one_or_more_years_below_network_coverage_threshold",
            "action": "interpret this as compatibility with a fixed 2023 centreline, not historical network completeness",
        },
        {
            "decision": "2018_2024_measured_station_osm_coverage_passes",
            "pass": station_pass,
            "evidence": f"minimum_station_coverage={coverage.loc[coverage['year'].isin(PRIMARY_YEARS), 'station_accepted_osm_share'].min():.3f}; threshold={STATION_COVERAGE_THRESHOLD:.2f}",
            "failed_criterion": "none" if station_pass else "one_or_more_years_below_station_coverage_threshold",
            "action": "retain the station-to-2023-segment mapping share as a separate denominator",
        },
        {
            "decision": "osm_has_minimum_observed_within_segment_tag_variation",
            "pass": variation_pass,
            "evidence": f"pooled_primary_transition_tag_change_share={tag_change_share:.3f}; threshold={MINIMUM_TAG_CHANGE_SHARE:.2f}",
            "failed_criterion": "none" if variation_pass else "tag_variation_below_predeclared_5pct_floor",
            "action": "variation is necessary but not sufficient for a temporal traffic model",
        },
        {
            "decision": "osm_changes_are_proven_real_road_changes",
            "pass": False,
            "evidence": "OSM alone cannot distinguish mapping edits, rematching, and physical road changes",
            "failed_criterion": "change_provenance_not_identified",
            "action": "do not interpret tag churn as traffic change without an external change source",
        },
        {
            "decision": "osm_tags_explain_most_annual_aadt_change",
            "pass": False,
            "evidence": f"share_absolute_aadt_change_when_osm_unchanged={all_alignment['share_total_absolute_aadt_change_on_osm_unchanged_pairs']:.3f}",
            "failed_criterion": "descriptive_alignment_is_not_temporal_identification",
            "action": "report the unchanged-OSM share rather than claiming annual explanatory power",
        },
        {
            "decision": "step23b_data_authorises_historical_osm_modelling",
            "pass": False,
            "evidence": "data availability, fixed-support coverage, and edit churn do not validate segment-level temporal downscaling",
            "failed_criterion": "step23a1_and_nonartifactual_temporal_signal_gates_are_separate",
            "action": "keep historical modelling locked; use this audit to decide whether another dynamic source is required",
        },
        {
            "decision": "step23b_data_establishes_multiyear_segment_backcasting",
            "pass": False,
            "evidence": "no historical AADT model is trained in Step 23B-data",
            "failed_criterion": "data_audit_only",
            "action": "do not call these OSM snapshots a validated backcast",
        },
    ]
    frame = pd.DataFrame(rows)
    save_csv(frame, DECISION_PATH)
    return frame


def plots(coverage: pd.DataFrame, churn: pd.DataFrame, alignment: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(coverage["year"], 100 * coverage["accepted_length_share"], marker="o", label="Network length")
    axis.plot(coverage["year"], 100 * coverage["station_accepted_osm_share"], marker="s", label="Measured stations")
    axis.axhline(80, color="#C44E52", linestyle="--", linewidth=1)
    axis.set_ylabel("Accepted OSM match coverage (%)")
    axis.set_title("Historical OSM coverage on a fixed 2023 compatibility support")
    axis.legend()
    figure.tight_layout()
    figure.savefig(COVERAGE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    primary = churn[churn["transition_role"] == "primary_annual"].copy()
    labels = primary["previous_year"].astype(str) + "-" + primary["current_year"].astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(labels, 100 * primary["any_transport_tag_change_share"], color="#4C78A8")
    axes[0].axhline(100 * MINIMUM_TAG_CHANGE_SHARE, color="#C44E52", linestyle="--")
    axes[0].set_ylabel("Segments with any OSM tag change (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("OSM churn is not automatically road change")
    all_row = alignment[alignment["transition"] == "all_2018_2024"].iloc[0]
    shares = [
        all_row["share_total_absolute_aadt_change_on_osm_unchanged_pairs"],
        1 - all_row["share_total_absolute_aadt_change_on_osm_unchanged_pairs"],
    ]
    axes[1].bar(["OSM unchanged", "OSM changed"], 100 * np.asarray(shares), color=["#9E9E9E", "#F28E2B"])
    axes[1].set_ylabel("Share of total absolute station AADT change (%)")
    axes[1].set_title("Where measured AADT change occurs")
    figure.tight_layout()
    figure.savefig(CHURN_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def update_manifest(paths: list[Path]) -> None:
    existing = (
        pd.read_csv(REPORT_MANIFEST_PATH)
        if REPORT_MANIFEST_PATH.exists()
        else pd.DataFrame(columns=["artifact", "status", "reason"])
    )
    names = {str(path.relative_to(PROJECT_ROOT)) for path in paths}
    existing = existing[~existing["artifact"].isin(names)]
    added = pd.DataFrame(
        {
            "artifact": sorted(names),
            "status": "step23b_data_audit",
            "reason": "historical_osm_availability_fixed_support_coverage_and_edit_churn_only",
        }
    )
    save_csv(pd.concat([existing, added], ignore_index=True), REPORT_MANIFEST_PATH)


def validate_inputs() -> None:
    required = [BASE_SCRIPT, STEP18_PANEL_PATH, STEP22_ROAD_MATCH_PATH]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run Steps 18, 22, and 23A first; missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=list(SNAPSHOT_YEARS),
        help="Snapshot years to obtain. Complete decisions require the default set.",
    )
    args = parser.parse_args()
    years = tuple(sorted(set(args.years)))
    if years != SNAPSHOT_YEARS:
        raise ValueError(f"The frozen complete audit uses years {SNAPSHOT_YEARS}; received {years}")
    validate_inputs()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base = load_module("hk_aadt_step23a_for_history", BASE_SCRIPT)
    pre2012 = pre2012_audit(base)
    centerline, road_geometries = base.read_centerline(base.find_road_geodatabase())
    measured = pd.read_csv(STEP18_PANEL_PATH)
    station_matches = pd.read_csv(STEP22_ROAD_MATCH_PATH)[
        ["station_id", "road_2023_segment_index"]
    ].drop_duplicates("station_id")

    segment_frames = []
    station_frames = []
    source_frames = []
    coverage_rows = []
    for year in years:
        crosswalk, source = obtain_snapshot(base, year, centerline, road_geometries)
        station_year, mapped_share, station_osm_share = station_support_for_year(
            year, crosswalk, measured, station_matches
        )
        segment_frames.append(crosswalk)
        station_frames.append(station_year)
        source_frames.append(source)
        coverage_rows.append(snapshot_coverage(crosswalk, mapped_share, station_osm_share))

    segment_year = pd.concat(segment_frames, ignore_index=True, sort=False)
    station_year = pd.concat(station_frames, ignore_index=True, sort=False)
    source = pd.concat(source_frames, ignore_index=True, sort=False)
    coverage = pd.DataFrame(coverage_rows)
    save_csv(segment_year, SEGMENT_YEAR_PATH)
    save_csv(station_year, STATION_YEAR_PATH)
    save_csv(source, SOURCE_AUDIT_PATH)
    save_csv(coverage, COVERAGE_PATH)

    _, churn = build_churn(segment_year)
    _, alignment = build_station_change_alignment(station_year)
    decision = decisions(pre2012, coverage, churn, alignment)
    plots(coverage, churn, alignment)
    paths = [
        PRE2012_PATH, SOURCE_AUDIT_PATH, COVERAGE_PATH, CHURN_PATH,
        ALIGNMENT_PATH, DECISION_PATH, SEGMENT_YEAR_PATH, STATION_YEAR_PATH,
        STATION_PAIR_PATH, COVERAGE_FIGURE_PATH, CHURN_FIGURE_PATH,
    ]
    update_manifest(paths)

    print("\nStep 23B-data is complete.")
    print("  2011 remains structurally unavailable through the public Overpass attic.")
    print("  2016 is supplemental; 2018-2024 is the primary label-feature overlap window.")
    print("  Decision: this audit does not authorise or claim historical segment-level backcasting.")


if __name__ == "__main__":
    main()
