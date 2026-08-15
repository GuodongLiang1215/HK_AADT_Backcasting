from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PANEL_ANCHOR_PATH = PROCESSED_DIR / "atc_three_year_panel_current_spatial_anchor.csv"
REVIEW_ANCHOR_PATH = PROCESSED_DIR / "atc_crosswalk_review_current_spatial_anchor.csv"

CORE_PANEL_PATH = PROCESSED_DIR / "atc_core_three_year_observed_panel.csv"
CORE_SPATIAL_PATH = PROCESSED_DIR / "atc_core_spatial_model_panel.csv"
CORE_LONG_PATH = PROCESSED_DIR / "atc_core_panel_long.csv"
CORE_CHANGE_PATH = PROCESSED_DIR / "atc_core_panel_change.csv"
CORE_GEOJSON_PATH = PROCESSED_DIR / "atc_core_spatial_model_panel.geojson"
REVIEW_PRIORITY_PATH = PROCESSED_DIR / "atc_manual_review_priority.csv"
BASELINE_SUMMARY_PATH = PROCESSED_DIR / "atc_core_panel_baseline_summary.csv"
AUDIT_PATH = PROCESSED_DIR / "atc_core_panel_audit.csv"

YEARS = (2011, 2016, 2021)
ADJACENT_PAIRS = ((2011, 2016), (2016, 2021))


def read_bool(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Complete Step 5 first."
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


def aadt(row: dict[str, str], year: int) -> int:
    return int(float(row[f"aadt_{year}"]))


def percent_change(start: int, end: int) -> float:
    return round((end / start - 1) * 100, 4)


def annualised_change(start: int, end: int, years: int) -> float:
    return round(((end / start) ** (1 / years) - 1) * 100, 4)


def freeze_core_panel(
    panel_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    core_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    for row in panel_rows:
        if not read_bool(row["recommended_three_year_observed_panel"]):
            continue
        frozen = {
            **row,
            "analysis_role": "primary_three_year_observed_panel",
            "historical_match_rule": "high_confidence_both_adjacent_pairs",
            "spatial_support_rule": (
                "current_anchor_available"
                if read_bool(row["current_point_present"])
                else "current_anchor_missing"
            ),
        }
        core_rows.append(frozen)
        if read_bool(row["current_point_present"]):
            spatial_rows.append(
                {
                    **frozen,
                    "analysis_role": "primary_spatial_model_panel",
                }
            )
    return core_rows, spatial_rows


def build_long_panel(core_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    long_rows: list[dict[str, object]] = []
    for row in core_rows:
        for year in YEARS:
            long_rows.append(
                {
                    "station_id": row["station_id"],
                    "year": year,
                    "aadt": aadt(row, year),
                    "station_type": row[f"station_type_{year}"],
                    "road_type": row[f"road_type_{year}"],
                    "segment_text": row[f"segment_text_{year}"],
                    "source_page": row[f"source_page_{year}"],
                    "current_point_present": row["current_point_present"],
                    "current_longitude": row["current_longitude"],
                    "current_latitude": row["current_latitude"],
                    "geometry_reference": row["geometry_reference"],
                    "historical_geometry_status": row["historical_geometry_status"],
                    "analysis_role": "primary_three_year_observed_panel",
                    "temporal_interpretation": (
                        "shock_sensitivity_year"
                        if year == 2021
                        else "historical_observed_anchor"
                    ),
                }
            )
    return long_rows


def build_change_table(core_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in core_rows:
        aadt_2011 = aadt(row, 2011)
        aadt_2016 = aadt(row, 2016)
        aadt_2021 = aadt(row, 2021)
        rows.append(
            {
                "station_id": row["station_id"],
                "aadt_2011": aadt_2011,
                "aadt_2016": aadt_2016,
                "aadt_2021": aadt_2021,
                "absolute_change_2011_2016": aadt_2016 - aadt_2011,
                "percent_change_2011_2016": percent_change(aadt_2011, aadt_2016),
                "annualised_percent_change_2011_2016": annualised_change(
                    aadt_2011, aadt_2016, 5
                ),
                "absolute_change_2016_2021": aadt_2021 - aadt_2016,
                "percent_change_2016_2021": percent_change(aadt_2016, aadt_2021),
                "annualised_percent_change_2016_2021": annualised_change(
                    aadt_2016, aadt_2021, 5
                ),
                "absolute_change_2011_2021": aadt_2021 - aadt_2011,
                "percent_change_2011_2021": percent_change(aadt_2011, aadt_2021),
                "annualised_percent_change_2011_2021": annualised_change(
                    aadt_2011, aadt_2021, 10
                ),
                "station_type": row["station_type_2011"],
                "road_type": row["road_type_2011"],
                "segment_text": row["segment_text_2011"],
                "current_point_present": row["current_point_present"],
                "current_longitude": row["current_longitude"],
                "current_latitude": row["current_latitude"],
                "historical_geometry_status": row["historical_geometry_status"],
                "interpretation_2016_2021": "contains_2021_shock_effect",
            }
        )
    return rows


def build_geojson(spatial_rows: list[dict[str, object]]) -> dict[str, object]:
    features: list[dict[str, object]] = []
    for row in spatial_rows:
        aadt_2011 = aadt(row, 2011)
        aadt_2016 = aadt(row, 2016)
        aadt_2021 = aadt(row, 2021)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "station_id": row["station_id"],
                    "aadt_2011": aadt_2011,
                    "aadt_2016": aadt_2016,
                    "aadt_2021": aadt_2021,
                    "pct_2011_2016": percent_change(aadt_2011, aadt_2016),
                    "pct_2016_2021": percent_change(aadt_2016, aadt_2021),
                    "road_type": row["road_type_2011"],
                    "segment_text": row["segment_text_2011"],
                    "geometry_reference": row["geometry_reference"],
                    "historical_geometry_status": row["historical_geometry_status"],
                    "analysis_role": "primary_spatial_model_panel",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(row["current_longitude"]),
                        float(row["current_latitude"]),
                    ],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "atc_core_spatial_model_panel",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def write_geojson(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def build_review_priority(
    panel_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    panel_lookup = {row["station_id"]: row for row in panel_rows}
    adjacent_by_station: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        pair = (int(row["year_from"]), int(row["year_to"]))
        if pair in ADJACENT_PAIRS:
            adjacent_by_station[row["station_id"]].append(row)

    priority_rows: list[dict[str, object]] = []
    for station_id in sorted(adjacent_by_station, key=int):
        panel = panel_lookup[station_id]
        adjacent = adjacent_by_station[station_id]
        current_point_present = any(
            read_bool(row["current_point_present"]) for row in adjacent
        )
        road_type_changed = any(
            row["road_type_from"]
            and row["road_type_to"]
            and row["road_type_from"] != row["road_type_to"]
            for row in adjacent
        )
        confidences = {row["physical_match_confidence"] for row in adjacent}
        all_three_present = read_bool(panel["all_three_present"])
        all_three_measured = read_bool(panel["all_three_measured"])
        potential_core_recovery = (
            all_three_present
            and all_three_measured
            and not read_bool(panel["recommended_three_year_observed_panel"])
        )

        if potential_core_recovery:
            if road_type_changed or not current_point_present:
                priority_group = "P3_archival_evidence_required"
                priority_reason = (
                    "road_type_changed_or_current_anchor_missing;do_not_upgrade_from_text"
                )
            elif "low" in confidences:
                priority_group = "P2_material_change_review"
                priority_reason = "material_description_change;historical_map_needed"
            else:
                priority_group = "P1_plausible_recovery_review"
                priority_reason = "medium_description_change;current_anchor_available"
        else:
            priority_group = "P4_defer_not_core_recovery"
            priority_reason = "missing_year_or_non_measured_label"

        point_source = next(
            (row for row in adjacent if read_bool(row["current_point_present"])),
            adjacent[0],
        )
        adjacent_reasons = sorted(
            {
                reason
                for row in adjacent
                for reason in row["review_reason"].split(";")
                if reason
            }
        )
        priority_rows.append(
            {
                "station_id": station_id,
                "priority_group": priority_group,
                "priority_reason": priority_reason,
                "potential_core_recovery": potential_core_recovery,
                "retain_in_primary_panel": False,
                "all_three_present": all_three_present,
                "all_three_measured": all_three_measured,
                "confidence_2011_2016": panel["confidence_2011_2016"],
                "confidence_2016_2021": panel["confidence_2016_2021"],
                "confidence_2011_2021_diagnostic": panel["confidence_2011_2021"],
                "road_type_changed_adjacent": road_type_changed,
                "adjacent_review_reasons": ";".join(adjacent_reasons),
                "segment_text_2011": panel["segment_text_2011"],
                "segment_text_2016": panel["segment_text_2016"],
                "segment_text_2021": panel["segment_text_2021"],
                "road_type_2011": panel["road_type_2011"],
                "road_type_2016": panel["road_type_2016"],
                "road_type_2021": panel["road_type_2021"],
                "current_point_present": current_point_present,
                "current_longitude": point_source["current_longitude"],
                "current_latitude": point_source["current_latitude"],
                "historical_geometry_status": "not_proven",
                "manual_review_decision": "pending",
                "historical_evidence_source": "",
                "manual_review_notes": "",
            }
        )
    return priority_rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_baseline_summary(
    core_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for year in YEARS:
        values = [float(aadt(row, year)) for row in core_rows]
        summary.append(
            {
                "summary_type": "aadt_distribution",
                "period": str(year),
                "n": len(values),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "p10": round(percentile(values, 0.10), 2),
                "p90": round(percentile(values, 0.90), 2),
                "share_increase": "",
                "share_decrease": "",
                "interpretation": (
                    "shock_sensitivity_year_not_ordinary_trend_endpoint"
                    if year == 2021
                    else "balanced_core_panel_distribution"
                ),
            }
        )

    for year_from, year_to in ((2011, 2016), (2016, 2021), (2011, 2021)):
        values = [
            (aadt(row, year_to) / aadt(row, year_from) - 1) * 100
            for row in core_rows
        ]
        summary.append(
            {
                "summary_type": "station_percent_change",
                "period": f"{year_from}-{year_to}",
                "n": len(values),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "p10": round(percentile(values, 0.10), 2),
                "p90": round(percentile(values, 0.90), 2),
                "share_increase": round(sum(value > 0 for value in values) / len(values), 4),
                "share_decrease": round(sum(value < 0 for value in values) / len(values), 4),
                "interpretation": (
                    "contains_2021_shock_effect"
                    if year_to == 2021
                    else "pre_2021_change_on_balanced_core_panel"
                ),
            }
        )
    return summary


def build_audit(
    panel_rows: list[dict[str, str]],
    core_rows: list[dict[str, object]],
    spatial_rows: list[dict[str, object]],
    priority_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    counts = defaultdict(int)
    for row in priority_rows:
        counts[str(row["priority_group"])] += 1
    return [
        {
            "metric": "all_three_present",
            "count": sum(read_bool(row["all_three_present"]) for row in panel_rows),
            "decision": "candidate_balanced_station_set",
        },
        {
            "metric": "all_three_measured",
            "count": sum(read_bool(row["all_three_measured"]) for row in panel_rows),
            "decision": "measured_labels_before_physical_stability_filter",
        },
        {
            "metric": "primary_core_panel",
            "count": len(core_rows),
            "decision": "freeze_for_primary_longitudinal_analysis",
        },
        {
            "metric": "primary_spatial_model_panel",
            "count": len(spatial_rows),
            "decision": "core_panel_with_current_point_anchor",
        },
        {
            "metric": "potential_core_recovery",
            "count": sum(bool(row["potential_core_recovery"]) for row in priority_rows),
            "decision": "sensitivity_only_until_manual_evidence",
        },
        {
            "metric": "P1_plausible_recovery_review",
            "count": counts["P1_plausible_recovery_review"],
            "decision": "review_first",
        },
        {
            "metric": "P2_material_change_review",
            "count": counts["P2_material_change_review"],
            "decision": "historical_map_required",
        },
        {
            "metric": "P3_archival_evidence_required",
            "count": counts["P3_archival_evidence_required"],
            "decision": "do_not_upgrade_from_current_geometry",
        },
        {
            "metric": "P4_defer_not_core_recovery",
            "count": counts["P4_defer_not_core_recovery"],
            "decision": "not_blocking_primary_panel",
        },
    ]


def main() -> None:
    panel_rows = read_csv(PANEL_ANCHOR_PATH)
    review_rows = read_csv(REVIEW_ANCHOR_PATH)

    core_rows, spatial_rows = freeze_core_panel(panel_rows)
    long_rows = build_long_panel(core_rows)
    change_rows = build_change_table(core_rows)
    priority_rows = build_review_priority(panel_rows, review_rows)
    baseline_summary = build_baseline_summary(core_rows)
    audit_rows = build_audit(panel_rows, core_rows, spatial_rows, priority_rows)

    write_csv(CORE_PANEL_PATH, core_rows)
    write_csv(CORE_SPATIAL_PATH, spatial_rows)
    write_csv(CORE_LONG_PATH, long_rows)
    write_csv(CORE_CHANGE_PATH, change_rows)
    write_geojson(CORE_GEOJSON_PATH, build_geojson(spatial_rows))
    write_csv(REVIEW_PRIORITY_PATH, priority_rows)
    write_csv(BASELINE_SUMMARY_PATH, baseline_summary)
    write_csv(AUDIT_PATH, audit_rows)

    print("\nCore panel is frozen.")
    print(f"Primary three-year observed panel: {len(core_rows)} stations")
    print(f"Primary spatial panel: {len(spatial_rows)} stations")
    print(
        "Manual review queue: "
        f"{sum(row['priority_group'] == 'P1_plausible_recovery_review' for row in priority_rows)} P1, "
        f"{sum(row['priority_group'] == 'P2_material_change_review' for row in priority_rows)} P2, "
        f"{sum(row['priority_group'] == 'P3_archival_evidence_required' for row in priority_rows)} P3."
    )
    print("The 2011-2021 comparison remains diagnostic and is not reviewed twice.")


if __name__ == "__main__":
    main()
