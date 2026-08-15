from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "atc_appendix_b_station_year.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

YEARS = (2011, 2016, 2021)
PAIR_YEARS = ((2011, 2016), (2016, 2021), (2011, 2021))

HIGH_NAME_THRESHOLD = 0.90
HIGH_ENDPOINT_THRESHOLD = 0.85


PAIRWISE_FIELDS = [
    "year_from",
    "year_to",
    "station_id",
    "station_type_from",
    "station_type_to",
    "station_type_same",
    "road_type_from",
    "road_type_to",
    "road_type_same",
    "road_name_from",
    "road_name_to",
    "road_from_from",
    "road_from_to",
    "road_to_from",
    "road_to_to",
    "segment_text_from",
    "segment_text_to",
    "road_name_similarity",
    "endpoint_similarity_direct",
    "endpoint_similarity_reversed",
    "endpoint_similarity_used",
    "orientation_relation",
    "physical_match_confidence",
    "confidence_reason",
    "geometry_required",
    "label_eligible_from",
    "label_eligible_to",
    "both_labels_measured",
    "aadt_from",
    "aadt_to",
    "recommended_observed_pair",
    "source_page_from",
    "source_page_to",
]

REVIEW_FIELDS = [
    "review_type",
    "year_from",
    "year_to",
    "station_id",
    "present_from",
    "present_to",
    "physical_match_confidence",
    "review_reason",
    "segment_text_from",
    "segment_text_to",
    "road_type_from",
    "road_type_to",
    "source_page_from",
    "source_page_to",
]


def read_bool(value: str) -> bool:
    return value.strip().casefold() == "true"


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

    sequence_score = SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return 0.65 * sequence_score + 0.35 * token_score


def assign_confidence(road_name_score: float, endpoint_score: float) -> str:
    if (
        road_name_score >= HIGH_NAME_THRESHOLD
        and endpoint_score >= HIGH_ENDPOINT_THRESHOLD
    ):
        return "high"
    if (
        (road_name_score >= 0.90 and endpoint_score >= 0.60)
        or (road_name_score >= 0.70 and endpoint_score >= 0.75)
        or (road_name_score >= 0.50 and endpoint_score >= 0.90)
    ):
        return "medium"
    return "low"


def orientation_relation(direct_score: float, reversed_score: float) -> str:
    if reversed_score >= 0.85 and reversed_score > direct_score + 0.10:
        return "reversed"
    if direct_score >= 0.85:
        return "direct"
    if reversed_score > direct_score + 0.10:
        return "changed_or_possible_reversal"
    return "changed_or_ambiguous"


def confidence_reason(
    confidence: str,
    orientation: str,
    station_type_same: bool,
    road_type_same: bool,
) -> str:
    if confidence == "high":
        reasons = ["road_name_and_endpoints_consistent"]
    elif confidence == "medium":
        reasons = ["partial_description_change"]
    else:
        reasons = ["material_description_change"]

    if orientation == "reversed":
        reasons.append("endpoints_reversed")
    elif orientation.startswith("changed_or"):
        reasons.append("endpoint_orientation_uncertain")
    if not station_type_same:
        reasons.append("station_type_changed")
    if not road_type_same:
        reasons.append("road_type_changed")
    return ";".join(reasons)


def build_pair_record(
    year_from: int,
    year_to: int,
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, object]:
    road_name_score = text_similarity(left["road_name"], right["road_name"])
    direct_score = (
        text_similarity(left["road_from"], right["road_from"])
        + text_similarity(left["road_to"], right["road_to"])
    ) / 2
    reversed_score = (
        text_similarity(left["road_from"], right["road_to"])
        + text_similarity(left["road_to"], right["road_from"])
    ) / 2
    endpoint_score = max(direct_score, reversed_score)
    orientation = orientation_relation(direct_score, reversed_score)
    confidence = assign_confidence(road_name_score, endpoint_score)
    station_type_same = left["station_type"] == right["station_type"]
    road_type_same = left["road_type"] == right["road_type"]
    label_eligible_from = read_bool(left["primary_label_eligible"])
    label_eligible_to = read_bool(right["primary_label_eligible"])
    both_labels_measured = label_eligible_from and label_eligible_to
    geometry_required = confidence != "high" or not road_type_same
    recommended_observed_pair = (
        confidence == "high"
        and station_type_same
        and road_type_same
        and both_labels_measured
    )

    return {
        "year_from": year_from,
        "year_to": year_to,
        "station_id": left["station_id"],
        "station_type_from": left["station_type"],
        "station_type_to": right["station_type"],
        "station_type_same": station_type_same,
        "road_type_from": left["road_type"],
        "road_type_to": right["road_type"],
        "road_type_same": road_type_same,
        "road_name_from": left["road_name"],
        "road_name_to": right["road_name"],
        "road_from_from": left["road_from"],
        "road_from_to": right["road_from"],
        "road_to_from": left["road_to"],
        "road_to_to": right["road_to"],
        "segment_text_from": left["road_segment_text"],
        "segment_text_to": right["road_segment_text"],
        "road_name_similarity": round(road_name_score, 4),
        "endpoint_similarity_direct": round(direct_score, 4),
        "endpoint_similarity_reversed": round(reversed_score, 4),
        "endpoint_similarity_used": round(endpoint_score, 4),
        "orientation_relation": orientation,
        "physical_match_confidence": confidence,
        "confidence_reason": confidence_reason(
            confidence,
            orientation,
            station_type_same,
            road_type_same,
        ),
        "geometry_required": geometry_required,
        "label_eligible_from": label_eligible_from,
        "label_eligible_to": label_eligible_to,
        "both_labels_measured": both_labels_measured,
        "aadt_from": left["aadt_current"],
        "aadt_to": right["aadt_current"],
        "recommended_observed_pair": recommended_observed_pair,
        "source_page_from": left["source_page"],
        "source_page_to": right["source_page"],
    }


def load_station_year() -> tuple[
    list[dict[str, str]],
    dict[int, dict[str, dict[str, str]]],
]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run src\\03_extract_appendix_b.py first."
        )

    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as source_file:
        rows = list(csv.DictReader(source_file))

    duplicate_keys = [
        key
        for key, count in Counter(
            (int(row["year"]), row["station_id"]) for row in rows
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"Duplicate year-station rows: {duplicate_keys[:10]}")

    by_year = {
        year: {
            row["station_id"]: row
            for row in rows
            if int(row["year"]) == year
        }
        for year in YEARS
    }
    return rows, by_year


def build_pairwise(
    by_year: dict[int, dict[str, dict[str, str]]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[int, int, str], dict[str, object]],
]:
    pairwise_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    pair_lookup: dict[tuple[int, int, str], dict[str, object]] = {}

    for year_from, year_to in PAIR_YEARS:
        ids_from = set(by_year[year_from])
        ids_to = set(by_year[year_to])

        for station_id in sorted(ids_from & ids_to, key=int):
            pair_row = build_pair_record(
                year_from,
                year_to,
                by_year[year_from][station_id],
                by_year[year_to][station_id],
            )
            pairwise_rows.append(pair_row)
            pair_lookup[(year_from, year_to, station_id)] = pair_row

            if bool(pair_row["geometry_required"]) or not bool(
                pair_row["station_type_same"]
            ):
                review_rows.append(
                    {
                        "review_type": "same_id_description_change",
                        "year_from": year_from,
                        "year_to": year_to,
                        "station_id": station_id,
                        "present_from": True,
                        "present_to": True,
                        "physical_match_confidence": pair_row[
                            "physical_match_confidence"
                        ],
                        "review_reason": pair_row["confidence_reason"],
                        "segment_text_from": pair_row["segment_text_from"],
                        "segment_text_to": pair_row["segment_text_to"],
                        "road_type_from": pair_row["road_type_from"],
                        "road_type_to": pair_row["road_type_to"],
                        "source_page_from": pair_row["source_page_from"],
                        "source_page_to": pair_row["source_page_to"],
                    }
                )

        for station_id in sorted(ids_from ^ ids_to, key=int):
            left = by_year[year_from].get(station_id)
            right = by_year[year_to].get(station_id)
            review_rows.append(
                {
                    "review_type": "id_not_present_in_both_years",
                    "year_from": year_from,
                    "year_to": year_to,
                    "station_id": station_id,
                    "present_from": left is not None,
                    "present_to": right is not None,
                    "physical_match_confidence": "unmatched",
                    "review_reason": "requires_geometry_before_cross_id_matching",
                    "segment_text_from": left["road_segment_text"] if left else "",
                    "segment_text_to": right["road_segment_text"] if right else "",
                    "road_type_from": left["road_type"] if left else "",
                    "road_type_to": right["road_type"] if right else "",
                    "source_page_from": left["source_page"] if left else "",
                    "source_page_to": right["source_page"] if right else "",
                }
            )

    return pairwise_rows, review_rows, pair_lookup


def build_three_year(
    by_year: dict[int, dict[str, dict[str, str]]],
    pair_lookup: dict[tuple[int, int, str], dict[str, object]],
) -> list[dict[str, object]]:
    all_station_ids = sorted(
        set().union(*(set(by_year[year]) for year in YEARS)),
        key=int,
    )
    rows: list[dict[str, object]] = []

    for station_id in all_station_ids:
        year_rows = {year: by_year[year].get(station_id) for year in YEARS}
        adjacent_pairs = [
            pair_lookup.get((2011, 2016, station_id)),
            pair_lookup.get((2016, 2021, station_id)),
        ]
        all_three_present = all(year_rows[year] is not None for year in YEARS)
        stable_physical_segment = (
            all_three_present
            and all(pair is not None for pair in adjacent_pairs)
            and all(
                pair["physical_match_confidence"] == "high"
                and bool(pair["road_type_same"])
                for pair in adjacent_pairs
                if pair is not None
            )
        )
        all_three_measured = all_three_present and all(
            read_bool(year_rows[year]["primary_label_eligible"])
            for year in YEARS
            if year_rows[year] is not None
        )
        recommended_panel = stable_physical_segment and all_three_measured

        review_reasons: list[str] = []
        if not all_three_present:
            review_reasons.append("not_present_in_all_three_years")
        elif not stable_physical_segment:
            review_reasons.append("adjacent_pair_not_stable_high_confidence")
        if all_three_present and not all_three_measured:
            review_reasons.append("not_measured_in_all_three_years")

        output_row: dict[str, object] = {
            "station_id": station_id,
            "all_three_present": all_three_present,
            "confidence_2011_2016": (
                adjacent_pairs[0]["physical_match_confidence"]
                if adjacent_pairs[0]
                else "unmatched"
            ),
            "confidence_2016_2021": (
                adjacent_pairs[1]["physical_match_confidence"]
                if adjacent_pairs[1]
                else "unmatched"
            ),
            "confidence_2011_2021": (
                pair_lookup[(2011, 2021, station_id)]["physical_match_confidence"]
                if (2011, 2021, station_id) in pair_lookup
                else "unmatched"
            ),
            "stable_physical_segment": stable_physical_segment,
            "all_three_measured": all_three_measured,
            "recommended_three_year_observed_panel": recommended_panel,
            "three_year_review_reason": ";".join(review_reasons),
        }
        for year in YEARS:
            source = year_rows[year]
            output_row.update(
                {
                    f"present_{year}": source is not None,
                    f"station_type_{year}": source["station_type"] if source else "",
                    f"road_type_{year}": source["road_type"] if source else "",
                    f"segment_text_{year}": (
                        source["road_segment_text"] if source else ""
                    ),
                    f"aadt_{year}": source["aadt_current"] if source else "",
                    f"label_eligible_{year}": (
                        read_bool(source["primary_label_eligible"])
                        if source
                        else False
                    ),
                    f"source_page_{year}": source["source_page"] if source else "",
                }
            )
        rows.append(output_row)

    return rows


def build_audit(
    by_year: dict[int, dict[str, dict[str, str]]],
    pairwise_rows: list[dict[str, object]],
    three_year_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    audit_rows: list[dict[str, object]] = []
    for year_from, year_to in PAIR_YEARS:
        relevant = [
            row
            for row in pairwise_rows
            if row["year_from"] == year_from and row["year_to"] == year_to
        ]
        counts = Counter(str(row["physical_match_confidence"]) for row in relevant)
        ids_from = set(by_year[year_from])
        ids_to = set(by_year[year_to])
        audit_rows.append(
            {
                "scope": "pairwise",
                "comparison": f"{year_from}-{year_to}",
                "source_ids": len(ids_from),
                "target_ids": len(ids_to),
                "common_ids": len(relevant),
                "only_source_ids": len(ids_from - ids_to),
                "only_target_ids": len(ids_to - ids_from),
                "high_matches": counts["high"],
                "medium_matches": counts["medium"],
                "low_matches": counts["low"],
                "station_type_changes": sum(
                    not bool(row["station_type_same"]) for row in relevant
                ),
                "road_type_changes": sum(
                    not bool(row["road_type_same"]) for row in relevant
                ),
                "reversed_endpoints": sum(
                    row["orientation_relation"] == "reversed" for row in relevant
                ),
                "recommended_observed_pairs": sum(
                    bool(row["recommended_observed_pair"]) for row in relevant
                ),
                "all_three_present": "",
                "stable_three_year_segments": "",
                "all_three_measured": "",
                "recommended_three_year_observed_panel": "",
            }
        )

    audit_rows.append(
        {
            "scope": "three_year",
            "comparison": "2011-2016-2021",
            "source_ids": "",
            "target_ids": "",
            "common_ids": "",
            "only_source_ids": "",
            "only_target_ids": "",
            "high_matches": "",
            "medium_matches": "",
            "low_matches": "",
            "station_type_changes": "",
            "road_type_changes": "",
            "reversed_endpoints": "",
            "recommended_observed_pairs": "",
            "all_three_present": sum(
                bool(row["all_three_present"]) for row in three_year_rows
            ),
            "stable_three_year_segments": sum(
                bool(row["stable_physical_segment"]) for row in three_year_rows
            ),
            "all_three_measured": sum(
                bool(row["all_three_measured"]) for row in three_year_rows
            ),
            "recommended_three_year_observed_panel": sum(
                bool(row["recommended_three_year_observed_panel"])
                for row in three_year_rows
            ),
        }
    )
    return audit_rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, by_year = load_station_year()
    pairwise_rows, review_rows, pair_lookup = build_pairwise(by_year)
    three_year_rows = build_three_year(by_year, pair_lookup)
    audit_rows = build_audit(by_year, pairwise_rows, three_year_rows)

    pairwise_path = OUTPUT_DIR / "atc_station_crosswalk_pairwise.csv"
    three_year_path = OUTPUT_DIR / "atc_station_crosswalk_three_year.csv"
    review_path = OUTPUT_DIR / "atc_station_crosswalk_review.csv"
    audit_path = OUTPUT_DIR / "atc_station_crosswalk_audit.csv"

    three_year_fields = list(three_year_rows[0].keys())
    audit_fields = list(audit_rows[0].keys())
    write_csv(pairwise_path, pairwise_rows, PAIRWISE_FIELDS)
    write_csv(three_year_path, three_year_rows, three_year_fields)
    write_csv(review_path, review_rows, REVIEW_FIELDS)
    write_csv(audit_path, audit_rows, audit_fields)

    print(f"Saved: {pairwise_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {three_year_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {review_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {audit_path.relative_to(PROJECT_ROOT)}")
    print(
        "\nPrimary longitudinal rule: use only "
        "recommended_observed_pair == True or "
        "recommended_three_year_observed_panel == True."
    )


if __name__ == "__main__":
    main()
