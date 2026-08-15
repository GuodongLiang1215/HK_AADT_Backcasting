from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

STATION_ID_PATTERN = re.compile(r"^\d{4,5}$")
INTEGER_PATTERN = re.compile(r"^\d{1,3}(?:,\d{3})*$")
PERCENT_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class YearSpec:
    year: int
    first_page: int
    last_page: int
    report_name: str
    reported_station_total: int
    reported_surveyed_total: int
    bands: dict[str, tuple[float, float]]

    @property
    def previous_year(self) -> int:
        return self.year - 1


SPECS = (
    YearSpec(
        year=2011,
        first_page=47,
        last_page=96,
        report_name="ATC_2011.pdf",
        reported_station_total=1649,
        reported_surveyed_total=844,
        bands={
            "station_id": (60.0, 100.0),
            "station_type": (100.0, 125.0),
            "road_type": (125.0, 150.0),
            "road_name": (150.0, 220.0),
            "road_from": (220.0, 295.0),
            "road_to": (295.0, 375.0),
            "aadt_previous": (375.0, 425.0),
            "aadt_current": (425.0, 480.0),
            "reported_change": (480.0, 540.0),
        },
    ),
    YearSpec(
        year=2016,
        first_page=47,
        last_page=107,
        report_name="ATC_2016.pdf",
        reported_station_total=1651,
        reported_surveyed_total=846,
        bands={
            "station_id": (50.0, 90.0),
            "station_type": (90.0, 115.0),
            "road_type": (115.0, 140.0),
            "road_name": (140.0, 210.0),
            "road_from": (210.0, 285.0),
            "road_to": (285.0, 360.0),
            "aadt_previous": (360.0, 410.0),
            "aadt_current": (410.0, 465.0),
            "reported_change": (465.0, 525.0),
        },
    ),
    YearSpec(
        year=2021,
        first_page=50,
        last_page=106,
        report_name="ATC_2021.pdf",
        reported_station_total=1678,
        reported_surveyed_total=873,
        bands={
            "station_id": (50.0, 90.0),
            "station_type": (90.0, 115.0),
            "road_type": (115.0, 140.0),
            "road_name": (140.0, 210.0),
            "road_from": (210.0, 285.0),
            "road_to": (285.0, 360.0),
            "aadt_previous": (360.0, 410.0),
            "aadt_current": (410.0, 465.0),
            "reported_change": (465.0, 525.0),
        },
    ),
)


OUTPUT_FIELDS = [
    "year",
    "previous_year",
    "station_id",
    "station_type",
    "road_type",
    "road_name",
    "road_from",
    "road_to",
    "road_segment_text",
    "aadt_previous",
    "aadt_current",
    "previous_aadt_estimated",
    "current_aadt_estimated",
    "reported_change_pct",
    "calculated_change_pct",
    "change_check",
    "primary_label_eligible",
    "parse_status",
    "review_reason",
    "source_pdf",
    "source_page",
    "row_text_raw",
]


def word_x0(word: dict[str, object]) -> float:
    return float(word["x0"])


def word_top(word: dict[str, object]) -> float:
    return float(word["top"])


def words_in_band(
    words: list[dict[str, object]], band: tuple[float, float]
) -> list[dict[str, object]]:
    lower, upper = band
    return [word for word in words if lower <= word_x0(word) < upper]


def join_words(words: list[dict[str, object]]) -> str:
    ordered = sorted(words, key=lambda word: (round(word_top(word), 1), word_x0(word)))
    return " ".join(str(word["text"]) for word in ordered).strip()


def first_text(words: list[dict[str, object]]) -> str:
    return str(min(words, key=word_x0)["text"]) if words else ""


def parse_integer_band(
    words: list[dict[str, object]],
) -> tuple[int | None, bool, list[str]]:
    numeric_tokens = [
        str(word["text"])
        for word in words
        if INTEGER_PATTERN.fullmatch(str(word["text"]))
    ]
    estimated = any(str(word["text"]) == "*" for word in words)
    issues: list[str] = []

    if len(numeric_tokens) > 1:
        issues.append("multiple_numeric_tokens_in_aadt_column")
    value = int(numeric_tokens[0].replace(",", "")) if numeric_tokens else None

    if estimated and value is None:
        issues.append("estimated_marker_without_value")
    return value, estimated, issues


def parse_change_band(
    words: list[dict[str, object]],
) -> tuple[float | None, list[str]]:
    numeric_tokens = [
        str(word["text"])
        for word in words
        if PERCENT_PATTERN.fullmatch(str(word["text"]))
    ]
    issues: list[str] = []
    if len(numeric_tokens) > 1:
        issues.append("multiple_numeric_tokens_in_change_column")
    value = float(numeric_tokens[0]) if numeric_tokens else None
    return value, issues


def footer_top(words: list[dict[str, object]], page_height: float) -> float:
    candidates = [
        word_top(word)
        for word in words
        if str(word["text"]) == "AADT" and word_top(word) > page_height * 0.75
    ]
    return min(candidates) - 2.0 if candidates else page_height - 60.0


def extract_page_records(
    page: pdfplumber.page.Page,
    spec: YearSpec,
    page_number: int,
) -> list[dict[str, object]]:
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    words = [word for word in words if word_top(word) < footer_top(words, page.height)]

    station_band = spec.bands["station_id"]
    station_starts = [
        word
        for word in words
        if station_band[0] <= word_x0(word) < station_band[1]
        and STATION_ID_PATTERN.fullmatch(str(word["text"]))
    ]
    station_starts.sort(key=word_top)

    records: list[dict[str, object]] = []
    for index, station_word in enumerate(station_starts):
        start_top = word_top(station_word) - 1.0
        end_top = (
            word_top(station_starts[index + 1]) - 1.0
            if index + 1 < len(station_starts)
            else footer_top(words, page.height)
        )
        block = [word for word in words if start_top <= word_top(word) < end_top]

        station_words = words_in_band(block, spec.bands["station_id"])
        station_type_words = words_in_band(block, spec.bands["station_type"])
        road_type_words = words_in_band(block, spec.bands["road_type"])
        road_name_words = words_in_band(block, spec.bands["road_name"])
        road_from_words = words_in_band(block, spec.bands["road_from"])
        road_to_words = words_in_band(block, spec.bands["road_to"])
        previous_words = words_in_band(block, spec.bands["aadt_previous"])
        current_words = words_in_band(block, spec.bands["aadt_current"])
        change_words = words_in_band(block, spec.bands["reported_change"])

        station_id = first_text(station_words)
        station_type = first_text(station_type_words)
        road_type = first_text(road_type_words)
        road_name = join_words(road_name_words)
        road_from = join_words(road_from_words)
        road_to = join_words(road_to_words)

        aadt_previous, previous_estimated, previous_issues = parse_integer_band(
            previous_words
        )
        aadt_current, current_estimated, current_issues = parse_integer_band(
            current_words
        )
        reported_change, change_issues = parse_change_band(change_words)

        review_reasons = previous_issues + current_issues + change_issues
        if not station_type:
            review_reasons.append("missing_station_type")
        if not road_type:
            review_reasons.append("missing_road_type")
        if not road_name or not road_from or not road_to:
            review_reasons.append("incomplete_road_description")
        if aadt_previous is None:
            review_reasons.append("missing_previous_aadt")
        if aadt_current is None:
            review_reasons.append("missing_current_aadt")

        calculated_change: float | None = None
        change_check = "not_checkable"
        if aadt_previous is not None and aadt_current is not None and aadt_previous > 0:
            calculated_change = round(
                (aadt_current - aadt_previous) / aadt_previous * 100,
                1,
            )
            if reported_change is None:
                review_reasons.append("missing_reported_change")
            else:
                previous_low = max(aadt_previous - 5, 1)
                previous_high = aadt_previous + 5
                current_low = max(aadt_current - 5, 0)
                current_high = aadt_current + 5
                possible_change_low = (
                    (current_low - previous_high) / previous_high * 100
                )
                possible_change_high = (
                    (current_high - previous_low) / previous_low * 100
                )
                if (
                    possible_change_low - 0.15
                    <= reported_change
                    <= possible_change_high + 0.15
                ):
                    change_check = "consistent_with_rounded_aadt"
                else:
                    change_check = "outside_rounded_aadt_range"
                    review_reasons.append("reported_change_inconsistent")
        elif reported_change is not None:
            review_reasons.append("reported_change_without_complete_aadt_pair")

        review_reasons = list(dict.fromkeys(review_reasons))
        road_segment_text = " | ".join(
            part for part in (road_name, road_from, road_to) if part
        )

        records.append(
            {
                "year": spec.year,
                "previous_year": spec.previous_year,
                "station_id": station_id,
                "station_type": station_type,
                "road_type": road_type,
                "road_name": road_name,
                "road_from": road_from,
                "road_to": road_to,
                "road_segment_text": road_segment_text,
                "aadt_previous": aadt_previous,
                "aadt_current": aadt_current,
                "previous_aadt_estimated": previous_estimated,
                "current_aadt_estimated": current_estimated,
                "reported_change_pct": reported_change,
                "calculated_change_pct": calculated_change,
                "change_check": change_check,
                "primary_label_eligible": (
                    aadt_current is not None and not current_estimated
                ),
                "parse_status": "review" if review_reasons else "ok",
                "review_reason": ";".join(review_reasons),
                "source_pdf": spec.report_name,
                "source_page": page_number,
                "row_text_raw": join_words(block),
            }
        )

    return records


def extract_report(spec: YearSpec) -> list[dict[str, object]]:
    report_path = REPORT_DIR / spec.report_name
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path}. Run src\\01_download_atc_reports.py first."
        )

    records: list[dict[str, object]] = []
    print(
        f"Extracting {spec.year} Appendix B, "
        f"PDF pages {spec.first_page}-{spec.last_page}"
    )
    with pdfplumber.open(report_path) as report:
        for page_number in range(spec.first_page, spec.last_page + 1):
            page = report.pages[page_number - 1]
            records.extend(extract_page_records(page, spec, page_number))

    print(f"Extracted {len(records)} station rows for {spec.year}")
    return records


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    audit_rows: list[dict[str, object]] = []
    for spec in SPECS:
        year_rows = [row for row in records if row["year"] == spec.year]
        station_ids = [str(row["station_id"]) for row in year_rows]
        audit_rows.append(
            {
                "year": spec.year,
                "appendix_b_pages": f"{spec.first_page}-{spec.last_page}",
                "reported_all_counting_stations": spec.reported_station_total,
                "rows_extracted": len(year_rows),
                "appendix_b_gap_vs_reported_total": (
                    spec.reported_station_total - len(year_rows)
                ),
                "unique_station_ids": len(set(station_ids)),
                "duplicate_station_ids": len(station_ids) - len(set(station_ids)),
                "current_values_present": sum(
                    row["aadt_current"] is not None for row in year_rows
                ),
                "current_values_estimated": sum(
                    bool(row["current_aadt_estimated"]) for row in year_rows
                ),
                "primary_measured_labels": sum(
                    bool(row["primary_label_eligible"]) for row in year_rows
                ),
                "reported_surveyed_stations": spec.reported_surveyed_total,
                "measured_label_gap_vs_reported_surveyed": (
                    spec.reported_surveyed_total
                    - sum(bool(row["primary_label_eligible"]) for row in year_rows)
                ),
                "previous_values_missing": sum(
                    row["aadt_previous"] is None for row in year_rows
                ),
                "current_values_missing": sum(
                    row["aadt_current"] is None for row in year_rows
                ),
                "rows_requiring_review": sum(
                    row["parse_status"] == "review" for row in year_rows
                ),
            }
        )
    return audit_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, object]] = []
    for spec in SPECS:
        all_records.extend(extract_report(spec))

    station_path = OUTPUT_DIR / "atc_appendix_b_station_year.csv"
    review_path = OUTPUT_DIR / "atc_appendix_b_review_rows.csv"
    audit_path = OUTPUT_DIR / "atc_appendix_b_extraction_audit.csv"

    write_csv(station_path, all_records, OUTPUT_FIELDS)
    review_rows = [row for row in all_records if row["parse_status"] == "review"]
    write_csv(review_path, review_rows, OUTPUT_FIELDS)

    audit_rows = build_audit(all_records)
    audit_fields = list(audit_rows[0].keys())
    write_csv(audit_path, audit_rows, audit_fields)

    print(f"Saved: {station_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {review_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {audit_path.relative_to(PROJECT_ROOT)}")
    print("\nPrimary modelling rule: use only primary_label_eligible == True.")


if __name__ == "__main__":
    main()
