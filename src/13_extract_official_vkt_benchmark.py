"""Step 13: extract the official Annual Traffic Census vehicle-kilometrage benchmark.

The Annual Traffic Census publishes, in section 3.4 of every report, the average
daily vehicle-kilometrage on the roads covered by the census, split by region
(Hong Kong Island, Kowloon, New Territories) and by road network (Major, Minor).

This is an official published aggregate constraint for the road support covered
by the census.  It is not an independent label set: Appendix K shows that the
published vehicle-kilometrage is itself calculated from ATC AADT and road length.
For that reason this step also parses the Appendix H road-network lengths.  VKT,
road length, and their ratio (the implied length-weighted mean AADT) must be kept
together whenever the model support is compared with the official support.

This step only reads the layout text produced in Step 2. It fits no model and
makes no prediction.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = PROJECT_ROOT / "data" / "interim" / "atc_text"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

OFFICIAL_VKT_PATH = PROCESSED_DIR / "atc_official_vehicle_kilometrage.csv"
OFFICIAL_LENGTH_PATH = PROCESSED_DIR / "atc_official_road_network_length.csv"
BENCHMARK_PATH = TABLE_DIR / "step13_official_vkt_benchmark.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step13_official_vkt_decision_audit.csv"

YEARS = (2011, 2016, 2021)
REGIONS = ("hong_kong_island", "kowloon", "new_territories")
NETWORKS = ("major", "minor")

TABLE_HEADER = re.compile(
    r"(\d{4})\s+and\s+(\d{4})\s+Average\s+Daily\s+Vehicle-kilometre",
    re.IGNORECASE,
)
APPENDIX_H_HEADER = re.compile(r"APPENDIX\s+H\b", re.IGNORECASE)
APPENDIX_I_HEADER = re.compile(r"APPENDIX\s+I\b", re.IGNORECASE)
# Official numbers are printed with a single space as the thousands separator.
SPACED_NUMBER = re.compile(r"\d{1,3}(?: \d{3})+")
SCAN_LINES = 60
# The published tables round each cell independently, so sub-totals can differ
# from major + minor by a vehicle-kilometre or two.
ROUNDING_TOLERANCE = 10


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def read_layout_text(year: int) -> list[str]:
    path = TEXT_DIR / f"ATC_{year}_layout.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Run src/02_extract_report_text.py first."
        )
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def region_from_label(label: str) -> str | None:
    lowered = label.casefold()
    if "hong kong island" in lowered:
        return "hong_kong_island"
    if "kowloon" in lowered:
        return "kowloon"
    if "new" in lowered and "territor" in lowered:
        return "new_territories"
    return None


def network_from_label(label: str) -> str | None:
    lowered = label.casefold()
    if "sub-total" in lowered or "subtotal" in lowered:
        return "subtotal"
    if re.fullmatch(r"total", lowered.strip()):
        return "territory_total"
    if "major" in lowered:
        return "major"
    if "minor" in lowered:
        return "minor"
    return None


def parse_vkt_table(year: int) -> list[dict[str, object]]:
    """Read the section 3.4 vehicle-kilometrage table for one census year."""
    lines = read_layout_text(year)
    header_index = None
    previous_year = None
    for index, line in enumerate(lines):
        match = TABLE_HEADER.search(line)
        if match:
            header_index = index
            previous_year = int(match.group(1))
            reported_year = int(match.group(2))
            if reported_year != year:
                raise ValueError(
                    f"ATC {year} report: vehicle-kilometrage table reports {reported_year}."
                )
            break
    if header_index is None:
        raise ValueError(
            f"ATC {year} report: no 'Average Daily Vehicle-kilometre' table header found."
        )

    rows: list[dict[str, object]] = []
    current_region: str | None = None
    for line in lines[header_index + 1 : header_index + 1 + SCAN_LINES]:
        numbers = SPACED_NUMBER.findall(line)
        if len(numbers) != 2:
            continue
        label = line[: line.index(numbers[0])].strip()
        region = region_from_label(label)
        if region is not None:
            current_region = region
        network = network_from_label(label)
        if network is None:
            continue
        previous_value = int(numbers[0].replace(" ", ""))
        current_value = int(numbers[1].replace(" ", ""))
        rows.append(
            {
                "census_year": year,
                "previous_year": previous_year,
                "region": current_region if network != "territory_total" else "territory",
                "road_network": network,
                "previous_year_daily_vehicle_km": previous_value,
                "census_year_daily_vehicle_km": current_value,
                "source_pdf": f"ATC_{year}.pdf",
                "source_section": "3.4_vehicle_kilometrage",
                "measurement_scope": "roads_covered_in_the_annual_traffic_census",
            }
        )
        if network == "territory_total":
            break
    return rows


def parse_road_network_lengths(year: int) -> list[dict[str, object]]:
    """Parse Appendix H trafficable lengths by region and network.

    The layout extraction occasionally separates a printed ``Sub-total`` from
    its value and the 2011 minor subtotal has an OCR inconsistency.  Major and
    total-covered lengths are stable in all three reports, so the minor length
    is derived as total minus major and recorded explicitly as such.
    """
    text = "\n".join(read_layout_text(year))
    appendix_match = APPENDIX_H_HEADER.search(text)
    if appendix_match is None:
        raise ValueError(f"ATC {year}: Appendix H road-network section not found.")
    appendix_end_match = APPENDIX_I_HEADER.search(text, appendix_match.end())
    appendix_end = appendix_end_match.start() if appendix_end_match else len(text)
    appendix = text[appendix_match.start() : appendix_end]

    region_patterns = (
        ("hong_kong_island", re.compile(r"Hong Kong Island\s*:", re.IGNORECASE)),
        (
            "kowloon",
            re.compile(r"Kowloon(?: and New Kowloon)?\s*:", re.IGNORECASE),
        ),
        ("new_territories", re.compile(r"New Territories\s*:", re.IGNORECASE)),
    )
    matches = []
    for region, pattern in region_patterns:
        match = pattern.search(appendix)
        if match is None:
            raise ValueError(f"ATC {year}: Appendix H region {region} not found.")
        matches.append((region, match.start()))
    matches.sort(key=lambda item: item[1])

    rows: list[dict[str, object]] = []
    for index, (region, start) in enumerate(matches):
        end = matches[index + 1][1] if index + 1 < len(matches) else len(appendix)
        block = appendix[start:end]
        major_match = re.search(
            r"Major(?:(?!Minor).)*?Sub-total\s*([0-9]+(?:\.[0-9]+)?)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        total_match = re.search(
            r"Total Covered by Census\s*([0-9]+(?:\.[0-9]+)?)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if major_match is None or total_match is None:
            raise ValueError(
                f"ATC {year}: could not parse Appendix H lengths for {region}."
            )
        major = float(major_match.group(1))
        total = float(total_match.group(1))
        minor = total - major
        for network, length, derivation in (
            ("major", major, "published_appendix_h_subtotal"),
            ("minor", minor, "derived_total_minus_major"),
            ("subtotal", total, "published_total_covered_by_census"),
        ):
            rows.append(
                {
                    "census_year": year,
                    "region": region,
                    "road_network": network,
                    "trafficable_length_km": round(length, 4),
                    "length_derivation": derivation,
                    "source_pdf": f"ATC_{year}.pdf",
                    "source_appendix": "H_road_network",
                    "measurement_scope": "roads_covered_in_the_annual_traffic_census",
                }
            )
    return rows


def validate_year_rows(year: int, rows: list[dict[str, object]]) -> int:
    """Internal arithmetic check: major + minor must reproduce each sub-total.

    The published tables carry their own rounding, so an exact match is not
    required. A discrepancy larger than ROUNDING_TOLERANCE means the parser has
    picked up the wrong lines and must fail loudly.
    """
    by_key = {(row["region"], row["road_network"]): row for row in rows}
    territory_total = by_key.get(("territory", "territory_total"))
    if territory_total is None:
        raise ValueError(f"ATC {year}: territory total row not parsed.")

    largest_discrepancy = 0
    subtotal_sum = 0
    for region in REGIONS:
        parts = [by_key.get((region, network)) for network in NETWORKS]
        if any(part is None for part in parts):
            raise ValueError(f"ATC {year}: missing major/minor rows for {region}.")
        expected = sum(int(part["census_year_daily_vehicle_km"]) for part in parts)
        subtotal = by_key.get((region, "subtotal"))
        if subtotal is None:
            raise ValueError(f"ATC {year}: missing sub-total row for {region}.")
        published = int(subtotal["census_year_daily_vehicle_km"])
        discrepancy = abs(published - expected)
        if discrepancy > ROUNDING_TOLERANCE:
            raise ValueError(
                f"ATC {year}: {region} sub-total does not equal major + minor "
                f"({published} vs {expected}). The table parser is misaligned."
            )
        largest_discrepancy = max(largest_discrepancy, discrepancy)
        subtotal_sum += published

    published_total = int(territory_total["census_year_daily_vehicle_km"])
    discrepancy = abs(published_total - subtotal_sum)
    if discrepancy > ROUNDING_TOLERANCE:
        raise ValueError(
            f"ATC {year}: territory total does not equal the sum of regional sub-totals "
            f"({published_total} vs {subtotal_sum}). The table parser is misaligned."
        )
    return max(largest_discrepancy, discrepancy)


def build_benchmark(
    all_rows: list[dict[str, object]],
    length_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """One row per year with VKT, support length, and implied mean AADT."""
    lookup = {
        (int(row["census_year"]), row["region"], row["road_network"]): int(
            row["census_year_daily_vehicle_km"]
        )
        for row in all_rows
    }
    length_lookup = {
        (int(row["census_year"]), row["region"], row["road_network"]): float(
            row["trafficable_length_km"]
        )
        for row in length_rows
    }
    benchmark: list[dict[str, object]] = []
    for year in YEARS:
        major = sum(lookup[(year, region, "major")] for region in REGIONS)
        minor = sum(lookup[(year, region, "minor")] for region in REGIONS)
        total = lookup[(year, "territory", "territory_total")]
        major_length = sum(length_lookup[(year, region, "major")] for region in REGIONS)
        minor_length = sum(length_lookup[(year, region, "minor")] for region in REGIONS)
        total_length = sum(length_lookup[(year, region, "subtotal")] for region in REGIONS)
        row: dict[str, object] = {
            "census_year": year,
            "territory_total_daily_vehicle_km": total,
            "territory_major_daily_vehicle_km": major,
            "territory_minor_daily_vehicle_km": minor,
            "territory_total_road_length_km": round(total_length, 4),
            "territory_major_road_length_km": round(major_length, 4),
            "territory_minor_road_length_km": round(minor_length, 4),
            "territory_total_implied_mean_aadt": round(total / total_length, 3),
            "territory_major_implied_mean_aadt": round(major / major_length, 3),
            "territory_minor_implied_mean_aadt": round(minor / minor_length, 3),
            "major_share_pct": round(100 * major / total, 3),
        }
        for region in REGIONS:
            row[f"{region}_major_daily_vehicle_km"] = lookup[(year, region, "major")]
            row[f"{region}_minor_daily_vehicle_km"] = lookup[(year, region, "minor")]
            row[f"{region}_total_daily_vehicle_km"] = lookup[(year, region, "subtotal")]
            for network in ("major", "minor", "subtotal"):
                output_network = "total" if network == "subtotal" else network
                length = length_lookup[(year, region, network)]
                vkt_network = output_network if output_network != "total" else "subtotal"
                vkt = lookup[(year, region, vkt_network)]
                row[f"{region}_{output_network}_road_length_km"] = round(length, 4)
                row[f"{region}_{output_network}_implied_mean_aadt"] = round(
                    vkt / length, 3
                )
        benchmark.append(row)

    for index in range(1, len(benchmark)):
        earlier = benchmark[index - 1]
        later = benchmark[index]
        start = int(earlier["territory_total_daily_vehicle_km"])
        end = int(later["territory_total_daily_vehicle_km"])
        later["change_vs_previous_study_year_pct"] = round(100 * (end - start) / start, 3)
        start_major = int(earlier["territory_major_daily_vehicle_km"])
        end_major = int(later["territory_major_daily_vehicle_km"])
        later["major_change_vs_previous_study_year_pct"] = round(
            100 * (end_major - start_major) / start_major, 3
        )
        for prefix in ("territory_total", "territory_major", "territory_minor"):
            start_length = float(earlier[f"{prefix}_road_length_km"])
            end_length = float(later[f"{prefix}_road_length_km"])
            start_mean = float(earlier[f"{prefix}_implied_mean_aadt"])
            end_mean = float(later[f"{prefix}_implied_mean_aadt"])
            later[f"{prefix}_road_length_change_pct"] = round(
                100 * (end_length - start_length) / start_length, 3
            )
            later[f"{prefix}_implied_mean_aadt_change_pct"] = round(
                100 * (end_mean - start_mean) / start_mean, 3
            )
    benchmark[0]["change_vs_previous_study_year_pct"] = ""
    benchmark[0]["major_change_vs_previous_study_year_pct"] = ""
    for prefix in ("territory_total", "territory_major", "territory_minor"):
        benchmark[0][f"{prefix}_road_length_change_pct"] = ""
        benchmark[0][f"{prefix}_implied_mean_aadt_change_pct"] = ""
    return benchmark


def build_decision_audit(
    all_rows: list[dict[str, object]],
    length_rows: list[dict[str, object]],
    benchmark: list[dict[str, object]],
    largest_rounding_discrepancy: int,
) -> list[dict[str, object]]:
    by_year = {int(row["census_year"]): row for row in benchmark}
    audit: list[dict[str, object]] = [
        {
            "metric": "official_vkt_rows_parsed",
            "count": len(all_rows),
            "value": "",
            "decision": "ten_rows_per_year_three_regions_major_minor_subtotal_and_total",
        },
        {
            "metric": "internal_arithmetic_check_passed",
            "count": 1,
            "value": "major_plus_minor_equals_subtotal_and_subtotals_equal_total",
            "decision": "parser_is_verified_against_the_published_arithmetic",
        },
        {
            "metric": "largest_published_rounding_discrepancy_vehicle_km",
            "count": largest_rounding_discrepancy,
            "value": f"tolerance_{ROUNDING_TOLERANCE}",
            "decision": "official_table_rounds_each_cell_independently",
        },
        {
            "metric": "official_appendix_h_length_rows_parsed",
            "count": len(length_rows),
            "value": "three_regions_by_major_minor_and_total",
            "decision": "vkt_must_be_interpreted_with_the_road_support_length",
        },
    ]
    for year in YEARS:
        row = by_year[year]
        audit.append(
            {
                "metric": f"territory_total_daily_vehicle_km_{year}",
                "count": row["territory_total_daily_vehicle_km"],
                "value": row["territory_total_road_length_km"],
                "decision": "official_aggregate_constraint_pair_vkt_with_length",
            }
        )
        audit.append(
            {
                "metric": f"territory_major_daily_vehicle_km_{year}",
                "count": row["territory_major_daily_vehicle_km"],
                "value": row["territory_major_road_length_km"],
                "decision": "major_network_constraint_not_a_route_number_subset_target",
            }
        )
    audit.extend(
        [
            {
                "metric": "official_change_2011_2016_pct",
                "count": "",
                "value": by_year[2016]["change_vs_previous_study_year_pct"],
                "decision": "vkt_change_contains_both_intensity_and_network_length_change",
            },
            {
                "metric": "official_change_2016_2021_pct",
                "count": "",
                "value": by_year[2021]["change_vs_previous_study_year_pct"],
                "decision": "total_vkt_rose_but_this_does_not_prove_mean_traffic_intensity_rose",
            },
            {
                "metric": "official_implied_mean_aadt_change_2016_2021_pct",
                "count": "",
                "value": by_year[2021][
                    "territory_total_implied_mean_aadt_change_pct"
                ],
                "decision": "same_metric_as_length_weighted_mean_aadt_not_total_vkt",
            },
            {
                "metric": "scope_limitation",
                "count": "",
                "value": "roads_covered_in_the_annual_traffic_census_not_every_road",
                "decision": "benchmark_bounds_the_census_network_not_the_whole_centreline",
            },
            {
                "metric": "step13_decision_signal",
                "count": "",
                "value": "official_vkt_length_and_implied_mean_aadt_are_available_as_a_support_consistency_check",
                "decision": "published_constraint_not_an_independent_label_set",
            },
        ]
    )
    return audit


def main() -> None:
    all_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    largest_rounding_discrepancy = 0
    for year in YEARS:
        rows = parse_vkt_table(year)
        largest_rounding_discrepancy = max(
            largest_rounding_discrepancy,
            validate_year_rows(year, rows),
        )
        all_rows.extend(rows)
        parsed_lengths = parse_road_network_lengths(year)
        length_rows.extend(parsed_lengths)
        print(f"Parsed ATC {year} section 3.4 vehicle-kilometrage table: {len(rows)} rows")
        print(f"Parsed ATC {year} Appendix H road-network lengths: {len(parsed_lengths)} rows")

    benchmark = build_benchmark(all_rows, length_rows)
    audit = build_decision_audit(
        all_rows, length_rows, benchmark, largest_rounding_discrepancy
    )

    write_csv(OFFICIAL_VKT_PATH, all_rows)
    write_csv(OFFICIAL_LENGTH_PATH, length_rows)
    write_csv(BENCHMARK_PATH, benchmark)
    write_csv(DECISION_AUDIT_PATH, audit)

    print("\nStep 13 official benchmark extraction is complete.")
    for row in benchmark:
        print(
            f"  {row['census_year']}: territory {int(row['territory_total_daily_vehicle_km']):,} "
            f"veh-km/day (major {int(row['territory_major_daily_vehicle_km']):,}, "
            f"minor {int(row['territory_minor_daily_vehicle_km']):,}); "
            f"covered length {float(row['territory_total_road_length_km']):,.2f} km; "
            f"implied mean AADT {float(row['territory_total_implied_mean_aadt']):,.0f}"
        )
    print(
        "  official change 2011-2016: "
        f"{benchmark[1]['change_vs_previous_study_year_pct']}%; "
        "2016-2021: "
        f"{benchmark[2]['change_vs_previous_study_year_pct']}%"
    )
    print(
        "\nInterpretation rule: compare like support with like support. A VKT ratio alone "
        "conflates road length and traffic intensity; report VKT, support length, and the "
        "implied length-weighted mean AADT together."
    )
    print("Next: python src\\14_validate_network_against_official_vkt.py")


if __name__ == "__main__":
    main()
