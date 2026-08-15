from __future__ import annotations

import csv
import re
from pathlib import Path

import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "reports"
TEXT_DIR = PROJECT_ROOT / "data" / "interim" / "atc_text"

REPORTS = {
    2011: REPORT_DIR / "ATC_2011.pdf",
    2016: REPORT_DIR / "ATC_2016.pdf",
    2021: REPORT_DIR / "ATC_2021.pdf",
}

APPENDIX_B_PATTERN = re.compile(
    r"Appendix\s+B\s*-?\s*(?:A\.?A\.?D\.?T\.?|AADT)\s+of\s+Counting\s+Stations",
    re.IGNORECASE,
)
APPENDIX_C_PATTERN = re.compile(
    r"Appendix\s+C\s*-?\s*(?:A\.?A\.?D\.?T\.?|AADT)\s+of\s+Counting\s+Stations",
    re.IGNORECASE,
)


def extract_report(year: int, pdf_path: Path) -> list[dict[str, object]]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing {pdf_path}. Run python src\\01_download_atc_reports.py first."
        )

    output_path = TEXT_DIR / f"ATC_{year}_layout.txt"
    page_records: list[dict[str, object]] = []

    print(f"Extracting layout text: {pdf_path.name}")
    with pdfplumber.open(pdf_path) as report, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for page_number, page in enumerate(report.pages, start=1):
            text = page.extract_text(layout=True) or ""
            output_file.write(f"\n===== PDF PAGE {page_number} =====\n")
            output_file.write(text)
            output_file.write("\n")

            page_records.append(
                {
                    "year": year,
                    "pdf_page": page_number,
                    "contains_appendix_b_heading": bool(
                        APPENDIX_B_PATTERN.search(text)
                    ),
                    "contains_appendix_c_heading": bool(
                        APPENDIX_C_PATTERN.search(text)
                    ),
                }
            )

    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return page_records


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    all_page_records: list[dict[str, object]] = []
    for year, report_path in REPORTS.items():
        all_page_records.extend(extract_report(year, report_path))

    index_path = TEXT_DIR / "page_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=[
                "year",
                "pdf_page",
                "contains_appendix_b_heading",
                "contains_appendix_c_heading",
            ],
        )
        writer.writeheader()
        writer.writerows(all_page_records)

    print(f"Saved: {index_path.relative_to(PROJECT_ROOT)}")
    print("\nText preparation is complete. Next: extract Appendix B station-year rows.")


if __name__ == "__main__":
    main()
