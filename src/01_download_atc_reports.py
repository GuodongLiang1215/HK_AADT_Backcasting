from __future__ import annotations

import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = PROJECT_ROOT / "config" / "official_sources.json"
REPORT_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "reports"


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already available: {destination.relative_to(PROJECT_ROOT)}")
        return

    print(f"Downloading: {destination.name}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Saved: {destination.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with SOURCE_CONFIG.open("r", encoding="utf-8") as source_file:
        source_config = json.load(source_file)

    for report in source_config["atc_reports"]:
        destination = REPORT_DIR / report["filename"]
        download_file(report["url"], destination)

    print("\nOfficial ATC reports are ready.")
    print("Next command: python src\\02_extract_report_text.py")


if __name__ == "__main__":
    main()
