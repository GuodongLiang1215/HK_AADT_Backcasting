"""Step 25A: materialise and validate a bounded annual strategic-detector signal.

This step is deliberately narrower than a full-network backcast.  It asks:

1. Is the public strategic-detector archive sufficiently complete for a
   comparable 2021--2024 sample?
2. Can a balanced annual detector proxy be built for a stable detector panel?
3. Does detector change improve on a no-change benchmark for measured ATC
   AADT change, both at colocated sites and after holding out a spatial fold?

The annual proxy is not AADT.  It is the equal-weighted mean of six fixed
weekday/weekend-by-time strata, sampled on the second Tuesday and Saturday of
March--December at 08:00, 13:00 and 18:00.  March--December is fixed because
2021 is the first archive year and lacks January--February coverage.

All gates and thresholds below are fixed before inspecting Step 25A results.
Passing authorises only a later major-road temporal downscaling experiment.
It does not establish local-road reconstruction, a full-network backcast, or
an equity trend.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "step25a_strategic_history"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

STRATEGIC_RAW_URL = (
    "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
)
ARCHIVE_LIST_ROOT = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
ARCHIVE_GET_ROOT = "https://app.data.gov.hk/v1/historical-archive/get-file"

MEASURED_PANEL_PATH = PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
DETECTOR_LOCATION_PATH = (
    PROCESSED_DIR / "atc_step24_public_dynamic_detector_locations.csv"
)

SNAPSHOT_LONG_PATH = PROCESSED_DIR / "atc_step25a_strategic_snapshot_long.csv"
ANNUAL_PROXY_PATH = PROCESSED_DIR / "atc_step25a_strategic_annual_proxy.csv"
STABLE_PANEL_PATH = PROCESSED_DIR / "atc_step25a_stable_detector_panel.csv"
CROSSWALK_PATH = PROCESSED_DIR / "atc_step25a_detector_atc_crosswalk.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step25a_temporal_predictions.csv"

ARCHIVE_INVENTORY_PATH = TABLE_DIR / "step25a_archive_year_inventory.csv"
SAMPLING_AUDIT_PATH = TABLE_DIR / "step25a_sampling_audit.csv"
ANNUAL_AUDIT_PATH = TABLE_DIR / "step25a_annual_proxy_audit.csv"
CROSSWALK_AUDIT_PATH = TABLE_DIR / "step25a_crosswalk_audit.csv"
DETECTOR_ID_AUDIT_PATH = TABLE_DIR / "step25a_detector_id_linkage_audit.csv"
TRANSITION_METRICS_PATH = TABLE_DIR / "step25a_metrics_by_transition.csv"
FOLD_METRICS_PATH = TABLE_DIR / "step25a_metrics_by_fold.csv"
PAIRED_COMPARISON_PATH = TABLE_DIR / "step25a_paired_model_comparison.csv"
DECISION_PATH = TABLE_DIR / "step25a_decision_audit.csv"

COVERAGE_FIGURE_PATH = FIGURE_DIR / "step25a_archive_and_panel_coverage.png"
CHANGE_FIGURE_PATH = FIGURE_DIR / "step25a_change_identification.png"

INVENTORY_YEARS = tuple(range(2018, 2025))
PRIMARY_YEARS = (2021, 2022, 2023, 2024)
COMMON_MONTHS = tuple(range(3, 13))
SAMPLE_HOURS = (800, 1300, 1800)
DAY_TYPES = ("weekday", "weekend")
TIME_BLOCKS = ("am", "midday", "pm")
EXPECTED_SNAPSHOTS_PER_YEAR = len(COMMON_MONTHS) * len(DAY_TYPES) * len(SAMPLE_HOURS)

MAX_TIME_OFFSET_MINUTES = 15
MIN_YEAR_SAMPLE_SHARE = 0.90
MIN_DETECTOR_YEAR_SAMPLE_SHARE = 0.80
MIN_DETECTOR_YEAR_MONTHS = 8
MIN_STABLE_DETECTORS = 100
MIN_STABLE_SPATIAL_FOLDS = 4
MAX_CROSSWALK_DISTANCE_M = 100.0
HIGH_CONFIDENCE_DISTANCE_M = 20.0
MIN_ROAD_NAME_SIMILARITY = 0.50
MIN_CROSSWALK_PAIRS = 20

MIN_MAE_IMPROVEMENT_PCT = 5.0
MIN_IMPROVED_TRANSITIONS = 2
MIN_IMPROVED_FOLDS = 3
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 250817
DOWNLOAD_WORKERS = 3

STEP25A_REVISION = "2026-08-17.2"

# The historical-archive API currently returns timestamps as
# ``YYYYMMDD-HHMM`` (for example, ``20230314-0800``).  Some archive metadata
# and filenames may expose the same value without the hyphen.  Canonicalise
# both representations to the hyphenated form required by the get-file API.
TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{6})-?(\d{4})(?!\d)")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def normalise_identifier(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def find_column(
    frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True
) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(
            f"None of {candidates} found. Available columns: {list(frame.columns)}"
        )
    return None


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def request_bytes(url: str, timeout: int = 240, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HK-AADT-public-data-research/1.0",
            "Accept": "application/json, application/xml, text/xml, */*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 2.0 * (attempt + 1)
            time.sleep(min(delay, 20.0))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Request failed: {last_error}")


def archive_list_url(start: str, end: str) -> str:
    query = urllib.parse.urlencode(
        {"url": STRATEGIC_RAW_URL, "start": start, "end": end}
    )
    return f"{ARCHIVE_LIST_ROOT}?{query}"


def historical_file_url(timestamp: str) -> str:
    query = urllib.parse.urlencode(
        {"url": STRATEGIC_RAW_URL, "time": timestamp}
    )
    return f"{ARCHIVE_GET_ROOT}?{query}"


def read_json_cache(path: Path, url: str, refresh: bool) -> dict[str, object]:
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Archive API cache is not a JSON object: {path}")
            return payload
    payload = json.loads(request_bytes(url, timeout=180).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Archive API response is not a JSON object")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def recursive_values(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from recursive_values(child)
    elif isinstance(value, list):
        for child in value:
            yield "", child
            yield from recursive_values(child)


def extract_timestamps(payload: dict[str, object]) -> list[str]:
    values: set[str] = set()
    for _, value in recursive_values(payload):
        if not isinstance(value, str):
            continue
        for match in TIMESTAMP_PATTERN.finditer(value):
            values.add(f"{match.group(1)}-{match.group(2)}")
    return sorted(values)


def extract_total_version_count(payload: dict[str, object]) -> int | None:
    preferred = {
        "total",
        "total_count",
        "totalcount",
        "record_count",
        "version_count",
        "total_versions",
    }
    candidates: list[int] = []
    for key, value in recursive_values(payload):
        key_clean = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if key_clean not in preferred:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            candidates.append(number)
    return max(candidates) if candidates else None


def extract_zip_inventory(payload: dict[str, object]) -> tuple[int, float | None]:
    zip_names: set[str] = set()
    size_values: list[float] = []
    for key, value in recursive_values(payload):
        if isinstance(value, str) and ".zip" in value.lower():
            zip_names.add(value)
        key_clean = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if key_clean not in {"size", "file_size", "compressed_size", "filesize"}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            size_values.append(number)
    total_gb = sum(size_values) / 1e9 if size_values else None
    return len(zip_names), total_gb


def inventory_archive(refresh: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in INVENTORY_YEARS:
        start = f"{year}0101"
        end = f"{year}1231"
        cache = RAW_DIR / "queries" / f"year_{year}.json"
        try:
            payload = read_json_cache(cache, archive_list_url(start, end), refresh)
            timestamps = extract_timestamps(payload)
            zip_count, compressed_gb = extract_zip_inventory(payload)
            rows.append(
                {
                    "year": year,
                    "query_status": "obtained",
                    "total_version_count_reported": extract_total_version_count(payload),
                    "timestamp_count_returned": len(timestamps),
                    "first_returned_timestamp": min(timestamps) if timestamps else "",
                    "last_returned_timestamp": max(timestamps) if timestamps else "",
                    "monthly_package_count_discovered": zip_count,
                    "compressed_package_size_gb_discovered": compressed_gb,
                    "primary_step25a_year": year in PRIMARY_YEARS,
                    "interpretation": (
                        "year-level inventory; returned timestamp lists may be API-capped, "
                        "so sampling continuity is evaluated from fixed daily queries"
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "year": year,
                    "query_status": f"retrieval_error:{type(exc).__name__}",
                    "total_version_count_reported": np.nan,
                    "timestamp_count_returned": np.nan,
                    "first_returned_timestamp": "",
                    "last_returned_timestamp": "",
                    "monthly_package_count_discovered": np.nan,
                    "compressed_package_size_gb_discovered": np.nan,
                    "primary_step25a_year": year in PRIMARY_YEARS,
                    "interpretation": "retrieval failure is not evidence that the archive is absent",
                }
            )
    result = pd.DataFrame(rows)
    save_csv(result, ARCHIVE_INVENTORY_PATH)
    return result


def nth_weekday(year: int, month: int, weekday: int, occurrence: int = 2) -> date:
    days = [
        day
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if date(year, month, day).weekday() == weekday
    ]
    return date(year, month, days[occurrence - 1])


def sample_day_specifications() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in PRIMARY_YEARS:
        for month in COMMON_MONTHS:
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "sample_date": nth_weekday(year, month, calendar.TUESDAY),
                    "day_type": "weekday",
                }
            )
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "sample_date": nth_weekday(year, month, calendar.SATURDAY),
                    "day_type": "weekend",
                }
            )
    return rows


def timestamp_offset_minutes(timestamp: str, target_hhmm: int) -> int:
    hhmm = int(timestamp[-4:])
    observed = (hhmm // 100) * 60 + hhmm % 100
    target = (target_hhmm // 100) * 60 + target_hhmm % 100
    return abs(observed - target)


def closest_timestamp(
    timestamps: list[str], day_text: str, target_hhmm: int
) -> tuple[str | None, int | None]:
    same_day = [value for value in timestamps if value.startswith(day_text)]
    if not same_day:
        return None, None
    chosen = min(same_day, key=lambda value: timestamp_offset_minutes(value, target_hhmm))
    offset = timestamp_offset_minutes(chosen, target_hhmm)
    if offset > MAX_TIME_OFFSET_MINUTES:
        return None, offset
    return chosen, offset


def download_historical(timestamp: str, path: Path, refresh: bool) -> Path:
    if path.exists() and path.stat().st_size > 100 and not refresh:
        return path
    payload = request_bytes(historical_file_url(timestamp), timeout=300)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    root = ET.parse(temporary).getroot()
    if not root.findall(".//detector"):
        temporary.unlink(missing_ok=True)
        raise ValueError("Archived XML contains no detector records")
    temporary.replace(path)
    return path


def obtain_sample_day(
    specification: dict[str, object], refresh: bool
) -> list[dict[str, object]]:
    sample_date = specification["sample_date"]
    if not isinstance(sample_date, date):
        raise TypeError("sample_date must be datetime.date")
    day_text = sample_date.strftime("%Y%m%d")
    query_cache = RAW_DIR / "queries" / f"day_{day_text}.json"
    try:
        payload = read_json_cache(
            query_cache, archive_list_url(day_text, day_text), refresh
        )
        timestamps = extract_timestamps(payload)
        query_status = "obtained"
        query_error = ""
    except Exception as exc:
        timestamps = []
        query_status = "retrieval_error"
        query_error = f"{type(exc).__name__}:{exc}"

    rows: list[dict[str, object]] = []
    for requested_hour in SAMPLE_HOURS:
        time_block = (
            "am" if requested_hour < 1000 else "midday" if requested_hour < 1600 else "pm"
        )
        base = {
            "year": specification["year"],
            "month": specification["month"],
            "sample_date": sample_date.isoformat(),
            "day_type": specification["day_type"],
            "requested_hour": requested_hour,
            "time_block": time_block,
            "daily_query_status": query_status,
            "daily_query_timestamp_count": len(timestamps),
        }
        if query_status != "obtained":
            rows.append(
                {
                    **base,
                    "archive_timestamp": "",
                    "time_offset_minutes": np.nan,
                    "status": "retrieval_error",
                    "error": query_error,
                    "local_path": "",
                    "download_size_mb": np.nan,
                }
            )
            continue
        timestamp, offset = closest_timestamp(timestamps, day_text, requested_hour)
        if timestamp is None:
            rows.append(
                {
                    **base,
                    "archive_timestamp": "",
                    "time_offset_minutes": offset,
                    "status": "archive_absent_near_requested_time",
                    "error": "",
                    "local_path": "",
                    "download_size_mb": np.nan,
                }
            )
            continue
        path = RAW_DIR / "xml" / str(specification["year"]) / f"strategic_{timestamp}.xml"
        try:
            download_historical(timestamp, path, refresh)
            rows.append(
                {
                    **base,
                    "archive_timestamp": timestamp,
                    "time_offset_minutes": offset,
                    "status": "obtained",
                    "error": "",
                    "local_path": str(path.relative_to(PROJECT_ROOT)),
                    "download_size_mb": path.stat().st_size / 1e6,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "archive_timestamp": timestamp,
                    "time_offset_minutes": offset,
                    "status": "retrieval_or_parse_error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "local_path": "",
                    "download_size_mb": np.nan,
                }
            )
    return rows


def obtain_samples(refresh: bool, workers: int) -> pd.DataFrame:
    specifications = sample_day_specifications()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(obtain_sample_day, specification, refresh): specification
            for specification in specifications
        }
        completed = 0
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                specification = futures[future]
                for requested_hour in SAMPLE_HOURS:
                    rows.append(
                        {
                            "year": specification["year"],
                            "month": specification["month"],
                            "sample_date": specification["sample_date"].isoformat(),
                            "day_type": specification["day_type"],
                            "requested_hour": requested_hour,
                            "time_block": "",
                            "daily_query_status": "job_error",
                            "daily_query_timestamp_count": np.nan,
                            "archive_timestamp": "",
                            "time_offset_minutes": np.nan,
                            "status": "job_error",
                            "error": f"{type(exc).__name__}:{exc}",
                            "local_path": "",
                            "download_size_mb": np.nan,
                        }
                    )
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                print(f"Completed {completed}/{len(futures)} fixed sampling days.")
    audit = pd.DataFrame(rows).sort_values(
        ["year", "sample_date", "requested_hour"]
    )
    save_csv(audit, SAMPLING_AUDIT_PATH)
    return audit


def parse_snapshot_file(sample: object) -> list[dict[str, object]]:
    path = PROJECT_ROOT / str(sample.local_path)
    root = ET.parse(path).getroot()
    per_detector: dict[str, list[tuple[float, float, float, int]]] = defaultdict(list)
    for period in root.findall(".//period"):
        for detector in period.findall("./detectors/detector"):
            detector_id = normalise_identifier(detector.findtext("detector_id", ""))
            if not detector_id:
                continue
            lane_values: list[tuple[float, float, float]] = []
            for lane in detector.findall("./lanes/lane"):
                if str(lane.findtext("valid", "")).strip().upper() != "Y":
                    continue
                volume = safe_float(lane.findtext("volume"))
                speed = safe_float(lane.findtext("speed"))
                occupancy = safe_float(lane.findtext("occupancy"))
                if np.isfinite(volume):
                    lane_values.append((volume, speed, occupancy))
            if not lane_values:
                continue
            array = np.asarray(lane_values, dtype=float)
            weights = np.maximum(array[:, 0], 1.0)
            finite_speed = np.isfinite(array[:, 1])
            mean_speed = (
                float(np.average(array[finite_speed, 1], weights=weights[finite_speed]))
                if finite_speed.any()
                else np.nan
            )
            per_detector[detector_id].append(
                (
                    float(np.nansum(array[:, 0])),
                    mean_speed,
                    float(np.nanmean(array[:, 2])) if np.isfinite(array[:, 2]).any() else np.nan,
                    len(lane_values),
                )
            )
    rows: list[dict[str, object]] = []
    for detector_id, values in per_detector.items():
        frame = pd.DataFrame(
            values, columns=["minute_volume", "speed", "occupancy", "valid_lanes"]
        )
        rows.append(
            {
                "detector_id": detector_id,
                "year": int(sample.year),
                "month": int(sample.month),
                "sample_date": sample.sample_date,
                "day_type": sample.day_type,
                "requested_hour": int(sample.requested_hour),
                "time_block": sample.time_block,
                "archive_timestamp": sample.archive_timestamp,
                "sampled_minute_volume": frame["minute_volume"].mean(),
                "sampled_speed": frame["speed"].mean(),
                "sampled_occupancy": frame["occupancy"].mean(),
                "valid_lane_count": frame["valid_lanes"].median(),
            }
        )
    return rows


def parse_all_samples(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = audit[audit["status"].eq("obtained")]
    for completed, sample in enumerate(samples.itertuples(index=False), start=1):
        rows.extend(parse_snapshot_file(sample))
        if completed % 20 == 0 or completed == len(samples):
            print(f"Parsed {completed}/{len(samples)} strategic detector snapshots.")
    long_columns = [
        "detector_id",
        "year",
        "month",
        "sample_date",
        "day_type",
        "requested_hour",
        "time_block",
        "archive_timestamp",
        "sampled_minute_volume",
        "sampled_speed",
        "sampled_occupancy",
        "valid_lane_count",
    ]
    long = pd.DataFrame(rows, columns=long_columns)
    if not long.empty:
        long = long.sort_values(["detector_id", "year", "sample_date", "requested_hour"])
    save_csv(long, SNAPSHOT_LONG_PATH)
    return long


def modal_integer(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().round().astype(int)
    if clean.empty:
        return np.nan
    counts = clean.value_counts()
    return float(sorted(counts[counts.eq(counts.max())].index)[0])


def annualise_detector_proxy(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if long.empty:
        annual = pd.DataFrame(
            columns=[
                "detector_id",
                "year",
                "snapshot_count",
                "sample_share_of_expected",
                "distinct_months",
                "stratum_count",
                "balanced_annual_volume_proxy",
                "modal_valid_lane_count",
                "mean_sampled_speed",
                "detector_year_qualified",
                "observed_year_count",
                "qualified_all_primary_years",
                "modal_lane_configuration_stable",
                "stable_detector_panel_member",
            ]
        )
        stable = annual.copy()
        audit = pd.DataFrame(
            [
                {
                    "year": year,
                    "detector_year_rows": 0,
                    "qualified_detector_years": 0,
                    "stable_panel_detectors": 0,
                    "median_balanced_annual_proxy_stable_panel": np.nan,
                    "expected_snapshots_per_detector_year": EXPECTED_SNAPSHOTS_PER_YEAR,
                }
                for year in PRIMARY_YEARS
            ]
        )
        save_csv(annual, ANNUAL_PROXY_PATH)
        save_csv(stable, STABLE_PANEL_PATH)
        save_csv(audit, ANNUAL_AUDIT_PATH)
        return annual, stable

    rows: list[dict[str, object]] = []
    for (detector_id, year), group in long.groupby(["detector_id", "year"]):
        stratum = (
            group.groupby(["day_type", "time_block"], as_index=False)[
                "sampled_minute_volume"
            ]
            .mean()
        )
        snapshot_count = group["archive_timestamp"].nunique()
        distinct_months = group["month"].nunique()
        stratum_count = len(stratum)
        qualified = (
            snapshot_count >= math.ceil(EXPECTED_SNAPSHOTS_PER_YEAR * MIN_DETECTOR_YEAR_SAMPLE_SHARE)
            and distinct_months >= MIN_DETECTOR_YEAR_MONTHS
            and stratum_count == len(DAY_TYPES) * len(TIME_BLOCKS)
        )
        rows.append(
            {
                "detector_id": detector_id,
                "year": int(year),
                "snapshot_count": snapshot_count,
                "sample_share_of_expected": snapshot_count / EXPECTED_SNAPSHOTS_PER_YEAR,
                "distinct_months": distinct_months,
                "stratum_count": stratum_count,
                "balanced_annual_volume_proxy": stratum["sampled_minute_volume"].mean(),
                "modal_valid_lane_count": modal_integer(group["valid_lane_count"]),
                "mean_sampled_speed": group["sampled_speed"].mean(),
                "detector_year_qualified": qualified,
            }
        )
    annual = pd.DataFrame(rows).sort_values(["detector_id", "year"])

    support_rows: list[dict[str, object]] = []
    stable_ids: list[str] = []
    for detector_id, group in annual.groupby("detector_id"):
        years = set(group["year"].astype(int))
        all_years = set(PRIMARY_YEARS).issubset(years)
        qualified_all = all_years and bool(
            group.set_index("year").reindex(PRIMARY_YEARS)["detector_year_qualified"].fillna(False).all()
        )
        lane_values = (
            group.set_index("year").reindex(PRIMARY_YEARS)["modal_valid_lane_count"].dropna().unique()
            if all_years
            else np.asarray([])
        )
        stable_lanes = qualified_all and len(lane_values) == 1
        stable = bool(qualified_all and stable_lanes)
        if stable:
            stable_ids.append(detector_id)
        support_rows.append(
            {
                "detector_id": detector_id,
                "observed_year_count": len(years.intersection(PRIMARY_YEARS)),
                "qualified_all_primary_years": qualified_all,
                "modal_lane_configuration_stable": stable_lanes,
                "stable_detector_panel_member": stable,
            }
        )
    support = pd.DataFrame(support_rows)
    annual = annual.merge(support, on="detector_id", how="left", validate="many_to_one")
    stable = annual[annual["stable_detector_panel_member"].fillna(False)].copy()
    save_csv(annual, ANNUAL_PROXY_PATH)
    save_csv(stable, STABLE_PANEL_PATH)

    audit_rows: list[dict[str, object]] = []
    for year in PRIMARY_YEARS:
        subset = annual[annual["year"].eq(year)]
        audit_rows.append(
            {
                "year": year,
                "detector_year_rows": len(subset),
                "qualified_detector_years": int(subset["detector_year_qualified"].fillna(False).sum()),
                "stable_panel_detectors": len(stable_ids),
                "median_balanced_annual_proxy_stable_panel": stable.loc[
                    stable["year"].eq(year), "balanced_annual_volume_proxy"
                ].median(),
                "expected_snapshots_per_detector_year": EXPECTED_SNAPSHOTS_PER_YEAR,
            }
        )
    save_csv(pd.DataFrame(audit_rows), ANNUAL_AUDIT_PATH)
    return annual, stable


def projected_xy(longitude: pd.Series, latitude: pd.Series) -> np.ndarray:
    reference_latitude = float(pd.to_numeric(latitude, errors="coerce").median())
    x = pd.to_numeric(longitude, errors="coerce").to_numpy(float)
    y = pd.to_numeric(latitude, errors="coerce").to_numpy(float)
    x = x * 111_320.0 * math.cos(math.radians(reference_latitude))
    y = y * 110_540.0
    return np.column_stack([x, y])


def normalise_road_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    substitutions = {
        "RD": "ROAD",
        "ST": "STREET",
        "AVE": "AVENUE",
        "HWY": "HIGHWAY",
    }
    tokens = [substitutions.get(token, token) for token in text.split()]
    return " ".join(tokens)


def road_name_similarity(left: object, right: object) -> float:
    a = normalise_road_name(left)
    b = normalise_road_name(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def mode_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if clean.empty:
        return np.nan
    modes = clean.mode()
    return float(modes.min())


def prepare_detector_locations(stable_ids: set[str]) -> pd.DataFrame:
    locations = pd.read_csv(DETECTOR_LOCATION_PATH)
    source_col = find_column(locations, ("source",))
    id_col = find_column(locations, ("device_id", "detector_id"))
    lon_col = find_column(locations, ("longitude", "lon"))
    lat_col = find_column(locations, ("latitude", "lat"))
    road_col = find_column(locations, ("road_name", "road_en"), required=False)
    fold_col = find_column(
        locations, ("nearest_spatial_fold", "spatial_fold"), required=False
    )
    strategic = locations[locations[source_col].astype(str).eq("strategic_detector")].copy()
    strategic["detector_id"] = strategic[id_col].map(normalise_identifier)
    current_ids = set(strategic["detector_id"].dropna().astype(str))
    linked_ids = stable_ids.intersection(current_ids)
    audit = pd.DataFrame(
        [
            {
                "stable_detector_id_count": len(stable_ids),
                "current_strategic_location_id_count": len(current_ids),
                "stable_id_with_current_location_count": len(linked_ids),
                "stable_id_linkage_share": len(linked_ids) / len(stable_ids) if stable_ids else np.nan,
                "stable_only_id_examples": "|".join(sorted(stable_ids - current_ids)[:20]),
                "current_only_id_examples": "|".join(sorted(current_ids - stable_ids)[:20]),
                "interpretation": (
                    "empty linkage can mean either no detector passed the four-year stable-panel rule "
                    "or historical feed IDs do not match the current location-table namespace"
                ),
            }
        ]
    )
    save_csv(audit, DETECTOR_ID_AUDIT_PATH)

    subset = strategic[strategic["detector_id"].isin(stable_ids)].copy()
    subset["longitude"] = pd.to_numeric(subset[lon_col], errors="coerce")
    subset["latitude"] = pd.to_numeric(subset[lat_col], errors="coerce")
    subset["road_name"] = subset[road_col].fillna("").astype(str) if road_col else ""
    subset["spatial_fold"] = (
        pd.to_numeric(subset[fold_col], errors="coerce") if fold_col else np.nan
    )
    subset = subset.dropna(subset=["longitude", "latitude"])
    output_columns = [
        "detector_id",
        "longitude",
        "latitude",
        "road_name",
        "spatial_fold",
        "current_location_record_count",
    ]
    rows: list[dict[str, object]] = []
    for detector_id, group in subset.groupby("detector_id"):
        road_names = sorted({value.strip() for value in group["road_name"] if value.strip()})
        rows.append(
            {
                "detector_id": detector_id,
                "longitude": group["longitude"].median(),
                "latitude": group["latitude"].median(),
                "road_name": " | ".join(road_names),
                "spatial_fold": mode_or_nan(group["spatial_fold"]),
                "current_location_record_count": len(group),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def prepare_measured_panel() -> pd.DataFrame:
    panel = pd.read_csv(MEASURED_PANEL_PATH)
    year_col = find_column(panel, ("year",))
    station_col = find_column(panel, ("station_id", "station_number"))
    aadt_col = find_column(panel, ("aadt", "observed_aadt"))
    lon_col = find_column(panel, ("longitude", "station_longitude"))
    lat_col = find_column(panel, ("latitude", "station_latitude"))
    fold_col = find_column(panel, ("spatial_fold", "step17_spatial_fold"))
    network_col = find_column(panel, ("road_network", "structured_road_network"))
    name_col = find_column(
        panel,
        ("road_name", "official_name", "link_description", "step17_link_description"),
        required=False,
    )
    result = pd.DataFrame(
        {
            "year": pd.to_numeric(panel[year_col], errors="coerce"),
            "station_id": panel[station_col].map(normalise_identifier),
            "aadt": pd.to_numeric(panel[aadt_col], errors="coerce"),
            "longitude": pd.to_numeric(panel[lon_col], errors="coerce"),
            "latitude": pd.to_numeric(panel[lat_col], errors="coerce"),
            "spatial_fold": pd.to_numeric(panel[fold_col], errors="coerce"),
            "road_network": panel[network_col].fillna("").astype(str).str.upper(),
            "station_road_name": panel[name_col].fillna("").astype(str) if name_col else "",
        }
    )
    result = result[
        result["year"].isin(PRIMARY_YEARS)
        & result["aadt"].gt(0)
        & result[["longitude", "latitude"]].notna().all(axis=1)
    ].copy()
    result["year"] = result["year"].astype(int)
    result["spatial_fold"] = result["spatial_fold"].astype("Int64")
    return result


def build_crosswalk(
    panel: pd.DataFrame, detectors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk_columns = [
        "station_id",
        "detector_id",
        "distance_m",
        "road_name_similarity",
        "station_road_name",
        "detector_road_name",
        "station_spatial_fold",
        "detector_spatial_fold",
        "candidate_accepted",
        "acceptance_rule",
    ]
    station_metadata = (
        panel.sort_values(["station_id", "year"])
        .groupby("station_id", as_index=False)
        .first()
    )
    station_metadata = station_metadata[station_metadata["road_network"].eq("MAJOR")].copy()
    if station_metadata.empty or detectors.empty:
        crosswalk = pd.DataFrame(columns=crosswalk_columns)
        audit = pd.DataFrame(
            [
                {
                    "metric": "accepted_pairs",
                    "value": 0,
                    "interpretation": (
                        "no eligible station-detector geometry; inspect "
                        "step25a_detector_id_linkage_audit.csv before interpreting the cause"
                    ),
                }
            ]
        )
        save_csv(crosswalk, CROSSWALK_PATH)
        save_csv(audit, CROSSWALK_AUDIT_PATH)
        return crosswalk, audit

    reference_latitude = float(
        pd.concat([station_metadata["latitude"], detectors["latitude"]]).median()
    )

    def xy(frame: pd.DataFrame) -> np.ndarray:
        x = frame["longitude"].to_numpy(float) * 111_320.0 * math.cos(
            math.radians(reference_latitude)
        )
        y = frame["latitude"].to_numpy(float) * 110_540.0
        return np.column_stack([x, y])

    detector_xy = xy(detectors)
    station_xy = xy(station_metadata)
    tree = cKDTree(detector_xy)
    candidates: list[dict[str, object]] = []
    for station_position, detector_positions in enumerate(
        tree.query_ball_point(station_xy, r=MAX_CROSSWALK_DISTANCE_M)
    ):
        station = station_metadata.iloc[station_position]
        for detector_position in detector_positions:
            detector = detectors.iloc[detector_position]
            distance = float(
                np.linalg.norm(station_xy[station_position] - detector_xy[detector_position])
            )
            similarity = road_name_similarity(
                station["station_road_name"], detector["road_name"]
            )
            accepted = (
                distance <= HIGH_CONFIDENCE_DISTANCE_M
                or similarity >= MIN_ROAD_NAME_SIMILARITY
            )
            candidates.append(
                {
                    "station_id": station["station_id"],
                    "detector_id": detector["detector_id"],
                    "distance_m": distance,
                    "road_name_similarity": similarity,
                    "station_road_name": station["station_road_name"],
                    "detector_road_name": detector["road_name"],
                    "station_spatial_fold": station["spatial_fold"],
                    "detector_spatial_fold": detector["spatial_fold"],
                    "candidate_accepted": accepted,
                    "acceptance_rule": (
                        "distance<=20m" if distance <= HIGH_CONFIDENCE_DISTANCE_M
                        else "road_name_similarity>=0.50" if accepted else "rejected"
                    ),
                }
            )
    candidate_frame = pd.DataFrame(candidates, columns=crosswalk_columns)
    accepted = candidate_frame[candidate_frame["candidate_accepted"]].copy()
    if not accepted.empty:
        accepted = accepted.sort_values(
            ["road_name_similarity", "distance_m"], ascending=[False, True]
        )
        used_stations: set[str] = set()
        used_detectors: set[str] = set()
        selected: list[dict[str, object]] = []
        for row in accepted.to_dict("records"):
            if row["station_id"] in used_stations or row["detector_id"] in used_detectors:
                continue
            used_stations.add(row["station_id"])
            used_detectors.add(row["detector_id"])
            selected.append(row)
        crosswalk = pd.DataFrame(selected)
    else:
        crosswalk = pd.DataFrame(columns=list(candidate_frame.columns))
    save_csv(crosswalk, CROSSWALK_PATH)

    audit = pd.DataFrame(
        [
            {
                "metric": "major_atc_stations_within_100m_candidate",
                "value": candidate_frame["station_id"].nunique() if not candidate_frame.empty else 0,
                "interpretation": "current-coordinate candidate support",
            },
            {
                "metric": "stable_detectors_within_100m_candidate",
                "value": candidate_frame["detector_id"].nunique() if not candidate_frame.empty else 0,
                "interpretation": "current-coordinate candidate support",
            },
            {
                "metric": "accepted_one_to_one_pairs",
                "value": len(crosswalk),
                "interpretation": "distance<=20m or road-name similarity>=0.50, then greedy one-to-one assignment",
            },
            {
                "metric": "historical_detector_location_available",
                "value": False,
                "interpretation": "current coordinates are used only for stable detector IDs; survivor/current-location limitation",
            },
        ]
    )
    save_csv(audit, CROSSWALK_AUDIT_PATH)
    return crosswalk, audit


def build_station_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for base_year, target_year in zip(PRIMARY_YEARS[:-1], PRIMARY_YEARS[1:]):
        base = panel[panel["year"].eq(base_year)].copy()
        target = panel[panel["year"].eq(target_year)].copy()
        merged = base.merge(
            target[["station_id", "aadt", "road_network"]],
            on="station_id",
            how="inner",
            suffixes=("_base", "_target"),
            validate="one_to_one",
        )
        merged = merged[
            merged["road_network_base"].eq("MAJOR")
            & merged["road_network_target"].eq("MAJOR")
        ].copy()
        merged["base_year"] = base_year
        merged["target_year"] = target_year
        merged["transition"] = f"{base_year}-{target_year}"
        merged["observed_change"] = merged["aadt_target"] - merged["aadt_base"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_detector_ratios(stable: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for base_year, target_year in zip(PRIMARY_YEARS[:-1], PRIMARY_YEARS[1:]):
        base = stable[stable["year"].eq(base_year)][
            ["detector_id", "balanced_annual_volume_proxy"]
        ].rename(columns={"balanced_annual_volume_proxy": "proxy_base"})
        target = stable[stable["year"].eq(target_year)][
            ["detector_id", "balanced_annual_volume_proxy"]
        ].rename(columns={"balanced_annual_volume_proxy": "proxy_target"})
        merged = base.merge(target, on="detector_id", how="inner", validate="one_to_one")
        merged = merged[merged["proxy_base"].gt(0) & merged["proxy_target"].gt(0)].copy()
        merged["detector_ratio"] = merged["proxy_target"] / merged["proxy_base"]
        merged["base_year"] = base_year
        merged["target_year"] = target_year
        merged["transition"] = f"{base_year}-{target_year}"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_temporal_predictions(
    panel: pd.DataFrame,
    stable: pd.DataFrame,
    detectors: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    station_pairs = build_station_pairs(panel)
    detector_ratios = build_detector_ratios(stable)
    location_fields = detectors[["detector_id", "spatial_fold"]].rename(
        columns={"spatial_fold": "detector_fold"}
    )
    detector_ratios = detector_ratios.merge(
        location_fields, on="detector_id", how="left", validate="many_to_one"
    )
    rows: list[pd.DataFrame] = []

    if not crosswalk.empty:
        colocated = station_pairs.merge(
            crosswalk[["station_id", "detector_id", "distance_m", "road_name_similarity"]],
            on="station_id",
            how="inner",
            validate="many_to_one",
        ).merge(
            detector_ratios[["detector_id", "transition", "detector_ratio"]],
            on=["detector_id", "transition"],
            how="inner",
            validate="many_to_one",
        )
        colocated["task"] = "colocated_temporal_transfer"
        colocated["temporal_factor"] = colocated["detector_ratio"]
        colocated["prediction"] = colocated["aadt_base"] * colocated["temporal_factor"]
        colocated["predicted_change"] = colocated["prediction"] - colocated["aadt_base"]
        colocated["cluster_id"] = colocated["detector_id"]
        colocated["factor_training_detector_count"] = 1
        rows.append(colocated)

    for transition, station_transition in station_pairs.groupby("transition"):
        ratios = detector_ratios[detector_ratios["transition"].eq(transition)].copy()
        available_folds = sorted(
            pd.to_numeric(station_transition["spatial_fold"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        for fold in available_folds:
            detector_folds = pd.to_numeric(ratios["detector_fold"], errors="coerce")
            training_ratios = ratios[
                detector_folds.notna() & detector_folds.ne(fold)
            ]["detector_ratio"].dropna()
            if training_ratios.empty:
                continue
            factor = float(training_ratios.median())
            heldout = station_transition[
                pd.to_numeric(station_transition["spatial_fold"], errors="coerce").eq(fold)
            ].copy()
            heldout["task"] = "heldout_network_factor"
            heldout["detector_id"] = ""
            heldout["distance_m"] = np.nan
            heldout["road_name_similarity"] = np.nan
            heldout["temporal_factor"] = factor
            heldout["prediction"] = heldout["aadt_base"] * factor
            heldout["predicted_change"] = heldout["prediction"] - heldout["aadt_base"]
            heldout["cluster_id"] = heldout["station_id"]
            heldout["factor_training_detector_count"] = len(training_ratios)
            rows.append(heldout)

    predictions = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not predictions.empty:
        predictions["no_change_prediction"] = predictions["aadt_base"]
        predictions["no_change_predicted_change"] = 0.0
        predictions["model_absolute_change_error"] = (
            predictions["predicted_change"] - predictions["observed_change"]
        ).abs()
        predictions["no_change_absolute_change_error"] = predictions["observed_change"].abs()
        predictions["absolute_error_difference_model_minus_no_change"] = (
            predictions["model_absolute_change_error"]
            - predictions["no_change_absolute_change_error"]
        )
    save_csv(predictions, PREDICTION_PATH)
    return predictions


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 3 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return np.nan
    return float(valid["left"].corr(valid["right"]))


def metric_row(group: pd.DataFrame) -> dict[str, object]:
    model_mae = group["model_absolute_change_error"].mean()
    no_change_mae = group["no_change_absolute_change_error"].mean()
    improvement = (
        100.0 * (no_change_mae - model_mae) / no_change_mae
        if no_change_mae > 0
        else np.nan
    )
    observed_sign = np.sign(group["observed_change"].to_numpy(float))
    predicted_sign = np.sign(group["predicted_change"].to_numpy(float))
    return {
        "n": len(group),
        "cluster_count": group["cluster_id"].nunique(),
        "model_change_mae": model_mae,
        "no_change_change_mae": no_change_mae,
        "mae_improvement_pct_vs_no_change": improvement,
        "model_change_rmse": float(
            np.sqrt(np.mean((group["predicted_change"] - group["observed_change"]) ** 2))
        ),
        "change_correlation": safe_correlation(
            group["observed_change"], group["predicted_change"]
        ),
        "direction_accuracy": float(np.mean(observed_sign == predicted_sign)),
        "mean_observed_change": group["observed_change"].mean(),
        "mean_predicted_change": group["predicted_change"].mean(),
    }


def clustered_bootstrap_interval(
    frame: pd.DataFrame, cluster_column: str = "cluster_id"
) -> tuple[float, float, float, int]:
    cluster_means = (
        frame.groupby(cluster_column)["absolute_error_difference_model_minus_no_change"]
        .mean()
        .dropna()
        .to_numpy(float)
    )
    if len(cluster_means) < 2:
        return np.nan, np.nan, np.nan, len(cluster_means)
    rng = np.random.default_rng(RANDOM_SEED)
    statistics = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.choice(cluster_means, size=len(cluster_means), replace=True)
        statistics[index] = sample.mean()
    return (
        float(cluster_means.mean()),
        float(np.quantile(statistics, 0.025)),
        float(np.quantile(statistics, 0.975)),
        len(cluster_means),
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    if predictions.empty:
        transition = pd.DataFrame()
        folds = pd.DataFrame()
        comparison = pd.DataFrame()
        save_csv(transition, TRANSITION_METRICS_PATH)
        save_csv(folds, FOLD_METRICS_PATH)
        save_csv(comparison, PAIRED_COMPARISON_PATH)
        return transition, folds, comparison

    for (task, transition_name), group in predictions.groupby(["task", "transition"]):
        transition_rows.append(
            {"task": task, "transition": transition_name, **metric_row(group)}
        )
    transition = pd.DataFrame(transition_rows)
    save_csv(transition, TRANSITION_METRICS_PATH)

    fold_frame = predictions.dropna(subset=["spatial_fold"]).copy()
    for (task, fold), group in fold_frame.groupby(["task", "spatial_fold"]):
        fold_rows.append(
            {"task": task, "spatial_fold": int(fold), **metric_row(group)}
        )
    folds = pd.DataFrame(fold_rows)
    save_csv(folds, FOLD_METRICS_PATH)

    for task, group in predictions.groupby("task"):
        summary = metric_row(group)
        difference, lower, upper, cluster_count = clustered_bootstrap_interval(group)
        task_transition = transition[transition["task"].eq(task)]
        task_folds = folds[folds["task"].eq(task)]
        improved_transitions = int(
            task_transition["mae_improvement_pct_vs_no_change"].gt(0).sum()
        )
        improved_folds = int(task_folds["mae_improvement_pct_vs_no_change"].gt(0).sum())
        gate = (
            summary["mae_improvement_pct_vs_no_change"] >= MIN_MAE_IMPROVEMENT_PCT
            and upper < 0
            and improved_transitions >= MIN_IMPROVED_TRANSITIONS
            and improved_folds >= MIN_IMPROVED_FOLDS
            and summary["change_correlation"] > 0
        )
        failed: list[str] = []
        if not summary["mae_improvement_pct_vs_no_change"] >= MIN_MAE_IMPROVEMENT_PCT:
            failed.append("pooled_effect_below_5pct")
        if not upper < 0:
            failed.append("cluster_interval_includes_zero")
        if improved_transitions < MIN_IMPROVED_TRANSITIONS:
            failed.append("fewer_than_2_of_3_transitions_improve")
        if improved_folds < MIN_IMPROVED_FOLDS:
            failed.append("fewer_than_3_of_5_folds_improve")
        if not summary["change_correlation"] > 0:
            failed.append("change_correlation_not_positive")
        comparison_rows.append(
            {
                "task": task,
                **summary,
                "mean_absolute_error_difference_model_minus_no_change": difference,
                "cluster_bootstrap_lower_95": lower,
                "cluster_bootstrap_upper_95": upper,
                "bootstrap_cluster_count": cluster_count,
                "improved_transition_count": improved_transitions,
                "improved_spatial_fold_count": improved_folds,
                "predeclared_task_gate_pass": gate,
                "failed_criterion": "|".join(failed),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, PAIRED_COMPARISON_PATH)
    return transition, folds, comparison


def archive_sample_gate(audit: pd.DataFrame) -> tuple[bool, str]:
    evidence: list[str] = []
    passed = True
    for year in PRIMARY_YEARS:
        subset = audit[audit["year"].eq(year)]
        obtained = subset[subset["status"].eq("obtained")]
        count = len(obtained)
        months = obtained["month"].nunique()
        day_types = obtained["day_type"].nunique()
        blocks = obtained["time_block"].nunique()
        year_pass = (
            count >= math.ceil(EXPECTED_SNAPSHOTS_PER_YEAR * MIN_YEAR_SAMPLE_SHARE)
            and months == len(COMMON_MONTHS)
            and day_types == len(DAY_TYPES)
            and blocks == len(TIME_BLOCKS)
        )
        passed = passed and year_pass
        evidence.append(
            f"{year}:obtained={count}/{EXPECTED_SNAPSHOTS_PER_YEAR},months={months},day_types={day_types},blocks={blocks}"
        )
    return passed, "; ".join(evidence)


def make_decisions(
    sampling_audit: pd.DataFrame,
    stable: pd.DataFrame,
    detectors: pd.DataFrame,
    crosswalk: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    archive_pass, archive_evidence = archive_sample_gate(sampling_audit)
    stable_count = stable["detector_id"].nunique() if not stable.empty else 0
    linked_location_count = detectors["detector_id"].nunique() if not detectors.empty else 0
    stable_folds = (
        pd.to_numeric(detectors["spatial_fold"], errors="coerce").dropna().astype(int).nunique()
        if not detectors.empty
        else 0
    )
    stable_panel_pass = stable_count >= MIN_STABLE_DETECTORS
    location_linkage_pass = (
        linked_location_count >= MIN_STABLE_DETECTORS
        and stable_folds >= MIN_STABLE_SPATIAL_FOLDS
    )
    panel_pass = stable_panel_pass and location_linkage_pass
    crosswalk_pass = len(crosswalk) >= MIN_CROSSWALK_PAIRS
    task_records = (
        comparison.set_index("task").to_dict("index") if not comparison.empty else {}
    )
    colocated_record = task_records.get("colocated_temporal_transfer", {})
    heldout_record = task_records.get("heldout_network_factor", {})
    colocated_pass = bool(colocated_record.get("predeclared_task_gate_pass", False))
    heldout_pass = bool(heldout_record.get("predeclared_task_gate_pass", False))

    def task_evidence(record: dict[str, object]) -> str:
        if not record:
            return "no evaluable paired predictions"
        return (
            f"MAE improvement={record.get('mae_improvement_pct_vs_no_change', np.nan):.2f}%; "
            f"cluster interval=[{record.get('cluster_bootstrap_lower_95', np.nan):.1f},"
            f" {record.get('cluster_bootstrap_upper_95', np.nan):.1f}]; "
            f"improved transitions={int(record.get('improved_transition_count', 0))}/3; "
            f"improved folds={int(record.get('improved_spatial_fold_count', 0))}/5; "
            f"change correlation={record.get('change_correlation', np.nan):.3f}"
        )
    final_pass = archive_pass and panel_pass and crosswalk_pass and colocated_pass and heldout_pass

    rows = [
        {
            "decision": "public_archive_fixed_sample_materialised",
            "pass": archive_pass,
            "evidence": archive_evidence,
            "failed_criterion": "" if archive_pass else "one_or_more_years_fail_90pct_and_complete_strata_rule",
            "action": "annualise the fixed sample" if archive_pass else "do not interpret retrieval success as a comparable annual panel",
        },
        {
            "decision": "stable_detector_panel_has_minimum_support",
            "pass": stable_panel_pass,
            "evidence": f"stable detectors={stable_count}; required detectors>={MIN_STABLE_DETECTORS}",
            "failed_criterion": "" if stable_panel_pass else "stable_detector_count_below_threshold",
            "action": "attempt linkage to current locations" if stable_panel_pass else "do not generalise the sampled proxy across the monitored network",
        },
        {
            "decision": "stable_detector_ids_link_to_current_locations_and_folds",
            "pass": location_linkage_pass,
            "evidence": f"linked stable detectors={linked_location_count}; spatial folds={stable_folds}; required detectors>={MIN_STABLE_DETECTORS}, folds>={MIN_STABLE_SPATIAL_FOLDS}",
            "failed_criterion": "" if location_linkage_pass else "current_location_id_linkage_or_fold_support_below_threshold",
            "action": "evaluate spatial temporal transfer" if location_linkage_pass else "inspect ID namespace and do not run a spatial transfer claim",
        },
        {
            "decision": "current_coordinate_detector_atc_crosswalk_has_minimum_support",
            "pass": crosswalk_pass,
            "evidence": f"accepted one-to-one major-road pairs={len(crosswalk)}; required>={MIN_CROSSWALK_PAIRS}; historical detector coordinates are unavailable",
            "failed_criterion": "" if crosswalk_pass else "accepted_pair_count_below_threshold",
            "action": "run colocated transfer with survivor/current-location limitation" if crosswalk_pass else "do not claim colocated temporal validation",
        },
        {
            "decision": "colocated_detector_change_beats_no_change",
            "pass": colocated_pass,
            "evidence": task_evidence(colocated_record),
            "failed_criterion": "" if colocated_pass else str(colocated_record.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as necessary calibration evidence" if colocated_pass else "do not transfer detector ratios to ATC AADT",
        },
        {
            "decision": "heldout_network_factor_beats_no_change",
            "pass": heldout_pass,
            "evidence": task_evidence(heldout_record) + "; factors exclude the held-out spatial fold",
            "failed_criterion": "" if heldout_pass else str(heldout_record.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as deployment-oriented temporal evidence" if heldout_pass else "do not downscale the monitored-network trend",
        },
        {
            "decision": "step25b_major_road_temporal_downscaling_authorised",
            "pass": final_pass,
            "evidence": f"archive={archive_pass}; stable_panel={stable_panel_pass}; location_linkage={location_linkage_pass}; crosswalk={crosswalk_pass}; colocated={colocated_pass}; heldout={heldout_pass}",
            "failed_criterion": "" if final_pass else "at_least_one_required_gate_failed",
            "action": "proceed only to a bounded major-road temporal experiment" if final_pass else "stop before major-road temporal downscaling and report the failed gate",
        },
        {
            "decision": "full_network_local_road_backcast_or_equity_trend_authorised",
            "pass": False,
            "evidence": "Step 25A uses strategic detectors and measured major-road ATC stations; Step 24 found no representative local-road dynamic label support",
            "failed_criterion": "estimand_outside_step25a_support",
            "action": "keep local-road reconstruction and equity trends outside the authorised claims",
        },
    ]
    decisions = pd.DataFrame(rows)
    save_csv(decisions, DECISION_PATH)
    return decisions


def write_figures(
    sampling_audit: pd.DataFrame,
    annual_audit: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    obtained = (
        sampling_audit.assign(obtained=sampling_audit["status"].eq("obtained"))
        .groupby("year")["obtained"]
        .sum()
        .reindex(PRIMARY_YEARS, fill_value=0)
    )
    axes[0].bar([str(year) for year in PRIMARY_YEARS], obtained, color="#2878B5")
    axes[0].axhline(
        math.ceil(EXPECTED_SNAPSHOTS_PER_YEAR * MIN_YEAR_SAMPLE_SHARE),
        color="#C44E52",
        linestyle="--",
        label="90% gate",
    )
    axes[0].set_title("Fixed archive snapshots obtained")
    axes[0].set_ylabel(f"Snapshots (expected {EXPECTED_SNAPSHOTS_PER_YEAR})")
    axes[0].legend(frameon=False)

    if not annual_audit.empty:
        axes[1].plot(
            annual_audit["year"],
            annual_audit["qualified_detector_years"],
            marker="o",
            label="qualified detector-years",
        )
        axes[1].plot(
            annual_audit["year"],
            annual_audit["stable_panel_detectors"],
            marker="s",
            label="stable four-year panel",
        )
    axes[1].axhline(MIN_STABLE_DETECTORS, color="#C44E52", linestyle="--", label="support gate")
    axes[1].set_title("Annual proxy support")
    axes[1].set_ylabel("Unique detectors")
    axes[1].set_xticks(PRIMARY_YEARS)
    axes[1].legend(frameon=False)
    figure.suptitle("Step 25A archive and stable-panel gates")
    figure.tight_layout()
    figure.savefig(COVERAGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    tasks = ["colocated_temporal_transfer", "heldout_network_factor"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for axis, task in zip(axes, tasks):
        subset = predictions[predictions["task"].eq(task)] if not predictions.empty else pd.DataFrame()
        if not subset.empty:
            axis.scatter(
                subset["observed_change"],
                subset["predicted_change"],
                alpha=0.5,
                s=22,
                color="#2878B5",
            )
            limit = float(
                np.nanmax(
                    np.abs(
                        np.concatenate(
                            [subset["observed_change"].to_numpy(), subset["predicted_change"].to_numpy()]
                        )
                    )
                )
            )
            if np.isfinite(limit) and limit > 0:
                axis.plot([-limit, limit], [-limit, limit], linestyle="--", color="black", linewidth=1)
                axis.axhline(0, color="grey", linewidth=0.8)
                axis.axvline(0, color="grey", linewidth=0.8)
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("Observed AADT change (vehicles/day)")
        axis.set_ylabel("Predicted AADT change (vehicles/day)")
    figure.suptitle("Does the public detector proxy identify measured annual change?")
    figure.tight_layout()
    figure.savefig(CHANGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {CHANGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def update_report_manifest() -> None:
    rows = [
        (ARCHIVE_INVENTORY_PATH, "reportable_data_audit", "year-level public archive inventory; timestamp lists may be capped"),
        (SAMPLING_AUDIT_PATH, "reportable_data_audit", "fixed-date and fixed-time archive materialisation record"),
        (ANNUAL_AUDIT_PATH, "reportable_data_audit", "balanced proxy and stable detector support"),
        (DETECTOR_ID_AUDIT_PATH, "reportable_data_audit", "historical feed ID overlap with the current strategic-detector location table"),
        (CROSSWALK_AUDIT_PATH, "reportable_validation_audit", "current-coordinate major-road crosswalk support and limitation"),
        (TRANSITION_METRICS_PATH, "reportable_validation_result", "strict annual change metrics by transition"),
        (FOLD_METRICS_PATH, "reportable_validation_result", "strict annual change metrics by frozen spatial fold"),
        (PAIRED_COMPARISON_PATH, "reportable_validation_result", "paired no-change comparison with clustered interval"),
        (DECISION_PATH, "reportable_decision_audit", "predeclared Step 25A gates and authorised claim boundary"),
        (COVERAGE_FIGURE_PATH, "reportable_data_audit", "archive and stable-panel support"),
        (CHANGE_FIGURE_PATH, "reportable_validation_result", "observed versus predicted annual AADT change"),
        (SNAPSHOT_LONG_PATH, "provenance_only", "sampled public strategic-detector observations"),
        (ANNUAL_PROXY_PATH, "provenance_only", "balanced annual detector proxy; not AADT"),
        (STABLE_PANEL_PATH, "provenance_only", "qualified stable detector panel"),
        (CROSSWALK_PATH, "provenance_only", "current-coordinate detector-to-ATC links"),
        (PREDICTION_PATH, "provenance_only", "paired temporal predictions and errors"),
    ]
    additions = pd.DataFrame(
        [
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "status": status,
                "reason": reason,
                "step": "25A",
            }
            for path, status, reason in rows
        ]
    )
    if REPORT_MANIFEST_PATH.exists():
        existing = pd.read_csv(REPORT_MANIFEST_PATH)
        for column in additions.columns:
            if column not in existing:
                existing[column] = ""
        for column in existing.columns:
            if column not in additions:
                additions[column] = ""
        existing = existing[
            ~existing.get("step", pd.Series(index=existing.index, dtype=str)).astype(str).eq("25A")
        ]
        additions = additions[existing.columns]
        output = pd.concat([existing, additions], ignore_index=True)
    else:
        output = additions
    save_csv(output, REPORT_MANIFEST_PATH)


def run_sample_phase(refresh: bool, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = obtain_samples(refresh, workers)
    long = parse_all_samples(audit)
    annual, stable = annualise_detector_proxy(long)
    return audit, annual, stable


def load_sample_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for path in (SAMPLING_AUDIT_PATH, ANNUAL_PROXY_PATH, STABLE_PANEL_PATH)
        if not path.exists()
    ]
    if missing:
        relative = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Step 25A sample outputs are missing: {relative}. Run --phase sample first."
        )
    return (
        pd.read_csv(SAMPLING_AUDIT_PATH),
        pd.read_csv(ANNUAL_PROXY_PATH),
        pd.read_csv(STABLE_PANEL_PATH),
    )


def run_evaluate_phase(
    sampling_audit: pd.DataFrame, annual: pd.DataFrame, stable: pd.DataFrame
) -> pd.DataFrame:
    for path in (MEASURED_PANEL_PATH, DETECTOR_LOCATION_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input is missing: {path.relative_to(PROJECT_ROOT)}. "
                "Run Steps 18 and 24.1 before Step 25A evaluation."
            )
    stable_ids = set(stable["detector_id"].map(normalise_identifier)) if not stable.empty else set()
    detectors = prepare_detector_locations(stable_ids)
    panel = prepare_measured_panel()
    crosswalk, _ = build_crosswalk(panel, detectors)
    predictions = build_temporal_predictions(panel, stable, detectors, crosswalk)
    _, _, comparison = evaluate_predictions(predictions)
    decisions = make_decisions(
        sampling_audit, stable, detectors, crosswalk, comparison
    )
    annual_audit = pd.read_csv(ANNUAL_AUDIT_PATH) if ANNUAL_AUDIT_PATH.exists() else pd.DataFrame()
    write_figures(sampling_audit, annual_audit, predictions)
    update_report_manifest()
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("inventory", "sample", "evaluate", "all"),
        default="all",
        help="inventory public coverage, materialise fixed samples, evaluate, or run all phases",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh cached API queries and XML snapshots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help=f"concurrent sampling-day downloads (default {DOWNLOAD_WORKERS})",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be between 1 and 4")

    for directory in (RAW_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Step 25A revision: {STEP25A_REVISION}")

    if args.phase in {"inventory", "all"}:
        print("Inventorying the public strategic-detector archive...")
        inventory_archive(args.refresh)

    if args.phase in {"sample", "all"}:
        print(
            "Materialising the fixed 2021-2024 March-December strategic-detector sample..."
        )
        sampling_audit, annual, stable = run_sample_phase(args.refresh, args.workers)
    elif args.phase == "evaluate":
        sampling_audit, annual, stable = load_sample_outputs()
    else:
        sampling_audit = annual = stable = pd.DataFrame()

    if args.phase in {"evaluate", "all"}:
        print("Running the two predeclared temporal change gates...")
        decisions = run_evaluate_phase(sampling_audit, annual, stable)
        print("\nStep 25A strategic-detector temporal-signal gate is complete.")
        for row in decisions.to_dict("records"):
            if row["decision"] in {
                "public_archive_fixed_sample_materialised",
                "stable_detector_panel_has_minimum_support",
                "stable_detector_ids_link_to_current_locations_and_folds",
                "colocated_detector_change_beats_no_change",
                "heldout_network_factor_beats_no_change",
                "step25b_major_road_temporal_downscaling_authorised",
                "full_network_local_road_backcast_or_equity_trend_authorised",
            }:
                print(f"  {row['decision']}: {row['pass']}")
        print(
            "  The balanced detector proxy is not AADT. Passing authorises only a bounded major-road next step."
        )


if __name__ == "__main__":
    main()
