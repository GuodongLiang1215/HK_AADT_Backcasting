"""Step 17: build a conservative annual ATC panel and test temporal identification.

The official structured ATC package contains yearly archives for 2018--2024.
Each archive has one ``Current/Sxxxx.xls`` workbook for a conservative subset
of directly measured core and Coverage (B) stations.  Those files are used here
because the archive-wide GISDB XML
also contains estimated AADT values but does not preserve the measured versus
estimated flag needed for a clean modelling label.

This step answers three separate questions.

1. Do the official files provide a continuous, internally consistent panel?
2. How well can next-year AADT be forecast at a station whose past AADT is
   already known?
3. When a station is held out as a location in every year, do annual refits or
   a simple road-hierarchy trend recover its year-to-year change better than a
   no-change baseline?

The third question is the relevant gate for an unmonitored-road backcast.
Current station coordinates are used only to form spatial blocks and are not
claimed to be historical station geometries.  No hyperparameter search is run.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hk_aadt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "official_sources.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "atc" / "annual"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PACKAGE_PATH = RAW_DIR / "ATC_TRAFFIC_DATA.zip"
CURRENT_POINTS_PATH = PROCESSED_DIR / "atc_current_station_points.csv"
STEP3_LABEL_PATH = PROCESSED_DIR / "atc_appendix_b_station_year.csv"

PANEL_PATH = PROCESSED_DIR / "atc_step17_directly_measured_annual_panel.csv"
KNOWN_PREDICTION_PATH = PROCESSED_DIR / "atc_step17_known_station_predictions.csv"
SPATIOTEMPORAL_PREDICTION_PATH = (
    PROCESSED_DIR / "atc_step17_spatiotemporal_predictions.csv"
)

SOURCE_AUDIT_PATH = TABLE_DIR / "step17_source_extraction_audit.csv"
PANEL_AUDIT_PATH = TABLE_DIR / "step17_annual_panel_audit.csv"
METADATA_AUDIT_PATH = TABLE_DIR / "step17_station_metadata_stability.csv"
KNOWN_METRIC_PATH = TABLE_DIR / "step17_known_station_forward_metrics.csv"
SPATIOTEMPORAL_METRIC_PATH = (
    TABLE_DIR / "step17_spatiotemporal_forward_metrics.csv"
)
CHANGE_METRIC_PATH = TABLE_DIR / "step17_spatiotemporal_change_metrics.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step17_temporal_identification_decision.csv"

LEVEL_FIGURE_PATH = FIGURE_DIR / "step17_measured_station_annual_levels.png"
FORWARD_FIGURE_PATH = FIGURE_DIR / "step17_forward_level_mae.png"
CHANGE_FIGURE_PATH = FIGURE_DIR / "step17_change_identification.png"

YEARS = tuple(range(2018, 2025))
TARGET_YEARS = tuple(range(2019, 2025))
CHANGE_YEARS = tuple(range(2020, 2025))
FOLDS = tuple(range(1, 6))
RANDOM_SEED = 42

STATION_MEMBER_PATTERN = re.compile(
    r"(?:^|/)(?P<version>Current|Revision\s+(?P<revision>\d+))/"
    r"S(?P<station_id>\d{4,5})\.xls$",
    re.IGNORECASE,
)

KNOWN_MODEL_ORDER = [
    "persistence",
    "expanding_station_mean",
    "station_linear_trend",
]
SPATIOTEMPORAL_MODEL_ORDER = [
    "training_median",
    "latest_year_road_hierarchy",
    "pooled_static_hgb",
    "latest_year_static_hgb",
    "road_hierarchy_linear_trend",
]

MODEL_LABELS = {
    "persistence": "Previous-year AADT",
    "expanding_station_mean": "Station expanding mean",
    "station_linear_trend": "Station linear trend",
    "training_median": "Prior-years median",
    "latest_year_road_hierarchy": "Latest-year hierarchy lookup",
    "pooled_static_hgb": "Pooled static HGB",
    "latest_year_static_hgb": "Latest-year static HGB",
    "road_hierarchy_linear_trend": "Hierarchy linear trend",
    "zero_change": "No-change baseline",
}

MODEL_COLORS = {
    "persistence": "#2E86AB",
    "expanding_station_mean": "#7F8C8D",
    "station_linear_trend": "#D35400",
    "training_median": "#B0B7BC",
    "latest_year_road_hierarchy": "#2E86AB",
    "pooled_static_hgb": "#7F8C8D",
    "latest_year_static_hgb": "#D35400",
    "road_hierarchy_linear_trend": "#6C5CE7",
    "zero_change": "#1F1F1F",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def read_package_url() -> str:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing source configuration: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    matches = [
        source["url"]
        for source in config.get("later_sources", [])
        if source.get("name") == "ATC traffic package"
    ]
    if len(matches) != 1:
        raise ValueError("Expected one ATC traffic package in official_sources.json.")
    return str(matches[0])


def download_package() -> None:
    if PACKAGE_PATH.exists() and PACKAGE_PATH.stat().st_size > 0:
        print(f"Already available: {PACKAGE_PATH.relative_to(PROJECT_ROOT)}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = PACKAGE_PATH.with_suffix(".download")
    print("Downloading official ATC structured package (2018-2024)...")
    with urlopen(read_package_url(), timeout=180) as response:
        with temporary_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if chunk:
                    output_file.write(chunk)
    temporary_path.replace(PACKAGE_PATH)
    print(
        f"Saved: {PACKAGE_PATH.relative_to(PROJECT_ROOT)} "
        f"({PACKAGE_PATH.stat().st_size / (1024 * 1024):.1f} MB)"
    )


def normalise_label(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def next_nonblank_value(sheet: xlrd.sheet.Sheet, row: int, column: int) -> object:
    for candidate_column in range(column + 1, sheet.ncols):
        value = sheet.cell_value(row, candidate_column)
        if str(value).strip():
            return value
    return ""


def parse_measured_station_workbook(
    workbook_bytes: bytes,
    expected_year: int,
    expected_station_id: int,
) -> dict[str, object]:
    workbook = xlrd.open_workbook(file_contents=workbook_bytes, on_demand=True)
    sheet = workbook.sheet_by_index(0)

    metadata: dict[str, object] = {}
    metadata_labels = {
        "YEAR",
        "CORESTATION",
        "COVERAGEBSTATION",
        "ROADNETWORK",
        "ROADTYPE",
        "LINK",
    }
    for row in range(min(10, sheet.nrows)):
        for column in range(sheet.ncols):
            label = normalise_label(sheet.cell_value(row, column))
            if label in metadata_labels:
                metadata[label] = next_nonblank_value(sheet, row, column)

    all_day_columns = [
        column
        for row in range(sheet.nrows)
        for column in range(sheet.ncols)
        if normalise_label(sheet.cell_value(row, column)) == "ALLDAY"
    ]
    if not all_day_columns:
        raise ValueError("All-Day column not found.")
    all_day_column = all_day_columns[0]

    direction_values: list[float] = []
    for row in range(sheet.nrows):
        row_has_aadt = any(
            normalise_label(sheet.cell_value(row, column)) == "AADT"
            for column in range(sheet.ncols)
        )
        if not row_has_aadt:
            continue
        value = sheet.cell_value(row, all_day_column)
        if isinstance(value, (int, float)) and float(value) > 0:
            direction_values.append(float(value))

    workbook.release_resources()

    parsed_year = int(float(metadata.get("YEAR", -1)))
    station_label = (
        "CORESTATION" if "CORESTATION" in metadata else "COVERAGEBSTATION"
    )
    parsed_station_id = int(float(metadata.get(station_label, -1)))
    if parsed_year != expected_year:
        raise ValueError(f"Workbook year {parsed_year} != archive year {expected_year}.")
    if parsed_station_id != expected_station_id:
        raise ValueError(
            f"Workbook station {parsed_station_id} != filename station "
            f"{expected_station_id}."
        )
    if not direction_values:
        raise ValueError("No positive All-Day AADT direction values found.")

    return {
        "year": parsed_year,
        "station_id": parsed_station_id,
        "aadt": int(round(sum(direction_values))),
        "direction_count": len(direction_values),
        "station_class": "core" if station_label == "CORESTATION" else "coverage_b",
        "road_network_reported": str(metadata.get("ROADNETWORK", "")).strip().upper(),
        "road_type_reported": str(metadata.get("ROADTYPE", "")).strip().upper(),
        "link_description_reported": str(metadata.get("LINK", "")).strip(),
    }


def parse_official_package() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ZipFile(PACKAGE_PATH) as outer_archive:
        outer_names = set(outer_archive.namelist())
        for year in YEARS:
            nested_name = f"ATC_TRAFFIC_DATA/{year}.zip"
            if nested_name not in outer_names:
                raise ValueError(f"Official package is missing {nested_name}.")
            print(f"Parsing directly measured station workbooks for {year}...")
            with ZipFile(BytesIO(outer_archive.read(nested_name))) as annual_archive:
                selected_members: dict[int, tuple[int, str, str]] = {}
                for member in annual_archive.namelist():
                    match = STATION_MEMBER_PATTERN.search(member)
                    if match is None:
                        continue
                    station_id = int(match.group("station_id"))
                    revision = int(match.group("revision") or 0)
                    version = str(match.group("version"))
                    current = selected_members.get(station_id)
                    if current is None or revision > current[0]:
                        selected_members[station_id] = (revision, member, version)
                if not selected_members:
                    raise ValueError(
                        f"No Current or Revision station workbooks found for {year}."
                    )
                for station_id, (revision, member, version) in sorted(
                    selected_members.items()
                ):
                    record = parse_measured_station_workbook(
                        annual_archive.read(member), year, station_id
                    )
                    record.update(
                        {
                            "directly_measured_station_workbook": True,
                            "source_nested_archive": f"{year}.zip",
                            "source_member": member,
                            "source_workbook_version": version,
                            "source_revision_number": revision,
                        }
                    )
                    rows.append(record)

    panel = pd.DataFrame(rows)
    if panel.duplicated(["year", "station_id"]).any():
        duplicates = panel[panel.duplicated(["year", "station_id"], keep=False)]
        raise ValueError(
            "Duplicate measured station-year rows found: "
            f"{duplicates[['year', 'station_id']].to_dict('records')[:10]}"
        )
    if set(panel["year"].astype(int)) != set(YEARS):
        raise ValueError("Parsed panel does not contain every year from 2018 to 2024.")
    if (panel.groupby("year").size() < 100).any():
        raise ValueError("A year has fewer than 100 directly measured stations.")
    return freeze_station_metadata(panel)


def freeze_station_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    """Use only information available at a station's first panel appearance.

    The official workbooks occasionally reclassify Road Network, Road Type, or
    the written link description.  Feeding the target-year version of those
    fields into a forward validation would leak target-year information.  The
    reported fields are retained for audit, while the modelling fields below
    are frozen at the first observed year for each station.
    """
    ordered = panel.sort_values(["station_id", "year"])
    first = (
        ordered.groupby("station_id", as_index=False)
        .first()[
            [
                "station_id",
                "road_network_reported",
                "road_type_reported",
                "link_description_reported",
            ]
        ]
        .rename(
            columns={
                "road_network_reported": "road_network",
                "road_type_reported": "road_type",
                "link_description_reported": "link_description",
            }
        )
    )
    return panel.merge(first, on="station_id", how="left", validate="many_to_one")


def attach_current_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    if not CURRENT_POINTS_PATH.exists():
        raise FileNotFoundError("Missing Step 5 station points. Complete Step 5 first.")
    points = pd.read_csv(
        CURRENT_POINTS_PATH,
        usecols=[
            "station_id",
            "longitude",
            "latitude",
            "geometry_reference",
            "historical_geometry_status",
        ],
    )
    output = panel.merge(points, on="station_id", how="left", validate="many_to_one")
    return output


def validate_2021_against_step3(panel: pd.DataFrame) -> dict[str, object]:
    if not STEP3_LABEL_PATH.exists():
        raise FileNotFoundError("Missing Step 3 labels. Complete Step 3 first.")
    step3 = pd.read_csv(STEP3_LABEL_PATH)
    measured = step3[
        (step3["year"] == 2021) & (step3["primary_label_eligible"] == True)  # noqa: E712
    ][["station_id", "aadt_current"]]
    current = panel[panel["year"] == 2021][["station_id", "aadt"]]
    comparison = current.merge(measured, on="station_id", how="left", validate="one_to_one")
    comparison["absolute_difference"] = (
        comparison["aadt"] - comparison["aadt_current"]
    ).abs()
    matched = int(comparison["aadt_current"].notna().sum())
    exact = int((comparison["absolute_difference"].fillna(np.inf) < 1).sum())
    if matched != len(current) or exact != len(current):
        raise ValueError(
            "The structured 2021 measured-station files do not reproduce all matching "
            "Step 3 measured labels exactly."
        )
    return {
        "check": "structured_2021_measured_aadt_vs_step3_pdf",
        "structured_rows": len(current),
        "rows_matched_to_step3_measured_labels": matched,
        "exact_aadt_matches": exact,
        "maximum_absolute_difference": round(
            float(comparison["absolute_difference"].max()), 4
        ),
        "status": "pass",
    }


def assign_spatial_folds(panel: pd.DataFrame) -> pd.DataFrame:
    stations = (
        panel[["station_id", "longitude", "latitude"]]
        .drop_duplicates("station_id")
        .dropna(subset=["longitude", "latitude"])
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    coordinates = stations[["longitude", "latitude"]].to_numpy(dtype=float)
    labels = KMeans(
        n_clusters=len(FOLDS),
        random_state=RANDOM_SEED,
        n_init=20,
    ).fit_predict(coordinates)
    stations["spatial_fold"] = labels + 1
    output = panel.merge(
        stations[["station_id", "spatial_fold"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    output["spatial_fold"] = output["spatial_fold"].astype("Int64")
    return output.sort_values(["year", "station_id"]).reset_index(drop=True)


def correlation(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_source_audit(
    panel: pd.DataFrame,
    step3_validation: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, group in panel.groupby("year"):
        rows.append(
            {
                "check": "yearly_directly_measured_station_workbooks",
                "year": int(year),
                "rows": len(group),
                "unique_stations": group["station_id"].nunique(),
                "one_direction_stations": int((group["direction_count"] == 1).sum()),
                "two_direction_stations": int((group["direction_count"] == 2).sum()),
                "missing_aadt": int(group["aadt"].isna().sum()),
                "missing_current_geometry": int(group["longitude"].isna().sum()),
                "revised_workbooks_used": int(
                    (group["source_revision_number"] > 0).sum()
                ),
                "status": "pass",
            }
        )
    rows.append(
        {
            "check": step3_validation["check"],
            "year": 2021,
            "rows": step3_validation["structured_rows"],
            "unique_stations": step3_validation[
                "rows_matched_to_step3_measured_labels"
            ],
            "one_direction_stations": "",
            "two_direction_stations": "",
            "missing_aadt": step3_validation["maximum_absolute_difference"],
            "missing_current_geometry": int(
                panel.loc[panel["year"] == 2021, "longitude"].isna().sum()
            ),
            "revised_workbooks_used": int(
                (
                    panel.loc[panel["year"] == 2021, "source_revision_number"]
                    > 0
                ).sum()
            ),
            "status": step3_validation["status"],
        }
    )
    return rows


def build_panel_audit(panel: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous: pd.DataFrame | None = None
    for year in YEARS:
        current = panel[panel["year"] == year]
        row: dict[str, object] = {
            "year": year,
            "station_count": len(current),
            "mean_aadt": round(float(current["aadt"].mean()), 4),
            "median_aadt": round(float(current["aadt"].median()), 4),
            "total_aadt_across_measured_stations": int(current["aadt"].sum()),
            "stations_shared_with_previous_year": "",
            "level_correlation_with_previous_year": "",
            "mean_change_from_previous_year": "",
            "median_change_from_previous_year": "",
            "mean_absolute_change_from_previous_year": "",
        }
        if previous is not None:
            paired = previous[["station_id", "aadt"]].merge(
                current[["station_id", "aadt"]],
                on="station_id",
                suffixes=("_previous", "_current"),
            )
            change = paired["aadt_current"] - paired["aadt_previous"]
            row.update(
                {
                    "stations_shared_with_previous_year": len(paired),
                    "level_correlation_with_previous_year": round(
                        correlation(paired["aadt_previous"], paired["aadt_current"]),
                        6,
                    ),
                    "mean_change_from_previous_year": round(float(change.mean()), 4),
                    "median_change_from_previous_year": round(
                        float(change.median()), 4
                    ),
                    "mean_absolute_change_from_previous_year": round(
                        float(change.abs().mean()), 4
                    ),
                }
            )
        rows.append(row)
        previous = current
    return rows


def build_metadata_audit(panel: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    field_pairs = [
        ("road_network_reported", "road_network"),
        ("road_type_reported", "road_type"),
        ("link_description_reported", "link_description"),
    ]
    for reported_field, frozen_field in field_pairs:
        unique_counts = panel.groupby("station_id")[reported_field].nunique(dropna=False)
        rows.append(
            {
                "reported_field": reported_field,
                "station_count": len(unique_counts),
                "stations_with_multiple_reported_values": int((unique_counts > 1).sum()),
                "maximum_values_for_one_station": int(unique_counts.max()),
                "model_field": frozen_field,
                "model_rule": "freeze_at_first_year_the_station_appears",
            }
        )
    return rows


def metric_record(
    evaluation: str,
    target_year: int | str,
    spatial_fold: int | str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    return {
        "evaluation": evaluation,
        "target_year": target_year,
        "spatial_fold": spatial_fold,
        "model": model,
        "n": len(observed),
        "mae": round(mean_absolute_error(observed, predicted), 4),
        "rmse": round(math.sqrt(mean_squared_error(observed, predicted)), 4),
        "mean_bias": round(float(np.mean(predicted - observed)), 4),
        "r2": round(r2_score(observed, predicted), 6),
        "observed_predicted_correlation": round(correlation(observed, predicted), 6),
    }


def linear_station_prediction(history: pd.DataFrame, target_year: int) -> float:
    ordered = history.sort_values("year")
    if ordered["year"].nunique() < 2:
        return float(ordered.iloc[-1]["aadt"])
    years = ordered["year"].to_numpy(dtype=float)
    values = ordered["aadt"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(years, values, 1)
    return max(0.0, float(intercept + slope * target_year))


def run_known_station_forward(
    panel: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for target_year in TARGET_YEARS:
        target = panel[panel["year"] == target_year].copy()
        previous = panel[panel["year"] == target_year - 1][
            ["station_id", "aadt"]
        ].rename(columns={"aadt": "previous_aadt"})
        evaluation = target.merge(previous, on="station_id", how="inner")
        history = panel[panel["year"] < target_year]
        history_groups = {station_id: group for station_id, group in history.groupby("station_id")}

        predictions = {
            "persistence": evaluation["previous_aadt"].to_numpy(dtype=float),
            "expanding_station_mean": np.array(
                [
                    float(history_groups[station_id]["aadt"].mean())
                    for station_id in evaluation["station_id"]
                ]
            ),
            "station_linear_trend": np.array(
                [
                    linear_station_prediction(history_groups[station_id], target_year)
                    for station_id in evaluation["station_id"]
                ]
            ),
        }
        observed = evaluation["aadt"].to_numpy(dtype=float)
        for model, predicted in predictions.items():
            metric_rows.append(
                metric_record(
                    "known_station_forward",
                    target_year,
                    "not_spatially_held_out",
                    model,
                    observed,
                    predicted,
                )
            )
            for position, row in enumerate(evaluation.itertuples(index=False)):
                prediction_rows.append(
                    {
                        "station_id": int(row.station_id),
                        "target_year": target_year,
                        "model": model,
                        "observed_aadt": int(row.aadt),
                        "previous_aadt": int(row.previous_aadt),
                        "predicted_aadt": round(float(predicted[position]), 4),
                        "observed_change": int(row.aadt - row.previous_aadt),
                        "predicted_change": round(
                            float(predicted[position] - row.previous_aadt), 4
                        ),
                        "deployment_scope": "previous_station_aadt_required",
                    }
                )
    return prediction_rows, metric_rows


def static_feature_columns(panel: pd.DataFrame) -> list[str]:
    return [
        "longitude",
        "latitude",
        *(
            f"road_network_{value}"
            for value in sorted(panel["road_network"].dropna().unique())
        ),
        *(f"road_type_{value}" for value in sorted(panel["road_type"].dropna().unique())),
    ]


def static_feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    values = pd.get_dummies(
        frame[["longitude", "latitude", "road_network", "road_type"]],
        columns=["road_network", "road_type"],
        dtype=float,
    )
    return values.reindex(columns=feature_columns, fill_value=0.0).astype(float)


def fixed_static_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )


def road_hierarchy_lookup_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    lookup = train.groupby(["road_network", "road_type"], observed=True)["aadt"].median()
    network_lookup = train.groupby("road_network", observed=True)["aadt"].median()
    fallback = float(train["aadt"].median())
    predictions = []
    for row in test.itertuples(index=False):
        value = lookup.get((row.road_network, row.road_type), np.nan)
        if pd.isna(value):
            value = network_lookup.get(row.road_network, fallback)
        predictions.append(float(value))
    return np.array(predictions)


def fitted_linear_value(years: np.ndarray, values: np.ndarray, target_year: int) -> float:
    if len(np.unique(years)) < 2:
        return float(values[-1])
    slope, intercept = np.polyfit(years.astype(float), values.astype(float), 1)
    return max(0.0, float(intercept + slope * target_year))


def hierarchy_trend_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_year: int,
) -> np.ndarray:
    yearly_group = (
        train.groupby(["road_network", "road_type", "year"], observed=True)["aadt"]
        .median()
        .reset_index()
    )
    yearly_network = (
        train.groupby(["road_network", "year"], observed=True)["aadt"]
        .median()
        .reset_index()
    )
    yearly_all = train.groupby("year")["aadt"].median().reset_index()

    group_series = {
        key: group.sort_values("year")
        for key, group in yearly_group.groupby(["road_network", "road_type"])
    }
    network_series = {
        key: group.sort_values("year")
        for key, group in yearly_network.groupby("road_network")
    }

    predictions: list[float] = []
    for row in test.itertuples(index=False):
        series = group_series.get((row.road_network, row.road_type))
        if series is None or series["year"].nunique() < 2:
            series = network_series.get(row.road_network)
        if series is None or series["year"].nunique() < 2:
            series = yearly_all
        predictions.append(
            fitted_linear_value(
                series["year"].to_numpy(),
                series["aadt"].to_numpy(),
                target_year,
            )
        )
    return np.array(predictions)


def run_spatiotemporal_forward(
    panel: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    feature_columns = static_feature_columns(panel)

    for target_year in TARGET_YEARS:
        for fold in FOLDS:
            train = panel[
                (panel["year"] < target_year)
                & panel["spatial_fold"].notna()
                & (panel["spatial_fold"] != fold)
            ].copy()
            latest_train = train[train["year"] == target_year - 1].copy()
            test = panel[
                (panel["year"] == target_year)
                & panel["spatial_fold"].notna()
                & (panel["spatial_fold"] == fold)
            ].copy()
            if train.empty or latest_train.empty or test.empty:
                raise ValueError(
                    f"Empty spatiotemporal split: year={target_year}, fold={fold}."
                )

            pooled_model = fixed_static_model().fit(
                static_feature_frame(train, feature_columns),
                train["aadt"].to_numpy(dtype=float),
            )
            latest_model = fixed_static_model().fit(
                static_feature_frame(latest_train, feature_columns),
                latest_train["aadt"].to_numpy(dtype=float),
            )
            predictions = {
                "training_median": np.full(len(test), train["aadt"].median()),
                "latest_year_road_hierarchy": road_hierarchy_lookup_predictions(
                    latest_train, test
                ),
                "pooled_static_hgb": pooled_model.predict(
                    static_feature_frame(test, feature_columns)
                ),
                "latest_year_static_hgb": latest_model.predict(
                    static_feature_frame(test, feature_columns)
                ),
                "road_hierarchy_linear_trend": hierarchy_trend_predictions(
                    train, test, target_year
                ),
            }
            observed = test["aadt"].to_numpy(dtype=float)
            for model, predicted in predictions.items():
                metric_rows.append(
                    metric_record(
                        "future_year_and_held_out_spatial_fold",
                        target_year,
                        fold,
                        model,
                        observed,
                        predicted,
                    )
                )
                for position, row in enumerate(test.itertuples(index=False)):
                    prediction_rows.append(
                        {
                            "station_id": int(row.station_id),
                            "target_year": target_year,
                            "spatial_fold": fold,
                            "model": model,
                            "observed_aadt": int(row.aadt),
                            "predicted_aadt": round(float(predicted[position]), 4),
                            "error": round(float(predicted[position] - row.aadt), 4),
                            "deployment_scope": (
                                "station_held_out_in_all_training_years"
                            ),
                        }
                    )
    return prediction_rows, metric_rows


def pooled_metric_rows(
    predictions: pd.DataFrame,
    evaluation: str,
    models: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        for model in models:
            group = predictions[
                (predictions["target_year"] == target_year)
                & (predictions["model"] == model)
            ]
            rows.append(
                metric_record(
                    evaluation,
                    target_year,
                    "pooled",
                    model,
                    group["observed_aadt"].to_numpy(dtype=float),
                    group["predicted_aadt"].to_numpy(dtype=float),
                )
            )
    for model in models:
        group = predictions[predictions["model"] == model]
        rows.append(
            metric_record(
                evaluation,
                "all_target_years",
                "pooled",
                model,
                group["observed_aadt"].to_numpy(dtype=float),
                group["predicted_aadt"].to_numpy(dtype=float),
            )
        )
    return rows


def change_metric_record(
    change_year: int | str,
    model: str,
    observed_change: np.ndarray,
    predicted_change: np.ndarray,
) -> dict[str, object]:
    observed_sign = np.sign(observed_change)
    predicted_sign = np.sign(predicted_change)
    nonzero = observed_sign != 0
    return {
        "change_year": change_year,
        "model": model,
        "n": len(observed_change),
        "observed_mean_change": round(float(np.mean(observed_change)), 4),
        "predicted_mean_change": round(float(np.mean(predicted_change)), 4),
        "change_mae": round(
            mean_absolute_error(observed_change, predicted_change), 4
        ),
        "change_rmse": round(
            math.sqrt(mean_squared_error(observed_change, predicted_change)), 4
        ),
        "change_correlation": round(
            correlation(observed_change, predicted_change), 6
        ),
        "direction_accuracy_nonzero_observed": round(
            float(np.mean(observed_sign[nonzero] == predicted_sign[nonzero]))
            if nonzero.any()
            else float("nan"),
            6,
        ),
    }


def build_spatiotemporal_change_metrics(
    prediction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = pd.DataFrame(prediction_rows)
    rows: list[dict[str, object]] = []
    all_model_changes: dict[str, list[pd.DataFrame]] = {
        model: [] for model in SPATIOTEMPORAL_MODEL_ORDER
    }
    all_zero_changes: list[pd.DataFrame] = []

    for change_year in CHANGE_YEARS:
        for model in SPATIOTEMPORAL_MODEL_ORDER:
            previous = predictions[
                (predictions["target_year"] == change_year - 1)
                & (predictions["model"] == model)
            ][["station_id", "observed_aadt", "predicted_aadt"]]
            current = predictions[
                (predictions["target_year"] == change_year)
                & (predictions["model"] == model)
            ][["station_id", "observed_aadt", "predicted_aadt"]]
            paired = previous.merge(
                current,
                on="station_id",
                suffixes=("_previous", "_current"),
                validate="one_to_one",
            )
            paired["observed_change"] = (
                paired["observed_aadt_current"] - paired["observed_aadt_previous"]
            )
            paired["predicted_change"] = (
                paired["predicted_aadt_current"] - paired["predicted_aadt_previous"]
            )
            rows.append(
                change_metric_record(
                    change_year,
                    model,
                    paired["observed_change"].to_numpy(dtype=float),
                    paired["predicted_change"].to_numpy(dtype=float),
                )
            )
            all_model_changes[model].append(paired)

        reference = all_model_changes[SPATIOTEMPORAL_MODEL_ORDER[0]][-1]
        rows.append(
            change_metric_record(
                change_year,
                "zero_change",
                reference["observed_change"].to_numpy(dtype=float),
                np.zeros(len(reference)),
            )
        )
        all_zero_changes.append(reference)

    for model, frames in all_model_changes.items():
        combined = pd.concat(frames, ignore_index=True)
        rows.append(
            change_metric_record(
                "all_change_years",
                model,
                combined["observed_change"].to_numpy(dtype=float),
                combined["predicted_change"].to_numpy(dtype=float),
            )
        )
    zero_combined = pd.concat(all_zero_changes, ignore_index=True)
    rows.append(
        change_metric_record(
            "all_change_years",
            "zero_change",
            zero_combined["observed_change"].to_numpy(dtype=float),
            np.zeros(len(zero_combined)),
        )
    )
    return rows


def build_decision_audit(
    panel: pd.DataFrame,
    known_metrics: pd.DataFrame,
    change_metrics: pd.DataFrame,
) -> list[dict[str, object]]:
    common_stations = set.intersection(
        *(
            set(panel.loc[panel["year"] == year, "station_id"])
            for year in YEARS
        )
    )
    known_all = known_metrics[
        known_metrics["target_year"].astype(str) == "all_target_years"
    ].set_index("model")
    change_all = change_metrics[
        change_metrics["change_year"].astype(str) == "all_change_years"
    ].set_index("model")
    deployable = change_all.loc[
        [model for model in SPATIOTEMPORAL_MODEL_ORDER if model in change_all.index]
    ]
    best_model = str(deployable["change_mae"].idxmin())
    best_mae = float(deployable.loc[best_model, "change_mae"])
    best_correlation = float(deployable.loc[best_model, "change_correlation"])
    zero_mae = float(change_all.loc["zero_change", "change_mae"])
    improvement = (zero_mae - best_mae) / zero_mae * 100
    persistence_mae = float(known_all.loc["persistence", "mae"])

    change_supported = improvement > 0 and best_correlation > 0
    return [
        {
            "question": "is_a_continuous_directly_measured_station_panel_available",
            "evidence": (
                f"{len(common_stations)} stations appear in all seven years; "
                f"yearly counts range from {panel.groupby('year').size().min()} to "
                f"{panel.groupby('year').size().max()}"
            ),
            "decision": "yes_for_core_and_coverage_b_subset_only",
        },
        {
            "question": "how_strong_is_the_known_station_benchmark",
            "evidence": (
                f"previous-year persistence pooled forward MAE={persistence_mae:.1f}"
            ),
            "decision": "must_be_reported_but_not_deployable_to_never_counted_roads",
        },
        {
            "question": "does_a_deployable_static_or_hierarchy_model_beat_zero_change",
            "evidence": (
                f"best={best_model}; change MAE={best_mae:.1f}; "
                f"zero-change MAE={zero_mae:.1f}; improvement={improvement:.2f}%; "
                f"change correlation={best_correlation:.3f}"
            ),
            "decision": "provisional_yes" if change_supported else "no",
        },
        {
            "question": "can_segment_level_annual_change_maps_be_restored_now",
            "evidence": (
                "The evidence is from 173-191 directly measured core and Coverage "
                "(B) stations and "
                "uses current station coordinates for spatial blocks."
            ),
            "decision": (
                "not_yet_requires_dynamic_covariates_and_full_station_validation"
            ),
        },
        {
            "question": "does_step17_validate_full_network_backcasting",
            "evidence": "No local-road or full-network annual ground truth is introduced.",
            "decision": "no_measured_station_temporal_gate_only",
        },
    ]


def plot_annual_levels(panel: pd.DataFrame) -> None:
    common = set.intersection(
        *(
            set(panel.loc[panel["year"] == year, "station_id"])
            for year in YEARS
        )
    )
    balanced = panel[panel["station_id"].isin(common)]
    yearly = balanced.groupby("year")["aadt"].agg(["mean", "median", "count"])

    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    axis.plot(yearly.index, yearly["mean"], marker="o", label="Mean AADT")
    axis.plot(yearly.index, yearly["median"], marker="s", label="Median AADT")
    axis.set_title(
        f"Directly measured station panel ({len(common)} stations in every year)"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("AADT (vehicles/day)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEVEL_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {LEVEL_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_forward_metrics(
    known_metrics: pd.DataFrame,
    spatiotemporal_metrics: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=False)
    panels = [
        (
            axes[0],
            known_metrics,
            KNOWN_MODEL_ORDER,
            "Known station: next-year forecast",
        ),
        (
            axes[1],
            spatiotemporal_metrics,
            SPATIOTEMPORAL_MODEL_ORDER,
            "Future year + held-out spatial block",
        ),
    ]
    for axis, metrics, models, title in panels:
        yearly = metrics[
            metrics["target_year"].astype(str) != "all_target_years"
        ].copy()
        expected_fold_label = (
            "not_spatially_held_out" if "Known station" in title else "pooled"
        )
        yearly = yearly[
            yearly["spatial_fold"].astype(str) == expected_fold_label
        ]
        for model in models:
            group = yearly[yearly["model"] == model].sort_values("target_year")
            axis.plot(
                group["target_year"].astype(int),
                group["mae"],
                marker="o",
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        axis.set_title(title)
        axis.set_xlabel("Target year")
        axis.set_ylabel("MAE (vehicles/day)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Forward validation: knowing a station is not the same as a new location")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FORWARD_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FORWARD_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_change_metrics(change_metrics: pd.DataFrame) -> None:
    overall = change_metrics[
        change_metrics["change_year"].astype(str) == "all_change_years"
    ].copy()
    order = [*SPATIOTEMPORAL_MODEL_ORDER, "zero_change"]
    overall["order"] = overall["model"].map({model: i for i, model in enumerate(order)})
    overall = overall.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].barh(
        [MODEL_LABELS[model] for model in overall["model"]],
        overall["change_mae"],
        color=[MODEL_COLORS[model] for model in overall["model"]],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Pooled annual-change error")
    axes[0].set_xlabel("Change MAE (vehicles/day)")
    axes[0].grid(axis="x", alpha=0.25)

    yearly = change_metrics[
        change_metrics["change_year"].astype(str) != "all_change_years"
    ]
    observed = (
        yearly[yearly["model"] == "zero_change"]
        .sort_values("change_year")
        .set_index("change_year")["observed_mean_change"]
    )
    axes[1].plot(
        observed.index.astype(int),
        observed.values,
        marker="o",
        color="#1F1F1F",
        linewidth=2.2,
        label="Observed mean change",
    )
    for model in SPATIOTEMPORAL_MODEL_ORDER:
        group = yearly[yearly["model"] == model].sort_values("change_year")
        axes[1].plot(
            group["change_year"].astype(int),
            group["predicted_mean_change"],
            marker="o",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            alpha=0.85,
        )
    axes[1].axhline(0, color="#888888", linewidth=1)
    axes[1].set_title("Observed versus predicted mean annual change")
    axes[1].set_xlabel("Change ending in year")
    axes[1].set_ylabel("Mean AADT change (vehicles/day)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    fig.suptitle("Temporal identification gate on spatially held-out measured stations")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHANGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CHANGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    download_package()
    raw_panel = parse_official_package()
    raw_panel = attach_current_coordinates(raw_panel)
    step3_validation = validate_2021_against_step3(raw_panel)
    panel = assign_spatial_folds(raw_panel)

    source_rows = build_source_audit(panel, step3_validation)
    panel_rows = build_panel_audit(panel)
    metadata_rows = build_metadata_audit(panel)
    write_csv(SOURCE_AUDIT_PATH, source_rows)
    write_csv(PANEL_AUDIT_PATH, panel_rows)
    write_csv(METADATA_AUDIT_PATH, metadata_rows)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {PANEL_PATH.relative_to(PROJECT_ROOT)}")

    known_prediction_rows, known_fold_rows = run_known_station_forward(panel)
    spatiotemporal_prediction_rows, spatiotemporal_fold_rows = (
        run_spatiotemporal_forward(panel)
    )

    known_predictions = pd.DataFrame(known_prediction_rows)
    spatiotemporal_predictions = pd.DataFrame(spatiotemporal_prediction_rows)
    known_predictions.to_csv(KNOWN_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    spatiotemporal_predictions.to_csv(
        SPATIOTEMPORAL_PREDICTION_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {KNOWN_PREDICTION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {SPATIOTEMPORAL_PREDICTION_PATH.relative_to(PROJECT_ROOT)}")

    known_pooled_rows = pooled_metric_rows(
        known_predictions,
        "known_station_forward",
        KNOWN_MODEL_ORDER,
    )
    known_metric_rows = [
        *known_fold_rows,
        *[
            row
            for row in known_pooled_rows
            if str(row["target_year"]) == "all_target_years"
        ],
    ]
    spatiotemporal_metric_rows = [
        *spatiotemporal_fold_rows,
        *pooled_metric_rows(
            spatiotemporal_predictions,
            "future_year_and_held_out_spatial_fold",
            SPATIOTEMPORAL_MODEL_ORDER,
        ),
    ]
    change_metric_rows = build_spatiotemporal_change_metrics(
        spatiotemporal_prediction_rows
    )

    write_csv(KNOWN_METRIC_PATH, known_metric_rows)
    write_csv(SPATIOTEMPORAL_METRIC_PATH, spatiotemporal_metric_rows)
    write_csv(CHANGE_METRIC_PATH, change_metric_rows)

    known_metrics = pd.DataFrame(known_metric_rows)
    spatiotemporal_metrics = pd.DataFrame(spatiotemporal_metric_rows)
    change_metrics = pd.DataFrame(change_metric_rows)
    decision_rows = build_decision_audit(panel, known_metrics, change_metrics)
    write_csv(DECISION_AUDIT_PATH, decision_rows)

    plot_annual_levels(panel)
    plot_forward_metrics(known_metrics, spatiotemporal_metrics)
    plot_change_metrics(change_metrics)

    overall_known = known_metrics[
        known_metrics["target_year"].astype(str) == "all_target_years"
    ].set_index("model")
    overall_change = change_metrics[
        change_metrics["change_year"].astype(str) == "all_change_years"
    ].set_index("model")
    deployable = overall_change.loc[SPATIOTEMPORAL_MODEL_ORDER]
    best_model = str(deployable["change_mae"].idxmin())

    print("\nStep 17 annual temporal-identification gate is complete.")
    print(
        f"  Directly measured stations: {panel.groupby('year').size().min()}-"
        f"{panel.groupby('year').size().max()} per year; "
        f"{panel.groupby('station_id')['year'].nunique().eq(len(YEARS)).sum()} "
        "appear in all seven years."
    )
    print(
        "  Known-station previous-year benchmark MAE: "
        f"{overall_known.loc['persistence', 'mae']:,.0f} vehicles/day."
    )
    print(
        f"  Best spatially held-out change model: {best_model}, "
        f"MAE {deployable.loc[best_model, 'change_mae']:,.0f}; "
        "no-change MAE "
        f"{overall_change.loc['zero_change', 'change_mae']:,.0f}."
    )
    print(
        "  Decision: this is a measured-station temporal gate, not validation of "
        "full-network or local-road annual backcasting."
    )


if __name__ == "__main__":
    main()
