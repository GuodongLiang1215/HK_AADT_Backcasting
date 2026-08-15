"""Step 16: validate neighbourhood predictions and stress-test the equity estimand.

This corrected step deliberately stops before a full-network equity estimate.
Step 14 no longer freezes or calibrates the route-number subset, so a calibrated
strategic-road description would not have a matched official support.

The valid questions here are narrower:

1. Do station-level out-of-fold errors cancel after aggregation to 2016 Large
   TPU Groups?
2. On the same station-supported units, is there a stable monotonic association
   between income and mean station AADT?
3. Does a simple unit-resampling sensitivity interval exclude zero for the
   lowest-minus-highest income-quintile difference?

The resampling interval is descriptive. It treats units as exchangeable and is
not a spatial or survey-design confidence interval. A population-weighted
near-road activity estimand on finer geography is deferred to P1.
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
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAINING_PATH = PROCESSED_DIR / "atc_high_confidence_training_table.csv"
OOF_PATH = PROCESSED_DIR / "atc_step15_oof_predictions.csv"
CENSUS_PATH = PROCESSED_DIR / "census_ltpug_standardised_panel.csv"
BOUNDARY_PATH = PROCESSED_DIR / "census_ltpug_2016_reference_boundaries.geojson.gz"
BOUNDARY_MATCH_PATH = TABLE_DIR / "step12_boundary_match_review.csv"

STATION_UNIT_PATH = PROCESSED_DIR / "atc_step16_station_neighbourhood_panel.csv"
UNIT_PANEL_PATH = PROCESSED_DIR / "atc_step16_neighbourhood_validation_panel.csv"
VALIDATION_PATH = TABLE_DIR / "step16_neighbourhood_validation.csv"
ESTIMAND_CHECK_PATH = TABLE_DIR / "step16_income_estimand_check.csv"
SENSITIVITY_PATH = TABLE_DIR / "step16_income_association_sensitivity.csv"
DECISION_AUDIT_PATH = TABLE_DIR / "step16_equity_decision_audit.csv"

VALIDATION_FIGURE_PATH = FIGURE_DIR / "step16_neighbourhood_validation.png"
ESTIMAND_FIGURE_PATH = FIGURE_DIR / "step16_income_estimand_check.png"

YEARS = (2011, 2016, 2021)
PRIMARY_YEAR = 2016
MODEL = "hist_gradient_boosting"
QUINTILE_LABELS = ["Q1_lowest_income", "Q2", "Q3", "Q4", "Q5_highest_income"]
MINIMUM_STATIONS_PER_UNIT = 1
NEAREST_FALLBACK_LIMIT_M = 1000.0
LATITUDE_ORIGIN = 22.35
X_SCALE = 111_320.0 * math.cos(math.radians(LATITUDE_ORIGIN))
Y_SCALE = 110_540.0
INCOME_COLUMN = "median_monthly_domestic_household_income_hkd_nominal"
UNIT_BOOTSTRAP_DRAWS = 2000
UNIT_BOOTSTRAP_SEED = 20260815


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
    if not text or text.casefold() in {"nan", "none"}:
        return ""
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = (
        TRAINING_PATH,
        OOF_PATH,
        CENSUS_PATH,
        BOUNDARY_PATH,
        BOUNDARY_MATCH_PATH,
    )
    missing = [path.relative_to(PROJECT_ROOT) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}. Run Steps 12 and 15 first.")
    training = pd.read_csv(TRAINING_PATH)
    oof = pd.read_csv(OOF_PATH)
    oof = oof[oof["model"] == MODEL].copy()
    census = pd.read_csv(CENSUS_PATH, dtype={"ltpug_id": str})
    census["ltpug_id"] = census["ltpug_id"].map(normalise_code)
    boundary_matches = pd.read_csv(
        BOUNDARY_MATCH_PATH,
        dtype={"source_ltpug_id": str, "target_ltpug_id": str},
    )
    boundary_matches["source_ltpug_id"] = boundary_matches[
        "source_ltpug_id"
    ].map(normalise_code)
    boundary_matches["target_ltpug_id"] = boundary_matches[
        "target_ltpug_id"
    ].map(normalise_code)
    return training, oof, census, boundary_matches


def assign_stations_to_units(training: pd.DataFrame) -> pd.DataFrame:
    if STATION_UNIT_PATH.exists():
        prior = pd.read_csv(STATION_UNIT_PATH, dtype={"reference_ltpug_id_2016": str})
        expected = set(training["station_id"].astype(int))
        available = set(prior["station_id"].astype(int))
        if expected == available and not prior["station_id"].duplicated().any():
            prior["reference_ltpug_id_2016"] = prior[
                "reference_ltpug_id_2016"
            ].map(normalise_code)
            return prior

    import shapely
    from shapely import STRtree
    from shapely.geometry import shape

    with gzip.open(BOUNDARY_PATH, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    geometries = [shape(feature["geometry"]) for feature in payload["features"]]
    identifiers = [str(feature["properties"]["ltpug_id"]) for feature in payload["features"]]
    tree = STRtree(geometries)
    points = shapely.points(
        training["station_longitude"].to_numpy(dtype=float),
        training["station_latitude"].to_numpy(dtype=float),
    )
    assigned = np.full(len(training), "", dtype=object)
    method = np.full(len(training), "outside_all_reference_units", dtype=object)
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
    result = training[
        ["station_id", "station_longitude", "station_latitude", "spatial_fold"]
    ].copy()
    result["reference_ltpug_id_2016"] = assigned
    result["reference_unit_assignment"] = method
    return result


def build_station_unit_panel(
    station_units: pd.DataFrame, oof: pd.DataFrame
) -> pd.DataFrame:
    panel = oof.merge(station_units, on="station_id", how="left", validate="many_to_one")
    if panel["reference_ltpug_id_2016"].isna().any():
        raise ValueError("A station has no reference-unit assignment.")
    return panel[panel["reference_ltpug_id_2016"] != ""].copy()


def build_neighbourhood_validation(
    panel: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    unit_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for year in YEARS:
        frame = panel[panel["year"] == year]
        grouped = (
            frame.groupby("reference_ltpug_id_2016")
            .agg(
                station_count=("station_id", "size"),
                observed_mean_aadt=("observed_aadt", "mean"),
                predicted_mean_aadt=("predicted_aadt", "mean"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["station_count"] >= MINIMUM_STATIONS_PER_UNIT]
        grouped["year"] = year
        grouped["bias_aadt"] = grouped["predicted_mean_aadt"] - grouped["observed_mean_aadt"]
        grouped["bias_pct"] = 100 * grouped["bias_aadt"] / grouped["observed_mean_aadt"]
        unit_frames.append(grouped)

        observed = grouped["observed_mean_aadt"].to_numpy(dtype=float)
        predicted = grouped["predicted_mean_aadt"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "year": year,
                "units_with_stations": len(grouped),
                "stations_used": int(grouped["station_count"].sum()),
                "units_with_one_station": int((grouped["station_count"] == 1).sum()),
                "units_with_two_or_fewer_stations": int((grouped["station_count"] <= 2).sum()),
                "maximum_stations_in_one_unit": int(grouped["station_count"].max()),
                "neighbourhood_mae_aadt": round(float(np.abs(predicted - observed).mean()), 2),
                "neighbourhood_mae_pct_of_observed_mean": round(
                    100 * float(np.abs(predicted - observed).mean()) / float(observed.mean()), 2
                ),
                "neighbourhood_mean_bias_pct": round(
                    100 * float(predicted.mean() - observed.mean()) / float(observed.mean()), 2
                ),
                "station_weighted_mean_bias_pct": round(
                    100
                    * (
                        float(np.average(predicted, weights=grouped["station_count"]))
                        - float(np.average(observed, weights=grouped["station_count"]))
                    )
                    / float(np.average(observed, weights=grouped["station_count"])),
                    2,
                ),
                "pearson_r": round(float(np.corrcoef(observed, predicted)[0, 1]), 4),
                "spearman_rho": round(float(spearmanr(observed, predicted).statistic), 4),
                "units_within_20pct": int((np.abs(predicted - observed) / observed <= 0.20).sum()),
                "units_over_50pct_off": int((np.abs(predicted - observed) / observed > 0.50).sum()),
                "decision": "overall_spatial_rank_is_informative_but_group_contrasts_need_direct_validation",
            }
        )
    return summary_rows, pd.concat(unit_frames, ignore_index=True)


def attach_income(
    unit_frame: pd.DataFrame,
    census: pd.DataFrame,
    boundary_matches: pd.DataFrame,
) -> pd.DataFrame:
    income = census[
        ["year", "ltpug_id", INCOME_COLUMN, "total_population", "boundary_area_km2"]
    ]
    merged = unit_frame.merge(
        income,
        left_on=["year", "reference_ltpug_id_2016"],
        right_on=["year", "ltpug_id"],
        how="left",
    )
    merged["population_density_per_km2"] = np.where(
        merged["boundary_area_km2"] > 0,
        merged["total_population"] / merged["boundary_area_km2"],
        np.nan,
    )
    merged["region"] = merged["reference_ltpug_id_2016"].str[0].map(
        {"1": "hong_kong_island", "2": "kowloon"}
    ).fillna("new_territories")
    stable = boundary_matches[
        boundary_matches["eligible_for_direct_panel_comparison"].astype(str).str.casefold()
        == "true"
    ]
    stable_keys = set(
        zip(
            stable["source_year"].astype(int),
            stable["source_ltpug_id"].astype(str),
            stable["target_ltpug_id"].astype(str),
        )
    )
    merged["boundary_stable_for_2016_reference"] = [
        year == PRIMARY_YEAR
        or (int(year), str(source_id), str(reference_id)) in stable_keys
        for year, source_id, reference_id in zip(
            merged["year"],
            merged["ltpug_id"],
            merged["reference_ltpug_id_2016"],
        )
    ]
    return merged


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = np.empty(UNIT_BOOTSTRAP_DRAWS)
    for index in range(UNIT_BOOTSTRAP_DRAWS):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_difference(
    lowest: np.ndarray, highest: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    draws = np.empty(UNIT_BOOTSTRAP_DRAWS)
    for index in range(UNIT_BOOTSTRAP_DRAWS):
        draws[index] = (
            rng.choice(lowest, len(lowest), replace=True).mean()
            - rng.choice(highest, len(highest), replace=True).mean()
        )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_income_estimand_check(unit_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        frame = unit_frame[
            (unit_frame["year"] == year)
            & unit_frame[INCOME_COLUMN].notna()
            & unit_frame["boundary_stable_for_2016_reference"]
        ].copy()
        frame["income_quintile"] = pd.qcut(
            frame[INCOME_COLUMN], 5, labels=QUINTILE_LABELS, duplicates="drop"
        )
        for source_index, (source, column) in enumerate(
            (
                ("observed", "observed_mean_aadt"),
                ("out_of_fold_predicted", "predicted_mean_aadt"),
            )
        ):
            rng = np.random.default_rng(
                UNIT_BOOTSTRAP_SEED + (year - 2000) * 100 + source_index
            )
            grouped = frame.groupby("income_quintile", observed=True)[column]
            means = grouped.mean()
            lowest_values = grouped.get_group(QUINTILE_LABELS[0]).to_numpy(dtype=float)
            highest_values = grouped.get_group(QUINTILE_LABELS[-1]).to_numpy(dtype=float)
            difference = float(lowest_values.mean() - highest_values.mean())
            difference_low, difference_high = bootstrap_difference(
                lowest_values, highest_values, rng
            )
            mean_differences = np.diff(means.to_numpy(dtype=float))
            if np.all(mean_differences > 0):
                shape_label = "monotonic_increasing_with_income"
            elif np.all(mean_differences < 0):
                shape_label = "monotonic_decreasing_with_income"
            else:
                shape_label = "non_monotonic_q1_minus_q5_is_not_a_sufficient_summary"
            rank_correlation = float(spearmanr(frame[INCOME_COLUMN], frame[column]).statistic)
            for quintile in means.index:
                values = grouped.get_group(quintile).to_numpy(dtype=float)
                mean_low, mean_high = bootstrap_mean(values, rng)
                rows.append(
                    {
                        "year": year,
                        "source": source,
                        "income_quintile": str(quintile),
                        "units": len(values),
                        "mean_station_aadt": round(float(values.mean()), 2),
                        "quintile_mean_resampling_low": round(mean_low, 2),
                        "quintile_mean_resampling_high": round(mean_high, 2),
                        "q1_minus_q5_aadt": round(difference, 2),
                        "q1_minus_q5_resampling_low": round(difference_low, 2),
                        "q1_minus_q5_resampling_high": round(difference_high, 2),
                        "q1_minus_q5_interval_includes_zero": bool(
                            difference_low <= 0 <= difference_high
                        ),
                        "unit_level_income_aadt_spearman": round(rank_correlation, 4),
                        "gradient_shape": shape_label,
                        "resampling_status": "iid_unit_resampling_sensitivity_not_spatial_or_survey_ci",
                    }
                )
    return rows


def build_income_sensitivity(unit_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        frame = unit_frame[
            (unit_frame["year"] == year)
            & unit_frame[INCOME_COLUMN].notna()
            & unit_frame["population_density_per_km2"].notna()
            & unit_frame["boundary_stable_for_2016_reference"]
        ].copy()
        frame["density_tertile"] = pd.qcut(
            frame["population_density_per_km2"],
            3,
            labels=["low_density", "middle_density", "high_density"],
        )
        for stratification, group_column in (
            ("density_tertile", "density_tertile"),
            ("region", "region"),
        ):
            for stratum, group in frame.groupby(group_column, observed=True):
                for source, column in (
                    ("observed", "observed_mean_aadt"),
                    ("out_of_fold_predicted", "predicted_mean_aadt"),
                ):
                    rows.append(
                        {
                            "year": year,
                            "stratification": stratification,
                            "stratum": str(stratum),
                            "source": source,
                            "units": len(group),
                            "income_aadt_spearman": round(
                                float(spearmanr(group[INCOME_COLUMN], group[column]).statistic),
                                4,
                            ),
                            "decision": "sensitivity_only_same_station_mean_estimand",
                        }
                    )
    return rows


def build_decision_audit(
    validation_rows: list[dict[str, object]],
    estimand_rows: list[dict[str, object]],
    station_units: pd.DataFrame,
    unit_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    validation = pd.DataFrame(validation_rows)
    primary_validation = validation[validation["year"] == PRIMARY_YEAR].iloc[0]
    estimand = pd.DataFrame(estimand_rows)
    primary_observed = estimand[
        (estimand["year"] == PRIMARY_YEAR) & (estimand["source"] == "observed")
    ].iloc[0]
    eligible_units = {
        year: int(
            unit_frame[
                (unit_frame["year"] == year)
                & unit_frame[INCOME_COLUMN].notna()
                & unit_frame["boundary_stable_for_2016_reference"]
            ]["reference_ltpug_id_2016"].nunique()
        )
        for year in YEARS
    }
    return [
        {"metric": "reference_geography", "count": "", "value": "large_tpu_group_2016", "decision": "coarse_diagnostic_geography_not_final_equity_estimand"},
        {"metric": "reference_units_containing_a_training_station", "count": int(primary_validation["units_with_stations"]), "value": "of_154", "decision": "validation_only_on_station_supported_units"},
        {"metric": "units_with_one_station_2016", "count": int(primary_validation["units_with_one_station"]), "value": "", "decision": "unit_means_have_heterogeneous_measurement_support"},
        {"metric": "units_with_two_or_fewer_stations_2016", "count": int(primary_validation["units_with_two_or_fewer_stations"]), "value": "", "decision": "unit_means_have_heterogeneous_measurement_support"},
        {"metric": "stations_outside_all_reference_units", "count": int((station_units["reference_ltpug_id_2016"] == "").sum()), "value": "", "decision": "excluded_from_neighbourhood_validation"},
        {"metric": "income_analysis_units_with_stable_boundary_2011", "count": eligible_units[2011], "value": "", "decision": "stable_same_code_to_2016_reference_only"},
        {"metric": "income_analysis_units_2016", "count": eligible_units[2016], "value": "", "decision": "reference_year_units_with_income"},
        {"metric": "income_analysis_units_with_stable_boundary_2021", "count": eligible_units[2021], "value": "", "decision": "stable_same_code_to_2016_reference_only"},
        {"metric": "neighbourhood_mae_pct_of_mean_2016", "count": "", "value": primary_validation["neighbourhood_mae_pct_of_observed_mean"], "decision": "errors_do_not_cancel_at_large_TPU_group_scale"},
        {"metric": "neighbourhood_spearman_rho_2016", "count": "", "value": primary_validation["spearman_rho"], "decision": "overall_rank_agreement_does_not_validate_income_group_direction"},
        {"metric": "observed_q1_minus_q5_2016", "count": "", "value": primary_observed["q1_minus_q5_aadt"], "decision": "descriptive_difference_on_station_mean_estimand"},
        {"metric": "observed_q1_minus_q5_resampling_interval_2016", "count": "", "value": f"{primary_observed['q1_minus_q5_resampling_low']}_to_{primary_observed['q1_minus_q5_resampling_high']}", "decision": "iid_unit_sensitivity_interval_includes_zero"},
        {"metric": "observed_unit_level_income_aadt_spearman_2016", "count": "", "value": primary_observed["unit_level_income_aadt_spearman"], "decision": "no_monotonic_signal_on_current_proxy"},
        {"metric": "attenuation_factor_defined", "count": 0, "value": "", "decision": "do_not_divide_by_a_near_zero_nonmonotonic_observed_difference"},
        {"metric": "MAUP_demonstrated", "count": 0, "value": "plausible_not_proven", "decision": "requires_comparison_across_finer_geographies_and_zoning_rules"},
        {"metric": "equity_direction_reportable", "count": 0, "value": "", "decision": "current_station_mean_estimand_has_no_stable_monotonic_income_signal"},
        {"metric": "equity_magnitude_reportable", "count": 0, "value": "", "decision": "population_weighted_near_road_estimand_required"},
        {"metric": "preliminary_calibrated_E1_description_reportable", "count": 0, "value": "", "decision": "withdrawn_after_step14_support_correction"},
        {"metric": "step16_decision_signal", "count": "", "value": "report_neither_equity_direction_nor_magnitude_under_current_estimand", "decision": "use_as_an_estimand_failure_diagnostic"},
    ]


def plot_neighbourhood_validation(unit_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
    limit = float(max(unit_frame["observed_mean_aadt"].max(), unit_frame["predicted_mean_aadt"].max()))
    for axis, year in zip(axes, YEARS):
        frame = unit_frame[unit_frame["year"] == year]
        sizes = 12 + 4 * np.sqrt(frame["station_count"].to_numpy(dtype=float))
        axis.scatter(
            frame["observed_mean_aadt"], frame["predicted_mean_aadt"],
            s=sizes, alpha=0.7, color="#2E86AB", linewidths=0,
        )
        axis.plot([0, limit], [0, limit], color="#202124", linestyle="--", linewidth=1)
        axis.set_title(str(year))
        axis.set_xlabel("Observed mean station AADT")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Out-of-fold predicted mean")
    fig.suptitle("Large TPU Group diagnostic; marker size reflects station count")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VALIDATION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {VALIDATION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def plot_income_estimand_check(estimand_rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(estimand_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
    positions = np.arange(len(QUINTILE_LABELS))
    for axis, year in zip(axes, YEARS):
        for source, colour, marker in (
            ("observed", "#202124", "o"),
            ("out_of_fold_predicted", "#D35400", "s"),
        ):
            rows = (
                frame[(frame["year"] == year) & (frame["source"] == source)]
                .set_index("income_quintile")
                .loc[QUINTILE_LABELS]
            )
            values = rows["mean_station_aadt"].to_numpy(dtype=float)
            low = values - rows["quintile_mean_resampling_low"].to_numpy(dtype=float)
            high = rows["quintile_mean_resampling_high"].to_numpy(dtype=float) - values
            axis.errorbar(
                positions, values, yerr=np.vstack([low, high]), marker=marker,
                color=colour, capsize=2.5, linewidth=1.5,
                label=source.replace("_", " "),
            )
        axis.set_xticks(positions, ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], fontsize=8)
        axis.set_title(str(year))
        axis.set_xlabel("Household-income quintile")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean station AADT (vehicles/day)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Estimand check: no stable monotonic income gradient on mean station AADT")
    fig.tight_layout()
    fig.savefig(ESTIMAND_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ESTIMAND_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    training, oof, census, boundary_matches = read_inputs()
    print("Assigning counting stations to 2016 Large TPU Group units...")
    station_units = assign_stations_to_units(training)
    panel = build_station_unit_panel(station_units, oof)
    validation_rows, unit_frame = build_neighbourhood_validation(panel)
    unit_frame = attach_income(unit_frame, census, boundary_matches)
    estimand_rows = build_income_estimand_check(unit_frame)
    sensitivity_rows = build_income_sensitivity(unit_frame)
    audit_rows = build_decision_audit(
        validation_rows,
        estimand_rows,
        station_units,
        unit_frame,
    )

    STATION_UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    station_units.to_csv(STATION_UNIT_PATH, index=False, encoding="utf-8-sig")
    unit_frame.to_csv(UNIT_PANEL_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {STATION_UNIT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {UNIT_PANEL_PATH.relative_to(PROJECT_ROOT)}")
    write_csv(VALIDATION_PATH, validation_rows)
    write_csv(ESTIMAND_CHECK_PATH, estimand_rows)
    write_csv(SENSITIVITY_PATH, sensitivity_rows)
    write_csv(DECISION_AUDIT_PATH, audit_rows)
    plot_neighbourhood_validation(unit_frame)
    plot_income_estimand_check(estimand_rows)

    print("\nStep 16 neighbourhood and estimand diagnostics are complete.")
    for row in validation_rows:
        print(
            f"  {row['year']}  units {row['units_with_stations']:3d}  "
            f"MAE {float(row['neighbourhood_mae_aadt']):,.0f} "
            f"({float(row['neighbourhood_mae_pct_of_observed_mean']):.1f}% of mean)  "
            f"Spearman {float(row['spearman_rho']):.2f}"
        )
    estimand = pd.DataFrame(estimand_rows)
    print("\nObserved income contrast on the current station-mean proxy:")
    for year in YEARS:
        row = estimand[(estimand["year"] == year) & (estimand["source"] == "observed")].iloc[0]
        print(
            f"  {year} Q1-Q5 {float(row['q1_minus_q5_aadt']):+,.0f}; "
            f"unit-resampling range [{float(row['q1_minus_q5_resampling_low']):+,.0f}, "
            f"{float(row['q1_minus_q5_resampling_high']):+,.0f}]; "
            f"Spearman {float(row['unit_level_income_aadt_spearman']):+.3f}"
        )
    print(
        "\nDecision: report neither equity direction nor magnitude from this estimand. "
        "The next equity step is a population-weighted near-road traffic-activity measure "
        "on finer geography; temporal identification remains a separate required gate."
    )


if __name__ == "__main__":
    main()
