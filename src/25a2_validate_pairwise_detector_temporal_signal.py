"""Step 25A.2: adjacent-year, unbalanced strategic-detector validation.

This prospectively specified sensitivity asks whether the negative Step 25A
and Step 25A.1 results were caused mainly by the requirement that one detector
must remain qualified across all four years.  It uses the already materialised
April--December Step 25A.1 detector-year proxy, but constructs a separate
eligible detector panel for each adjacent transition (2021--2022,
2022--2023 and 2023--2024).

The detector proxy is not AADT.  No Step 25A or Step 25A.1 output is
overwritten.  All thresholds below are fixed before Step 25A.2 outcomes are
inspected:

* at least 100 qualified, positive-proxy, lane-stable detectors per transition;
* at least four inherited frozen spatial folds per transition;
* at least 20 accepted one-to-one detector--major-ATC pairs that also have
  evaluable adjacent-year ATC observations per transition for the colocated
  task;
* the held-out network factor for each ATC fold excludes detectors in that
  fold;
* at least 5% pooled MAE improvement over no change;
* the cluster-bootstrap upper 95% bound for model-minus-no-change absolute
  error must be below zero;
* at least two of three transitions and three of five spatial folds improve;
* pooled change correlation must be positive; and
* at least two transitions must meet the relevant support gate.

Passing can authorise only a bounded recent-year major-road Step 25B.  It
cannot authorise local-road reconstruction, a full-network backcast, a 2011
backcast or an equity trend.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "report_manifest.csv"

BASE_SCRIPT_PATH = (
    PROJECT_ROOT / "src" / "25a_validate_strategic_detector_temporal_signal.py"
)
PAIRWISE_ANNUAL_INPUT = (
    PROCESSED_DIR / "atc_step25a1_strategic_annual_proxy.csv"
)
DETECTOR_LOCATION_INPUT = (
    PROCESSED_DIR / "atc_step24_public_dynamic_detector_locations.csv"
)
MEASURED_PANEL_INPUT = (
    PROCESSED_DIR / "atc_step18_measured_station_annual_panel.csv"
)

PAIRWISE_PANEL_PATH = (
    PROCESSED_DIR / "atc_step25a2_pairwise_detector_panel.csv"
)
CROSSWALK_PATH = PROCESSED_DIR / "atc_step25a2_detector_atc_crosswalk.csv"
PREDICTION_PATH = PROCESSED_DIR / "atc_step25a2_temporal_predictions.csv"

DESIGN_AUDIT_PATH = TABLE_DIR / "step25a2_design_audit.csv"
SUPPORT_PATH = TABLE_DIR / "step25a2_transition_support.csv"
FOLD_SUPPORT_PATH = TABLE_DIR / "step25a2_detector_fold_support.csv"
CROSSWALK_AUDIT_PATH = TABLE_DIR / "step25a2_crosswalk_audit.csv"
TRANSITION_METRICS_PATH = TABLE_DIR / "step25a2_metrics_by_transition.csv"
FOLD_METRICS_PATH = TABLE_DIR / "step25a2_metrics_by_fold.csv"
PAIRED_COMPARISON_PATH = TABLE_DIR / "step25a2_paired_model_comparison.csv"
DECISION_PATH = TABLE_DIR / "step25a2_decision_audit.csv"

SUPPORT_FIGURE_PATH = FIGURE_DIR / "step25a2_pairwise_support.png"
CHANGE_FIGURE_PATH = FIGURE_DIR / "step25a2_change_identification.png"

PRIMARY_YEARS = (2021, 2022, 2023, 2024)
TRANSITIONS = tuple(zip(PRIMARY_YEARS[:-1], PRIMARY_YEARS[1:]))
MIN_PAIRWISE_DETECTORS = 100
MIN_PAIRWISE_SPATIAL_FOLDS = 4
MIN_CROSSWALK_PAIRS = 20
MIN_MAE_IMPROVEMENT_PCT = 5.0
MIN_SUPPORTED_TRANSITIONS = 2
MIN_IMPROVED_TRANSITIONS = 2
MIN_IMPROVED_FOLDS = 3
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 250819

MAX_CROSSWALK_DISTANCE_M = 100.0
HIGH_CONFIDENCE_DISTANCE_M = 20.0
MIN_ROAD_NAME_SIMILARITY = 0.50
STEP25A2_REVISION = "2026-08-17.2"


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def load_base_module():
    if not BASE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "Missing frozen Step 25A source: "
            f"{BASE_SCRIPT_PATH.relative_to(PROJECT_ROOT)}"
        )
    specification = importlib.util.spec_from_file_location(
        "step25a_base_for_pairwise_design", BASE_SCRIPT_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the frozen Step 25A module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def require_inputs() -> None:
    required = (
        BASE_SCRIPT_PATH,
        PAIRWISE_ANNUAL_INPUT,
        DETECTOR_LOCATION_INPUT,
        MEASURED_PANEL_INPUT,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Step 25A.2 inputs are missing: {joined}. "
            "Run the corrected Steps 18 and 24 and Step 25A.1 first."
        )


def normalise_identifier(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def mode_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().round().astype(int)
    if clean.empty:
        return np.nan
    modes = clean.mode()
    return float(modes.min())


def make_design_audit() -> pd.DataFrame:
    rows = [
        {
            "design_item": "analysis_status",
            "value": "prospectively_specified_pairwise_sensitivity",
            "evidence": "Step 25A and Step 25A.1 outputs and decisions remain unchanged",
        },
        {
            "design_item": "detector_panel_definition",
            "value": "separate_unbalanced_panel_for_each_adjacent_year_pair",
            "evidence": "both detector-years must pass the frozen Step 25A.1 detector-year rule, have positive proxies and have the same non-missing modal lane count",
        },
        {
            "design_item": "temporal_window",
            "value": "2021-2022|2022-2023|2023-2024",
            "evidence": "uses the already materialised April-December Step 25A.1 proxy; no new archive window is selected from outcomes",
        },
        {
            "design_item": "support_thresholds",
            "value": f"detectors>={MIN_PAIRWISE_DETECTORS};folds>={MIN_PAIRWISE_SPATIAL_FOLDS};colocated_pairs>={MIN_CROSSWALK_PAIRS}",
            "evidence": "a transition below support is non-evaluable, not a negative skill result",
        },
        {
            "design_item": "performance_thresholds",
            "value": f"MAE_gain>={MIN_MAE_IMPROVEMENT_PCT:.0f}%;bootstrap_upper<0;improved_transitions>={MIN_IMPROVED_TRANSITIONS};improved_folds>={MIN_IMPROVED_FOLDS};correlation>0",
            "evidence": "all comparisons use measured AADT change and the no-change prediction",
        },
        {
            "design_item": "spatial_leakage_rule",
            "value": "exclude_target_fold_detectors",
            "evidence": "each held-out ATC fold receives the median detector ratio from other inherited frozen folds only",
        },
        {
            "design_item": "claim_boundary",
            "value": "bounded_recent_year_major_road_only",
            "evidence": "passing cannot authorise local roads, full-network backcasting, 2011, or an equity trend",
        },
    ]
    audit = pd.DataFrame(rows)
    save_csv(audit, DESIGN_AUDIT_PATH)
    return audit


def prepare_locations(base) -> pd.DataFrame:
    frame = pd.read_csv(DETECTOR_LOCATION_INPUT)
    source_col = base.find_column(frame, ("source",))
    id_col = base.find_column(frame, ("device_id", "detector_id"))
    lon_col = base.find_column(frame, ("longitude", "lon"))
    lat_col = base.find_column(frame, ("latitude", "lat"))
    road_col = base.find_column(frame, ("road_name", "road_en"), required=False)
    fold_col = base.find_column(
        frame, ("nearest_spatial_fold", "spatial_fold"), required=False
    )
    strategic = frame[frame[source_col].astype(str).eq("strategic_detector")].copy()
    strategic["detector_id"] = strategic[id_col].map(normalise_identifier)
    strategic["longitude"] = pd.to_numeric(strategic[lon_col], errors="coerce")
    strategic["latitude"] = pd.to_numeric(strategic[lat_col], errors="coerce")
    strategic["road_name"] = (
        strategic[road_col].fillna("").astype(str) if road_col else ""
    )
    strategic["spatial_fold"] = (
        pd.to_numeric(strategic[fold_col], errors="coerce") if fold_col else np.nan
    )
    strategic = strategic.dropna(subset=["longitude", "latitude"])
    rows: list[dict[str, object]] = []
    for detector_id, group in strategic.groupby("detector_id"):
        road_names = sorted(
            {value.strip() for value in group["road_name"].astype(str) if value.strip()}
        )
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
    return pd.DataFrame(rows)


def build_pairwise_detector_panel(
    annual: pd.DataFrame, locations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual = annual.copy()
    annual["detector_id"] = annual["detector_id"].map(normalise_identifier)
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    annual["balanced_annual_volume_proxy"] = pd.to_numeric(
        annual["balanced_annual_volume_proxy"], errors="coerce"
    )
    annual["modal_valid_lane_count"] = pd.to_numeric(
        annual["modal_valid_lane_count"], errors="coerce"
    )
    annual["detector_year_qualified"] = as_bool(annual["detector_year_qualified"])

    pair_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for base_year, target_year in TRANSITIONS:
        transition = f"{base_year}-{target_year}"
        base_frame = annual[annual["year"].eq(base_year)][
            [
                "detector_id",
                "balanced_annual_volume_proxy",
                "modal_valid_lane_count",
                "detector_year_qualified",
            ]
        ].rename(
            columns={
                "balanced_annual_volume_proxy": "proxy_base",
                "modal_valid_lane_count": "lane_base",
                "detector_year_qualified": "qualified_base",
            }
        )
        target_frame = annual[annual["year"].eq(target_year)][
            [
                "detector_id",
                "balanced_annual_volume_proxy",
                "modal_valid_lane_count",
                "detector_year_qualified",
            ]
        ].rename(
            columns={
                "balanced_annual_volume_proxy": "proxy_target",
                "modal_valid_lane_count": "lane_target",
                "detector_year_qualified": "qualified_target",
            }
        )
        paired = base_frame.merge(
            target_frame, on="detector_id", how="inner", validate="one_to_one"
        )
        paired["both_years_qualified"] = (
            paired["qualified_base"] & paired["qualified_target"]
        )
        paired["positive_proxies"] = paired["proxy_base"].gt(0) & paired[
            "proxy_target"
        ].gt(0)
        paired["lane_pair_stable"] = (
            paired["lane_base"].notna()
            & paired["lane_target"].notna()
            & paired["lane_base"].eq(paired["lane_target"])
        )
        paired["pairwise_eligible_before_location"] = (
            paired["both_years_qualified"]
            & paired["positive_proxies"]
            & paired["lane_pair_stable"]
        )
        eligible_before = paired[paired["pairwise_eligible_before_location"]].copy()
        eligible = eligible_before.merge(
            locations,
            on="detector_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        eligible["current_location_linked"] = eligible["_merge"].eq("both")
        eligible["valid_spatial_fold"] = pd.to_numeric(
            eligible["spatial_fold"], errors="coerce"
        ).between(1, 5)
        eligible["pairwise_eligible"] = (
            eligible["current_location_linked"] & eligible["valid_spatial_fold"]
        )
        eligible = eligible[eligible["pairwise_eligible"]].copy()
        eligible["spatial_fold"] = pd.to_numeric(
            eligible["spatial_fold"], errors="coerce"
        ).astype(int)
        eligible["detector_ratio"] = eligible["proxy_target"] / eligible[
            "proxy_base"
        ]
        eligible["base_year"] = base_year
        eligible["target_year"] = target_year
        eligible["transition"] = transition
        eligible = eligible.drop(columns=["_merge"], errors="ignore")
        pair_rows.append(eligible)

        fold_count = eligible["spatial_fold"].nunique()
        support_pass = (
            eligible["detector_id"].nunique() >= MIN_PAIRWISE_DETECTORS
            and fold_count >= MIN_PAIRWISE_SPATIAL_FOLDS
        )
        support_rows.append(
            {
                "transition": transition,
                "base_year_detector_rows": len(base_frame),
                "target_year_detector_rows": len(target_frame),
                "detectors_observed_both_years": len(paired),
                "detectors_both_years_qualified": int(
                    paired["both_years_qualified"].sum()
                ),
                "detectors_positive_proxy": int(
                    (paired["both_years_qualified"] & paired["positive_proxies"]).sum()
                ),
                "detectors_pairwise_lane_stable": len(eligible_before),
                "detectors_linked_with_valid_fold": eligible[
                    "detector_id"
                ].nunique(),
                "spatial_fold_count": fold_count,
                "minimum_detector_requirement": MIN_PAIRWISE_DETECTORS,
                "minimum_fold_requirement": MIN_PAIRWISE_SPATIAL_FOLDS,
                "network_support_gate_pass": support_pass,
                "failed_criterion": ""
                if support_pass
                else "|".join(
                    item
                    for item, failed in (
                        (
                            "pairwise_detector_count_below_100",
                            eligible["detector_id"].nunique()
                            < MIN_PAIRWISE_DETECTORS,
                        ),
                        (
                            "pairwise_fold_count_below_4",
                            fold_count < MIN_PAIRWISE_SPATIAL_FOLDS,
                        ),
                    )
                    if failed
                ),
            }
        )
        for fold in range(1, 6):
            count = int(eligible["spatial_fold"].eq(fold).sum())
            fold_rows.append(
                {
                    "transition": transition,
                    "spatial_fold": fold,
                    "eligible_detector_count": count,
                    "eligible_detector_share": (
                        count / len(eligible) if len(eligible) else np.nan
                    ),
                    "detectors_available_outside_fold": int(
                        eligible["spatial_fold"].ne(fold).sum()
                    ),
                    "interpretation": "target-fold detectors are excluded when its temporal factor is estimated",
                }
            )

    pairwise = (
        pd.concat(pair_rows, ignore_index=True, sort=False)
        if pair_rows
        else pd.DataFrame()
    )
    support = pd.DataFrame(support_rows)
    fold_support = pd.DataFrame(fold_rows)
    save_csv(pairwise, PAIRWISE_PANEL_PATH)
    save_csv(support, SUPPORT_PATH)
    save_csv(fold_support, FOLD_SUPPORT_PATH)
    return pairwise, support, fold_support


def station_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        panel.sort_values(["station_id", "year"])
        .groupby("station_id", as_index=False)
        .first()
    )
    return metadata[metadata["road_network"].eq("MAJOR")].copy()


def crosswalk_for_transition(
    base,
    stations: pd.DataFrame,
    detectors: pd.DataFrame,
    transition: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "transition",
        "station_id",
        "detector_id",
        "distance_m",
        "road_name_similarity",
        "station_road_name",
        "detector_road_name",
        "station_spatial_fold",
        "detector_spatial_fold",
        "acceptance_rule",
    ]
    if stations.empty or detectors.empty:
        return pd.DataFrame(columns=columns), {
            "transition": transition,
            "candidate_station_count": 0,
            "candidate_detector_count": 0,
            "accepted_one_to_one_pairs": 0,
            "minimum_pair_requirement": MIN_CROSSWALK_PAIRS,
            "colocated_support_gate_pass": False,
            "failed_criterion": "no_eligible_station_detector_geometry",
        }

    reference_latitude = float(
        pd.concat([stations["latitude"], detectors["latitude"]]).median()
    )

    def xy(frame: pd.DataFrame) -> np.ndarray:
        x = frame["longitude"].to_numpy(float) * 111_320.0 * math.cos(
            math.radians(reference_latitude)
        )
        y = frame["latitude"].to_numpy(float) * 110_540.0
        return np.column_stack([x, y])

    station_xy = xy(stations)
    detector_xy = xy(detectors)
    tree = cKDTree(detector_xy)
    candidates: list[dict[str, object]] = []
    for station_position, detector_positions in enumerate(
        tree.query_ball_point(station_xy, r=MAX_CROSSWALK_DISTANCE_M)
    ):
        station = stations.iloc[station_position]
        for detector_position in detector_positions:
            detector = detectors.iloc[detector_position]
            distance = float(
                np.linalg.norm(
                    station_xy[station_position] - detector_xy[detector_position]
                )
            )
            similarity = base.road_name_similarity(
                station["station_road_name"], detector["road_name"]
            )
            accepted = (
                distance <= HIGH_CONFIDENCE_DISTANCE_M
                or similarity >= MIN_ROAD_NAME_SIMILARITY
            )
            if not accepted:
                continue
            candidates.append(
                {
                    "transition": transition,
                    "station_id": station["station_id"],
                    "detector_id": detector["detector_id"],
                    "distance_m": distance,
                    "road_name_similarity": similarity,
                    "station_road_name": station["station_road_name"],
                    "detector_road_name": detector["road_name"],
                    "station_spatial_fold": station["spatial_fold"],
                    "detector_spatial_fold": detector["spatial_fold"],
                    "acceptance_rule": (
                        "distance<=20m"
                        if distance <= HIGH_CONFIDENCE_DISTANCE_M
                        else "road_name_similarity>=0.50"
                    ),
                }
            )
    candidate_frame = pd.DataFrame(candidates, columns=columns)
    if candidate_frame.empty:
        crosswalk = pd.DataFrame(columns=columns)
    else:
        ordered = candidate_frame.sort_values(
            ["road_name_similarity", "distance_m"], ascending=[False, True]
        )
        used_stations: set[str] = set()
        used_detectors: set[str] = set()
        selected: list[dict[str, object]] = []
        for row in ordered.to_dict("records"):
            if row["station_id"] in used_stations:
                continue
            if row["detector_id"] in used_detectors:
                continue
            used_stations.add(row["station_id"])
            used_detectors.add(row["detector_id"])
            selected.append(row)
        crosswalk = pd.DataFrame(selected, columns=columns)
    pair_count = len(crosswalk)
    audit = {
        "transition": transition,
        "candidate_station_count": candidate_frame["station_id"].nunique()
        if not candidate_frame.empty
        else 0,
        "candidate_detector_count": candidate_frame["detector_id"].nunique()
        if not candidate_frame.empty
        else 0,
        "accepted_one_to_one_pairs": pair_count,
        "minimum_pair_requirement": MIN_CROSSWALK_PAIRS,
        "colocated_support_gate_pass": pair_count >= MIN_CROSSWALK_PAIRS,
        "failed_criterion": ""
        if pair_count >= MIN_CROSSWALK_PAIRS
        else "accepted_pair_count_below_20",
    }
    return crosswalk, audit


def build_crosswalks(
    base,
    panel: pd.DataFrame,
    pairwise: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = station_metadata(panel)
    station_pairs = build_station_pairs(base, panel)
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    support_map = support.set_index("transition")[
        "network_support_gate_pass"
    ].to_dict()
    for base_year, target_year in TRANSITIONS:
        transition = f"{base_year}-{target_year}"
        detectors = pairwise[pairwise["transition"].eq(transition)][
            [
                "detector_id",
                "longitude",
                "latitude",
                "road_name",
                "spatial_fold",
            ]
        ].drop_duplicates("detector_id")
        crosswalk, audit = crosswalk_for_transition(
            base, stations, detectors, transition
        )
        evaluable_station_ids = set(
            station_pairs.loc[
                station_pairs["transition"].eq(transition), "station_id"
            ].astype(str)
        )
        evaluable_pair_count = int(
            crosswalk["station_id"].astype(str).isin(evaluable_station_ids).sum()
        )
        accepted_pair_count = int(audit["accepted_one_to_one_pairs"])
        audit["geometric_pair_gate_pass"] = bool(
            accepted_pair_count >= MIN_CROSSWALK_PAIRS
        )
        audit["evaluable_adjacent_year_atc_pairs"] = evaluable_pair_count
        audit["evaluable_share_of_accepted_pairs"] = (
            evaluable_pair_count / accepted_pair_count
            if accepted_pair_count > 0
            else np.nan
        )
        audit["colocated_support_gate_pass"] = bool(
            evaluable_pair_count >= MIN_CROSSWALK_PAIRS
        )
        audit["failed_criterion"] = (
            ""
            if audit["colocated_support_gate_pass"]
            else "evaluable_adjacent_year_pair_count_below_20"
        )
        audit["network_support_gate_pass"] = bool(
            support_map.get(transition, False)
        )
        audit["transition_fully_supported"] = bool(
            audit["network_support_gate_pass"]
            and audit["colocated_support_gate_pass"]
        )
        rows.append(crosswalk)
        audits.append(audit)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    audit_frame = pd.DataFrame(audits)
    save_csv(result, CROSSWALK_PATH)
    save_csv(audit_frame, CROSSWALK_AUDIT_PATH)
    return result, audit_frame


def build_station_pairs(base, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for base_year, target_year in TRANSITIONS:
        base_frame = panel[panel["year"].eq(base_year)].copy()
        target_frame = panel[panel["year"].eq(target_year)].copy()
        merged = base_frame.merge(
            target_frame[["station_id", "aadt", "road_network"]],
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


def add_prediction_errors(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["no_change_prediction"] = result["aadt_base"]
    result["no_change_predicted_change"] = 0.0
    result["model_absolute_change_error"] = (
        result["predicted_change"] - result["observed_change"]
    ).abs()
    result["no_change_absolute_change_error"] = result["observed_change"].abs()
    result["absolute_error_difference_model_minus_no_change"] = (
        result["model_absolute_change_error"]
        - result["no_change_absolute_change_error"]
    )
    return result


def build_predictions(
    base,
    panel: pd.DataFrame,
    pairwise: pd.DataFrame,
    support: pd.DataFrame,
    crosswalk: pd.DataFrame,
    crosswalk_audit: pd.DataFrame,
) -> pd.DataFrame:
    station_pairs = build_station_pairs(base, panel)
    network_support = support.set_index("transition")[
        "network_support_gate_pass"
    ].to_dict()
    colocated_support = crosswalk_audit.set_index("transition")[
        "transition_fully_supported"
    ].to_dict()
    rows: list[pd.DataFrame] = []

    for base_year, target_year in TRANSITIONS:
        transition = f"{base_year}-{target_year}"
        station_transition = station_pairs[
            station_pairs["transition"].eq(transition)
        ].copy()
        ratio_frame = pairwise[pairwise["transition"].eq(transition)].copy()

        transition_crosswalk = crosswalk[
            crosswalk["transition"].eq(transition)
        ].copy()
        if not transition_crosswalk.empty:
            colocated = station_transition.merge(
                transition_crosswalk[
                    [
                        "station_id",
                        "detector_id",
                        "distance_m",
                        "road_name_similarity",
                    ]
                ],
                on="station_id",
                how="inner",
                validate="many_to_one",
            ).merge(
                ratio_frame[["detector_id", "detector_ratio"]],
                on="detector_id",
                how="inner",
                validate="many_to_one",
            )
            colocated["task"] = "colocated_temporal_transfer"
            colocated["temporal_factor"] = colocated["detector_ratio"]
            colocated["prediction"] = (
                colocated["aadt_base"] * colocated["temporal_factor"]
            )
            colocated["predicted_change"] = (
                colocated["prediction"] - colocated["aadt_base"]
            )
            colocated["cluster_id"] = colocated["detector_id"]
            colocated["factor_training_detector_count"] = 1
            colocated["transition_support_pass"] = bool(
                colocated_support.get(transition, False)
            )
            rows.append(colocated)

        for fold in range(1, 6):
            detector_folds = pd.to_numeric(
                ratio_frame["spatial_fold"], errors="coerce"
            )
            training = ratio_frame[
                detector_folds.notna() & detector_folds.ne(fold)
            ]["detector_ratio"].dropna()
            heldout = station_transition[
                pd.to_numeric(
                    station_transition["spatial_fold"], errors="coerce"
                ).eq(fold)
            ].copy()
            if training.empty or heldout.empty:
                continue
            factor = float(training.median())
            heldout["task"] = "heldout_network_factor"
            heldout["detector_id"] = ""
            heldout["distance_m"] = np.nan
            heldout["road_name_similarity"] = np.nan
            heldout["temporal_factor"] = factor
            heldout["prediction"] = heldout["aadt_base"] * factor
            heldout["predicted_change"] = (
                heldout["prediction"] - heldout["aadt_base"]
            )
            heldout["cluster_id"] = heldout["station_id"]
            heldout["factor_training_detector_count"] = len(training)
            heldout["transition_support_pass"] = bool(
                network_support.get(transition, False)
            )
            rows.append(heldout)

    predictions = (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    )
    predictions = add_prediction_errors(predictions)
    save_csv(predictions, PREDICTION_PATH)
    return predictions


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 3:
        return np.nan
    if valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return np.nan
    return float(valid["left"].corr(valid["right"]))


def metric_row(group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return {
            "n": 0,
            "cluster_count": 0,
            "model_change_mae": np.nan,
            "no_change_change_mae": np.nan,
            "mae_improvement_pct_vs_no_change": np.nan,
            "model_change_rmse": np.nan,
            "change_correlation": np.nan,
            "direction_accuracy": np.nan,
            "mean_observed_change": np.nan,
            "mean_predicted_change": np.nan,
        }
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
            np.sqrt(
                np.mean(
                    (group["predicted_change"] - group["observed_change"]) ** 2
                )
            )
        ),
        "change_correlation": safe_correlation(
            group["observed_change"], group["predicted_change"]
        ),
        "direction_accuracy": float(np.mean(observed_sign == predicted_sign)),
        "mean_observed_change": group["observed_change"].mean(),
        "mean_predicted_change": group["predicted_change"].mean(),
    }


def clustered_bootstrap_interval(
    frame: pd.DataFrame,
) -> tuple[float, float, float, int]:
    cluster_means = (
        frame.groupby("cluster_id")[
            "absolute_error_difference_model_minus_no_change"
        ]
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
        empty = pd.DataFrame()
        save_csv(empty, TRANSITION_METRICS_PATH)
        save_csv(empty, FOLD_METRICS_PATH)
        save_csv(empty, PAIRED_COMPARISON_PATH)
        return empty, empty, empty

    for (task, transition), group in predictions.groupby(["task", "transition"]):
        supported = bool(group["transition_support_pass"].all())
        difference, lower, upper, cluster_count = clustered_bootstrap_interval(group)
        transition_rows.append(
            {
                "task": task,
                "transition": transition,
                "transition_support_pass": supported,
                **metric_row(group),
                "mean_absolute_error_difference_model_minus_no_change": difference,
                "cluster_bootstrap_lower_95": lower,
                "cluster_bootstrap_upper_95": upper,
                "bootstrap_cluster_count": cluster_count,
                "transition_improves": bool(
                    supported
                    and metric_row(group)[
                        "mae_improvement_pct_vs_no_change"
                    ]
                    > 0
                ),
            }
        )
    transition_metrics = pd.DataFrame(transition_rows)
    save_csv(transition_metrics, TRANSITION_METRICS_PATH)

    supported_predictions = predictions[
        predictions["transition_support_pass"].fillna(False)
    ].copy()
    for (task, fold), group in supported_predictions.groupby(
        ["task", "spatial_fold"]
    ):
        metrics = metric_row(group)
        fold_rows.append(
            {
                "task": task,
                "spatial_fold": int(fold),
                **metrics,
                "fold_improves": bool(
                    metrics["mae_improvement_pct_vs_no_change"] > 0
                ),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    save_csv(fold_metrics, FOLD_METRICS_PATH)

    for task in sorted(predictions["task"].dropna().unique()):
        task_all = predictions[predictions["task"].eq(task)]
        task_supported = task_all[
            task_all["transition_support_pass"].fillna(False)
        ].copy()
        task_transition = transition_metrics[
            transition_metrics["task"].eq(task)
        ]
        task_folds = fold_metrics[fold_metrics["task"].eq(task)]
        supported_transition_count = int(
            task_transition["transition_support_pass"].sum()
        )
        improved_transition_count = int(task_transition["transition_improves"].sum())
        improved_fold_count = int(task_folds["fold_improves"].sum())
        metrics = metric_row(task_supported)
        difference, lower, upper, cluster_count = clustered_bootstrap_interval(
            task_supported
        )
        failed: list[str] = []
        if supported_transition_count < MIN_SUPPORTED_TRANSITIONS:
            failed.append("fewer_than_2_supported_transitions")
        if not metrics["mae_improvement_pct_vs_no_change"] >= MIN_MAE_IMPROVEMENT_PCT:
            failed.append("pooled_effect_below_5pct")
        if not upper < 0:
            if cluster_count < 2:
                failed.append("insufficient_clusters_for_interval")
            elif pd.notna(lower) and lower > 0:
                failed.append("cluster_interval_entirely_above_zero")
            else:
                failed.append("cluster_interval_includes_zero")
        if improved_transition_count < MIN_IMPROVED_TRANSITIONS:
            failed.append("fewer_than_2_of_3_transitions_improve")
        if improved_fold_count < MIN_IMPROVED_FOLDS:
            failed.append("fewer_than_3_of_5_folds_improve")
        if not metrics["change_correlation"] > 0:
            failed.append(
                "change_correlation_not_estimable"
                if pd.isna(metrics["change_correlation"])
                else "change_correlation_not_positive"
            )
        comparison_rows.append(
            {
                "task": task,
                **metrics,
                "mean_absolute_error_difference_model_minus_no_change": difference,
                "cluster_bootstrap_lower_95": lower,
                "cluster_bootstrap_upper_95": upper,
                "bootstrap_cluster_count": cluster_count,
                "supported_transition_count": supported_transition_count,
                "improved_transition_count": improved_transition_count,
                "improved_spatial_fold_count": improved_fold_count,
                "predeclared_task_gate_pass": len(failed) == 0,
                "failed_criterion": "|".join(failed),
                "unsupported_predictions_excluded_from_primary_score": len(task_all)
                - len(task_supported),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, PAIRED_COMPARISON_PATH)
    return transition_metrics, fold_metrics, comparison


def task_record(comparison: pd.DataFrame, task: str) -> dict[str, object]:
    if comparison.empty:
        return {}
    subset = comparison[comparison["task"].eq(task)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def task_evidence(record: dict[str, object]) -> str:
    if not record:
        return "no evaluable predictions"
    return (
        f"n={int(record.get('n', 0))}; supported transitions="
        f"{int(record.get('supported_transition_count', 0))}/3; MAE improvement="
        f"{record.get('mae_improvement_pct_vs_no_change', np.nan):.2f}%; interval=["
        f"{record.get('cluster_bootstrap_lower_95', np.nan):.1f},"
        f"{record.get('cluster_bootstrap_upper_95', np.nan):.1f}]; improved transitions="
        f"{int(record.get('improved_transition_count', 0))}/3; improved folds="
        f"{int(record.get('improved_spatial_fold_count', 0))}/5; correlation="
        f"{record.get('change_correlation', np.nan):.3f}"
    )


def make_decisions(
    support: pd.DataFrame,
    crosswalk_audit: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    network_supported = int(support["network_support_gate_pass"].sum())
    colocated_supported = int(
        crosswalk_audit["transition_fully_supported"].sum()
    )
    heldout = task_record(comparison, "heldout_network_factor")
    colocated = task_record(comparison, "colocated_temporal_transfer")
    heldout_pass = bool(heldout.get("predeclared_task_gate_pass", False))
    colocated_pass = bool(colocated.get("predeclared_task_gate_pass", False))
    final_pass = (
        network_supported >= MIN_SUPPORTED_TRANSITIONS
        and colocated_supported >= MIN_SUPPORTED_TRANSITIONS
        and heldout_pass
        and colocated_pass
    )
    rows = [
        {
            "decision": "primary_step25a_and_step25a1_results_preserved",
            "pass": True,
            "evidence": "Step 25A.2 writes only step25a2_* outputs",
            "failed_criterion": "",
            "action": "report all three designs separately",
        },
        {
            "decision": "pairwise_network_support_available_in_at_least_two_transitions",
            "pass": network_supported >= MIN_SUPPORTED_TRANSITIONS,
            "evidence": f"supported transitions={network_supported}/3; each requires >={MIN_PAIRWISE_DETECTORS} detectors in >={MIN_PAIRWISE_SPATIAL_FOLDS} folds",
            "failed_criterion": ""
            if network_supported >= MIN_SUPPORTED_TRANSITIONS
            else "fewer_than_2_supported_transitions",
            "action": "evaluate held-out factors" if network_supported >= 2 else "stop the pairwise network-factor route",
        },
        {
            "decision": "pairwise_colocated_support_available_in_at_least_two_transitions",
            "pass": colocated_supported >= MIN_SUPPORTED_TRANSITIONS,
            "evidence": f"fully supported colocated transitions={colocated_supported}/3; each also requires >={MIN_CROSSWALK_PAIRS} accepted one-to-one pairs with adjacent-year ATC observations",
            "failed_criterion": ""
            if colocated_supported >= MIN_SUPPORTED_TRANSITIONS
            else "fewer_than_2_transitions_with_20_colocated_pairs",
            "action": "evaluate colocated transfer" if colocated_supported >= 2 else "retain colocated calibration as non-identifiable",
        },
        {
            "decision": "pairwise_colocated_detector_change_beats_no_change",
            "pass": colocated_pass,
            "evidence": task_evidence(colocated),
            "failed_criterion": ""
            if colocated_pass
            else str(colocated.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as calibration evidence" if colocated_pass else "do not transfer detector ratios to ATC AADT",
        },
        {
            "decision": "pairwise_heldout_network_factor_beats_no_change",
            "pass": heldout_pass,
            "evidence": task_evidence(heldout) + "; every target-fold factor excludes detectors in that fold",
            "failed_criterion": ""
            if heldout_pass
            else str(heldout.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as deployment-oriented temporal evidence" if heldout_pass else "do not downscale the monitored-network trend",
        },
        {
            "decision": "step25b_recent_major_road_temporal_downscaling_authorised_by_pairwise_design",
            "pass": final_pass,
            "evidence": f"network_support={network_supported}/3; colocated_support={colocated_supported}/3; colocated_skill={colocated_pass}; heldout_skill={heldout_pass}",
            "failed_criterion": ""
            if final_pass
            else "at_least_one_predeclared_support_or_skill_gate_failed",
            "action": "proceed only to bounded recent-year major-road Step 25B" if final_pass else "stop before Step 25B and retain Step 25A.2 as a boundary result",
        },
        {
            "decision": "full_network_local_road_backcast_or_equity_trend_authorised",
            "pass": False,
            "evidence": "Step 25A.2 still uses strategic detectors and measured major-road ATC stations only",
            "failed_criterion": "estimand_outside_step25a2_support",
            "action": "keep local-road reconstruction and equity trends outside the authorised claims",
        },
    ]
    decisions = pd.DataFrame(rows)
    save_csv(decisions, DECISION_PATH)
    return decisions


def write_figures(
    support: pd.DataFrame,
    fold_support: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    transitions = support["transition"].astype(str).tolist()
    stages = [
        ("detectors_both_years_qualified", "both years qualified"),
        ("detectors_pairwise_lane_stable", "+ stable lane count"),
        ("detectors_linked_with_valid_fold", "+ linked frozen fold"),
    ]
    x = np.arange(len(transitions))
    width = 0.24
    for index, (column, label) in enumerate(stages):
        axes[0].bar(
            x + (index - 1) * width,
            support[column],
            width=width,
            label=label,
        )
    axes[0].axhline(
        MIN_PAIRWISE_DETECTORS,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="100-detector gate",
    )
    axes[0].set_xticks(x, transitions)
    axes[0].set_ylabel("Detector count")
    axes[0].set_title("Adjacent-year support attrition")
    axes[0].legend(fontsize=8)

    pivot = fold_support.pivot(
        index="transition", columns="spatial_fold", values="eligible_detector_count"
    ).reindex(transitions)
    bottom = np.zeros(len(pivot))
    for fold in range(1, 6):
        values = pivot.get(fold, pd.Series(0, index=pivot.index)).fillna(0).to_numpy()
        axes[1].bar(pivot.index, values, bottom=bottom, label=f"fold {fold}")
        bottom += values
    axes[1].set_ylabel("Eligible detectors")
    axes[1].set_title("Inherited spatial support by transition")
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle("Step 25A.2 pairwise detector support")
    figure.tight_layout()
    figure.savefig(SUPPORT_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {SUPPORT_FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    supported = (
        predictions[predictions["transition_support_pass"].fillna(False)].copy()
        if not predictions.empty
        else pd.DataFrame()
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for axis, task, title in zip(
        axes,
        ("colocated_temporal_transfer", "heldout_network_factor"),
        ("Colocated transfer", "Held-out network factor"),
    ):
        subset = supported[supported["task"].eq(task)] if not supported.empty else pd.DataFrame()
        if subset.empty:
            axis.text(0.5, 0.5, "No supported predictions", ha="center", va="center")
            axis.set_axis_off()
            continue
        for transition, group in subset.groupby("transition"):
            axis.scatter(
                group["observed_change"],
                group["predicted_change"],
                s=18,
                alpha=0.5,
                label=transition,
            )
        values = pd.concat(
            [subset["observed_change"], subset["predicted_change"]]
        ).dropna()
        lower, upper = values.min(), values.max()
        axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        axis.axhline(0, color="grey", linewidth=0.7)
        axis.axvline(0, color="grey", linewidth=0.7)
        axis.set_xlabel("Observed AADT change")
        axis.set_ylabel("Predicted AADT change")
        axis.set_title(title)
        axis.legend(fontsize=8)
    figure.suptitle("Step 25A.2: does the public detector proxy identify change?")
    figure.tight_layout()
    figure.savefig(CHANGE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {CHANGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def update_manifest() -> None:
    items = [
        (DESIGN_AUDIT_PATH, "reportable_design_audit", "predeclared adjacent-year design"),
        (SUPPORT_PATH, "reportable_validation_audit", "pairwise detector support by transition"),
        (FOLD_SUPPORT_PATH, "reportable_validation_audit", "pairwise frozen-fold support"),
        (CROSSWALK_AUDIT_PATH, "reportable_validation_audit", "pairwise colocated support"),
        (TRANSITION_METRICS_PATH, "reportable_validation_audit", "change skill by adjacent transition"),
        (FOLD_METRICS_PATH, "reportable_validation_audit", "change skill by frozen spatial fold"),
        (PAIRED_COMPARISON_PATH, "reportable_validation_audit", "no-change comparisons on supported transitions"),
        (DECISION_PATH, "reportable_decision_audit", "Step 25A.2 bounded temporal decision"),
        (SUPPORT_FIGURE_PATH, "reportable_data_audit", "adjacent-year support and fold coverage"),
        (CHANGE_FIGURE_PATH, "reportable_validation_diagnostic", "change-identification diagnostics"),
        (PAIRWISE_PANEL_PATH, "provenance_only", "eligible detector ratios; not AADT"),
        (CROSSWALK_PATH, "provenance_only", "transition-specific detector-to-major-ATC links"),
        (PREDICTION_PATH, "provenance_only", "pairwise temporal predictions"),
    ]
    additions = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "report_status": status,
                "interpretation": interpretation,
            }
            for path, status, interpretation in items
            if path.exists()
        ]
    )
    if not REPORT_MANIFEST_PATH.exists():
        save_csv(additions, REPORT_MANIFEST_PATH)
        return
    existing = pd.read_csv(REPORT_MANIFEST_PATH)
    lookup = {str(column).strip().lower(): str(column) for column in existing.columns}
    path_column = next(
        (
            lookup[candidate]
            for candidate in (
                "path",
                "artifact_path",
                "file_path",
                "relative_path",
                "output_path",
                "artifact",
                "file",
                "filename",
            )
            if candidate in lookup
        ),
        None,
    )
    if path_column is None:
        existing = existing.copy()
        existing["path"] = ""
        path_column = "path"
    additions = additions.rename(columns={"path": path_column})
    existing = existing[
        ~existing[path_column].astype(str).isin(additions[path_column].astype(str))
    ]
    output = pd.concat([existing, additions], ignore_index=True, sort=False)
    save_csv(output, REPORT_MANIFEST_PATH)


def run() -> pd.DataFrame:
    require_inputs()
    base = load_base_module()
    make_design_audit()
    annual = pd.read_csv(PAIRWISE_ANNUAL_INPUT)
    locations = prepare_locations(base)
    pairwise, support, fold_support = build_pairwise_detector_panel(
        annual, locations
    )
    panel = base.prepare_measured_panel()
    crosswalk, crosswalk_audit = build_crosswalks(
        base, panel, pairwise, support
    )
    predictions = build_predictions(
        base, panel, pairwise, support, crosswalk, crosswalk_audit
    )
    _, _, comparison = evaluate_predictions(predictions)
    decisions = make_decisions(support, crosswalk_audit, comparison)
    write_figures(support, fold_support, predictions)
    update_manifest()
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for directory in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Step 25A.2 revision: {STEP25A2_REVISION}")
    print("Step 25A and Step 25A.1 outputs will not be overwritten.")
    print("Running the predeclared adjacent-year unbalanced-panel validation...")
    decisions = run()
    print("\nStep 25A.2 adjacent-year temporal validation is complete.")
    for row in decisions.to_dict("records"):
        print(f"  {row['decision']}: {row['pass']}")
    print(
        "  Passing can authorise only a bounded recent-year major-road Step 25B; "
        "the detector proxy is not AADT."
    )


if __name__ == "__main__":
    main()
