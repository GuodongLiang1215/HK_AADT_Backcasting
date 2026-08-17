"""Step 25A.1: revised-window strategic-detector temporal validation.

This is a prospectively labelled sensitivity/revised-design analysis.  It
does not overwrite or retroactively pass Step 25A.  Step 25A.1 asks whether a
common April--December sample, justified by the observed public-archive
availability rather than by model performance, is sufficient to materialise
the two temporal-change tests that Step 25A could not identify.

The following Step 25A rules remain unchanged:

* 90% archive-year retrieval support and complete month/day-type/time-block
  coverage;
* 80% detector-year support, at least eight months and all six strata;
* at least 100 stable detectors in at least four spatial folds;
* the frozen 100 m/20 m/name-similarity detector--ATC crosswalk;
* at least 20 accepted one-to-one pairs; and
* the frozen no-change comparison, effect-size, transition, fold,
  cluster-interval and positive-change-correlation gates.

The balanced detector proxy is not AADT.  Passing this step could authorise
only a later bounded major-road experiment.  It cannot authorise local-road
reconstruction, a full-network backcast or an equity trend.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "src" / "25a_validate_strategic_detector_temporal_signal.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

BASE_SAMPLING_AUDIT_PATH = TABLE_DIR / "step25a_sampling_audit.csv"
BASE_SNAPSHOT_LONG_PATH = PROCESSED_DIR / "atc_step25a_strategic_snapshot_long.csv"

REVISED_SAMPLING_AUDIT_PATH = TABLE_DIR / "step25a1_sampling_audit.csv"
REVISED_SNAPSHOT_LONG_PATH = PROCESSED_DIR / "atc_step25a1_strategic_snapshot_long.csv"
REVISED_ANNUAL_PROXY_PATH = PROCESSED_DIR / "atc_step25a1_strategic_annual_proxy.csv"
REVISED_STABLE_PANEL_PATH = PROCESSED_DIR / "atc_step25a1_stable_detector_panel.csv"
REVISED_CROSSWALK_PATH = PROCESSED_DIR / "atc_step25a1_detector_atc_crosswalk.csv"
REVISED_PREDICTION_PATH = PROCESSED_DIR / "atc_step25a1_temporal_predictions.csv"

DESIGN_AUDIT_PATH = TABLE_DIR / "step25a1_design_audit.csv"
ANNUAL_AUDIT_PATH = TABLE_DIR / "step25a1_annual_proxy_audit.csv"
SUPPORT_ATTRITION_PATH = TABLE_DIR / "step25a1_support_attrition.csv"
DETECTOR_RETENTION_PATH = TABLE_DIR / "step25a1_detector_retention.csv"
DETECTOR_ID_AUDIT_PATH = TABLE_DIR / "step25a1_detector_id_linkage_audit.csv"
FOLD_INTERFACE_AUDIT_PATH = TABLE_DIR / "step25a1_fold_interface_audit.csv"
CROSSWALK_AUDIT_PATH = TABLE_DIR / "step25a1_crosswalk_audit.csv"
TRANSITION_METRICS_PATH = TABLE_DIR / "step25a1_metrics_by_transition.csv"
FOLD_METRICS_PATH = TABLE_DIR / "step25a1_metrics_by_fold.csv"
PAIRED_COMPARISON_PATH = TABLE_DIR / "step25a1_paired_model_comparison.csv"
DECISION_PATH = TABLE_DIR / "step25a1_decision_audit.csv"

COVERAGE_FIGURE_PATH = FIGURE_DIR / "step25a1_archive_and_panel_coverage.png"
CHANGE_FIGURE_PATH = FIGURE_DIR / "step25a1_change_identification.png"

REVISED_COMMON_MONTHS = tuple(range(4, 13))
EXPECTED_REVISED_SNAPSHOTS = len(REVISED_COMMON_MONTHS) * 2 * 3
STEP25A1_REVISION = "2026-08-17.1"


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def load_base_module():
    if not BASE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "Step 25A source is missing: "
            f"{BASE_SCRIPT_PATH.relative_to(PROJECT_ROOT)}"
        )
    specification = importlib.util.spec_from_file_location(
        "step25a_base_for_revised_design", BASE_SCRIPT_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the frozen Step 25A source module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def configure_revised_outputs(base) -> None:
    """Redirect every inherited writer away from the primary Step 25A files."""

    base.COMMON_MONTHS = REVISED_COMMON_MONTHS
    base.EXPECTED_SNAPSHOTS_PER_YEAR = EXPECTED_REVISED_SNAPSHOTS

    base.SNAPSHOT_LONG_PATH = REVISED_SNAPSHOT_LONG_PATH
    base.ANNUAL_PROXY_PATH = REVISED_ANNUAL_PROXY_PATH
    base.STABLE_PANEL_PATH = REVISED_STABLE_PANEL_PATH
    base.CROSSWALK_PATH = REVISED_CROSSWALK_PATH
    base.PREDICTION_PATH = REVISED_PREDICTION_PATH

    base.SAMPLING_AUDIT_PATH = REVISED_SAMPLING_AUDIT_PATH
    base.ANNUAL_AUDIT_PATH = ANNUAL_AUDIT_PATH
    base.CROSSWALK_AUDIT_PATH = CROSSWALK_AUDIT_PATH
    base.DETECTOR_ID_AUDIT_PATH = DETECTOR_ID_AUDIT_PATH
    base.TRANSITION_METRICS_PATH = TRANSITION_METRICS_PATH
    base.FOLD_METRICS_PATH = FOLD_METRICS_PATH
    base.PAIRED_COMPARISON_PATH = PAIRED_COMPARISON_PATH
    base.DECISION_PATH = DECISION_PATH

    base.COVERAGE_FIGURE_PATH = COVERAGE_FIGURE_PATH
    base.CHANGE_FIGURE_PATH = CHANGE_FIGURE_PATH


def require_inputs(base) -> None:
    required = (
        BASE_SAMPLING_AUDIT_PATH,
        BASE_SNAPSHOT_LONG_PATH,
        base.MEASURED_PANEL_PATH,
        base.DETECTOR_LOCATION_PATH,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Step 25A.1 inputs are missing: {joined}. "
            "Run the corrected Steps 18, 24 and Step 25A sample phase first."
        )


def prepare_revised_window(base) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(BASE_SAMPLING_AUDIT_PATH)
    long = pd.read_csv(BASE_SNAPSHOT_LONG_PATH)

    audit["year"] = pd.to_numeric(audit["year"], errors="coerce")
    audit["month"] = pd.to_numeric(audit["month"], errors="coerce")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["month"] = pd.to_numeric(long["month"], errors="coerce")
    revised_audit = audit[
        audit["year"].isin(base.PRIMARY_YEARS)
        & audit["month"].isin(REVISED_COMMON_MONTHS)
    ].copy()
    revised_long = long[
        long["year"].isin(base.PRIMARY_YEARS)
        & long["month"].isin(REVISED_COMMON_MONTHS)
    ].copy()

    revised_audit["step25a1_design"] = "revised_april_december_sensitivity"
    revised_long["step25a1_design"] = "revised_april_december_sensitivity"
    save_csv(revised_audit, REVISED_SAMPLING_AUDIT_PATH)
    save_csv(revised_long, REVISED_SNAPSHOT_LONG_PATH)
    return revised_audit, revised_long


def make_design_audit(base, revised_audit: pd.DataFrame) -> pd.DataFrame:
    march_2021 = pd.read_csv(BASE_SAMPLING_AUDIT_PATH)
    march_2021 = march_2021[
        march_2021["year"].eq(2021) & march_2021["month"].eq(3)
    ]
    march_obtained = int(march_2021["status"].eq("obtained").sum())
    revised_2021 = revised_audit[revised_audit["year"].eq(2021)]
    revised_2021_obtained = int(revised_2021["status"].eq("obtained").sum())
    rows = [
        {
            "design_item": "analysis_status",
            "value": "prospectively_labelled_revised_design_sensitivity",
            "evidence": "Step 25A primary outputs remain unchanged and retain their failed/non-evaluable decision",
        },
        {
            "design_item": "window_change_justification",
            "value": "public_archive_availability_not_model_performance",
            "evidence": (
                f"2021 March obtained={march_obtained}/6; revised April-December "
                f"obtained={revised_2021_obtained}/{EXPECTED_REVISED_SNAPSHOTS}"
            ),
        },
        {
            "design_item": "revised_common_months",
            "value": "April-December",
            "evidence": "nine common calendar months; 54 fixed requested snapshots per year",
        },
        {
            "design_item": "model_and_validation_thresholds",
            "value": "unchanged_from_step25a",
            "evidence": (
                f"archive={base.MIN_YEAR_SAMPLE_SHARE:.0%}; detector-year={base.MIN_DETECTOR_YEAR_SAMPLE_SHARE:.0%}; "
                f"stable_detectors={base.MIN_STABLE_DETECTORS}; folds={base.MIN_STABLE_SPATIAL_FOLDS}; "
                f"crosswalk_pairs={base.MIN_CROSSWALK_PAIRS}; MAE_gain={base.MIN_MAE_IMPROVEMENT_PCT:.0f}%"
            ),
        },
        {
            "design_item": "claim_boundary",
            "value": "bounded_major_road_only",
            "evidence": "the detector proxy is not AADT and cannot authorise local-road or equity-trend claims",
        },
    ]
    design = pd.DataFrame(rows)
    save_csv(design, DESIGN_AUDIT_PATH)
    return design


def detector_retention_table(base, annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_years = set(base.PRIMARY_YEARS)
    for detector_id, group in annual.groupby("detector_id"):
        observed_years = set(pd.to_numeric(group["year"], errors="coerce").dropna().astype(int))
        all_years = primary_years.issubset(observed_years)
        indexed = group.set_index("year").reindex(base.PRIMARY_YEARS)
        qualified_all = bool(
            all_years
            and indexed["detector_year_qualified"].fillna(False).all()
        )
        lane_values = (
            pd.to_numeric(indexed["modal_valid_lane_count"], errors="coerce")
            .dropna()
            .round()
            .astype(int)
            .unique()
            if all_years
            else np.asarray([])
        )
        stable_lanes = bool(qualified_all and len(lane_values) == 1)
        rows.append(
            {
                "detector_id": base.normalise_identifier(detector_id),
                "observed_primary_year_count": len(observed_years.intersection(primary_years)),
                "observed_all_primary_years": all_years,
                "qualified_all_primary_years": qualified_all,
                "modal_lane_configuration_stable": stable_lanes,
                "stable_detector_panel_member": bool(qualified_all and stable_lanes),
                "modal_lane_configuration_values": "|".join(map(str, sorted(lane_values))),
            }
        )
    retention = pd.DataFrame(rows).sort_values("detector_id")
    save_csv(retention, DETECTOR_RETENTION_PATH)
    return retention


def inspect_fold_interface(base, stable_ids: set[str]) -> pd.DataFrame:
    locations = pd.read_csv(base.DETECTOR_LOCATION_PATH)
    source_col = base.find_column(locations, ("source",))
    id_col = base.find_column(locations, ("device_id", "detector_id"))
    fold_col = base.find_column(
        locations, ("nearest_spatial_fold", "spatial_fold"), required=False
    )
    strategic = locations[locations[source_col].astype(str).eq("strategic_detector")].copy()
    strategic["normalised_detector_id"] = strategic[id_col].map(base.normalise_identifier)
    linked = strategic[strategic["normalised_detector_id"].isin(stable_ids)].copy()
    if fold_col:
        linked_folds = pd.to_numeric(linked[fold_col], errors="coerce")
    else:
        linked_folds = pd.Series(np.nan, index=linked.index, dtype=float)

    rows = [
        {
            "metric": "fold_source_column",
            "value": fold_col or "missing",
            "interpretation": "Step 25A.1 reads the frozen Step 24 nearest-fold assignment; it does not create new folds",
        },
        {
            "metric": "stable_ids_requested",
            "value": len(stable_ids),
            "interpretation": "revised-window stable detector IDs",
        },
        {
            "metric": "stable_ids_linked_to_current_rows",
            "value": linked["normalised_detector_id"].nunique(),
            "interpretation": "ID linkage is reported separately from minimum sample support",
        },
        {
            "metric": "linked_stable_ids_with_valid_fold",
            "value": linked.loc[linked_folds.notna(), "normalised_detector_id"].nunique(),
            "interpretation": "linked detectors with an inherited non-missing fold",
        },
        {
            "metric": "linked_stable_spatial_fold_count",
            "value": linked_folds.dropna().astype(int).nunique(),
            "interpretation": "must cover at least four frozen spatial folds before spatial transfer is evaluable",
        },
        {
            "metric": "linked_rows_with_invalid_fold",
            "value": int(linked_folds.isna().sum()),
            "interpretation": "a non-zero value is an interface/data-field failure, not evidence against the temporal signal",
        },
    ]
    audit = pd.DataFrame(rows)
    save_csv(audit, FOLD_INTERFACE_AUDIT_PATH)
    return audit


def audit_value(frame: pd.DataFrame, metric: str, default: int = 0) -> int:
    if frame.empty or "metric" not in frame or "value" not in frame:
        return default
    values = frame.loc[frame["metric"].eq(metric), "value"]
    if values.empty:
        return default
    value = pd.to_numeric(values.iloc[0], errors="coerce")
    return int(value) if pd.notna(value) else default


def make_support_attrition(
    base,
    annual: pd.DataFrame,
    retention: pd.DataFrame,
    stable: pd.DataFrame,
    detectors: pd.DataFrame,
    fold_audit: pd.DataFrame,
    crosswalk_audit: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    archive_ids = set(annual["detector_id"].map(base.normalise_identifier))
    observed_all = set(
        retention.loc[retention["observed_all_primary_years"], "detector_id"]
    )
    qualified_all = set(
        retention.loc[retention["qualified_all_primary_years"], "detector_id"]
    )
    stable_ids = set(stable["detector_id"].map(base.normalise_identifier)) if not stable.empty else set()
    linked_ids = set(detectors["detector_id"].map(base.normalise_identifier)) if not detectors.empty else set()
    linked_with_fold = audit_value(fold_audit, "linked_stable_ids_with_valid_fold")
    candidate_detectors = audit_value(
        crosswalk_audit, "stable_detectors_within_100m_candidate"
    )

    stages = [
        ("archive_detector_seen", len(archive_ids), "appears at least once in the revised-window archive"),
        ("observed_all_four_years", len(observed_all), "appears in every primary year"),
        ("qualified_all_four_years", len(qualified_all), "passes unchanged detector-year coverage and six-strata rules every year"),
        ("stable_lane_configuration", len(stable_ids), "also has one modal valid-lane count across all four years"),
        ("linked_to_current_location", len(linked_ids), "stable ID links to the current strategic-detector location table"),
        ("linked_with_frozen_spatial_fold", linked_with_fold, "linked stable ID inherits a valid Step 24 frozen fold"),
        ("within_100m_major_atc_candidate", candidate_detectors, "stable detector is within the frozen 100 m ATC candidate radius"),
        ("accepted_one_to_one_pair", len(crosswalk), "passes the frozen distance/name acceptance and one-to-one assignment"),
    ]
    rows: list[dict[str, object]] = []
    starting_count = stages[0][1] if stages else 0
    previous_count: int | None = None
    for order, (stage, count, definition) in enumerate(stages, start=1):
        rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "eligible_count": count,
                "retention_from_previous_pct": (
                    100.0 * count / previous_count if previous_count else np.nan
                ),
                "retention_from_archive_start_pct": (
                    100.0 * count / starting_count if starting_count else np.nan
                ),
                "definition": definition,
            }
        )
        previous_count = count
    attrition = pd.DataFrame(rows)
    save_csv(attrition, SUPPORT_ATTRITION_PATH)
    return attrition


def correct_comparison_semantics(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        save_csv(comparison, PAIRED_COMPARISON_PATH)
        return comparison
    revised = comparison.copy()
    for index, row in revised.iterrows():
        failed = [item for item in str(row.get("failed_criterion", "")).split("|") if item]
        cluster_count = pd.to_numeric(
            pd.Series([row.get("bootstrap_cluster_count", np.nan)]), errors="coerce"
        ).iloc[0]
        if pd.isna(cluster_count) or cluster_count < 2:
            failed = [
                "insufficient_clusters_for_interval"
                if item == "cluster_interval_includes_zero"
                else item
                for item in failed
            ]
        correlation = pd.to_numeric(
            pd.Series([row.get("change_correlation", np.nan)]), errors="coerce"
        ).iloc[0]
        if pd.isna(correlation):
            failed = [
                "change_correlation_not_estimable"
                if item == "change_correlation_not_positive"
                else item
                for item in failed
            ]
        revised.at[index, "failed_criterion"] = "|".join(dict.fromkeys(failed))
    save_csv(revised, PAIRED_COMPARISON_PATH)
    return revised


def task_record(comparison: pd.DataFrame, task: str) -> dict[str, object]:
    if comparison.empty or "task" not in comparison:
        return {}
    subset = comparison[comparison["task"].eq(task)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def task_evidence(record: dict[str, object]) -> str:
    if not record:
        return "no evaluable paired predictions"
    return (
        f"n={int(record.get('n', 0))}; clusters={int(record.get('cluster_count', 0))}; "
        f"MAE improvement={record.get('mae_improvement_pct_vs_no_change', np.nan):.2f}%; "
        f"cluster interval=[{record.get('cluster_bootstrap_lower_95', np.nan):.1f}, "
        f"{record.get('cluster_bootstrap_upper_95', np.nan):.1f}]; "
        f"improved transitions={int(record.get('improved_transition_count', 0))}/3; "
        f"improved folds={int(record.get('improved_spatial_fold_count', 0))}/5; "
        f"change correlation={record.get('change_correlation', np.nan):.3f}"
    )


def make_revised_decisions(
    base,
    sampling_audit: pd.DataFrame,
    stable: pd.DataFrame,
    detectors: pd.DataFrame,
    fold_audit: pd.DataFrame,
    crosswalk: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    archive_pass, archive_evidence = base.archive_sample_gate(sampling_audit)
    stable_count = stable["detector_id"].nunique() if not stable.empty else 0
    linked_count = detectors["detector_id"].nunique() if not detectors.empty else 0
    fold_linked_count = audit_value(fold_audit, "linked_stable_ids_with_valid_fold")
    fold_count = audit_value(fold_audit, "linked_stable_spatial_fold_count")

    stable_pass = stable_count >= base.MIN_STABLE_DETECTORS
    id_linkage_pass = stable_count > 0 and linked_count == stable_count
    fold_support_pass = (
        fold_linked_count >= base.MIN_STABLE_DETECTORS
        and fold_count >= base.MIN_STABLE_SPATIAL_FOLDS
    )
    crosswalk_pass = len(crosswalk) >= base.MIN_CROSSWALK_PAIRS

    colocated = task_record(comparison, "colocated_temporal_transfer")
    heldout = task_record(comparison, "heldout_network_factor")
    colocated_pass = bool(colocated.get("predeclared_task_gate_pass", False))
    heldout_pass = bool(heldout.get("predeclared_task_gate_pass", False))
    final_pass = all(
        [
            archive_pass,
            stable_pass,
            id_linkage_pass,
            fold_support_pass,
            crosswalk_pass,
            colocated_pass,
            heldout_pass,
        ]
    )

    rows = [
        {
            "decision": "primary_step25a_result_preserved",
            "pass": True,
            "evidence": "Step 25A.1 writes only step25a1_* outputs and does not overwrite the primary failed/non-evaluable gate",
            "failed_criterion": "",
            "action": "report primary and revised-design results separately",
        },
        {
            "decision": "revised_april_december_archive_materialised",
            "pass": archive_pass,
            "evidence": archive_evidence,
            "failed_criterion": "" if archive_pass else "one_or_more_years_fail_unchanged_archive_rule",
            "action": "annualise revised-window proxy" if archive_pass else "retain as a failed revised data-window gate",
        },
        {
            "decision": "revised_stable_detector_panel_has_minimum_support",
            "pass": stable_pass,
            "evidence": f"stable detectors={stable_count}; unchanged requirement>={base.MIN_STABLE_DETECTORS}",
            "failed_criterion": "" if stable_pass else "stable_detector_count_below_threshold",
            "action": "continue support audit" if stable_pass else "do not generalise across the monitored network",
        },
        {
            "decision": "stable_detector_ids_link_to_current_locations",
            "pass": id_linkage_pass,
            "evidence": f"linked stable IDs={linked_count}/{stable_count}; ID linkage is separated from sample-size and fold gates",
            "failed_criterion": "" if id_linkage_pass else "stable_id_current_location_linkage_incomplete",
            "action": "retain current-coordinate survivor limitation" if id_linkage_pass else "repair the ID namespace/interface",
        },
        {
            "decision": "stable_detector_spatial_fold_support",
            "pass": fold_support_pass,
            "evidence": f"linked IDs with valid fold={fold_linked_count}; folds={fold_count}; unchanged requirements>={base.MIN_STABLE_DETECTORS} IDs and >={base.MIN_STABLE_SPATIAL_FOLDS} folds",
            "failed_criterion": "" if fold_support_pass else "fold_interface_or_spatial_support_below_threshold",
            "action": "run held-out spatial transfer" if fold_support_pass else "do not interpret a missing fold field as evidence against the temporal signal",
        },
        {
            "decision": "revised_current_coordinate_crosswalk_has_minimum_support",
            "pass": crosswalk_pass,
            "evidence": f"accepted one-to-one major-road pairs={len(crosswalk)}; unchanged requirement>={base.MIN_CROSSWALK_PAIRS}",
            "failed_criterion": "" if crosswalk_pass else "accepted_pair_count_below_threshold",
            "action": "run colocated transfer" if crosswalk_pass else "the frozen colocated validation remains non-identifiable",
        },
        {
            "decision": "revised_colocated_detector_change_beats_no_change",
            "pass": colocated_pass,
            "evidence": task_evidence(colocated),
            "failed_criterion": "" if colocated_pass else str(colocated.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as necessary temporal calibration evidence" if colocated_pass else "do not transfer colocated detector ratios to AADT",
        },
        {
            "decision": "revised_heldout_network_factor_beats_no_change",
            "pass": heldout_pass,
            "evidence": task_evidence(heldout) + "; each factor excludes the held-out frozen spatial fold",
            "failed_criterion": "" if heldout_pass else str(heldout.get("failed_criterion", "no_evaluable_predictions")),
            "action": "retain as deployment-oriented temporal evidence" if heldout_pass else "do not downscale the monitored-network trend",
        },
        {
            "decision": "step25b_major_road_temporal_downscaling_authorised_by_revised_design",
            "pass": final_pass,
            "evidence": (
                f"archive={archive_pass}; stable_panel={stable_pass}; ID_linkage={id_linkage_pass}; "
                f"fold_support={fold_support_pass}; crosswalk={crosswalk_pass}; "
                f"colocated={colocated_pass}; heldout={heldout_pass}"
            ),
            "failed_criterion": "" if final_pass else "at_least_one_unchanged_required_gate_failed",
            "action": "proceed only to bounded major-road Step 25B" if final_pass else "stop before Step 25B and report which support or skill gate failed",
        },
        {
            "decision": "full_network_local_road_backcast_or_equity_trend_authorised",
            "pass": False,
            "evidence": "Step 25A.1 still uses strategic detectors and measured major-road ATC stations only",
            "failed_criterion": "estimand_outside_step25a1_support",
            "action": "keep local-road reconstruction and equity trends outside the authorised claims",
        },
    ]
    decisions = pd.DataFrame(rows)
    save_csv(decisions, DECISION_PATH)
    return decisions


def update_manifest(base) -> None:
    items = [
        (DESIGN_AUDIT_PATH, "reportable_design_audit", "revised window justification and unchanged gates"),
        (SUPPORT_ATTRITION_PATH, "reportable_validation_audit", "detector support attrition from archive to accepted ATC pair"),
        (FOLD_INTERFACE_AUDIT_PATH, "reportable_validation_audit", "ID linkage and frozen-fold interface kept separate"),
        (DECISION_PATH, "reportable_decision_audit", "Step 25A.1 bounded temporal decision"),
        (PAIRED_COMPARISON_PATH, "reportable_validation_audit", "revised-window no-change comparisons"),
        (COVERAGE_FIGURE_PATH, "reportable_data_audit", "revised-window archive and stable-panel support"),
        (CHANGE_FIGURE_PATH, "reportable_validation_diagnostic", "revised-window temporal change identification"),
        (REVISED_ANNUAL_PROXY_PATH, "provenance_only", "revised-window detector-year proxy; not AADT"),
        (REVISED_STABLE_PANEL_PATH, "provenance_only", "revised-window stable detector panel"),
        (REVISED_PREDICTION_PATH, "provenance_only", "revised-window paired predictions"),
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
    if base.REPORT_MANIFEST_PATH.exists():
        existing = pd.read_csv(base.REPORT_MANIFEST_PATH)
        normalised_columns = {
            str(column).strip().lower(): str(column) for column in existing.columns
        }
        path_column = next(
            (
                normalised_columns[candidate]
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
                if candidate in normalised_columns
            ),
            None,
        )
        if path_column is None:
            path_column = next(
                (
                    str(column)
                    for column in existing.columns
                    if any(
                        token in str(column).strip().lower()
                        for token in ("path", "file", "artifact")
                    )
                ),
                None,
            )

        if path_column is not None:
            additions = additions.rename(columns={"path": path_column})
            existing = existing[
                ~existing[path_column]
                .astype(str)
                .isin(additions[path_column].astype(str))
            ]

        status_column = next(
            (
                normalised_columns[candidate]
                for candidate in ("report_status", "status", "classification")
                if candidate in normalised_columns
            ),
            None,
        )
        if status_column and status_column != "report_status":
            additions = additions.rename(columns={"report_status": status_column})

        interpretation_column = next(
            (
                normalised_columns[candidate]
                for candidate in ("interpretation", "description", "note", "notes")
                if candidate in normalised_columns
            ),
            None,
        )
        if interpretation_column and interpretation_column != "interpretation":
            additions = additions.rename(
                columns={"interpretation": interpretation_column}
            )

        output = pd.concat([existing, additions], ignore_index=True, sort=False)
    else:
        output = additions
    save_csv(output, base.REPORT_MANIFEST_PATH)


def run() -> pd.DataFrame:
    base = load_base_module()
    configure_revised_outputs(base)
    require_inputs(base)

    revised_audit, revised_long = prepare_revised_window(base)
    make_design_audit(base, revised_audit)
    annual, stable = base.annualise_detector_proxy(revised_long)
    retention = detector_retention_table(base, annual)

    stable_ids = (
        set(stable["detector_id"].map(base.normalise_identifier))
        if not stable.empty
        else set()
    )
    fold_audit = inspect_fold_interface(base, stable_ids)
    detectors = base.prepare_detector_locations(stable_ids)
    panel = base.prepare_measured_panel()
    crosswalk, crosswalk_audit = base.build_crosswalk(panel, detectors)
    make_support_attrition(
        base,
        annual,
        retention,
        stable,
        detectors,
        fold_audit,
        crosswalk_audit,
        crosswalk,
    )

    predictions = base.build_temporal_predictions(panel, stable, detectors, crosswalk)
    _, _, comparison = base.evaluate_predictions(predictions)
    comparison = correct_comparison_semantics(comparison)
    decisions = make_revised_decisions(
        base,
        revised_audit,
        stable,
        detectors,
        fold_audit,
        crosswalk,
        comparison,
    )

    annual_audit = pd.read_csv(ANNUAL_AUDIT_PATH)
    base.write_figures(revised_audit, annual_audit, predictions)
    update_manifest(base)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for directory in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Step 25A.1 revision: {STEP25A1_REVISION}")
    print("Primary Step 25A outputs will not be overwritten.")
    print("Using the source-justified April-December revised window with unchanged gates...")
    decisions = run()
    print("\nStep 25A.1 revised-window temporal validation is complete.")
    for row in decisions.to_dict("records"):
        print(f"  {row['decision']}: {row['pass']}")
    print("  The detector proxy is not AADT; only a fully passing result can authorise bounded major-road Step 25B.")


if __name__ == "__main__":
    main()
