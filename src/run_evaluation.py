"""
End-to-end evaluation runner - AI Trust Score DMS Framework.

Usage (from project root):
    python src/run_evaluation.py [label_dir] [--sample N]

Defaults to ../../DMS-SafetyEstimateLimittime/data/raw relative to this file
if no label_dir is passed, which points at the same dataset used in the
SafetyEstimateLimit prototype.

Outputs
-------
  - Console report (scorecards, RQ results, CI table)
  - outputs/trust_scores.csv
  - outputs/kpi_detail.csv
  - outputs/rq_results.json
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_labels, summary
from kpi_engine import compute_all_kpis
from trust_score import compute_trust_score, format_scorecard, AHP_WEIGHTS
from rq_analysis import (
    rq1_kpi_scenario_comparison,
    rq3_discriminative_power,
    rq4_dimension_influence,
    confidence_intervals,
)

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_LABEL_DIR = str(
    Path(__file__).parent.parent.parent /
    "DMS-SafetyEstimateLimittime" / "data" / "raw"
)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _section(title: str):
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def _build_scenario_score_records(df: pd.DataFrame, model: str) -> list[dict]:
    """
    Split data into per-scenario batches, compute trust score for each.
    Returns records suitable for RQ4 regression.
    """
    records = []
    for sc in df["scenario_type"].unique():
        sub = df[df["scenario_type"] == sc]
        if len(sub) < 20:
            continue
        kpis = compute_all_kpis(sub, model)
        ts   = compute_trust_score(kpis)
        rec  = {**ts["dimension_scores"], "trust_score": ts["trust_score"], "scenario": sc}
        records.append(rec)
    return records


def _edge_failure_count(df: pd.DataFrame, model: str) -> int:
    """Count wrong predictions on edge-case (non-normal) scenarios."""
    edge = df[df["scenario_type"] != "normal"]
    y_true = edge["y_true"].values
    y_pred = edge[f"y_pred_{model}"].values
    return int((y_true != y_pred).sum())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(label_dir: str, sample_limit: int = None):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    _section("1. DATA LOADING")
    print(f"Label directory : {label_dir}")
    df = load_labels(label_dir, sample_limit=sample_limit)
    if df.empty:
        print("ERROR: No label files found. Check label_dir path.")
        return
    summary(df)

    # ------------------------------------------------------------------ #
    # 2. KPI computation - Model A and Model B
    # ------------------------------------------------------------------ #
    _section("2. KPI COMPUTATION")
    kpis_a = compute_all_kpis(df, model="a")
    kpis_b = compute_all_kpis(df, model="b")
    print("KPIs computed for Model A and Model B.")

    # ------------------------------------------------------------------ #
    # 3. Trust Scores
    # ------------------------------------------------------------------ #
    _section("3. AI TRUST SCORES")
    ts_a = compute_trust_score(kpis_a)
    ts_b = compute_trust_score(kpis_b)

    print(format_scorecard("Model A (Higher Accuracy)", kpis_a, ts_a))
    print(format_scorecard("Model B (Lower Accuracy)",  kpis_b, ts_b))

    # ------------------------------------------------------------------ #
    # 4. RQ1 - KPI Scenario Sensitivity
    # ------------------------------------------------------------------ #
    _section("4. RQ1 - KPI SCENARIO SENSITIVITY")
    rq1_a = rq1_kpi_scenario_comparison(df, model="a")
    kw = rq1_a.get("confidence_kruskal", {})
    print(f"Kruskal-Wallis H = {kw.get('H_statistic', '?')}, "
          f"p = {kw.get('p_value', '?')}")
    print(kw.get("interpretation", ""))
    print("\nF1 by scenario (Model A):")
    for sc, f1 in sorted(rq1_a.get("f1_by_scenario", {}).items()):
        bar = "#" * int(f1 * 20)
        print(f"  {sc:<22} F1={f1:.3f}  {bar}")

    # ------------------------------------------------------------------ #
    # 5. RQ2 - Confidence Intervals (N=385 validation)
    # ------------------------------------------------------------------ #
    _section("5. RQ2 - SAMPLE SIZE & CONFIDENCE INTERVALS")
    ci = confidence_intervals(df, model="a")
    print(f"  N = {ci['n_samples']} samples")
    print(f"  Accuracy: {ci['accuracy']:.4f}  "
          f"95% CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Margin of error : {ci['margin_of_error']:.4f}")
    req = "YES" if ci["meets_n385_requirement"] else "NO"
    print(f"  Meets N>={ci['required_n']} requirement : {req}")

    # ------------------------------------------------------------------ #
    # 6. RQ3 - Discriminative Power
    # ------------------------------------------------------------------ #
    _section("6. RQ3 - DISCRIMINATIVE POWER")
    # Build multiple model variants by sub-sampling to simulate more models
    rng = np.random.default_rng(99)
    multi_models = []
    for i, (name, model_col) in enumerate([
        ("Model_A_full", "a"), ("Model_B_full", "b"),
        ("Model_A_day",  "a"), ("Model_B_day",  "b"),
        ("Model_A_night","a"), ("Model_B_night","b"),
    ]):
        sub = df.sample(min(500, len(df)), random_state=int(rng.integers(1, 1000)))
        kpis_sub = compute_all_kpis(sub, model=model_col)
        ts_sub   = compute_trust_score(kpis_sub)
        multi_models.append({
            "model_name":         name,
            "trust_score":        ts_sub["trust_score"],
            "map_proxy":          kpis_sub["performance"]["map_proxy"],
            "edge_failure_count": _edge_failure_count(sub, model_col),
        })

    rq3 = rq3_discriminative_power(multi_models)
    print(f"  Spearman (Trust Score vs failures) : "
          f"rho={rq3['spearman_trust_vs_failures']['rho']}, "
          f"p={rq3['spearman_trust_vs_failures']['p_value']}")
    print(f"  Spearman (mAP vs failures)         : "
          f"rho={rq3['spearman_map_vs_failures']['rho']}, "
          f"p={rq3['spearman_map_vs_failures']['p_value']}")
    print(f"\n  {rq3['verdict']}")

    # ------------------------------------------------------------------ #
    # 7. RQ4 - Dimension Influence
    # ------------------------------------------------------------------ #
    _section("7. RQ4 - DIMENSION INFLUENCE (OLS Regression)")
    records_a = _build_scenario_score_records(df, "a")
    records_b = _build_scenario_score_records(df, "b")
    all_records = records_a + records_b

    rq4 = rq4_dimension_influence(all_records)
    if "error" in rq4:
        print(f"  {rq4['error']}")
    else:
        print(f"  R^2 = {rq4['r_squared']}")
        print(f"  Ranked by influence:")
        for rank, dim in enumerate(rq4["ranked_by_influence"], 1):
            beta = rq4["standardised_betas"][dim]
            p    = rq4["p_values"][dim]
            bar  = "#" * int(abs(beta) * 20)
            sig  = "*" if p < 0.05 else " "
            print(f"    {rank}. {dim:<16} b={beta:+.4f}  p={p:.4f} {sig}  {bar}")
        print(f"\n  Top driver: {rq4.get('top_driver', '?')}")

    # ------------------------------------------------------------------ #
    # 8. Export outputs
    # ------------------------------------------------------------------ #
    _section("8. EXPORTING OUTPUTS")

    # Trust scores CSV
    ts_rows = []
    for name, kpis, ts in [
        ("Model_A", kpis_a, ts_a),
        ("Model_B", kpis_b, ts_b),
    ]:
        row = {"model": name, **ts["dimension_scores"],
               "trust_score": ts["trust_score"],
               "trust_score_100": ts["trust_score_100"],
               "trust_grade": ts["trust_grade"]}
        ts_rows.append(row)
    ts_df = pd.DataFrame(ts_rows)
    ts_path = OUTPUT_DIR / "trust_scores.csv"
    ts_df.to_csv(ts_path, index=False)
    print(f"  Trust scores  -> {ts_path}")

    # Detailed KPI CSV
    kpi_rows = []
    for name, kpis in [("Model_A", kpis_a), ("Model_B", kpis_b)]:
        kpi_rows.append({
            "model": name,
            # Performance
            "accuracy":    kpis["performance"]["accuracy"],
            "precision":   kpis["performance"]["precision"],
            "recall":      kpis["performance"]["recall"],
            "f1":          kpis["performance"]["f1"],
            "map_proxy":   kpis["performance"]["map_proxy"],
            # Fairness
            "disparate_impact_ratio":  kpis["fairness"]["disparate_impact_ratio"],
            "equalized_odds_gap":      kpis["fairness"]["equalized_odds_gap"],
            # Explainability
            "shap_consistency_score":  kpis["explainability"]["shap_consistency_score"],
            # Robustness
            "adversarial_accuracy_drop": kpis["robustness"]["adversarial_accuracy_drop"],
            "accuracy_clean":            kpis["robustness"]["accuracy_clean"],
            "accuracy_adversarial":      kpis["robustness"]["accuracy_adversarial"],
            # Privacy
            "entropy_normalised":  kpis["privacy"]["entropy_normalised"],
            "entropy_model":       kpis["privacy"]["entropy_model_a"],
            # Safety
            "sotif_pass_rate":       kpis["safety"]["sotif_pass_rate"],
            "edge_case_pass_rate":   kpis["safety"]["edge_case_pass_rate"],
        })
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_path = OUTPUT_DIR / "kpi_detail.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"  KPI details   -> {kpi_path}")

    # RQ results JSON
    rq_out = {
        "rq1_scenario_sensitivity": {
            k: v for k, v in rq1_a.items() if k != "per_scenario_mean_confidence"
        },
        "rq2_confidence_intervals": ci,
        "rq3_discriminative_power": rq3,
        "rq4_dimension_influence": rq4,
    }
    rq_path = OUTPUT_DIR / "rq_results.json"
    with open(rq_path, "w") as f:
        json.dump(rq_out, f, indent=2, default=str)
    print(f"  RQ results    -> {rq_path}")

    _section("EVALUATION COMPLETE")
    print(f"  Model A Trust Score : {ts_a['trust_score_100']}/100  Grade {ts_a['trust_grade']}")
    print(f"  Model B Trust Score : {ts_b['trust_score_100']}/100  Grade {ts_b['trust_grade']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Trust Score DMS - end-to-end evaluation runner"
    )
    parser.add_argument(
        "label_dir", nargs="?", default=DEFAULT_LABEL_DIR,
        help="Path to directory containing YOLO .txt label files"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit total records loaded (useful for quick tests)"
    )
    args = parser.parse_args()
    run(args.label_dir, sample_limit=args.sample)
