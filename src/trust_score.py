"""
Trust Score Engine — AI Trust Score Framework for DMS.

Implements RQ2:
  Unified Score = sum(w_i * M_i)
  where M_i = normalised KPI (0–1), w_i = AHP-derived weights.

AHP weights (from research proposal):
  Safety        : 0.40
  Robustness    : 0.25
  Fairness      : 0.15
  Explainability: 0.10
  Privacy       : 0.10

Normalisation maps each raw KPI value to [0, 1] using defined ideal/anti-ideal
bounds so heterogeneous metrics can be combined on a common scale.
"""

import numpy as np
from typing import Optional

# --------------------------------------------------------------------------- #
# AHP Weights
# --------------------------------------------------------------------------- #

AHP_WEIGHTS = {
    "safety":         0.40,
    "robustness":     0.25,
    "fairness":       0.15,
    "explainability": 0.10,
    "privacy":        0.10,
}

# Sanity check
assert abs(sum(AHP_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #

def _normalise(value: Optional[float], ideal: float, anti_ideal: float,
               higher_is_better: bool = True) -> float:
    """
    Min-max normalise a raw KPI value to [0, 1].
    higher_is_better=True  → ideal maps to 1.0, anti_ideal maps to 0.0
    higher_is_better=False → anti_ideal maps to 0.0, ideal maps to 1.0
    Returns 0.5 if value is None (neutral score for missing data).
    """
    if value is None:
        return 0.5
    if ideal == anti_ideal:
        return 0.5
    if higher_is_better:
        score = (value - anti_ideal) / (ideal - anti_ideal)
    else:
        score = (anti_ideal - value) / (anti_ideal - ideal)
    return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Per-dimension KPI extraction and normalisation
# --------------------------------------------------------------------------- #

def _normalise_performance(perf: dict) -> float:
    """Composite: 40 % F1, 30 % recall, 20 % accuracy, 10 % mAP."""
    f1       = _normalise(perf.get("f1"),        ideal=1.0, anti_ideal=0.0)
    recall   = _normalise(perf.get("recall"),    ideal=1.0, anti_ideal=0.0)
    accuracy = _normalise(perf.get("accuracy"),  ideal=1.0, anti_ideal=0.0)
    map_p    = _normalise(perf.get("map_proxy"), ideal=1.0, anti_ideal=0.0)
    return round(0.40 * f1 + 0.30 * recall + 0.20 * accuracy + 0.10 * map_p, 4)


def _normalise_fairness(fair: dict) -> float:
    """
    DIR = 1.0 → perfectly fair (score 1.0).
    DIR < 0.8 → concerning (score degrades rapidly below 0.8).
    DIR > 1.2 → reverse bias (also penalised — score from 1.2 down).
    """
    dir_val = fair.get("disparate_impact_ratio")
    if dir_val is None:
        return 0.5
    # Distance from 1.0, penalised symmetrically
    dist = abs(dir_val - 1.0)
    score = max(0.0, 1.0 - dist * 2.0)
    # Additional penalty for sub-0.8 threshold (80 % rule)
    if dir_val < 0.8:
        score = min(score, 0.60)
    return round(float(score), 4)


def _normalise_explainability(expl: dict) -> float:
    """SHAP consistency score already in [0, 1]."""
    return round(_normalise(
        expl.get("shap_consistency_score"), ideal=1.0, anti_ideal=0.0
    ), 4)


def _normalise_robustness(rob: dict) -> float:
    """
    AAD is a drop value (0 = best, 1 = worst).
    Score = 1 - AAD  (normalised so 0 drop → score 1.0).
    Also penalise if edge-case scenario accuracy falls below 0.6.
    """
    aad = rob.get("adversarial_accuracy_drop")
    aad_score = _normalise(aad, ideal=0.0, anti_ideal=0.50, higher_is_better=False)

    # Edge-case scenario penalty: average accuracy across non-normal scenarios
    sc_acc = rob.get("scenario_accuracy", {})
    edge_accs = [v for k, v in sc_acc.items() if k != "normal"]
    if edge_accs:
        mean_edge = np.mean(edge_accs)
        edge_score = _normalise(mean_edge, ideal=1.0, anti_ideal=0.40)
        return round(0.60 * aad_score + 0.40 * edge_score, 4)

    return round(aad_score, 4)


def _normalise_privacy(priv: dict) -> float:
    """Use pre-normalised entropy value (already 0–1)."""
    return round(_normalise(
        priv.get("entropy_normalised"), ideal=1.0, anti_ideal=0.0
    ), 4)


def _normalise_safety(saf: dict) -> float:
    """
    Composite: 60 % overall SOTIF pass rate + 40 % edge-case pass rate.
    Both already in [0, 1].
    """
    overall_pr = _normalise(saf.get("sotif_pass_rate"),       ideal=1.0, anti_ideal=0.0)
    edge_pr    = _normalise(saf.get("edge_case_pass_rate"),   ideal=1.0, anti_ideal=0.0)
    return round(0.60 * overall_pr + 0.40 * edge_pr, 4)


# --------------------------------------------------------------------------- #
# Trust Score
# --------------------------------------------------------------------------- #

def compute_trust_score(kpis: dict, weights: dict = None) -> dict:
    """
    Compute the composite AI Trust Score from a full KPI result dict
    (as returned by kpi_engine.compute_all_kpis).

    Parameters
    ----------
    kpis    : dict with keys 'performance', 'fairness', 'explainability',
              'robustness', 'privacy', 'safety'
    weights : optional override for AHP_WEIGHTS

    Returns
    -------
    dict with:
        dimension_scores  — normalised per-dimension score (0–1)
        weights           — weights used
        trust_score       — weighted composite (0–1)
        trust_score_100   — scaled to 0–100
        trust_grade       — letter grade
        interpretation    — plain-language summary
    """
    w = weights or AHP_WEIGHTS

    dim_scores = {
        "performance":     _normalise_performance(kpis.get("performance", {})),
        "fairness":        _normalise_fairness(kpis.get("fairness", {})),
        "explainability":  _normalise_explainability(kpis.get("explainability", {})),
        "robustness":      _normalise_robustness(kpis.get("robustness", {})),
        "privacy":         _normalise_privacy(kpis.get("privacy", {})),
        "safety":          _normalise_safety(kpis.get("safety", {})),
    }

    # Weighted sum — note: performance is not in the AHP weights (it's diagnostic)
    # Include performance as an informational score but not in the AHP composite
    ahp_dims = ["safety", "robustness", "fairness", "explainability", "privacy"]
    trust = sum(w[d] * dim_scores[d] for d in ahp_dims)
    trust = round(float(trust), 4)
    trust_100 = round(trust * 100, 1)

    grade = (
        "A" if trust >= 0.85 else
        "B" if trust >= 0.70 else
        "C" if trust >= 0.55 else
        "D" if trust >= 0.40 else
        "F"
    )

    interpretations = {
        "A": "High Trust - model meets or exceeds all safety and ethical thresholds.",
        "B": "Acceptable - model is deployable with monitoring; minor improvements recommended.",
        "C": "Marginal - significant gaps in one or more dimensions; review before deployment.",
        "D": "Poor - substantial ethical or safety failures; re-training or re-design required.",
        "F": "Unacceptable - model must not be deployed in safety-critical contexts.",
    }

    return {
        "dimension_scores": dim_scores,
        "weights": w,
        "trust_score": trust,
        "trust_score_100": trust_100,
        "trust_grade": grade,
        "interpretation": interpretations[grade],
    }


# --------------------------------------------------------------------------- #
# Scorecard table formatter
# --------------------------------------------------------------------------- #

def format_scorecard(model_name: str, kpis: dict, trust: dict) -> str:
    """Return a formatted scorecard string for printing / reporting."""
    ds = trust["dimension_scores"]
    w  = trust["weights"]
    perf = kpis.get("performance", {})
    fair = kpis.get("fairness", {})
    rob  = kpis.get("robustness", {})
    saf  = kpis.get("safety", {})
    priv = kpis.get("privacy", {})
    expl = kpis.get("explainability", {})

    lines = [
        f"\n{'='*62}",
        f"  AI TRUST SCORECARD - {model_name}",
        f"{'='*62}",
        f"  {'Dimension':<18} {'Raw KPI':<22} {'Norm Score':>10} {'Weight':>7}",
        f"  {'-'*58}",
        f"  {'Safety (SOTIF)':<18} {'PassRate={:.2f}'.format(saf.get('sotif_pass_rate','?')):<22} {ds['safety']:>10.3f} {w.get('safety',0):>7.2f}",
        f"  {'Robustness':<18} {'AAD={:.3f}'.format(rob.get('adversarial_accuracy_drop') or 0):<22} {ds['robustness']:>10.3f} {w.get('robustness',0):>7.2f}",
        f"  {'Fairness':<18} {'DIR={}'.format(fair.get('disparate_impact_ratio','?')):<22} {ds['fairness']:>10.3f} {w.get('fairness',0):>7.2f}",
        f"  {'Explainability':<18} {'SHAP_C={:.3f}'.format(expl.get('shap_consistency_score',0)):<22} {ds['explainability']:>10.3f} {w.get('explainability',0):>7.2f}",
        f"  {'Privacy':<18} {'H(X)_norm={:.3f}'.format(priv.get('entropy_normalised',0)):<22} {ds['privacy']:>10.3f} {w.get('privacy',0):>7.2f}",
        f"  {'-'*58}",
        f"  {'TRUST SCORE':<18} {'':<22} {trust['trust_score']:>10.3f} {'(AHP)':>7}",
        f"  {'TRUST SCORE /100':<18} {'':<22} {trust['trust_score_100']:>10.1f}",
        f"  {'GRADE':<18} {'':<22} {'  ' + trust['trust_grade']:>10}",
        f"{'='*62}",
        f"  Performance (info):",
        f"    Accuracy={perf.get('accuracy','?')}  Precision={perf.get('precision','?')}",
        f"    Recall={perf.get('recall','?')}  F1={perf.get('f1','?')}  mAP={perf.get('map_proxy','?')}",
        f"  {trust['interpretation']}",
        f"{'='*62}\n",
    ]
    return "\n".join(lines)
