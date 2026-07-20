"""
KPI Engine — AI Trust Score Framework for DMS.

Implements all KPI equations from the research proposal across six dimensions:

  Dimension 1 — Performance  : Accuracy, Precision, Recall, F1, mAP-proxy
  Dimension 2 — Fairness     : Disparate Impact Ratio (TPR_A / TPR_B)
  Dimension 3 — Explainability: SHAP Consistency Score
  Dimension 4 — Robustness   : Adversarial Accuracy Drop (AAD)
  Dimension 5 — Privacy      : Feature-embedding entropy H(X)
  Dimension 6 — Safety (SOTIF): Scenario Pass Rate

Each public function accepts a DataFrame (output of data_loader.load_labels)
and a model column suffix ('a' or 'b') and returns a named dict of KPI values.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _safe_div(num, den, default=0.0):
    return num / den if den > 0 else default


# --------------------------------------------------------------------------- #
# Dimension 1 — Performance
# --------------------------------------------------------------------------- #

def compute_performance(df: pd.DataFrame, model: str = "a") -> dict:
    """
    Returns: accuracy, precision, recall, f1, map_proxy

    map_proxy approximates mAP using average_precision_score on confidence values.
    For multi-class, macro-average across classes with at least two labels.
    """
    y_true = df["y_true"].values
    y_pred = df[f"y_pred_{model}"].values
    y_score = df[f"model_pred_{model}"].values

    tp, tn, fp, fn = _confusion(y_true, y_pred)

    accuracy  = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    # mAP proxy: average precision for the positive class
    try:
        map_proxy = float(average_precision_score(y_true, y_score))
    except Exception:
        map_proxy = float(precision)   # fallback if only one class present

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map_proxy": round(map_proxy, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# --------------------------------------------------------------------------- #
# Dimension 2 — Fairness (Disparate Impact Ratio)
# --------------------------------------------------------------------------- #

def compute_fairness(df: pd.DataFrame, model: str = "a",
                     privileged: str = "skin_tone_light",
                     unprivileged: str = "skin_tone_dark") -> dict:
    """
    Disparate Impact Ratio = TPR_unprivileged / TPR_privileged

    Score close to 1.0 → fair. Below 0.8 → potential bias (80 % rule).

    Also reports equalized odds gap (FPR difference) as a secondary metric.
    """
    y_true = df["y_true"].values
    y_pred = df[f"y_pred_{model}"].values
    group  = df["demographic_group"].values

    def tpr_for_group(g):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        return _safe_div(tp, tp + fn, default=None)

    def fpr_for_group(g):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        return _safe_div(fp, fp + tn, default=None)

    tpr_priv   = tpr_for_group(privileged)
    tpr_unpriv = tpr_for_group(unprivileged)
    fpr_priv   = fpr_for_group(privileged)
    fpr_unpriv = fpr_for_group(unprivileged)

    if tpr_priv is None or tpr_unpriv is None or tpr_priv == 0:
        dir_score = None
    else:
        dir_score = round(tpr_unpriv / tpr_priv, 4)

    eod_gap = None
    if fpr_priv is not None and fpr_unpriv is not None:
        eod_gap = round(abs(fpr_unpriv - fpr_priv), 4)

    return {
        "disparate_impact_ratio": dir_score,
        "tpr_privileged": round(tpr_priv, 4) if tpr_priv is not None else None,
        "tpr_unprivileged": round(tpr_unpriv, 4) if tpr_unpriv is not None else None,
        "equalized_odds_gap": eod_gap,
        "privileged_group": privileged,
        "unprivileged_group": unprivileged,
    }


# --------------------------------------------------------------------------- #
# Dimension 3 — Explainability (SHAP Consistency Score)
# --------------------------------------------------------------------------- #

def compute_explainability(df: pd.DataFrame, model: str = "a") -> dict:
    """
    SHAP Consistency Score = 1 - std(SHAP_proxy) / mean(SHAP_proxy)

    True SHAP requires the actual model. Here we approximate SHAP importance
    with a proxy: the magnitude of bbox_w * bbox_h (object size → attention area)
    normalised per class, treated as the dominant feature attribution value.
    This is consistent with the paper's intention: high consistency means the
    model relies on stable features rather than noise.

    For the prototype we compute per-class feature-attribution variability using
    the bounding-box features (cx, cy, w, h) as a stand-in for SHAP values.
    """
    feature_cols = ["bbox_cx", "bbox_cy", "bbox_w", "bbox_h"]
    # Proxy SHAP: absolute contribution of each bbox coordinate to detection area
    shap_proxy = df[feature_cols].abs().values  # shape (N, 4)

    # SHAP consistency per feature dimension
    col_means = shap_proxy.mean(axis=0)
    col_stds  = shap_proxy.std(axis=0)

    # Coefficient of variation per feature; consistency = 1 - CV
    cvs = np.where(col_means > 0, col_stds / col_means, 1.0)
    shap_consistency = float(np.clip(1.0 - cvs.mean(), 0.0, 1.0))

    # Per-class consistency (higher variance between classes = less consistent)
    class_means = df.groupby("class_id")[feature_cols].mean()
    inter_class_cv = (class_means.std() / class_means.mean().replace(0, np.nan)).mean()
    inter_class_consistency = float(np.clip(1.0 - inter_class_cv, 0.0, 1.0))

    return {
        "shap_consistency_score": round(shap_consistency, 4),
        "inter_class_consistency": round(inter_class_consistency, 4) if not np.isnan(inter_class_consistency) else None,
        "feature_cv_mean": round(float(cvs.mean()), 4),
    }


# --------------------------------------------------------------------------- #
# Dimension 4 — Robustness (Adversarial Accuracy Drop)
# --------------------------------------------------------------------------- #

def compute_robustness(df: pd.DataFrame, model: str = "a") -> dict:
    """
    Adversarial Accuracy Drop (AAD) = Acc_clean - Acc_adversarial

    Lower AAD → more robust. Also reports per-scenario accuracy.
    """
    y_true = df["y_true"].values
    y_pred = df[f"y_pred_{model}"].values
    is_adv = df["is_adversarial"].values

    acc_clean = accuracy_score(y_true[~is_adv], y_pred[~is_adv]) if (~is_adv).sum() > 0 else None
    acc_adv   = accuracy_score(y_true[is_adv],  y_pred[is_adv])  if is_adv.sum() > 0  else None

    aad = None
    if acc_clean is not None and acc_adv is not None:
        aad = round(acc_clean - acc_adv, 4)

    # Per-scenario accuracy
    scenario_acc = {}
    for sc in df["scenario_type"].unique():
        mask = df["scenario_type"] == sc
        if mask.sum() >= 5:
            scenario_acc[sc] = round(accuracy_score(y_true[mask], y_pred[mask]), 4)

    return {
        "adversarial_accuracy_drop": aad,
        "accuracy_clean": round(acc_clean, 4) if acc_clean is not None else None,
        "accuracy_adversarial": round(acc_adv, 4) if acc_adv is not None else None,
        "scenario_accuracy": scenario_acc,
    }


# --------------------------------------------------------------------------- #
# Dimension 5 — Privacy (Feature Entropy)
# --------------------------------------------------------------------------- #

def compute_privacy(df: pd.DataFrame) -> dict:
    """
    H(X) = -sum p(x_i) * log p(x_i)

    Computed over the distribution of predicted confidence values (model_pred_a),
    treating them as the latent face-embedding projection. Higher entropy →
    more anonymised (identity-revealing structure is 'scrambled').

    We discretise the confidence distribution into bins and compute entropy.
    Also compute entropy over bbox spatial positions as a proxy for face-location
    leakage: high spatial entropy → less location-leakage.
    """
    def _entropy_from_array(arr: np.ndarray, bins: int = 20) -> float:
        counts, _ = np.histogram(arr, bins=bins, range=(0, 1))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(scipy_entropy(probs, base=2))

    h_conf_a = _entropy_from_array(df["model_pred_a"].values)
    h_conf_b = _entropy_from_array(df["model_pred_b"].values)
    h_spatial = _entropy_from_array(df["bbox_cx"].values)

    # Theoretical max entropy for 20 bins = log2(20) ≈ 4.32
    max_h = np.log2(20)
    normalised = float(np.clip(h_conf_a / max_h, 0, 1))

    return {
        "entropy_model_a": round(h_conf_a, 4),
        "entropy_model_b": round(h_conf_b, 4),
        "entropy_spatial": round(h_spatial, 4),
        "entropy_normalised": round(normalised, 4),
    }


# --------------------------------------------------------------------------- #
# Dimension 6 — Safety / SOTIF (Scenario Pass Rate)
# --------------------------------------------------------------------------- #

def compute_safety(df: pd.DataFrame, model: str = "a",
                   pass_threshold: float = 0.70) -> dict:
    """
    SOTIF Scenario Pass Rate = Scenarios_Passed / Total_Scenarios

    A scenario 'passes' if the model's accuracy on that scenario's samples
    meets or exceeds pass_threshold.

    Edge-case scenarios (non-normal) are evaluated separately for the
    safety-critical pass rate required by ISO 21448.
    """
    y_true = df["y_true"].values
    y_pred = df[f"y_pred_{model}"].values

    scenarios = df["scenario_type"].unique()
    passed = 0
    total  = 0
    detail = {}

    for sc in scenarios:
        mask = df["scenario_type"] == sc
        if mask.sum() < 5:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        ok  = acc >= pass_threshold
        detail[sc] = {"accuracy": round(acc, 4), "passed": ok, "n": int(mask.sum())}
        passed += int(ok)
        total  += 1

    overall_pass_rate = _safe_div(passed, total)

    # Edge-case-only pass rate (excluding 'normal')
    edge_passed = sum(1 for sc, v in detail.items() if sc != "normal" and v["passed"])
    edge_total  = sum(1 for sc in detail if sc != "normal")
    edge_pass_rate = _safe_div(edge_passed, edge_total)

    return {
        "sotif_pass_rate": round(overall_pass_rate, 4),
        "edge_case_pass_rate": round(edge_pass_rate, 4),
        "scenarios_passed": passed,
        "scenarios_total": total,
        "pass_threshold": pass_threshold,
        "scenario_detail": detail,
    }


# --------------------------------------------------------------------------- #
# Aggregate: compute all KPIs for one model
# --------------------------------------------------------------------------- #

def compute_all_kpis(df: pd.DataFrame, model: str = "a") -> dict:
    """Compute and return all six KPI dimension results for a given model."""
    return {
        "performance":     compute_performance(df, model),
        "fairness":        compute_fairness(df, model),
        "explainability":  compute_explainability(df, model),
        "robustness":      compute_robustness(df, model),
        "privacy":         compute_privacy(df),
        "safety":          compute_safety(df, model),
    }
