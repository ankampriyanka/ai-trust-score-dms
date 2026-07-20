"""
RQ Analysis - AI Trust Score Framework for DMS.

RQ3: Scorecard Discriminative Power
    Hypothesis: AI Trust Score ranks models more consistently with safety
    outcomes than mAP alone.
    Test: Spearman correlation between {Trust Score, mAP} and real-world
    incident proxy (edge-case failure count).

RQ4: KPI Dimension Influence
    OLS regression: Trust Score ~ Safety + Robustness + Fairness +
                                  Explainability + Privacy
    Standardised beta coefficients indicate relative influence.
    Also ANOVA-style comparison of KPI values across edge-case scenarios.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# --------------------------------------------------------------------------- #
# RQ3 - Discriminative Power
# --------------------------------------------------------------------------- #

def rq3_discriminative_power(model_results: list[dict]) -> dict:
    """
    Compare how well Trust Score vs mAP correlates with safety incidents.

    Parameters
    ----------
    model_results : list of dicts, each containing:
        {
          'model_name': str,
          'trust_score': float,          # from trust_score.compute_trust_score
          'map_proxy': float,            # from kpi_engine performance
          'edge_failure_count': int,     # proxy for real-world incidents
        }

    Returns
    -------
    dict with Spearman correlations and comparison verdict.
    """
    if len(model_results) < 3:
        return {
            "error": "Need at least 3 models for correlation analysis.",
            "n_models": len(model_results),
        }

    trust_scores    = np.array([r["trust_score"] for r in model_results])
    map_scores      = np.array([r["map_proxy"] for r in model_results])
    failure_counts  = np.array([r["edge_failure_count"] for r in model_results])

    rho_trust, p_trust = scipy_stats.spearmanr(trust_scores, -failure_counts)
    rho_map,   p_map   = scipy_stats.spearmanr(map_scores,   -failure_counts)

    trust_wins = abs(rho_trust) > abs(rho_map)

    return {
        "n_models": len(model_results),
        "spearman_trust_vs_failures": {
            "rho": round(float(rho_trust), 4),
            "p_value": round(float(p_trust), 4),
        },
        "spearman_map_vs_failures": {
            "rho": round(float(rho_map), 4),
            "p_value": round(float(p_map), 4),
        },
        "trust_score_more_discriminative": bool(trust_wins),
        "verdict": (
            "SUPPORTED - Trust Score correlates more strongly with safety outcomes than mAP alone."
            if trust_wins else
            "NOT SUPPORTED - mAP shows equal or stronger correlation with safety outcomes."
        ),
    }


# --------------------------------------------------------------------------- #
# RQ4 - Dimension Influence (OLS)
# --------------------------------------------------------------------------- #

def rq4_dimension_influence(score_records: list[dict]) -> dict:
    """
    OLS regression: Trust Score ~ Safety + Robustness + Fairness +
                                  Explainability + Privacy

    Standardised beta coefficients rank relative influence.

    Parameters
    ----------
    score_records : list of dicts, each with dimension scores and trust score,
        e.g. as produced per-scenario or per-image-batch.

    Returns
    -------
    dict with standardised betas, R², p-values.
    """
    dims = ["safety", "robustness", "fairness", "explainability", "privacy"]

    required = dims + ["trust_score"]
    records = [r for r in score_records if all(k in r for k in required)]

    if len(records) < 10:
        return {"error": f"Need at least 10 records; got {len(records)}."}

    df = pd.DataFrame(records)
    X = df[dims].values
    y = df["trust_score"].values

    # Standardise
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    y_std = (y - y.mean()) / (y.std() + 1e-9)

    # OLS via numpy (avoids statsmodels dependency)
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X_aug, y_std, rcond=None)
    beta0 = coeffs[0]
    betas = coeffs[1:]

    # R-squared
    y_hat = X_aug @ coeffs
    ss_res = np.sum((y_std - y_hat) ** 2)
    ss_tot = np.sum((y_std - y_std.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Approximate p-values via t-test on standardised coefficients
    n, k = X_aug.shape
    mse = ss_res / max(n - k, 1)
    var_cov = mse * np.linalg.pinv(X_aug.T @ X_aug)
    se = np.sqrt(np.diag(var_cov))
    t_stats = coeffs / (se + 1e-12)
    p_vals  = [2 * float(scipy_stats.t.sf(abs(t), df=max(n - k, 1))) for t in t_stats]

    # Rank by absolute beta
    ranked = sorted(
        zip(dims, betas.tolist(), p_vals[1:]),
        key=lambda x: abs(x[1]), reverse=True
    )

    return {
        "n_records": len(records),
        "r_squared": round(r2, 4),
        "standardised_betas": {d: round(b, 4) for d, b, _ in ranked},
        "p_values": {d: round(p, 4) for d, _, p in ranked},
        "ranked_by_influence": [d for d, _, _ in ranked],
        "top_driver": ranked[0][0] if ranked else None,
    }


# --------------------------------------------------------------------------- #
# RQ1b - KPI comparison across scenarios (ANOVA / Kruskal-Wallis)
# --------------------------------------------------------------------------- #

def rq1_kpi_scenario_comparison(df: pd.DataFrame, model: str = "a") -> dict:
    """
    Compare prediction quality (confidence) across scenarios using
    Kruskal-Wallis H-test (nonparametric ANOVA alternative).

    A significant H-test (p < 0.05) means KPI values differ meaningfully
    across edge-case scenarios - i.e., the KPI is sensitive to scenario type.
    """
    results = {}
    scenarios = df["scenario_type"].unique()

    col = f"model_pred_{model}"
    groups = [df.loc[df["scenario_type"] == sc, col].values for sc in scenarios
              if (df["scenario_type"] == sc).sum() >= 5]

    if len(groups) >= 2:
        h_stat, p_val = scipy_stats.kruskal(*groups)
        results["confidence_kruskal"] = {
            "H_statistic": round(float(h_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "interpretation": (
                "KPI values differ significantly across scenarios (p<0.05) - "
                "this KPI is scenario-sensitive and captures edge-case degradation."
                if p_val < 0.05 else
                "No significant difference across scenarios."
            ),
        }

    # Per-scenario mean confidence
    results["per_scenario_mean_confidence"] = (
        df.groupby("scenario_type")[col]
        .agg(["mean", "std", "count"])
        .round(4)
        .rename(columns={"mean": "mean_conf", "std": "std_conf", "count": "n"})
        .to_dict("index")
    )

    # Per-scenario F1 (binary)
    y_true_col = "y_true"
    y_pred_col = f"y_pred_{model}"
    f1_by_sc = {}
    for sc in scenarios:
        mask = df["scenario_type"] == sc
        if mask.sum() < 5:
            continue
        yt = df.loc[mask, y_true_col].values
        yp = df.loc[mask, y_pred_col].values
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_by_sc[sc] = round(f1, 4)

    results["f1_by_scenario"] = f1_by_sc

    return results


# --------------------------------------------------------------------------- #
# Confidence intervals for KPI point estimates (RQ1 sample size validation)
# --------------------------------------------------------------------------- #

def confidence_intervals(df: pd.DataFrame, model: str = "a",
                         confidence: float = 0.95) -> dict:
    """
    Compute 95 % Wilson score confidence intervals for accuracy and pass-rate.
    Validates that N=385 gives margin of error ≤ 0.05.
    """
    n = len(df)
    y_true = df["y_true"].values
    y_pred = df[f"y_pred_{model}"].values
    correct = (y_true == y_pred)

    p_hat = correct.mean()
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)

    # Wilson interval
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    lower = float(centre - margin)
    upper = float(centre + margin)

    # Required N for e=0.05, p=0.5, z=1.96
    required_n = int(np.ceil((1.96**2 * 0.5 * 0.5) / (0.05**2)))

    return {
        "n_samples": n,
        "accuracy": round(p_hat, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "margin_of_error": round(float(margin), 4),
        "meets_n385_requirement": bool(n >= required_n),
        "required_n": required_n,
        "confidence_level": confidence,
    }
