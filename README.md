> **Repository status — legacy duplicate**
>
> This repository is retained for historical reference. The structured canonical doctoral research implementation is **[DBA-ai-trust-score-dms](https://github.com/ankampriyanka/DBA-ai-trust-score-dms)**. New development should happen there.
>
---
# AI Trust Score Framework — Driver Monitoring Systems

> A mathematically grounded, model-agnostic scorecard for evaluating the **trustworthiness** of Driver Monitoring System (DMS) computer vision models beyond accuracy alone.

---

## Table of Contents

1. [Business Problem](#1-business-problem)
2. [Solution Approach](#2-solution-approach)
3. [Architecture](#3-architecture)
4. [KPI Dimensions](#4-kpi-dimensions)
5. [Trust Score Formula](#5-trust-score-formula)
6. [Research Questions](#6-research-questions)
7. [Dataset](#7-dataset)
8. [Project Structure](#8-project-structure)
9. [Quick Start](#9-quick-start)
10. [Results](#10-results)
11. [Outputs](#11-outputs)
12. [Regulatory Alignment](#12-regulatory-alignment)

---

## 1. Business Problem

Modern vehicles equipped with Level 2 (L2) Advanced Driver Assistance Systems (ADAS) require the driver to remain attentive and in control at all times. Driver Monitoring Systems detect drowsiness, distraction, phone use, and other unsafe behaviours using in-cabin cameras.

**The problem is that current DMS evaluation relies almost entirely on accuracy or mAP** — metrics that tell you whether a model is correct on average, but not whether it is:

| Gap | Why it matters for safety |
|---|---|
| **Fair** across demographics | A model that works well for light-skinned drivers but fails for darker skin tones creates unequal safety protection |
| **Robust** under real-world conditions | Low-light, occlusion, sunglasses, and adversarial noise degrade detection — accuracy on a clean test set does not reveal this |
| **Explainable** | Regulators and OEMs need evidence that decisions are based on relevant facial features, not spurious correlations |
| **Privacy-preserving** | In-cabin cameras capture biometric data; a model whose internal representations leak identity poses GDPR and CCPA risk |
| **Safe under edge cases** | ISO 21448 (SOTIF) requires demonstrable performance in known hazardous scenarios, not just average performance |

Without a unified evaluation methodology, OEMs cannot objectively compare DMS models, cannot demonstrate regulatory compliance, and cannot prioritise which failure modes to fix.

---

## 2. Solution Approach

This framework transforms raw model predictions into a **composite AI Trust Score** — a single 0–100 number with an A–F grade — by:

1. **Defining quantitative KPI equations** for five trustworthy-AI dimensions derived from EU AI Act, NIST AI RMF, ISO/IEC 42001, and SAE J3016.
2. **Normalising heterogeneous metrics** onto a common 0–1 scale using min-max normalisation with domain-specific ideal/anti-ideal bounds.
3. **Aggregating via AHP-derived weights** that reflect the relative importance of each dimension in a safety-critical automotive context.
4. **Running statistical analyses** (Kruskal-Wallis, Spearman correlation, OLS regression) to answer four research questions about KPI sensitivity, discriminative power, and dimension influence.

The framework is **model-agnostic** — it consumes any model's output confidence scores and ground-truth labels. It operates exclusively at the **evaluation stage** of the AI lifecycle and produces reproducible, auditable results.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                     │
│                                                                         │
│   YOLO .txt label files          Ground-truth annotations               │
│   (class_id  cx  cy  w  h)       (neutral / microsleep /                │
│                                   distraction / phone_use)              │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       data_loader.py                                    │
│                                                                         │
│  • Parses YOLO labels into a flat DataFrame                             │
│  • Assigns scenario tags   (normal / low_light / occlusion /            │
│                              adversarial / glasses / sunglasses /       │
│                              extreme_head_pose / motion_blur)           │
│  • Assigns demographic groups (skin_tone_light / skin_tone_dark /       │
│                                 age_young / age_middle / age_older /    │
│                                 glasses_wearer)                         │
│  • Simulates Model A (higher accuracy) and Model B (lower accuracy)     │
│    prediction confidence scores with scenario-specific degradation      │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        kpi_engine.py                                    │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  Performance    │  │    Fairness       │  │   Explainability     │   │
│  │  Accuracy       │  │  Disparate Impact │  │  SHAP Consistency    │   │
│  │  Precision      │  │  Ratio (DIR)      │  │  Score               │   │
│  │  Recall / F1    │  │  TPR_A / TPR_B    │  │  1 - std/mean(SHAP)  │   │
│  │  mAP proxy      │  │                  │  │                      │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────────┘   │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  Robustness     │  │    Privacy        │  │   Safety (SOTIF)     │   │
│  │  Adversarial    │  │  Entropy H(X)     │  │  Scenario Pass Rate  │   │
│  │  Accuracy Drop  │  │  -sum p*log(p)    │  │  Passed / Total      │   │
│  │  Acc_clean -    │  │                  │  │  scenarios           │   │
│  │  Acc_adversarial│  │                  │  │                      │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       trust_score.py                                    │
│                                                                         │
│  Step 1 — Normalise each raw KPI to [0, 1]                             │
│           M_i = (value - anti_ideal) / (ideal - anti_ideal)            │
│                                                                         │
│  Step 2 — Apply AHP weights                                             │
│           Trust Score = Σ w_i × M_i                                    │
│                                                                         │
│           Safety        0.40  ████████████████                         │
│           Robustness    0.25  ██████████                                │
│           Fairness      0.15  ██████                                    │
│           Explainability 0.10  ████                                     │
│           Privacy       0.10  ████                                      │
│                                                                         │
│  Step 3 — Scale to 0-100 and assign grade A / B / C / D / F            │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       rq_analysis.py                                    │
│                                                                         │
│  RQ1  Kruskal-Wallis H-test — KPI sensitivity across scenarios          │
│  RQ2  Wilson score CIs — sample size adequacy (N >= 385)               │
│  RQ3  Spearman rho — Trust Score vs mAP discriminative power           │
│  RQ4  OLS regression — standardised beta coefficients per dimension     │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                    │
│                                                                         │
│   trust_scores.csv     Per-model dimension scores + composite grade     │
│   kpi_detail.csv       All raw KPI values side-by-side                  │
│   rq_results.json      Statistical test results for RQ1-RQ4             │
│   dashboard.html       Interactive 6-chart HTML dashboard               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. KPI Dimensions

| # | Dimension | KPI | Equation | Target |
|---|---|---|---|---|
| 1 | **Performance** | F1 Score | 2PR / (P+R) | Maximise |
| 1 | **Performance** | mAP proxy | Average Precision Score | Maximise |
| 2 | **Fairness** | Disparate Impact Ratio | TPR\_unprivileged / TPR\_privileged | Close to 1.0 (> 0.8 rule) |
| 3 | **Explainability** | SHAP Consistency Score | 1 - std(SHAP) / mean(SHAP) | Maximise (high = stable features) |
| 4 | **Robustness** | Adversarial Accuracy Drop | Acc\_clean - Acc\_adversarial | Minimise (0 = perfect robustness) |
| 5 | **Privacy** | Feature Entropy H(X) | -Σ p(x) log p(x) | Maximise (high = scrambled identity) |
| 6 | **Safety** | SOTIF Scenario Pass Rate | Scenarios\_passed / Total\_scenarios | Maximise (ISO 21448) |

### KPI explanations

**Fairness — Disparate Impact Ratio (DIR)**
Compares the True Positive Rate (correct detection of an unsafe driver state) between demographic groups. DIR = 1.0 means the model is equally accurate regardless of skin tone, age, or eyewear. DIR < 0.8 indicates potential bias under the 80% rule used in algorithmic auditing.

**Explainability — SHAP Consistency Score**
Measures whether the same input features (eye region, head position, hands on wheel) are consistently identified as the reason for a decision across similar images. A high score means the model relies on stable, interpretable signals rather than noise or background artefacts.

**Robustness — Adversarial Accuracy Drop (AAD)**
After applying controlled perturbations to input images (simulating sensor noise, light reflections, or intentional attacks), a small AAD indicates a robust model. A large drop signals brittleness — dangerous in production where lighting and sensor conditions are unpredictable.

**Privacy — Entropy H(X)**
Measured over the distribution of the model's internal confidence embeddings. High entropy means the internal representations are evenly spread and do not cluster by identity — making it mathematically difficult to reconstruct a specific driver's face from the model's internal state.

**Safety — SOTIF Scenario Pass Rate**
Unlike general accuracy, this metric strictly evaluates model performance on safety-critical scenario tags: night driving, face masks, occlusions, extreme head poses. A scenario passes if accuracy meets the minimum threshold, satisfying ISO 21448 requirements for known hazardous conditions.

---

## 5. Trust Score Formula

```
Trust Score = Σ (w_i × M_i)     where M_i ∈ [0, 1]

           = 0.40 × Safety
           + 0.25 × Robustness
           + 0.15 × Fairness
           + 0.10 × Explainability
           + 0.10 × Privacy

Scaled:  Trust Score /100 ∈ [0, 100]

Grade:   A  ≥ 85   High Trust — meets all safety and ethical thresholds
         B  ≥ 70   Acceptable — deployable with monitoring
         C  ≥ 55   Marginal — significant gaps; review before deployment
         D  ≥ 40   Poor — substantial failures; re-training required
         F  < 40   Unacceptable — must not be deployed
```

Weights are derived from the Analytic Hierarchy Process (AHP) reflecting the relative importance of each dimension in a safety-critical L2 ADAS context. Safety carries the highest weight (0.40) because SOTIF edge-case failures have the most direct link to road accidents.

---

## 6. Research Questions

| RQ | Question | Method | Status |
|---|---|---|---|
| **RQ1a** | Which KPI equations best quantify DMS trustworthiness? | Literature review + framework design | Addressed |
| **RQ1b** | How do KPIs compare empirically across edge-case scenarios? | Kruskal-Wallis H-test per scenario | H = 1027.9, p < 0.001 — KPIs differ significantly |
| **RQ2** | How can technical and ethical KPIs be integrated into one score? | AHP-weighted normalised composite | Trust Score formula above |
| **RQ3** | Does the Trust Score rank models more effectively than mAP alone? | Spearman correlation vs edge-failure count | rho\_trust = 0.77 > rho\_mAP = -0.54 |
| **RQ4** | Which dimensions contribute most to the overall score? | OLS regression with standardised betas | Safety β=0.84 > Fairness β=0.31 > Robustness β=0.18 |

---

## 7. Dataset

This prototype uses the **DMS-SafetyEstimateLimit** YOLO-format label dataset (same dataset as the companion SafetyEstimateLimit prototype). It contains annotated in-cabin driver images with four behaviour classes:

| Class ID | Label | Impairment Level |
|---|---|---|
| 0 | neutral | None |
| 1 | microsleep | Critical (0.95) |
| 2 | distraction | High (0.55) |
| 3 | phone_use | High (0.70) |

**Scenario tags** are assigned heuristically per image for edge-case evaluation:

`normal` · `low_light` · `occlusion` · `glasses` · `sunglasses` · `extreme_head_pose` · `motion_blur` · `adversarial`

**Demographic groups** are assigned deterministically per image for fairness evaluation:

`skin_tone_light` · `skin_tone_dark` · `age_young` · `age_middle` · `age_older` · `glasses_wearer`

> **N = 1,361 detections** across 1,232 unique images — exceeds the minimum N = 385 required at 95% confidence, 5% margin of error (Wilson score CI: margin = 2.6 pp).

---

## 8. Project Structure

```
ai-trust-score-dms/
│
├── src/
│   ├── data_loader.py       Label ingestion, scenario/demographic enrichment,
│   │                        dual-model prediction simulation
│   ├── kpi_engine.py        All KPI equations across 6 dimensions
│   ├── trust_score.py       Normalisation, AHP weighting, grade assignment,
│   │                        scorecard formatter
│   ├── rq_analysis.py       Statistical analyses for RQ1-RQ4
│   └── run_evaluation.py    End-to-end CLI runner
│
├── outputs/
│   ├── trust_scores.csv     Composite scores + grades per model
│   ├── kpi_detail.csv       All raw KPI values side-by-side
│   ├── rq_results.json      Statistical test outputs (RQ1-RQ4)
│   └── dashboard.html       Interactive visualisation dashboard
│
├── requirements.txt
└── README.md
```

---

## 9. Quick Start

### Prerequisites

```bash
python >= 3.10
pip install -r requirements.txt
```

Dependencies: `numpy` · `pandas` · `scipy` · `scikit-learn`

### Run the evaluation

```bash
# Against the default label directory (DMS-SafetyEstimateLimit dataset)
python src/run_evaluation.py

# Against a custom label directory
python src/run_evaluation.py /path/to/yolo/labels

# Quick test with a sample of 600 records
python src/run_evaluation.py --sample 600
```

### View the dashboard

Open `outputs/dashboard.html` in any browser. No server required.

The dashboard includes:
- **Stat tiles** — Trust Score, grade, sample size, margin of error, top driver
- **Spider chart** — all 5 AHP-weighted dimensions for both models
- **Dimension comparison bar chart** — side-by-side normalised scores
- **F1 by scenario** — edge-case degradation with severity colour coding
- **Raw performance KPIs** — accuracy / precision / recall / F1 / mAP
- **Beta coefficients** — RQ4 OLS dimension influence ranking
- **Robustness chart** — clean vs adversarial accuracy gap

Dark mode toggle is available in the top-right corner. All chart marks show a tooltip on hover.

---

## 10. Results

Results from the full dataset run (N = 1,361):

### Trust Scorecard

| | Model A | Model B |
|---|---|---|
| **Trust Score /100** | **39.5** | **40.2** |
| **Grade** | **F** | **D** |
| Safety (SOTIF Pass Rate) | 0.132 | 0.000 |
| Robustness (1 - AAD norm) | 0.293 | 0.460 |
| Fairness (DIR) | 0.869 | 0.989 |
| Explainability (SHAP-C) | 0.650 | 0.650 |
| Privacy (H norm) | 0.738 | 0.738 |

### Performance Metrics (informational)

| | Model A | Model B |
|---|---|---|
| Accuracy | 0.613 | 0.587 |
| Precision | 0.643 | 0.635 |
| Recall | 0.907 | 0.858 |
| F1 | 0.753 | 0.730 |
| mAP | 0.536 | 0.615 |

> Model A has higher accuracy and F1, but scores **lower** than Model B on the composite Trust Score because its SOTIF Safety pass rate (0.125) is worse. This demonstrates the framework's key value: accuracy alone does not capture deployment readiness.

### RQ4 — Dimension Influence

| Rank | Dimension | Std. Beta | p-value |
|---|---|---|---|
| 1 | Safety | 0.838 | < 0.001 |
| 2 | Fairness | 0.314 | < 0.001 |
| 3 | Robustness | 0.177 | < 0.001 |
| 4 | Privacy | 0.061 | < 0.001 |
| 5 | Explainability | 0.013 | < 0.001 |

---

## 11. Outputs

| File | Description |
|---|---|
| `outputs/trust_scores.csv` | Per-model dimension scores, composite Trust Score, and grade |
| `outputs/kpi_detail.csv` | All raw KPI values (accuracy, DIR, AAD, entropy, SOTIF pass rate, etc.) |
| `outputs/rq_results.json` | Full statistical output for RQ1–RQ4 including p-values, rho, and betas |
| `outputs/dashboard.html` | Self-contained interactive HTML dashboard with 6 charts |

---

## 12. Regulatory Alignment

| Standard | Addressed dimension | How this framework covers it |
|---|---|---|
| **EU AI Act** (high-risk AI systems) | Transparency, fairness, human oversight | Explainability (SHAP-C) and Fairness (DIR) KPIs; scorecard provides auditable evidence |
| **ISO 21448 SOTIF** | Safety under edge cases | SOTIF Scenario Pass Rate KPI evaluated across 8 edge-case scenario tags |
| **ISO/IEC 42001** | AI management system | Reproducible, documented evaluation methodology with versioned outputs |
| **NIST AI RMF** | Trustworthy AI properties | All five trustworthiness pillars (Safe, Explainable, Fair, Privacy-preserving, Robust) mapped to KPIs |
| **SAE J3016 L2 ADAS** | Driver monitoring at Level 2 automation | Framework scoped specifically to in-cabin camera DMS at L2 |

---

## Acknowledgements

KPI formulations for Fairness (Disparate Impact), Robustness (Adversarial Accuracy Drop), and Explainability (SHAP Consistency) are adapted from established open-source evaluation frameworks in algorithmic auditing (Barocas et al., 2019; Molnar, 2022; NIST, 2023). Mathematical refinements developed with AI assistance (Gemini; Zambrano et al., 2026).

---

*Part of the Walsh DBA — AI/ML Prototypes research series.*
