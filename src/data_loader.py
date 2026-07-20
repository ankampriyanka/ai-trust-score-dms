"""
Data loader for AI Trust Score DMS framework.

Reads the same YOLO-format label files used in DMS-SafetyEstimateLimittime,
enriches each detection with:
  - scenario_type  (heuristic from class distribution + filename hash)
  - demographic_group (synthetic, reproducible from filename hash)
  - model_pred / model_pred_b (simulated second model for comparison)
  - is_adversarial flag

Produces a flat DataFrame with one row per detection, ready for KPI computation.

Class map (same as feature_pipeline.py):
  0 = neutral   1 = microsleep   2 = distraction   3 = phone_use
"""

import os
import glob
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASS_NAMES = {0: "neutral", 1: "microsleep", 2: "distraction", 3: "phone_use"}

# Map each class to a safety-critical label (positive = impaired/unsafe)
POSITIVE_CLASSES = {1, 2, 3}   # microsleep, distraction, phone_use

SCENARIOS = ["normal", "low_light", "occlusion", "glasses", "sunglasses",
             "extreme_head_pose", "motion_blur", "adversarial"]

DEMOGRAPHICS = ["skin_tone_light", "skin_tone_dark", "age_young",
                "age_middle", "age_older", "glasses_wearer"]

# Adversarial perturbation is applied to ~10 % of samples
ADVERSARIAL_RATE = 0.10

# Second model (Model B) has slightly lower clean accuracy to enable comparison
MODEL_B_NOISE = 0.12


def _stable_hash(s: str) -> int:
    """Deterministic integer hash of a string (for reproducible pseudo-random splits)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _assign_scenario(filename: str, class_id: int) -> str:
    h = _stable_hash(filename + "scenario")
    # microsleep / phone_use are more likely in adverse conditions
    weights = [0.35, 0.12, 0.10, 0.10, 0.08, 0.10, 0.08, 0.07]
    idx = h % 1000
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w * 1000
        if idx < cumulative:
            return SCENARIOS[i]
    return SCENARIOS[0]


def _assign_demographic(filename: str) -> str:
    h = _stable_hash(filename + "demo")
    return DEMOGRAPHICS[h % len(DEMOGRAPHICS)]


def _simulate_prediction(class_id: int, scenario: str, is_adversarial: bool,
                         noise_factor: float, rng: np.random.Generator) -> float:
    """
    Simulate model confidence for ground-truth class.
    Returns value in [0, 1] representing probability assigned to the true class.
    """
    base = 0.82 if class_id in POSITIVE_CLASSES else 0.88
    # Degrade in adverse scenarios
    penalty = {
        "normal": 0.0,
        "low_light": 0.12,
        "occlusion": 0.15,
        "glasses": 0.05,
        "sunglasses": 0.09,
        "extreme_head_pose": 0.13,
        "motion_blur": 0.10,
        "adversarial": 0.22,
    }.get(scenario, 0.0)
    if is_adversarial:
        penalty += 0.18
    conf = base - penalty + rng.normal(0, 0.04 + noise_factor)
    return float(np.clip(conf, 0.0, 1.0))


def load_labels(label_dir: str, sample_limit: int = None) -> pd.DataFrame:
    """
    Parse YOLO .txt label files, enrich with metadata, simulate model predictions.

    Parameters
    ----------
    label_dir : path to directory containing YOLO .txt files
    sample_limit : if set, cap total records (useful for quick tests)

    Returns
    -------
    DataFrame with columns:
        image_id, scenario_type, demographic_group, label_class, class_id,
        is_adversarial, bbox (list), confidence_gt,
        model_pred_a, model_pred_b, y_true, y_pred_a, y_pred_b
    """
    rng = np.random.default_rng(42)
    files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    # Exclude classes.txt
    files = [f for f in files if Path(f).name != "classes.txt"]

    records = []
    for fpath in files:
        stem = Path(fpath).stem
        with open(fpath) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            parts = line.split()
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            if cls not in CLASS_NAMES:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            scenario = _assign_scenario(stem, cls)
            demo = _assign_demographic(stem)
            is_adv = (_stable_hash(stem + "adv") % 100) < int(ADVERSARIAL_RATE * 100)
            if is_adv:
                scenario = "adversarial"

            conf_a = _simulate_prediction(cls, scenario, is_adv, 0.0, rng)
            conf_b = _simulate_prediction(cls, scenario, is_adv, MODEL_B_NOISE, rng)

            # Binary prediction: positive (impaired) if conf >= 0.5
            y_true = int(cls in POSITIVE_CLASSES)
            y_pred_a = int(conf_a >= 0.50)
            y_pred_b = int(conf_b >= 0.50)

            records.append({
                "image_id": stem,
                "scenario_type": scenario,
                "demographic_group": demo,
                "label_class": CLASS_NAMES[cls],
                "class_id": cls,
                "is_adversarial": is_adv,
                "bbox": [cx, cy, w, h],
                "bbox_cx": cx,
                "bbox_cy": cy,
                "bbox_w": w,
                "bbox_h": h,
                "confidence_gt": 1.0,
                "model_pred_a": conf_a,
                "model_pred_b": conf_b,
                "y_true": y_true,
                "y_pred_a": y_pred_a,
                "y_pred_b": y_pred_b,
            })

        if sample_limit and len(records) >= sample_limit:
            break

    df = pd.DataFrame(records)
    if sample_limit:
        df = df.head(sample_limit)
    return df.reset_index(drop=True)


def summary(df: pd.DataFrame) -> None:
    print(f"Total detections : {len(df)}")
    print(f"Unique images    : {df['image_id'].nunique()}")
    print(f"Classes          :\n{df['label_class'].value_counts().to_string()}")
    print(f"Scenarios        :\n{df['scenario_type'].value_counts().to_string()}")
    print(f"Demographics     :\n{df['demographic_group'].value_counts().to_string()}")
    print(f"Adversarial      : {df['is_adversarial'].sum()} ({df['is_adversarial'].mean():.1%})")
