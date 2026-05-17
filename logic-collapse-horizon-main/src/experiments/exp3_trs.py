"""
Experiment 3 — ESC: Explanation Stability Certification
=======================================================
Metric: TRS — Teacher-Relative Stability

  TRS_ε(f, T) = E_{x, δ~N(0,I)} [ corr(GI_T(x+εδ), GI_f(x+εδ)) ]

Measures: does model f track the teacher's feature attributions
as inputs drift? Models trained with SHAP guidance (LoRA-SHAP)
explicitly optimize for this and should maintain high TRS.

Compliance thresholds (EU AI Act Article 13 framing):
  TRS ≥ 0.90  →  GREEN  (certified deployment)
  TRS ≥ 0.85  →  AMBER  (conditional approval)
  TRS  < 0.85  →  RED    (non-compliant)

Outputs:
  - results/exp3_trs.csv
  - figures/fig_exp3_trs.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from src.metrics import gi_eval


def compute_trs(model, teacher, X, eps_list, n_noise=15):
    """
    Compute TRS at each noise level in eps_list.

    At ε=0: correlation on clean inputs (no noise).
    At ε>0: average across n_noise random noise draws.

    Returns:
        dict { eps: TRS_score }
    """
    results = {}
    for eps in eps_list:
        if eps == 0.0:
            gi_t = gi_eval(teacher, X).numpy()
            gi_s = gi_eval(model,   X).numpy()
            corrs = [
                float(np.corrcoef(gi_t[:, j], gi_s[:, j])[0, 1])
                for j in range(gi_t.shape[1])
                if gi_t[:, j].std() > 1e-8 and gi_s[:, j].std() > 1e-8
            ]
            results[0.0] = round(float(np.mean(corrs)) if corrs else 0.0, 4)
        else:
            all_corrs = []
            for _ in range(n_noise):
                Xn   = (X + torch.randn_like(X) * eps).detach()
                gi_t = gi_eval(teacher, Xn).numpy()
                gi_s = gi_eval(model,   Xn).numpy()
                for j in range(gi_t.shape[1]):
                    if gi_t[:, j].std() > 1e-8 and gi_s[:, j].std() > 1e-8:
                        all_corrs.append(
                            float(np.corrcoef(gi_t[:, j], gi_s[:, j])[0, 1])
                        )
            results[eps] = round(float(np.mean(all_corrs)) if all_corrs else 0.0, 4)
    return results


def run_exp3(models, teacher, Xq,
             eps_list=None, n_noise=15, save_dir="results"):
    """
    Run TRS experiment for all models.

    Returns:
        trs_data : dict {name: {eps: score}}
        trs_df   : long-format DataFrame
    """
    if eps_list is None:
        eps_list = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]

    print("=" * 60)
    print("EXP 3 — ESC / TRS CERTIFICATION")
    print("=" * 60)

    X_trs    = Xq[:60]
    trs_data = {}
    rows     = []
    print()

    for name, model in models.items():
        scores = compute_trs(model, teacher, X_trs, eps_list, n_noise)
        trs_data[name] = scores
        line = "  ".join([f"ε={e:.2f}→{scores[e]:.3f}" for e in eps_list])
        print(f"  {name:14s}  {line}")
        for ep, v in scores.items():
            rows.append({"Method": name, "eps": ep, "TRS": v})

    trs_df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    trs_df.to_csv(os.path.join(save_dir, "exp3_trs.csv"), index=False)
    print(f"\n  Saved {save_dir}/exp3_trs.csv")
    return trs_data, trs_df
