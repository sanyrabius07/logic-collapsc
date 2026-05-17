"""
Experiment 2 — SEPA: Silent Explanation Poisoning Attack
========================================================
Metric: MPRF — Minimum Perturbation for Rank Flip

For each model f and test instance x:
  Binary-search for the smallest ε such that the teacher's top-1 GI
  feature falls out of f's top-3 GI features under noise δ~N(0, εI).

  MPRF(f, T, x) = min{ ε : top1_T ∉ top3_f(x + εδ) }

Higher MPRF = model explanations are harder to silently corrupt.
LoRA-SHAP requires the largest perturbation → most robust.

Outputs:
  - results/exp2_mprf.csv
  - figures/fig_exp2_mprf.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from src.metrics import gi_eval


MAX_EPS   = 3.0
N_BSEARCH = 18
N_DRAWS   = 5


def mprf_model(model, X, top1_per_instance, max_eps=MAX_EPS,
               n_bsearch=N_BSEARCH, n_draws=N_DRAWS):
    """
    Compute per-instance MPRF and return summary statistics.

    Returns dict with keys: MPRF_mean, MPRF_median, MPRF_pct_robust
    """
    model.eval()
    flip_eps = []

    for i in range(len(X)):
        xi   = X[i:i+1]
        t1   = int(top1_per_instance[i])
        lo, hi, found = 0.0, max_eps, max_eps

        for _ in range(n_bsearch):
            mid     = (lo + hi) / 2.0
            flipped = 0
            for _ in range(n_draws):
                Xn   = (xi + torch.randn_like(xi) * mid).detach()
                top3 = set(gi_eval(model, Xn).abs()[0].topk(3).indices.tolist())
                if t1 not in top3:
                    flipped += 1
            if flipped >= n_draws // 2 + 1:
                found = mid
                hi    = mid
            else:
                lo    = mid

        flip_eps.append(found)

    arr = np.array(flip_eps)
    return {
        "MPRF_mean":       round(float(arr.mean()),                4),
        "MPRF_median":     round(float(np.median(arr)),            4),
        "MPRF_pct_robust": round(float((arr >= max_eps*0.95).mean()), 4),
    }


def run_exp2(models, teacher, Xq, lci_map, save_dir="results"):
    """
    Run MPRF experiment for all models and compute correlation with LCI.

    Returns:
        mprf_df : DataFrame with MPRF statistics per model
        r, p    : Pearson correlation between MPRF and LCI
    """
    print("=" * 60)
    print("EXP 2 — SEPA / MPRF ATTACK")
    print("=" * 60)

    X_mprf   = Xq[:80]
    gi_teach = gi_eval(teacher, X_mprf).abs()
    top1     = gi_teach.argmax(dim=1).numpy()

    rows  = []
    order = list(models.keys())
    print()
    for name in order:
        r = mprf_model(models[name], X_mprf, top1)
        rows.append({"Method": name, **r, "LCI": lci_map[name]})
        print(f"  {name:14s}  MPRF_mean={r['MPRF_mean']:.3f}  "
              f"MPRF_median={r['MPRF_median']:.3f}")

    df     = pd.DataFrame(rows)
    r_val, p_val = pearsonr(df["MPRF_mean"].tolist(), df["LCI"].tolist())
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "exp2_mprf.csv"), index=False)

    print(f"\n  Pearson r(MPRF, LCI) = {r_val:.4f}   p = {p_val:.4f}")
    print(f"  Saved {save_dir}/exp2_mprf.csv")
    return df, r_val, p_val
