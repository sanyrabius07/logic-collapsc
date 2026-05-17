"""
run_lch.py — Logic Collapse Horizon
=====================================
Reproduces Figure 1: Logic Collapse Horizon (LCH) curve.
Plots LCI vs pruning ratio and marks the collapse threshold at LCI=0.85.

Usage:
    python experiments/run_lch.py
    python experiments/run_lch.py --dataset unsw --seed 42

Outputs:
    figures/fig1_lch_curve.pdf
    figures/fig1_lch_curve.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, warnings
warnings.filterwarnings("ignore")

from src.data   import load_dataset
from src.models import ResidualMLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="unsw")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def make_loader(X, y, bs=1024, shuffle=False):
    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y).long()),
        batch_size=bs, shuffle=shuffle
    )


def train_model(model, tr_ldr, va_ldr, epochs=30):
    from sklearn.metrics import accuracy_score
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc, best_st = 0., None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ldr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sch.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in va_ldr:
                ps.append(model(xb).argmax(1)); ys.append(yb)
        v = accuracy_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        if v > best_acc:
            best_acc = v
            best_st  = {k: v2.clone() for k, v2 in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model


def get_shap(model, X_np):
    model.eval()
    bg  = torch.tensor(X_np[:50],  dtype=torch.float32)
    inp = torch.tensor(X_np[:300], dtype=torch.float32)
    sv  = shap.GradientExplainer(model, bg).shap_values(inp)
    if isinstance(sv, list): sv = sv[1]
    sv  = np.array(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv.astype(np.float32)


def compute_lci(sv_t, sv_s, w=0.5, topk=10):
    n = min(len(sv_t), len(sv_s))
    sv_t, sv_s = sv_t[:n], sv_s[:n]
    corrs = []
    for i in range(n):
        try:
            c = float(pearsonr(sv_t[i].ravel(), sv_s[i].ravel())[0])
            if not np.isnan(c): corrs.append(c)
        except: pass
    corr    = float(np.mean(corrs)) if corrs else 0.
    top_t   = set(int(x) for x in np.argsort(np.abs(sv_t).mean(0))[-topk:])
    top_s   = set(int(x) for x in np.argsort(np.abs(sv_s).mean(0))[-topk:])
    jaccard = len(top_t & top_s) / len(top_t | top_s)
    return round(w * corr + (1 - w) * jaccard, 4)


def apply_pruning(model, ratio):
    import torch.nn.utils.prune as prune
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            prune.l1_unstructured(mod, name="weight", amount=ratio)
            prune.remove(mod, "weight")
    return m


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset(args.dataset)
    tr_ldr = make_loader(X_tr, y_tr, shuffle=True)
    va_ldr = make_loader(X_va, y_va)

    print("Training teacher ...")
    teacher  = train_model(ResidualMLP(X_tr.shape[1]), tr_ldr, va_ldr)
    sv_teach = get_shap(teacher, X_tr)

    ratios    = [0.0, 0.10, 0.30, 0.50, 0.70, 0.85, 0.95]
    lci_vals  = []
    for r in ratios:
        if r == 0.0:
            lci_vals.append(1.0)
        else:
            pruned = apply_pruning(teacher, r)
            sv_p   = get_shap(pruned, X_tr)
            lci_vals.append(compute_lci(sv_teach, sv_p))
        print(f"  Pruning {int(r*100):>3}%  →  LCI={lci_vals[-1]:.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r * 100 for r in ratios]
    ax.plot(xs, lci_vals, "o-", color="#1565C0", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label="ResidualMLP (Pruning)")
    ax.axhline(0.85, color="#D32F2F", linestyle="--", linewidth=2,
               label="Logic Collapse Threshold (LCI = 0.85)")
    ax.fill_between(xs, lci_vals, 0.85,
                    where=[l < 0.85 for l in lci_vals],
                    color="#FFCDD2", alpha=0.5, label="Logic Collapse Region")
    ax.set_xlabel("Compression Ratio (% weights pruned)", fontsize=12)
    ax.set_ylabel("Logic Collapse Index (LCI)",           fontsize=12)
    ax.set_title(f"Logic Collapse Horizon (LCH)\n"
                 f"{args.dataset.upper()} × ResidualMLP", fontsize=13)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig1_lch_curve.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("figures/fig1_lch_curve.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved → figures/fig1_lch_curve.pdf / .png")


if __name__ == "__main__":
    main()
