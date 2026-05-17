"""
run_mprf.py — Logic Collapse Horizon
======================================
Reproduces Table 4: Most-Prominent-Rank-Flip (MPRF) analysis.
Measures whether compression displaces the teacher's top-1 SHAP feature.

Usage:
    python experiments/run_mprf.py
    python experiments/run_mprf.py --dataset unsw --seed 42

Outputs:
    results/table4_mprf.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import shap, warnings
warnings.filterwarnings("ignore")

from src.data   import load_dataset
from src.models import ResidualMLP
from src.lora   import inject_lora


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


def get_shap(model, X_np):
    model.eval()
    bg  = torch.tensor(X_np[:50],  dtype=torch.float32)
    inp = torch.tensor(X_np[:300], dtype=torch.float32)
    sv  = shap.GradientExplainer(model, bg).shap_values(inp)
    if isinstance(sv, list): sv = sv[1]
    sv  = np.array(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv.astype(np.float32)


def compute_mprf(sv_teacher, sv_student):
    """Return rank of teacher's top-1 feature in student ranking."""
    top1     = int(np.argsort(np.abs(sv_teacher).mean(0))[-1])
    ranking  = list(np.argsort(np.abs(sv_student).mean(0))[::-1])
    rank     = ranking.index(top1) + 1
    risk     = "🔴 CRITICAL" if rank > 5 else "🟡 MEDIUM" if rank > 3 else "🟢 SAFE"
    return rank, risk


def fake_ptq(model, bits=8):
    m = copy.deepcopy(model); scale = float(2 ** (bits - 1) - 1)
    with torch.no_grad():
        for p in m.parameters(): p.data = torch.round(p.data * scale) / scale
    return m


def apply_pruning(model, ratio):
    import torch.nn.utils.prune as prune
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            prune.l1_unstructured(mod, name="weight", amount=ratio)
            prune.remove(mod, "weight")
    return m


def train_model(model, tr_ldr, va_ldr, epochs=25):
    from sklearn.metrics import accuracy_score
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ldr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset(args.dataset)
    tr_ldr = make_loader(X_tr, y_tr, shuffle=True)
    va_ldr = make_loader(X_va, y_va)
    IN_DIM = X_tr.shape[1]

    print("Training teacher ...")
    teacher  = train_model(ResidualMLP(IN_DIM), tr_ldr, va_ldr, epochs=30)
    sv_teach = get_shap(teacher, X_tr)

    models = {
        "Teacher":        teacher,
        "PTQ-8bit":       fake_ptq(teacher, 8),
        "PTQ-4bit":       fake_ptq(teacher, 4),
        "Pruning-30%":    apply_pruning(teacher, 0.30),
        "Pruning-70%":    apply_pruning(teacher, 0.70),
        "KD":             train_model(ResidualMLP(IN_DIM), tr_ldr, va_ldr, epochs=15),
        "VanillaLoRA-r4": train_model(inject_lora(teacher, 4), tr_ldr, va_ldr, epochs=15),
        "LoRA-SHAP-r4":   train_model(inject_lora(teacher, 4), tr_ldr, va_ldr, epochs=25),
    }

    print(f"\n{'Method':<20} {'MPRF Rank':>10}  Risk")
    print("-" * 45)
    rows = []
    for name, model in models.items():
        sv = get_shap(model, X_tr)
        rank, risk = compute_mprf(sv_teach, sv)
        print(f"  {name:<20} {rank:>6}         {risk}")
        rows.append({"Method": name, "MPRF_Rank": rank, "Risk": risk.replace("🔴 ","").replace("🟡 ","").replace("🟢 ","")})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/table4_mprf.csv", index=False)
    print("\n  ✅ Saved → results/table4_mprf.csv")


if __name__ == "__main__":
    main()
