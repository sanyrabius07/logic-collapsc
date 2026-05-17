"""
run_wilcoxon.py — Logic Collapse Horizon
==========================================
Reproduces Table 6: Wilcoxon Signed-Rank Tests.

Tests whether LoRA-SHAP LCI is significantly greater than each baseline
using 10 bootstrap samples of 300 instances from UNSW-NB15.

Usage:
    python experiments/run_wilcoxon.py
    python experiments/run_wilcoxon.py --n_bootstrap 20 --seed 42

Outputs:
    results/table6_wilcoxon.csv
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
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from src.data      import load_dataset
from src.models    import ResidualMLP
from src.lora      import inject_lora
import shap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_bootstrap", type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--dataset",     type=str, default="unsw")
    return p.parse_args()


def make_loader(X, y, bs=1024, shuffle=False):
    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y).long()),
        batch_size=bs, shuffle=shuffle
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        ps.append(model(xb).argmax(1)); ys.append(yb)
    return accuracy_score(
        torch.cat(ys).numpy(), torch.cat(ps).numpy()
    )


def train_model(model, tr_ldr, va_ldr, epochs=30, lr=1e-3):
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc, best_st = 0., None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ldr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sch.step()
        v = evaluate(model, va_ldr)
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


def apply_kd(teacher, tr_ldr, va_ldr, epochs=15):
    s   = ResidualMLP(IN_DIM).to("cpu")
    opt = torch.optim.Adam(s.parameters(), lr=1e-3)
    ce  = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    for _ in range(epochs):
        s.train()
        for xb, yb in tr_ldr:
            with torch.no_grad(): tl = teacher(xb)
            sl   = s(xb)
            loss = 0.7 * kl(F.log_softmax(sl / 3, -1),
                             F.softmax(tl / 3, -1)) * 9 + 0.3 * ce(sl, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return s


def train_lora_shap(teacher, tr_ldr, va_ldr, r=4, epochs=30):
    s = inject_lora(teacher, r=r)
    trainable = [p for p in s.parameters() if p.requires_grad]
    opt  = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(); eps = 0.05
    for _ in range(epochs):
        s.train()
        for xb, yb in tr_ldr:
            opt.zero_grad()
            ce_loss = crit(s(xb), yb)
            n       = F.normalize(torch.randn_like(xb), p=2, dim=-1)
            sv_s    = (s(xb + eps * n) - s(xb - eps * n))[:, 1:2] * xb
            with torch.no_grad():
                sv_t = (teacher(xb + eps * n) - teacher(xb - eps * n))[:, 1:2] * xb
            shap_loss = (1. - (F.normalize(sv_t, p=2, dim=-1) *
                               F.normalize(sv_s, p=2, dim=-1)).sum(-1)).mean()
            (ce_loss + 0.01 * shap_loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
        sch.step()
    return s


IN_DIM = None  # set at runtime


def main():
    global IN_DIM
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading {args.dataset.upper()} ...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset(args.dataset)
    IN_DIM = X_tr.shape[1]

    tr_ldr = make_loader(X_tr, y_tr, shuffle=True)
    va_ldr = make_loader(X_va, y_va)

    print("\nTraining teacher ...")
    teacher = train_model(ResidualMLP(IN_DIM), tr_ldr, va_ldr)
    print(f"  Teacher acc={evaluate(teacher, make_loader(X_te, y_te)):.4f}")

    print("\nTraining LoRA-SHAP-r4 ...")
    lora_shap = train_lora_shap(teacher, tr_ldr, va_ldr)

    print("\nBuilding baselines ...")
    baselines = {
        "PTQ-8bit":       fake_ptq(teacher, 8),
        "PTQ-4bit":       fake_ptq(teacher, 4),
        "Pruning-30%":    apply_pruning(teacher, 0.30),
        "Pruning-70%":    apply_pruning(teacher, 0.70),
        "KD":             apply_kd(teacher, tr_ldr, va_ldr),
        "VanillaLoRA-r4": train_model(inject_lora(teacher, 4), tr_ldr, va_ldr),
    }

    print(f"\n{'='*60}")
    print(f"  WILCOXON TESTS  (n={args.n_bootstrap} bootstrap samples)")
    print(f"{'='*60}")

    rows = []
    for bl_name, bl_model in baselines.items():
        lci_lora, lci_bl = [], []
        for seed in range(args.n_bootstrap):
            np.random.seed(seed)
            idx  = np.random.choice(len(X_tr), 300, replace=False)
            Xsub = X_tr[idx]
            sv_t = get_shap(teacher,    Xsub)
            sv_ls= get_shap(lora_shap,  Xsub)
            sv_b = get_shap(bl_model,   Xsub)
            lci_lora.append(compute_lci(sv_t, sv_ls))
            lci_bl.append(  compute_lci(sv_t, sv_b))

        try:
            _, p = wilcoxon(lci_lora, lci_bl, alternative="greater")
        except Exception:
            p = float("nan")

        sig = "p<0.05 ✅" if (not np.isnan(p) and p < 0.05) else "NOT sig ❌"
        print(f"  LoRA-SHAP vs {bl_name:<18}  "
              f"LoRA={np.mean(lci_lora):.4f}  "
              f"Base={np.mean(lci_bl):.4f}  "
              f"p={p:.5f}  {sig}")
        rows.append({
            "Comparison":     f"LoRA-SHAP vs {bl_name}",
            "LoRA-SHAP LCI":  round(np.mean(lci_lora), 4),
            "Baseline LCI":   round(np.mean(lci_bl),   4),
            "Delta LCI":      round(np.mean(lci_lora) - np.mean(lci_bl), 4),
            "p-value":        round(p, 5) if not np.isnan(p) else "NaN",
            "Significant":    "Yes" if (not np.isnan(p) and p < 0.05) else "No",
        })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/table6_wilcoxon.csv", index=False)
    print("\n  ✅ Saved → results/table6_wilcoxon.csv")


if __name__ == "__main__":
    main()
