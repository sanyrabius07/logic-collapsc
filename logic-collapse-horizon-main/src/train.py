"""
train.py — Logic Collapse Horizon
==================================
Full training pipeline:
  1. Load Phishing dataset (UCI #967)
  2. Train Teacher (ResidualMLP)
  3. Train LoRA-SHAP  (SHAP-guided LoRA)
  4. Train VanillaLoRA (no SHAP guidance)
  5. Train KD student  (TinyMLP)
  6. Apply Pruning-70% + fine-tune
  7. Compute and print LCI for all models

Usage:
  python src/train.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .models       import ResidualMLP, TinyMLP, inject_lora
from .compression  import train_lora_shap, train_vanilla_lora, train_kd_student, apply_pruning
from .metrics      import compute_lci, evaluate_accuracy

np.random.seed(42)
torch.manual_seed(42)


def load_phishing():
    """Load and preprocess the UCI Phishing Websites dataset."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds    = fetch_ucirepo(id=967)
        X_raw = ds.data.features.values.astype(np.float32)
        y_raw = ds.data.targets.values.ravel()
    except Exception:
        print("  ucimlrepo unavailable — using synthetic fallback")
        from sklearn.datasets import make_classification
        X_raw, y_raw = make_classification(
            n_samples=11055, n_features=30, n_informative=20, random_state=42
        )
        X_raw = X_raw.astype(np.float32)

    le    = LabelEncoder()
    y_raw = (le.fit_transform(y_raw) > 0).astype(np.int64)
    X_raw = StandardScaler().fit_transform(X_raw).astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y_raw, test_size=0.30, stratify=y_raw, random_state=42
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def make_loader(X, y, batch_size=512, shuffle=False):
    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y).long()),
        batch_size=batch_size, shuffle=shuffle
    )


def train_teacher(in_dim, nc, tr_ldr, va_ldr, epochs=30, lr=1e-3):
    model = ResidualMLP(in_dim, nc)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_ldr:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        v = evaluate_accuracy(model, va_ldr)
        if v > best_acc:
            best_acc   = v
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"    ep{ep:02d}  val={v:.4f}", end="  ", flush=True)
    print()
    model.load_state_dict(best_state)
    return model


def main():
    print("Loading Phishing dataset...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_phishing()
    IN_DIM, NC = X_tr.shape[1], len(np.unique(y_tr))
    print(f"  IN_DIM={IN_DIM}  NC={NC}  Train={len(X_tr)}  Val={len(X_va)}  Test={len(X_te)}")

    tr_ldr = make_loader(X_tr, y_tr, shuffle=True)
    va_ldr = make_loader(X_va, y_va)
    te_ldr = make_loader(X_te, y_te)
    Xbg    = torch.tensor(X_tr[:100])
    Xq     = torch.tensor(X_te[:200])

    # Teacher
    print("\nTraining Teacher...")
    teacher  = train_teacher(IN_DIM, NC, tr_ldr, va_ldr, epochs=30)
    te_acc   = evaluate_accuracy(teacher, te_ldr)
    print(f"  Teacher  acc={te_acc:.4f}")

    # LoRA-SHAP
    print("\nTraining LoRA-SHAP...")
    lora_shap = train_lora_shap(
        inject_lora(teacher, r=4), teacher, tr_ldr, va_ldr, Xbg, Xq, epochs=25
    )
    _, _, ls_lci = compute_lci(teacher, lora_shap, Xbg, Xq)
    print(f"  LoRA-SHAP  acc={evaluate_accuracy(lora_shap, te_ldr):.4f}  LCI={ls_lci:.4f}")

    # VanillaLoRA
    print("\nTraining VanillaLoRA...")
    vanilla_lora = train_vanilla_lora(inject_lora(teacher, r=4), tr_ldr, va_ldr, epochs=25)
    _, _, vl_lci = compute_lci(teacher, vanilla_lora, Xbg, Xq)
    print(f"  VanillaLoRA  acc={evaluate_accuracy(vanilla_lora, te_ldr):.4f}  LCI={vl_lci:.4f}")

    # KD student
    print("\nTraining KD student...")
    kd = train_kd_student(TinyMLP(IN_DIM, NC), teacher, tr_ldr, va_ldr, epochs=25)
    _, _, kd_lci = compute_lci(teacher, kd, Xbg, Xq)
    print(f"  KD  acc={evaluate_accuracy(kd, te_ldr):.4f}  LCI={kd_lci:.4f}")

    # Pruning
    print("\nApplying Pruning-70%...")
    pruned = apply_pruning(teacher, tr_ldr, va_ldr, sparsity=0.70, finetune_epochs=7)
    _, _, pr_lci = compute_lci(teacher, pruned, Xbg, Xq)
    print(f"  Pruning  acc={evaluate_accuracy(pruned, te_ldr):.4f}  LCI={pr_lci:.4f}")

    print("\n  All models trained. Ready for experiments.")
    return {
        "Teacher":     teacher,
        "LoRA-SHAP":   lora_shap,
        "VanillaLoRA": vanilla_lora,
        "KD":          kd,
        "Pruning":     pruned,
    }


if __name__ == "__main__":
    main()
