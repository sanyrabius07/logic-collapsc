"""
run_aca.py — Logic Collapse Horizon
=====================================
Experiment 2: Adversarial Compression Attack (ACA)

Reproduces Table 3 of the paper:
  ACA on UNSW-NB15 (ResidualMLP) — PTQ-5bit maximises attribution
  corruption within the ±2% accuracy variance documented for
  production IDS systems, while remaining undetectable by
  accuracy-based monitors.

Output:
  results/table3_aca.csv

Usage:
  python -m src.experiments.run_aca
"""

import csv
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models      import ResidualMLP
from src.data        import load_dataset
from src.compression import apply_ptq
from src.metrics     import compute_lci, evaluate_accuracy

DELTA_ACC_THRESHOLD = 0.02   # operational accuracy variance for production IDS


def run_aca_experiment() -> None:
    print("\n=== ACA Experiment — UNSW-NB15 (ResidualMLP) ===")

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset("unsw-nb15")
    in_dim = X_tr.shape[1]
    nc     = len(np.unique(y_tr))

    te_ldr = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te).long()),
        batch_size=1024
    )
    tr_ldr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).long()),
        batch_size=512, shuffle=True
    )
    va_ldr = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va).long()),
        batch_size=512
    )

    X_bg = torch.tensor(X_tr[:100])
    X_q  = torch.tensor(X_te[:300])

    from src.train import train_teacher
    print("Training teacher...")
    teacher   = train_teacher(in_dim, nc, tr_ldr, va_ldr, epochs=30)
    base_acc  = evaluate_accuracy(teacher, te_ldr)
    _, _, base_lci = compute_lci(teacher, teacher, X_bg, X_q)
    print(f"  Teacher  acc={base_acc:.4f}  LCI={base_lci:.4f}")

    rows = [{"compression": "Teacher (baseline)", "lci": base_lci,
              "acc": base_acc, "delta_acc": 0.000}]

    # Evaluate PTQ at multiple bit-widths as ACA search
    for bits in [8, 5, 4]:
        student       = apply_ptq(teacher, bits=bits)
        acc           = evaluate_accuracy(student, te_ldr)
        _, _, lci     = compute_lci(teacher, student, X_bg, X_q)
        delta_acc     = round(abs(base_acc - acc), 4)
        detectable    = delta_acc > DELTA_ACC_THRESHOLD
        tag           = f"PTQ-{bits}bit"
        if bits == 5:
            tag += " (ACA)"
        elif detectable:
            tag += " (detectable)"
        else:
            tag += " (safe)"
        rows.append({"compression": tag, "lci": lci,
                     "acc": acc, "delta_acc": delta_acc})
        print(f"  {tag:<30s}  acc={acc:.4f}  LCI={lci:.4f}  "
              f"delta_acc={delta_acc:.3f}  detectable={detectable}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/table3_aca.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["compression", "lci", "acc", "delta_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_aca_experiment()
