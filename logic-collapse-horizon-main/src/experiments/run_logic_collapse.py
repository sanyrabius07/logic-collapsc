"""
run_logic_collapse.py — Logic Collapse Horizon
================================================
Experiment 1: Logic Collapse at Scale

Reproduces Table 2 of the paper:
  LCI and accuracy across all compression methods × architectures × datasets.

Output:
  results/table2_lci_accuracy.csv

Usage:
  python -m src.experiments.run_logic_collapse --dataset phishing
"""

import argparse
import csv
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models       import ResidualMLP
from src.data         import load_dataset
from src.compression  import apply_ptq, apply_pruning, train_kd_student
from src.lora_shap    import train_lora_shap
from src.lora         import inject_lora
from src.metrics      import compute_lci, evaluate_accuracy


def run_experiment(dataset_name: str = "phishing") -> None:
    print(f"\n=== Logic Collapse at Scale — dataset: {dataset_name} ===")

    # Load data
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset(dataset_name)
    in_dim = X_tr.shape[1]
    nc     = len(np.unique(y_tr))

    tr_ldr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).long()),
        batch_size=512, shuffle=True
    )
    va_ldr = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va).long()),
        batch_size=512
    )
    te_ldr = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te).long()),
        batch_size=512
    )

    X_bg = torch.tensor(X_tr[:100])
    X_q  = torch.tensor(X_te[:300])

    # Train teacher
    from src.train import train_teacher
    print("Training teacher...")
    teacher = train_teacher(in_dim, nc, tr_ldr, va_ldr, epochs=30)

    rows = []
    base_acc = evaluate_accuracy(teacher, te_ldr)
    _, _, base_lci = compute_lci(teacher, teacher, X_bg, X_q)
    rows.append({"method": "Teacher", "lci": base_lci, "acc": base_acc})
    print(f"  Teacher  acc={base_acc:.4f}  LCI={base_lci:.4f}")

    # PTQ variants
    for bits in [8, 4]:
        student = apply_ptq(teacher, bits=bits)
        acc     = evaluate_accuracy(student, te_ldr)
        _, _, lci = compute_lci(teacher, student, X_bg, X_q)
        tag = f"PTQ-{bits}bit"
        rows.append({"method": tag, "lci": lci, "acc": acc})
        print(f"  {tag:<20s}  acc={acc:.4f}  LCI={lci:.4f}")

    # Pruning variants
    for sparsity in [0.30, 0.50, 0.70]:
        student = apply_pruning(teacher, tr_ldr, va_ldr, sparsity=sparsity, finetune_epochs=5)
        acc     = evaluate_accuracy(student, te_ldr)
        _, _, lci = compute_lci(teacher, student, X_bg, X_q)
        tag = f"Prune-{int(sparsity*100)}%"
        rows.append({"method": tag, "lci": lci, "acc": acc})
        print(f"  {tag:<20s}  acc={acc:.4f}  LCI={lci:.4f}")

    # Knowledge Distillation
    from src.models import TinyMLP
    kd_student = train_kd_student(TinyMLP(in_dim, nc), teacher, tr_ldr, va_ldr, epochs=15)
    acc         = evaluate_accuracy(kd_student, te_ldr)
    _, _, lci   = compute_lci(teacher, kd_student, X_bg, X_q)
    rows.append({"method": "KD", "lci": lci, "acc": acc})
    print(f"  {'KD':<20s}  acc={acc:.4f}  LCI={lci:.4f}")

    # Vanilla LoRA
    from src.compression import train_vanilla_lora
    vl = train_vanilla_lora(inject_lora(teacher, r=4), tr_ldr, va_ldr, epochs=25)
    acc       = evaluate_accuracy(vl, te_ldr)
    _, _, lci = compute_lci(teacher, vl, X_bg, X_q)
    rows.append({"method": "VanillaLoRA", "lci": lci, "acc": acc})
    print(f"  {'VanillaLoRA':<20s}  acc={acc:.4f}  LCI={lci:.4f}")

    # LoRA-SHAP
    ls = train_lora_shap(teacher, tr_ldr, va_ldr, X_bg_np=X_tr[:300], rank=4, epochs=25)
    acc       = evaluate_accuracy(ls, te_ldr)
    _, _, lci = compute_lci(teacher, ls, X_bg, X_q)
    rows.append({"method": "LoRA-SHAP", "lci": lci, "acc": acc})
    print(f"  {'LoRA-SHAP':<20s}  acc={acc:.4f}  LCI={lci:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/table2_lci_accuracy_{dataset_name}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "lci", "acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="phishing",
                        choices=["phishing", "nsl-kdd", "unsw-nb15", "rt-iot2022"])
    args = parser.parse_args()
    run_experiment(args.dataset)
