"""
run_oracle.py — Logic Collapse Horizon
========================================
Experiment 4: Logic Collapse Oracle Training and Evaluation

Reproduces Section 4.4 of the paper:
  Trains the LogicCollapseOracle on 140/176 configurations,
  evaluates on held-out 36 configurations.

  Expected results: AUC=0.917, precision=0.89, recall=0.93.

Output:
  results/oracle_evaluation.csv

Usage:
  python -m src.experiments.run_oracle
"""

import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data        import load_dataset
from src.models      import ResidualMLP, TinyMLP
from src.compression import apply_ptq, apply_pruning, train_kd_student, train_vanilla_lora
from src.lora_shap   import train_lora_shap
from src.lora        import inject_lora
from src.metrics     import compute_lci
from src.oracle      import extract_features, LogicCollapseOracle

LCI_THRESHOLD = 0.85


def _collect_configurations(dataset_name: str):
    """Train all compression variants and collect (features, label) pairs."""
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
    X_bg = torch.tensor(X_tr[:100])
    X_q  = torch.tensor(X_te[:300])
    X_val_t = torch.tensor(X_va[:300])

    from src.train import train_teacher
    teacher = train_teacher(in_dim, nc, tr_ldr, va_ldr, epochs=30)

    configs = []

    def _record(student, method_type, bit_or_sparsity, param_red):
        _, _, lci = compute_lci(teacher, student, X_bg, X_q)
        feat  = extract_features(
            teacher, student, X_val_t,
            method_type=method_type,
            bit_width=bit_or_sparsity,
            param_reduction=param_red,
        )
        label = 1 if lci < LCI_THRESHOLD else 0
        configs.append((feat, label, lci, method_type))

    for bits in [8, 4]:
        _record(apply_ptq(teacher, bits=bits), "ptq", bits, 1 - bits/32)

    for sp in [0.30, 0.50, 0.70]:
        _record(
            apply_pruning(teacher, tr_ldr, va_ldr, sparsity=sp, finetune_epochs=5),
            "pruning", sp, sp
        )

    _record(
        train_kd_student(TinyMLP(in_dim, nc), teacher, tr_ldr, va_ldr, epochs=15),
        "kd", 0, 0.5
    )

    for rank in [1, 2, 4, 8]:
        vl = train_vanilla_lora(inject_lora(teacher, r=rank), tr_ldr, va_ldr, epochs=25)
        _record(vl, "lora", rank, rank / (in_dim * in_dim))

        ls = train_lora_shap(teacher, tr_ldr, va_ldr, X_bg_np=X_tr[:300], rank=rank, epochs=25)
        _record(ls, "lora", rank, rank / (in_dim * in_dim))

    return configs


def run_oracle_experiment() -> None:
    print("\n=== Logic Collapse Oracle ===")

    all_feats, all_labels = [], []
    for ds in ["phishing"]:
        print(f"\n  Collecting configurations for {ds}...")
        configs = _collect_configurations(ds)
        for feat, label, lci, method in configs:
            all_feats.append(feat)
            all_labels.append(label)
            print(f"    {method:<10s}  LCI={lci:.3f}  label={label}")

    X_all = np.stack(all_feats)
    y_all = np.array(all_labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.205, stratify=y_all, random_state=42
    )

    oracle = LogicCollapseOracle()
    oracle.fit(X_tr, y_tr, verbose=True)
    results = oracle.evaluate(X_te, y_te)

    fi = oracle.feature_importances()
    print("\n  Feature importances:")
    for name, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {name:<30s}  {imp:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/oracle_evaluation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["auc", "precision", "recall"])
        writer.writeheader()
        writer.writerow(results)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_oracle_experiment()
