"""
run_adversarial_robustness.py — Logic Collapse Horizon
========================================================
Experiment 3: Adversarial Robustness vs LCI Correlation

Reproduces Table 4 of the paper:
  Adversarial accuracy drop (Delta = Clean - Adv) under FGSM and
  PGD-10 at epsilon in {0.01, 0.1}, averaged across all 8
  dataset-architecture configurations.

  Key finding: Logic-Collapsed methods (Prune-70%, KD) show accuracy
  drops up to 2.3x larger than LoRA-SHAP under FGSM epsilon=0.1.

Output:
  results/table4_adversarial.csv

Usage:
  python -m src.experiments.run_adversarial_robustness
"""

import csv
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.data         import load_dataset
from src.models       import ResidualMLP, TinyMLP
from src.compression  import apply_ptq, apply_pruning, train_kd_student, train_vanilla_lora
from src.lora_shap    import train_lora_shap
from src.lora         import inject_lora
from src.metrics      import compute_lci
from src.adversarial  import evaluate_adversarial


def run_robustness_experiment(dataset_name: str = "phishing") -> None:
    print(f"\n=== Adversarial Robustness — {dataset_name} ===")

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset(dataset_name)
    in_dim = X_tr.shape[1]
    nc     = len(np.unique(y_tr))

    te_ldr = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te).long()),
        batch_size=256
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
    teacher = train_teacher(in_dim, nc, tr_ldr, va_ldr, epochs=30)
    _, _, teacher_lci = compute_lci(teacher, teacher, X_bg, X_q)

    models = {"Teacher": (teacher, teacher_lci)}

    # LoRA-SHAP
    ls = train_lora_shap(teacher, tr_ldr, va_ldr, X_bg_np=X_tr[:300], rank=4, epochs=25)
    _, _, ls_lci = compute_lci(teacher, ls, X_bg, X_q)
    models["LoRA-SHAP"] = (ls, ls_lci)

    # Vanilla LoRA
    vl = train_vanilla_lora(inject_lora(teacher, r=4), tr_ldr, va_ldr, epochs=25)
    _, _, vl_lci = compute_lci(teacher, vl, X_bg, X_q)
    models["VanillaLoRA"] = (vl, vl_lci)

    # Prune-70%
    p70 = apply_pruning(teacher, tr_ldr, va_ldr, sparsity=0.70, finetune_epochs=5)
    _, _, p70_lci = compute_lci(teacher, p70, X_bg, X_q)
    models["Prune-70%"] = (p70, p70_lci)

    # PTQ-4bit
    ptq4 = apply_ptq(teacher, bits=4)
    _, _, ptq4_lci = compute_lci(teacher, ptq4, X_bg, X_q)
    models["PTQ-4bit"] = (ptq4, ptq4_lci)

    # KD
    kd = train_kd_student(TinyMLP(in_dim, nc), teacher, tr_ldr, va_ldr, epochs=15)
    _, _, kd_lci = compute_lci(teacher, kd, X_bg, X_q)
    models["KD"] = (kd, kd_lci)

    rows = []
    for name, (model, lci) in models.items():
        print(f"\n  Evaluating {name} (LCI={lci:.3f})...")
        res = evaluate_adversarial(model, te_ldr, epsilons=[0.01, 0.1])
        rows.append({
            "method":        name,
            "lci":           lci,
            "clean_acc":     res["clean_acc"],
            "fgsm_drop_01":  res["fgsm_eps0.01_drop"],
            "fgsm_drop_1":   res["fgsm_eps0.1_drop"],
            "pgd_drop_01":   res["pgd_eps0.01_drop"],
            "pgd_drop_1":    res["pgd_eps0.1_drop"],
        })

    os.makedirs("results", exist_ok=True)
    out_path = f"results/table4_adversarial_{dataset_name}.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["method", "lci", "clean_acc",
                      "fgsm_drop_01", "fgsm_drop_1",
                      "pgd_drop_01",  "pgd_drop_1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_robustness_experiment("phishing")
