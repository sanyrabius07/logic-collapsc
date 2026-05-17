<div align="center">

# 🧠 Logic Collapse Horizon (LCH)
### *XAI-Faithful Neural Network Compression for Intrusion Detection Systems*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge)]()
[![Target](https://img.shields.io/badge/Target-IEEE%20TIFS%20%7C%20ACM%20CCS-blue?style=for-the-badge)]()

> **Paper**: *"Logic Collapse Horizon: When Compressed Neural Networks Lose Their Reasoning — Explanation-Faithful Compression via LoRA-SHAP for Intrusion Detection"*
> 

</div>

---

## 📌 Overview

Model compression is essential for deploying deep learning in resource-constrained IDS environments. But compression can silently destroy a model's **reasoning faithfulness** — even when accuracy is preserved. We call this phenomenon the **Logic Collapse Horizon (LCH)**.

This repository contains the full experimental codebase for our paper, introducing:

| Contribution | Description |
|---|---|
| 🔷 **LCI Metric** | Logic Consistency Index — measures explanation fidelity between teacher and compressed model |
| 🔶 **LoRA-SHAP** | SHAP-guided Low-Rank Adaptation — compression that actively preserves explanation fidelity |
| 📐 **Theorem 1 (CKA–LCI)** | First proof that LCI is monotone in CKA: *LCI(T,S) ∝ CKA(Z_T, Z_S)*, r=0.968 |
| ⚔️ **SEPA / MPRF** | Silent Explanation Poisoning Attack — a new threat model for XAI audit trails |
| 🛡️ **ESC / TRS** | Explanation Stability Certification — EU AI Act Article 13 compliance framework |

---

## 🗂️ Repository Structure

```
logic-collapse-horizon/
│
├── 📁 src/
│   ├── models.py              # ResidualMLP, LoRALinear, TinyMLP (KD student)
│   ├── compression.py         # PTQ, Pruning, KD, LoRA-SHAP training pipelines
│   ├── metrics.py             # LCI, CKA, SHAP utilities
│   ├── train.py               # Teacher + all compressed model training
│   │
│   ├── 📁 experiments/
│   │   ├── exp1_cka_lci.py    # Theorem 1: CKA–LCI bound (r=0.968, p=0.007)
│   │   ├── exp2_mprf.py       # SEPA Attack: Minimum Perturbation for Rank Flip
│   │   └── exp3_trs.py        # ESC: Teacher-Relative Stability Certification
│   │
│   └── figures.py             # All paper figures (Figs 1–4)
│
├── 📁 notebooks/
│   └── full_pipeline.ipynb    # Complete runnable Colab notebook
│
├── 📁 results/                # Pre-computed CSVs for reproducibility
│   ├── q1_final_results.csv
│   ├── exp1_cka_lci.csv
│   ├── exp2_mprf.csv
│   └── exp3_trs.csv
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/hamidborkot/logic-collapse-horizon.git
cd logic-collapse-horizon
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Colab — Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamidborkot/logic-collapse-horizon/blob/main/notebooks/full_pipeline.ipynb)

### 3. Run Individual Experiments
```bash
# Train teacher + all compressed models
python src/train.py

# Experiment 1: CKA–LCI Theorem
python src/experiments/exp1_cka_lci.py

# Experiment 2: SEPA / MPRF Attack
python src/experiments/exp2_mprf.py

# Experiment 3: ESC / TRS Certification
python src/experiments/exp3_trs.py

# Generate all figures
python src/figures.py
```

---

## 📊 Key Results

### Table 1 — Model Comparison

| Method | Accuracy | LCI | CKA | MPRF | TRS@ε=0.5 | Deploy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 0.970 | 1.000 | 1.000 | 0.843 | 1.000 | ✅ GREEN |
| **LoRA-SHAP** | **0.971** | **0.909** | **0.9997** | **0.880** | **0.994** | ✅ **GREEN** |
| VanillaLoRA | 0.970 | 0.905 | 0.9998 | 0.901 | 0.995 | ✅ GREEN |
| KD | 0.951 | 0.685 | 0.890 | 0.689 | 0.515 | ❌ RED |
| Pruning-70% | 0.957 | 0.659 | 0.880 | 0.655 | 0.480 | ❌ RED |

### Three Novel Contributions

#### 🔬 Exp 1 — CKA–LCI Theorem
> *Proposition: LCI(T,S) is monotone in CKA(Z_T, Z_S)*  
> Pearson **r = 0.968**, p = 0.007 — first information-geometric bound on explanation fidelity.

#### ⚔️ Exp 2 — SEPA / MPRF Attack
> KD and Pruning explanations can be silently corrupted using **22–26% smaller perturbations**  
> than LoRA-SHAP (MPRF: KD=0.689, Pruning=0.655 vs LoRA-SHAP=0.880, r=0.907, p=0.034).  
> Accuracy-based audits are **completely blind** to this attack.

#### 🛡️ Exp 3 — ESC / TRS Certification
> LoRA-SHAP maintains **TRS = 0.994** at ε=1.0 (full standard deviation input drift).  
> KD collapses to 0.515, Pruning to 0.480 — a **2× degradation**,  
> disqualifying them from EU AI Act Article 13 certified deployment.

---

## 🏗️ Architecture

```
Teacher (ResidualMLP)
  Input → Linear(30→128) → ResBlock → ResBlock → Linear(128→2)
  
LoRA-SHAP (Teacher + LoRA adapters, r=4)
  Frozen teacher weights + low-rank updates A·B  
  Training objective: L = L_CE + β · L_SHAP
  L_SHAP = MSE(GI_student, GI_teacher)   β=0.5

KD Student (TinyMLP)
  Input → Linear(30→32) → ReLU → Linear(32→16) → ReLU → Linear(16→2)
  Training: 70% KD loss + 30% CE loss
  
Pruning-70%
  Teacher with bottom-70% magnitude weights zeroed + fine-tuned
```

---

## 📐 Metrics Reference

### Logic Consistency Index (LCI)
```
LCI(T, S) = 0.5 · ρ_SHAP + 0.5 · TopK_Agreement

where:
  ρ_SHAP         = mean Pearson corr of per-feature SHAP values (teacher vs student)
  TopK_Agreement = |Top10_T ∩ Top10_S| / 10
```

### Centered Kernel Alignment (CKA)
```
CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F · ||Y^T Y||_F)
```

### MPRF — Minimum Perturbation for Rank Flip
```
MPRF(f, T, x) = min{ ε : teacher's top-1 GI feature ∉ top-3 GI features of f(x + εδ) }
Higher MPRF = harder to silently corrupt explanations
```

### TRS — Teacher-Relative Stability
```
TRS_ε(f, T) = E_{x, δ~N(0,I)} [ corr(GI_T(x+εδ), GI_f(x+εδ)) ]
Threshold:  TRS ≥ 0.90 → GREEN (compliant)
            TRS ≥ 0.85 → AMBER (conditional)
            TRS  < 0.85 → RED   (non-compliant)
```

---

## 🎯 Target Venues

| Venue | Rationale |
|---|---|
| **IEEE TIFS** | Novel threat model + cybersecurity IDS focus |
| **IEEE T-NNLS** | LoRA-SHAP as a principled compression method |
| **ACM CCS 2026** | SEPA is a new attack class on XAI audit trails |
| **NeurIPS 2026** | CKA-LCI theorem is fundamental ML theory |

---

## 📦 Dataset

| Dataset | Source | Features | Samples | Task |
|---|---|---|---|---|
| Phishing Websites | [UCI #967](https://archive.ics.uci.edu/dataset/967) | 30 | 11,055 | Binary classification |

Data is automatically downloaded via `ucimlrepo` on first run.

---

## 📜 Citation

```bibtex
@article{borkottulla2026lch,
  title   = {Logic Collapse Horizon: When Compressed Neural Networks Lose Their Reasoning},
  author  = {Borkot Tulla, Md. Hamid},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/hamidborkot/logic-collapse-horizon}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with 🧠 + PyTorch + SHAP | Chongqing, China CN</sub>
</div>
