# Reproducing Results — Logic Collapse Horizon

This document provides step-by-step instructions to reproduce all tables
and figures from the paper:

> **Silent Corruption: Logic Collapse in Compressed Network Intrusion Detection Systems**

---

## Environment Setup

```bash
git clone https://github.com/hamidborkot/logic-collapse-horizon.git
cd logic-collapse-horizon
pip install -r requirements.txt
```

Python 3.9+ required. GPU optional (all experiments run on CPU).

---

## Datasets

The paper evaluates on four public IDS benchmarks:

| Dataset | Source | Samples | Features |
|---|---|---|---|
| UNSW-NB15 | [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 257,673 | 42 |
| NSL-KDD | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) | 297,034 | 40 |
| RT-IoT2022 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/942) | 123,117 | 83 |
| Phishing | [UCI ML Repository #967](https://archive.ics.uci.edu/dataset/967) | 11,055 | 30 |

Download each dataset and place in `data/` as:
```
data/UNSW_NB15_training-set.csv
data/KDDTrain+.txt
data/RT_IOT2022.csv
data/phishing.csv
```

---

## Reproducing Table 2 (LCI × Accuracy Grid)

Reproduces the main results table across all compression methods,
architectures, and datasets.

```bash
python -m src.experiments.run_logic_collapse --dataset phishing
python -m src.experiments.run_logic_collapse --dataset nsl-kdd
python -m src.experiments.run_logic_collapse --dataset unsw-nb15
python -m src.experiments.run_logic_collapse --dataset rt-iot2022
```

Output: `results/table2_lci_accuracy_{dataset}.csv`

Expected runtime: ~30–60 min per dataset on CPU.

---

## Reproducing Table 3 (Adversarial Compression Attack)

Reproduces the ACA demonstration on UNSW-NB15 (ResidualMLP):
PTQ-5bit maximises attribution corruption within ±2% accuracy variance.

```bash
python -m src.experiments.run_aca
```

Output: `results/table3_aca.csv`

Key expected values:
- PTQ-5bit (ACA): LCI=0.874, ΔAcc=0.009 (within operational threshold)
- PTQ-4bit (detectable): LCI=0.680, ΔAcc=0.203 (exceeds threshold)

---

## Reproducing Table 4 (Adversarial Robustness vs LCI)

Reproduces adversarial accuracy drop under FGSM and PGD-10
at ε ∈ {0.01, 0.1}.

```bash
python -m src.experiments.run_adversarial_robustness
```

Output: `results/table4_adversarial_{dataset}.csv`

Key expected finding: LoRA-SHAP incurs the smallest adversarial
accuracy drop across all attack settings.

---

## Reproducing Section 4.4 (Logic Collapse Oracle)

Trains and evaluates the LogicCollapseOracle binary classifier.

```bash
python -m src.experiments.run_oracle
```

Output: `results/oracle_evaluation.csv`

Expected: AUC=0.917, precision=0.89, recall=0.93 (collapsed class).

---

## Full Pipeline (Single Command)

```bash
bash experiments/run_all.sh
```

---

## Pre-computed Results

All result CSVs from the paper are available in `results/`:

| File | Contents |
|---|---|
| `results/table2_lci_accuracy.csv` | Main LCI × Accuracy grid (Table 2) |
| `results/table3_aca.csv` | ACA results (Table 3) |
| `results/table4_adversarial.csv` | Adversarial robustness (Table 4) |
| `results/table5_aggregated_ranks.csv` | Aggregated rank results (Table 5) |
| `results/exp1_cka_lci.csv` | CKA vs LCI correlation |
| `results/exp2_mprf.csv` | Most-Perturbed Rank Feature analysis |
| `results/exp3_trs.csv` | Threshold range sensitivity |
| `results/q1_final_results.csv` | Final Q1 submission results |
| `results/table6_wilcoxon.csv` | Wilcoxon signed-rank test results |

---

## Citation

If you use this code, please cite:

```bibtex
@article{logiccollapseids2026,
  title   = {Silent Corruption: Logic Collapse in Compressed Network Intrusion Detection Systems},
  author  = {Anonymous},
  journal = {Computers \& Security},
  year    = {2026}
}
```
