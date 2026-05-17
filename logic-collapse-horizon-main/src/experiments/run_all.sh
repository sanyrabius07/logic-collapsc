#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Logic Collapse Horizon — Full Reproduction Script
#  Reproduces all tables and figures from one command.
#
#  Usage:
#    bash experiments/run_all.sh
#
#  Expected runtime:
#    CPU  : ~4–6 hours
#    GPU  : ~40–60 minutes
# ═══════════════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Logic Collapse Horizon — Full Reproduction     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

mkdir -p results figures

echo "[1/3] Figure 1 — LCH Curve (fig1_lch_curve.pdf)"
python experiments/run_lch.py --dataset unsw --seed 42
echo ""

echo "[2/3] Table 4 — MPRF Analysis (table4_mprf.csv)"
python experiments/run_mprf.py --dataset unsw --seed 42
echo ""

echo "[3/3] Table 6 — Wilcoxon Significance Tests (table6_wilcoxon.csv)"
python experiments/run_wilcoxon.py --dataset unsw --n_bootstrap 10 --seed 42
echo ""

echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅ All experiments complete.                   ║"
echo "║                                                  ║"
echo "║   results/table4_mprf.csv                        ║"
echo "║   results/table6_wilcoxon.csv                    ║"
echo "║   figures/fig1_lch_curve.pdf                     ║"
echo "║   figures/fig1_lch_curve.png                     ║"
echo "╚══════════════════════════════════════════════════╝"
