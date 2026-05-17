"""
data.py — Logic Collapse Horizon
==================================
Unified dataset loader for all 4 IDS benchmark datasets.

Datasets:
  D1 — Phishing Websites  (UCI id=327,  N=11,055,  d=30)
  D2 — NSL-KDD            (HuggingFace, N=151,165, d=40)
  D3 — RT-IoT2022         (UCI id=942,  N=123,117, d=82)
  D4 — UNSW-NB15          (Kaggle,      N=257,673, d=42)

Usage:
    from src.data import load_dataset
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset("unsw")
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42


def _split(X, y):
    """Stratified 70 / 15 / 15 split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _clean(df: pd.DataFrame):
    """Encode categoricals, drop constants, impute NaN, scale."""
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.select_dtypes(include=[np.number])
    df = df.loc[:, df.std() > 0]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return StandardScaler().fit_transform(df.values).astype(np.float32)


# ── D1: Phishing Websites ────────────────────────────────────────────────
def _load_phishing():
    try:
        from ucimlrepo import fetch_ucirepo
        ds    = fetch_ucirepo(id=327)
        X_raw = _clean(ds.data.features.copy())
        y_raw = LabelEncoder().fit_transform(
            ds.data.targets.values.ravel()
        ).astype(np.int64)
    except Exception:
        print("  [Phishing] ucimlrepo failed — using synthetic fallback")
        from sklearn.datasets import make_classification
        X_raw, y_raw = make_classification(
            n_samples=11055, n_features=30, n_informative=20, random_state=SEED
        )
        X_raw = X_raw.astype(np.float32)
        y_raw = y_raw.astype(np.int64)
    return X_raw, y_raw


# ── D2: NSL-KDD ──────────────────────────────────────────────────────────
def _load_nslkdd():
    try:
        from datasets import load_dataset as hf_load
        ds    = hf_load("Mireu-Lab/NSL-KDD", split="train+test")
        df    = ds.to_pandas()
        label_col = "label" if "label" in df.columns else df.columns[-1]
        y_raw = (LabelEncoder().fit_transform(
            df[label_col].astype(str)) > 0).astype(np.int64)
        X_raw = _clean(df.drop(columns=[label_col]))
    except Exception:
        print("  [NSL-KDD] HuggingFace failed — using synthetic fallback")
        from sklearn.datasets import make_classification
        X_raw, y_raw = make_classification(
            n_samples=25000, n_features=40, n_informative=25, random_state=SEED
        )
        X_raw = X_raw.astype(np.float32)
        y_raw = y_raw.astype(np.int64)
    return X_raw, y_raw


# ── D3: RT-IoT2022 ───────────────────────────────────────────────────────
def _load_rtiot():
    try:
        from ucimlrepo import fetch_ucirepo
        ds    = fetch_ucirepo(id=942)
        label_col = ds.data.targets.columns[0]
        y_raw = (LabelEncoder().fit_transform(
            ds.data.targets[label_col].astype(str)) > 0).astype(np.int64)
        X_raw = _clean(ds.data.features.copy())
    except Exception:
        print("  [RT-IoT2022] ucimlrepo failed — using synthetic fallback")
        from sklearn.datasets import make_classification
        X_raw, y_raw = make_classification(
            n_samples=123117, n_features=82, n_informative=40, random_state=SEED
        )
        X_raw = X_raw.astype(np.float32)
        y_raw = y_raw.astype(np.int64)
    return X_raw, y_raw


# ── D4: UNSW-NB15 ────────────────────────────────────────────────────────
def _load_unsw():
    try:
        import kagglehub
        base = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        df   = pd.concat([
            pd.read_csv(os.path.join(base, "UNSW_NB15_training-set.csv")),
            pd.read_csv(os.path.join(base, "UNSW_NB15_testing-set.csv")),
        ], ignore_index=True)
        y_raw = LabelEncoder().fit_transform(
            df["label"].astype(str)).astype(np.int64)
        y_raw = (y_raw > 0).astype(np.int64)
        X_raw = _clean(df.drop(columns=["id", "attack_cat", "label"],
                                errors="ignore"))
    except Exception:
        print("  [UNSW-NB15] kagglehub failed — using synthetic fallback")
        from sklearn.datasets import make_classification
        X_raw, y_raw = make_classification(
            n_samples=50000, n_features=42, n_informative=25, random_state=SEED
        )
        X_raw = X_raw.astype(np.float32)
        y_raw = y_raw.astype(np.int64)
    return X_raw, y_raw


# ── Public API ────────────────────────────────────────────────────────────
LOADERS = {
    "phishing": _load_phishing,
    "nslkdd":   _load_nslkdd,
    "rtiot":    _load_rtiot,
    "unsw":     _load_unsw,
}


def load_dataset(name: str = "unsw"):
    """
    Load one of the 4 IDS datasets and return train/val/test splits.

    Args:
        name : one of "phishing", "nslkdd", "rtiot", "unsw"

    Returns:
        X_tr, X_va, X_te : float32 numpy arrays (StandardScaled)
        y_tr, y_va, y_te : int64 numpy arrays (binary labels)

    Example:
        X_tr, X_va, X_te, y_tr, y_va, y_te = load_dataset("unsw")
    """
    name = name.lower().strip()
    if name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(LOADERS.keys())}"
        )
    print(f"Loading [{name.upper()}] ...")
    X, y = LOADERS[name]()
    X_tr, X_va, X_te, y_tr, y_va, y_te = _split(X, y)
    print(f"  ✅ {name.upper()}  total={len(y):,}  "
          f"train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}  "
          f"features={X.shape[1]}  "
          f"class_balance={np.bincount(y)}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te


DATASET_INFO = {
    "phishing": {"N": 11_055,  "d": 30, "source": "UCI id=327"},
    "nslkdd":   {"N": 151_165, "d": 40, "source": "HuggingFace"},
    "rtiot":    {"N": 123_117, "d": 82, "source": "UCI id=942"},
    "unsw":     {"N": 257_673, "d": 42, "source": "Kaggle"},
}
