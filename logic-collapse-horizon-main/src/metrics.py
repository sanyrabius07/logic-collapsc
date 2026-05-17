"""
Metrics — Logic Collapse Horizon

Implements:
  LCI   : Logic Consistency Index
  CKA   : Centered Kernel Alignment
  gi_eval : Gradient × Input explanation map
"""
import numpy as np
import torch
import shap
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


# ── Gradient × Input ─────────────────────────────────────────────────────
def gi_eval(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Compute Gradient × Input attribution map.
    gi_i = x_i ⊙ ∇_{x_i} f(x_i)[class=1]

    Args:
        model : trained PyTorch model
        X     : (N, D) float tensor
    Returns:
        gi    : (N, D) float tensor (detached)
    """
    model.eval()
    Xc = X.clone().float().detach().requires_grad_(True)
    model(Xc)[:, 1].sum().backward()
    return (Xc * Xc.grad).detach()


# ── SHAP explanations ─────────────────────────────────────────────────────
def get_shap_values(
    model: torch.nn.Module,
    X_background: torch.Tensor,
    X_query: torch.Tensor,
    n_samples: int = 60,
) -> np.ndarray:
    """
    Compute SHAP values using GradientExplainer.

    Returns:
        sv : (N, D) numpy array for positive class
    """
    model.eval()
    sv = shap.GradientExplainer(model, X_background).shap_values(
        X_query, nsamples=n_samples
    )
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


# ── Logic Consistency Index ───────────────────────────────────────────────
def compute_lci(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    X_background: torch.Tensor,
    X_query: torch.Tensor,
    k: int = 10,
) -> tuple[float, float, float]:
    """
    Compute the Logic Consistency Index between teacher and student.

    LCI = 0.5 · ρ_SHAP + 0.5 · TopK_Agreement

    Returns:
        shap_corr    : mean per-feature Pearson correlation of SHAP values
        topk_agree   : |Top-k_T ∩ Top-k_S| / k
        lci          : combined score
    """
    sv_t = get_shap_values(teacher, X_background, X_query)
    sv_s = get_shap_values(student, X_background, X_query)

    corrs = [
        float(np.corrcoef(sv_t[:, j], sv_s[:, j])[0, 1])
        for j in range(sv_t.shape[1])
        if sv_t[:, j].std() > 1e-8 and sv_s[:, j].std() > 1e-8
    ]
    shap_corr = float(np.mean(corrs)) if corrs else 0.0

    top_t = set(np.argsort(-np.abs(sv_t).mean(0))[:k])
    top_s = set(np.argsort(-np.abs(sv_s).mean(0))[:k])
    topk_agree = len(top_t & top_s) / k

    lci = 0.5 * shap_corr + 0.5 * topk_agree
    return round(shap_corr, 4), round(topk_agree, 4), round(lci, 4)


# ── Centered Kernel Alignment ─────────────────────────────────────────────
def compute_cka(
    Z1: np.ndarray,
    Z2: np.ndarray,
) -> float:
    """
    Linear CKA between representation matrices Z1, Z2 (shape: N × D).
    CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F · ||Y^T Y||_F)
    """
    def _center(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K1 = _center(Z1 @ Z1.T)
    K2 = _center(Z2 @ Z2.T)
    num   = np.linalg.norm(K1 * K2, "fro") ** 2
    denom = np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")
    return round(float(num / (denom + 1e-10)), 6)


# ── Accuracy ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
) -> float:
    """Return classification accuracy over a DataLoader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x).argmax(1))
        ys.append(y)
    return round(
        accuracy_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()), 4
    )
