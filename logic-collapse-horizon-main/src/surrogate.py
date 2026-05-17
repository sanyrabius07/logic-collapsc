"""
surrogate.py — Logic Collapse Horizon
=======================================
FastSHAP surrogate model: a lightweight MLP trained to approximate
the teacher's SHAP attribution vectors at training speed.

This avoids calling GradientSHAP inside the LoRA-SHAP training loop
(which would be O(N × 2^d)) and instead uses a pre-trained surrogate
g_θ(x) ≈ φ(f_T, x) for fast SHAP alignment.

Usage:
    from src.surrogate import FastSHAPSurrogate, train_surrogate, predict_shap
    surrogate = train_surrogate(teacher, X_background, epochs=40)
    shap_approx = predict_shap(surrogate, X_batch)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import shap


class FastSHAPSurrogate(nn.Module):
    """
    3-layer MLP that maps input x → approximate SHAP attribution vector.

    Architecture: Linear(d→h) → ReLU → Linear(h→h) → ReLU → Linear(h→d)

    Args:
        in_dim : input feature dimension (must match teacher)
        h      : hidden layer width (default 128)
    """

    def __init__(self, in_dim: int, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h),      nn.ReLU(),
            nn.Linear(h, in_dim),
        )
        self.sv_std: float = 1.0   # set during training, used for denormalisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _compute_shap(
    model: torch.nn.Module,
    X_np: np.ndarray,
    n_bg: int = 50,
) -> np.ndarray:
    """Compute GradientSHAP values for X_np using model."""
    model.eval()
    bg  = torch.tensor(X_np[:n_bg],  dtype=torch.float32)
    inp = torch.tensor(X_np[:300],   dtype=torch.float32)
    sv  = shap.GradientExplainer(model, bg).shap_values(inp)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv.astype(np.float32)


def train_surrogate(
    teacher: nn.Module,
    X_np: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    h: int = 128,
    device: str = "cpu",
) -> FastSHAPSurrogate:
    """
    Train a FastSHAP surrogate to approximate teacher SHAP attributions.

    Args:
        teacher : trained teacher model (frozen)
        X_np    : numpy array of training samples used for SHAP computation
        epochs  : training epochs (default 40)
        lr      : Adam learning rate (default 1e-3)
        h       : hidden layer width (default 128)
        device  : "cpu" or "cuda"

    Returns:
        Trained FastSHAPSurrogate with sv_std attribute set for denormalisation
    """
    in_dim = X_np.shape[1]
    print("  Computing teacher SHAP values for surrogate training...")
    sv     = _compute_shap(teacher, X_np)

    surrogate = FastSHAPSurrogate(in_dim, h=h).to(device)
    opt       = torch.optim.Adam(surrogate.parameters(), lr=lr)

    Xt  = torch.tensor(X_np[:300], dtype=torch.float32).to(device)
    SVt = torch.tensor(sv,         dtype=torch.float32).to(device)

    # Normalise targets for stable training
    std = float(SVt.std().item()) + 1e-8
    SVn = SVt / std
    surrogate.sv_std = std

    ds = DataLoader(TensorDataset(Xt, SVn), batch_size=64, shuffle=True)
    best_loss, best_state = 1e9, None

    for ep in range(1, epochs + 1):
        surrogate.train()
        epoch_loss = 0.0
        for xb, sb in ds:
            opt.zero_grad()
            loss = F.mse_loss(surrogate(xb), sb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(ds)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in surrogate.state_dict().items()}
        if ep % 10 == 0:
            print(f"    surrogate ep{ep:02d}  mse={avg:.5f}")

    surrogate.load_state_dict(best_state)
    surrogate.eval()
    print(f"  ✅ Surrogate trained  best_mse={best_loss:.5f}  sv_std={std:.4f}")
    return surrogate


def predict_shap(
    surrogate: FastSHAPSurrogate,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Predict SHAP attribution values using the trained surrogate.
    Returns denormalised attribution tensor (same scale as GradientSHAP output).

    Args:
        surrogate : trained FastSHAPSurrogate
        X         : input tensor [N × d]

    Returns:
        approx SHAP values [N × d] (denormalised)
    """
    surrogate.eval()
    with torch.no_grad():
        return surrogate(X) * surrogate.sv_std
