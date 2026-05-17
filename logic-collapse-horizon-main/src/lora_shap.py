"""
lora_shap.py — Logic Collapse Horizon
=======================================
LoRA-SHAP: the first LoRA-based compression algorithm for tabular IDS
classifiers that treats SHAP attribution fidelity as a first-class
training objective.

Key components:
  - Attribution cosine loss (L_ar) penalises divergence from teacher SHAP
  - Delayed activation: attribution loss activates at epoch e_start
    to allow cross-entropy to converge first (prevents gradient conflict)
  - Uses pre-trained FastSHAPSurrogate for tractable attribution at scale

Reference: Algorithm 1 in the paper (Section 5).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .lora      import inject_lora
from .surrogate import train_surrogate, predict_shap
from .metrics   import evaluate_accuracy


# ── Attribution cosine loss ───────────────────────────────────────────────

def attribution_cosine_loss(
    phi_student: torch.Tensor,
    phi_surrogate: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cosine dissimilarity between student attribution and surrogate target.

    L_ar = 1 - mean_i [ cos(phi_S(x_i), g_theta(x_i)) ]

    Args:
        phi_student   : (N, D) student attribution tensor
        phi_surrogate : (N, D) surrogate target tensor
        eps           : numerical stability term

    Returns:
        scalar loss tensor
    """
    s_norm = F.normalize(phi_student,   dim=1, eps=eps)
    g_norm = F.normalize(phi_surrogate, dim=1, eps=eps)
    return 1.0 - (s_norm * g_norm).sum(dim=1).mean()


# ── Gradient-sensitivity attribution approximation ───────────────────────

def grad_sensitivity(
    model: nn.Module,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate per-sample SHAP attribution via gradient x input.

    phi_hat_S(x) = x * grad_{x} f_S(x)[class=1]

    Args:
        model : student model (LoRA-injected)
        X     : (N, D) float tensor

    Returns:
        phi_hat : (N, D) attribution tensor (detached)
    """
    model.train()           # keep BN running stats live
    Xc = X.clone().float().detach().requires_grad_(True)
    logits = model(Xc)
    logits[:, 1].sum().backward()
    return (Xc * Xc.grad).detach()


# ── LoRA-SHAP training ────────────────────────────────────────────────────

def train_lora_shap(
    teacher:     nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    X_bg_np,                    # numpy array for surrogate training
    rank:        int   = 4,
    beta:        float = 0.1,
    epochs:      int   = 25,
    e_start:     int   = 15,
    lr:          float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip:   float = 0.5,
    device:      str   = "cpu",
) -> nn.Module:
    """
    Train a LoRA-SHAP compressed student from a frozen teacher.

    Algorithm (matches Algorithm 1 in the paper):
      1. Inject rank-r LoRA adapters into all linear layers; freeze base weights
      2. Train FastSHAP surrogate g_theta once on 300 teacher SHAP samples
      3. For each epoch:
           a. Compute cross-entropy loss L_CE
           b. After e_start: compute L_ar = attribution_cosine_loss(phi_hat_S, g_theta(x))
           c. L = L_CE + beta * L_ar  (if epoch >= e_start, else L = L_CE)
           d. Update LoRA parameters via AdamW with gradient clipping

    Args:
        teacher      : trained teacher model (not modified)
        train_loader : DataLoader for training set
        val_loader   : DataLoader for validation set
        X_bg_np      : numpy array (N_bg, D) for surrogate SHAP computation
        rank         : LoRA rank r (default 4)
        beta         : attribution loss weight (default 0.1)
        epochs       : total training epochs (default 25)
        e_start      : epoch at which attribution loss is activated (default 15)
        lr           : AdamW learning rate (default 3e-4)
        weight_decay : AdamW weight decay (default 1e-4)
        grad_clip    : gradient clipping norm (default 0.5)
        device       : 'cpu' or 'cuda'

    Returns:
        Trained student model (LoRA adapters fine-tuned)
    """
    # Step 1: inject LoRA into a deep copy of the teacher
    student = inject_lora(copy.deepcopy(teacher), r=rank).to(device)

    # Step 2: train surrogate once, then freeze
    print("  [LoRA-SHAP] Training FastSHAP surrogate...")
    surrogate = train_surrogate(teacher, X_bg_np, epochs=40, device=device)
    surrogate.to(device).eval()

    # Optimiser over LoRA adapter params only
    lora_params = [p for p in student.parameters() if p.requires_grad]
    optimiser   = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion   = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        student.train()
        total_lce = total_lar = n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            optimiser.zero_grad()

            # Classification loss
            logits = student(X_batch)
            l_ce   = criterion(logits, y_batch)

            # Attribution loss (delayed)
            if epoch >= e_start:
                phi_hat  = grad_sensitivity(student, X_batch)
                phi_surr = predict_shap(surrogate, X_batch).detach()
                l_ar     = attribution_cosine_loss(phi_hat, phi_surr)
                loss     = l_ce + beta * l_ar
                total_lar += l_ar.item()
            else:
                loss = l_ce

            loss.backward()
            nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimiser.step()

            total_lce += l_ce.item()
            n_batches += 1

        scheduler.step()
        val_acc = evaluate_accuracy(student, val_loader)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(student.state_dict())

        if epoch % 5 == 0:
            avg_lce = total_lce / n_batches
            avg_lar = total_lar / n_batches if epoch >= e_start else 0.0
            print(
                f"    ep{epoch:02d}  L_CE={avg_lce:.4f}  "
                f"L_ar={avg_lar:.4f}  val_acc={val_acc:.4f}"
            )

    student.load_state_dict(best_state)
    student.eval()
    print(f"  [LoRA-SHAP] Done. Best val_acc={best_val:.4f}")
    return student
