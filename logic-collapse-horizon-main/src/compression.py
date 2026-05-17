"""
Compression Pipelines — Logic Collapse Horizon

Pipelines:
  train_lora_shap    : SHAP-guided LoRA fine-tuning (LoRA-SHAP)
  train_vanilla_lora : Standard LoRA fine-tuning (baseline)
  train_kd_student   : Knowledge Distillation (TinyMLP student)
  apply_pruning      : Magnitude-based unstructured pruning + fine-tune
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import gi_eval


BETA = 0.5   # SHAP regularisation weight in LoRA-SHAP objective


# ── LoRA-SHAP ─────────────────────────────────────────────────────────────
def train_lora_shap(
    lora_model: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    Xbg: torch.Tensor,
    Xq: torch.Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    beta: float = BETA,
    shap_start_epoch: int = 5,
    gi_batch: int = 80,
    gi_samples: int = 20,
) -> nn.Module:
    """
    Train a LoRA-wrapped model with SHAP-guided regularisation.

    Objective:
        L = L_CE(f_S(x), y)  +  β · MSE(GI_S(x), GI_T(x))

    The SHAP term aligns the student's Gradient×Input attribution map
    with the teacher's, explicitly preserving explanation fidelity.

    Args:
        lora_model       : inject_lora(teacher) — model with LoRA adapters
        teacher          : frozen teacher for GI reference
        train_loader     : training DataLoader
        val_loader       : validation DataLoader
        Xbg, Xq          : SHAP background and query tensors
        epochs           : number of training epochs
        lr               : Adam learning rate
        beta             : SHAP regularisation weight
        shap_start_epoch : epoch from which to apply SHAP loss
        gi_batch         : number of query samples per GI computation
        gi_samples       : kept for API compatibility

    Returns:
        best model (by validation accuracy)
    """
    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    # Pre-compute teacher GI reference
    teacher.eval()
    gi_target = gi_eval(teacher, Xq[:gi_batch]).detach()

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        lora_model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(lora_model(xb), yb)

            if ep >= shap_start_epoch:
                lora_model.eval()
                gi_s  = gi_eval(lora_model, Xq[:gi_batch])
                shap_reg = F.mse_loss(gi_s, gi_target)
                lora_model.train()
                loss = loss + beta * shap_reg

            loss.backward()
            opt.step()

        sched.step()
        from .metrics import evaluate_accuracy
        v = evaluate_accuracy(lora_model, val_loader)
        if v > best_acc:
            best_acc   = v
            best_state = {k: val.clone() for k, val in lora_model.state_dict().items()}

    lora_model.load_state_dict(best_state)
    return lora_model


# ── VanillaLoRA ───────────────────────────────────────────────────────────
def train_vanilla_lora(
    lora_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
) -> nn.Module:
    """Standard LoRA fine-tuning without any explanation guidance."""
    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        lora_model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            crit(lora_model(xb), yb).backward()
            opt.step()
        sched.step()
        from .metrics import evaluate_accuracy
        v = evaluate_accuracy(lora_model, val_loader)
        if v > best_acc:
            best_acc   = v
            best_state = {k: val.clone() for k, val in lora_model.state_dict().items()}

    lora_model.load_state_dict(best_state)
    return lora_model


# ── Knowledge Distillation ────────────────────────────────────────────────
def train_kd_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    alpha: float = 0.7,
) -> nn.Module:
    """
    Train a student via Knowledge Distillation.

    Objective:
        L = α · MSE(logits_S, logits_T) + (1−α) · L_CE(logits_S, y)

    Args:
        alpha : weight on KD term (default 0.7)
    """
    opt   = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for _ in range(1, epochs + 1):
        student.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits_s = student(xb)
            logits_t = teacher(xb).detach()
            loss = alpha * F.mse_loss(logits_s, logits_t) + (1 - alpha) * crit(logits_s, yb)
            loss.backward()
            opt.step()
        from .metrics import evaluate_accuracy
        v = evaluate_accuracy(student, val_loader)
        if v > best_acc:
            best_acc   = v
            best_state = {k: val.clone() for k, val in student.state_dict().items()}

    student.load_state_dict(best_state)
    return student


# ── Magnitude Pruning ─────────────────────────────────────────────────────
def apply_pruning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sparsity: float = 0.70,
    finetune_epochs: int = 7,
    lr: float = 5e-4,
) -> nn.Module:
    """
    Unstructured magnitude pruning: zero out the lowest-magnitude weights,
    then fine-tune to recover accuracy.

    Args:
        sparsity       : fraction of weights to zero (default 0.70)
        finetune_epochs: number of fine-tuning epochs after pruning
    """
    pruned = copy.deepcopy(model)
    for name, param in pruned.named_parameters():
        if "weight" in name:
            threshold = torch.quantile(param.data.abs().flatten(), sparsity)
            param.data.mul_((param.data.abs() >= threshold).float())

    opt  = torch.optim.Adam(pruned.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for _ in range(1, finetune_epochs + 1):
        pruned.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            crit(pruned(xb), yb).backward()
            opt.step()
        from .metrics import evaluate_accuracy
        v = evaluate_accuracy(pruned, val_loader)
        if v > best_acc:
            best_acc   = v
            best_state = {k: val.clone() for k, val in pruned.state_dict().items()}

    pruned.load_state_dict(best_state)
    return pruned
