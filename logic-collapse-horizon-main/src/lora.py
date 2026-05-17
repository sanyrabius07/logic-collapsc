"""
lora.py — Logic Collapse Horizon
==================================
LoRA (Low-Rank Adaptation) injection utilities.

Provides:
  LoRALinear  : drop-in replacement for nn.Linear with trainable A·B adapters
  inject_lora : recursively wraps all nn.Linear layers in a model with LoRALinear
  lora_param_count : counts trainable vs total parameters after injection

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Replaces nn.Linear with a frozen base weight + trainable low-rank adapter.

    Forward pass:
        y = x @ (W_0 + A @ B)^T + b
        where W_0 is frozen, A ∈ R^{d×r}, B ∈ R^{r×k} are trainable.

    Args:
        original_layer : the nn.Linear layer to wrap
        r              : LoRA rank (default 4)
    """

    def __init__(self, original_layer: nn.Linear, r: int = 4):
        super().__init__()
        d, k = original_layer.out_features, original_layer.in_features

        # Freeze base weights
        self.base_weight = nn.Parameter(
            original_layer.weight.data.clone(), requires_grad=False
        )
        self.base_bias = (
            nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
            if original_layer.bias is not None
            else None
        )

        # Trainable low-rank adapters
        # A initialised with small Gaussian, B initialised to zero
        # so that A@B = 0 at init (preserves teacher behaviour at start)
        self.lora_A = nn.Parameter(torch.randn(d, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, k))
        self.r = r

    @property
    def effective_weight(self) -> torch.Tensor:
        return self.base_weight + self.lora_A @ self.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight, self.base_bias)

    def extra_repr(self) -> str:
        d, k = self.base_weight.shape
        return f"in={k}, out={d}, rank={self.r}"


def inject_lora(model: nn.Module, r: int = 4) -> nn.Module:
    """
    Recursively replace all nn.Linear layers in model with LoRALinear.
    Returns a deep copy — original model is not modified.

    Args:
        model : any nn.Module (teacher)
        r     : LoRA rank

    Returns:
        model copy with LoRA adapters injected
        Only lora_A and lora_B parameters require grad.

    Example:
        lora_model = inject_lora(teacher, r=4)
        trainable  = [p for p in lora_model.parameters() if p.requires_grad]
    """
    import copy
    model = copy.deepcopy(model)
    _replace_linear(model, r)
    return model


def _replace_linear(module: nn.Module, r: int) -> None:
    """In-place recursive replacement of nn.Linear → LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r))
        else:
            _replace_linear(child, r)


def lora_param_count(model: nn.Module) -> dict:
    """
    Count trainable (LoRA) vs total parameters.

    Returns:
        dict with keys: trainable, frozen, total, compression_ratio
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    return {
        "trainable":        trainable,
        "frozen":           frozen,
        "total":            total,
        "compression_ratio": round(trainable / total, 4),
    }
