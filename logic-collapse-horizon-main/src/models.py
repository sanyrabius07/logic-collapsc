"""
Models — Logic Collapse Horizon

Architectures:
  ResidualMLP   : Teacher model (ResBlock × 2, hidden=128)
  LoRALinear    : LoRA adapter wrapper for any nn.Linear
  TinyMLP       : KD student model (2-layer, 3× smaller)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Linear → BN → ReLU → Dropout → Linear → BN + skip."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class ResidualMLP(nn.Module):
    """
    Teacher model: Input → Linear(d→h) → ResBlock → ResBlock → Linear(h→nc)

    Args:
        in_dim: number of input features
        n_classes: number of output classes
        hidden: hidden dimension (default 128)
    """

    def __init__(self, in_dim: int, n_classes: int = 2, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.r1   = ResBlock(hidden)
        self.r2   = ResBlock(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.proj(x))
        z = self.r1(z)
        z = self.r2(z)
        return self.head(z)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate representation Z (before classification head)."""
        z = F.relu(self.proj(x))
        z = self.r1(z)
        z = self.r2(z)
        return z


class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping a frozen nn.Linear layer.
    W_effective = W_frozen + A @ B   (rank-r decomposition)

    Args:
        base_layer: original nn.Linear to wrap
        r: LoRA rank (default 4)
    """

    def __init__(self, base_layer: nn.Linear, r: int = 4):
        super().__init__()
        d_out, d_in = base_layer.out_features, base_layer.in_features
        self.base = copy.deepcopy(base_layer)
        for p in self.base.parameters():
            p.requires_grad_(False)   # freeze base
        self.A = nn.Parameter(torch.randn(d_out, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base.weight + self.A @ self.B, self.base.bias)


def inject_lora(model: ResidualMLP, r: int = 4) -> ResidualMLP:
    """
    Replace all Linear layers in a ResidualMLP with LoRALinear adapters.
    Returns a new model with frozen base weights and trainable A, B.
    """
    m = copy.deepcopy(model)
    lora_targets = [
        "proj", "r1.net.0", "r1.net.4",
        "r2.net.0", "r2.net.4", "head"
    ]
    for path in lora_targets:
        parts  = path.split(".")
        parent = m
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        if isinstance(orig, nn.Linear):
            setattr(parent, parts[-1], LoRALinear(orig, r))
    return m


class TinyMLP(nn.Module):
    """
    Knowledge Distillation student — 3× smaller than ResidualMLP.
    Input → Linear(d→32) → ReLU → Dropout → Linear(32→16) → ReLU → Linear(16→nc)
    """

    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),     nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x
