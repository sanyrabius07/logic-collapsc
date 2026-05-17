"""
splra.py — Logic Collapse Horizon
===================================
SPLRA: Sensitivity-Profiled Low-Rank Allocation

The first rank allocation strategy that profiles layers by their
SHAP explanation sensitivity rather than task-accuracy criteria.

Key insight (Paper Section 5.4):
  Attribution-critical capacity is systematically localised in the
  input-proximal layers of IDS classifiers. SPLRA allocates higher
  LoRA rank to layers whose weight perturbations most strongly shift
  SHAP attributions, protecting explanation fidelity where it matters.

Usage:
    from src.splra import compute_sensitivity_scores, allocate_ranks
    scores = compute_sensitivity_scores(teacher, X_sample)
    ranks  = allocate_ranks(scores, rank_budget=16)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List


# ── Sensitivity scoring ────────────────────────────────────────────────────

def _get_named_linear_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """Return {name: layer} for all nn.Linear (or LoRALinear) in model."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module
        # support LoRALinear wrapper
        elif hasattr(module, 'base_weight') and hasattr(module, 'lora_A'):
            layers[name] = module
    return layers


def _perturb_layer(
    model: nn.Module,
    layer_name: str,
    sigma: float = 0.01,
) -> nn.Module:
    """
    Return a deep copy of model with Gaussian noise added to the
    weights of layer `layer_name`.

    sigma = 0.01 matches typical 4-bit quantisation noise magnitude
    (as used in the paper's sensitivity profiling).
    """
    perturbed = copy.deepcopy(model)
    for name, module in perturbed.named_modules():
        if name == layer_name:
            if hasattr(module, 'base_weight'):
                module.base_weight.data += torch.randn_like(
                    module.base_weight.data
                ) * sigma
            elif isinstance(module, nn.Linear):
                module.weight.data += torch.randn_like(
                    module.weight.data
                ) * sigma
    return perturbed


def _shap_l1_distance(
    model_a: nn.Module,
    model_b: nn.Module,
    X: torch.Tensor,
) -> float:
    """
    Approximate SHAP attribution distance between two models
    using gradient × input as a fast proxy.

    Returns mean L1 distance over the sample batch.
    """
    def _gi(m, x):
        m.eval()
        xc = x.clone().float().detach().requires_grad_(True)
        m(xc)[:, 1].sum().backward()
        return (xc * xc.grad).detach()

    with torch.enable_grad():
        phi_a = _gi(model_a, X)
        phi_b = _gi(model_b, X)

    return float((phi_a - phi_b).abs().mean().item())


def compute_sensitivity_scores(
    teacher: nn.Module,
    X_sample: torch.Tensor,
    sigma: float = 0.01,
    n_trials: int = 3,
) -> Dict[str, float]:
    """
    Compute explanation sensitivity score s_l for each linear layer l.

    s_l = E_{perturbation} [ || phi_T^{delta_l}(x) - phi_T(x) ||_1 ]

    Higher score → layer l has greater influence on SHAP attributions
    → should receive higher LoRA rank allocation.

    Args:
        teacher  : trained teacher model (read-only)
        X_sample : (N, D) tensor used for attribution comparison
        sigma    : perturbation noise std (default 0.01)
        n_trials : number of perturbation trials per layer (default 3)

    Returns:
        dict {layer_name: sensitivity_score}
    """
    teacher.eval()
    layer_names = list(_get_named_linear_layers(teacher).keys())
    scores: Dict[str, float] = {}

    for name in layer_names:
        trial_scores = []
        for _ in range(n_trials):
            perturbed = _perturb_layer(teacher, name, sigma=sigma)
            s = _shap_l1_distance(teacher, perturbed, X_sample)
            trial_scores.append(s)
        scores[name] = float(np.mean(trial_scores))
        print(f"    sensitivity  {name:<40s}  s={scores[name]:.6f}")

    return scores


def allocate_ranks(
    sensitivity_scores: Dict[str, float],
    rank_budget: int = 16,
    min_rank: int = 1,
) -> Dict[str, int]:
    """
    Allocate LoRA rank to each layer proportional to its sensitivity score.

    Allocation rule (Equation 10 in the paper):
        r_l = max(min_rank, round(rank_budget * s_l / sum(s)))

    Scores are normalised to a probability simplex before allocation.
    Every layer receives at least min_rank to ensure coverage.

    Args:
        sensitivity_scores : dict from compute_sensitivity_scores()
        rank_budget        : total rank budget across all layers (default 16)
        min_rank           : minimum rank per layer (default 1)

    Returns:
        dict {layer_name: allocated_rank}
    """
    names  = list(sensitivity_scores.keys())
    values = np.array([sensitivity_scores[n] for n in names], dtype=float)

    # Normalise to probability simplex
    total = values.sum()
    if total < 1e-12:
        # Uniform fallback if all sensitivities are near zero
        probs = np.ones(len(values)) / len(values)
    else:
        probs = values / total

    raw_ranks = np.round(probs * rank_budget).astype(int)
    ranks     = np.maximum(raw_ranks, min_rank)

    rank_dict = {name: int(r) for name, r in zip(names, ranks)}

    print("  [SPLRA] Rank allocation:")
    for name, r in rank_dict.items():
        bar = '|' * r
        print(f"    {name:<40s}  r={r:2d}  {bar}")

    return rank_dict


def inject_lora_with_splra(
    teacher: nn.Module,
    rank_dict: Dict[str, int],
) -> nn.Module:
    """
    Inject LoRA adapters into a teacher copy using per-layer rank
    allocation from SPLRA instead of uniform rank.

    Args:
        teacher   : teacher model to copy and inject
        rank_dict : {layer_name: rank} from allocate_ranks()

    Returns:
        Deep copy of teacher with heterogeneous-rank LoRA adapters.
    """
    from .lora import LoRALinear
    model = copy.deepcopy(teacher)

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}".lstrip(".")
            if isinstance(child, nn.Linear) and full_name in rank_dict:
                r = rank_dict[full_name]
                setattr(module, child_name, LoRALinear(child, r=r))
            else:
                _replace(child, prefix=full_name)

    _replace(model)
    return model
