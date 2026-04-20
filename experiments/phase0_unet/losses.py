"""Loss functions for Phase-0 UNet.

Total training loss = BCE + soft_Dice + soft_clDice (unit weights).

  • BCE with logits — standard; tolerates soft targets.
  • soft Dice — per-channel, mean-averaged, ε-smoothed. Handles soft targets.
  • soft clDice — topology-aware centreline Dice (Shit et al. 2021, CVPR).
    Uses iterative soft morphological skeletonisation so the whole term
    stays differentiable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---- Dice ----


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
                   ) -> torch.Tensor:
    """1 − Dice averaged over channels. `probs` and `target` in [0, 1]."""
    dims = (0, 2, 3)
    num = 2.0 * (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims) + eps
    return 1.0 - (num / den).mean()


# ---- soft clDice (Shit 2021) ----
#
# Idea: the morphological skeleton is thinning iterated to convergence.
# Replace hard min/max with differentiable max-pool approximations so the
# centreline extraction is end-to-end differentiable.


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    # min over 3×3 neighbourhood = −max_pool(−x)
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skel(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Iterative differentiable skeletonisation."""
    img1 = _soft_open(x)
    skel = F.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x)
        img1 = _soft_open(x)
        delta = F.relu(x - img1)
        skel = skel + F.relu(delta - delta * skel)
    return skel


def soft_cldice_loss(probs: torch.Tensor, target: torch.Tensor,
                     iters: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """1 − centreline Dice. Topology-aware; penalises broken skeletons.

    `probs` and `target` in [0, 1], shape (B, C, H, W). Soft-skel is run
    per channel independently.
    """
    sk_p = _soft_skel(probs, iters)
    sk_t = _soft_skel(target, iters)
    # Per-channel t-prec and t-sens, averaged.
    dims = (0, 2, 3)
    tprec = (sk_p * target).sum(dim=dims) / (sk_p.sum(dim=dims) + eps)
    tsens = (sk_t * probs).sum(dim=dims) / (sk_t.sum(dim=dims) + eps)
    cld = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cld.mean()


# ---- Combined ----


def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cldice_iters: int = 5,
    cldice_weight: float = 1.0,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (scalar_loss, per-component-dict) for logging."""
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(probs, targets)
    cld = soft_cldice_loss(probs, targets, iters=cldice_iters) if cldice_weight > 0 else \
        torch.tensor(0.0, device=logits.device)
    total = bce_weight * bce + dice_weight * dice + cldice_weight * cld
    return total, {
        "loss_total": float(total.item()),
        "loss_bce": float(bce.item()),
        "loss_dice": float(dice.item()),
        "loss_cldice": float(cld.item()) if cldice_weight > 0 else 0.0,
    }
