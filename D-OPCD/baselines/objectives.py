from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square()
    return error.reshape(error.shape[0], -1).mean(dim=1)


def flow_dpo_loss_per_sample(
    model_losses: torch.Tensor,
    reference_losses: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pair flow-DPO loss and its implicit preference logit.

    Both loss vectors are ordered as all chosen examples followed by all
    rejected examples, matching the reference Z-Image flow-DPO prototype.
    """
    if model_losses.ndim != 1 or reference_losses.ndim != 1:
        raise ValueError("model/reference losses must be one-dimensional")
    if model_losses.shape != reference_losses.shape or model_losses.numel() % 2:
        raise ValueError("model/reference losses must be equal-length chosen+rejected vectors")
    if beta <= 0:
        raise ValueError("beta must be positive")
    model_chosen, model_rejected = model_losses.chunk(2)
    reference_chosen, reference_rejected = reference_losses.chunk(2)
    model_difference = model_chosen - model_rejected
    reference_difference = reference_chosen - reference_rejected
    implicit_logit = -0.5 * beta * (model_difference - reference_difference)
    return -F.logsigmoid(implicit_logit), implicit_logit

