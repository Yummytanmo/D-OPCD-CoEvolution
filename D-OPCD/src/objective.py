from __future__ import annotations

import torch


def distillation_loss_per_sample(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    current_latents: torch.Tensor,
    timestep: float,
    loss_type: str,
) -> torch.Tensor:
    """Return one D-OPCD loss value per example in the leading batch dimension."""
    if loss_type == "velocity":
        error = (
            student_prediction.float() - teacher_prediction.detach().float()
        ).square()
    elif loss_type == "endpoint":
        teacher_endpoint = current_latents + (1.0 - timestep) * teacher_prediction
        student_endpoint = current_latents + (1.0 - timestep) * student_prediction
        error = (
            student_endpoint.float() - teacher_endpoint.detach().float()
        ).square()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
    if error.ndim == 0:
        return error.reshape(1)
    if error.ndim == 1:
        return error
    return error.mean(dim=tuple(range(1, error.ndim)))


def distillation_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    current_latents: torch.Tensor,
    timestep: float,
    loss_type: str,
) -> torch.Tensor:
    """On-policy diffusion distillation with velocity or clean-endpoint loss."""
    return distillation_loss_per_sample(
        student_prediction,
        teacher_prediction,
        current_latents,
        timestep,
        loss_type,
    ).mean()
