from __future__ import annotations

import torch


def _marker(adapter_name: str) -> str:
    return f".{adapter_name}."


def paired_adapter_parameter_names(model, source_adapter: str, target_adapter: str):
    parameters = dict(model.named_parameters())
    source_marker = _marker(source_adapter)
    target_marker = _marker(target_adapter)
    for source_name in sorted(parameters):
        if source_marker not in source_name:
            continue
        target_name = source_name.replace(source_marker, target_marker)
        if target_name in parameters:
            yield source_name, target_name


def set_adapter_trainable(model, adapter_name: str, trainable: bool) -> None:
    found = False
    for name, parameter in model.named_parameters():
        if _marker(adapter_name) in name:
            parameter.requires_grad = trainable
            found = True
    if not found:
        raise ValueError(f"No parameters found for adapter {adapter_name!r}")


@torch.no_grad()
def copy_adapter(model, source_adapter: str, target_adapter: str) -> int:
    parameters = dict(model.named_parameters())
    copied = 0
    for source_name, target_name in paired_adapter_parameter_names(
        model, source_adapter, target_adapter
    ):
        parameters[target_name].copy_(parameters[source_name])
        copied += 1
    if copied == 0:
        raise ValueError(f"No paired parameters for {source_adapter!r} -> {target_adapter!r}")
    return copied


@torch.no_grad()
def ema_update_adapter(model, source_adapter: str, target_adapter: str, decay: float) -> int:
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
    parameters = dict(model.named_parameters())
    updated = 0
    for source_name, target_name in paired_adapter_parameter_names(
        model, source_adapter, target_adapter
    ):
        parameters[target_name].mul_(decay).add_(parameters[source_name], alpha=1.0 - decay)
        updated += 1
    if updated == 0:
        raise ValueError(f"No paired parameters for {source_adapter!r} -> {target_adapter!r}")
    return updated


@torch.no_grad()
def adapter_l2_norm(model, adapter_name: str) -> torch.Tensor:
    squared = None
    for name, parameter in model.named_parameters():
        if _marker(adapter_name) not in name:
            continue
        value = torch.linalg.vector_norm(parameter.detach().float()).square()
        squared = value if squared is None else squared + value
    if squared is None:
        raise ValueError(f"No parameters found for adapter {adapter_name!r}")
    return squared.sqrt()


@torch.no_grad()
def adapter_gap_l2_norm(model, source_adapter: str, target_adapter: str) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    squared = None
    for source_name, target_name in paired_adapter_parameter_names(
        model, source_adapter, target_adapter
    ):
        value = torch.linalg.vector_norm(
            parameters[source_name].detach().float()
            - parameters[target_name].detach().float()
        ).square()
        squared = value if squared is None else squared + value
    if squared is None:
        raise ValueError(f"No paired parameters for {source_adapter!r} -> {target_adapter!r}")
    return squared.sqrt()
