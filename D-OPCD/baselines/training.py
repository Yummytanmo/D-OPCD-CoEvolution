#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Qwen2Tokenizer, Qwen3Model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import ensure_within, sha256_file, write_json  # noqa: E402
from prompt_encoder import encode_prompts  # noqa: E402

try:  # Package import in tests; file-local import under ``accelerate launch``.
    from .data import ContextImageDataset, collate_context_images
    from .objectives import flow_dpo_loss_per_sample, flow_matching_mse_per_sample
except ImportError:
    from data import ContextImageDataset, collate_context_images
    from objectives import flow_dpo_loss_per_sample, flow_matching_mse_per_sample


logger = get_logger(__name__)
BaselineMethod = Literal["sft", "flow_dpo"]
DEFAULT_LORA_LAYERS = [
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "attention.to_k",
    "attention.to_q",
    "attention.to_v",
    "attention.to_out.0",
]


def parse_args(method: BaselineMethod, input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Z-Image {method} baseline on D-OPCD context rows; student conditioning is q only."
    )
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--context_jsonl", required=True)
    parser.add_argument("--image_manifest_jsonl", required=True)
    parser.add_argument("--sample_weights_jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", default=",".join(DEFAULT_LORA_LAYERS))
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--tracker_project_name", default="dopcd-zimage-baselines")
    parser.add_argument("--tracker_run_name", default=None)
    parser.add_argument("--tracker_run_id", default=None)
    parser.add_argument("--wandb_init_timeout", type=int, default=300)
    parser.add_argument("--parameter_logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timestep_sampling",
        choices=["uniform", "logit_normal"],
        default="logit_normal",
    )
    parser.add_argument("--timestep_sampling_logit_mean", type=float, default=0.0)
    parser.add_argument("--timestep_sampling_logit_std", type=float, default=1.0)
    parser.add_argument("--num_train_timesteps_per_batch", type=int, default=1)
    parser.add_argument("--beta_dpo", type=float, default=2500.0)
    parser.add_argument("--random_horizontal_flip", action="store_true")
    args = parser.parse_args(input_args)
    args.method = method

    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    if args.max_sequence_length <= 0 or args.max_sequence_length > 1536:
        raise ValueError("max_sequence_length must be in [1, 1536]")
    if args.rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if args.train_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_train_steps <= 0 or args.checkpointing_steps <= 0:
        raise ValueError("training/checkpoint steps must be positive")
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit <= 0:
        raise ValueError("checkpoints_total_limit must be positive when set")
    if args.parameter_logging_steps <= 0 or args.wandb_init_timeout <= 0:
        raise ValueError("logging steps and W&B init timeout must be positive")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if args.num_train_timesteps_per_batch <= 0:
        raise ValueError("num_train_timesteps_per_batch must be positive")
    if args.timestep_sampling_logit_std <= 0:
        raise ValueError("timestep_sampling_logit_std must be positive")
    if args.beta_dpo <= 0:
        raise ValueError("beta_dpo must be positive")
    return args


def unwrap_model(accelerator: Accelerator, model):
    unwrapped = accelerator.unwrap_model(model)
    return unwrapped._orig_mod if is_compiled_module(unwrapped) else unwrapped


def transformer_forward(model, latents: torch.Tensor, timesteps: torch.Tensor, prompt_embeds):
    latent_list = list(latents.unsqueeze(2).unbind(dim=0))
    predictions = model(latent_list, timesteps, prompt_embeds, return_dict=False)[0]
    return -torch.stack(predictions, dim=0).squeeze(2)


def get_sigmas(
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def sample_timesteps(
    args: argparse.Namespace,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if args.timestep_sampling == "uniform":
        density = torch.rand(batch_size, device=device)
    else:
        density = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=args.timestep_sampling_logit_mean,
            logit_std=args.timestep_sampling_logit_std,
            mode_scale=1.29,
            device=device,
        )
    indices = (density * noise_scheduler.config.num_train_timesteps).long()
    indices = indices.clamp(max=noise_scheduler.config.num_train_timesteps - 1)
    return noise_scheduler.timesteps.to(device=device)[indices]


def encode_images(
    vae: AutoencoderKL,
    pixel_values: torch.Tensor,
    shift_factor: float,
    scaling_factor: float,
) -> torch.Tensor:
    latents = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.mode()
    return (latents - shift_factor) * scaling_factor


def find_resume_checkpoint(output_dir: Path, requested: str | None) -> Path | None:
    if not requested:
        return None
    if requested != "latest":
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate = candidate.resolve()
        return candidate if candidate.is_dir() and (candidate / "checkpoint_complete.json").is_file() else None
    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        if not candidate.is_dir() or not (candidate / "checkpoint_complete.json").is_file():
            continue
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, candidate))
    return max(checkpoints, default=(None, None))[1]


def build_tracker_init_kwargs(args: argparse.Namespace) -> dict | None:
    if args.report_to != "wandb":
        return None
    wandb_init: dict = {
        "settings": {"init_timeout": args.wandb_init_timeout},
        "dir": str(Path(args.output_dir).expanduser().resolve() / "logs"),
    }
    if args.tracker_run_name:
        wandb_init["name"] = args.tracker_run_name
    if args.tracker_run_id:
        wandb_init["id"] = args.tracker_run_id
        if os.environ.get("WANDB_MODE", "online").lower() != "offline":
            wandb_init["resume"] = "allow"
    return {"wandb": wandb_init}


def configure_logging(accelerator: Accelerator, output_dir: Path) -> None:
    level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if accelerator.is_main_process:
        handler = logging.FileHandler(output_dir / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)


def adapter_l2_norm(model) -> float:
    state = get_peft_model_state_dict(model)
    squared = sum(value.detach().float().square().sum() for value in state.values())
    return float(squared.sqrt().item())


def prune_checkpoints(output_dir: Path, total_limit: int | None) -> None:
    if total_limit is None:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    checkpoints.sort()
    for _, path in checkpoints[: max(0, len(checkpoints) - total_limit + 1)]:
        shutil.rmtree(path)


def validate_training_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    training_root = (project_root / "data").resolve()
    context_path = ensure_within(args.context_jsonl, training_root)
    manifest_path = ensure_within(args.image_manifest_jsonl, training_root)
    if context_path.parent != manifest_path.parent:
        raise ValueError("context_jsonl and image_manifest_jsonl must belong to the same immutable data ID")
    if not context_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("context_jsonl and image_manifest_jsonl must exist")
    weights_path = None
    if args.sample_weights_jsonl:
        weights_path = ensure_within(args.sample_weights_jsonl, training_root)
        if weights_path.parent != context_path.parent or not weights_path.is_file():
            raise ValueError("sample_weights_jsonl must exist in the same immutable data ID")
    output_boundary = Path(
        os.environ.get("DOPCD_RESULTS_ROOT", str(project_root / "results"))
    ).expanduser().resolve()
    output_dir = ensure_within(args.output_dir, output_boundary)
    return context_path, manifest_path, weights_path, output_dir


def main(args: argparse.Namespace) -> None:
    context_path, manifest_path, weights_path, output_dir = validate_training_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.report_to == "wandb":
        for key, subdir in {
            "WANDB_DIR": "logs",
            "WANDB_CACHE_DIR": "wandb-cache",
            "WANDB_DATA_DIR": "wandb-data",
            "WANDB_ARTIFACT_DIR": "wandb-artifacts",
        }.items():
            directory = output_dir / subdir
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(directory)
    logging_dir = output_dir / args.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir),
    )
    configure_logging(accelerator, output_dir)
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        saved_args = vars(args).copy()
        saved_args.update(
            {
                "context_jsonl": str(context_path),
                "context_sha256": sha256_file(context_path),
                "image_manifest_jsonl": str(manifest_path),
                "image_manifest_sha256": sha256_file(manifest_path),
                "sample_weights_jsonl": str(weights_path) if weights_path else None,
                "sample_weights_sha256": sha256_file(weights_path) if weights_path else None,
                "student_context": "q",
                "privileged_prompt_usage": "offline_supervision_only",
            }
        )
        write_json(output_dir / "args.json", saved_args)
    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = str(Path(args.pretrained_model_name_or_path).expanduser().resolve())
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = Qwen3Model.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    transformer.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    if args.max_sequence_length > int(transformer.config.axes_lens[0]):
        raise ValueError(
            f"max_sequence_length={args.max_sequence_length} exceeds text-axis limit="
            f"{transformer.config.axes_lens[0]}"
        )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    target_modules = [value.strip() for value in args.lora_layers.split(",") if value.strip()]
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    if accelerator.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    dataset = ContextImageDataset(
        context_path,
        manifest_path,
        args.method,
        args.resolution,
        sample_weights_jsonl=weights_path,
        image_root=context_path.parent,
        random_horizontal_flip=args.random_horizontal_flip,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_context_images,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def save_model_hook(models, weights, checkpoint_dir):
        if accelerator.is_main_process:
            state = get_peft_model_state_dict(unwrap_model(accelerator, transformer))
            ZImagePipeline.save_lora_weights(checkpoint_dir, transformer_lora_layers=state)
            for _ in models:
                if weights:
                    weights.pop()

    def load_model_hook(models, checkpoint_dir):
        state = ZImagePipeline.lora_state_dict(
            checkpoint_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        transformer_state = {
            key.replace("transformer.", ""): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        transformer_state = convert_unet_state_dict_to_peft(transformer_state)
        incompatible = set_peft_model_state_dict(
            unwrap_model(accelerator, transformer), transformer_state, adapter_name="default"
        )
        if getattr(incompatible, "unexpected_keys", None):
            raise ValueError(f"Unexpected LoRA keys: {incompatible.unexpected_keys}")
        while models:
            models.pop()
        if accelerator.mixed_precision == "fp16":
            cast_training_params([unwrap_model(accelerator, transformer)], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    if len(dataloader) == 0:
        raise ValueError(
            "Prepared dataloader is empty; provide at least one full per-rank training batch"
        )
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / updates_per_epoch)

    if args.report_to != "none":
        accelerator.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs=build_tracker_init_kwargs(args),
        )

    resume_path = find_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    global_step = 0
    first_epoch = 0
    resume_batch = 0
    if resume_path is not None:
        accelerator.print(f"Resuming from {resume_path}")
        accelerator.load_state(str(resume_path))
        global_step = int(resume_path.name.rsplit("-", 1)[1])
        consumed_batches = global_step * args.gradient_accumulation_steps
        first_epoch = consumed_batches // len(dataloader)
        resume_batch = consumed_batches % len(dataloader)
    elif args.resume_from_checkpoint:
        accelerator.print(
            f"No complete checkpoint found for {args.resume_from_checkpoint!r}; starting at step 0"
        )

    if accelerator.is_main_process:
        logger.info("***** Running %s baseline *****", args.method)
        logger.info("Examples: %d", len(dataset))
        logger.info("Context routing: student=q; p=offline supervision provenance only")
        logger.info(
            "Global context-row batch: %d",
            args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        )
        logger.info("Trainable parameters: %d", sum(parameter.numel() for parameter in parameters))

    progress = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    loss_log_path = output_dir / "loss.jsonl"
    optimizer_step_started = None

    for epoch in range(first_epoch, num_epochs):
        active_dataloader = dataloader
        if epoch == first_epoch and resume_batch:
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_batch)
        transformer.train()
        for batch in active_dataloader:
            if optimizer_step_started is None:
                optimizer_step_started = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
            with accelerator.accumulate(transformer):
                with torch.no_grad(), accelerator.autocast():
                    prompt_embeds, prompt_lengths = encode_prompts(
                        text_encoder,
                        tokenizer,
                        batch["prompts"],
                        accelerator.device,
                        args.max_sequence_length,
                        dtype=weight_dtype,
                    )
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    if args.method == "sft":
                        model_input = encode_images(
                            vae,
                            pixel_values,
                            vae.config.shift_factor,
                            vae.config.scaling_factor,
                        )
                    else:
                        chosen_pixels = pixel_values[:, 0]
                        rejected_pixels = pixel_values[:, 1]
                        model_input = encode_images(
                            vae,
                            torch.cat([chosen_pixels, rejected_pixels], dim=0),
                            vae.config.shift_factor,
                            vae.config.scaling_factor,
                        )

                pair_batch_size = len(batch["sample_ids"])
                sample_weights = batch["sample_weights"].to(accelerator.device, dtype=torch.float32)
                loss_for_log = torch.zeros((), device=accelerator.device)
                raw_model_loss = torch.zeros((), device=accelerator.device)
                raw_reference_loss = torch.zeros((), device=accelerator.device)
                implicit_correct = torch.zeros((), device=accelerator.device)
                implicit_count = torch.zeros((), device=accelerator.device)

                for _ in range(args.num_train_timesteps_per_batch):
                    if args.method == "sft":
                        timesteps = sample_timesteps(
                            args, noise_scheduler, model_input.shape[0], model_input.device
                        )
                        noise = torch.randn_like(model_input)
                        active_prompt_embeds = prompt_embeds
                    else:
                        base_timesteps = sample_timesteps(
                            args, noise_scheduler, pair_batch_size, model_input.device
                        )
                        timesteps = base_timesteps.repeat(2)
                        noise = torch.randn_like(model_input[:pair_batch_size]).repeat(2, 1, 1, 1)
                        active_prompt_embeds = prompt_embeds + prompt_embeds

                    sigmas = get_sigmas(
                        noise_scheduler,
                        timesteps,
                        model_input.ndim,
                        model_input.dtype,
                        model_input.device,
                    )
                    noisy_input = (1.0 - sigmas) * model_input + sigmas * noise
                    normalized_timesteps = (1000 - timesteps) / 1000
                    target = noise - model_input
                    with accelerator.autocast():
                        prediction = transformer_forward(
                            transformer, noisy_input, normalized_timesteps, active_prompt_embeds
                        )
                    model_losses = flow_matching_mse_per_sample(prediction, target)
                    raw_model_loss += model_losses.mean().detach() / args.num_train_timesteps_per_batch

                    if args.method == "sft":
                        per_sample_loss = model_losses
                    else:
                        unwrapped = unwrap_model(accelerator, transformer)
                        unwrapped.disable_adapters()
                        try:
                            with torch.no_grad(), accelerator.autocast():
                                reference_prediction = transformer_forward(
                                    transformer,
                                    noisy_input,
                                    normalized_timesteps,
                                    active_prompt_embeds,
                                )
                            reference_losses = flow_matching_mse_per_sample(
                                reference_prediction, target
                            )
                        finally:
                            unwrapped.enable_adapters()
                        raw_reference_loss += (
                            reference_losses.mean().detach() / args.num_train_timesteps_per_batch
                        )
                        per_sample_loss, implicit_logit = flow_dpo_loss_per_sample(
                            model_losses, reference_losses, args.beta_dpo
                        )
                        implicit_correct += (
                            (implicit_logit > 0).float().sum()
                            + 0.5 * (implicit_logit == 0).float().sum()
                        )
                        implicit_count += implicit_logit.numel()

                    step_loss = (per_sample_loss * sample_weights).mean()
                    loss_for_log += step_loss.detach() / args.num_train_timesteps_per_batch
                    accelerator.backward(step_loss / args.num_train_timesteps_per_batch)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                        raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            progress.update(1)
            step_seconds_local = time.perf_counter() - optimizer_step_started
            optimizer_step_started = None
            gathered_loss = accelerator.gather(loss_for_log.reshape(1)).mean().item()
            if not math.isfinite(gathered_loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {global_step}: {gathered_loss}")
            step_seconds = accelerator.gather(
                torch.tensor([step_seconds_local], device=accelerator.device, dtype=torch.float64)
            ).max().item()
            token_max = accelerator.gather(
                torch.tensor([max(prompt_lengths)], device=accelerator.device, dtype=torch.int64)
            ).max().item()
            global_batch = (
                args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
            )
            record = {
                "step": global_step,
                "epoch": epoch,
                "method": args.method,
                "loss": gathered_loss,
                "raw_model_loss": accelerator.gather(raw_model_loss.reshape(1)).mean().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "prompt_token_max": int(token_max),
                "step_time_seconds": step_seconds,
                "context_rows_per_second": global_batch / step_seconds,
            }
            if args.method == "flow_dpo":
                total_correct = accelerator.gather(implicit_correct.reshape(1)).sum().item()
                total_count = accelerator.gather(implicit_count.reshape(1)).sum().item()
                record["raw_reference_loss"] = accelerator.gather(
                    raw_reference_loss.reshape(1)
                ).mean().item()
                record["implicit_accuracy"] = total_correct / total_count
            if torch.cuda.is_available():
                memory = accelerator.gather(
                    torch.tensor(
                        [[
                            torch.cuda.max_memory_allocated(accelerator.device),
                            torch.cuda.max_memory_reserved(accelerator.device),
                        ]],
                        device=accelerator.device,
                        dtype=torch.float64,
                    )
                ).amax(dim=0)
                record["max_gpu_memory_allocated_gb"] = memory[0].item() / 1024**3
                record["max_gpu_memory_reserved_gb"] = memory[1].item() / 1024**3
            if accelerator.is_main_process and (
                global_step == 1
                or global_step % args.parameter_logging_steps == 0
                or global_step >= args.max_train_steps
            ):
                record["lora_l2_norm"] = adapter_l2_norm(unwrap_model(accelerator, transformer))
            progress.set_postfix(loss=f"{gathered_loss:.6f}")
            if accelerator.is_main_process:
                with loss_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            if args.report_to != "none":
                accelerator.log(
                    {
                        f"train/{key}": value
                        for key, value in record.items()
                        if isinstance(value, (int, float)) and key not in {"step", "epoch"}
                    },
                    step=global_step,
                )

            if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    prune_checkpoints(output_dir, args.checkpoints_total_limit)
                accelerator.wait_for_everyone()
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(checkpoint_dir))
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    write_json(
                        checkpoint_dir / "checkpoint_complete.json",
                        {"global_step": global_step, "method": args.method, "status": "complete"},
                    )
                accelerator.wait_for_everyone()
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = get_peft_model_state_dict(unwrap_model(accelerator, transformer))
        ZImagePipeline.save_lora_weights(str(output_dir), transformer_lora_layers=state)
        write_json(
            output_dir / "training_complete.json",
            {
                "global_step": global_step,
                "method": args.method,
                "student_context": "q",
                "privileged_prompt_usage": "offline_supervision_only",
                "context_sha256": sha256_file(context_path),
                "image_manifest_sha256": sha256_file(manifest_path),
            },
        )
    accelerator.end_training()
