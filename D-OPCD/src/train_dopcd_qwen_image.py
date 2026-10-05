#!/usr/bin/env python
"""D-OPCD LoRA trainer for Qwen-Image."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, broadcast_object_list, set_seed
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration

from common import (
    DEFAULT_CONTEXT_LABEL, PROJECT_ROOT, ensure_within,
    resolve_teacher_context_mode, teacher_context_description, write_json,
)
from dataset import PromptContextDataset, collate_prompt_contexts
from lora_ema import (
    adapter_gap_l2_norm,
    adapter_l2_norm,
    copy_adapter,
    ema_update_adapter,
    set_adapter_trainable,
)
from objective import distillation_loss
from qwen_image_prompt_encoder import (
    QWEN_IMAGE_TOKENIZER_MAX_LENGTH,
    encode_qwen_image_prompts,
    load_qwen_image_tokenizer,
)


logger = get_logger(__name__)
STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher"
TEACHER_STATE_FILE = "teacher_lora.safetensors"
DEFAULT_LORA_LAYERS = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def parse_timesteps(value: str) -> list[float]:
    try:
        timesteps = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timesteps must be comma-separated floats") from exc
    if not timesteps:
        raise argparse.ArgumentTypeError("at least one timestep is required")
    if any(timestep < 0.0 or timestep >= 1.0 for timestep in timesteps):
        raise argparse.ArgumentTypeError("timesteps must be in [0, 1)")
    if any(left >= right for left, right in zip(timesteps, timesteps[1:])):
        raise argparse.ArgumentTypeError("timesteps must be strictly increasing")
    return timesteps


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D-OPCD LoRA training for Qwen-Image: student=q, teacher context is configurable."
    )
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--context_label", default=DEFAULT_CONTEXT_LABEL)
    parser.add_argument("--teacher_context_mode", choices=("p_only", "q_plus_p", "q_only"), default=None,
                        help="q_plus_p (paper main) or p_only (bare CLI fallback). Resumes preserve the saved mode.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--loss_type", choices=["velocity", "endpoint"], default="velocity")
    parser.add_argument(
        "--timesteps", type=parse_timesteps, default=parse_timesteps("0.0,0.1,0.25,0.5")
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--vae_scale_factor", type=int, default=8)
    parser.add_argument("--student_max_sequence_length", type=int, default=512)
    parser.add_argument("--teacher_max_sequence_length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", default=",".join(DEFAULT_LORA_LAYERS))
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
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
    parser.add_argument("--tracker_project_name", default="qwen-image-dopcd")
    parser.add_argument("--tracker_run_name", default=None)
    parser.add_argument("--tracker_run_id", default=None)
    parser.add_argument("--wandb_init_timeout", type=int, default=300)
    parser.add_argument("--parameter_logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(input_args)

    if args.vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")
    latent_divisor = args.vae_scale_factor * 2
    if args.resolution <= 0 or args.resolution % latent_divisor:
        raise ValueError(f"resolution must be a positive multiple of {latent_divisor}")
    if args.student_max_sequence_length <= 0 or args.teacher_max_sequence_length <= 0:
        raise ValueError("prompt sequence lengths must be positive")
    if args.teacher_max_sequence_length > QWEN_IMAGE_TOKENIZER_MAX_LENGTH:
        raise ValueError(
            f"Qwen-Image prompt length limit is {QWEN_IMAGE_TOKENIZER_MAX_LENGTH}"
        )
    if args.max_train_steps <= 0 or args.checkpointing_steps <= 0:
        raise ValueError("training/checkpoint steps must be positive")
    if args.parameter_logging_steps <= 0:
        raise ValueError("parameter_logging_steps must be positive")
    if args.wandb_init_timeout <= 0:
        raise ValueError("wandb_init_timeout must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not str(args.context_label).strip():
        raise ValueError("context_label must be non-empty")
    return args


def build_tracker_init_kwargs(args: argparse.Namespace) -> dict | None:
    if args.report_to != "wandb":
        return None
    wandb_init: dict = {
        "settings": {"init_timeout": args.wandb_init_timeout},
    }
    if args.tracker_run_name:
        wandb_init["name"] = args.tracker_run_name
    if args.tracker_run_id:
        wandb_init["id"] = args.tracker_run_id
        if os.environ.get("WANDB_MODE", "online").lower() != "offline":
            wandb_init["resume"] = "allow"
    return {"wandb": wandb_init}


def initialize_trackers(
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> None:
    """Initialize tracking and propagate rank-0 failures to every process."""
    tracker_error: list[str | None] = [None]
    tracker_exception: Exception | None = None
    try:
        accelerator.init_trackers(
            args.tracker_project_name,
            config={**vars(args), "timesteps": list(args.timesteps)},
            init_kwargs=build_tracker_init_kwargs(args),
        )
    except Exception as exc:  # tracker backends run only on the main process
        tracker_exception = exc
        tracker_error[0] = f"{type(exc).__name__}: {exc}"

    broadcast_object_list(tracker_error, from_process=0)
    if tracker_error[0] is not None:
        message = f"Tracker initialization failed on rank 0: {tracker_error[0]}"
        if tracker_exception is not None:
            raise RuntimeError(message) from tracker_exception
        raise RuntimeError(message)


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def set_active_adapter(accelerator: Accelerator, model, adapter_name: str) -> None:
    unwrap_model(accelerator, model).set_adapter(adapter_name)


def qwen_image_shape(batch_size: int, resolution: int, vae_scale_factor: int):
    packed_side = resolution // vae_scale_factor // 2
    return [[(1, packed_side, packed_side)]] * batch_size


def transformer_forward(
    model,
    latents: torch.Tensor,
    progress_timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    img_shapes,
) -> torch.Tensor:
    """Return velocity in progress time u=1-sigma (noise at u=0, data at u=1)."""
    sigma = 1.0 - progress_timesteps
    scheduler_velocity = model(
        hidden_states=latents,
        timestep=sigma,
        guidance=None,
        encoder_hidden_states_mask=prompt_mask,
        encoder_hidden_states=prompt_embeds,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    return -scheduler_velocity


def predict_teacher(
    accelerator: Accelerator,
    model,
    latents: torch.Tensor,
    progress_timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    img_shapes,
) -> torch.Tensor:
    unwrapped = unwrap_model(accelerator, model)
    unwrapped.set_adapter(TEACHER_ADAPTER)
    try:
        with torch.no_grad(), accelerator.autocast():
            return transformer_forward(
                model,
                latents,
                progress_timesteps,
                prompt_embeds,
                prompt_mask,
                img_shapes,
            )
    finally:
        unwrapped.set_adapter(STUDENT_ADAPTER)


def find_resume_checkpoint(output_dir: Path, requested: str | None) -> Path | None:
    if not requested:
        return None
    if requested != "latest":
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate = candidate.resolve()
        return (
            candidate
            if candidate.is_dir() and (candidate / "checkpoint_complete.json").is_file()
            else None
        )
    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        if not candidate.is_dir() or not (candidate / "checkpoint_complete.json").is_file():
            continue
        try:
            step = int(candidate.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, candidate))
    return max(checkpoints, default=(None, None))[1]


def configure_logging(accelerator: Accelerator, output_dir: Path) -> None:
    level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if accelerator.is_main_process:
        handler = logging.FileHandler(output_dir / "train.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)


def resolve_output_directory(path: str | Path, project_root: str | Path) -> Path:
    """Resolve a run output beneath the configured canonical results root."""
    project_boundary = Path(project_root).expanduser().resolve()
    output_boundary = Path(
        os.environ.get("DOPCD_RESULTS_ROOT", str(project_boundary))
    ).expanduser().resolve()
    return ensure_within(path, output_boundary)


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = resolve_output_directory(args.output_dir, project_root)
    args.teacher_context_mode = resolve_teacher_context_mode(
        args.teacher_context_mode, output_dir / "args.json", bool(args.resume_from_checkpoint)
    )
    if args.teacher_context_mode not in {"p_only", "q_plus_p", "q_only"}:
        raise ValueError("Qwen-Image trainer supports text-only teacher context modes")
    data_jsonl = ensure_within(args.data_jsonl, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / args.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_DIR", str(logging_dir))

    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    log_with = None if args.report_to == "none" else args.report_to
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    configure_logging(accelerator, output_dir)
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        saved_args = vars(args).copy()
        saved_args["timesteps"] = list(args.timesteps)
        saved_args["method"] = "D-OPCD"
        saved_args["model_family"] = "qwen_image"
        saved_args["student_context"] = "q"
        saved_args["teacher_context"] = teacher_context_description(args.teacher_context_mode)
        saved_args["flow_coordinate"] = "progress_u=1-sigma"
        saved_args["ddp_find_unused_parameters"] = True
        write_json(output_dir / "args.json", saved_args)
    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = str(Path(args.pretrained_model_name_or_path).expanduser().resolve())
    vae_config_path = Path(model_path) / "vae" / "config.json"
    with vae_config_path.open("r", encoding="utf-8") as handle:
        vae_config = json.load(handle)
    model_vae_scale_factor = 2 ** len(vae_config.get("temperal_downsample") or [])
    if args.vae_scale_factor != model_vae_scale_factor:
        raise ValueError(
            f"vae_scale_factor={args.vae_scale_factor}, model requires "
            f"{model_vae_scale_factor} from {vae_config_path}"
        )
    tokenizer = load_qwen_image_tokenizer(model_path)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    if bool(transformer.config.guidance_embeds):
        raise ValueError("This D-OPCD backend currently requires guidance_embeds=false")
    text_encoder.requires_grad_(False).eval()
    transformer.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    target_modules = [value.strip() for value in args.lora_layers.split(",") if value.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config, adapter_name=STUDENT_ADAPTER)
    transformer.set_adapter(STUDENT_ADAPTER)
    transformer.add_adapter(lora_config, adapter_name=TEACHER_ADAPTER)
    copy_adapter(transformer, STUDENT_ADAPTER, TEACHER_ADAPTER)
    set_adapter_trainable(transformer, TEACHER_ADAPTER, False)
    transformer.set_adapter(STUDENT_ADAPTER)

    train_dataset = PromptContextDataset(
        data_jsonl, context_label=args.context_label,
        teacher_context_mode=args.teacher_context_mode,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_prompt_contexts,
        pin_memory=True,
        drop_last=True,
    )
    parameters_to_optimize = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    def save_model_hook(models, weights, checkpoint_dir):
        if not accelerator.is_main_process:
            return
        checkpoint_path = Path(checkpoint_dir)
        model = unwrap_model(accelerator, transformer)
        student_state = get_peft_model_state_dict(model, adapter_name=STUDENT_ADAPTER)
        QwenImagePipeline.save_lora_weights(
            str(checkpoint_path), transformer_lora_layers=student_state
        )
        teacher_state = {
            key: value.detach().cpu().contiguous()
            for key, value in get_peft_model_state_dict(
                model, adapter_name=TEACHER_ADAPTER
            ).items()
        }
        save_safetensors(teacher_state, str(checkpoint_path / TEACHER_STATE_FILE))
        for _ in models:
            if weights:
                weights.pop()

    def load_model_hook(models, checkpoint_dir):
        model = unwrap_model(accelerator, transformer)
        lora_state = QwenImagePipeline.lora_state_dict(
            checkpoint_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        student_state = {
            key.replace("transformer.", ""): value
            for key, value in lora_state.items()
            if key.startswith("transformer.")
        }
        student_state = convert_unet_state_dict_to_peft(student_state)
        incompatible = set_peft_model_state_dict(
            model, student_state, adapter_name=STUDENT_ADAPTER
        )
        if getattr(incompatible, "unexpected_keys", None):
            raise ValueError(f"Unexpected student LoRA keys: {incompatible.unexpected_keys}")
        teacher_path = Path(checkpoint_dir) / TEACHER_STATE_FILE
        if not teacher_path.is_file():
            raise FileNotFoundError(f"EMA checkpoint is missing {teacher_path}")
        teacher_state = load_safetensors(str(teacher_path), device="cpu")
        incompatible = set_peft_model_state_dict(
            model, teacher_state, adapter_name=TEACHER_ADAPTER
        )
        if getattr(incompatible, "unexpected_keys", None):
            raise ValueError(f"Unexpected teacher LoRA keys: {incompatible.unexpected_keys}")
        model.set_adapter(STUDENT_ADAPTER)
        while models:
            models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / updates_per_epoch)

    if log_with is not None:
        initialize_trackers(accelerator, args)

    resume_path = find_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    global_step = 0
    first_epoch = 0
    resume_batch = 0
    if resume_path is not None:
        accelerator.print(f"Resuming from {resume_path}")
        accelerator.load_state(str(resume_path))
        global_step = int(resume_path.name.split("-")[-1])
        consumed_batches = global_step * args.gradient_accumulation_steps
        first_epoch = consumed_batches // len(train_dataloader)
        resume_batch = consumed_batches % len(train_dataloader)
    elif args.resume_from_checkpoint:
        accelerator.print(
            f"No complete checkpoint found for {args.resume_from_checkpoint!r}; starting at step 0"
        )

    if accelerator.is_main_process:
        logger.info("***** Running Qwen-Image D-OPCD *****")
        logger.info("Examples: %d", len(train_dataset))
        logger.info("Processes: %d", accelerator.num_processes)
        logger.info(
            "Global batch: %d",
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps,
        )
        logger.info("Context routing: student=q; teacher=%s (mode=%s)",
                    teacher_context_description(args.teacher_context_mode), args.teacher_context_mode)
        logger.info("Loss: %s; progress timesteps=%s", args.loss_type, args.timesteps)
        logger.info("Flow coordinate: u=1-sigma; progress velocity=-scheduler output")
        logger.info(
            "Trainable parameters: %d",
            sum(parameter.numel() for parameter in parameters_to_optimize),
        )

    progress = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    loss_log_path = output_dir / "loss.jsonl"
    packed_side = args.resolution // args.vae_scale_factor // 2
    optimizer_step_started = None

    for epoch in range(first_epoch, num_epochs):
        active_dataloader = train_dataloader
        if epoch == first_epoch and resume_batch:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch)
        for batch in active_dataloader:
            if optimizer_step_started is None:
                optimizer_step_started = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
            with accelerator.accumulate(transformer):
                with torch.no_grad(), accelerator.autocast():
                    student_embeds, student_mask, student_lengths = encode_qwen_image_prompts(
                        text_encoder,
                        tokenizer,
                        batch["student_prompts"],
                        accelerator.device,
                        args.student_max_sequence_length,
                        dtype=weight_dtype,
                    )
                    teacher_embeds, teacher_mask, teacher_lengths = encode_qwen_image_prompts(
                        text_encoder,
                        tokenizer,
                        batch["teacher_prompts"],
                        accelerator.device,
                        args.teacher_max_sequence_length,
                        dtype=weight_dtype,
                    )

                batch_size = len(batch["sample_ids"])
                packed_channels = int(unwrap_model(accelerator, transformer).config.in_channels)
                student_latents = torch.randn(
                    (batch_size, packed_side * packed_side, packed_channels),
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                img_shapes = qwen_image_shape(
                    batch_size, args.resolution, args.vae_scale_factor
                )
                step_losses = []
                for step_index, timestep in enumerate(args.timesteps):
                    next_timestep = (
                        args.timesteps[step_index + 1]
                        if step_index + 1 < len(args.timesteps)
                        else 1.0
                    )
                    dt = next_timestep - timestep
                    current_latents = student_latents.detach()
                    timestep_batch = torch.full(
                        (batch_size,),
                        timestep,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    teacher_prediction = predict_teacher(
                        accelerator,
                        transformer,
                        current_latents,
                        timestep_batch,
                        teacher_embeds,
                        teacher_mask,
                        img_shapes,
                    )
                    set_active_adapter(accelerator, transformer, STUDENT_ADAPTER)
                    is_last_timestep = step_index + 1 == len(args.timesteps)
                    sync_context = (
                        nullcontext()
                        if accelerator.sync_gradients and is_last_timestep
                        else accelerator.no_sync(transformer)
                    )
                    with sync_context:
                        with accelerator.autocast():
                            student_prediction = transformer_forward(
                                transformer,
                                current_latents,
                                timestep_batch,
                                student_embeds,
                                student_mask,
                                img_shapes,
                            )
                        step_loss = distillation_loss(
                            student_prediction,
                            teacher_prediction,
                            current_latents,
                            timestep,
                            args.loss_type,
                        )
                        finite_loss = accelerator.gather(
                            torch.isfinite(step_loss.detach()).reshape(1)
                        ).all()
                        if not bool(finite_loss.item()):
                            raise FloatingPointError(
                                f"Non-finite loss at optimizer step {global_step + 1}, "
                                f"timestep={timestep}: {step_loss.detach().float().item()}"
                            )
                        accelerator.backward(step_loss / len(args.timesteps))
                    step_losses.append(step_loss.detach())
                    student_latents = (
                        current_latents + dt * student_prediction.detach()
                    )

                loss = torch.stack(step_losses).mean()
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )
                    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at optimizer step {global_step + 1}: "
                            f"{grad_norm}"
                        )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            progress.update(1)
            step_seconds_local = time.perf_counter() - optimizer_step_started
            optimizer_step_started = None
            ema_update_adapter(
                unwrap_model(accelerator, transformer),
                STUDENT_ADAPTER,
                TEACHER_ADAPTER,
                args.ema_decay,
            )
            set_active_adapter(accelerator, transformer, STUDENT_ADAPTER)

            gathered_loss = accelerator.gather(loss.detach().reshape(1)).mean().item()
            gathered_step_seconds = accelerator.gather(
                torch.tensor([step_seconds_local], device=accelerator.device, dtype=torch.float64)
            ).max().item()
            token_maxima = accelerator.gather(
                torch.tensor(
                    [[max(student_lengths), max(teacher_lengths)]],
                    device=accelerator.device,
                    dtype=torch.int64,
                )
            ).amax(dim=0)
            peak_memory_gb = 0.0
            reserved_memory_gb = 0.0
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
                peak_memory_gb = memory[0].item() / (1024**3)
                reserved_memory_gb = memory[1].item() / (1024**3)
            global_batch_size = (
                args.train_batch_size
                * accelerator.num_processes
                * args.gradient_accumulation_steps
            )
            record = {
                "step": global_step,
                "epoch": epoch,
                "loss": gathered_loss,
                "loss_type": args.loss_type,
                "step_losses": [
                    accelerator.gather(value.detach().reshape(1)).mean().item()
                    for value in step_losses
                ],
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "student_token_max": int(token_maxima[0].item()),
                "teacher_token_max": int(token_maxima[1].item()),
                "step_time_seconds": gathered_step_seconds,
                "samples_per_second": global_batch_size / gathered_step_seconds,
                "max_gpu_memory_allocated_gb": peak_memory_gb,
                "max_gpu_memory_reserved_gb": reserved_memory_gb,
            }
            if accelerator.is_main_process and (
                global_step == 1
                or global_step % args.parameter_logging_steps == 0
                or global_step >= args.max_train_steps
            ):
                model = unwrap_model(accelerator, transformer)
                record["student_lora_l2_norm"] = float(
                    adapter_l2_norm(model, STUDENT_ADAPTER).item()
                )
                record["ema_teacher_student_l2_gap"] = float(
                    adapter_gap_l2_norm(model, STUDENT_ADAPTER, TEACHER_ADAPTER).item()
                )
            progress.set_postfix(loss=f"{gathered_loss:.6f}")
            if accelerator.is_main_process:
                with loss_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            if log_with is not None:
                tracker_metrics = {
                    "train/loss": gathered_loss,
                    "train/learning_rate": record["lr"],
                    "train/grad_norm": record["grad_norm"],
                    "performance/step_time_seconds": gathered_step_seconds,
                    "performance/samples_per_second": record["samples_per_second"],
                    "performance/max_gpu_memory_allocated_gb": record[
                        "max_gpu_memory_allocated_gb"
                    ],
                    "performance/max_gpu_memory_reserved_gb": record[
                        "max_gpu_memory_reserved_gb"
                    ],
                    "data/student_token_max": record["student_token_max"],
                    "data/teacher_token_max": record["teacher_token_max"],
                    "train/epoch": record["epoch"],
                }
                tracker_metrics.update(
                    {
                        f"train/loss_t_{timestep:g}": step_loss
                        for timestep, step_loss in zip(args.timesteps, record["step_losses"])
                    }
                )
                if "student_lora_l2_norm" in record:
                    tracker_metrics["parameters/student_lora_l2_norm"] = record[
                        "student_lora_l2_norm"
                    ]
                    tracker_metrics["parameters/ema_teacher_student_l2_gap"] = record[
                        "ema_teacher_student_l2_gap"
                    ]
                accelerator.log(tracker_metrics, step=global_step)

            if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                if accelerator.is_main_process:
                    logger.info("Saving checkpoint to %s", checkpoint_dir)
                accelerator.save_state(str(checkpoint_dir))
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    write_json(
                        checkpoint_dir / "checkpoint_complete.json",
                        {
                            "global_step": global_step,
                            "method": "D-OPCD",
                            "model_family": "qwen_image",
                            "loss_type": args.loss_type,
                            "status": "complete",
                        },
                    )
                accelerator.wait_for_everyone()
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = unwrap_model(accelerator, transformer)
        model.set_adapter(STUDENT_ADAPTER)
        final_state = get_peft_model_state_dict(model, adapter_name=STUDENT_ADAPTER)
        QwenImagePipeline.save_lora_weights(
            str(output_dir), transformer_lora_layers=final_state
        )
        write_json(
            output_dir / "training_complete.json",
            {
                "global_step": global_step,
                "method": "D-OPCD",
                "model_family": "qwen_image",
                "student_context": "q",
                "teacher_context": teacher_context_description(args.teacher_context_mode),
                "teacher_context_mode": args.teacher_context_mode,
                "loss_type": args.loss_type,
                "student_adapter": STUDENT_ADAPTER,
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
