#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import ZImagePipeline, ZImageTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Qwen2Tokenizer, Qwen3Model

from common import (
    DEFAULT_CONTEXT_LABEL, PROJECT_ROOT, TEACHER_CONTEXT_MODES, ensure_within,
    resolve_teacher_context_mode, sha256_file, teacher_context_description, write_json,
)
from dataset import PromptContextDataset, collate_prompt_contexts
from lora_ema import (
    adapter_gap_l2_norm,
    adapter_l2_norm,
    copy_adapter,
    ema_update_adapter,
    set_adapter_trainable,
)
from objective import distillation_loss_per_sample
from prompt_encoder import encode_prompts
from vlm_utils import get_qwen3vl_zimage_prompt_embeds, load_matching_state_dict


logger = get_logger(__name__)
STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher"
TEACHER_STATE_FILE = "teacher_lora.safetensors"
DEFAULT_LORA_LAYERS = [
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "attention.to_k",
    "attention.to_q",
    "attention.to_v",
    "attention.to_out.0",
]


def required_text_axis_length(max_sequence_length: int) -> int:
    """Allow for 32-token caption padding and the generated image's first position."""
    padded_caption_length = ((max_sequence_length + 31) // 32) * 32
    return padded_caption_length + 2


def configure_text_axis(transformer: ZImageTransformer2DModel, args: argparse.Namespace) -> int:
    """Extend only the runtime RoPE lookup table; model weights remain unchanged."""
    base_length = int(transformer.config.axes_lens[0])
    axis_length = args.teacher_rope_axis_length or base_length
    required_length = required_text_axis_length(
        max(args.student_max_sequence_length, args.teacher_max_sequence_length)
    )
    if axis_length < required_length:
        raise ValueError(
            f"text-axis length={axis_length} is too short for the configured prompt; "
            f"requires at least {required_length} positions including padding and image offset"
        )
    if axis_length != base_length:
        transformer.rope_embedder.axes_lens = [
            axis_length, *transformer.rope_embedder.axes_lens[1:]
        ]
        transformer.rope_embedder.freqs_cis = None
    return axis_length


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
        description="D-OPCD LoRA training for Z-Image-Turbo: student=q, teacher context is configurable."
    )
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--sample_weights_jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--context_label", default=DEFAULT_CONTEXT_LABEL)
    parser.add_argument("--teacher_context_mode", choices=TEACHER_CONTEXT_MODES, default=None,
                        help="q_plus_p (paper main), p_only (bare CLI fallback), q_only, vlm_q_image (D-OPSD), or vlm_q_p_image (VLM([q;p]+image)). Resumes preserve the saved mode.")
    parser.add_argument("--image_manifest_jsonl", default=None,
                        help="Image sidecar with target_image/target_sha256 per sample; required for VLM image modes.")
    parser.add_argument("--vlm_model_path", default=os.environ.get("DOPCD_VLM_MODEL_PATH", "models/Qwen3-VL-4B-Instruct"),
                        help="Qwen3-VL checkpoint used to build teacher embeds for VLM image modes.")
    parser.add_argument("--vlm_min_pixels", type=int, default=512 * 512)
    parser.add_argument("--vlm_max_pixels", type=int, default=768 * 768)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--loss_type", choices=["velocity", "endpoint"], default="velocity")
    parser.add_argument("--timesteps", type=parse_timesteps, default=parse_timesteps("0.0,0.1,0.25,0.5"))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--student_max_sequence_length", type=int, default=512)
    parser.add_argument("--teacher_max_sequence_length", type=int, default=1024)
    parser.add_argument("--teacher_rope_axis_length", type=int, default=None,
                        help="Opt-in runtime text-axis RoPE length; does not alter the base checkpoint.")
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
    parser.add_argument("--tracker_project_name", default="zimage-dopcd")
    parser.add_argument("--tracker_run_name", default=None)
    parser.add_argument("--tracker_run_id", default=None)
    parser.add_argument("--wandb_init_timeout", type=int, default=300)
    parser.add_argument("--parameter_logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(input_args)

    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    if args.student_max_sequence_length <= 0 or args.teacher_max_sequence_length <= 0:
        raise ValueError("prompt sequence lengths must be positive")
    if args.teacher_rope_axis_length is not None and args.teacher_rope_axis_length <= 0:
        raise ValueError("teacher_rope_axis_length must be positive")
    if args.vlm_min_pixels <= 0 or args.vlm_max_pixels < args.vlm_min_pixels:
        raise ValueError("VLM pixel budget must satisfy 0 < min_pixels <= max_pixels")
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


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def set_active_adapter(accelerator: Accelerator, model, adapter_name: str) -> None:
    unwrap_model(accelerator, model).set_adapter(adapter_name)


def transformer_forward(model, latents: torch.Tensor, timesteps: torch.Tensor, prompt_embeds):
    latent_list = list(latents.unsqueeze(2).unbind(dim=0))
    predictions = model(latent_list, timesteps, prompt_embeds, return_dict=False)[0]
    return torch.stack(predictions, dim=0).squeeze(2)


def predict_teacher(
    accelerator: Accelerator,
    model,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds,
) -> torch.Tensor:
    unwrapped = unwrap_model(accelerator, model)
    unwrapped.set_adapter(TEACHER_ADAPTER)
    try:
        with torch.no_grad(), accelerator.autocast():
            return transformer_forward(model, latents, timesteps, prompt_embeds)
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
        return candidate if candidate.is_dir() and (candidate / "checkpoint_complete.json").is_file() else None
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
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)


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


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_boundary = Path(os.environ.get("DOPCD_RESULTS_ROOT", str(project_root))).expanduser().resolve()
    output_dir = ensure_within(args.output_dir, output_boundary)
    args.teacher_context_mode = resolve_teacher_context_mode(
        args.teacher_context_mode, output_dir / "args.json", bool(args.resume_from_checkpoint)
    )
    data_jsonl = ensure_within(args.data_jsonl, project_root)
    sample_weights_jsonl = (
        ensure_within(args.sample_weights_jsonl, project_root)
        if args.sample_weights_jsonl
        else None
    )
    if args.teacher_context_mode in {"vlm_q_image", "vlm_q_p_image"}:
        if not args.image_manifest_jsonl:
            raise ValueError(f"{args.teacher_context_mode} requires --image_manifest_jsonl")
        image_manifest_jsonl = ensure_within(args.image_manifest_jsonl, project_root)
    else:
        if args.image_manifest_jsonl:
            raise ValueError("--image_manifest_jsonl is only valid with a VLM image mode")
        image_manifest_jsonl = None
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.report_to == "wandb":
        # Keep all writable W&B state with this run, including direct launches.
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

    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    log_with = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=project_config,
    )
    configure_logging(accelerator, output_dir)
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        saved_args = vars(args).copy()
        saved_args["timesteps"] = list(args.timesteps)
        saved_args["method"] = "D-OPSD" if args.teacher_context_mode == "vlm_q_image" else "D-OPCD"
        saved_args["student_context"] = "q"
        saved_args["teacher_context"] = teacher_context_description(args.teacher_context_mode)
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
    transformer = ZImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    text_encoder.requires_grad_(False).eval()
    transformer.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_axis_length = configure_text_axis(transformer, args)
    accelerator.print(
        f"Z-Image text-axis RoPE length={text_axis_length} "
        f"(checkpoint config={transformer.config.axes_lens[0]}, "
        f"teacher prompt limit={args.teacher_max_sequence_length})"
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    vl_model = None
    vl_processor = None
    if args.teacher_context_mode in {"vlm_q_image", "vlm_q_p_image"}:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        vlm_path = str(Path(args.vlm_model_path).expanduser().resolve())
        vl_processor = AutoProcessor.from_pretrained(
            vlm_path, min_pixels=args.vlm_min_pixels, max_pixels=args.vlm_max_pixels,
            local_files_only=True,
        )
        vl_model = AutoModelForImageTextToText.from_pretrained(
            vlm_path, torch_dtype=weight_dtype, local_files_only=True
        )
        load_matching_state_dict(
            target_module=vl_model.model.language_model,
            source_state_dict=text_encoder.state_dict(),
            verbose=False,
        )
        vl_model.requires_grad_(False).eval()
        vl_model.to(accelerator.device, dtype=weight_dtype)
        accelerator.print(f"Teacher VLM loaded from {vlm_path} ({teacher_context_description(args.teacher_context_mode)} embeds)")

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
        data_jsonl,
        context_label=args.context_label,
        sample_weights_jsonl=sample_weights_jsonl,
        teacher_context_mode=args.teacher_context_mode,
        image_manifest_jsonl=image_manifest_jsonl,
        resolution=args.resolution,
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
    parameters_to_optimize = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
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
        ZImagePipeline.save_lora_weights(
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
        lora_state = ZImagePipeline.lora_state_dict(
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
        accelerator.init_trackers(
            args.tracker_project_name,
            config={**vars(args), "timesteps": list(args.timesteps)},
            init_kwargs=build_tracker_init_kwargs(args),
        )

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
        method_name = "D-OPSD" if args.teacher_context_mode == "vlm_q_image" else "D-OPCD"
        logger.info("***** Running %s *****", method_name)
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
        if sample_weights_jsonl is None:
            logger.info("Sample weighting: uniform")
        else:
            weights = list(train_dataset.sample_weights.values())
            logger.info(
                "Sample weighting: %s; min=%.6f mean=%.6f max=%.6f",
                sample_weights_jsonl,
                min(weights),
                sum(weights) / len(weights),
                max(weights),
            )
        logger.info("Loss: %s; timesteps=%s", args.loss_type, args.timesteps)
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
    latent_spatial = 2 * (args.resolution // 16)
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
                    student_embeds, student_lengths = encode_prompts(
                        text_encoder,
                        tokenizer,
                        batch["student_prompts"],
                        accelerator.device,
                        args.student_max_sequence_length,
                        dtype=weight_dtype,
                    )
                    if args.teacher_context_mode in {"vlm_q_image", "vlm_q_p_image"}:
                        teacher_images = (batch["teacher_images"] + 1.0) / 2.0
                        teacher_embeds = get_qwen3vl_zimage_prompt_embeds(
                            vl_model=vl_model,
                            processor=vl_processor,
                            prompts=batch["teacher_prompts"],
                            images=list(teacher_images.unbind(dim=0)),
                            device=accelerator.device,
                            dtype=weight_dtype,
                            max_sequence_length=args.teacher_max_sequence_length,
                            num_images_per_prompt=1,
                            hidden_state_layer=-2,
                            use_system_prompt=False,
                        )
                        teacher_lengths = [int(embed.shape[0]) for embed in teacher_embeds]
                    else:
                        teacher_embeds, teacher_lengths = encode_prompts(
                            text_encoder,
                            tokenizer,
                            batch["teacher_prompts"],
                            accelerator.device,
                            args.teacher_max_sequence_length,
                            dtype=weight_dtype,
                        )

                batch_size = len(batch["sample_ids"])
                sample_weights = torch.tensor(
                    batch["sample_weights"],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                num_channels = int(unwrap_model(accelerator, transformer).config.in_channels)
                student_latents = torch.randn(
                    (batch_size, num_channels, latent_spatial, latent_spatial),
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                step_losses = []
                for step_index, timestep in enumerate(args.timesteps):
                    next_timestep = (
                        args.timesteps[step_index + 1]
                        if step_index + 1 < len(args.timesteps)
                        else 1.0
                    )
                    dt = next_timestep - timestep
                    current_latents = student_latents.detach().requires_grad_(True)
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
                    )
                    set_active_adapter(accelerator, transformer, STUDENT_ADAPTER)
                    with accelerator.autocast():
                        student_prediction = transformer_forward(
                            transformer,
                            current_latents,
                            timestep_batch,
                            student_embeds,
                        )
                    per_sample_step_loss = distillation_loss_per_sample(
                        student_prediction,
                        teacher_prediction,
                        current_latents,
                        timestep,
                        args.loss_type,
                    )
                    if per_sample_step_loss.shape != sample_weights.shape:
                        raise RuntimeError(
                            "Per-sample loss/weight shape mismatch: "
                            f"{tuple(per_sample_step_loss.shape)} vs {tuple(sample_weights.shape)}"
                        )
                    step_loss = (per_sample_step_loss * sample_weights).mean()
                    step_losses.append(step_loss)
                    student_latents = current_latents + dt * student_prediction

                loss = torch.stack(step_losses).mean()
                finite_loss = accelerator.gather(torch.isfinite(loss.detach()).reshape(1)).all()
                if not bool(finite_loss.item()):
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer step {global_step + 1}: "
                        f"{loss.detach().float().item()}"
                    )
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )
                    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at optimizer step {global_step + 1}: {grad_norm}"
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
                torch.tensor(
                    [step_seconds_local], device=accelerator.device, dtype=torch.float64
                )
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
                            "method": "D-OPSD" if args.teacher_context_mode == "vlm_q_image" else "D-OPCD",
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
        ZImagePipeline.save_lora_weights(
            str(output_dir), transformer_lora_layers=final_state
        )
        write_json(
            output_dir / "training_complete.json",
            {
                "global_step": global_step,
                "method": "D-OPSD" if args.teacher_context_mode == "vlm_q_image" else "D-OPCD",
                "student_context": "q",
                "teacher_context": teacher_context_description(args.teacher_context_mode),
                "teacher_context_mode": args.teacher_context_mode,
                "loss_type": args.loss_type,
                "student_adapter": STUDENT_ADAPTER,
                "context_sha256": sha256_file(data_jsonl),
                "image_manifest_sha256": (
                    sha256_file(image_manifest_jsonl) if image_manifest_jsonl is not None else None
                ),
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
