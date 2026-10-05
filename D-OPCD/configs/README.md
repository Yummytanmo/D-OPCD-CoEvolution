# Training Configuration

- [task.example.json](task.example.json): reusable D-OPCD training task.
- [dopsd_task.example.json](dopsd_task.example.json): D-OPSD with query-and-image
  teacher conditioning and endpoint loss.
- [e2e.example.json](e2e.example.json): training, validation checkpoint selection,
  and test evaluation.
- [runtime.json](runtime.json): trainer, Python and Accelerate paths, model
  defaults, and shared trainer settings.
- [accelerate_single_gpu.yaml](accelerate_single_gpu.yaml) and
  [accelerate_ddp.yaml](accelerate_ddp.yaml): single-GPU and multi-GPU layouts.

Copy `task.example.json` to an ignored `*.local.json` file. Set:

- `experiment_id`, `dataset`, and `selection`.
- `data_path` and `expected_count` to match the converted Agent data.
- `model_path` to the local base generator.
- `trainer_args`: teacher context mode, training steps, checkpoint interval,
  learning rate, LoRA settings, batch size, and gradient accumulation.

Paths are relative to `D-OPCD/`. Training data belongs in
`data/<dataset>/<data-id>/train.jsonl`. Task trainer settings override matching runtime settings.
D-OPCD uses teacher context mode `q_plus_p`. The D-OPSD template uses
`vlm_q_image` and `endpoint` loss; set `trainer_args.image_manifest_jsonl`
and `trainer_args.vlm_model_path` to the Agent image manifest and local
Qwen3-VL-4B-Instruct model. It shares the runtime's optimization settings.

From `D-OPCD/`:

```bash
cp configs/task.example.json configs/my-task.local.json
uv run python train/dopcd.py --config configs/my-task.local.json \
  --runtime-config configs/runtime.json
uv run python train/dopcd.py --config configs/my-task.local.json \
  --runtime-config configs/runtime.json --execute
```

The first Python command checks the configuration; the second starts training.
See the [runner guide](../runners/README.md) for
D-OPCD experiment definitions that include checkpoint selection and test
evaluation. The [baseline guide](../baselines/README.md#d-opsd) provides
D-OPSD setup and commands.
