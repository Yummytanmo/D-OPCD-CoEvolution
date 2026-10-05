# Baseline Training

- **Vanilla SFT**: trains on Agent-selected images using the original query;
  entry: [train/sft.py](../train/sft.py).
- **Diffusion-DPO**: trains on chosen and rejected images under the original
  query; entry: [train/flow_dpo.py](../train/flow_dpo.py).
- **D-OPSD**: the student receives the original query, and an EMA teacher receives
  the query and Agent-selected image encoded by Qwen3-VL. The teacher supervises
  clean endpoints on the student's own sampling trajectory;
  entry: [train/dopsd.py](../train/dopsd.py).

This directory implements SFT and DPO: `training.py` contains their training
loop, `objectives.py` their losses, `data.py` their loaders, and `launch.py`
their launcher. D-OPSD uses the on-policy trainer in
[src/train_dopcd.py](../src/train_dopcd.py).

## Data

Store context records and image manifests together in
`D-OPCD/data/<dataset>/<data-id>/`. Records are joined by `sample_id`.

- `train.jsonl`: `sample_id`, `source_sample_id`, `tag`, `original_query`, and
  `privileged_prompt`; produced by the Agent data converter.
- `sft.images.jsonl`: `sample_id`, `target_image`, `target_sha256`, and
  `target_prompt_role: privileged_prompt`; shared by SFT and D-OPSD.
- `flow_dpo.images.jsonl`: `sample_id`, `chosen_image`, `chosen_sha256`,
  `rejected_image`, `rejected_sha256`, `chosen_prompt_role: privileged_prompt`, and
  `rejected_prompt_role: original_query`.

Image paths are relative to the image manifest and stay within its data
directory. The chosen image uses the Agent-selected prompt; the rejected
image uses the original query with matching generation settings.

## SFT and Diffusion-DPO

[configs/zimage_fair.json](configs/zimage_fair.json) defines the method parameters
and reads shared settings from [configs/runtime.json](../configs/runtime.json).
Copy the [SFT task template](configs/sft_task.example.json) or
[DPO task template](configs/flow_dpo_task.example.json) and set `run_id`, `method`,
`context_jsonl`, `image_manifest_jsonl`, and `expected_count`.

From `D-OPCD/`, copy the SFT template:

```bash
cp baselines/configs/sft_task.example.json configs/my-sft.local.json
```

Prepare the inputs, edit the copied task, then check and train:

```bash
uv run python train/sft.py --config configs/my-sft.local.json --dry-run
uv run python train/sft.py --config configs/my-sft.local.json
```

Use `train/flow_dpo.py` with a matching DPO task. Both entries can use a custom
baseline protocol file:

```bash
uv run python train/sft.py --config configs/my-sft.local.json \
  --protocol /path/to/baseline-protocol.json
```

## D-OPSD

From `D-OPCD/`, copy the [D-OPSD task template](../configs/dopsd_task.example.json):

```bash
cp configs/dopsd_task.example.json configs/my-dopsd.local.json
```

Set the context data, expected count, image manifest, generator, and Qwen3-VL
model paths. The template uses `teacher_context_mode: vlm_q_image`,
`loss_type: endpoint`, EMA decay `0.9999`, and the four-step student trajectory.
The Agent prompt remains in the context records; D-OPSD conditions the student
on the query and the teacher on the query plus image.

```bash
uv run python train/dopsd.py --config configs/my-dopsd.local.json \
  --runtime-config configs/runtime.json
uv run python train/dopsd.py --config configs/my-dopsd.local.json \
  --runtime-config configs/runtime.json --execute
```

The first command checks the task definition; the second runs the full
preflight and starts training. See the [configuration guide](../configs/README.md)
for shared optimization settings.
