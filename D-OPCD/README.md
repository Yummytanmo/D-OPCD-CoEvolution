# D-OPCD

This module trains diffusion generators with D-OPCD and the comparison
baselines. In the paper's main D-OPCD setting, the student receives the
original query `q`, and the EMA teacher receives `q` together with the
Agent-selected prompt `p` along the student's denoising trajectory.

## Contents

- `src/`: datasets, prompt encoding, the D-OPCD objective, and numerical trainers.
- `train/`: training entries for D-OPCD, Vanilla SFT, Diffusion-DPO, and D-OPSD.
- [configs/](configs/README.md): training task template and shared runtime settings.
- [baselines/](baselines/README.md): baseline inputs, SFT/DPO implementations, and D-OPSD setup.
- [runners/](runners/README.md): generation, validation-based checkpoint selection, and test evaluation.
- [scripts/](scripts/README.md): data conversion, inspection, and generation utilities.

Agent feedback and benchmark scoring are described in the
[evaluation guide](../evaluation/README.md).

## Training

Install from the repository root:

```bash
uv sync --project D-OPCD
```

Generate training pairs from the Agent's output using the main README's
[data preparation instructions](../README.md#data-preparation).

From `D-OPCD/`, copy the task template and set its data, model, and training
parameters according to the [configuration guide](configs/README.md):

```bash
cp configs/task.example.json configs/my-task.local.json
uv run python train/dopcd.py --config configs/my-task.local.json \
  --runtime-config configs/runtime.json
uv run python train/dopcd.py --config configs/my-task.local.json \
  --runtime-config configs/runtime.json --execute
```

The first Python command prints a dry-run summary; the second starts training.
D-OPSD reuses the on-policy trainer with a query-and-image-conditioned EMA
teacher. Use the [baseline guide](baselines/README.md) for its setup and the
other methods, and the
[runner guide](runners/README.md) for the full training-to-test workflow.
