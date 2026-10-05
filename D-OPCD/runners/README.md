# Generator Training and Evaluation

- `e2e.py`: trains D-OPCD, evaluates the configured checkpoints on validation,
  selects the highest-scoring checkpoint, and evaluates it on test.
- `generate.py`: generates and stages checkpoint images for the benchmark
  adapters in [evaluation/](../../evaluation/README.md).
- `common.py`: shared configuration and file helpers.

The full workflow supports GenEval, GenEval2, WISE Verified, and R2I-Bench.
Provide the trainer environment, generator weights, and the evaluator code,
weights, or judge endpoints needed for the selected benchmark.

## Experiment Definition

Copy [e2e.example.json](../configs/e2e.example.json) to an ignored local configuration:

```bash
cp configs/e2e.example.json configs/my-experiment.local.json
```

Set:

- `schema_version: 1`, `method: dopcd`, `experiment_id`, and `benchmark`.
- `training.task_template` and `training.runtime_config`: paths to the
  [training configurations](../configs/README.md).
- `splits.validation` and `splits.test`: each split's `path` and `count`, using the included [task manifests](../../evaluation/data/README.md).
- `checkpoint_steps`: checkpoint steps matching the task's training schedule.
- `generation`: model path, Accelerate executable, image and sampling settings,
  and optional `validation_parallelism`.
- `evaluation.python`: Python executable for scoring, including the evaluator dependencies.
- GenEval: `evaluation.evaluator_root`, `detector_model`, and `clip_cache`.
- GenEval2: `evaluation.evaluator_root` and `judge_model`.
- WISE: `evaluation.evaluator_root` and `judge` with `api_base`, `model`, and optional `workers`.
- R2I-Bench: `evaluation.judge` with `api_base`, `model`, and optional `workers`.

Use the same experiment ID and benchmark in the experiment and training task.
Paths are relative to `D-OPCD/`; the included task manifests are under
`../evaluation/data/<dataset>/`. External evaluator checkouts default to
`../evaluation/.assets/<benchmark>/`; change `evaluation.evaluator_root` for another installation.

## Run and Resume

From `D-OPCD/`:

```bash
uv run python runners/e2e.py --config configs/my-experiment.local.json
uv run python runners/e2e.py --config configs/my-experiment.local.json --execute
uv run python runners/e2e.py --config configs/my-experiment.local.json --resume my-run-id
```

The first command checks the configuration and inputs. The second creates a
new run; the third continues a saved run with the same configuration and inputs.
Replace `my-run-id` with the saved run ID.
Checkpoint selection uses validation only; test evaluation starts after all
configured validation checkpoints have been scored.
