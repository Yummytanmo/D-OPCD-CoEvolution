# Agent and Generator Launchers

- [experiment/train.sh](experiment/train.sh): runs Agent evolution in the foreground.
- [experiment/start-training.sh](experiment/start-training.sh): starts Agent evolution in a tmux session.
- [experiment/evaluate.sh](experiment/evaluate.sh): evaluates a saved harness
  without updating its Memory or Skills.
- [generator/z-image.sh](generator/z-image.sh): starts the local Z-Image generator;
  requires `Z_IMAGE_MODEL_PATH` and optionally accepts `Z_IMAGE_LORA_PATH`.
- [generator/z_image_background_load.py](generator/z_image_background_load.py):
  optional background generation, disabled by default and controlled by
  `background_load` in `configs/generator.z-image.json`.

Run commands from the repository root. Start Agent evolution in the foreground:

```bash
bash scripts/experiment/train.sh configs/evolution.local.json
```

To start it in a tmux session:

```bash
bash scripts/experiment/start-training.sh configs/evolution.local.json
```

Evaluate a frozen harness:

```bash
bash scripts/experiment/evaluate.sh configs/evolution.local.json test full latest
```

The evaluation wrapper defaults to validation, all variants, the latest
checkpoint, and the complete split. Set `MAX_TASKS` only for a shorter check.

Start the local generator in a separate terminal:

```bash
Z_IMAGE_MODEL_PATH=/path/to/Z-Image-Turbo bash scripts/generator/z-image.sh
```

See the main README for [Agent configuration](../README.md#installation),
the [D-OPCD utilities](../D-OPCD/scripts/README.md) for data conversion and
training tools, and the [evaluation guide](../evaluation/README.md) for scoring.
