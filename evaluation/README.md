# Evaluation and Agent Feedback

- [data/](data/README.md): input task splits for the four in-domain benchmarks.
- `client.py`: the Agent's HTTP feedback client.
- `feedback_agents/`: converts evaluator outputs into rewards and task feedback.
- `services/`: feedback-service entry points for GenEval, GenEval2, WISE, and R2I-Bench.
- `evaluators/`: single-image scoring runtimes used by the feedback agents.
- `benchmarks/`: offline scoring adapters called after generator evaluation.
- `ood/`: T2I-CompBench++ image staging, category scoring, and summaries.

## Feedback Services

Run commands from the repository root. Copy [config.example.json](config.example.json)
to the ignored local configuration:

```bash
cp evaluation/config.example.json evaluation/config.json
```

Set local model paths and the feedback language-model endpoint. Configure the
Agent's service URLs in its evolution configuration under `feedback.services`.
Start the service for the selected benchmark:

- **GenEval**, port `8101`. Install the official evaluator separately and set
  `GENEVAL_EVALUATOR_SCRIPT` to its `evaluate_images.py` path. Configure the
  detector and CLIP weights locally.

  ```bash
  uv run python -m evaluation.services.geneval_service
  ```

- **GenEval2**, port `8104`. Install dependencies, then start the service:

  ```bash
  uv sync --extra evaluation
  uv run python -m evaluation.services.geneval2_service \
    --model-path /path/to/Qwen3-VL-8B-Instruct
  ```

  Feedback rewards use the Soft-TIFA geometric mean; offline scoring reports
  Soft-TIFA AM and GM.

- **WISE Verified**, port `8105`. Supply a judge API through `WISE_JUDGE_API_BASE`,
  `WISE_JUDGE_MODEL`, and `WISE_JUDGE_API_KEY`. Feedback is a binary reward.

  ```bash
  uv run python -m evaluation.services.wise_service
  ```

- **R2I-Bench**, port `8108`. Set `R2I_JUDGE_API_BASE`, `R2I_JUDGE_MODEL`, and
  `R2I_JUDGE_API_KEY`. The service scores the task checklist and returns its
  weighted reward.

  ```bash
  uv run python -m evaluation.services.r2ibench_service
  ```

The task manifests contain the scoring annotations described in the
[data guide](data/README.md). The runner passes them to the feedback service
after image selection. Set `feedback.evolution_input` to `reward` for WISE;
the other benchmarks support `text`, `reward`, or `both`.

## Offline Scoring

The [generator runner](../D-OPCD/runners/README.md) calls the adapters under
`benchmarks/`; GenEval and GenEval2 also accept staged Agent runs through
`agent_run_dir`. Evaluator source code and model weights are separate dependencies.
Place external checkouts under `evaluation/.assets/<benchmark>/`, or set
`evaluation.evaluator_root` in the E2E configuration. Set `evaluation.python`
to the evaluator's Python executable; it may use a separate environment.
For the GenEval feedback service, the default evaluator script is
`evaluation/.assets/geneval/evaluation/evaluate_images.py`;
`GENEVAL_EVALUATOR_SCRIPT` overrides this path.
T2I-CompBench++ scoring utilities are under `ood/`.
