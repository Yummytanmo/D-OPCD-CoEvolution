# Data and Generation Utilities

- [build_prompt_context_data.py](build_prompt_context_data.py): converts Agent
  query–prompt pairs into `data/<dataset>/<data-id>/train.jsonl`.
- [validate_data.py](validate_data.py): checks the training record schema and selection policy.
- [audit_token_lengths.py](audit_token_lengths.py): measures student and teacher prompt lengths.
- [analyze_pq_text_diversity.py](analyze_pq_text_diversity.py): summarizes query and selected-prompt diversity.
- [launch_training.py](launch_training.py): prepares and launches training commands.
- [train.sh](train.sh): shell wrapper for the training launcher, using a task with `run_id`.
- [infer_geneval_test.py](infer_geneval_test.py): generates benchmark images with
  the base model or a trained LoRA; supports all four in-domain benchmarks and
  T2I-CompBench++.

Run these utilities from `D-OPCD/`. Convert the Agent's saved query–prompt pairs:

```bash
uv run python scripts/build_prompt_context_data.py \
  --input /path/to/agent-run/contexts.jsonl \
  --dataset geneval --data-id my-context-pairs \
  --query-field q --privileged-prompt-field p \
  --selection changed-only
```

Check the schema and token lengths with the paper's main teacher input:

```bash
uv run python scripts/validate_data.py \
  --data data/geneval/my-context-pairs/train.jsonl \
  --teacher-context-mode q_plus_p

uv run python scripts/audit_token_lengths.py \
  --data data/geneval/my-context-pairs/train.jsonl \
  --model-path /path/to/Z-Image-Turbo \
  --teacher-context-mode q_plus_p \
  --report /tmp/token-lengths.json
```

See the main README for [data preparation](../../README.md#data-preparation),
the [training guide](../README.md) for training entries, and the
[evaluation guide](../../evaluation/README.md) for benchmark scoring.
