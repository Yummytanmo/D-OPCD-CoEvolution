# Agent Task Splits

Each dataset directory contains `train.jsonl`, `validation.jsonl`, and
`test.jsonl`. Task counts below follow training / validation / test order:

- `geneval/`: 276 / 83 / 194.
- `geneval2/`: 400 / 120 / 280.
- `wise/`: 500 / 150 / 350 (WISE Verified).
- `r2ibench/`: 480 / 100 / 200.

Use training for Agent evolution, validation for harness or generator
checkpoint selection, and test for final evaluation. From the repository
root, configure the Agent input, for example:

```json
"manifests": ["evaluation/data/geneval/train.jsonl"]
```

## Record Fields

All records contain `sample_id`, `benchmark`, `split`, `prompt`, and
`inference_seed`, plus benchmark annotations:

- GenEval: `include`, `exclude`, and `tag`.
- GenEval2: `vqa_list`, `skills`, and `atom_count`.
- WISE: `prompt_id`, `category`, `subcategory`, and `explanation`.
- R2I-Bench: `checklist`, `category`, `subcategory`, `reference_caption`, and
  `assessment_point`.

GenEval and WISE retain `source_metadata` for offline scoring. Reference
annotations are passed to feedback services separately from the public task input.

## D-OPCD Training Data

The Agent writes query–prompt pairs to `runs/<run-id>/contexts.jsonl`.
The [data converter](../../D-OPCD/scripts/build_prompt_context_data.py) turns
these pairs into `D-OPCD/data/<dataset>/<data-id>/train.jsonl`.
See the main README's [data preparation instructions](../../README.md#data-preparation).
