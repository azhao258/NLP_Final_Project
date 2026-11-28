# P3/P4 validation-only diagnostic

No retraining. Use the same 200 frozen validation questions and BF16 weights for v1, P3 and P4. Results are exploratory because these questions have already informed development. Do not use test outcomes to tune this probe.

## Run at repository root on CUDA

```bash
python e3/download_checkpoints.py --names control
python e5/download_checkpoints.py --names p3 p4
python e5/check_reference_retrieval.py --checkpoint e3_runs/control/last.pt --output e5/probe_results/v1
python e5/check_reference_retrieval.py --checkpoint e5/p3/checkpoint/last.pt --output e5/probe_results/p3
python e5/check_reference_retrieval.py --checkpoint e5/p4/checkpoint/last.pt --output e5/probe_results/p4
```

Requires the project's existing Python environment and assets/gpt2. Output directories must not already exist. A smoke run can use --n 32 with separate output directories; main comparisons must use identical n.

## P3 reference path

The original question/CoT/answer packing is reused. Causal logits predict the next token; question tokens, fixed answer prefix and answer EOS are excluded from the relevant averages. Report token-weighted CoT NLL, numeric-answer NLL and token accuracy. Answer NLL is conditioned on gold CoT AND preceding gold answer tokens, not question-only answer accuracy. Compare with v1. Lower NLL means better gold-conditioned prediction, not automatically a better hidden-state distillation target or a reliable free-generating teacher. Existing question-only free-generation probes remain separate evidence.

## P4 retrieval

Collect projected z0-z4 from question-only recurrence. Collect last-layer reference states at closing >> positions, using P4's original position mapping and merged-tail rule. Report cosine positive-minus-hardest-negative margin and unique top-1 retrieval in two candidate sets: same-question different slots, and different-question same slot. Exact normalized duplicate reference strings are excluded as negatives. Algebraic equivalents are NOT fully detected, so inspect negative pairs manually before interpreting failures. Cosine ties count as failures, avoiding arbitrary argmax tie wins. Candidate counts vary; retain them in comparisons.

Report consumed slots z0-z3 separately from readout-only z4. No-reference slots are excluded. Each checkpoint uses its own reference encoder; cross-model score differences can reflect changes on either side and are not a pure latent-only comparison. Reference representations contain preceding context and may encode question identity. Retrieval is not proof of semantic correctness or causal faithfulness.

## Return files

Return all three directories under e5/probe_results: config.json, summary.json, reference.jsonl, retrieval.jsonl. representations.pt is useful for secondary checks but not needed for an initial table. Config records checkpoint/data/source hashes and IDs. Inspect hardest negatives and paired per-question NLL changes; do not rely solely on means.

## Local validation status

Python compilation and synthetic retrieval identity/tie tests passed. No real-checkpoint CUDA execution was possible on the preparation machine (CPU-only Torch; P3/P4 weights not present). GPU smoke testing is still required. P3 intermediate checkpoint can be passed via --checkpoint once restored, using a distinct output folder; no intermediate path is assumed here.
