# P0 diagnostic protocol

No optimizer or parameter update. No test split is read. Start with frozen
control_v1 (`e3_runs/control/last.pt`); inspect results before changing training.
Default: 32 training + 32 validation examples, sampled separately with seed 11.
The next sampled example cyclically supplies donor latents (no self donors).
Use the same n/seed for any later checkpoint comparison.

## Measurements

- Per-example norms and pairwise gradient dot products/cosines for the four
  existing weighted losses: answer CE, distillation, reference CE, explanation CE.
  Coordinates include trainable reasoner and projector parameters, with separate
  reports per group. Decoder-only parameters are excluded. Unused coordinates
  are zero; zero-norm cosines are null, not counted as agreement or conflict.
- FP32 master weights + BF16 autocast, dropout disabled for reproducible local
  gradient comparison. These are not stochastic training-gradient estimates.
- Decoder-only swap: each valid reference slot is read from normal vs donor z.
  Record teacher-forced NLL, free generations and exact token matches.
- Reasoner swap: retain recipient question cache, replace inputs z0..z3 with
  donor z0..z3, and recompute every subsequent latent call and answer cache.
  z4 is not consumed by the answer path. Report free answers and gold-answer
  NLL (excluding the fixed `The answer is:` prefix, including EOS).
- Normal inference must match existing generation. Self-swap must match normal
  inference exactly. Parameter version counters must remain unchanged, with no
  accumulated `.grad` tensors.

The source's returned diagnostic losses are detached. This script creates an
in-memory copy of the upstream forward with only four final detach assignments
removed. It fails closed if these assignments change. It does not modify the
vendored source. The CPU integration test compares original and exposed loss
values and verifies that component gradients sum to the total gradient.

## Run on the existing server environment

Use the same environment as E3/E5 (torch 2.7.1, transformers 4.52.4, peft 0.15.2).
The previously recorded project path is `/home/user/experiment`; verify it exists.

```bash
cd /home/user/experiment
python -m unittest discover -s e5 -p 'test_p0.py' -v
python e5/p0_diagnose.py --checkpoint e3_runs/control/last.pt --output e5_runs/p0_control_smoke --n 2 --smoke
python e5/p0_diagnose.py --checkpoint e3_runs/control/last.pt --output e5_runs/p0_control_v1 --n 32
```

Only start the full run after the test and smoke run succeed. Output directories
must not exist, preventing accidental overwrite. On interruption, use a new
directory; there is no resume. Individual records are flushed as JSONL.
Do not launch training concurrently with this diagnostic on the same GPU.

Outputs: `config.json` (checkpoint/data/source hashes and conventions),
`samples.json`, `parameters.json`, `gradients.jsonl`, `interventions.jsonl`,
and `summary.json` (written only on successful completion).

## Interpretation

Negative explanation/answer cosine is local conflict, not proof of persistent
training harm. Compare frequency and gradient size across both splits. Large
explanation gradients with negative cosine motivate testing P1, not declaring
P1 effective. Small greedy-answer changes alone are inconclusive at low accuracy;
inspect answer NLL deltas too. Small decoder swap effects do not establish that
latent contains no information. Swap sensitivity does not establish faithful
reasoning. Cross-question swaps are out-of-distribution interventions. This P0
does not measure explicit-CoT generation quality and cannot by itself justify P4.

Summary norms and cosines are means of per-example measurements, not the norm
or cosine of an averaged minibatch gradient. Raw records retain sample IDs for
paired inspection. A single final checkpoint cannot describe early training.
