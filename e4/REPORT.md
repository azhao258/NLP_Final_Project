# Part 2 / E4: Reasoning compression and latent-step explanation diagnostics

## Compression, accuracy, and latency across baselines

This section uses three fixed test subsets of 200 questions each. The step-level analysis below uses a separate set of 200 held-out validation questions

| Dataset | Model | Correct / total | EM | Latent steps | Reference type | Mean reference tokens | Length ratio | Mean / median / P95 latency (ms) | Parse failures / truncated |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| gsm8k | CODI_released | 74/200 | 37.0% | 6 | natural_language | 98.855 | 16.476 | 241.59 / 237.54 / 270.76 | 0/0 |
| gsm8k | SIM-CoT_released | 67/200 | 33.5% | 6 | natural_language | 98.855 | 16.476 | 247.83 / 244.47 / 278.34 | 0/0 |
| gsm8k | control_v1 | 7/200 | 3.5% | 4 | natural_language | 98.855 | 24.714 | 207.10 / 197.59 / 257.34 | 0/0 |
| gsm_hard | CODI_released | 18/200 | 9.0% | 6 | missing | N/A | N/A | 268.63 / 273.95 / 299.35 | 0/0 |
| gsm_hard | SIM-CoT_released | 18/200 | 9.0% | 6 | missing | N/A | N/A | 275.09 / 279.73 / 319.13 | 2/0 |
| gsm_hard | control_v1 | 1/200 | 0.5% | 4 | missing | N/A | N/A | 226.07 / 247.56 / 295.53 | 1/0 |
| svamp | CODI_released | 88/200 | 44.0% | 6 | equation | 10.855 | 1.809 | 242.70 / 251.23 / 280.69 | 0/0 |
| svamp | SIM-CoT_released | 85/200 | 42.5% | 6 | equation | 10.855 | 1.809 | 190.38 / 184.20 / 242.98 | 0/0 |
| svamp | control_v1 | 2/200 | 1.0% | 4 | equation | 10.855 | 2.714 | 208.12 / 197.55 / 254.14 | 0/0 |

The length ratio is total reference tokens divided by total latent steps for examples with references. GSM8K provides natural-language chains; SVAMP provides equations. GSM-Hard has no reference chain, so its ratio is N/A. This is not the length of explanations generated during evaluation.

Released models use 6 latent steps; v1 uses 4. The latent budget is 33.3% smaller, and the ratio on the same references is 50% larger. GSM8K accuracy is 37.0%/33.5% for released CODI/SIM-CoT versus 3.5% for v1; SVAMP accuracy is 44.0%/42.5% versus 1.0%. A larger ratio is not accompanied by reliable answer accuracy.

The released models have the same length budget. SIM-CoT has EM scores 3.5 percentage points lower on GSM8K and 1.5 points lower on SVAMP than CODI, with equal EM on GSM-Hard. No significance test was performed; small question-count differences do not establish stable superiority.

Released checkpoints and v1 differ in training scale and settings. This is a descriptive comparison, not a controlled experiment changing only 6 to 4 steps. Accuracy differences cannot be attributed to latent-step count alone.

The length ratio is not a speedup factor; no explicit CoT generation baseline was run. Timing used A6000 GPUs, BF16, batch size 1, greedy decoding, a 64-token cap, 5 warm-ups, and CUDA synchronization, but sessions and host conditions differed. In refreshed SVAMP results, SIM-CoT has lower median latency than both CODI and 4-step v1, so latent-step count alone does not explain latency. Auxiliary explanation decoding is excluded from answer latency.

## Auxiliary decoder and held-out examples

Analysis uses free-decoded v1 auxiliary-decoder outputs from e3_runs/control_diagnostics/validation.decoded.jsonl. This decoder provided step supervision during control training. Inputs are 200 validation questions excluded from training, including 6 correct and 194 incorrect answers. Saved outputs are reused; GPU inference was not rerun.

To reproduce the export, restore the control checkpoint and assets following Part 1, then run this command from the repository root with a new output directory:

```powershell
python e3/evaluate.py --checkpoint e3_runs/control/last.pt --split validation --explain --output e3_runs/v1_validation_recheck
```

This command is for reproduction and was not executed in this analysis. PAD positions have no reference steps to match; their outputs are retained to inspect template distributions.

Manual review and reference-data issues are documented in `e4/CASE_REVIEW.md`. Automated classification is not per-question manual semantic annotation.

## Scope and findings

Across 200 validation questions, 6 answers are correct. There are 478 valid reference slots, with 6/478 strict matches (1.26%).
Conservative equation compatibility holds for 9/478 slots, and for all valid slots in 1/200 questions.
Of 1,000 decoded slots, 1000 contain valid arithmetic. The main failure is mismatch with reference steps; these failures should not all be called arithmetic errors.
Answer parse failures: 0; truncated answers: 0; truncated explanations: 0.

## Scoring conventions

- Align decoded outputs, predictions, and validation records by question ID; check manifest SHA256, recorded baseline checkpoint SHA256, answers, and reference chains.
- Strict matching removes only whitespace and EOS, preserving equation markers, consistent with contributor A's summarize_handoff.py.
- Score explanation errors only at non-PAD slots. Retain PAD outputs for distribution analysis, but exclude them from error denominators.
- Operator matching compares operator multisets. Operand matching compares left-hand-side number multisets, retaining multiplicity but ignoring order. Result matching compares right-hand-side values. These are reference-relative metrics, not standalone semantic correctness judgments.
- Compare merged slots as whole segments: aggregate operators and operands and retain ordered lists of equation results. One generated equation does not fully reproduce multiple reference equations.
- Conservative compatibility requires valid equations, matching equation counts, operator and operand multisets, evaluated left-hand-side values, and right-hand-side results. It accepts 2*3 versus 3*2 but does not recognize all equivalent derivations or establish semantic correctness.
- Use an AST whitelist and exact Fraction arithmetic, supporting only numbers, parentheses, +, -, *, and /. Unsupported expressions are unparseable, not arithmetic errors. Reference equalities may fail because of intentional rounding and require manual review.
- missing_output means an empty output with a reference. reference_step_not_reproduced means failure of conservative compatibility, not proof of an internally missing step.
- frequent_template means the same normalized text at the same slot in at least 100/200 questions. This distribution label can coexist with correct steps and is not an independent error criterion.
- Categories overlap. Slot-level counts use 478 valid slots; question-level counts use 200 questions. Each slot uses all 200 outputs for template frequencies.

## Statistics by slot

| Slot | Consumed by reasoner | Valid references | Strict matches | Conservative matches | Unique outputs | Most common output (all 200 questions) |
|---|---|---:|---:|---:|---:|---|
| 0 | Yes | 200 | 4 | 5 | 5 | `<<2*15=30>>`: 80/200 |
| 1 | Yes | 159 | 1 | 2 | 2 | `<<3*2=6>>`: 188/200 |
| 2 | Yes | 82 | 1 | 2 | 2 | `<<3*2=6>>`: 187/200 |
| 3 | Yes | 29 | 0 | 0 | 2 | `<<3*2=6>>`: 175/200 |
| 4 | No | 8 | 0 | 0 | 2 | `<<3*2=6>>`: 177/200 |

Slots 0-3 contain 470 valid references; slot 4 contains 8. Slot 4 is exported after the last recurrence, not a fifth reasoning input.

Reference chains contain 483 equations. Slot 4 merges multiple equations in 2 questions. Generated outputs have 5 fewer equations than these merged references, despite every valid slot having nonempty output.

## Overlapping category counts

| Label | Valid slots / 478 | Questions / 200 |
|---|---:|---:|
| arithmetic_inconsistency | 0 | 0 |
| correct_answer_explanation_mismatch | N/A (question-level label) | 6 |
| fewer_equations_than_reference | 2 | 2 |
| frequent_template_on_valid_slot | N/A (question-level label) | 156 |
| intermediate_result_mismatch | 452 | 198 |
| missing_output | 0 | 0 |
| operand_mismatch | 468 | 199 |
| operator_mismatch | 288 | 172 |
| reference_arithmetic_inconsistency | 7 | 7 |
| reference_step_not_reproduced | 469 | 199 |
| unparseable_output | 0 | 0 |

Unobserved labels have count zero. Full definitions are in `e4/README.md`.

## Additional step-matching examples

These two examples illustrate a strict step match and conservative compatibility without exact text matching. Cases covered in the manual review, including all correctly answered questions, appear only in the next section to avoid duplication. This is illustrative post-analysis selection, not preregistered sampling; cases are not used to estimate prevalence.

### gsm8k-aug-train-2007: A strict step match does not establish full-chain correctness

In a basketball game, The Giants scored 3 points in the first quarter and twice that amount in the second quarter.  The opponents, The Eagles, scored half the amount that The Giants scored in the first quarter and three times the amount that The Giants scored in their second quarter.  How many points did both teams score in total?

- Gold: 29; prediction: 8; answer incorrect.
- Reference chain: `<<3*2=6>> <<.5*3=1.5>> <<3*6=18>> <<3+6+2+18=29>>`
- Decoded slots 0-4: `<<3*2=6>> / <<3*2=6>> / <<3*2=6>> / <<3*2=6>> / <<3*2=6>>`
- Non-PAD slots: 4; strict matches: 1; conservative matches: 1.
- Labels: intermediate_result_mismatch, operand_mismatch, operator_mismatch, reference_step_not_reproduced, frequent_template_on_valid_slot.

### gsm8k-aug-train-2013: Different text with conservative compatibility

Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?

- Gold: 16; prediction: 8; answer incorrect.
- Reference chain: `<<2*3=6>> <<6+2=8>> <<8*2=16>>`
- Decoded slots 0-4: `<<3*2=6>> / <<3*2=6>> / <<3*2=6>> / <<3*2=6>> / <<3*2=6>>`
- Non-PAD slots: 3; strict matches: 0; conservative matches: 1.
- Labels: intermediate_result_mismatch, operand_mismatch, operator_mismatch, reference_step_not_reproduced, frequent_template_on_valid_slot.

## Manual case interpretation and reference-quality review

This is the manual review. It examines the questions, reference chains, and decoded outputs below. Automated classifications for all 200 questions are in results/questions.csv; detailed outputs are in REPORT.md. 

### Core finding

Across 1,000 decoded slots, v1 produces only 5 distinct equations, all arithmetically valid multiplications. Of 478 valid reference slots, 6 strictly match (1.26%) and 9 pass conservative equation compatibility (1.88%). Operators, operands, and right-hand-side results mismatch the references in 288, 468, and 452 slots, respectively. These categories overlap and must not be added together.

The main observation is poor differentiation between questions or supervised steps in auxiliary explanations. The generated equations themselves perform multiplication correctly.

### 1. Multistep reasoning replaced by unrelated multiplication templates

**gsm8k-aug-train-2000**: Vanessa has 38 candies, which is 10 more than twice Sofia's amount. Find their total.

The reference computes `(38-10)/2=14`, then `38+14=52`, represented as `38-10=28`, `28/2=14`, and `38+14=52`. The model instead outputs `2*15=30`, `3*2=6`, and `3*2=6`, with final answer 8.

These multiplications are valid, but their operands and operations do not correspond to the problem. They also do not demonstrate use of a preceding step's result. Slots 3 and 4 are PAD and are not counted as reference steps that should match.

### 2. All 6 correct answers still have mismatched explanations

All IDs below have prefix `gsm8k-aug-train-`. Each of these 6 questions has zero strict matches and zero conservative matches across its valid slots.

| ID suffix | Question and manually checked calculation | Model answer | Why the answer does not establish explanation correctness |
|---|---|---:|---|
| 2004 | Half of 200 robots work, each using 2 batteries: 100*2=200 | 200 | Decoded 2*15=30 and 3*2=6 do not correspond to the robot counts |
| 2005 | The first participant spells 20 words, the second twice as many, and the third half the second: 20*2/2=20 | 20 | The explanation does not represent the calculations involving 20 and 40 |
| 2030 | Subtract chocolates costing 15 and flowers costing 25 from 50: 50-(15+25)=10 | 10 | The reference requires addition and subtraction, but the output is fixed multiplication |
| 2057 | Seven people need 3 sandwiches each; each box contains 5: ceil(21/5)=5 | 5 | The final reference value is 4.2 and gold is 5, a reasonable whole-box rounding rule; the explanation does not express this calculation |
| 2160 | A 300-liter tank fills at 10 liters per minute: 300/10=30 | 30 | Decoded 2*15=30 has the same result but different operands and operation |
| 2192 | Start with 20 cookies and give 4 to each of 3 friends: 20-4*3=8 | 8 | The explanation does not represent distribution or the remaining quantity |

Case 2160 shows that checking only the right-hand-side result can incorrectly accept unrelated reasoning. Case 2057 shows that a difference between the last reference value and gold does not automatically indicate a data error: buying whole boxes requires rounding up.

### 3. A fully matched explanation can still accompany an incorrect answer

**gsm8k-aug-train-2068**: Each loaf needs 2 cups of flour, so 15 loaves require 30 cups. The only valid reference slot is `2*15=30`, exactly matched by model slot 0, but the final answer is 15.

This is the only question among all 200 with every valid slot strictly matched, and it is not among the 6 correctly answered questions. Thus, no complete explanation match among correct answers does not mean no complete match anywhere in the dataset. This case limits the use of auxiliary decoded explanations as reliable evidence for final answers.

### 4. Long reference chains merged into slot 4

- **2048**: Average height of four people. There are 6 reference equations; slot 4 contains both `58+175=233` and `233/4=58.25`. The model generates one multiplication, missing one equation and mismatching the content.
- **2195**: Total revenue from cakes and cookies over two days. There are 9 reference equations; slot 4 merges the final 5. The model generates only one, so 4 equations are missing. Slot 2 happens to match cookie revenue, `3*2=6`, but the full chain does not match, and the final answer is 8 rather than 999.

Together these questions have 5 fewer generated equations than reference equations. This is an equation-count shortage within a supervision slot, not empty output. These merges explain why the dataset contains 483 reference equations but only 478 non-PAD supervision slots. Slot 4 is not fed into another reasoning recurrence and must not be called a fifth latent reasoning step.

### 5. Seven exact-equality flags: Separate approximation from data issues

Exact Fraction arithmetic flags one reference equality in each of the following 7 questions. All are finite-decimal approximations to repeating decimals; they must not simply be reported as seven reference arithmetic errors.

| ID suffix | Approximate reference equation | Review against question semantics |
|---|---|---|
| 2015 | 125/15=8.33 | Weekly net income is 15 and initial cost is 125; profit begins in week 9, so gold=9 is reasonable |
| 2021 | 5/1.5=3.33333 | Each mix makes 1.5 dozen cupcakes; 5 dozen requires buying 4 mixes, so gold=4 is reasonable |
| 2049 | 800/9=88.89 | "Eight times bigger than" is ambiguous: eight times the quantity gives 100, while eight times more gives 800/9, also a noninteger book count. Gold=89 requires source review by contributor A; do not directly replace it with 100 |
| 2088 | 40/3=13.33 | After reserving 10 from 50 for a gift, 13 books costing 3 each can be bought; gold=13 is reasonable |
| 2127 | 40/15=2.67 | Forty bananas at 15 per day last approximately 2.67 days. Complete days would be 2, but the wording permits a duration interpretation; retain gold=2.67 |
| 2154 | 1/0.3=3.33 | Combined rate is 1/5+1/10=0.3, giving approximately 3.33 hours |
| 2189 | 20/0.75=26.67 | Reading 20 books as exactly 75% implies a noninteger total. Gold=27 is rounded; the stated conditions are not fully consistent with discrete objects and should be flagged as a data-quality concern |

No official scores were changed for these cases; all 200 questions remain included. Finite-precision reference values limit exact-equality metrics, but the model outputs here are still unrelated multiplication templates. Reference rounding cannot explain the overall low match rate.
