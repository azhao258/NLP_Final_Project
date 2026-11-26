# E4 case review: Text matching, question semantics, and reference quality

This is the manual review. It examines the questions, reference chains, and decoded outputs below. Automated classifications for all 200 questions are in results/questions.csv; detailed outputs are in REPORT.md. 

## Core finding

Across 1,000 decoded slots, v1 produces only 5 distinct equations, all arithmetically valid multiplications. Of 478 valid reference slots, 6 strictly match (1.26%) and 9 pass conservative equation compatibility (1.88%). Operators, operands, and right-hand-side results mismatch the references in 288, 468, and 452 slots, respectively. These categories overlap and must not be added together.

The main observation is poor differentiation between questions or supervised steps in auxiliary explanations. The generated equations themselves perform multiplication correctly.

## 1. Multistep reasoning replaced by unrelated multiplication templates

**gsm8k-aug-train-2000**: Vanessa has 38 candies, which is 10 more than twice Sofia's amount. Find their total.

The reference computes `(38-10)/2=14`, then `38+14=52`, represented as `38-10=28`, `28/2=14`, and `38+14=52`. The model instead outputs `2*15=30`, `3*2=6`, and `3*2=6`, with final answer 8.

These multiplications are valid, but their operands and operations do not correspond to the problem. They also do not demonstrate use of a preceding step's result. Slots 3 and 4 are PAD and are not counted as reference steps that should match.

## 2. All 6 correct answers still have mismatched explanations

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

## 3. A fully matched explanation can still accompany an incorrect answer

**gsm8k-aug-train-2068**: Each loaf needs 2 cups of flour, so 15 loaves require 30 cups. The only valid reference slot is `2*15=30`, exactly matched by model slot 0, but the final answer is 15.

This is the only question among all 200 with every valid slot strictly matched, and it is not among the 6 correctly answered questions. Thus, no complete explanation match among correct answers does not mean no complete match anywhere in the dataset. This case limits the use of auxiliary decoded explanations as reliable evidence for final answers.

## 4. Long reference chains merged into slot 4

- **2048**: Average height of four people. There are 6 reference equations; slot 4 contains both `58+175=233` and `233/4=58.25`. The model generates one multiplication, missing one equation and mismatching the content.
- **2195**: Total revenue from cakes and cookies over two days. There are 9 reference equations; slot 4 merges the final 5. The model generates only one, so 4 equations are missing. Slot 2 happens to match cookie revenue, `3*2=6`, but the full chain does not match, and the final answer is 8 rather than 999.

Together these questions have 5 fewer generated equations than reference equations. This is an equation-count shortage within a supervision slot, not empty output. These merges explain why the dataset contains 483 reference equations but only 478 non-PAD supervision slots. Slot 4 is not fed into another reasoning recurrence and must not be called a fifth latent reasoning step.

## 5. Seven exact-equality flags: Separate approximation from data issues

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

