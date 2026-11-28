GRPO no-update sampling feasibility check

Purpose: frozen v1; fixed 64 training questions; 8 stochastic answers per question;
K=4; temperature=1.0; full softmax (no top-k/top-p truncation); 64-token cap.
No optimizer, backward, or model updates. State hashes before/after must match.
One greedy answer per question provides a same-subset comparator.
Sampling varies answer tokens, not deterministic latent trajectories. No decoder explanations are sampled.
Correctness uses the existing e1 numeric parser. Parse failures receive reward 0.
This is not GRPO training, not a test-set evaluation, and does not demonstrate generalization.

Server directory example: ~/experiment/NLP_Final_Project-main
Place this GRPO folder directly under the project root, beside e3 and e5.

1. If environment already exists:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/experiment/.conda312"

If missing, create it first:
mkdir -p ~/experiment
conda create -y --override-channels -c conda-forge -p "$HOME/experiment/.conda312" python=3.12 pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/experiment/.conda312"

2. If code is missing, download repository ZIP:
cd ~/experiment
curl --http1.1 -fL --retry 5 -o NLP_Final_Project-main.zip https://codeload.github.com/azhao258/NLP_Final_Project/zip/refs/heads/main
python -m zipfile -e NLP_Final_Project-main.zip .
Do not extract over modified project files; reuse existing code instead.

cd ~/experiment/NLP_Final_Project-main
python -m pip install --prefer-binary -r SIM-CoT/CODI/requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported()); print(torch.ones(1,device='cuda').item())"

3. Required assets:
assets/gpt2/{config.json,merges.txt,vocab.json,tokenizer.json,tokenizer_config.json,model.safetensors}
e3_runs/control/last.pt (frozen v1)
Use existing local verified copies via scp, or python e3/download_checkpoints.py --names control.
No P3/P4/shared_init needed. The script checks v1 and training-data hashes.

4. Smoke test (4 questions, 8 samples each):
python -u GRPO/sample_check.py --n 4 --output GRPO/results/smoke4

5. Full diagnostic (64 questions, 512 sampled answers + 64 greedy answers):
python -u GRPO/sample_check.py --output GRPO/results/v1_train64_g8

Output directory must not exist. Do not overwrite failed or previous runs.
Smoke IDs are the first four of the fixed 64 IDs; per-question seeds match full run.
Outputs: config.json, groups.jsonl, summary.json. A summary is written only on completion.

6. Return results:
tar -czf grpo_sampling_results.tar.gz GRPO/results
Download archive to local returned_results directory for analysis.

Interpretation:
groups_mixed_rewards: some correct and some wrong in a group; provides task reward contrast.
groups_all_wrong / groups_all_correct: zero within-group correctness variance.
groups_any_correct: sampled coverage, not greedy accuracy and not an unbiased arbitrary pass@k estimate.
sample_accuracy: correct among all sampled answers; not best-of-eight selected accuracy.
mean_unique_texts / mean_unique_parsed_answers: formatting vs numeric diversity.
parse_failures / capped: inspect raw outputs before modifying parsing or rewards.
Mixed groups are necessary evidence for this binary group-relative reward, not proof RL will help.
Do not repeatedly change sampling settings until results look favorable; document any separate protocol.

Validation: local compilation and synthetic summary-count tests only. Real CUDA smoke test required.
