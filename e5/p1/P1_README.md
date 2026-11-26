# P1: explanation-loss ramp, full prefixes

本目录保存 P1（解释损失系数渐增）的代码、最终 checkpoint、验证/测试原始结果、诊断结果及最终版验证分析。本文按当前目录结构整理；历史记录中的服务器路径和哈希保留原样。

当前已完成 750 次更新，并已保存固定测试子集结果。独立 BF16 验证为 6/200；测试集 GSM8K、GSM-Hard、SVAMP 分别为 3/200、1/200、2/200。`P1_REPORT.md` 是此前的验证与诊断报告，`p1_analysis_final` 中的 final 指该分析的最终版本，不代表固定测试集分析。固定测试集与 v1 的完整比较见 [六模型统一报告](../test_suite_analysis_20261126/REPORT.md)。

P0 on control_v1 (32 train / 32 validation) found mean shared answer/explanation
gradient cosine about +0.118/+0.117, with 3/32 negative in each split. It does not
establish widespread conflict. Answer swaps changed 10/32 and 17/32 numeric
answers, without an original-latent advantage in mean gold-answer NLL. P1 is an
exploratory optimization-timing experiment, not a demonstrated conflict fix.

## Frozen method

- Start from original `e3_runs/shared_init.pt`, never from trained control or pilot.
- Keep full explanation prefixes. No masking hooks or operand reweighting.
- Preserve v1 data, order, seed 11, learning rate 5e-5, AdamW single decay group,
  batch size 1, accumulation 8, warmup 38, clipping 1, and 750 optimizer updates.
- For 1-based optimizer update u: `0.1 + 0.9 * min(u-1, 249) / 249`.
  Update 1 = 0.1, update 250 = 1.0; updates 251 through 750 = 1.0.
  All microbatches within one update have the same coefficient.
- Only `model.explain_loss_factor` changes. Upstream explanation normalization,
  answer/reference/distillation losses and model source remain intact.
- The common config retains the v1 target coefficient 1.0. The method and
  per-update coefficient are recorded separately to avoid ambiguous provenance.
- Use the final 750-update model for comparison; update250.pt is diagnostic only.
  Selection uses validation, not test. Prior test-influenced baseline selection
  and historical manifest discrepancy remain limitations.

The ramp's average coefficient over the full run is 0.85. This experiment changes
both timing and cumulative explanation-loss weight; a gain alone would not
isolate timing as the cause.

系数计算函数 `explain_factor()` 位于上一级 [p1_schedule.py](../p1_schedule.py)，由本目录的训练入口调用。该共用文件也包含 P2–P4 复用的基线预检查；它们复用预检查不代表采用 P1 渐增规则。共同训练配置见 [e3/config.json](../../e3/config.json)。

## Runtime and checks

Use `/home/user/experiment/.conda312/bin/python` (torch 2.7.1,
transformers 4.52.4, peft 0.15.2). There is no need to activate conda.

```bash
cd /home/user/experiment
./.conda312/bin/python -m unittest e5.p1.test_p1_schedule e5.p1.test_p1_model -v
./.conda312/bin/python e5/p1_schedule.py --require-weights
./.conda312/bin/python e5/p1/train_p1.py --output e5_runs/explain_ramp_pilot --pilot-updates 10
```

Check all tests pass and the pilot finishes with finite losses. Pilot coefficient
starts at 0.1 and reaches approximately 0.13253012 at update 10. The pilot uses
the full-run LR/ramp schedules; it does not compress them into 10 updates.

The local dependency-free schedule/budget tests and preflight passed. The model
integration test and CUDA pilot require the server environment. The model test
checks unchanged non-explanation losses, explanation weighting, and bitwise
CPU tiny-model resume across the ramp endpoint with optimizer/scheduler/RNG
restored. It does not claim GPU bitwise resume validation.

## Formal training, after successful pilot

```bash
./.conda312/bin/python e5/p1/train_p1.py --output e5_runs/explain_ramp_v1
```

Use a separate output directory: this restarts from shared_init, not pilot weights.
If interrupted, resume the SAME run with:

```bash
./.conda312/bin/python e5/p1/train_p1.py --output e5_runs/explain_ramp_v1 --resume
```

Resume restores weights, optimizer, scheduler and RNG, and derives the next
coefficient from the absolute completed-update count. It never restarts the ramp.
Checkpoints are saved atomically every 50 updates and at epoch/end boundaries.
Logs newer than the saved checkpoint are discarded on resume. Do not run two
processes against the same output directory. No test evaluation is launched.

`train_log.jsonl` records total and component weighted losses, coefficient,
`raw_explain_loss`, learning rate, gradient norm and epoch-end validation.
`last.pt` is resumable. `update250.pt` contains model weights/config for optional
P0 diagnosis, without optimizer state; it adds storage but no optimizer updates.
At initialization use original shared_init for optional diagnosis, at 750 use
last.pt. P0 uses standardized coefficient 1.0, not the early ramp coefficient.
`sessions.jsonl` measures loop time and peak allocated GPU memory. Validation and
checkpoint I/O are included; loading time is excluded.

以上是训练脚本的输出说明，不是当前本地文件清单。当前目录没有单独的 `training/`、`train_log.jsonl`、`sessions.jsonl` 或 `update250.pt`；不能将其写成已归档交付。当前已归档的模型文件是 `checkpoint/last.pt`。

## Validation after training

Use the existing evaluator explicitly with `--split validation` (its default is
test, so do not omit this flag). Compare both models under the same BF16 policy:

```bash
./.conda312/bin/python e3/evaluate.py --checkpoint e3_runs/control/last.pt --output e5_runs/p1_control_validation --split validation --explain
./.conda312/bin/python e3/evaluate.py --checkpoint e5_runs/explain_ramp_v1/last.pt --output e5_runs/p1_validation --split validation --explain
./.conda312/bin/python e5/p0_diagnose.py --checkpoint e5_runs/explain_ramp_v1/last.pt --output e5_runs/p0_p1_v1 --n 32 --seed 11
```

Review answers, reference-step/operand correspondence, arithmetic consistency
and paired latent-swap effects. Template diversity alone is insufficient.
Do not use the old masking-specific `e5/prefix_mask/evaluate.py` for P1; its provenance
validator deliberately requires a prefix-mask checkpoint.

## 已归档模型的重新评测

下面命令从项目根目录运行，使用整理后的 checkpoint 路径。输出到新的目录，避免覆盖现有归档；重新评测需要相应运行环境及 GPU。

```bash
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p1/checkpoint/last.pt --output e5_runs/p1_validation_recheck --split validation --explain
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p1/checkpoint/last.pt --output e5_runs/p1_test_recheck --split test --limit 200 --explain
./.conda312/bin/python e5/p0_diagnose.py --checkpoint e5/p1/checkpoint/last.pt --output e5_runs/p1_diagnostics_recheck --n 32 --seed 11
```

独立评测使用 BF16 权重和 autocast；诊断使用 FP32 master 参数和 BF16 autocast，关闭 dropout，不更新参数。两种协议应分别比较。测试延迟不含分词、文本解码、解释导出及模型加载，详见 `test/config.json`；六模型已有延迟来自不同会话，不能据此直接归因于方法带来的加速。

## 当前目录的每个文件

以下相对路径均以 `e5/p1/` 为起点。当前共有 30 个交付文件及 2 个 Python 缓存文件；旧版 `p1_analysis/` 和 `FILES.md` 已不在此目录。

### 根目录：代码与说明

| 文件 | 内容与用途 |
|---|---|
| [__init__.py](__init__.py) | Python 包标记，支持 `e5.p1` 下的模块导入。 |
| [configuration.json](configuration.json) | 从冻结设置汇总的配置参考：共同超参数及 P1 的 0.1→1.0、250 次更新渐增规则；不替代历史训练 contract，也不是训练入口默认读取的配置文件。 |
| [P1_README.md](P1_README.md) | 本文件：方法假设、训练与评测命令、当前文件说明及共用分析入口。 |
| [P1_REPORT.md](P1_REPORT.md) | 历史验证与诊断分析报告，解释 P1 与 v1 的答案、步骤及状态替换结果；其中当时的下载/实验进度不作为当前状态。固定测试结论应查看六模型统一报告。 |
| [train_p1.py](train_p1.py) | P1 训练入口：从共同初始化训练，逐次设置解释损失系数，记录损失和验证结果，保存模型并支持恢复优化器、调度器及随机状态。 |
| [test_p1_schedule.py](test_p1_schedule.py) | 检查系数边界、单调性、恢复后的更新索引，以及每轮 2000 题、共 750 次更新的样本预算。 |
| [test_p1_model.py](test_p1_model.py) | 使用小型 CPU 模型检查解释损失加权、其他损失不变，以及跨渐增终点恢复训练的一致性；不等于真实 GPU 训练的逐位一致性验证。 |

两个 `test_p1_*.py` 是代码正确性测试，与下方 `test/` 中的数学题测试集结果不同。

### checkpoint：最终模型

| 文件 | 内容与用途 |
|---|---|
| [checkpoint/last.pt](checkpoint/last.pt) | 750 次更新后的最终 checkpoint，约 2.23 GB。训练保存格式包括模型、优化器、调度器、随机状态、配置及训练 contract，支持恢复训练。各评测配置记录的 checkpoint SHA256 为 `4276c57a336117c29e8592c98ed1f4262fe37b05c3e3d8ed287bef9fcd2871f8`。 |

### validation：200 道验证题

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split validation --explain` 生成。

| 文件 | 内容与用途 |
|---|---|
| [validation/config.json](validation/config.json) | 评测模型哈希、训练更新数、配置、验证划分及运行环境。 |
| [validation/summary.csv](validation/summary.csv) | 验证汇总：6/200 正确、准确率 3%，以及隐式步数、参考/答案长度、延迟、解析失败及截断数。 |
| [validation/validation.predictions.jsonl](validation/validation.predictions.jsonl) | 200 道题逐题的原始输出、提取答案、正确性、生成 token、长度和延迟等记录。 |
| [validation/validation.decoded.jsonl](validation/validation.decoded.jsonl) | 200 道题的解码解释、参考信息及监督槽位，用于后续步骤级评分。 |

### test：固定测试子集

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split test --limit 200 --explain` 生成。三个数据集各 200 题，共 600 题；`decoded` 文件是解释输出，步骤评分和相对 v1 的收益分析由共用分析脚本完成。

| 文件 | 内容与用途 |
|---|---|
| [test/config.json](test/config.json) | 测试来源与条件：checkpoint、数据哈希、BF16、batch size 1、贪心生成、5 次预热及 CUDA 同步计时范围。 |
| [test/summary.csv](test/summary.csv) | 三个数据集的答案准确率、长度及延迟汇总；GSM8K 3/200、GSM-Hard 1/200、SVAMP 2/200。 |
| [test/gsm8k.predictions.jsonl](test/gsm8k.predictions.jsonl) | GSM8K 200 道题的逐题答案、正确性、长度及延迟。 |
| [test/gsm8k.decoded.jsonl](test/gsm8k.decoded.jsonl) | GSM8K 200 道题的解码解释及参考信息，用于步骤级分析。 |
| [test/gsm_hard.predictions.jsonl](test/gsm_hard.predictions.jsonl) | GSM-Hard 200 道题的逐题答案、正确性、长度及延迟。 |
| [test/gsm_hard.decoded.jsonl](test/gsm_hard.decoded.jsonl) | GSM-Hard 200 道题的解码解释；当前导出缺少参考步骤，不能按同样方式计算参考步骤匹配分数。 |
| [test/svamp.predictions.jsonl](test/svamp.predictions.jsonl) | SVAMP 200 道题的逐题答案、正确性、长度及延迟。 |
| [test/svamp.decoded.jsonl](test/svamp.decoded.jsonl) | SVAMP 200 道题的解码解释及参考算式；整题参考算式不等同于完整的多步推理标注。 |

### diagnostics：32 道训练题＋32 道验证题的机制诊断

由 [p0_diagnose.py](../p0_diagnose.py) 加载 P1 最终 checkpoint 生成，固定 `n=32`、`seed=11`，分别从训练和验证数据抽样。检查损失梯度关系及跨题隐式状态替换的影响，不训练模型、不读取测试集；小样本机制诊断不替代正式效果评测。

| 文件 | 内容与用途 |
|---|---|
| [diagnostics/config.json](diagnostics/config.json) | 诊断 checkpoint、抽样设置、梯度与替换协议、代码哈希及运行环境。 |
| [diagnostics/samples.json](diagnostics/samples.json) | 实际抽取的两组题目及 ID，确保跨模型使用同一批题。 |
| [diagnostics/parameters.json](diagnostics/parameters.json) | 参与梯度分析的参数名称和形状，不是模型权重。 |
| [diagnostics/gradients.jsonl](diagnostics/gradients.jsonl) | 64 道题各项损失的值、梯度范数、梯度内积和余弦，用于分析局部梯度关系。 |
| [diagnostics/interventions.jsonl](diagnostics/interventions.jsonl) | 64 道题使用本题或其他题隐式状态时的答案、解码解释及目标序列 NLL 等记录。 |
| [diagnostics/summary.json](diagnostics/summary.json) | 分训练/验证子集汇总梯度与状态替换结果，并记录无参数更新、自身替换一致等检查及局限。 |

### p1_analysis_final：验证和诊断的最终版对照分析

由 [analyze_p1.py](../analyze_p1.py) 生成，输入为 v1/P1 的 200 题验证输出和各自的 32＋32 诊断结果。相比旧版，补充了“解释发生变化但数学上等价”的统计；此目录不包含固定测试集的六模型分析。

| 文件 | 内容与用途 |
|---|---|
| [p1_analysis_final/summary.json](p1_analysis_final/summary.json) | v1/P1 验证指标、答案得失、步骤变化及诊断汇总；包括解释变化后的等价性分析。 |
| [p1_analysis_final/answer_pairs.csv](p1_analysis_final/answer_pairs.csv) | 200 道验证题的配对答案和正确性，用于定位新增答对、丢失和共同答对的题目。 |
| [p1_analysis_final/step_pairs.csv](p1_analysis_final/step_pairs.csv) | 200 题×5 槽位，共 1000 条步骤对照，包含参考、有效性、生成文本、严格匹配及算术有效性。有效参考槽位为 478，不能将 1000 当作参考步骤评分分母。 |
| [p1_analysis_final/provenance.json](p1_analysis_final/provenance.json) | 原始输入、分析代码、checkpoint 和验证数据的哈希及核验状态。保留生成当时的服务器备份路径和代码版本。 |

### Python 自动缓存

| 文件 | 内容与用途 |
|---|---|
| `__pycache__/__init__.cpython-312.pyc` | 包导入时自动生成的 Python 3.12 缓存，可重新生成，不需要 push。 |
| `__pycache__/test_p1_schedule.cpython-312.pyc` | 系数测试模块的 Python 3.12 缓存，可重新生成，不需要 push。 |

## 共用分析与提交范围

- [analyze_p1.py](../analyze_p1.py)：历史验证及诊断比较；默认读取项目根目录的 `server_backup_p1`，不是直接读取当前整理后的 P1 目录。
- [analyze_test_suite.py](../analyze_test_suite.py)：固定测试子集上的六模型统一比较，复用 [analyze.py](../analyze.py) 中的指标和步骤分析函数。
- [六模型统一报告](../test_suite_analysis_20261126/REPORT.md) 与 [汇总表](../test_suite_analysis_20261126/summary.csv)：查阅 P1 与 v1 的测试准确率、长度、延迟、步骤及分题型结果。
- [六模型延迟说明](../SIX_MODEL_LATENCY_SUMMARY.md)：现有测量的统一汇总及不同会话的比较限制。

建议提交本目录的代码、说明、配置和所有 JSON/JSONL/CSV 结果；`p1_analysis_final/` 四个文件应一起保留。`checkpoint/last.pt` 单独提供下载地址，不通过普通 Git 提交；`__pycache__/` 不提交。训练/分析还依赖上一级共用代码、E3/E4 代码及基础模型资源，仅复制本目录不足以独立复现。
