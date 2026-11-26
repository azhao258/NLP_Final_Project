# P3: reference-only warm start, then joint training

## 当前进度与结果

本目录保存 P3（参考路径预热后联合训练）的实现、最终 checkpoint、训练记录、验证/测试输出，以及两类机制诊断。当前已完成 750 次更新，最终权重已归档，固定测试子集的六模型步骤评分和分题型分析也已完成。正式 control 仍为 v1。

| 评测范围 | v1 答对数 | P3 答对数 | 说明 |
|---|---:|---:|---|
| 独立 BF16 验证集 | 6/200 | 2/200 | 严格步骤匹配分别为 6/478、5/478。 |
| GSM8K 固定测试子集 | 7/200 | 3/200 | 严格步骤匹配分别为 4/634、2/634。 |
| GSM-Hard 固定测试子集 | 1/200 | 2/200 | 缺少参考步骤，参考步骤匹配为 N/A。 |
| SVAMP 固定测试子集 | 2/200 | 5/200 | 整题参考算式不作为完整多步推理标注评分。 |

测试结果存在不同方向的变化：P3 在 GSM-Hard、SVAMP 上答对数增加，在 GSM8K 上下降，不能写成普遍有效。详见 [六模型统一报告](../test_suite_analysis_20261126/REPORT.md) 和 [分题型结果](../test_suite_analysis_20261126/problem_types.csv)。单种子、小样本和历史基线选择参考过测试结果的限制仍需披露。

训练日志末次验证为 3/200，使用 FP32 master 参数加 BF16 autocast；上述独立验证为 2/200，使用 BF16 权重加 autocast，不能混用两种评测口径。当前 `checkpoint/` 只有 `last.pt`；第 150 次更新的诊断结果已归档，但 `reference150.pt` 不在当前目录。

P3 的假设是先训练显式参考路径，可能为后续自蒸馏提供更好的参考。已有自由生成诊断没有建立“可靠强教师”的证据；这是待实验检验的训练顺序假设，而不是已证明成立的前提。

## Frozen design

Start from the original `e3_runs/shared_init.pt`, never a trained P1/P2 model.
Updates 1–150 optimize ONLY the existing reference path CE (masked question,
gold CoT and final answer targets). Updates 151–750 use the unchanged CODI/SIM-CoT
joint loss with all original factors equal to 1. Use full explanation prefixes,
no P1 ramp, no P2 token weighting, and four latent iterations at inference.

Same 2,000 training examples, seed 11, sample order, accumulation 8, optimizer,
38-update LR warmup and continuous 750-update LR schedule as control_v1.
Do not restart the optimizer or scheduler at update 151. This deliberately
reduces latent/decoder training to 600 updates; it is not an equal latent-update
comparison. Reference-only execution consumes fewer dropout RNG draws than
joint execution; same seed/order does not imply identical dropout masks.

The reference-only forward reconstructs the original shifted masked CE because
CODI returns a detached diagnostic `ref_ce_loss`. Projector/decoder have no
gradient in this phase and are skipped by AdamW, including weight decay.
Unused phase-one losses are logged as null, not measured zeros.

Final selection is always update 750. `reference150.pt` is diagnostic only.
The checkpoint stores model/optimizer/scheduler/RNG and the completed stage.
Resume derives the next stage from the global number of completed updates.
Source/data/order/init identities are checked and recorded. Frozen source files
are not edited. Historical manifest mismatch remains disclosed by preflight.

## 运行环境与目录要求

以下命令用于复现，不表示当前实验尚未完成。示例以项目根目录 `/home/user/experiment`、`./.conda312/bin/python` 为准，按实际环境调整；历史训练环境为 torch 2.7.1、transformers 4.52.4、peft 0.15.2。

同步代码应保留 `e5/p3/` 的包结构，不再把 P3 文件平铺到 `e5/`。还需共同的 E3/E5 代码、数据、基础模型资源和原始 `e3_runs/shared_init.pt`。[p1_schedule.py](../p1_schedule.py) 提供基线预检查，不表示采用 P1 渐增；[p3_reference_probe.py](../p3_reference_probe.py) 是共用参考路径诊断工具，仍在上一级。共同配置见 [e3/config.json](../../e3/config.json)。

## First: tests and reference-path diagnosis (remote Bash)

```bash
cd /home/user/experiment
./.conda312/bin/python -m unittest e5.p3.test_p3_stage e5.p3.test_p3_model -v && \
./.conda312/bin/python e5/p3_reference_probe.py --checkpoint e3_runs/control/last.pt --output e5_runs/p3_control_reference_recheck --n 32
```

The probe uses the same seed-11 validation subset as P0, BF16 weights + autocast,
and paired free generation. The reference prompt contains only the question,
without a bot token or gold CoT. The explicit path has up to 512 new tokens
(context permitting); latent answers retain the original 64-token limit.
Reference answers require an answer delimiter. It exports raw generations,
correctness, parse failures and truncation counts. Read the generated CoT too:
correct final answers alone do not establish correct intermediate reasoning.
These are differing inference budgets, not a speed comparison.

If most reference outputs truncate or fail parsing, the free-generation evidence
is inconclusive. If reference performance is also weak, P3 remains exploratory;
do not describe a strong teacher as an established fact. This probe does not
automatically change the predeclared training design or inspect test data.

## Pilot through the actual phase boundary

```bash
./.conda312/bin/python e5/p3/train_p3.py --output e5_runs/reference_warmstart_pilot --pilot-updates 151
```

Use a fresh directory. Verify updates 1–150 have `stage=reference_only`, only
`ref_ce_loss` is active, and update 151 has `stage=joint` with finite full losses.
A 10-update pilot would not exercise the phase transition. Pilot uses the full
750-step LR schedule. Do not continue the pilot into the formal run.

## 重新训练与阶段恢复

```bash
./.conda312/bin/python e5/p3/train_p3.py --output e5_runs/reference_warmstart_reproduction
```

For interruption recovery, rerun that exact command with `--resume`. Do not
allow multiple writers in one output directory. Use a fresh directory for a new run.

## 已归档最终模型的验证、测试与诊断

下列命令使用当前归档路径并写入新目录，避免覆盖现有交付结果：

```bash
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p3/checkpoint/last.pt --output e5_runs/p3_validation_recheck --split validation --explain
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p3/checkpoint/last.pt --output e5_runs/p3_test_recheck --split test --limit 200 --explain
./.conda312/bin/python e5/p0_diagnose.py --checkpoint e5/p3/checkpoint/last.pt --output e5_runs/p3_diagnostics_recheck --n 32 --seed 11
./.conda312/bin/python e5/p3_reference_probe.py --checkpoint e5/p3/checkpoint/last.pt --output e5_runs/p3_final_reference_recheck --n 32
```

如需重新生成第 150 次更新的 reference probe，必须先取得对应的 `reference150.pt`，再将它传给 `p3_reference_probe.py --checkpoint`。最终 `last.pt` 不能替代阶段 checkpoint。当前 `p3_reference150/` 的三个结果文件可以直接用于分析，无需仅为读取结果而重新训练。

Compare with existing control validation and P0 outputs; do not retrain control.
Use paired answer correctness, strict steps/operands/results, arithmetic validity
and latent intervention effects together. Training-time FP32-master validation
must not be mixed with standalone BF16-weight evaluation. P0 retains its own
FP32-master protocol; compare P0 results within that protocol.
Keep test data out of selection. Single seed, small validation set and earlier
test-influenced baseline selection remain limitations.

训练会话记录已归档在 `training/sessions.jsonl`：750 次更新的训练循环约 3476.25 秒（57.94 分钟），峰值 allocated 显存约 4.72 GiB。计时包括验证和 checkpoint I/O，不含模型加载，也不等同于总租赁时间。

## 历史实现核验记录

Local implementation checks covered boundaries, shifted/masked reference CE,
inactive-parameter AdamW behavior, continuous scheduler/RNG resume across the
transition, unchanged joint routing, and frozen train/validation/source checks.
The server subsequently reported five passing tests, completed the 151-update
pilot, and completed the formal 750-update run and all validation diagnostics.
These are historical records, not a claim that this documentation update reran training or tests.

## 历史验证分析（2026-11-23）

Downloaded exports were audited locally with `analyze_p3.py`. Standalone BF16
validation: P3 2/200 answers correct versus control 6/200; strict explanation
matches 5/478 versus 6/478. Reference free-generation answer matches: control
3/32, P3 update 150 0/32, P3 update 750 1/32. Every answer-matching reference
output contained arithmetic errors. P0 showed positive mean answer NLL deltas
under latent swaps, but no validation answer or strict-step accuracy benefit.
Retain control_v1; this is an exploratory negative result, not a successful
improvement or proof that all reference warm-start designs fail.

See [P3_REPORT.md](P3_REPORT.md) for methods, paired outcomes, generation review,
precision distinctions, training cost and limitations. Machine-readable results
and provenance are in [e5/p3_analysis](../p3_analysis/). Recompute into a fresh directory:

```powershell
python e5/analyze_p3.py --output e5/p3_analysis_recheck
```

该历史分析生成时尚未下载或重新哈希权重；这一描述不代表当前没有最终模型。当前已归档 `checkpoint/last.pt`，但仍未在本目录找到 `reference150.pt`。原始报告、provenance 和训练 contract 保留生成时内容，不随 README 更新改写。

## 当前目录的每个文件

以下相对路径以 `e5/p3/` 为起点。当前共有 36 个交付文件和 3 个缓存文件；`FILES.md` 已不在目录。下列“提交”表示交付建议，不表示已经执行 push。

### 根目录：实现与说明

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [__init__.py](__init__.py) | Python 包标记，支持从 `e5.p3` 导入模块。 | 提交 |
| [configuration.json](configuration.json) | 汇总共同超参数及 P3 的 150＋600 阶段规则；是便于阅读的配置参考，不替代历史 contract，也不是训练入口默认读取的配置。 | 提交 |
| [p3_stage.py](p3_stage.py) | 核心实现：根据已完成更新数选择阶段；前 150 次只计算参考路径 CE，第 151 次起调用原始联合前向；提供冻结协议检查。 | 提交 |
| [train_p3.py](train_p3.py) | 训练入口：共同初始化、分阶段训练、验证、日志、checkpoint 保存和断点恢复；第 150 次更新后另存诊断快照。 | 提交 |
| [test_p3_stage.py](test_p3_stage.py) | 检查阶段边界、参考损失的标签遮蔽/移位、未参与参数不被 AdamW 更新、跨阶段优化器/调度器/RNG 恢复及冻结协议。 | 建议提交 |
| [test_p3_model.py](test_p3_model.py) | 小型模型集成测试，核对参考损失与原有实现的一致性，以及联合阶段调用原模型的行为。 | 建议提交 |
| [P3_README.md](P3_README.md) | 本文件：方法、进度、复现命令、全部文件用途和提交范围。 | 提交 |
| [P3_REPORT.md](P3_REPORT.md) | 历史验证和诊断报告，比较 control/P1/P2/P3，分析步骤、状态替换和参考路径生成；当前固定测试结果应查看六模型统一报告。 | 作为历史分析提交 |

两个 `test_p3_*.py` 是代码正确性测试，不是下方 `test/` 中的数学题测试集评测。

### checkpoint：最终模型

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [checkpoint/last.pt](checkpoint/last.pt) | 最终 750 次更新的 checkpoint，约 2.23 GB。训练保存格式包括模型、优化器、调度器、随机状态、配置、contract 和阶段信息。评测记录中的 SHA256 为 `54237b3eccad670a6f71f3bc26f01e53957075cb5fcf5de44a20be18a2ce33c0`。 | 单独提供下载，不通过普通 Git 提交 |

### training：实际训练记录

由 `train_p3.py` 生成。当前这里只归档了三份记录；脚本生成的其他辅助文件不能视为已经保存在本目录。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [training/contract.json](training/contract.json) | 训练超参数、阶段规则、预算、初始化/数据/样本顺序/代码哈希及环境；历史代码路径反映当时结构。 | 提交 |
| [training/train_log.jsonl](training/train_log.jsonl) | 750 条更新记录：stage、损失、学习率、梯度范数、耗时及阶段性验证。前 150 次未计算的损失记录为 null，不是测得零损失。 | 提交 |
| [training/sessions.jsonl](training/sessions.jsonl) | 会话起止更新数、循环耗时及峰值 allocated 显存，包含计时范围说明。 | 提交 |

### validation：200 道验证题

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split validation --explain` 生成。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [validation/config.json](validation/config.json) | 独立验证的模型哈希、更新数、配置、数据和运行环境。 | 提交 |
| [validation/summary.csv](validation/summary.csv) | 200 题汇总：2/200 正确（1%）、长度、延迟、解析失败与截断统计。 | 提交 |
| [validation/validation.predictions.jsonl](validation/validation.predictions.jsonl) | 逐题原始输出、提取答案、正确性、token 数及延迟等。 | 提交 |
| [validation/validation.decoded.jsonl](validation/validation.decoded.jsonl) | 逐题解码解释、参考信息及监督槽位，供步骤级分析使用。 | 提交 |

### test：固定测试子集

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split test --limit 200 --explain` 生成。三个数据集各 200 题，共 600 题。`decoded` 是解释输出，步骤评分与相对 v1 的收益比较由共用分析代码完成。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [test/config.json](test/config.json) | checkpoint、数据哈希、BF16 权重及 autocast、batch size 1、贪心生成、预热和计时范围。 | 提交 |
| [test/summary.csv](test/summary.csv) | GSM8K、GSM-Hard、SVAMP 的准确率、长度及延迟汇总；正确数依次为 3/200、2/200、5/200。 | 提交 |
| [test/gsm8k.predictions.jsonl](test/gsm8k.predictions.jsonl) | GSM8K 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/gsm8k.decoded.jsonl](test/gsm8k.decoded.jsonl) | GSM8K 逐题解释和参考信息，用于步骤评分。 | 提交 |
| [test/gsm_hard.predictions.jsonl](test/gsm_hard.predictions.jsonl) | GSM-Hard 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/gsm_hard.decoded.jsonl](test/gsm_hard.decoded.jsonl) | GSM-Hard 逐题解释；缺少参考步骤，参考步骤匹配记为 N/A。 | 提交 |
| [test/svamp.predictions.jsonl](test/svamp.predictions.jsonl) | SVAMP 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/svamp.decoded.jsonl](test/svamp.decoded.jsonl) | SVAMP 逐题解释和整题参考算式，不将其视为完整的多步推理标注。 | 提交 |

测试采用 K=4，解码解释不是实际答案推理路径中的显式文本。延迟不含分词、文本解码、解释导出和模型加载；六模型已有延迟来自不同会话，不宜直接归因于方法带来的加速。

### diagnostics：梯度及跨题状态替换诊断

由 [p0_diagnose.py](../p0_diagnose.py) 加载最终 P3 模型生成，固定 seed=11，分别抽取 32 道训练题和 32 道验证题；不更新模型、不读取测试集。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [diagnostics/config.json](diagnostics/config.json) | checkpoint、抽样设置、环境、代码哈希及梯度/替换协议。 | 提交 |
| [diagnostics/samples.json](diagnostics/samples.json) | 实际使用的 64 道题，便于跨模型核对样本一致。 | 提交 |
| [diagnostics/parameters.json](diagnostics/parameters.json) | 梯度分析参数的名称及形状，不是权重。 | 提交 |
| [diagnostics/gradients.jsonl](diagnostics/gradients.jsonl) | 64 道题的各项损失、梯度范数、内积和余弦，分析局部梯度关系。 | 提交 |
| [diagnostics/interventions.jsonl](diagnostics/interventions.jsonl) | 使用本题或其他题的隐式状态时，答案、解释及目标序列 NLL 等逐题结果。 | 提交 |
| [diagnostics/summary.json](diagnostics/summary.json) | 分训练/验证汇总诊断，以及无参数更新、自身替换一致等检查和局限。 | 提交 |

### p3_reference150：第 150 次更新的参考路径检查

由 [p3_reference_probe.py](../p3_reference_probe.py) 生成，使用固定的 32 道验证题。只提供题目，让显式参考路径和隐式路径分别自由生成，不提供标准 CoT 或答案。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [p3_reference150/config.json](p3_reference150/config.json) | 第 150 次更新 checkpoint 的哈希、32 道题 ID、精度、生成预算和代码来源。 | 提交 |
| [p3_reference150/pairs.jsonl](p3_reference150/pairs.jsonl) | 两条路径的 32 组原始生成、提取答案及正确性。 | 提交 |
| [p3_reference150/summary.json](p3_reference150/summary.json) | 参考路径 0/32、隐式路径 0/32；参考输出截断 9 道、解析失败 9 道。 | 提交 |

这三个文件是中间 checkpoint 的已有评测结果，不包含 `reference150.pt` 本身。

### p3_final_reference：第 750 次更新的参考路径检查

由同一 `p3_reference_probe.py` 在最终模型上生成，题目与第 150 次检查相同。两个目录对应不同训练阶段，不是重复版本，应同时保留。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [p3_final_reference/config.json](p3_final_reference/config.json) | 最终 checkpoint、题目 ID、精度、生成预算和代码来源。 | 提交 |
| [p3_final_reference/pairs.jsonl](p3_final_reference/pairs.jsonl) | 最终模型两条路径的 32 组原始生成、提取答案及正确性。 | 提交 |
| [p3_final_reference/summary.json](p3_final_reference/summary.json) | 参考路径 1/32、隐式路径 0/32；参考输出截断 2 道、解析失败 2 道。 | 提交 |

参考路径最多生成 512 个新 token，隐式答案路径最多 64 个，不能用于等预算速度比较。最终答案匹配也不证明中间推理正确；历史报告记录了答案匹配输出中的算术错误。小样本 probe 和状态替换不能替代正式测试结果。

### Python 自动缓存

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| `__pycache__/__init__.cpython-312.pyc` | 包导入缓存，可重新生成。 | 不提交 |
| `__pycache__/p3_stage.cpython-312.pyc` | 阶段控制模块缓存，可重新生成。 | 不提交 |
| `__pycache__/test_p3_stage.cpython-312.pyc` | 阶段测试模块缓存，可重新生成。 | 不提交 |

## 共用分析与交付范围

以下内容位于上一级共用目录，不计入上述文件数量：

| 路径 | 用途 |
|---|---|
| [analyze_p3.py](../analyze_p3.py) | 核验训练并比较 control/P1/P2/P3 验证和诊断，读取历史 server_backup 目录，不是直接读取当前整理后的 P3 目录。 |
| [p3_analysis/summary.json](../p3_analysis/summary.json) | 验证、机制诊断、参考生成和训练核验的结构化汇总。 |
| [p3_analysis/answer_pairs.csv](../p3_analysis/answer_pairs.csv) | 逐题答案配对对照。 |
| [p3_analysis/step_pairs.csv](../p3_analysis/step_pairs.csv) | 逐槽位解释与参考步骤比较。 |
| [p3_analysis/reference_generations.csv](../p3_analysis/reference_generations.csv) | 参考路径生成的整理结果。 |
| [p3_analysis/cases.json](../p3_analysis/cases.json) | 报告分析使用的具体案例记录。 |
| [p3_analysis/provenance.json](../p3_analysis/provenance.json) | 分析输入、代码与 checkpoint 的来源和哈希。 |
| [analyze_test_suite.py](../analyze_test_suite.py) | 固定测试子集的六模型统一分析，复用 [analyze.py](../analyze.py) 的指标和步骤分析函数。 |
| [统一报告](../test_suite_analysis_20261126/REPORT.md)、[汇总表](../test_suite_analysis_20261126/summary.csv)、[分题型表](../test_suite_analysis_20261126/problem_types.csv) | P3 与 v1 的测试准确率、长度、延迟、步骤和分组收益结果。 |
| [六模型延迟说明](../SIX_MODEL_LATENCY_SUMMARY.md) | 现有测量的统一汇总及比较限制。 |

建议提交代码、说明、配置和全部小型结果文件，同时保留所依赖的共用分析。checkpoint 可沿用 E3 的压缩分片、Release 附件及哈希校验方式单独交付；这只是交付建议，不表示 P3 已发布到 Release。缓存不提交。当前目录不包含全部依赖，仅复制 P3 文件夹不足以独立复现。
