# P2: first operand token weighting

本目录保存 P2（首操作数 token 加权）的实现、最终 checkpoint、训练记录、验证/测试输出和机制诊断。本文按当前目录结构整理；历史记录中的服务器路径、代码哈希和当时的实验状态保留原样。

## 当前进度与结果

P2 已完成 750 次更新，最终权重已归档到 `checkpoint/last.pt`。验证集、固定测试子集及诊断结果均已保存；P3/P4 也已完成测试，六模型统一步骤评分和分题型分析已有结果。当前正式 control 仍为 v1。

| 评测范围 | v1 答对数 | P2 答对数 | 说明 |
|---|---:|---:|---|
| 独立 BF16 验证集 | 6/200 | 6/200 | 严格步骤匹配由 v1 的 6/478 降至 P2 的 3/478。 |
| GSM8K 固定测试子集 | 7/200 | 4/200 | 严格步骤匹配为 v1 4/634、P2 5/634；答案准确率下降。 |
| GSM-Hard 固定测试子集 | 1/200 | 1/200 | 缺少参考步骤，参考步骤匹配记为 N/A。 |
| SVAMP 固定测试子集 | 2/200 | 1/200 | 参考为整题算式，不作为完整多步推理标注评分。 |

P2 没有取得一致的答案准确率收益，不能因个别步骤指标增加而宣称整体改进。分题型结果也同时存在改善、持平和退化，详见 [六模型统一报告](../test_suite_analysis_20261126/REPORT.md) 和 [分题型表](../test_suite_analysis_20261126/problem_types.csv)。单种子、小样本及历史基线选择参考过测试集的限制仍需保留。

训练日志末次验证为 8/200，使用 FP32 master 参数加 BF16 autocast；独立评测为 6/200，使用 BF16 权重加 autocast。不能把前者直接与 v1 的独立评测比较，也不能仅凭两个数值断定差异只由精度造成。

[P2_REPORT.md](P2_REPORT.md) 和 [p2_analysis/summary.json](../p2_analysis/summary.json) 是此前的验证与诊断分析，其中“权重未下载”“尚未测试”“P3 尚未实施”等描述代表生成当时的状态，不是当前进度。当前固定测试结论以六模型统一分析为准。

P1 did not improve answer accuracy or strict step correspondence. P2 is an
independent engineering hypothesis, starting from original shared_init, not P1.

## Frozen experiment

- Full prefixes; explanation coefficient 1 from update 1 to 750. No P1 ramp.
- First signed numeric literal in EACH `<<equation>>`: all overlapping GPT-2
  tokens receive weight 2; all other valid tokens, including EOS, receive 1.
  Ignored labels receive 0. Decimal points, unary signs, comma-separated digits,
  and equations merged into the fifth slot are covered. Leading parentheses are
  skipped when locating the literal. A BPE token overlapping a span is weighted
  as a whole; no token is split. Unsupported equations fail loudly.
  Two training equations (`<<x=5>>`, `<<X/2=0.5X>>`) begin with a symbolic
  operand: retain uniform weight 1, without substituting the RHS/denominator.
- Per slot: sum(weight * token CE) / sum(valid weights). Then preserve the
  upstream mean over effective slots. This is NOT a global mean over all slots.
- No changes to answer/reference/distillation losses, decoder inputs, inference,
  parameter names, initialization, data order, optimizer, learning rate or budget.
- The loss hook is active only inside a training CODI forward, immediately after
  an auxiliary decoder call. The frozen upstream source stays unchanged.
- Hooks add no RNG or model parameters. Resume restores model, optimizer,
  scheduler and global RNG; token-weight cache is deterministic and rebuilt.
- Token weighting targets numeric operands, not template diversity itself.
  Any benefit must be established from validation results; no effectiveness claim.

## 运行环境和目录要求

下面是复现命令，不表示当前还需要重新训练。示例使用历史服务器项目根目录 `/home/user/experiment` 和 `./.conda312/bin/python`；实际运行时按当前环境调整。历史训练环境为 torch 2.7.1、transformers 4.52.4、peft 0.15.2。

同步代码时应保留当前的 `e5/p2/` 包结构，不再把 P2 脚本平铺到 `e5/`。同时需要共用的 E3/E5 代码、数据、基础模型资源及原始 `e3_runs/shared_init.pt`。上一级 [p1_schedule.py](../p1_schedule.py) 只提供基线预检查，P2 不调用 P1 渐增规则；共同配置见 [e3/config.json](../../e3/config.json)。仅复制本目录不足以独立复现。

## Server checks and 10-update pilot

In the remote VSCode Bash terminal:

```bash
cd /home/user/experiment
./.conda312/bin/python -m unittest e5.p2.test_p2_operand e5.p2.test_p2_model -v && \
./.conda312/bin/python e5/p2/p2_audit.py && \
./.conda312/bin/python e5/p2/train_p2.py --output e5_runs/first_operand_pilot --pilot-updates 10
```

The chained commands stop if tests/audit fail. Check ten finite update records,
`operand_weight=2.0`, `explain_loss_factor=1.0`, and final COMPLETE. The pilot uses
the full-run LR schedule. Its directory must be new; it is not the formal run.
Local span/value/gradient tests and train/validation audit can run without peft.
Model integration tests and CUDA pilot require the frozen server environment
(torch 2.7.1, transformers 4.52.4, peft 0.15.2).

此前本地检查记录：five operand/span/value/gradient/protocol tests passed.
All 2,000 training rows (5,174 equations; 5,172 numeric spans; 2 symbolic
exceptions; 50 merged tails) and 200 validation rows (483 numeric equations;
2 merged tails) passed the tokenizer audit. Unicode byte-BPE reconstruction
is covered. Frozen initialization/source/data/order checks passed; historical
manifest discrepancy remains recorded. These are historical check results, not
a claim that this documentation update reran tests or training. Formal training
was completed on the server and its outputs are archived below.

## Formal training after a successful pilot

```bash
./.conda312/bin/python e5/p2/train_p2.py --output e5_runs/first_operand_reproduction
```

This starts again from shared_init and completes 750 updates. To recover an
interruption, use the same command with `--resume`. Never run two writers in the
same output directory. No extra update250 snapshot is created. Checkpoints are
saved every 50 updates and epoch/end boundaries. `contract.json` records method,
initialization, data/order and source hashes. `sessions.jsonl` records loop time.

## 已归档模型的验证、测试和诊断

以下命令使用当前归档的 checkpoint，并写入新的输出目录。不要覆盖已有交付文件。

```bash
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p2/checkpoint/last.pt --output e5_runs/p2_validation_recheck --split validation --explain
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p2/checkpoint/last.pt --output e5_runs/p2_test_recheck --split test --limit 200 --explain
./.conda312/bin/python e5/p0_diagnose.py --checkpoint e5/p2/checkpoint/last.pt --output e5_runs/p2_diagnostics_recheck --n 32 --seed 11
```

Compare with existing `p1_control_validation` and `p0_control_v1` (same frozen
control). No need to train control again. The generic P0 tool uses UNWEIGHTED
explanation CE for both checkpoints; its gradient measurements compare a common
diagnostic objective, not the P2 training objective. Keep its source unchanged
to preserve comparability. Evaluation uses BF16 weights; P0 uses FP32 master +
BF16 autocast, so compare within each protocol.

Review paired answer correctness, strict steps/operands/results, arithmetic
consistency and latent swaps together. Do not select based only on template
frequency or textual swap sensitivity. Use final update 750, not best epoch.
Do not evaluate test while selecting. Single-seed evidence and historical
test-influenced baseline selection remain limitations.

## 当前目录的每个文件

以下相对路径以 `e5/p2/` 为起点。当前共 32 个交付文件和 1 个 Python 缓存文件，`FILES.md` 已不在此目录。表中的“提交”指建议纳入代码/结果交付，不表示已执行 Git push。

### 根目录：实现、配置和说明

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [__init__.py](__init__.py) | Python 包标记，支持从 `e5.p2` 导入模块。 | 提交 |
| [configuration.json](configuration.json) | 汇总共同超参数与 P2 专属规则，方便查阅；不是历史训练 contract 的替代品，也不是训练脚本默认读取的配置。 | 提交 |
| [p2_operand.py](p2_operand.py) | 核心实现：识别每个算式的首个数字操作数，将字符/字节范围映射到 GPT-2 token，按槽位计算加权 CE；通过训练钩子只替换解释损失。 | 提交 |
| [p2_protocol.py](p2_protocol.py) | 固定操作数权重 2、其他有效 token 权重 1、解释系数 1、不开启前缀遮蔽；复用共同初始化、数据和预算检查。 | 提交 |
| [p2_audit.py](p2_audit.py) | 使用真实 tokenizer 审计训练/验证中的全部算式，检查数字范围、合并尾槽和 token 权重并打印统计；不读取测试集、不训练模型。 | 提交 |
| [train_p2.py](train_p2.py) | 训练入口：加载共同初始化、安装加权逻辑、执行 750 次更新，记录训练和验证结果，保存 checkpoint 并支持断点续训。 | 提交 |
| [test_p2_operand.py](test_p2_operand.py) | 单元测试：完整数字范围、合并尾槽、加权损失及梯度、实验协议、特殊符号算式和 Unicode 分词。 | 建议提交 |
| [test_p2_model.py](test_p2_model.py) | 小型模型集成测试：权重为 1 时与原模型一致，权重为 2 时只改变解释损失；检查梯度、解码器输入、参数结构、重新加载及推理行为。 | 建议提交 |
| [P2_README.md](P2_README.md) | 本文件：当前进度、方法规则、复现命令、文件介绍及交付范围。 | 提交 |
| [P2_REPORT.md](P2_REPORT.md) | 根据结构化分析结果撰写的历史验证与诊断报告，涵盖 control/P1/P2 比较、训练核验、模板问题和状态替换；不是固定测试集最终报告。 | 保留为历史分析提交 |

两个 `test_p2_*.py` 检查代码正确性，与下方 `test/` 的数学题测试结果不同。

### checkpoint：最终模型

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [checkpoint/last.pt](checkpoint/last.pt) | 最终 750 次更新的 checkpoint，约 2.23 GB。训练保存格式包含模型、优化器、调度器、随机状态、配置和 contract，支持恢复训练。评测记录中的 SHA256 为 `8764f7f04060a34a319826fd478f17b4e3d6dd40d2ff6358ec23705d96bca599`。 | 单独提供下载地址，不通过普通 Git 提交 |

### training：实际训练记录

由 `train_p2.py` 生成。这里保存了三份原始记录；训练脚本还会生成其他辅助文件，但不能将当前目录中不存在的文件写成已归档。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [training/contract.json](training/contract.json) | 本次训练的正式设置：共同超参数、P2 方法、750 次更新预算、初始化/数据/样本顺序/代码哈希及运行环境；其中旧代码路径反映当时目录结构。 | 提交 |
| [training/train_log.jsonl](training/train_log.jsonl) | 750 条更新记录，包含各项损失、学习率、梯度范数、操作数权重、解释系数、耗时和阶段验证；末次训练内验证为 8/200。 | 提交 |
| [training/sessions.jsonl](training/sessions.jsonl) | 会话起止更新数、约 4064.05 秒训练循环耗时及约 4.72 GiB 峰值 allocated 显存；耗时包括验证和 checkpoint I/O，不含模型加载。 | 提交 |

### validation：200 道验证题

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split validation --explain` 生成，采用独立 BF16 评测协议。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [validation/config.json](validation/config.json) | 验证 checkpoint、更新数、训练配置、数据及运行环境记录。 | 提交 |
| [validation/summary.csv](validation/summary.csv) | 200 题汇总：6/200 正确、准确率 3%，以及长度、延迟、解析失败和截断统计。 | 提交 |
| [validation/validation.predictions.jsonl](validation/validation.predictions.jsonl) | 200 道题的原始输出、提取答案、正确性、生成 token、长度和延迟等。 | 提交 |
| [validation/validation.decoded.jsonl](validation/validation.decoded.jsonl) | 200 道题的解码解释、参考信息及监督槽位，供步骤级分析使用。 | 提交 |

### test：固定测试子集

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split test --limit 200 --explain` 生成。GSM8K、GSM-Hard、SVAMP 各 200 题，共 600 题。`decoded` 保存解释文本，步骤评分及相对 v1 的收益比较由共用分析脚本完成。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [test/config.json](test/config.json) | 测试来源和条件：checkpoint、数据哈希、BF16 权重及 autocast、batch size 1、贪心生成、5 次预热及 CUDA 同步计时范围。 | 提交 |
| [test/summary.csv](test/summary.csv) | 三个数据集的准确率、长度及延迟汇总，正确数依次为 4/200、1/200、1/200。 | 提交 |
| [test/gsm8k.predictions.jsonl](test/gsm8k.predictions.jsonl) | GSM8K 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/gsm8k.decoded.jsonl](test/gsm8k.decoded.jsonl) | GSM8K 逐题解码解释及参考信息，用于步骤级评分。 | 提交 |
| [test/gsm_hard.predictions.jsonl](test/gsm_hard.predictions.jsonl) | GSM-Hard 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/gsm_hard.decoded.jsonl](test/gsm_hard.decoded.jsonl) | GSM-Hard 逐题解码解释；缺少参考步骤，参考步骤匹配记为 N/A。 | 提交 |
| [test/svamp.predictions.jsonl](test/svamp.predictions.jsonl) | SVAMP 逐题答案、正确性、长度和延迟等。 | 提交 |
| [test/svamp.decoded.jsonl](test/svamp.decoded.jsonl) | SVAMP 逐题解码解释和参考算式；整题算式不能当作完整步骤标注。 | 提交 |

延迟不包含分词、文本解码、解释导出和模型加载。所有模型采用 K=4；答案 token 数、参考长度及隐式步数是不同口径，不能把解码解释长度当成实际答案推理路径的长度。六模型延迟来自已有不同会话的测量，不能直接归因于方法带来的加速。

### diagnostics：32 道训练题＋32 道验证题的机制诊断

由 [p0_diagnose.py](../p0_diagnose.py) 加载 P2 最终模型生成，固定 `n=32`、`seed=11`，不更新参数、不使用测试集。它使用与其他模型相同的普通解释 CE，而非 P2 训练时的首操作数加权 CE，以保证诊断目标一致。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [diagnostics/config.json](diagnostics/config.json) | checkpoint、抽样设置、运行环境、代码哈希、梯度和状态替换协议。 | 提交 |
| [diagnostics/samples.json](diagnostics/samples.json) | 实际抽取的 32 道训练题和 32 道验证题，便于跨模型核对样本一致性。 | 提交 |
| [diagnostics/parameters.json](diagnostics/parameters.json) | 参与梯度诊断的参数名称和形状，不是模型权重。 | 提交 |
| [diagnostics/gradients.jsonl](diagnostics/gradients.jsonl) | 64 道题各项损失、梯度范数及梯度内积/余弦，检查局部梯度关系。 | 提交 |
| [diagnostics/interventions.jsonl](diagnostics/interventions.jsonl) | 64 道题使用自身或其他题隐式状态时的答案、解释和目标序列 NLL 等，用于检查状态替换影响。 | 提交 |
| [diagnostics/summary.json](diagnostics/summary.json) | 分训练/验证子集的诊断汇总，以及无参数更新、自身替换一致等完整性检查和解释局限。 | 提交 |

小样本梯度关系和替换敏感性是辅助证据，不等同于泛化改善或忠实推理的证明。

### 自动缓存

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| `__pycache__/__init__.cpython-312.pyc` | Python 3.12 自动生成的包缓存，运行时可重新生成。 | 不提交 |

## 共用分析代码与结果

以下文件保留在上一级共用目录，不属于当前 P2 文件清单，但应随相关分析交付：

| 路径 | 用途 |
|---|---|
| [analyze_p2.py](../analyze_p2.py) | 核验训练记录，比较 control/P1/P2 的验证输出和诊断；读取历史 `server_backup_p1`、`server_backup_p2`，不是直接读取当前整理后的目录。 |
| [p2_analysis/summary.json](../p2_analysis/summary.json) | 三模型验证、诊断和训练核验汇总；其中进度限制是历史记录。 |
| [p2_analysis/answer_pairs.csv](../p2_analysis/answer_pairs.csv) | P2 与 control/P1 的逐题答案对照。 |
| [p2_analysis/step_pairs.csv](../p2_analysis/step_pairs.csv) | 三模型解释步骤的配对比较，含首操作数分析。 |
| [p2_analysis/provenance.json](../p2_analysis/provenance.json) | 分析输入、代码及 checkpoint 哈希和核验记录。 |
| [analyze_test_suite.py](../analyze_test_suite.py) | 固定测试子集六模型统一分析入口，复用 [analyze.py](../analyze.py) 的指标和步骤分析函数。 |
| [统一测试报告](../test_suite_analysis_20261126/REPORT.md)、[汇总表](../test_suite_analysis_20261126/summary.csv)、[分题型表](../test_suite_analysis_20261126/problem_types.csv) | P2 与 v1 的测试准确率、长度、延迟、步骤评分及分组收益分析。 |
| [六模型延迟说明](../SIX_MODEL_LATENCY_SUMMARY.md) | 现有测量的统一汇总和比较限制。 |

原始 JSON/JSONL/CSV、checkpoint 和训练 contract 不因文档整理而改写。建议提交代码、说明和全部小型结果文件；checkpoint 单独发布下载地址，缓存不提交。
