# P4：步骤末尾上下文表示对齐

## 当前进度与结果

本目录保存 P4 的实现、最终 checkpoint、训练记录、验证/测试输出、梯度与状态替换诊断，以及显式参考路径检查。P4 已完成 750 次更新；固定测试子集的六模型步骤评分和分题型分析已完成，正式 control 仍为 v1。

| 评测范围 | v1 答对数 | P4 答对数 | 说明 |
|---|---:|---:|---|
| 独立 BF16 验证集 | 6/200 | 6/200 | 严格步骤匹配为 v1 6/478、P4 1/478；答案持平不代表解释改善。 |
| GSM8K 固定测试子集 | 7/200 | 6/200 | 严格步骤匹配为 v1 4/634、P4 3/634。 |
| GSM-Hard 固定测试子集 | 1/200 | 2/200 | 缺少参考步骤，参考步骤匹配记为 N/A。 |
| SVAMP 固定测试子集 | 2/200 | 1/200 | 整题参考算式不作为完整的多步推理标注评分。 |

P4 没有取得一致的答案准确率收益，不能因某一数据集增加一题而宣称普遍有效。完整比较见 [六模型统一报告](../test_suite_analysis_20261126/REPORT.md)、[汇总表](../test_suite_analysis_20261126/summary.csv) 和 [分题型结果](../test_suite_analysis_20261126/problem_types.csv)。

训练内最后一次验证为 4/200，采用 FP32 master 参数加 BF16 autocast；上述独立验证为 6/200，采用 BF16 权重加 autocast。两种协议不可混用，也不能仅凭这两个数断定差异完全由精度造成。单种子、小样本及历史基线选择参考过测试结果的限制仍需披露。

[P4_REPORT.md](P4_REPORT.md) 是此前的验证与诊断分析报告，其中当时的下载及测试进度不作为当前状态。当前最终模型已位于 `checkpoint/last.pt`；原始报告、配置、日志和哈希记录保持历史内容。

## 假设与唯一改动

P4 的假设是：在标准算式上下文中取得的步骤末尾内部表示，可能为对应隐式状态提供额外的过程监督。P3 的自由生成检查没有建立可靠强教师的证据，因此这里不能把参考路径称为已经验证的强教师。Teacher forcing 下的内部状态与自由生成能力不同，对齐损失下降也不保证解题收益。

从原始 `e3_runs/shared_init.pt` 开始，保留 v1 的 2000 道训练题、200 道验证题、seed 11、样本顺序、750 次更新、micro batch 1、梯度累积 8、学习率 5e-5、weight decay 0.01、38 次学习率预热及 K=4。使用完整解释前缀，不叠加前缀遮蔽或 P1–P3。

唯一目标函数改动为：

```text
L_P4 = L_v1 + 0.1 * mean_valid_slots(1 - cosine(z_s, stop_gradient(h_ref[p_s])))
```

- `L_v1` 保留原有答案 CE、蒸馏、参考 CE 和解释 CE，各系数均为 1。
- `z_s` 是原 projector 输出，即解释 decoder 使用的表示。
- `h_ref[p_s]` 来自原训练流程已有的无梯度参考前向，取最后一层在对应算式闭合 `>>` token 的隐藏状态，包含题目和截至该步的标准上下文。
- 对齐项不向参考端传播梯度。参考路径仍通过原参考 CE 学习，与学生共享 reasoner；不是冻结的外部教师。
- 保留原 train-mode dropout，在 FP32 计算余弦距离并只对有效槽位平均。系数 0.1 是预先固定的工程选择，不是验证集最优值。
- 不增加参数或额外参考前向，不修改冻结的 `e3/common.py`、`e3/train.py` 或 vendored `model.py`；训练包装器通过临时 hooks 取得表示。
- 推理只输入题目，不需要参考算式或对齐目标。checkpoint 仍可由共用评测脚本加载。

| 监督槽 | 对齐目标 | 是否被后续 reasoner 作为隐式输入消费 |
|---|---|---|
| z0–z3 | 前四个算式各自末尾的上下文状态 | 是 |
| z4 | 第五个算式；若后续有多个算式，使用合并尾部的最终末尾状态 | 否，仅供解释读出 |
| 无参考步骤的槽 | 不参与对齐，不用 PAD 状态代替目标 | 沿用原流程 |

这是 5 个解释监督槽位，不是 5 步隐式推理。尾槽合并会带来信息压缩；逐步表示相似性约束也可能限制隐式表示的自由度，不能预设一定有益。

## 运行环境与目录要求

以下为复现命令，不表示实验尚未完成。以历史服务器项目根目录 `/home/user/experiment` 和 `./.conda312/bin/python` 为例，按实际环境调整。历史训练环境为 torch 2.7.1、transformers 4.52.4、peft 0.15.2、支持 BF16 的 CUDA GPU。

同步时保留 `e5/p4/` 包结构，不再把脚本平铺到 `e5/`。还需共同 E3/E5 代码、数据、基础模型资源及原始初始化。[p1_schedule.py](../p1_schedule.py) 仅提供基线预检查，不表示使用 P1 渐增；[p3_reference_probe.py](../p3_reference_probe.py) 是共用参考路径检查。共同配置为 [e3/config.json](../../e3/config.json)。

### 实现测试、数据审计和试跑

从项目根目录运行，使用新的试跑输出目录：

```bash
cd /home/user/experiment
./.conda312/bin/python -m unittest e5.p4.test_p4_alignment e5.p4.test_p4_model -v && \
./.conda312/bin/python e5/p4/p4_audit.py --require-weights && \
./.conda312/bin/python e5/p4/train_p4.py --output e5_runs/step_alignment_pilot_recheck --pilot-updates 10
```

检查损失及梯度有限，且 `loss ≈ base_loss + weighted_alignment_loss`，其中加权项为 `0.1 * alignment_loss`。`alignment_valid_slots` 在日志中是每次更新内 microbatch 的平均有效槽数，不是累计监督总量。试跑沿用完整 750 次更新的学习率计划，不从试跑权重继续正式训练。

`p4_audit.py` 只把 JSON 打印到终端，不自动创建 `p4_preflight.json`。当前目录已没有这个历史文件；如需保存新的检查结果，可在 Bash 中重定向到新文件：

```bash
./.conda312/bin/python e5/p4/p4_audit.py --require-weights > e5/p4/p4_preflight_recheck.json
```

该输出包含 `protocol`（来自 `p4_alignment.py` 的预检查）及 `splits`（来自 `p4_audit.py` 的全部训练/验证槽位审计），不是训练结果。正式训练保存的 `training/alignment_audit.json` 仍在当前目录，不应与预检查完整输出混淆。

### 重新训练与恢复

```bash
./.conda312/bin/python e5/p4/train_p4.py --output e5_runs/step_alignment_reproduction
```

新实验从共同初始化开始。中断后使用同一输出目录并加 `--resume`，恢复模型、优化器、调度器和随机状态；不要同时向同一目录写入。最终评测使用第 750 次更新，不按中途测试成绩选点。

### 已归档最终模型的重新评测

使用当前 checkpoint 路径，将结果写到新目录，避免覆盖归档：

```bash
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p4/checkpoint/last.pt --output e5_runs/p4_validation_recheck --split validation --explain
./.conda312/bin/python e3/evaluate.py --checkpoint e5/p4/checkpoint/last.pt --output e5_runs/p4_test_recheck --split test --limit 200 --explain
./.conda312/bin/python e5/p0_diagnose.py --checkpoint e5/p4/checkpoint/last.pt --output e5_runs/p4_diagnostics_recheck --n 32 --seed 11
./.conda312/bin/python e5/p3_reference_probe.py --checkpoint e5/p4/checkpoint/last.pt --output e5_runs/p4_final_reference_recheck --n 32
```

独立评测和参考 probe 使用 BF16 权重加 autocast；P0 诊断使用 FP32 master 参数加 BF16 autocast，应各自按相同协议比较。P0 计算原有共同损失的梯度，不是新增对齐项的专门梯度分析。不使用前缀遮蔽专用的 `e5/prefix_mask/evaluate.py` 来评测 P4。

## 历史核验与训练记录

此前记录了 8 项测试通过，覆盖位置偏移、尾部合并、停止 token、题目/答案边界、参考端停止梯度、PAD 排除、空监督、真实小型 CODI 前向、异常清理及恢复训练。系数为 0 时原损失、梯度和 RNG 一致；系数为 0.1 时原有损失及 RNG 不变，同时影响 reasoner/projector 梯度。这是历史核验记录，本次文档更新没有重跑训练或模型测试。

全数据审计为：训练集 5102 个有效槽位，其中 z4 为 146 个、50 道题合并尾部、1 道题无解释槽；验证集 478 个有效槽位，其中 z4 为 8 个、2 道题合并尾部。历史 manifest 差异仍按预检查记录披露。

实际训练会话耗时 4195.72 秒（69.93 分钟），峰值 allocated 显存约 4.723 GiB；计时包括验证及 checkpoint I/O，不含模型加载，不等同于总租赁时间。没有额外参考前向不代表新增损失没有反向计算或显存开销。

历史验证分析发现：答案 6/200 与 control 持平，严格步骤匹配 1/478 低于 control 的 6/478；解释仍高度模板化。对齐损失的 epoch 均值从约 0.92038 降到 0.60076，没有转化为验证准确率提升。参考自由生成只有 1/32 答案匹配，且对应生成中含算术错误。具体见 [P4_REPORT.md](P4_REPORT.md)，测试结果另见本文开头。

## 当前目录的每个文件

以下相对路径以 `e5/p4/` 为起点。当前共有 35 个交付文件和 3 个 Python 缓存文件。`p4_preflight.json` 不在现有文件清单中。“提交”表示交付建议，不表示已执行 push。

### 根目录：实现、配置和说明

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [__init__.py](__init__.py) | Python 包标记，支持从 `e5.p4` 导入模块。 | 提交 |
| [configuration.json](configuration.json) | 汇总共同超参数及 P4 规则，供阅读参考；不是历史训练 contract 的替代品，也不是训练入口默认读取的配置。 | 提交 |
| [p4_alignment.py](p4_alignment.py) | 核心实现：定位参考步骤末尾、排除错误边界/PAD、捕获隐式和参考表示、计算停止参考端梯度的余弦对齐项，并在结束或异常时清理 hooks。 | 提交 |
| [p4_audit.py](p4_audit.py) | 对全部训练/验证数据核对步骤提取、token 位置及对齐目标，打印协议与统计；不读取测试集、不训练模型。 | 提交 |
| [train_p4.py](train_p4.py) | 训练入口：共同初始化、数据检查、加入对齐损失、训练验证、日志和 checkpoint 保存及断点恢复。 | 提交 |
| [test_p4_alignment.py](test_p4_alignment.py) | 检查位置、尾槽合并、边界、参考端停止梯度、PAD 排除、空监督及系数缩放。 | 建议提交 |
| [test_p4_model.py](test_p4_model.py) | 小型模型集成测试，检查原有损失/RNG 一致性、学生梯度、无步骤情况、异常清理和恢复可复现性。 | 建议提交 |
| [P4_README.md](P4_README.md) | 本文件：当前进度、方法、复现命令、文件用途及提交建议。 | 提交 |
| [P4_REPORT.md](P4_REPORT.md) | 根据结构化结果撰写的历史验证和诊断报告；当时进度不作为当前状态，固定测试结果以六模型统一分析为准。 | 作为历史分析提交 |

两个 `test_p4_*.py` 是代码正确性测试，不是下方 `test/` 中的数学题测试集结果。

### checkpoint：最终模型

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [checkpoint/last.pt](checkpoint/last.pt) | 最终 750 次更新 checkpoint，约 2.23 GB。训练保存格式包括模型、优化器、调度器、随机状态、配置及 contract。评测记录的 SHA256 为 `aa55f07d580ebc9f8e3b5d33b51ff9abb24d37882831824c581a8f8bc527350e`。 | 单独提供下载，不通过普通 Git 提交 |

### training：正式训练记录

由 `train_p4.py` 生成。这里只列当前归档的四个文件，不将脚本可能生成的其他辅助文件视为已保存。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [training/contract.json](training/contract.json) | 正式训练超参数、P4 规则、更新预算、初始化/数据/代码哈希及环境；旧路径反映当时源代码结构。 | 提交 |
| [training/train_log.jsonl](training/train_log.jsonl) | 750 条更新记录，含四项原损失、base_loss、原始/加权对齐损失、平均有效槽位数、学习率、梯度范数及阶段验证；末次训练内验证为 4/200。 | 提交 |
| [training/sessions.jsonl](training/sessions.jsonl) | 会话起止更新数、训练循环耗时及显存峰值，说明计时范围。 | 提交 |
| [training/alignment_audit.json](training/alignment_audit.json) | 正式训练保存的槽位审计统计：训练 5102 个、验证 478 个有效槽位，以及参与推理/仅供读出槽数、合并尾槽及空步骤题数。 | 提交 |

### validation：200 道验证题

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split validation --explain` 生成。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [validation/config.json](validation/config.json) | 独立验证的 checkpoint、更新数、配置、数据及运行环境记录。 | 提交 |
| [validation/summary.csv](validation/summary.csv) | 200 题准确率、长度、延迟、解析失败和截断汇总；6/200 正确（3%）。 | 提交 |
| [validation/validation.predictions.jsonl](validation/validation.predictions.jsonl) | 逐题原始回答、提取答案、正确性、token 数、长度与延迟等。 | 提交 |
| [validation/validation.decoded.jsonl](validation/validation.decoded.jsonl) | 逐题解码解释、参考信息及监督槽位，供后续步骤评分。 | 提交 |

### test：固定测试子集

由 [e3/evaluate.py](../../e3/evaluate.py) 使用 `--split test --limit 200 --explain` 生成。三个数据集各 200 题，共 600 题。`decoded` 保存解释文本，步骤评分和相对 v1 的收益比较由共用分析代码完成。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [test/config.json](test/config.json) | checkpoint、数据哈希、BF16、batch size 1、贪心生成、5 次预热及 CUDA 同步计时范围。 | 提交 |
| [test/summary.csv](test/summary.csv) | GSM8K、GSM-Hard、SVAMP 的准确率、长度及延迟汇总；正确数依次为 6/200、2/200、1/200。 | 提交 |
| [test/gsm8k.predictions.jsonl](test/gsm8k.predictions.jsonl) | GSM8K 逐题答案、正确性、长度及延迟等。 | 提交 |
| [test/gsm8k.decoded.jsonl](test/gsm8k.decoded.jsonl) | GSM8K 逐题解释及参考信息，用于步骤评分。 | 提交 |
| [test/gsm_hard.predictions.jsonl](test/gsm_hard.predictions.jsonl) | GSM-Hard 逐题答案、正确性、长度及延迟等。 | 提交 |
| [test/gsm_hard.decoded.jsonl](test/gsm_hard.decoded.jsonl) | GSM-Hard 逐题解释；缺少参考步骤，参考步骤匹配为 N/A。 | 提交 |
| [test/svamp.predictions.jsonl](test/svamp.predictions.jsonl) | SVAMP 逐题答案、正确性、长度及延迟等。 | 提交 |
| [test/svamp.decoded.jsonl](test/svamp.decoded.jsonl) | SVAMP 逐题解释和整题参考算式，不将后者视为完整多步推理标注。 | 提交 |

测试采用 K=4；解释解码文本不是答案推理路径中的显式 CoT。延迟不含分词、文本解码、解释导出和模型加载。六模型已有延迟来自不同会话，不能直接归因于方法带来的加速。

### diagnostics：32 道训练题＋32 道验证题的机制诊断

由 [p0_diagnose.py](../p0_diagnose.py) 加载最终 P4 模型生成，固定 seed=11，不更新模型、不读取测试集。使用原有共同损失检查梯度关系，新增对齐项的训练变化应查看训练日志。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [diagnostics/config.json](diagnostics/config.json) | 模型、抽样、精度、运行环境、代码哈希及梯度/状态替换协议。 | 提交 |
| [diagnostics/samples.json](diagnostics/samples.json) | 实际使用的两组 32 道题，便于跨模型核对样本一致。 | 提交 |
| [diagnostics/parameters.json](diagnostics/parameters.json) | 参与梯度分析的参数名称和形状，不是模型权重。 | 提交 |
| [diagnostics/gradients.jsonl](diagnostics/gradients.jsonl) | 64 道题的原有损失、梯度范数、内积及余弦，用于局部梯度关系分析。 | 提交 |
| [diagnostics/interventions.jsonl](diagnostics/interventions.jsonl) | 使用本题或其他题隐式状态时的答案、解释及目标序列 NLL 等逐题记录。 | 提交 |
| [diagnostics/summary.json](diagnostics/summary.json) | 分训练/验证子集的诊断汇总、无参数更新和自身替换一致等检查及局限。 | 提交 |

局部梯度关系和跨题替换敏感性不能单独证明忠实推理或泛化改善。

### p4_final_reference：最终模型的显式参考路径检查

由共用 [p3_reference_probe.py](../p3_reference_probe.py) 在最终 P4 模型上生成，使用固定的 32 道验证题。只提供题目，让显式参考路径和隐式路径自由生成，不输入标准 CoT 或答案。

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [p4_final_reference/config.json](p4_final_reference/config.json) | checkpoint、题目 ID、精度、生成预算和代码来源。 | 提交 |
| [p4_final_reference/pairs.jsonl](p4_final_reference/pairs.jsonl) | 两条路径的 32 组原始生成、提取答案及正确性。 | 提交 |
| [p4_final_reference/summary.json](p4_final_reference/summary.json) | 参考路径 1/32、隐式路径 2/32；参考输出截断 2 道、解析失败 2 道。 | 提交 |

显式路径最多生成 512 个新 token，隐式答案路径最多 64 个，不能作等预算速度比较。P4 的训练对齐目标来自标准上下文中的内部状态，不能据此推定自由生成的参考推理可靠。

### Python 自动缓存

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| `__pycache__/__init__.cpython-312.pyc` | 包导入缓存，可重新生成。 | 不提交 |
| `__pycache__/p4_alignment.cpython-312.pyc` | 对齐模块缓存，可重新生成。 | 不提交 |
| `__pycache__/test_p4_alignment.cpython-312.pyc` | 对齐测试模块缓存，可重新生成。 | 不提交 |

## 共用分析与交付范围

下列文件位于上一级共用目录，不计入当前 P4 文件数量，但应随相关分析一并交付：

| 路径 | 用途 |
|---|---|
| [analyze_p4.py](../analyze_p4.py) | 核验训练、比较 control/P1/P2/P3/P4 的验证和诊断；读取历史 server_backup 数据，不是直接读取当前整理后的 P4 目录。 |
| [p4_analysis/summary.json](../p4_analysis/summary.json) | 验证、诊断、参考生成和训练核验的结构化汇总，部分进度描述属于历史记录。 |
| [p4_analysis/answer_pairs.csv](../p4_analysis/answer_pairs.csv) | 逐题答案配对比较。 |
| [p4_analysis/step_pairs.csv](../p4_analysis/step_pairs.csv) | 逐槽位步骤与参考的对照。 |
| [p4_analysis/reference_generations.csv](../p4_analysis/reference_generations.csv) | 参考自由生成的整理结果。 |
| [p4_analysis/cases.json](../p4_analysis/cases.json) | 报告所分析的具体案例记录。 |
| [p4_analysis/provenance.json](../p4_analysis/provenance.json) | 分析输入、代码及 checkpoint 来源和哈希。 |
| [analyze_test_suite.py](../analyze_test_suite.py) | 固定测试子集六模型统一分析入口，复用 [analyze.py](../analyze.py) 的指标与步骤分析函数。 |
| [统一测试报告](../test_suite_analysis_20261126/REPORT.md)、[汇总表](../test_suite_analysis_20261126/summary.csv)、[分题型表](../test_suite_analysis_20261126/problem_types.csv) | P4 与 v1 的测试准确率、长度、延迟、步骤和分组收益结果。 |
| [六模型延迟说明](../SIX_MODEL_LATENCY_SUMMARY.md) | 现有延迟测量的统一汇总及比较限制。 |

历史验证分析可在项目根目录重新输出到一个新目录，无需加载模型或重新训练：

```powershell
python e5/analyze_p4.py --output e5/p4_analysis_recheck
```

建议提交代码、文档、配置及全部小型结果，同时保留共用依赖；仅复制 P4 目录不足以独立复现。checkpoint 可沿用 E3 的压缩分片、GitHub Release 附件及哈希校验方式单独交付，这不表示 P4 已发布到 Release。缓存不提交。文档更新不改写原始结果、训练 contract 或历史哈希。
