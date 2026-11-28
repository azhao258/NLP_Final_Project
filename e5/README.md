> 2026-11-28 目录调整：测试比较见 [E6](../E6/README.md)，步骤/检索/案例见 [E7](../E7/README.md)。旧运行记录中的路径是生成时路径，迁移不代表重训或重算分数。

# Part 3 / E5：改进实验与交付索引

## 范围与当前进度

当前正式交付范围为 **v1 对照、前缀遮蔽、P1、P2、P3、P4**。v1 是冻结基线；五个改进方案各自从共同初始化开始，不依次叠加。v2/v3、额外基线调优及其他探索不作为本目录的正式比较模型。

五个方案均已保存最终 checkpoint、验证及固定测试输出。固定测试为 GSM8K、GSM-Hard、SVAMP 各 200 道题，已完成六模型统一答案、长度、延迟、步骤评分与分题型分析。现有延迟来自分批会话，不是完整六模型同场基准。

| 模型 | GSM8K 答对 /200 | GSM-Hard 答对 /200 | SVAMP 答对 /200 |
|---|---:|---:|---:|
| v1 | 7 | 1 | 2 |
| 前缀遮蔽 | 4 | 1 | 2 |
| P1 | 3 | 1 | 2 |
| P2 | 4 | 1 | 1 |
| P3 | 3 | 2 | 5 |
| P4 | 6 | 2 | 1 |

没有方案在三个数据集上都取得答案准确率提升。应报告改善、持平和退化，避免把个别题目的得失或步骤指标变化解释为稳定收益。

当前记录的边界：

- K=4，但解释读出有 5 个槽位；最后一个不作为后续 reasoner 的隐式输入。
- GSM8K 测试参考步骤评分覆盖 195 道题、634 个有效槽位；GSM-Hard 没有参考步骤，匹配指标为 N/A；SVAMP 只有整题表达式兼容性检查，不作为完整步骤链评分。
- 所有方案只有单种子结果；v1 的历史选择参考过测试结果，因此这些是探索性比较。
- 解释匹配、算术有效性或状态替换敏感性不能单独证明隐式推理忠实性。
- P1 当前目录没有单独归档的完整训练日志；P3 有第 150 次更新的 probe 输出，但其目录没有 `reference150.pt`。
- 整理前源码 ZIP 和 P4 预检查 JSON 当前缺失，对部分历史分析的重跑有影响，见后文“复现与输入依赖”。

## 五个方案的入口

每个方案 README 介绍其当前文件、方法、运行命令与提交建议。

| 方案 | 唯一改动 | 文档入口 |
|---|---|---|
| 前缀遮蔽 | 训练解释 decoder 时，以 0.3 概率将部分文本前缀 embedding 置零，保留规定的边界及标签 | [prefix_mask/README.md](prefix_mask/README.md) |
| P1 | 前 250 次更新将解释损失系数从 0.1 增至 1，之后保持 1 | [p1/P1_README.md](p1/P1_README.md) |
| P2 | 对每个参考算式的首个数字操作数 token 使用权重 2 | [p2/P2_README.md](p2/P2_README.md) |
| P3 | 前 150 次只优化参考路径 CE，后 600 次恢复联合训练 | [p3/P3_README.md](p3/P3_README.md) |
| P4 | 在原损失上增加系数 0.1 的步骤末尾上下文表示对齐项 | [p4/P4_README.md](p4/P4_README.md) |

共同配置见 [e3/config.json](../e3/config.json)，冻结基线记录见 [e3/frozen_v1.json](../e3/frozen_v1.json)。v1 checkpoint 位于 [e3_runs/control/last.pt](../e3_runs/control/last.pt)，其交付下载方式见 [E3 交接文档](../e3/HANDOFF_TO_B.md)。

## 课程要求与主要交付

| 要求 | 主要入口 | 说明 |
|---|---|---|
| 方法假设与单项改动 | 五个方案 README、[RESEARCH.md](RESEARCH.md) | 研究文档包含历史计划；当前状态以各方案 README 及测试分析为准。 |
| 改进代码、checkpoint、配置 | 五个方案目录 | checkpoint 单独交付下载，不纳入普通 Git 历史。 |
| 固定测试答案准确率、长度、延迟、步骤指标 | [统一汇总表](../E6/legacy_six_model_analysis/summary.csv) | 18 行，六模型×三个数据集。 |
| 逐题相对 v1 的得失 | [逐题对照](../E6/legacy_six_model_analysis/paired_questions.csv) | 3000 个模型与 v1 的题目配对。 |
| 第二部分步骤规则的应用 | [步骤明细](../E6/legacy_six_model_analysis/steps.csv)、[错误类别](../E6/legacy_six_model_analysis/categories.csv) | 18000 个解码槽位，参考缺失不记成匹配失败。 |
| 收益是否跨题型成立 | [分题型结果](../E6/legacy_six_model_analysis/problem_types.csv) | 保留改善/持平/退化与小组样本量，分组轴重叠不能相加。 |
| 延迟口径与限制 | [延迟说明](../E6/SIX_MODEL_LATENCY_SUMMARY.md)、[延迟明细](../E6/SIX_MODEL_LATENCY.csv) | 历史测量描述性比较，不作跨会话方法加速归因。 |
| 汇总报告 | [六模型固定测试报告](../E6/legacy_six_model_analysis/REPORT.md) | 包含结果、配对变化、分组与限制；长度和延迟数值详见汇总表。 |

## 根目录共用文件

本节及下方目录表覆盖五个方案目录之外的现有非缓存文件。当前为 76 个非缓存文件，另有 31 个根目录 Python 缓存；这个数量不包含五个方案内部文件。“提交”是建议，不代表已执行 Git push。

### 分析、训练协议与诊断代码

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [analyze.py](../E7/scoring.py) | v1/前缀遮蔽配对分析；其中步骤评分、准确率、长度、延迟函数也被统一分析复用。现仅生成结构化数据，不再生成旧 Markdown 报告。 | 提交 |
| [analyze_test_suite.py](../E6/analyze_test_suite.py) | 六模型固定测试统一分析入口，核验原始输出，计算配对、步骤、题型、模板等，并生成统一 REPORT.md；不重新运行模型。 | 提交 |
| [analyze_p1.py](analyze_p1.py) | v1/P1 的 200 题验证和 32＋32 诊断比较；辅助函数被后续脚本复用。 | 提交 |
| [analyze_p2.py](analyze_p2.py) | v1/P1/P2 验证和诊断，核验 P2 日志并分析首操作数。 | 提交 |
| [analyze_p3.py](analyze_p3.py) | v1/P1/P2/P3 验证和诊断、分阶段训练核验、参考生成及案例分析。 | 提交 |
| [analyze_p4.py](analyze_p4.py) | v1/P1/P2/P3/P4 验证和诊断、对齐损失训练核验、参考生成及案例分析；当前重跑仍依赖缺失的 P4 预检查文件。 | 提交；依赖见后文 |
| [p0_diagnose.py](p0_diagnose.py) | 各项原有损失的梯度关系和跨题隐式状态替换，默认 32 道训练题＋32 道验证题，不更新模型。 | 提交 |
| [p3_reference_probe.py](p3_reference_probe.py) | 对固定验证题比较显式参考与隐式路径的自由生成能力，供 v1/P3/P4 等使用。 | 提交 |
| [p1_schedule.py](p1_schedule.py) | P1 解释系数渐增函数，以及 P1–P4 复用的初始化、训练/验证数据、顺序和预算预检查。 | 提交 |
| [protocol.py](protocol.py) | JSON 读写、哈希、样本顺序等共用工具；完整 preflight 和 checkpoint 验证包含前缀遮蔽专属约束。 | 提交 |
| [problem_types.py](problem_types.py) | 基于参考运算、步骤数、SVAMP 原类型、数字大小/小数等规则分组，不读取模型预测。 | 提交 |
| [source_history.py](source_history.py) | 当前源码不匹配历史哈希时，从整理前源码快照核验旧版本；当前所需 ZIP 缺失。 | 提交；依赖见后文 |
| [test_analysis.py](../E7/test_scoring.py) | 检查协议、类型清单、步骤评分、配对完整性、不覆盖结果和不生成旧报告等行为。 | 建议提交 |
| [test_p0.py](test_p0.py) | 检查诊断损失梯度图及状态替换实现。 | 建议提交 |
| [test_test_suite.py](../E6/test_exports.py) | 确认统一分析拒绝乱序题目、错误正确性标签、缺失解释槽位和错误参考。 | 建议提交 |

`analyze_p1.py`–`analyze_p4.py` 支持历史验证报告，不能替代固定测试的 `analyze_test_suite.py`。P1 最终验证分析已放在 [p1/p1_analysis_final](p1/p1_analysis_final/)；P2–P4 的跨模型结果保留在共用目录。

### 文档与固定清单

| 文件 | 内容与用途 | 提交建议 |
|---|---|---|
| [README.md](README.md) | 本总入口：范围、现状、文件、依赖与提交建议。 | 提交 |
| [P0_README.md](P0_README.md) | 梯度和状态替换诊断的协议、运行方式及解释边界。 | 提交 |
| [RESEARCH.md](RESEARCH.md) | 早期研究依据、课程要求对应和候选方案讨论。含历史计划；其旧报告链接需另外修正，当前前缀遮蔽报告见 [PREFIX_MASK_REPORT.md](prefix_mask/PREFIX_MASK_REPORT.md)。 | 保留历史定位，修正链接后提交 |
| [problem_types.json](problem_types.json) | 固定 600 道测试题的分组标签、分类政策和来源哈希；所有模型使用相同分组。 | 提交 |
| [SIX_MODEL_LATENCY.csv](../E6/SIX_MODEL_LATENCY.csv) | 六模型×三个数据集共 18 行延迟及相关指标明细。 | 提交 |
| [SIX_MODEL_LATENCY_SUMMARY.md](../E6/SIX_MODEL_LATENCY_SUMMARY.md) | 平均值、中位数、P95 及计时范围、跨会话比较限制；旧文本中的 experiment_summary 路径可用本目录 CSV 对照。 | 提交；建议同步路径文字 |
| [ORGANIZATION_MANIFEST.json](ORGANIZATION_MANIFEST.json) | 文件整理时的移动/复制来源、目标、哈希和核验结果。包括源码快照路径及哈希，但当前快照文件缺失；其中 current_source_sha256 是整理当时记录，不代表后来修改文件的最新哈希。 | 作为历史记录提交 |

## v1 共用对照数据

### control_validation：独立验证，4 个文件

由共用评测流程生成，供 P1–P4 历史验证比较。

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [control_validation/config.json](control_validation/config.json) | v1 验证 checkpoint、配置、数据及环境。 | 提交 |
| [control_validation/summary.csv](control_validation/summary.csv) | 200 题准确率、长度与延迟，6/200 正确。 | 提交 |
| [control_validation/validation.predictions.jsonl](control_validation/validation.predictions.jsonl) | 逐题答案、正确性、长度和延迟。 | 提交 |
| [control_validation/validation.decoded.jsonl](control_validation/validation.decoded.jsonl) | 逐题解释、参考及监督槽位。 | 提交 |

### control_test：固定测试，9 个文件

保存与前缀遮蔽配对测量时的 v1 输出，也是六模型分析采用的 v1 对照来源。

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [control_test/complete.json](../E6/exports/v1/complete.json) | 600 题导出完成记录、配对标识及文件哈希。 | 提交 |
| [control_test/config.json](../E6/exports/v1/config.json) | v1 身份、训练来源、配置、数据/代码/题型哈希、配对与运行设置。 | 提交 |
| [control_test/summary.csv](../E6/exports/v1/summary.csv) | 三个数据集汇总，正确数为 7/200、1/200、2/200。 | 提交 |
| [control_test/gsm8k.predictions.jsonl](../E6/exports/v1/gsm8k.predictions.jsonl) | GSM8K 逐题答案、正确性、长度和延迟。 | 提交 |
| [control_test/gsm8k.decoded.jsonl](../E6/exports/v1/gsm8k.decoded.jsonl) | GSM8K 解释及参考。 | 提交 |
| [control_test/gsm_hard.predictions.jsonl](../E6/exports/v1/gsm_hard.predictions.jsonl) | GSM-Hard 逐题答案、正确性、长度和延迟。 | 提交 |
| [control_test/gsm_hard.decoded.jsonl](../E6/exports/v1/gsm_hard.decoded.jsonl) | GSM-Hard 解释，缺少参考步骤。 | 提交 |
| [control_test/svamp.predictions.jsonl](../E6/exports/v1/svamp.predictions.jsonl) | SVAMP 逐题答案、正确性、长度和延迟。 | 提交 |
| [control_test/svamp.decoded.jsonl](../E6/exports/v1/svamp.decoded.jsonl) | SVAMP 解释及整题参考算式。 | 提交 |

### control_diagnostics：32＋32 机制诊断，6 个文件

由 `p0_diagnose.py` 生成，是改进模型相同诊断的共同基准。

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [control_diagnostics/config.json](control_diagnostics/config.json) | 模型、抽样、精度、环境和诊断协议。 | 提交 |
| [control_diagnostics/samples.json](control_diagnostics/samples.json) | 实际 32 道训练题和 32 道验证题。 | 提交 |
| [control_diagnostics/parameters.json](control_diagnostics/parameters.json) | 参与梯度分析的参数名称及形状，不是权重。 | 提交 |
| [control_diagnostics/gradients.jsonl](control_diagnostics/gradients.jsonl) | 逐题损失、梯度范数、内积与余弦。 | 提交 |
| [control_diagnostics/interventions.jsonl](control_diagnostics/interventions.jsonl) | 原始和跨题替换状态下的答案、解释、NLL 等。 | 提交 |
| [control_diagnostics/summary.json](control_diagnostics/summary.json) | 分训练/验证汇总、完整性检查及局限。 | 提交 |

### control_reference：32 道验证题的参考路径 probe，3 个文件

由 `p3_reference_probe.py` 生成，作为 P3/P4 参考生成的对照。

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [control_reference/config.json](control_reference/config.json) | 模型、题目 ID、精度、生成预算和代码来源。 | 提交 |
| [control_reference/pairs.jsonl](control_reference/pairs.jsonl) | 每题显式参考和隐式路径的原始生成及正确性。 | 提交 |
| [control_reference/summary.json](control_reference/summary.json) | 参考路径 3/32、隐式路径 1/32，及截断/解析失败统计。 | 提交 |

## 历史验证对比结果

以下目录不包含固定测试的最终六模型分析。保留它们可以追溯方法诊断、训练核验和报告论据；其中“权重未下载”等描述代表生成当时状态。

### p2_analysis：v1/P1/P2，4 个文件

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [p2_analysis/summary.json](p2_analysis/summary.json) | 验证、首操作数、诊断和 P2 训练核验汇总。 | 提交 |
| [p2_analysis/answer_pairs.csv](p2_analysis/answer_pairs.csv) | P2 分别对 v1/P1 的 200 题答案配对，共 400 行。 | 提交 |
| [p2_analysis/step_pairs.csv](p2_analysis/step_pairs.csv) | 1000 个槽位上的三模型解释、严格匹配和首操作数比较。 | 提交 |
| [p2_analysis/provenance.json](p2_analysis/provenance.json) | 输入、代码及 checkpoint 哈希与核验结果。 | 提交 |

### p3_analysis：v1/P1/P2/P3，6 个文件

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [p3_analysis/summary.json](p3_analysis/summary.json) | 验证、诊断、参考生成及阶段训练核验汇总。 | 提交 |
| [p3_analysis/answer_pairs.csv](p3_analysis/answer_pairs.csv) | P3 对 v1/P1/P2 的答案比较，共 600 行。 | 提交 |
| [p3_analysis/step_pairs.csv](p3_analysis/step_pairs.csv) | 四模型在同一 1000 个槽位的步骤比较。 | 提交 |
| [p3_analysis/reference_generations.csv](p3_analysis/reference_generations.csv) | v1、P3 第 150 次更新和 P3 最终模型的参考生成，共 96 行。 | 提交 |
| [p3_analysis/cases.json](p3_analysis/cases.json) | 参考答案匹配输出、P3 答对题及算术错误案例。 | 提交 |
| [p3_analysis/provenance.json](p3_analysis/provenance.json) | 输入、代码、checkpoint 来源与哈希。 | 提交 |

### p4_analysis：v1/P1/P2/P3/P4，6 个文件

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [p4_analysis/summary.json](p4_analysis/summary.json) | 验证、诊断、对齐训练核验及参考生成汇总。 | 提交 |
| [p4_analysis/answer_pairs.csv](p4_analysis/answer_pairs.csv) | P4 对 v1/P1/P2/P3 的答案比较，共 800 行。 | 提交 |
| [p4_analysis/step_pairs.csv](p4_analysis/step_pairs.csv) | 五模型在同一 1000 个槽位的步骤比较。 | 提交 |
| [p4_analysis/reference_generations.csv](p4_analysis/reference_generations.csv) | v1、P3 最终模型及 P4 最终模型的参考生成，共 96 行。 | 提交 |
| [p4_analysis/cases.json](p4_analysis/cases.json) | 参考答案匹配输出、P4 答对题及算术错误案例。 | 提交 |
| [p4_analysis/provenance.json](p4_analysis/provenance.json) | 输入、代码、checkpoint 来源与哈希。 | 提交 |

## test_suite_analysis_20261126：固定测试统一分析，16 个文件

由 `analyze_test_suite.py` 离线处理已保存结果生成，不重新训练或推理。这是 Part3 效果比较的主要交付。

| 文件 | 内容 | 提交建议 |
|---|---|---|
| [REPORT.md](../E6/legacy_six_model_analysis/REPORT.md) | 六模型准确率、步骤匹配、逐题得失、分组变化及局限报告。 | 提交 |
| [summary.csv](../E6/legacy_six_model_analysis/summary.csv) | 六模型×三数据集的 18 行汇总，包含答案、长度、延迟和步骤指标。 | 提交 |
| [summary.json](../E6/legacy_six_model_analysis/summary.json) | 同一总体指标的 JSON 数组。 | 提交 |
| [summary.jsonl](../E6/legacy_six_model_analysis/summary.jsonl) | 同一总体指标的逐行 JSON。 | 一起提交 |
| [steps.csv](../E6/legacy_six_model_analysis/steps.csv) | 18000 条逐槽位评分，含生成、参考、算术有效性、匹配、错误标签及证据。 | 提交 |
| [steps.jsonl](../E6/legacy_six_model_analysis/steps.jsonl) | 同一逐槽位结果的结构化格式，方便保留列表等字段。 | 提交 |
| [paired_questions.csv](../E6/legacy_six_model_analysis/paired_questions.csv) | 五方案分别对 v1、各 600 题，共 3000 组答案和解释得失。 | 提交 |
| [paired_questions.jsonl](../E6/legacy_six_model_analysis/paired_questions.jsonl) | 同一逐题配对结果的结构化格式。 | 提交 |
| [problem_types.csv](../E6/legacy_six_model_analysis/problem_types.csv) | 170 行总体和分组收益，含得失、准确率及步骤变化和小组标记。 | 提交 |
| [problem_types.jsonl](../E6/legacy_six_model_analysis/problem_types.jsonl) | 同一分组比较的结构化格式。 | 提交 |
| [categories.csv](../E6/legacy_six_model_analysis/categories.csv) | 162 行步骤错误类别计数及分母。 | 提交 |
| [categories.jsonl](../E6/legacy_six_model_analysis/categories.jsonl) | 同一步骤错误类别统计的结构化格式。 | 提交 |
| [slots.csv](../E6/legacy_six_model_analysis/slots.csv) | 90 行槽位模板统计，含不同输出数、最常见解释及占比。 | 提交 |
| [slots.jsonl](../E6/legacy_six_model_analysis/slots.jsonl) | 同一槽位模板统计的结构化格式。 | 提交 |
| [provenance.json](../E6/legacy_six_model_analysis/provenance.json) | 输入配置及哈希、题型清单、分析代码、原有汇总回归检查和局限。 | 提交 |
| [complete.json](../E6/legacy_six_model_analysis/complete.json) | 完成状态、模型/题目/槽位/配对数量，以及输出文件哈希。 | 提交 |

根目录 `problem_types.json` 定义“题目属于哪些组”；这里的 `problem_types.csv/jsonl` 回答“各组收益怎样”。`categories` 是错误类别，`slots` 是模板统计。CSV 和 JSON/JSONL 是不同读取格式，建议按完整导出一起提交，避免删去文件后留下不完整的 complete 哈希清单。

## 缓存文件

根目录 `__pycache__/` 当前有 31 个 `.pyc`，全部不提交。它们是 Python 自动编译缓存，包含共用分析和部分整理前方法模块的遗留缓存；不能代替源码或实验输出。运行时可重新生成，已有忽略规则覆盖这些文件。

## 复现与输入依赖

查看已有 CSV/JSON/Markdown 不需要 GPU。重新分析也不需要模型推理，但**当前脚本仍依赖原始备份位置，不能只克隆 e5 就假定能立即重跑**。

### 六模型固定测试分析

默认依赖项目根目录下：

- `server_backup_eval_20261126/`：含 `VERIFICATION.json` 和 P1–P4 原始测试导出；
- `server_backup_part3/nlp_part3_backup.tar`：含 v1/前缀遮蔽配对导出；
- `e5_eval_backfill_20261126.tar.gz` 及其 `.sha256`：补跑原始包及校验；
- `e1/data/`、`e3/frozen_v1.json`、本目录冻结题型清单及相关 E1/E4 评分代码。

当前主要备份目录及两个压缩包仍在本地。脚本支持通过 `--input`、`--paired-archive`、`--backfill-archive` 指定相应位置。确保依赖齐全后，从项目根目录写到新的输出目录：

```powershell
python e5/analyze_test_suite.py --output e5/test_suite_analysis_recheck
```

这个命令是复现说明；本次 README 更新没有重跑分析或核验所有备份文件的字节。

### P1–P4 历史验证分析

- `analyze_p1.py` 默认读取 `server_backup_p1`，可用 `--input` 指定原结构的替代位置。
- `analyze_p2.py`、`analyze_p3.py`、`analyze_p4.py` 仍按源码读取对应 `server_backup_p1` 至 `server_backup_p4`，并依赖 E3 数据及冻结记录。
- `source_history.py` 在当前源码与旧哈希不同的情况下，要求 `e5/source_before_organization_20261126.zip`。该文件曾作为历史源码证据保存，**当前已不在 e5**。整理清单记录的预期 SHA256 为 `f8c927696f7e099aa0d5bb626e0be620f723f39cb783ceaaa2fe5a19d94b64c0`。需要恢复原字节快照才能继续该核验，不能用当前源码随意重新打包替代。
- `analyze_p4.py` 还直接读取 `e5/p4/p4_preflight.json` 的 protocol.variant 和 splits。**当前该文件也缺失**。已有 P4 分析仍可查看，但按当前代码直接重跑会受阻。应恢复匹配的预检查记录，或另行调整并验证输入核验方式；不能把 `training/alignment_audit.json` 原样改名，因为它不含完整 protocol。

这些缺失不改变已归档结果本身，但影响当前复现路径。本文不把旧 complete/PASS 标志解释为今天所有重跑依赖都已齐备，也不通过跳过哈希检查掩盖缺失。

## 提交建议与历史记录

- 提交五个方案的实现、配置、说明与原始小型结果，以及本目录共用代码、v1 对照数据、历史分析和统一测试结果。
- 五个大 checkpoint 单独提供下载地址。可沿用 E3 的压缩分片、GitHub Release 附件、manifest 和下载校验脚本方式；这里不表示 E5 权重已经发布。
- 缓存不提交。源码快照虽然是 ZIP，却是历史核验依赖，恢复后应随交付提供。
- `ORGANIZATION_MANIFEST.json`、训练 contract、评测配置及 provenance 保留生成当时的路径和哈希，不为“匹配当前文件”而重写历史证据。
- 方案 README 中的重跑命令仍需满足本节依赖。当前总入口反映最新缺失情况；历史报告和研究文档中的阶段进度不作为现状。

本 README 按当前文件结构整理，仅更新文档，不改变训练方案、评分规则或原始结果。
