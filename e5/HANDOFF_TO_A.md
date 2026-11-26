# 给 A 的交接：E4、E5 完成情况及 E6、E7 已覆盖内容

**E4 的分析与案例文档、E5 的五个改进模型训练和测试已经有交付；B 在实施 Part 3 时，也完成了 E6 的主要量化比较和 E7 的自动步骤评分。A 不需要从零重做这些统计，后续重点是人工复核、报告整合及交付验收。**

文中路径链接相对于本文件；代码块中的路径相对于项目根目录 `D:\NLP_Final_Project`。只保留 **v1 control、prefix_mask、P1、P2、P3、P4** 为本次正式比较范围
## 1. 按 E4–E7 分工核对

| 任务及原负责人 | B 已完成的内容 | 当前状态与 A 的后续工作 |
|---|---|---|
| E4 压缩与步骤诊断（B） | 发布版 CODI、SIM-CoT 与 v1 的准确率/长度/延迟整理；v1 held-out 解释解码结果分析；步骤评分规则、算术一致性、重叠错误标签、模板统计；成功/失败案例及参考数据问题解读；据此形成改进假设 |  |
| E5 训练改进模型（B） | 五个独立单改动方案的假设、代码、配置、最终 750-update checkpoint；验证和固定测试导出；机制诊断；四个方案有独立训练日志和会话耗时 | |
| E6 答案与效率对照评测（A） | 六模型同一固定测试子集的统一离线分析；答案准确率、latent 预算、答案长度、参考长度比、历史延迟；逐题错转对/对转错；分题型收益和退化 | 主要量化交付已覆盖。A 重点整理最终表格和跨题型结论 |
| E7 步骤可解释性对照评测（A） | 沿用 E4 规则，对六模型测试解释评分；参考匹配、算术有效性、错误类型、模板分布及逐题步骤变化；前缀遮蔽已有案例讨论 | 自动分析已完成，**未见覆盖五方案的完整测试集配对案例人工核验交付**。A 需核验初评、整理改善与退化案例，并解释答案和解释指标不一致的情况。 |



## 2. E4：已完成内容和阅读入口

建议先读 [E4 报告](../e4/REPORT.md)、[评分规则与文件说明](../e4/README.md)、[案例解读](../e4/CASE_REVIEW.md)。

| 交付 | 文件 | 内容与使用边界 |
|---|---|---|
| 压缩、答案、延迟比较 | [compression_comparison.csv](../e4/results/compression_comparison.csv) | 三个固定测试子集，各 200 题；比较发布版 CODI、发布版 SIM-CoT 和 v1。发布版 K=6，v1 K=4，但训练规模等也不同，不能将其当作只改变步数的受控消融。 |
| 评分实现 | [analyze.py](../e4/analyze.py) | ID/来源核对、算式解析、严格匹配、限定兼容及错误标签。当前只写结构化统计，不生成或覆盖 E4 报告。 |
| v1 逐题与逐步结果 | [questions.csv](../e4/results/questions.csv)、[steps.csv](../e4/results/steps.csv) | 200 道独立验证题；共 1,000 个解释位置，其中 478 个有参考，522 个为 PAD。对应 JSONL 同目录可用。 |
| 汇总与错误分布 | [summary.json](../e4/results/summary.json)、[categories.csv](../e4/results/categories.csv)、[slots.csv](../e4/results/slots.csv) | 总体指标、重叠错误类别、按位置模板频率。 |
| 来源记录 | [provenance.json](../e4/results/provenance.json) | 数据、checkpoint 身份和来源哈希记录。 |
| 案例与参考质量核查 | [CASE_REVIEW.md](../e4/CASE_REVIEW.md) | 答对但解释不匹配、解释匹配但答案错、长链尾槽合并、参考舍入及题目歧义等。 |

E4 的关键发现是：v1 验证集答对 6/200；严格匹配 6/478，限定算式兼容 9/478；全部 1,000 个生成位置算术成立，但解释高度模板化且通常与本题不对应。因此改进目标是**在固定 K=4 下改善解释与题目/参考步骤的对应程度**，不能把问题简单写成“模型不会算乘法”。

评分约定已经落实：只在非 PAD 位置评价参考匹配；算术使用受限解析和精确分数；尾部多个参考算式合并到第五个监督槽；错误类别允许重叠；近似小数、缺失参考与不可解析分别处理。限定兼容不是完整的数学等价判定，也不是因果忠实性证明。

E4 案例文档明确留下两项参考质量问题给 A 复核：`gsm8k-aug-train-2049` 的倍数措辞/答案歧义，以及 `gsm8k-aug-train-2189` 的离散计数与比例不完全一致。当前未修改 gold，也未剔除这些题；如需要更改官方数据，应单独记录依据，不能默默改分。

## 3. E5：五个独立改进方案已完成什么

共同基线是 [冻结 v1](../e3/frozen_v1.json)，共同配置见 [e3/config.json](../e3/config.json)。五个方案各自从原始 `e3_runs/shared_init.pt` 开始，不从已训练 v1 继续训练，也不把五种改动叠加。

共同设置：2,000 道训练题、200 道验证题、3 epochs、750 次 optimizer updates、K=4、micro batch 1、梯度累积 8、seed 11、学习率 5e-5。最终比较使用第 750 次更新的 checkpoint。

| 方案 | 假设与唯一改动 | 核心代码 | 文档及配置 |
|---|---|---|---|
| 前缀遮蔽 | 真实算式前缀可能掩盖 decoder 对 latent 的利用；训练时以 0.3 概率将符合条件的辅助 decoder 前缀 token embedding 置零，保护 latent/边界等位置，评测关闭遮蔽 | [prefix_mask.py](prefix_mask/prefix_mask.py)、[train.py](prefix_mask/train.py) | [README](prefix_mask/README.md)、[报告](prefix_mask/PREFIX_MASK_REPORT.md)、[config.json](prefix_mask/config.json)、[configuration.json](prefix_mask/configuration.json) |
| P1 | 探索解释监督介入时机；前 250 次更新将解释损失系数由 0.1 升至 1，之后保持 1 | [train_p1.py](p1/train_p1.py)、共用目录中的 [p1_schedule.py](p1_schedule.py) | [P1_README](p1/P1_README.md)、[configuration.json](p1/configuration.json) |
| P2 | 加强解释对数值操作数的学习；每个参考算式首个数字操作数对应 token 的 CE 权重设为 2，其余有效 token 为 1 | [train_p2.py](p2/train_p2.py)、[p2_operand.py](p2/p2_operand.py) | [P2_README](p2/P2_README.md)、[configuration.json](p2/configuration.json) |
| P3 | 先训练参考路径，可能改善后续自蒸馏；前 150 次更新只优化原参考路径 CE，后 600 次恢复原联合损失 | [train_p3.py](p3/train_p3.py)、[p3_stage.py](p3/p3_stage.py) | [P3_README](p3/P3_README.md)、[configuration.json](p3/configuration.json) |
| P4 | 参考算式末尾的上下文表示可能提供过程监督；原目标上增加系数 0.1 的有效步骤表示余弦对齐损失，参考端 stop-gradient | [train_p4.py](p4/train_p4.py)、[p4_alignment.py](p4/p4_alignment.py) | [P4_README](p4/P4_README.md)、[configuration.json](p4/configuration.json) |

受控条件还需要保留两点解释：P1 同时改变解释损失的时间分布和累计权重，不能单独归因于“时机”；P3 总更新数仍为 750，但 latent/decoder 联合训练只有后 600 次，不能说 latent 路径也获得与 v1 相同的 750 次训练。

每个方案目录均有 `checkpoint/last.pt`、`validation/` 和 `test/`。例如 P2 的最终权重为 `e5/p2/checkpoint/last.pt`；其余方案按同样结构存放。验证/测试目录保存配置、汇总、逐题预测和解码解释。v1 的对应副本位于 [control_validation](control_validation/) 和 [control_test](control_test/)。

训练记录与耗时：

| 方案 | 原始训练记录 | 正式循环耗时 | 当前缺项 |
|---|---|---:|---|
| 前缀遮蔽 | [training/](prefix_mask/training/)：contract、初始化/顺序/优化器等审计、train_log、sessions、environment | 4,297.63 秒（71.63 分钟） | 本表所列正式训练记录可用 |
| P1 | `configuration.json` 为配置汇编；测试配置记录最终 750 updates | 当前不能由本目录独立会话日志核实 | 当前没有 `p1/training/`；需补原始 contract、train_log、sessions |
| P2 | [training/](p2/training/)：contract、train_log、sessions | 4,064.05 秒（67.73 分钟） | 本表所列正式训练记录可用 |
| P3 | [training/](p3/training/)：contract、train_log、sessions | 3,476.25 秒（57.94 分钟） | 150-update probe 结果保留，但中间 `reference150.pt` 不在当前目录 |
| P4 | [training/](p4/training/)：contract、train_log、sessions、alignment_audit | 4,195.72 秒（69.93 分钟） | 旧分析所需 `p4_preflight.json` 当前缺失 |

这些耗时是 sessions 记录的训练循环墙钟时间，包含循环内验证和 checkpoint I/O，排除模型加载；不等于整个项目的 GPU 租用时间，也未包含全部 pilot、诊断、测试成本。需要总预算时应分项补齐，不能用四项记录推算 P1。

附加诊断也已保存：共用入口 [p0_diagnose.py](p0_diagnose.py)，v1 的 [control_diagnostics](control_diagnostics/) 与各 P1–P4 的 `diagnostics/` 保存固定 32 道训练题＋32 道验证题的梯度/状态替换诊断；[p3_reference_probe.py](p3_reference_probe.py) 及 v1/P3/P4 参考路径结果用于机制检查。它们是小样本诊断，不能替代固定测试集评价，也没有证明 latent 完全不含信息或参考路径是可靠强教师。

## 4. E6：已经完成的答案、效率和题型比较

### 4.1 统一测试与代码分工

固定测试子集为 GSM8K（ID）、GSM-Hard（OOD）、SVAMP（OOD），各 200 题；六个模型每个均有 600 条逐题预测及对应解释。

| 工作 | 使用的代码 | 说明 |
|---|---|---|
| v1/前缀遮蔽同场推理、计时与解释导出 | [prefix_mask/evaluate.py](prefix_mask/evaluate.py) | 两模型同一进程先后评测；不是交错重复的六模型基准。 |
| P1–P4 通用推理与解释导出 | [e3/evaluate.py](../e3/evaluate.py) | 各模型独立 BF16 导出；原配置记录源码与 checkpoint 身份。 |
| 六模型正式固定测试统一比较 | [analyze_test_suite.py](analyze_test_suite.py) | 离线读取已保存预测/解释，核对固定 ID、gold、来源，生成统一比较。 |
| 共用步骤评分及汇总 | [analyze.py](analyze.py)、[e4/analyze.py](../e4/analyze.py) | 六模型分析复用这些函数；`analyze.py` 当前不再生成前缀遮蔽 Markdown 报告。 |
| 题型划分 | [problem_types.py](problem_types.py)、[problem_types.json](problem_types.json) | 分组不读取模型预测，六模型采用同一分类。 |

`analyze_p1.py`–`analyze_p4.py` 主要服务此前的 200 题验证集、训练审计与专项诊断。它们及 P1–P4 的历史 REPORT 不应替代本节正式测试结果；旧报告中的“尚未测试”等进度描述也不代表当前状态。

### 4.2 A 可以直接使用的结果

全部统一结果位于 [test_suite_analysis_20261126/](test_suite_analysis_20261126/)。

| 文件 | 已覆盖的交付 |
|---|---|
| [REPORT.md](test_suite_analysis_20261126/REPORT.md) | 六模型答案与步骤总表、配对得失、分组改善/持平/退化及局限 |
| [summary.csv](test_suite_analysis_20261126/summary.csv) | 18 行模型×数据集汇总：EM、K、答案/参考长度、长度比、平均/中位/P95 延迟、步骤指标 |
| [paired_questions.csv](test_suite_analysis_20261126/paired_questions.csv) | 3,000 条改进模型与 v1 的逐题配对：题干、gold、双方答案/解释、错转对和对转错、步骤变化 |
| [problem_types.csv](test_suite_analysis_20261126/problem_types.csv) | 170 行总体/分组比较：样本数、准确率及步骤变化、gained/lost、小组标记 |
| [SIX_MODEL_LATENCY_SUMMARY.md](SIX_MODEL_LATENCY_SUMMARY.md)、[SIX_MODEL_LATENCY.csv](SIX_MODEL_LATENCY.csv) | 六模型已有计时的集中整理与计时边界 |
| [provenance.json](test_suite_analysis_20261126/provenance.json)、[complete.json](test_suite_analysis_20261126/complete.json) | 来源配置/哈希、完成状态、输出文件校验清单 |

正式测试答案结果如下；每格均为“正确数/200（EM）”：

| 模型 | GSM8K | GSM-Hard | SVAMP |
|---|---:|---:|---:|
| v1 | 7/200（3.5%） | 1/200（0.5%） | 2/200（1.0%） |
| 前缀遮蔽 | 4/200（2.0%） | 1/200（0.5%） | 2/200（1.0%） |
| P1 | 3/200（1.5%） | 1/200（0.5%） | 2/200（1.0%） |
| P2 | 4/200（2.0%） | 1/200（0.5%） | 1/200（0.5%） |
| P3 | 3/200（1.5%） | 2/200（1.0%） | 5/200（2.5%） |
| P4 | 6/200（3.0%） | 2/200（1.0%） | 1/200（0.5%） |

没有方案在三个数据集上都取得净准确率提升。P3 在两个 OOD 子集增加正确数，但 GSM8K 下降；前缀遮蔽两个 OOD 总正确数不变，也存在新增正确与失去正确相互抵消，不能写成“逐题行为未变”。全部结果都是单 seed、小样本下的探索性比较。

题型分析不是空缺：GSM8K 包含参考步骤数/运算类型，SVAMP 使用原生 Type 和表达式信息，另有数字规模/小数等维度。GSM-Hard 无参考链，相关分组主要是表面数字特征，不能写成已经人工核验的完整语义题型。不同维度可能重叠，不能跨维度相加。A 需选取有代表性的改善和退化分组，带上样本数解释，不能只摘正收益组。

长度比较也已完成：所有模型 K=4，没有获得更少 latent 步数的结构性压缩收益。GSM8K 的参考 token/latent 步数比为 24.71375，SVAMP 为 2.71375，GSM-Hard 因无参考链为 N/A。该比值不是加速倍数；GSM8K 自然语言参考与 SVAMP 整式参考也不是同一种长度语义。答案 token 均值与含额外标记的 generated token 均值应分开引用。

延迟采用 A6000、BF16、batch size 1、greedy、K=4 的已有计时，覆盖题目前向、latent 递推和答案生成，排除加载、分词及额外解释生成。**现有六模型汇总足以报告历史延迟，但不足以将跨会话差异归因于方案加速。**当前交付沿用现有计时，不默认要求重跑；只有要主张可靠加速时，才另行安排同硬件、统一会话、重复/交错测量。

## 5. E7：已完成的自动分析与仍待人工完成的部分

### 5.1 自动步骤对比已有完整结果

| 文件 | 内容 |
|---|---|
| [steps.csv](test_suite_analysis_20261126/steps.csv) / 同名 JSONL | 18,000 个解释槽位：生成式、参考、算术有效性、匹配/错误标签及证据 |
| [categories.csv](test_suite_analysis_20261126/categories.csv) / 同名 JSONL | 162 行模型×数据集×错误标签计数及分母，可对照 v1 得到错误类型变化 |
| [slots.csv](test_suite_analysis_20261126/slots.csv) / 同名 JSONL | 90 行位置级模板统计，包括最常见解释及占比 |
| [paired_questions.csv](test_suite_analysis_20261126/paired_questions.csv) / 同名 JSONL | 同题双方解释、严格/兼容匹配数量及变化；可直接筛选人工核验候选 |
| [前缀遮蔽配对报告](prefix_mask/paired_analysis_v1/REPORT.md)、[叙述报告](prefix_mask/PREFIX_MASK_REPORT.md) | 前缀遮蔽已有专门统计、题型讨论和部分案例文字；历史相对链接和来源描述需按当前目录理解 |

GSM8K 测试集 200 题中，195 题能提取参考算式，共 634 个有效参考槽位。六模型结果为：

| 模型 | 严格匹配 /634 | 限定兼容 /634 |
|---|---:|---:|
| v1 | 4 | 8 |
| 前缀遮蔽 | 3 | 5 |
| P1 | 5 | 9 |
| P2 | 5 | 7 |
| P3 | 2 | 4 |
| P4 | 3 | 9 |

个别步骤数略增并未伴随普遍答案收益，不能据一两个槽位变化认定稳健改进。前缀遮蔽的生成算式自身成立率由 v1 的 100% 降至 GSM8K 73.7%、GSM-Hard 71.9%、SVAMP 74.5%，且参考对应未改善，应如实作为失败结果报告。

解释评价边界必须保留：

- K=4，但解码 z0–z4 共五个位置；最后一个是诊断/监督读出，不是第五次推理输入。
- E4 的 **478** 是验证集有效参考槽位；本次 **634** 是 GSM8K 测试集有效槽位，不能直接作前后提升比较。
- GSM-Hard 没有参考链，参考对齐指标为 N/A，不是零，也不是漏做。
- SVAMP 只评价整题表达式兼容，不能冒充逐步 gold 监督；当前六模型 slot 0/任一 slot 整式兼容均为 0/200。
- 自身算术有效性可统计全部生成槽位；参考错误率使用有参考位置。两者分母不同，PAD 不应作为“缺失了推理步骤”。
- 错误标签允许重叠；精确算术触发的参考舍入标记需结合语义复核；解码对应好坏不等于 latent 的因果忠实性。

### 5.2 A 仍需完成的 E7 人工交付

目前已有机器评分和部分案例文字， A 完成五方案统一测试对比的人工审核。

从 `paired_questions.csv` 按 `answer_outcome`、`strict_match_count_delta`、`conservative_equivalent_count_delta` 筛候选；再回查 `steps.csv` 和原始 `test/*.decoded.jsonl`。覆盖以下情况：答案错转对、答案对转错、步骤对应改善、步骤对应退化、答案与解释改善方向不一致。


## 6. Checkpoint 交付与使用

**B 已将前缀遮蔽、P1、P2、P3、P4 五个最终 checkpoint 全部上传到 GitHub Release。**每个模型采用与 v1 相同的压缩分片方式，共十个权重分片。本地原始文件保留在 `e5/<方案>/checkpoint/last.pt`。

Release 地址：

<https://github.com/azhao258/NLP_Final_Project/releases/tag/e5-checkpoints-2026-11-26>

下载恢复使用 [checkpoints_release.json](checkpoints_release.json) 和 [download_checkpoints.py](download_checkpoints.py)：前者记录分片、文件大小及 SHA-256，后者负责下载、校验并恢复权重。本地 `*.verified.json` 只是打包收据，A 不需要使用。

A 从项目根目录执行，即可下载并恢复全部五个模型：

```powershell
python e5/download_checkpoints.py
```

只恢复某一方案可执行：

```powershell
python e5/download_checkpoints.py --names p3
```

脚本校验分片及恢复后的 checkpoint，恢复到 `e5/<方案>/checkpoint/last.pt`。若只从 Release 下载了独立脚本，可把脚本放到项目根目录运行 `python download_checkpoints.py`。v1 与 shared_init 仍按 [E3 交接文档](../e3/HANDOFF_TO_B.md) 的原 Release 获取。



