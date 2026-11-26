# Part 3 / E5 改进方向调研

调研日期：2026-11-21。本文保留实验前的方案比较、研究依据和当时的待办，不作为当前运行状态。后续已实现候选 A，并完成远程 pilot、750 更新正式训练与匹配测试；结果未支持改进假设。最终结果、算术一致性下降、题型得失和失败讨论见 [PART3_REPORT.md](prefix_mask/PART3_REPORT.md)，当前执行约定见 [README.md](prefix_mask/README.md)。本次不新增训练；下文未实施的建议（例如 latent 因果干预）不算作已完成实验。

## 结论

优先研究如何让辅助 decoder 更依赖当前题目的 latent。建议首个单变量实验为 **辅助解释 decoder 的前缀 token dropout**：训练时遮蔽一部分真实算式前缀，目标算式保持不变。数字 token 加权 CE 是实现更简单的备选；逐步表示对齐是后续研究选项。不要在首轮同时加入这三项。

理由是现有错误主要表现为“输出自洽但不属于本题的固定算式”，而不是输出式子算错。这个现象支持研究条件信息利用不足，但不能证明 latent 完全没有信息，也不能把本模型直接诊断成 VAE 的 posterior collapse。

用户已于本次调研后补充课程 Part 3 原文，见下节。本建议选择其中“在明确的 latent 预算下改善步骤含义”路线：保持 4 次递推，研究解释对当前题目的对应程度。课程没有要求同时提高压缩率，也没有要求必须采用 GRPO。初始化、数据、训练更新数与评测必须匹配。

## 课程原文与方案对应

来源：用户在当前对话提供的课程要求，非本次从网站下载。

> Part 3: Develop and evaluate one focused improvement (35%)
>
> Choose either greater compression at comparable answer accuracy (e.g., fewer reasoning steps) or clearer latent-step meaning at a stated latent budget. Write a short hypothesis based on Part 2, implement one change within the existing CODI/SIM-CoT pipeline, and train it from the same GPT-2 initialization as the control. Keep the data, training-step budget, and evaluation procedure matched. Compare the modified model with the control on the test subsets using answer accuracy, reasoning length, latency, and the step-level analysis from Part 2. Analyze whether any gain holds across problem types and report unsuccessful results as well. Students may adopt cutting-edge post-training strategies to better improve the accuracy, e.g., Group Relative Policy Optimization.
>
> Expected output: One controlled comparison of the proposed change and the matched control, with code, metrics, and an explanation of the observed trade-off.

| 要求 | 本研究建议 | 尚待完成 |
|---|---|---|
| 两条目标路线任选其一 | 固定 4 次 latent 递推，改善步骤含义 | 冻结最终方法配置 |
| 基于 Part 2 写假设并只改一项 | 前缀信息可能掩盖 decoder 对 latent 的依赖；只增加辅助 decoder 前缀遮蔽 | 实现和验证；当前不能声称假设成立 |
| 相同初始化、数据和训练步数 | 同一个 `shared_init.pt`、2,000 条、750 更新 | 配置与运行证据 |
| 测试子集上的受控比较 | v1 与改进组在三个固定测试子集上比较答案、长度、延迟与可用的步骤参考 | 两组测试解释导出及统一分析；验证集分析不能替代这一项 |
| 跨题型及失败分析 | 先固定题型规则，再报告组内配对变化与样本数 | 题型清单、分组结果和失败案例 |
| 最终代码、指标与取舍解释 | 单一变体和 matched control 的完整结果表 | E5 实验与报告；本文件只完成调研 |

拟用短假设：Part 2 中，辅助 decoder 大量输出与题目无关的自洽算式，且已有小样本交换 latent 诊断影响很小。我们假设，训练时部分遮蔽辅助 decoder 的真实算式前缀，可以增强它对 latent 中题目相关信息的利用，在固定 4 次递推下改善操作数、运算和结果对应程度；同时测量答案准确率是否受损以及延迟是否变化。该假设针对解释可读出性，不预先承诺因果忠实性改善。

## 已完成工作与证据

| 内容 | 当前状态 | 证据 |
|---|---|---|
| 发布 CODI / SIM-CoT 评测 | 三个固定测试集各 200 题；SVAMP 已换成 test 子集补跑 | `e2/PART1_REPORT.md`、`e2/part1_baseline_table.csv` |
| 自训基线 | control_v1 冻结；v2/v3 是已完成的配置诊断 | `e3/active_baseline.json`、`e3/HANDOFF_TO_B.md` |
| Part 2 | 压缩比较、步骤分类、案例核查已形成报告 | `e4/results/REPORT.md`、`e4/CASE_REVIEW.md` |
| 训练/解码排查 | 官方源码与输入对齐、精度诊断、8 题过拟合已做 | `audit_official/FINDINGS.md`、`e3_runs/diagnostic_v1/summary.json` |
| Part 3 | 仓库未见已完成的改进训练与匹配结果 | 本次仅新增本研究记录 |

关键数字：

- v1 验证答案 6/200（3%）；测试 GSM8K 7/200、GSM-Hard 1/200、SVAMP 2/200。
- 验证有效解释位置严格匹配 6/478；限定算式兼容 9/478。
- 操作数不匹配 468/478，结果不匹配 452/478，运算符不匹配 288/478；这些标签重叠。
- 全部 1,000 个导出位置的算式自身成立，但 slot 1 在 188/200 题输出同一算式 `3*2=6`。这个频率包含 PAD 位置，不能当作有效位置错误率。
- 答对的 6 题全部没有完整解释匹配。解释读出与答案成功应分别评价。
- 既有 20 道训练题诊断包含 43 个有效位置，正常 latent 与另一题同位置 latent 交换后的平均 NLL 差仅约 0.0082；汇总 token accuracy 均为 62.9%。样本很小，不能据此断言完全忽略 latent。
- v1 的训练循环日志记录约 4,296.9 秒（1.19 小时），包括验证和 checkpoint I/O、排除加载；这不是总租用工时，也不是新方法耗时承诺。

本次重新运行已有 CPU 检查：E2 3/3、E4 9/9 全部通过，其中包括冻结结果的重新分析与可复算性检查。没有重新加载完整 checkpoint 或进行 GPU 复现。

## GitHub 参考：能借鉴什么，不能直接移植什么

以下核查了仓库页面、README；Sentence-VAE 另查看了具体模型代码。本次未安装、运行或完整审计这些外部工程。公开仓库不代表所有文件均可无条件复制，实际复用需固定提交并保留适用许可。

| 项目与来源 | 可借鉴部分 | 对本项目的适配判断 |
|---|---|---|
| [InternLM/SIM-CoT](https://github.com/InternLM/SIM-CoT) | 用辅助 decoder 从单个 latent 重建对应步骤；推理可移除 decoder | 本项目已采用，不能把“新增步骤监督”再作为 Part 3 创新。继续基于本地固定提交，而非升级当前上游 |
| [zhenyi4/codi](https://github.com/zhenyi4/codi) | 连续推理、自蒸馏的原始实现 | 用于解释和核对原方法；现有官方对齐检查已完成，不应重复以改超参数代替方法改进 |
| [J1mL1/causal-latent-cot](https://github.com/J1mL1/causal-latent-cot) | 逐步干预、充分性和因果影响分析；入口在 `experiments/`、配置在 `configs/` | 很适合借鉴评价设计：把“解码可读出”与“答案确实使用该信息”分开。它是诊断参考，不是直接提升训练效果的模块。仓库标明 CC BY-NC 4.0 |
| [Latent-CoT-Reasoning-as-Dynamical-Systems](https://github.com/SabariIyyappan/Latent-CoT-Reasoning-as-Dynamical-Systems) | GPT-2 CODI/Coconut 与 SIM-CoT 的轨迹分析；`analysis/stability.py`、`analysis/wrappers.py` | 可借鉴相邻 latent 位移、跨题多样性和扰动指标。不要照搬其数据加载流程；本项目继续使用冻结划分。几何变化本身不能证明推理正确 |
| [timbmg/Sentence-VAE 的 model.py](https://github.com/timbmg/Sentence-VAE/blob/master/model.py) | decoder 输入 word dropout，替换输入 token 而保留学习目标 | 提供可查看的最小实现思路，是独立的 RNN/VAE 实现，不是 SIM-CoT 方法，也不是在本任务有效的证据；需要针对 GPT-2 重新实现 |
| [facebookresearch/coconut](https://github.com/facebookresearch/coconut) | 显式 CoT 预热、逐阶段用 latent 替换文字步骤 | 方法相关，但官方 GSM8K 流程先训 CoT 再训 Coconut。直接采用会改变初始化、训练路径和预算，不适合作为现有 v1 的首个单变量增量 |
| [xiaomi-research/colar](https://github.com/xiaomi-research/colar) | 预测压缩 embedding 的辅助目标、随机压缩因子、后续 RL | 可参考“给 latent 更直接的表示目标”。其专用 latent head 和示例 LLaMA 训练流程不是现有 GPT-2 的即插即用模块；不建议整套迁移 |
| [DJC-GO-SOLO/Latent-GRPO](https://github.com/DJC-GO-SOLO/Latent-GRPO) | 已有 latent-SFT 模型上的 GRPO；自定义 SGLang 和 verl | 公开实现确实存在，但词表空间 latent、采样机制、初始化和执行栈与本项目不同。当前约 3% 正确率、固定预算下，工程及稀疏奖励风险较高，不列首选 |

补充的反面参考：[Oli-26/LatentReasoningNoDecode 的实验报告](https://github.com/Oli-26/LatentReasoningNoDecode/blob/main/REPORT.md)记录了另一种 Qwen 并行 latent 架构中，表示蒸馏与 context dropout 未建立稳定行为依赖的负结果。这是作者自报的不同架构实验，不是对本方案的直接否定；它提醒我们不能只凭训练 loss 下降或 latent 更分散宣布成功。

## 候选 A：辅助 decoder 前缀 token dropout（优先）

### 假设与唯一改动

辅助 decoder 在 teacher forcing 时能读取正确算式前缀，可能主要靠这个前缀预测后续 token。训练时削弱该信息来源，可能增加对题目条件 latent 的利用。

基线解释损失为 `CE(D(z_k, s_<t), s_t)`；改进为 `CE(D(z_k, corrupt(s_<t)), s_t)`。仍使用原来的有效位置平均和四项总损失；只改变解释路径的输入污染策略。

建议先固定一个探索性设置 `p=0.3`，不扫描多组参数。该数值是工程起点，不是已验证最优值。正常推理和解释评测关闭这一操作。若三轮内学不起来，应如实报告，不因测试结果持续更换方案。

### 与本地代码的连接

- 位置：`SIM-CoT/CODI/src/model.py` 中两处 `explain_embds` 构建，分别覆盖 z0 和递推后的 z1–z4。未来实现应放在独立 E5 入口/明确的模型变体中，保留可复现的 v1 路径。
- 原始 `indices` 继续生成 labels 和 attention mask，只对喂入辅助 decoder 的文本 embedding 副本施加 token 级遮蔽；latent 向量不遮蔽。
- 可使用零向量替换被选中的整个 token embedding，不缩放其余 embedding；这是针对 GPT-2 的适配，不等同原仓库用 UNK 替换的实现。不要把 GPT-2 的 UNK/EOS 混同为随意可用的 mask token。
- PAD、EOS 与显式边界标记不参与遮蔽；attention mask 仍依照原始 padding。保留真实标签，保持原来的 causal shift；不能把被遮蔽输入对应的监督目标也设成忽略。
- 不能改答案路径、显式参考路径、latent 递推输入或共享 embedding 权重本身。
- 用独立随机数发生器记录遮蔽状态，避免新随机采样改变原有 LoRA dropout 随机序列；恢复 checkpoint 时也恢复此状态。`p=0` 路径不额外消耗随机数。

不增加模型参数或额外 decoder 前向，预期额外成本小，但需用短 pilot 实测。若 latent 本身缺少题目信息，遮蔽会使解释更难，可能降低性能；它不是保证有效的修复。

## 候选 B：数字 token 加权 CE（最简单备选）

只改变解释损失内部各 token 的相对权重，提高数字跨度对应 token 的权重，例如数字为 2、其余为 1：

`L_k = sum_t(w_t * CE_t) / sum_t(w_t)`，再按原逻辑平均有效位置。

它针对 468/478 个位置操作数不匹配的问题，几乎不增加前向开销。数字跨度需覆盖多 token 数字、小数、负号等，并与 shifted labels 对齐；PAD/忽略标签不进入分母。不能只按一个孤立 token 的字符串是否全是数字判断。

这是本项目拟议的损失适配，不应冠以某个检索论文已验证的方法名。局限是数字本来就占算式较大比例，可能增益很小；即使解释更准确，也可能仍靠文字前缀预测，未增强 latent 使用。首轮若选 B，就不同时使用 A。

## 候选 C：逐步骤表示对齐（研究性更强，实施成本更高）

参考 CODI 的表示蒸馏和 CoLaR 的 embedding 目标思想，为每个有效位置构造对应参考步骤表示，增加 latent 与该表示的对齐损失。它是新的适配，不是原论文方法的完整复现。

难点：当前共同初始化的 GPT-2 文字路径本身未必有可靠算术表示；要明确目标在步骤前还是步骤后、是否含题目上下文、是否 stop-gradient、如何处理 slot 4 的合并步骤及 PAD。静态平均 token embedding 会弱化操作顺序，不能默认等同“执行该步骤后的状态”。

暂不优先用批内 InfoNCE：当前 micro batch=1，梯度累积 8 不会自动产生可供对比的 8 个样本。需要额外缓存或跨样本计算，且重复算式、同值异式可能造成假负例，已超出最小改动。

## 建议的最小实验协议

1. **先冻结方案。** 在 A/B/C 中只选一项，写定参数、假设和成功判据；不使用测试集选择参数。
2. **技术检查。** 若实施 A，检查 p=0 与 v1 数值路径一致；p>0 只改变解释输入、labels 不变、PAD 无损失、两处解码路径均生效；验证梯度和保存恢复。小样本调试模型单独归档。
3. **同一起点正式训练。** 从 `shared_init.pt` 开始；2,000 条、seed11、原样本顺序、4 次递推、micro1×累积8、750 更新、lr5e-5、统一 wd0.01，全都继承 v1；使用第 750 步，不挑测试最好的 checkpoint。
4. **统一评价。** 沿用 BF16 独立评测、greedy、64 token 上限。三个固定测试子集均报告答案 EM、latent 数、答案生成 token 数、适用的参考长度比，以及答案延迟均值/中位数/P95。解释导出另行运行、不计入答案延迟。两组最好在同一硬件会话按相同预热/计时规则重测；历史计时保留但不冒充同场测量。200 题验证集用于开发诊断；正式比较另导出两组的测试解释并按下节规则分析。4 个推理输入与 5 个解释位置继续分开。
5. **补充机制诊断。** 对 v1 和改进组采用相同预定题目与 donor 映射，比较正常/跨题同 slot 替换 latent 下的解释 NLL、自由解释和答案变化。只换辅助 decoder 的输入测到的是“解释读出依赖”；要测答案因果影响，必须在推理器实际消费 z0–z3 处干预并重新计算其后状态和 KV cache。slot 4 不作为额外推理输入。
6. **保留失败和成本。** 记录实际 GPU 时长、峰值显存及所有结果，不根据已看过的测试再改选基线。若额外多种子实验超预算，明确单 seed 局限。

### 测试步骤分析与题型划分的适配

现有 `e3/evaluate.py --split test --explain` 可以导出测试解释，但只有包含 `cot` 字段的验证数据会生成 `supervision_slots`。现有 E4 分析绑定验证集，不能不经适配就声称已覆盖 Part 3 测试比较。

- **GSM8K：** 从保存的自然语言参考中提取已有 `<<...>>` 算式，按原来的 5 个解释位置及尾部合并规则对齐。统计运算、操作数、结果和严格/限定兼容匹配，同时披露算式提取覆盖率；没有可用算式的题不计为错误步骤。
- **SVAMP：** 保存的是完整表达式，不是逐步参考链。例如 `(7+12)-15` 不包含中间步骤标注。可比较生成算式与给定表达式的运算/操作数及最终值，并按固定规则做人工案例核查；不得把自动展开的子表达式冒充原始 gold 步骤。逐 slot 严格监督匹配标为不适用，整式指标单列，不与 GSM8K 的步骤匹配率混合。
- **GSM-Hard：** 当前文件无参考链，参考步骤匹配标为 N/A；仍报告解码可解析性、自身算术一致性、模板分布和人工题意核查，清楚区分指标来源。不能套用 GSM8K 原数值链或以模型生成链当 gold。
- **题型：** SVAMP 优先沿用已有 `problem_type`；GSM8K 按参考算式预先固定单步/多步与加减/乘除/混合运算标签。GSM-Hard 没有现成类型，若做题型比较，应按题干独立人工标注固定规则，并在查看改进组输出前完成；不从预测倒推题型。三数据集之间的比较不替代数据集内部题型分析。
- 分组表给出题数、两组正确数、仅改进组正确/仅对照组正确的题数、EM 差和可用步骤指标。小样本组如实报告，不因少数题变化宣称稳定收益；跨类型不一致也是结果。

固定 4 次递推意味着本方案不预期增加结构性压缩率；答案生成长度仍可能变化，延迟必须实测。若解释对应改善但答案准确率降低，结论应明确为可解释性与答题能力的取舍，不能笼统称为整体性能提升。

建议同时报告：答案 EM；有效位置操作数/运算符/结果匹配；严格与限定兼容匹配；各 slot 的输出种类和最高模板占比；正常与交换 latent 的配对 NLL 差；答案与完整解释同时成功的题数。比较解释指标时按题做配对重采样，避免把同题各位置当成独立样本。已有测试集参与过 v1 选择，仍须披露其探索性局限。

判读原则：仅模板减少可能只是输出噪声增加；仅解释改善说明可读出性改善；仅答案改善但干预无影响不能证明逐步使用 latent。只有多项证据一致，才支持“题目相关信息保留或利用改善”的有限结论。

## 下一步交付范围

候选 A 的实现、固定配置、输入遮蔽/恢复检查、远程训练、匹配测试与报告整理均已完成，最终结论为本次固定预算下未获得预期收益，详情见 PART3_REPORT.md。GSM-Hard 实际采用预先固定的数字规模/小数写法分组，不声称完成了上文设想的人工语义题型标注；latent 干预诊断没有实施。本轮以既有负结果收尾，不开展新的训练。完整租用 GPU 工时及历史 manifest 差异原因仍需核实；原始结果文件的本地可访问性边界见正式报告末尾。
