# Part 3 / E5：辅助 decoder 前缀遮蔽

当前状态：已在远程 A6000 完成 10 更新技术 pilot、从共同初始化开始的 750 更新正式训练、两组匹配测试评测及分析。**本次作为负结果完成 Part 3 报告，不启动新的训练。** 完整讨论见 [PREFIX_MASK_REPORT.md](PREFIX_MASK_REPORT.md)：GSM8K 正确数 7/200→4/200，GSM-Hard 1/200→1/200，SVAMP 2/200→2/200；三个子集的生成算式自身一致性分别从 100% 降至 73.7%、71.9%、74.5%，未观察到解释对应改善。正式训练记录见 `training/`，实验前方案依据与课程原文见 [RESEARCH.md](../RESEARCH.md)。

## 假设与固定方案

Part 2 的解释经常是与本题无关的固定算式，已有小样本 latent 交换诊断影响很小。假设训练时削弱辅助 decoder 的真实算式前缀信息，能增强它对题目相关 latent 的利用，在固定 4 次递推下改善步骤含义；同时检验答案准确率和延迟的取舍。本次固定预算实验未支持该假设，失败现象与机制解释的边界见正式报告。

唯一方法变量：辅助解释 decoder 的文本前缀 token 以 **0.3** 概率替换成零 embedding，其余 embedding 不缩放。latent、PAD、EOS、算式边界标记保持原样；所有原始标签、attention mask、答案路径和参考路径不变。训练使用独立 CPU 随机数发生器（seed=405103），避免影响原有 dropout 的随机序列，状态单独保存并恢复。评测不启用遮蔽。

通过仅挂接到辅助 decoder 输入的 hooks 实现，不修改 `SIM-CoT/CODI/src/model.py`、`e3/common.py` 或 `e3/train.py`，也不新增参数。hook 仅在外层 CODI 的训练前向内启用；不能对该模型另行启用未经核验的梯度 checkpointing、编译或不同 decoder 配置。

## 与 control 保持一致

| 项目 | 冻结值 |
|---|---|
| control | `e3_runs/control/last.pt`，v1 |
| 初始化 | 原始 `e3_runs/shared_init.pt`，完整文件 SHA256 严格核验；缺失时拒绝训练，不重建 |
| 公共配置 | `e3/config.json` 必须与 `e3/frozen_v1.json` 完全相同 |
| 数据 | 原始前 2,000 条训练、200 条验证；三个固定测试子集各 200 题 |
| 训练预算 | 3 epochs、750 optimizer updates、6,000 次样本呈现 |
| 批量与顺序 | micro batch 1、累积 8、每轮 seed=11+epoch 的相同打乱顺序 |
| 优化器 | AdamW，lr=5e-5、所有可训练参数单组 wd=0.01 |
| 调度与裁剪 | warmup 38、原线性衰减、clip=1 |
| 推理 | 4 次 latent 递推；解释导出 z0–z4 共5个位置 |
| 最终 checkpoint | 第750次更新；不按测试表现挑选 |

`train.py` 保留 v1 的前向、梯度累积、验证时机、优化器和调度代码，仅增加遮蔽、严格来源检查及相应保存恢复。`contract.json`、`sample_order.json`、`optimizer_groups.json` 和初始化哈希便于审计；恢复训练要求完整 contract 一致。

历史 manifest 哈希差异继续显式记录。当前训练/验证文件的字节与冻结清单核验通过，但不宣称历史清单变化原因已经解决。v1 选择曾参考测试结果，这一局限会写入 E5 报告。

## 本地检查：不启动训练

在仓库根目录执行：

```powershell
python e5/protocol.py
python -m unittest discover -s e5 -p "test_analysis.py" -v
python -m unittest discover -s e5 -p "test_prefix_mask.py" -v
```

第一条不需要 torch，也不会创建模型。分析测试只依赖现有分析环境（含 numpy）；前缀遮蔽测试需要 torch、Transformers 4.52.4、PEFT 0.15.2 及本地 GPT-2 tokenizer。测试用随机 tiny GPT-2 做 CPU 前向/反向与状态恢复，不启动训练入口、不读取完整训练 checkpoint；临时模拟评测数据全部存入临时目录。

正式训练/评测入口强制使用 control 的核心版本：torch 2.7.1、Transformers 4.52.4、PEFT 0.15.2；完整服务器依赖继续用 `SIM-CoT/CODI/requirements.txt`。正式训练环境的软件版本见 `training/environment.txt`，运行耗时与显存记录见 `training/sessions.jsonl`。

## 复现命令（保留记录，本次整理不执行）

以下记录已采用的流程。重新复现时必须使用新输出目录，不能覆盖已完成实验。先恢复原权重并核验：

```bash
python e3/download_checkpoints.py --names shared_init control
python e5/protocol.py --require-weights
```

使用单独目录做技术 pilot，完成后正式训练仍从原始初始化重新开始：

```bash
python e5/prefix_mask/train.py --init e3_runs/shared_init.pt --output e5_runs/prefix_mask_pilot --pilot-updates 10
python e5/prefix_mask/train.py --init e3_runs/shared_init.pt --output e5_runs/prefix_mask_v1
```

中断后恢复正式运行：

```bash
python e5/prefix_mask/train.py --init e3_runs/shared_init.pt --output e5_runs/prefix_mask_v1 --resume
```

pilot 是技术检查，不能用于测试集正式比较，也不能作为正式训练的初始化。两次运行都使用750步的学习率调度；pilot 的额外成本计入全项目 GPU 工时。正式 checkpoint 同时保存优化器、调度器、全局 RNG 和 `prefix_mask_rng`。

训练结束后，用单个入口在同一硬件进程内重测两组测试集：

```bash
python e5/prefix_mask/evaluate.py --control e3_runs/control/last.pt --modified e5_runs/prefix_mask_v1/last.pt --output e5_runs/paired_test_v1
python e5/analyze.py --evaluation e5_runs/paired_test_v1 --output e5_runs/paired_analysis_v1
```

评测不启动训练，不下载权重；要求原 v1 control、同初始化、同数据顺序、同配置和完成750更新的 E5 checkpoint。任何输出目录必须为新目录。两组均输出完整600题答案及3,000个解释位置，完成标记保存文件哈希；分析拒绝缺题、重复/错序 ID、损坏文件、不同 pairing ID、不同硬件/精度/评测策略等不匹配输入。

答题延迟包含题目处理、latent 递推和答案生成，排除加载、tokenization、文字解码和辅助解释导出。每集5次预热、CUDA同步计时；两组先control后modified，仍须承认运行先后带来的计时波动。

## 测试步骤与题型分析

`problem_types.json` 已在未查看改进组结果时生成，固定600题的分类，分析时重新计算规则并核对。不得据改进组预测改标签。

- GSM8K：提取原参考中的 `<<...>>` 算式，按5个位置和尾部合并规则比较；当前200题中195题可提取算式，5题无可用算式。缺失参考不算步骤错误。按参考步骤数与运算类别分组。
- SVAMP：沿用原始 Type，另按参考表达式运算类别分组。只有完整表达式，因此单列 slot0 与任意slot的整式兼容率；不制造中间 gold 步骤、不把整式指标混入 GSM8K 步骤匹配。
- GSM-Hard：没有参考链，不填写参考步骤匹配率；保留可解析性、自身算术一致性、模板分布与案例。按题干最大数字和小数写法分析，明确这些是表面特征，尚无独立人工语义题型标注。
- 所有数据集均有数字规模/小数写法分组；数字规模包括题干中的年份、百分比等数字，不声称准确度量算术难度。各维度分组可重叠，不跨维度相加。

输出与文件说明（以下路径均相对于 `e5/prefix_mask/`）：

| 文件 | 内容 |
|---|---|
| `README.md` | 本方案的假设、唯一改动、受控训练条件、运行命令和文件说明 |
| `PREFIX_MASK_REPORT.md` | 整理撰写的完整实验报告：结果、分题型收益与退化、失败原因及局限 |
| `prefix_mask.py` | 前缀遮蔽核心实现：仅在训练时将选中的辅助 decoder 文本前缀 embedding 置零 |
| `train.py` | 从共同初始化进行前缀遮蔽训练，保存训练记录及 checkpoint |
| `evaluate.py` | 在同一进程依次评测 v1 与前缀遮蔽，导出三个固定测试子集的答案、解释和延迟 |
| `test_prefix_mask.py` | 遮蔽位置、标签保护、随机状态保存恢复及评测时关闭遮蔽等代码检查 |
| `__init__.py` | Python 包标识，支持导入本目录中的代码 |
| `config.json` | 前缀遮蔽专有设置：概率0.3、独立随机种子、零 embedding 替换及边界保护 |
| `configuration.json` | 公共训练配置与遮蔽设置的便查汇总；不替代原始训练 contract |
| `checkpoint/last.pt` | 第750次更新的最终 checkpoint，包含模型参数、优化器、调度器、配置与随机状态 |
| `training/contract.json` | 正式训练配置、初始化身份、数据与源码哈希、训练顺序及更新预算 |
| `training/environment.txt` | 正式训练环境的软件包及版本 |
| `training/initialization.json` | 共同初始化文件的路径、哈希与基础模型来源 |
| `training/optimizer_groups.json` | 优化器参数组、可训练参数名称与 weight decay 设置 |
| `training/parameters.json` | 总参数量、可训练参数量及各模块统计 |
| `training/preflight.json` | 正式训练前的配置、数据、顺序与初始化预检查记录 |
| `training/sample_order.json` | 750次更新对应的样本索引顺序，每次更新8个样本，共6,000次样本呈现 |
| `training/sessions.jsonl` | 训练会话的起止更新次数、循环耗时和峰值已分配显存 |
| `training/train_log.jsonl` | 750次更新的损失、学习率、梯度范数、累计耗时及训练内验证记录 |
| `validation/config.json` | 独立200题验证集评测的 checkpoint 身份、精度、生成与计时设置 |
| `validation/summary.csv` | 前缀遮蔽的验证集汇总：准确率、长度、延迟、解析失败与截断情况 |
| `validation/validation.predictions.jsonl` | 200道验证题的原始答案输出、正确性、token 长度及逐题耗时 |
| `validation/validation.decoded.jsonl` | 同一批验证题的参考算式与5个辅助解释槽输出 |
| `test/config.json` | 前缀遮蔽固定测试集评测配置，包含与 v1 对照的配对标识和训练来源 |
| `test/complete.json` | 600题测试导出的完成标记及结果文件哈希 |
| `test/summary.csv` | 前缀遮蔽在 GSM8K、GSM-Hard、SVAMP 上的准确率、长度和延迟汇总，各200题 |
| `test/*.predictions.jsonl` | `gsm8k`、`gsm_hard`、`svamp` 各自的200题答案、正确性、长度和逐题耗时，共3个文件 |
| `test/*.decoded.jsonl` | 上述三个测试子集的逐题辅助解释及可用参考信息，共3个文件；这是原始解释输出，尚不是步骤评分表 |
| `paired_analysis_v1/summary.csv` / `summary.json` / `summary.jsonl` | v1与前缀遮蔽两组×三个测试集的准确率、长度、延迟、步骤／整式指标及失败计数，共6组结果 |
| `paired_analysis_v1/paired_questions.csv` / `paired_questions.jsonl` | 600道测试题的两组答案、解释、答题得失与题型配对记录 |
| `paired_analysis_v1/steps.csv` / `steps.jsonl` | 两组共6,000个解释槽的算术、操作数、运算符、结果与参考匹配评分及证据；保留不适用项 N/A |
| `paired_analysis_v1/problem_types.csv` / `problem_types.jsonl` | 各题型两组指标、样本数、新增答对、失去正确及准确率变化 |
| `paired_analysis_v1/slots.csv` / `slots.jsonl` | 按模型、数据集和解释槽位置统计不同输出、最常见算式及模板占比 |
| `paired_analysis_v1/categories.csv` / `categories.jsonl` | 操作数不匹配、结果不匹配、算术不自洽等错误类别的数量与分母 |
| `paired_analysis_v1/REPORT.md` | 保留的历史配对分析报告，包含总体比较、分题型结果及成功／失败案例 |
| `paired_analysis_v1/provenance.json` | 配对分析的输入配置、来源、源码与分类哈希，以及分析局限 |

同一行中的同名 CSV、JSON、JSONL 文件均位于该行给出的子目录。CSV 便于表格查看，JSON／JSONL 便于程序读取。`__pycache__/` 是 Python 自动缓存，不属于实验交付结果。

`../analyze.py` 负责生成配对分析的 CSV、JSON、JSONL 文件；其中的 Markdown 报告生成函数已删除，因此重新运行不会生成或更新已有的 `paired_analysis_v1/REPORT.md`。六模型统一测试比较及其结果保留在上一级 `e5/`，不放入本方案目录。

整式/算式兼容属于限定的自动指标，不等于一般数学等价或题意忠实性。解释改善和答案改善分别报告；模板减少也可能来自噪声。没有实现额外的 latent 因果干预实验，因此报告不会声称证明因果忠实性。


