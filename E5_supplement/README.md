# E5 补充实验：GRPO 采样检查与 P5

2026-11-27 至 2026-11-28 完成。此目录归档 A 的补充实验，以及后续 E6/E7 离线分析。**冻结基线仍为 v1；没有进行 GRPO 参数更新。P5 完成独立的 750-update 训练，但没有获得整体准确率或解释质量提升。**

本次新增文件全部位于 `E5_supplement/`，不覆盖仓库原 E1–E5 文件。原代码按执行时字节保留，P5/GRPO 主代码哈希与返回配置核对一致。原 README 中的“尚未进行 GPU 验证”等表述是实验准备时记录；完成状态以本文件及返回结果为准。

## 目录与阅读顺序

| 路径 | 内容 |
|---|---|
| `GRPO/sample_check.py` | 不更新模型的答案采样检查 |
| `GRPO/results/smoke4/`、`v1_train64_g8/` | 配置、逐题候选、汇总及运行前后权重哈希 |
| `P5/train.py`、`method.py`、`negatives.py` | P5 训练、间隔损失、训练集负例构造 |
| `P5/train_negative_pairs.json` | 冻结训练集负例，非验证/测试负例 |
| `P5/runs/formal/` | 正式训练 contract、初始化、顺序、日志、环境和耗时 |
| `P5/runs/pilot*/` | 诊断运行记录，不作为正式效果评测 |
| `P5/results/validation/`、`test/`、`retrieval/` | 200 题验证、600 题测试、200 题验证集检索诊断 |
| `diagnostics/` | 检索脚本与原使用说明 |
| `E6_E7_completion/` | 七模型统计代码、原始补充输入、完整统计及 34 个案例 AI 初审 |
| `prepare_runtime.py` | 将原代码安全复制到所需运行路径；拒绝覆盖不同文件 |
| `MANIFEST_SHA256.json` | 本目录文件 SHA-256 清单（自身除外） |

先阅读 [E6/E7 报告](E6_E7_completion/results/REPORT.txt)、[配对案例](E6_E7_completion/results/CASE_REVIEW.txt) 和 [复现说明](E6_E7_completion/README.txt)。`E6_E7_completion/p5_outputs` 是离线分析输入快照，与上层 P5 返回记录重复，保留是为了按其 README 复现。

## GRPO：只做采样，不做训练

- 使用冻结 v1；从 2,000 道训练题固定抽 64 题，每题 8 个采样答案，另保留贪心答案。
- K=4，温度 1，最大 64 tokens；随机性施加于答案生成，latent 递推不随机。
- 权重运行前后哈希相同，未执行反向传播或优化器更新。
- 贪心正确 **2/64**，采样正确 **5/512**；**5/64** 组有正确/错误奖励差异，**59/64** 组全部答错。
- 每组平均 7.72 个不同解析答案；解析失败 3/512，无截断。

结论：输出多样性足够，但二元正确性奖励稀疏。不把“8次至少一次正确”的 5/64 当成部署准确率或训练收益。这是训练集可行性诊断，不是正式测试成绩。

## P5：训练集跨题同槽位间隔约束

`L = L_v1 + 0.1 × mean(relu(0.02 + cos(z,h_negative) - cos(z,h_positive)))`

- 从原共同 `shared_init.pt` 开始；2,000 train / 200 validation，seed=11，K=4，3 epochs，750 updates，lr=5e-5，micro batch=1、累积=8；其余公共配置保持 v1。
- 负例只从训练集构造：不同题目、相同槽位，算式精确成立、算式数相同、结果集合不相交。
- 1,984 个 anchor 有配对、3,744 个配对槽；剩余 16 题仍进行原训练。额外 donor 前向不改变 anchor 顺序。
- 正负参考均额外 eval/no_grad 前向，参考网络随训练变化；不是冻结强教师。与 P4 的训练模式参考前向不同。
- 训练更新预算一致，**不等于相同 FLOPs/耗时**。正式循环 4,419.79 秒，约 73.7 分钟，非总租赁时长。
- 保留第五个解释读出槽，但只有四次 latent 递推；不声称提高压缩率。

| 固定测试集 | v1 | P4 | P5 |
|---|---:|---:|---:|
| GSM8K | 7/200 | 6/200 | 6/200 |
| GSM-Hard | 1/200 | 2/200 | 1/200 |
| SVAMP | 2/200 | 1/200 | 2/200 |

P5 对 v1 全测试新增正确 3 题、丢失正确 4 题。GSM8K 测试严格步骤匹配 3/634、保守兼容 9/634；v1 为 4/634、8/634。P5 全部测试解释槽中 95.67% 为 `2*3=6` 或 `3*2=6`。

验证集检索：同题步骤 v1 187/437 → P5 194/437；跨题同槽位 6/478 → 7/478。小幅变化未转化为整体准确率或解释收益，不声称稳定显著提升。

## 运行方法

在完整仓库根目录执行；不要直接运行嵌套目录中的原训练脚本，否则其项目根目录解析会错。

```bash
python E5_supplement/prepare_runtime.py --project-root .
python -m unittest P5.test_rules
python P5/preflight.py
```

复制入口先检查全部目标，发现同名不同内容即停止，不修改现有代码。结果归档不会复制到运行目录，以免续训误用。

准备原 GPT-2 assets、原 shared_init（P5）及 v1 checkpoint（GRPO）；使用原 Python 3.12 / torch 2.7.1 / transformers 4.52.4 / peft 0.15.2 环境。依赖清单见项目 `SIM-CoT/CODI/requirements.txt`，GPU 环境复现，不在 CPU 上训练。

```bash
# GRPO 无更新检查
python -u GRPO/sample_check.py --n 4 --output GRPO/results/new_smoke4
python -u GRPO/sample_check.py --n 64 --output GRPO/results/new_train64

# P5 新运行必须用空输出目录
python -u P5/train.py --init e3_runs/shared_init.pt --pilot-updates 2 --output P5/runs/new_pilot
python -u P5/train.py --init e3_runs/shared_init.pt --output P5/runs/new_formal
python e3/evaluate.py --checkpoint P5/runs/new_formal/last.pt --split validation --explain --output P5/results/new_validation
python e3/evaluate.py --checkpoint P5/runs/new_formal/last.pt --split test --explain --output P5/results/new_test
python e5/check_reference_retrieval.py --checkpoint P5/runs/new_formal/last.pt --n 200 --output P5/results/new_retrieval
```

E6/E7 离线复算无需 GPU，使用新输出目录：

```bash
python E5_supplement/E6_E7_completion/build_analysis.py --repo . --p5 E5_supplement/P5 --output E5_supplement/analysis_rerun
python E5_supplement/E6_E7_completion/add_diagnostics.py --probes E5_supplement/E6_E7_completion/probe_inputs --output E5_supplement/analysis_rerun
python E5_supplement/E6_E7_completion/finalize_review.py --repo . --p5 E5_supplement/P5 --results E5_supplement/analysis_rerun
```

## 交接边界

- **本包不含 P5 `last.pt`**：本机收到的归档排除了权重。最终评测所记录 SHA-256 为 `07c3a6ccf9a254658198f7616f5a3b5e7710443473099b37403fdc6d9e6d544e`，尚不能据此声称已上传/独立校验权重文件。训练日志、配置与全部预测已归档。
- 原 shared_init、v1 不重复上传，使用 E3 的既有 checkpoint 发布。GRPO 没有新权重。
- E7 的 34 个案例是 AI 初审，A/B 人工签字待完成，见 `case_review.csv`；不是人类盲审。
- 单 seed、小样本、v1 选型曾参考测试集；结果为探索性比较。历史延迟不能证明跨模型加速。
- GSM-Hard 无参考链，对齐指标 N/A；SVAMP 只评完整表达式，不虚构步骤链。
- `gsm-hard-train-1051` 有速度/耗时措辞歧义，参考标记亦有不完整情况。保留官方gold与原统计，案例中披露，不剔除失败题。
