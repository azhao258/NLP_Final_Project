> 历史运行指南。当前最终基线与交接规则以 [HANDOFF_TO_B.md](HANDOFF_TO_B.md) 和 active_baseline.json 为准。

# E3：从 GPT-2 开始训练正式 SIM-CoT 对照组

负责人 A。对应 Part 1 的自训 checkpoint，并为 Part 2 解码诊断、Part 3 匹配实验提供基准。不是重训两篇论文，也不是继续微调发布权重。只训练一个带辅助 decoder 的 SIM-CoT control，使用仓库中已经验证过的 CODI 路线实现。

## 原理：这个 control 学什么

模型从同一个 GPT-2 小模型基座开始，有两个共享推理器参数的训练路径。

1. 文字路径看到问题、数据提供的算式链、答案，通过交叉熵学习显式推理和答案。
2. 隐式路径看到问题，递推连续向量，再生成答案，通过答案交叉熵学习答题。
3. 自蒸馏将隐式路径与文字路径在“开始预测数值答案之前”的各层隐藏状态对齐。文字路径用于对齐的目标停止梯度；另一次文字路径前向负责有梯度的显式任务学习。它不是直接把每个文字 token 一对一压缩成向量。
4. 辅助 GPT-2 decoder 从指定 latent 向量预测对应算式的文本，提供步骤监督；推理答题时不需要这个 decoder。解释导出单独运行，不计入答案延迟。

总损失沿用官方实现：L = L_answer + L_reference + L_distillation + L_explanation，三个可配置系数均为 1。蒸馏使用按参考状态标准差归一化的 Smooth L1；解释损失按有效监督位置平均。E3 不增加数字加权、GRPO 或其他改进。那些属于后续 E5 的单一变量。

这里的“正式”是指完成冻结数据和预算上的训练、保存可用 checkpoint；此前 one-batch smoke 只验证训练、保存和加载能运行，不能充当效果基线。不能用 smoke checkpoint 或发布版 SIM-CoT checkpoint 初始化 E3。

## 已冻结的设置

| 项目 | 设置 |
|---|---|
| 基座 | openai-community/gpt2；revision 及权重 SHA 见 assets/manifest.json |
| 实现 | vendored SIM-CoT/CODI/src/model.py，训练损失不改 |
| 训练 | zen-E/GSM8k-Aug 指定版本的原始前 2000 行，索引 0–1999 |
| 验证 | 从索引 2000 向后选 200 个不与训练重复的问题；此次恰为 2000–2199 |
| 测试 | 当前 e1/data 的三个固定 200 题，包括修正后的 SVAMP test 子集 |
| Latent 递推 | 4 次；仅对自训 control 和未来改进组保持相同预算 |
| Epoch | 3，不按测试结果挑 epoch |
| Micro batch / 累积 | 1 / 8；有效 batch 8 |
| 更新 | 每 epoch 250，总 750 次 optimizer update；共 6000 样本前向/反向 |
| 优化器 | AdamW，lr 5e-5，weight decay 0.01，warmup 38 更新后线性衰减 |
| 梯度裁剪 | 最大范数 1.0 |
| LoRA | r=128、alpha=32、dropout=0.1，c_attn/c_proj/c_fc |
| 可训练参数 | 推理器 LoRA、projector、完整辅助 decoder；推理器基座冻结 |
| 精度 | 官方加载路径先以 BF16 载入基座；训练转换为 FP32 master 参数并用 BF16 autocast；正式评测权重转 BF16 |
| 随机种子 | 11；每 epoch 顺序由 seed+epoch 生成 |
| 最终模型 | 固定第 750 次更新的 last.pt；验证只监控，不用测试选模型 |
| 评测 | greedy、batch 1、最多 64 tokens、每集 5 次预热、CUDA 同步计时 |

这是一套小规模课程控制配置，并非声称复现论文的完整训练超参数。发布版 E1/E2 继续保留各自受训步数；不能把发布模型与小规模自训差异归因于某一个改进。

## 4 次递推为什么导出 5 段解释

官方代码先把问题后的隐藏状态投影成 z0，并立即监督它；随后进行 4 次递推，得到 z1、z2、z3、z4，并分别接受解释监督。

- 输入推理器的 latent 向量是 z0–z3，共 4 个，故预算和压缩比分母为 4。
- z4 是第四次递推后的投影输出，接受 decoder 监督，但不再作为下一次 latent 输入。
- 导出标明 slot=0..4 及 consumed_by_reasoner，不能把 5 个解释位置误报为 5 次推理。
- 算式多于 5 步时，官方实现将第五步及之后合并到最后一个监督位置；不足 5 步的位置以 PAD 补齐，不计算解释损失。
- 解码出的文字与参考一致只表明可读出相应信息，不能证明模型因果上按这段文字计算。E4 仍应检查运算、数字、中间结果以及“答对但解释不一致”。

## 已做的数据检查

源文件 SHA 与下载清单一致。训练最大上下文 172 tokens，验证最大 146；无需截断。训练和验证内部及彼此均无规范化后的重复问题。与三个 E1 测试子集没有精确问题重合，但这不排除改写或共享原题。

训练行 67 无算式链，按课程“前 2000 条”保留，解释损失为零，其余损失正常；行 837、1334 的答案有句末句号，原样保留用于训练，评测使用答案解析器规范化。原始算式存在噪声的可能性，本次不按答案正确性人工筛选。

## 服务器运行指南

在 E1/E2 补跑结束后顺序进行；不要同时执行 GPU 任务。将完整 e3/ 上传到 ~/experiment/，沿用已有 assets、smoke.py、e1 和 SIM-CoT。Python 3.12 及现有固定 requirements 不变。

本机 PowerShell 上传：

```powershell
scp -r "C:\Users\83946\Desktop\NLP\AIAA4051_Projects\experiment\e3" NLP:~/experiment/
```

远程准备：

```bash
cd ~/experiment
export PATH="$HOME/experiment/.conda312/bin:$PATH"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
bash e3/run_e3.sh prepare
```

先跑 10 次更新的 pilot，处理 80 个训练样本；它使用正式训练相同的 750 步学习率计划，并保存 checkpoint 后重新加载、导出解释。

```bash
bash e3/run_e3.sh pilot
```

通过条件：没有非有限损失/梯度报错；last.pt 存在；reload_check.json 为 PASS；projector 确有更新；加载前后确定性输出一致。随机起步的 pilot 不要求答对或解释正确，空解释也会记录。

用 pilot 的每次更新耗时估计正式 750 次更新，加上 3 次验证、checkpoint 保存和评测。不要承诺未经实测的 GPU 小时数。建议 E3 先预留不超过约 10 GPU 小时，保留 E5 和修复预算；若预估超出预算，先统一修改两组训练预算再开始正式训练。改配置后必须使用新的初始化路径和新输出目录，不能与旧 pilot 续跑。整组累计低于 50 GPU 小时。

正式训练从 shared_init.pt 重新开始，不接着 pilot 训练：

```bash
bash e3/run_e3.sh train
```

建议在 tmux 中执行，避免关闭终端导致进程中断（若系统已有 tmux，可先运行 tmux new -s e3）。运行中的实例仍会计费。

每 50 更新和每个 epoch 结束保存 last.pt，原子替换避免半写文件。中断后：

```bash
bash e3/run_e3.sh resume
```

恢复包括模型、AdamW 状态、学习率、随机数状态和样本位置；配置或训练源码改变会拒绝混用。日志中超出最后完整 checkpoint 的记录会清理后重算。首次 50 次更新前崩溃可能还没有可恢复 checkpoint，应选择新目录从共享初始化重新开始。此时先修复具体错误，不要直接反复启动。

训练结束后再评测和导出验证集解释：

```bash
bash e3/run_e3.sh evaluate
```

test 评测三集各 200 题；diagnostics 对全部 200 个验证例导出 5 个解码位置及参考目标，不挑正确例。B 可从中按正确/错误分层抽样做 E4。两个输出目录须为空；重复运行使用新的 --output 目录。

## 交给 B 的内容

1. e3/ 全目录：配置、训练/验证清单、manifest、代码和说明。
2. e3_runs/shared_init.pt：共同初始化，包含 LoRA、projector 和 decoder 的起点；不是训练完成的 control。
3. e3_runs/control/：last.pt（含优化器，较大）、contract、初始化 SHA、参数数量、依赖、逐更新日志、会话耗时和 reload_check。
4. e3_runs/control_test/：三组逐题预测与 summary.csv。
5. e3_runs/control_diagnostics/：验证集答案、逐 latent 解码与监督目标。
6. control.log、pilot.log 和实际 GPU 起止时间。

E5 从相同 shared_init 权重开始，保留相同数据、4 次递推、750 更新和评测设置，只改变选定的改进。不能从训练好的 control 再训练 750 次而称为匹配对照。E5 的入口应在 E4 诊断明确后再实施，本代码的初始化配置严格校验不能绕过；届时仅对事先约定的改进项放开检查。

## 验证边界

已在本地用 tiny GPT-2 跑通官方 forward/backward、optimizer 更新、严格保存/加载、五个位置解码及无算式样本，并检查 4 步压缩比分母。结果见 cpu_test_results.json。它验证代码连接，不等于完整 GPT-2/A6000 训练成功。真实 GPU pilot、正式训练和准确率仍需在服务器运行。
