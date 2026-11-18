> 历史运行指南。当前最终基线与交接规则以 [HANDOFF_TO_B.md](HANDOFF_TO_B.md) 和 active_baseline.json 为准。

# 8 题过拟合调试：先验证接口，再判断模型能否学习固定题

本脚本是独立调试，不替代 E3 正式 control。不会修改 shared_init.pt、control/last.pt、正式配置或数据。仅将新的日志和 checkpoint 写入独立输出目录。已通过本地 tiny GPT-2 的 CPU 集成测试，真实 GPT-2/A6000 的 audit 和训练仍待运行。

## 固定数据与预算

使用原训练集索引 [2,3,18,7,9,11,23,36]，由代码写死并核对 source_index，不读取测试集挑样本。样本都来自课程要求的前 2000 条，算式和最终答案通过精确分数运算检查。

| 索引 | 题目概要 | 步数 | 答案 |
|---|---|---:|---:|
| 2 | 20 名学生各有 5 支铅笔 | 1 | 100 |
| 3 | 10 个礼篮各需 3 瓶酒 | 1 | 30 |
| 18 | 84 块饼干平分给 6 人 | 1 | 14 |
| 7 | 30 块糖给三人各 8 块，求剩余 | 2 | 6 |
| 9 | 15 只鸡，每只每周 5 个蛋，求三周总数 | 2 | 225 |
| 11 | 8 人每人需 1/4 千克，每千克 5 元 | 2 | 10 |
| 23 | 18 包每包 12 张，送出 6 包，求剩余打数 | 3 | 12 |
| 36 | 8 个苹果，连续两天每天吃 2 个，再买 5 个 | 3 | 9 |

每次更新包含全部 8 题：micro batch=1，累积 8 次；固定种子下每次重排顺序。最多 200 次 optimizer update，即最多 1600 次样本前向/反向，每题最多重复 200 次。这种重复次数专用于调试，不是满足正式“最多 3 epochs”要求的实验；不能把调试模型作为正式初始化或 control。调试 GPU 时间也计入小组预算。

损失、LoRA、dropout、优化器、梯度裁剪等直接使用 shared_init.pt 的配置，并与 e3/config.json 比对。lr=5e-5、warmup=38、线性衰减继续使用原 750 次更新计划，只执行前 200 步，不压缩成 200 步衰减，也不提高学习率。此测试失败不等于证明代码有 bug，仍可能是优化配置或预算不足。

## 检查内容

- 核对 encoder、student、reference 输入与标签；验证 student/reference 用于预测首个数值 token 的状态位置；输出可读文本供复核。
- 独立重算返回 logits 的答案交叉熵，与官方 forward 返回值比对。
- 按 4 次 latent 递推、5 个解码位置的官方约定核对目标；15 个有效算式，其余 PAD 不计解释准确率。
- 反向传播后分别统计 LoRA、projector、decoder 的梯度数量、非零数量和范数；任何组无有效梯度即停止。冻结参数不能有梯度。LoRA 零 B 初始化时，个别 A 的梯度为零是允许的。
- 定期检查每组代表性参数相对初始权重确实发生改变。
- 第 0、20、40、…、200 次更新导出全部 8 题的自由答案、全部解码位置和有效目标的 teacher-forcing 指标。

这些检查验证当前代码的接口一致性，不能单凭 PASS 证明整个方法与论文训练设置完全一致。梯度范数是裁剪前数值，超过 1 不代表裁剪失败。

## 上传与运行

本机 PowerShell：

```powershell
scp "C:\Users\83946\Desktop\NLP\AIAA4051_Projects\experiment\e3\overfit8.py" NLP:~/experiment/e3/
```

依赖服务器已有 common.py、train.py、diagnose.py、原模型代码、assets、e3/data、shared_init.pt。无需修改它们或重新安装环境。

远程只做检查，不更新参数：

```bash
cd ~/experiment
export PATH="$HOME/experiment/.conda312/bin:$PATH"
python -u e3/overfit8.py --init e3_runs/shared_init.pt --output e3_runs/overfit8_audit --audit-only
```

看到 PASS 且 status.json 为 AUDIT_PASS 后，再运行独立的 200 步调试：

```bash
python -u e3/overfit8.py --init e3_runs/shared_init.pt --output e3_runs/overfit8_v1
```

检查模式本身不做 optimizer.step；训练模式会再检查一次，然后恢复初始随机数状态，从共享初始化训练。不能传入已训练的 control/last.pt 或 pilot/last.pt。

每 20 步保存可恢复的 last.pt，使用原子替换。中断且已有完整 checkpoint 时：

```bash
python -u e3/overfit8.py --init e3_runs/shared_init.pt --output e3_runs/overfit8_v1 --resume
```

恢复校验代码、数据、初始化和配置。若第一次保存之前中断，使用新输出目录从初始化重来。没有自动扩展更新预算或提高学习率；到 200 步即停止，即便尚未学会。运行结束的 COMPLETE 表示执行完成，不表示已经过拟合成功。

## 回传与判断

本机下载两个完整目录：

```powershell
$dest = "C:\Users\83946\Desktop\NLP\AIAA4051_Projects\experiment\returned_results\e3_runs"
scp -r NLP:~/experiment/e3_runs/overfit8_audit "$dest"
scp -r NLP:~/experiment/e3_runs/overfit8_v1 "$dest"
```

last.pt 含优化器，约数 GB；如只先分析，可先下载 JSON/JSONL 文件，checkpoint 留在服务器并另行备份。

- interface_audit.json：标签、答案位置与算式监督。
- gradient_audit.json：更新前各组梯度。
- parameter_changes.json：最近评测点的代表参数变化。
- evaluations.jsonl：历次答案正确数、自由解码一致性、teacher-forcing 指标。
- predictions.jsonl：逐题逐阶段的详细输出。
- train_log.jsonl：训练损失、学习率、梯度。
- summary.json：最后一次评测。
- samples.json、contract.json、environment.txt、sessions.jsonl：复现与时间记录。

若能稳定答对大部分甚至全部 8 题并解码相应算式，说明现有流程具备基本学习能力，再分析完整训练配置。若答案能学会而解释仍固定，聚焦 decoder/latent 信息传递。若答案也学不会，继续定位实现或优化问题，暂不盲目重跑完整 control。不要以一次 8/8 推断泛化能力；先看多次检查趋势和实际输出。

评估保持 FP32 master 权重＋BF16 autocast，不在训练中反复将参数转 BF16，避免诊断本身改变待训练权重。
