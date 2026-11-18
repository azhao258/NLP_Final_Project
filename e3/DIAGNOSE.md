> 历史运行指南。当前最终基线与交接规则以 [HANDOFF_TO_B.md](HANDOFF_TO_B.md) 和 active_baseline.json 为准。

# E3 只读诊断

diagnose.py 加载现有 control/last.pt，不调用 backward、不创建优化器，不修改 checkpoint 或正式评测。使用固定随机种子 11，从训练集与验证集分别抽取 20 题，保存完整题目清单。不要根据输出好坏重新抽样。

## 运行

沿用已有 Python 3.12、模型资产和环境，只需上传 diagnose.py。服务器执行：

```bash
cd ~/experiment
export PATH="$HOME/experiment/.conda312/bin:$PATH"
python -u e3/diagnose.py --checkpoint e3_runs/control/last.pt --output e3_runs/diagnostic_v1
```

需要支持 BF16 的 CUDA GPU。输出目录必须为空，避免混合不同运行。失败重试时保留旧目录，使用 diagnostic_v2 等新目录。这是小样本诊断，不是正式准确率或延迟测量。

## 三项检查

1. 固定训练 20 题：BF16 权重及 autocast，自由生成答案和全部五个解释位置，记录准确数、答案集中程度和各位置解释的种类。
2. 同批训练题：只对非 PAD 的有效参考位置计算 teacher-forcing token 准确率与交叉熵（包含 EOS）。正常 latent 与循环移位后另一道题同位置的 latent 对比；正确文字前缀、目标和 decoder 都不变。正的 swapped_minus_normal_nll 表示正常 latent 更好。保存首 token 指标，避免只看正确前缀带来的高准确率。自由生成严格 token 一致率不等于数学等价率。
3. 固定验证 20 题：分别比较纯 FP32、FP32 权重加 BF16 autocast、BF16 权重加 BF16 autocast。后两者对应原训练期间验证和正式独立评测的约定；每次均从原 checkpoint 重新加载，避免 BF16 舍入后再转 FP32 影响对照。禁用 TF32。

teacher-forcing 给定真实前缀，而自由生成没有，因此二者并非同一难度，不能直接把高 token 准确率解释为真实推理能力。换 latent 后影响小支持“不敏感”的判断，但单次移位、20 题不能证明完全忽略 latent，也不能区分 latent 本身缺信息与 decoder 不会使用信息。

## 输出

- samples.json：可复现的两组固定题目。
- config.json：checkpoint 哈希、数据和代码哈希、精度条件及硬件。
- precision.jsonl：三种精度下同一批验证题的答案。
- train_free.jsonl：训练题答案、参考链、五个解释位置。
- teacher_forcing.jsonl：有效位置的正常/错配 latent 评分和自由生成一致性。
- summary.json：汇总，优先查看该文件。

已有数据清单可能因 JSON 键顺序不同而与训练记录哈希不同；脚本记录该事实并验证训练/验证文件各自哈希，不修改清单。若实际题目内容改变，应停止使用这份结果并核对来源。

完成后回传整个 diagnostic_v1。先结合三项检查判断是否需要代码修复或小样本过拟合测试，再决定正式 control 的重训配置；不要据 20 题直接挑超参数或宣称改进有效。
