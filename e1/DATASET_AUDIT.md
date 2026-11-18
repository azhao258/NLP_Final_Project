# E1 数据核对与重跑说明（2026-11-15）

## 结论与证据

| 数据 | 旧 E1 来源 | 课程要求及核对结果 | 操作 |
|---|---|---|---|
| GSM8K | OpenAI 官方 test，200/1319 | 课程要求 GSM8K test，符合；文件逐字节保留 | 保留 74/200、37% |
| GSM-Hard | juyoung-trl/gsm-hard，200/1319 | 指定 reasoning-machines/gsm-hard；全部 1319 问题精确匹配，数值答案全部一致 | 已改来源并保留原 200 题；保留 18/200、9% |
| SVAMP | arkilpatel/SVAMP 全量 1000 抽 200 | 指定 ChilleD/SVAMP；1000 道题的 Body、Question、Equation、Type 和数值答案等于其 train/test 并集；旧子集 135 train + 65 test | 改为 test 的 300 题中抽 200，重新评测 |

课程要求固定 SVAMP 子集，但没有明确限定 test split。因此旧结果并非因为数据内容错误而无效，也不能仅据 split 名称断言发生训练泄漏。我们选择 test-only，明确评测边界，便于两人复现。该选择也不能证明发布模型从未见过这些题。

新 SVAMP 使用种子 4051，按课程 test 文件行号抽样后排序，与旧子集重合 44 题，156 题不同。旧准确率 38%、参考长度 10.355、长度比 1.726 和延迟均不能作为新子集指标。为保持一次运行的计时条件，建议完整重跑新的 200 题，而非拼接旧预测。

GSM-Hard 的课程源含 Python 解题 code；它不是自然语言思维链，不执行这些代码，也不把其 token 长度混入自然语言压缩比。N/A 表示没有此处测量的自然语言参考链，不表示完全没有解题信息。为保持历史题目 ID 和答案字符串，准备脚本按问题映射课程源到旧顺序，并逐条断言数值答案相同；source_index 仍是旧来源索引。

## 冻结来源与复现

- GSM-Hard: reasoning-machines/gsm-hard @ 960448f73503112d4226baeb8eb41d3fb5ae2506，gsmhardv2.jsonl。
- SVAMP: ChilleD/SVAMP @ 5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef，data/train-00000-of-00001.parquet 与 data/test-00000-of-00001.parquet。
- 源文件在 audit/，SHA256 在 data/manifest.json；旧评测数据和旧 manifest 在 audit/legacy_data/。
- audit/report.json 由核对脚本生成。旧 e1_results/codi_full 的预测、配置和统计原样保留。

在仓库根目录执行（CPU 即可）：

```bash
python e1/prepare_data.py
python e1/audit_datasets.py
python e1/check_local.py
```

check_local.py 需要已下载的 GPT-2 tokenizer。prepare_data.py 显式固定 JSONL 换行方式，以便 Windows 和 Linux 重建相同哈希。

## 哪些实验需要重跑

- E1（A）：仅重跑 CODI 的 SVAMP 200 题。GSM8K、GSM-Hard 的逐题数据未改，准确率与原始延迟可以保留。
- E2（B）：若已经跑过旧数据，仅重跑 SIM-CoT 的 SVAMP；若尚未运行，直接用当前 600 题。
- E3 对照训练、E5 改进训练：本次只变更评测数据，不要求重新训练。其训练/验证数据仍需另按 GSM8k-Aug 2000/200 冻结，不能由本次 E1 核对推断已合规。
- E4 解码诊断：GSM8K 案例可保留；若用了旧 SVAMP 样本，必须注明历史集合或换为新集合，正式统计统一新清单。
- E6 对照比较及后续报告：任何已完成的 SVAMP 评测均更新；其余数据无需因本次修订重跑。E7 汇总、report、poster 更新来源、划分及结果。

如果未来要精确比较 E1 与 E2 的延迟，但硬件、驱动或环境已变化，应另外统一重测延迟。这与数据修订是否需要重新计算准确率是两件事。

## 远程运行

先上传更新后的整个 e1/ 目录，保留服务器 assets、环境和旧结果。无需重新下载模型或重装环境。在 ~/experiment 运行：

```bash
cd ~/experiment
export PATH="$HOME/experiment/.conda312/bin:$PATH"
python e1/audit_datasets.py
bash e1/run_e1.sh svamp
```

新结果：e1_results/codi_svamp_course_test。回传整个目录和 e1_results/svamp_course_test.log。

B 若已跑完旧 E2，只补 SVAMP：

```bash
python -u e1/evaluate.py --model sim --datasets svamp --output e1_results/sim_svamp_course_test --resume
```

B 若尚未运行 E2：

```bash
python -u e1/evaluate.py --model sim --output e1_results/sim_course_full --resume
```

不要向旧 codi_full/sim_full 目录追加新清单。新配置增加 datasets 字段且 manifest 已更新，断点续跑会拒绝混用。最终表的 GSM8K/GSM-Hard 指向旧结果，SVAMP 指向新结果，并注明各自运行日期。当前只完成 CPU 核对与代码检查，未执行新的 GPU 评测。
