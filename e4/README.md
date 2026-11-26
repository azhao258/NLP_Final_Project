# E4：v1 解释诊断

在仓库根目录执行（不需要 GPU、torch 或模型权重；复用 E1 评分器，需要项目依赖中的 numpy）：

```powershell
python e2/summarize_baselines.py
python e4/analyze.py
python -m unittest discover -s e4 -p "test_*.py" -v
```

`--output <目录>` 可另存评分结果；默认重新生成 `e4/results/` 中的 CSV/JSON/JSONL 统计文件及来源记录。脚本只读实验输入，不训练、不下载权重、不修改冻结数据，**不会创建或覆盖 REPORT.md**。

`REPORT.md` 保留为最终报告，可直接编辑；案例统一保存在 `CASE_REVIEW.md`，修改后需同步报告中的人工案例部分。

## 交付内容

| 文件 | 用途 |
|---|---|
| `analyze.py` | ID/哈希核查、算式解析、重叠标签及统计文件导出；不生成报告 |
| `results/questions.csv` / `.jsonl` | 全部 200 题的预测、参考、5 个解码位置、标签 |
| `results/steps.csv` / `.jsonl` | 全部 1,000 个位置，含 478 个有效位置、522 个 PAD；保存解析证据 |
| `results/summary.json` | 完整机器可读汇总 |
| `results/slots.csv` | 5 个位置分别统计，多样性以全部 200 题为分母 |
| `results/categories.csv` | 重叠错误分类；题级专有标签的位置数留空 |
| `results/provenance.json` | 输入与分析脚本 SHA256、checkpoint 身份、清单不一致说明 |
| `REPORT.md` | 保留的最终英文报告，包含可复算统计及案例分析；单独维护 |
| `results/compression_comparison.csv` | 三模型×三测试集的完整压缩、准确率与延迟比较 |
| `CASE_REVIEW.md` | 英文人工案例复核，包含案例解释、取整与歧义检查 |

CSV 使用 UTF-8 BOM，方便 Excel 打开。列表字段为 JSON；JSONL 保留布尔和空值类型。自动标签覆盖全部 200 题，但不声称对所有题目做了独立人工语义标注。

当前 REPORT.md 是完整英文 Part 2 报告：包含跨基线压缩比较、逐步骤统计和 CASE_REVIEW.md 中的案例解读。前者使用固定测试集，后者使用独立于训练的验证集；不混合两个统计总体。测试长度比不是加速倍数，发布模型与4步control的比较不是只改变latent步数的因果实验。案例文字的编辑不改变评分规则和数值结果。

## 标签定义

错误类别相对当前参考监督定义，允许多标签。

| 标签 | 中文含义与判据 |
|---|---|
| `missing_output` | 有参考但输出为空 |
| `unparseable_output` | 非空输出不在数字四则运算解析范围内 |
| `operator_mismatch` | 生成与参考的运算符多重集不同 |
| `operand_mismatch` | 左侧数字多重集不同，数字用精确分数规范化 |
| `intermediate_result_mismatch` | 等号右侧数值不同；含最终参考步骤，因此不全是中间步 |
| `arithmetic_inconsistency` | 生成式左右精确数值不等 |
| `reference_arithmetic_inconsistency` | 参考式左右精确数值不等；可能是近似取整，不直接等于坏标签 |
| `reference_step_not_reproduced` | 已解析但未通过限定算式兼容检查，不代表内部步骤缺失 |
| `fewer_equations_than_reference` | 同一监督位置生成算式条数少于参考条数 |
| `frequent_template_on_valid_slot` | 题级：有效位置用了该位置在 >=100/200 题出现的文本；不是错误真值 |
| `correct_answer_explanation_mismatch` | 题级：答案正确，但未做到全部有效位置限定兼容 |

`conservative_equivalent` 要求算式自身成立、算式条数相同、左侧操作数和运算符多重集相同、计算值与右侧结果相同。它可以识别本次出现的乘法交换顺序，不是一般数学等价或语义忠实性判定。PAD 的匹配字段为 null，不能按 false 计入错误。

参考链超过 5 条算式时，官方 `get_steps` 将第 5 条及以后合并到 slot 4。本脚本复核这一规则，避免误判缺失参考。两道题存在这种合并。


