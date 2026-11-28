# E7：步骤解释与表示对应性

| 路径 | 用途 |
|---|---|
| `scoring.py` | 共用步骤评分、汇总与配对分析实现；仍沿用E4算术/匹配规则 |
| `results/steps.*` | 七模型21,000个槽位评分 |
| `results/categories.*`、`slots.*` | 重叠错误类型及逐槽模板分布 |
| `results/CASE_REVIEW.txt`、`case_review.*` | 34个配对案例的AI初审与人工确认栏 |
| `results/case_coverage.json` | 六种改进的案例覆盖 |
| `results/retrieval_comparison.csv`、`reference_path.csv` | v1/P3/P4/P5验证集诊断汇总 |
| `probe_inputs/` | 四模型检索诊断原始JSON/JSONL，复算输入 |
| `p5_retrieval/` | 从补充实验原位置迁入的P5检索返回结果 |
| `check_reference_retrieval.py` | GPU参考路径与检索诊断脚本 |
| `add_diagnostics.py`、`review_cases.py` | CPU诊断汇总与案例初审生成 |

测试导出的逐题解码文件与对应答案共同保留在 [E6/exports](../E6/exports/)，验证导出继续保留在训练实验目录，不拆开配对输入。全部CPU复算命令见 [E6 README](../E6/README.md)。

已有checkpoint需要补跑检索时，在仓库根目录使用：

```bash
python E7/check_reference_retrieval.py --checkpoint PATH_TO_LAST_PT --n 200 --output NEW_DIAGNOSTIC_DIRECTORY
```

人工审核尚待A/B完成：请先阅读CASE_REVIEW，再填写CSV中的 `human_reviewer`、`human_agrees`、`human_notes`。禁止对已签字目录重跑review_cases，它会重建空白签字栏；复算使用新目录。

解释指标边界：四次latent递推、五个解码槽，末槽readout-only；GSM-Hard无参考链，SVAMP只有整式兼容。算术成立或检索命中不等于因果忠实性。案例是目的性选例、AI初审，不是人类盲审。

P5跨题检索仅比v1多1个命中，测试解释仍高度模板化。保留官方gold与原分数，同时披露速度题1051和参考标记不完整案例，未据此删除题目。
