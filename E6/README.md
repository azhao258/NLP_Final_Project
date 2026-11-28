# E6：答案、压缩与效率比较

此目录集中存放 v1、prefix_mask、P1–P5 的固定测试导出与统一比较。训练代码和训练日志仍在 `e5/` 与 `E5_supplement/P5/`；E7 的步骤分析见 [E7](../E7/README.md)。

| 路径 | 用途 |
|---|---|
| `exports/<模型>/` | 每模型600题原始测试预测、配套解码、配置与原汇总。解码作为配对输入与预测一起保留，评分结果归E7 |
| `results/summary.csv` | 七模型准确率、长度比、历史延迟及共享步骤汇总，共21行 |
| `results/paired_questions.csv` | 六种改进对v1的3600条配对，包含答案与解释变化 |
| `results/problem_types.csv` | 204行冻结题型分析、小组标记及得失 |
| `results/REPORT.txt` | 最近的E6/E7联合结论 |
| `legacy_six_model_analysis/` | B此前六模型完整历史分析，原样保留作为回归基准，不拆解历史审计包 |
| `SIX_MODEL_LATENCY*` | 六模型历史计时说明；P5计时已纳入新summary |
| `archive/` | 旧交付输入快照及历史清单，保留原始字节，不作为新运行入口 |
| `MIGRATION_MAP.json` | 旧路径→新路径和迁移前文件哈希 |
| `CURRENT_MANIFEST.json` | 当前E6/E7文件哈希；跨目录检查使用本清单 |

## CPU复算

从仓库根目录运行，临时输出目录必须不存在：

```bash
python E6/build_analysis.py --repo . --p5 E5_supplement/P5 --output analysis_new
python E7/add_diagnostics.py --probes E7/probe_inputs --output analysis_new
python E7/review_cases.py --repo . --p5 E5_supplement/P5 --results analysis_new
python E6/split_results.py --input analysis_new --output-root analysis_split
```

最后一步得到 `analysis_split/E6/results` 和 `analysis_split/E7/results`，不覆盖已交付结果或人工签字。该流程使用既有numpy环境，不加载权重、不运行GPU。

`analyze_test_suite.py` 是历史六模型归档入口，仍需要它原来要求的服务器备份归档；正常七模型复算使用 `build_analysis.py`。

冻结题型和共用训练审计依赖仍保留在 `e5/problem_types.py/json`、`e5/protocol.py`，避免改动原训练契约；`e5/analyze.py` 和 `e5/analyze_test_suite.py` 仅作为旧导入/命令的兼容转发。

统计事实保持不变：v1三个测试正确数7/1/2，P5为6/1/2，K均为4。历史延迟不是七模型同场基准，不声称加速。单seed、v1曾使用测试结果选型；题型各轴有重叠。

历史 `complete.json`/`provenance.json` 记录的是生成时路径与代码哈希，未改写为新实验。最近联合结果拆分后，E6旧清单中的E7文件应在 `../E7/results` 查找；迁移后的整体完整性以 `CURRENT_MANIFEST.json` 为准。
