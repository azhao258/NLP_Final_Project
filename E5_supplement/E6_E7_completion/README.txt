E6/E7 补齐交付（2026-11-28）

范围
七个自训模型：v1、prefix_mask、P1–P5。发布版CODI/SIM-CoT仍使用原E1/E2/E4结果，
不得把其K=6和自训K=4当成只改变latent预算的受控比较。
本包完成已有输出的CPU离线分析、AI案例初审；不含新训练，不含模型权重。
未自动上传GitHub。没有覆盖B的原分析文件。

优先阅读
results/REPORT.txt：E6/E7完成项、七模型结论、P5分题型分析及局限。
results/CASE_REVIEW.txt：34个配对案例；含题干、参考、双方答案/解释、独立核算与AI意见。
results/case_review.csv：同一批案例，供A/B填写human_reviewer、human_agrees、human_notes。
results/RETRIEVAL_NOTES.txt：验证集专项检索诊断的解释边界。

结果文件
summary.csv/jsonl：21行，答案准确率、K、答案长度、参考长度比、历史延迟、解释指标。
paired_questions.csv/jsonl：3600条（六种改进×600题），包括答案和步骤得失。
problem_types.csv/jsonl：204行，使用B的冻结题型，不按预测重新划分；含小组标记。
steps.csv/jsonl：21000个解码槽位，含参考可用性、算术有效性和可重叠错误标签。
slots.csv/jsonl：105行，逐模型/数据集/槽位模板频率。
categories.csv/jsonl：189行，错误分类；无参考时为N/A而非零。
retrieval_comparison.csv：v1/P3/P4/P5验证集检索，包括消耗槽和读出槽。
reference_path.csv：gold-conditioned参考路径NLL/token准确率；不是自由答案准确率。
case_coverage.json：六种改进案例覆盖；prefix_mask无步骤兼容提升候选，未虚构。
provenance.json、diagnostic_provenance.json：输入、代码、设置与哈希来源。
complete.json：统计完成状态与输出校验。人工确认仍未完成。
p5_outputs/：本次P5返回的文本/JSON原始输出，保留pilot与formal区别，不含权重。
probe_inputs/：四模型检索原始JSON/JSONL，不含大型表示张量。

复现（不需要GPU；需要原项目及numpy）
把E6_E7_completion目录放在项目根目录，保留原e1/e3/e4/e5和B导出的六模型文件。
在项目根目录运行，新输出路径必须不存在：

python E6_E7_completion/build_analysis.py --repo . --p5 E6_E7_completion/p5_outputs --output E6_E7_completion/results_rerun
python E6_E7_completion/add_diagnostics.py --probes E6_E7_completion/probe_inputs --output E6_E7_completion/results_rerun
python E6_E7_completion/finalize_review.py --repo . --p5 E6_E7_completion/p5_outputs --results E6_E7_completion/results_rerun

第一步核对固定题目、答案解析、推理设置和旧六模型所有汇总字段；若旧结果不同会停止。
第三步生成AI意见与空白人工栏；只对新目录运行，不覆盖已经签字的case_review.csv。
复现依赖现有项目的e5/test_suite_analysis_20261126/summary.jsonl作为回归基准。
权重身份来自评测配置，本分析不独立重新读取或哈希权重。

还需要人完成什么
1. A/B阅读34个案例、确认/修改AI初审意见，在CSV填姓名、意见和日期（可放human_notes）。
   尤其核对速度题gsm-hard-train-1051的措辞歧义，以及52/473题参考标记不完整。
2. 将本包结论与原E1–E5合并进最终report/poster。不可写成已由人类完成盲审。
3. P5最终last.pt另行备份；如要发布权重，核验文件哈希并按既有发布流程处理。
4. 项目总GPU预算另按全部训练/诊断/评测会话和租赁记录核对，未在此伪造总耗时。

不需要为了补齐当前E6/E7重训。没有跨七模型同场重复计时，因此不能声称加速；
如果未来要作这种主张，才需额外基准测试。当前报告保留历史延迟且明确其边界。
