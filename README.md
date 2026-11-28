> 2026-11-28 目录调整：测试比较见 [E6](E6/README.md)，步骤/检索/案例见 [E7](E7/README.md)。旧运行记录中的路径是生成时路径，迁移不代表重训或重算分数。

# NLP Final Project — A/B 实验交接

AIAA4051 Project 11 · Compressing Chain-of-Thought via Interpretable Latent Reasoning

**当前最终自训基线：control_v1。** 代码、冻结数据、逐题预测和日志在仓库；完整checkpoint在[GitHub Release](https://github.com/azhao258/NLP_Final_Project/releases/tag/e3-handoff-2026-11-18)，通过分片校验恢复。

B 请首先阅读 **[E3 交接说明](e3/HANDOFF_TO_B.md)**。权威配置选择记录：[active_baseline.json](e3/active_baseline.json)、[frozen_v1.json](e3/frozen_v1.json)。v2冻结记录是历史材料，已被当前选择取代。

## E3 版本与结果

三组均从相同GPT-2共同初始化开始，2000条训练、200条验证、4次latent、3epochs、750更新；下表验证使用统一独立BF16评测。

| 模型 | 改动 | 验证准确率 | 有效解释位置严格匹配 | GSM8K测试 | GSM-Hard测试 | SVAMP测试 |
|---|---|---:|---:|---:|---:|---:|
| **v1：当前基线** | lr5e-5，全部可训练参数统一wd0.01 | 3.0% | 6/478 | 3.5% | 0.5% | 1.0% |
| v2 | 仅lr改为1e-4 | 3.5% | 2/478 | 2.5% | 0% | 0.5% |
| v3 | 仅按官方Trainer规则排除bias/LayerNorm等的wd | 2.5% | 5/478 | 未运行 | 未运行 | 未运行 |

**选择限制：** 最初按验证准确率选v2，查看其测试结果后改选v1。这一过程已记录，不能声称测试集完全未参与选择。保留三版全部结果，不宣称少数题差异稳定，也不再根据测试结果切换基线。低绝对准确率是报告必须讨论的局限。

## 发布模型 E1/E2

| 发布模型 | GSM8K（200） | GSM-Hard（200） | 新SVAMP test子集（200） | Latent步数 |
|---|---:|---:|---:|---:|
| CODI | 37.0% | 9.0% | 44.0% | 6 |
| SIM-CoT | 33.5% | 9.0% | 42.5% | 6 |

GSM8K/GSM-Hard保留原codi_full/sim_full结果；SVAMP使用codi_svamp_course_test/sim_svamp_course_test。旧SVAMP混合划分结果只归档。数据核对见[e1/DATASET_AUDIT.md](e1/DATASET_AUDIT.md)。发布模型与自训模型训练规模不同，不作为单变量方法对比。

B原E2提交保留在[e2](e2)及[e1_results/sim_full](e1_results/sim_full)。其旧对比CSV含历史SVAMP，不能直接用作当前表格。

## 快速使用

Python3.12、A6000及可用NVIDIA驱动；依赖固定：

```bash
python -m pip install --prefer-binary -r SIM-CoT/CODI/requirements.txt
python download.py
python verify_assets.py
python e3/download_checkpoints.py --names shared_init control
```

已有assets可直接复制校验，无需重复下载。不要重新抽样或修改冻结数据。恢复全部诊断/变体checkpoint可用下载器的--all。Release分片只用于分发，不应直接torch.load；下载器会拼接、解压并检查原文件SHA256。

复测v1验证例：

```bash
python e3/evaluate.py --checkpoint e3_runs/control/last.pt --split validation --explain --output e3_runs/v1_validation_recheck
```

结果目录必须独立，不能覆盖归档。E5必须从shared_init.pt开始，使用v1公共配置，不能从已训练control继续训练。

## 目录

- [e3/HANDOFF_TO_B.md](e3/HANDOFF_TO_B.md)：最终交接、各实验含义、结果与E4/E5规则。
- [e3](e3)：训练/评测/诊断代码和2000/200固定数据。
- [e3_runs](e3_runs)：全部已返回文本结果，包括v1/v2/v3、pilot、精度与错配latent诊断、8题过拟合。
- [audit_official/FINDINGS.md](audit_official/FINDINGS.md)：官方固定提交核对，训练与解码一致性、数据与优化配置差异。
- [e1](e1)：发布模型共用评测和600条固定测试题。
- [e1_results](e1_results)：E1/E2历史结果及新版SVAMP补跑。
- [e2](e2)：B原E2交接与分析，原样保留。

## 复现与报告要求

总GPU预算低于50小时，代码与报告截止2026-12-14。会话日志不等于完整租用时长。E4须区分算术运算、操作数、中间结果及答对解释错；PAD位置不当作应监督的步骤。4次递推对应5个解释位置，二者不能混报。严格文本匹配不等同数学等价，解码一致不证明因果忠实性。

官方模型代码来自SIM-CoT提交cd3291dac9fe1e495da9fe00e61a9b96a884b2ed，许可保存在SIM-CoT/LICENSE；数据与模型遵循各自来源许可。课程规模适配没有声称复现论文完整训练规模。
