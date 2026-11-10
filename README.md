# NLP Final Project — A/B 实验交接

AIAA4051 Project 11 · Compressing Chain-of-Thought via Interpretable Latent Reasoning

本仓库保存 A 已完成的 E1，并提供 B 运行 E2 的相同数据与代码。课程代码和报告截止日期为 2026-12-14。

## 已完成的 E1

单张 RTX A6000，BF16，batch size 1，greedy，6 latent steps，最多生成 64 tokens。三个集合各 200 题，600 条答案均成功解析，无生成长度截断。

| 数据集 | 正确数 | EM | 参考类型 | 平均参考 tokens | 参考长度/6 | 延迟中位数 | P95 |
|---|---:|---:|---|---:|---:|---:|---:|
| GSM8K | 74/200 | 37.0% | 文字链 | 98.855 | 16.476 | 237.54 ms | 270.76 ms |
| GSM-Hard | 18/200 | 9.0% | 缺失 | N/A | N/A | 273.95 ms | 299.35 ms |
| SVAMP | 76/200 | 38.0% | 方程 | 10.355 | 1.726 | 239.06 ms | 257.40 ms |

原始结果在 [e1_results/codi_full](e1_results/codi_full)。长度比取决于参考文本类型，不能直接推算加速倍数。
GSM-Hard 的所用数据版本仅包含题目和答案，因此参考推理长度及长度比记为 N/A。所有数据集均使用 6 个 latent steps。

E2 待 B 运行。E3 正式对照组待 A 训练。smoke.py 中保留了早期单 batch 验证逻辑，正式训练入口将在 E3 准备。

## B 的比较规则

1. 直接使用 e1/data 中已冻结的 600 题及其 ID，抽样种子为 4051。保持 A/B 测试题一致。
2. 使用同一 evaluate.py、metrics.py 和依赖版本。保持 greedy、BF16、batch=1、6 latent steps、答案上限 64、运行种子 11、每集 5 次预热。
3. 发布模型的 6 步配置用于 E1/E2。后续对照组和改进组的自训预算另行冻结。
4. 延迟比较使用相同型号的 RTX A6000，每次只运行一个 GPU 作业。其他硬件结果单独标注。
5. 评分规则或推理程序修订后，同时复核两个模型的结果。保留解析失败、错误答案和截断输出。
6. B 输出写入 e1_results/sim_full。勿同时启动两份写入同一目录的任务。
7. GPU 使用起止时间由 A/B 共同登记，总预算低于 50 GPU 小时。

## 安装与资产准备

Git 仓库包含源码、固定评测数据和结果。大模型文件由固定 revision 下载。推荐 Python 3.12 和带 NVIDIA 驱动的 A6000 镜像。

```bash
git clone https://github.com/azhao258/NLP_Final_Project.git
cd NLP_Final_Project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --prefer-binary -r SIM-CoT/CODI/requirements.txt
```

若使用 conda，可用以下命令代替 venv 创建步骤，再安装相同依赖。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y --override-channels -c conda-forge -p "$PWD/.conda312" python=3.12 pip
conda activate "$PWD/.conda312"
```

检查 GPU 并准备文件。Linux 下载需要 curl，Windows 下载需要 PowerShell 7（pwsh）。权重与原始训练数据总下载量约 1.8 GB。依赖安装需要联网，正式评测使用离线模式。

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())"
python download.py
python verify_assets.py
python e1/check_local.py
```

若已收到原 experiment/assets 全部文件，可直接复制到本仓库 assets/，然后校验。verify_assets.py 检查完整资产清单，因此请保留全部文件。

## B 运行 E2

先每个数据集运行 5 题。

```bash
python -u e1/evaluate.py --model sim --limit 5 --output e1_results/sim_pilot --resume
```

确认 COMPLETE、三个数据集各 n=5，检查异常及解析失败，再运行正式评测。

```bash
python -u e1/evaluate.py --model sim --output e1_results/sim_full --resume
```

逐题保存结果。中断后在相同配置下重新执行即可续跑，并重新预热。代码、GPU 或设置改变时使用新目录。若驱动或后台负载变化可能影响延迟，应重新测量正式延迟。

e1/run_e1.sh 固定运行 CODI。仓库包含历史 codi_full，重新复现 E1 请用独立目录。

```bash
python -u e1/evaluate.py --model codi --output e1_results/codi_reproduction --resume
```

## B 交回的结果

交付整个 sim_full 文件夹、终端日志、异常说明及 GPU 起止时间。

- 三个 predictions.jsonl，每个 200 条，题目 ID 与清单一致。
- summary.csv，EM 为 0 到 1 的小数，报告百分数时乘 100。
- config.json、environment.txt、model.json、sessions.jsonl。

B 将 E1/E2 合成基线表。按 dataset 与 id 对齐逐题预测，统计两者都对、CODI 错而 SIM-CoT 对、CODI 对而 SIM-CoT 错、两者都错。

## 文件作用

| 文件 | 作用 |
|---|---|
| e1/data/*.jsonl 和 manifest.json | 固定测试题、来源、ID 和哈希 |
| e1/raw/ 和 prepare_data.py | 原始数据、元数据及可重建抽样脚本 |
| e1/evaluate.py | 模型推理、计时、断点续跑、逐题保存 |
| e1/metrics.py | 数值答案解析、准确率、长度及延迟统计 |
| e1/check_local.py | 解析规则、600 条 gold、ID 与上下文检查 |
| smoke.py | 模型结构构建、LoRA 配置和发布权重加载 |
| SIM-CoT/CODI/src/model.py | 官方 CODI＋SIM-CoT 模型实现 |
| SIM-CoT/CODI/requirements.txt | 固定依赖版本 |
| download.py / verify_assets.py | 固定版本资产下载及校验 |
| assets/manifest.json | 下载来源、revision、大小及哈希 |
| e1_results/codi_full/ | A 的正式 E1 结果 |
| source_revisions.txt | SIM-CoT、CODI 源码提交号，按此顺序 |

保留目录结构。evaluate.py 会导入 smoke.py，后者再导入官方模型文件。SIM-CoT checkpoint 包含辅助 decoder，回答路径和总参数量需区别记录。

## 数据与计时定义

- GSM8K 使用官方 test.jsonl，1319 题抽 200，参考链为 #### 前原文，含算式标记。
- GSM-Hard 使用 juyoung-trl/gsm-hard default/train 的 1319 条抽 200，仅用于本项目评测。缺少参考链，长度比为 N/A。
- SVAMP 使用官方完整 1000 题挑战集抽 200，该原始文件无独立 test split。参考类型标为 equation。
- 每集独立使用种子 4051 抽样，再按原始行号排序。GSM-Hard 源于 GSM8K，三个集合分别报告，避免将 600 题视为独立同分布样本合并推断。
- 计时前输入已 tokenize 并放到 GPU。CUDA 同步计时覆盖 prompt、latent 和答案生成。文本解码、文件保存和模型加载在计时区间外。
- answer_tokens 排除 EOS，generated_tokens 包含生成的 EOS。长度比为参考 tokens 总和除以覆盖样本的 latent 步数总和。
- sessions.jsonl 记录成功结束会话的加载后评测时间。完整 GPU 工时另行登记。

## 下一次交接

E3 完成后，A 交正式对照组、训练前共同初始化、2000/200 训练验证清单、训练代码配置、日志和 GPU 耗时。B 使用对照组开展 E4，再从共同初始化完成 E5 改进训练。当前仓库尚未冻结 E3 训练验证划分。

## 来源与许可

- [CODI 官方代码](https://github.com/zhenyi4/codi)
- [SIM-CoT 官方代码](https://github.com/InternLM/SIM-CoT)
- [CODI 权重](https://huggingface.co/zen-E/CODI-gpt2)
- [SIM-CoT 权重](https://huggingface.co/internlm/SIM_COT-GPT2-CODI)
- [GSM8K](https://github.com/openai/grade-school-math)
- [GSM-Hard](https://huggingface.co/datasets/juyoung-trl/gsm-hard)
- [SVAMP](https://github.com/arkilpatel/SVAMP)

官方模型代码和依赖来自 SIM-CoT 提交 cd3291dac9fe1e495da9fe00e61a9b96a884b2ed，Apache-2.0 许可保留在 SIM-CoT/LICENSE。数据与模型使用遵循原始来源条款。
