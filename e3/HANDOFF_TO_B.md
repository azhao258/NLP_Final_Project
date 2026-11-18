# A → B：E3 完整交接（2026-11-18）

**当前最终基线是 control_v1，即 e3_runs/control/last.pt。不是 v2，也不是8题调试模型。**

权威选择记录为 active_baseline.json 和 frozen_v1.json。frozen_v2.json、evaluate_frozen_v2.py 仅保留为历史追溯，不再代表当前选择。

## 三个 control 分别做了什么

三组共享初始权重、2000/200训练验证数据、seed11、样本顺序、4次latent递推、3epochs、750更新、micro batch1×累积8、warmup38、线性调度、clip1和四项损失。都从原shared_init.pt开始，没有继续训练已完成的control。

| 版本 | 学习率 | 优化器分组 | 实验目的 |
|---|---:|---|---|
| v1：control/ | 5e-5 | 所有可训练参数weight decay=0.01 | 原课程规模对照组，现选定基线 |
| v2：control_v2/ | 1e-4 | 同v1 | 单独检查提高学习率 |
| v3：control_v3/ | 5e-5 | 按Trainer规则排除bias、LayerNorm等的weight decay，其余0.01 | 单独检查优化器分组 |

v3的config字典与v1一致，区别记录在variant、optimizer_groups.json和训练入口中。v2只有一个衰减组，共250个参数张量；v3有148个衰减、102个不衰减参数张量。

## 结果

下表使用统一独立评测精度：BF16权重+BF16 autocast。不能混入训练日志中FP32 master权重的验证结果。

| 模型 | 验证准确率 | 验证解释位置严格匹配 | GSM8K测试 | GSM-Hard测试 | SVAMP测试 |
|---|---:|---:|---:|---:|---:|
| v1（当前基线） | 6/200，3.0% | 6/478，1.26% | 7/200，3.5% | 1/200，0.5% | 2/200，1.0% |
| v2 | 7/200，3.5% | 2/478，0.42% | 5/200，2.5% | 0/200，0% | 1/200，0.5% |
| v3 | 5/200，2.5% | 5/478，1.05% | 未运行 | 未运行 | 未运行 |

解释统计排除PAD，去除空白和EOS文本后比较算式，是严格文本一致性，不等同数学等价性。v1/v2/v3答对的6/7/5道验证题均没有做到所有有效解释位置一致。v2每个位置在200题上输出固定文本：位置0为3*12=36，位置1–4为3*2=6。

v1测试长度比：GSM8K=98.855/4=24.71375；SVAMP算式=10.855/4=2.71375；GSM-Hard无所用自然语言参考链，N/A。不可将长度比当作速度提升或成功保留推理能力。v1测试GSM-Hard有1条解析失败，其余两集0；三个集合均无生成截断。

## 基线选择的透明说明

最初依据验证准确率选出并冻结v2；看到v2测试结果后，A决定保留v1。已明确记录 selection_used_test_results=true。不能写成“始终仅按验证集选v1”，也不能把现有测试子集称为完全未参与选型的确认性证据。三版结果均保留，不隐去不利结果。几道题差异不证明稳定优劣。后续不得继续根据测试结果切换基线；如需独立确认，可预先固定额外未参与选择的样本。

## 已完成的诊断，不要重复误判

- pilot：技术验证，不是正式基线。
- diagnostic_v1：固定20训练题自由生成0/20；精度三条件在固定20验证题均0/20；正常/错配latent的teacher-forcing损失差很小。
- overfit8_v1：原初始化重复8题200更新，140步起连续四次评测8/8；200步解释仅5/15位置一致。说明可拟合少量题，不能推出通用算术或泛化能力。调试checkpoint绝不能用作E5初始化。
- overfit8_audit_v2：标签、答案CE、三组梯度通过。早期CE断言因比较不同autocast条件失败；相同条件重算完全一致，未发现该处损失错位。
- audit_official：模型源码与固定官方版本除换行外相同。当前保留最后算式，相当于include_last_cot=True；官方示例默认False。2,000条在保留末步条件下主要输入与标签逐token一致。微批1无padding。训练与解码使用推理器embedding，这也是官方训练设计，不应随意当成bug修改。
- 官方示例40epochs、lr3e-3、6latent，并含作者路径；我们是课程规模适配，不是完整论文规模复现。v1单组weight decay也是已声明的差异。

## 文件怎么使用

| 路径 | 用途 |
|---|---|
| e3/data/{train,validation}.jsonl、manifest.json | 冻结训练与验证清单，2000/200 |
| e1/data/ | 600条冻结测试题，SVAMP已改为ChilleD/SVAMP test抽200 |
| e3/config.json、common.py、train.py | v1公共配置、模型/输入/解码构建、训练与恢复 |
| e3/evaluate.py、metrics.py | 自训模型评测、严格答案解析与4步长度统计 |
| e3/train_variants.py、variant_support.py | v2/v3单变量实验入口；不要当作v1默认入口 |
| e3_runs/control/ | v1训练日志、配置、初始化证据、checkpoint |
| e3_runs/control_diagnostics/ | B做E4的主数据：200验证题逐题答案与5个解码位置 |
| e3_runs/control_test/ | 当前基线的正式测试结果 |
| e3_runs/control_v2*、control_v3* | 配置对照结果，完整保留 |
| e3_runs/diagnostic_v1、overfit8*、pilot/ | 技术和失败诊断 |
| audit_official/ | 固定官方源码、比较脚本、机器可读结果、审计说明 |
| e3/checkpoints_release.json | Release中完整checkpoint的分片、字节数与SHA256 |
| e3/download_checkpoints.py | 下载、校验并恢复checkpoint |
| e3/summarize_handoff.py | 无需GPU，从逐题文件复算三版验证准确数与步骤匹配数 |

## B先做E4，再做E5

1. 以control_diagnostics为主，核对运算、数字、中间结果；区分缺步骤、错误运算/操作数、固定算式、答对解释错。不要把PAD位置当作应当正确的参考步骤。
2. 4次latent递推实际输入z0–z3，辅助decoder监督z0–z4，共5个位置；slot4有时有参考，但它不是第5次推理输入。分开报告。
3. 根据E4写一个简短假设，只实现一个改动，例如增强latent与对应步骤的联系。当前材料不能证明内部向量完全相同或完全没有信息。
4. E5从shared_init.pt开始，公共设置使用v1：lr5e-5、单组wd0.01、4步、750更新；不要从control/last.pt继续训练，不要混用v3分组。
5. 若需要修改训练入口对初始化配置的校验，只允许事先确定的改进项；记录代码差异，保留相同初始权重、顺序及评测。
6. A统一评测改进组与v1，报告失败与各题型结果。基线接近性能下限是显著局限，不能夸大少数题提升。

## 下载权重并复现

普通Git仅放代码、数据与文本结果，避免超大二进制历史。所有已返回的checkpoint（包括optimizer/RNG）压缩分片放在Release：

https://github.com/azhao258/NLP_Final_Project/releases/tag/e3-handoff-2026-11-18

在仓库根目录：

```bash
python e3/download_checkpoints.py --names shared_init control
```

通常B只需要这两份。完整取回全部历史模型可用--all。解压恢复的.pt包含原优化器及随机数状态，校验SHA后才落到正式路径。需要足够磁盘空间。

安装SIM-CoT/CODI/requirements.txt（Python3.12），准备assets/gpt2。可用仓库download.py下载资产，也可复制原assets并运行verify_assets.py。不要重建或重抽数据。Git中保留原始文件字节以便SHA核对。

复测v1验证集的例子（新输出目录）：

```bash
python e3/evaluate.py --checkpoint e3_runs/control/last.pt --split validation --explain --output e3_runs/v1_validation_recheck
```

已有结果无需为交接重复跑。B如需从头复现v1，恢复shared_init.pt后使用e3/train.py并选择新输出目录。运行前阅读e3/README.md和本交接的覆盖说明。

## E1/E2数据更新提醒

B原E2代码及结果仍原样保留在e2/和e1_results/sim_full/。其中旧SVAMP为混合划分，不再用于当前表格；新版CODI=88/200（44%）、SIM-CoT=85/200（42.5%），分别见codi_svamp_course_test/、sim_svamp_course_test/。原GSM8K和GSM-Hard可保留。旧e2对比CSV属于历史版本，勿误当最新汇总。
