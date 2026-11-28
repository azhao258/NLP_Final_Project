P5：训练集跨题步骤间隔对齐（实验方案与运行指南）

研究假设
P4 仅提高正配对相似度，却未提升正确步骤的区分。P5 检验：固定 K=4 时，增加
正负步骤间隔约束，能否提高步骤对应性，同时保持答案表现。
L_total = 原 v1 联合损失 + 0.1 * mean(relu(0.02 + cos(z,h_neg) - cos(z,h_pos)))。
0.1 和 0.02 是本轮预设探索参数，不根据测试成绩扫描。无 GRPO、无前缀遮蔽、无预热。

比较边界
主对照为冻结 v1。P5 从原 shared_init.pt 开始，不从 P3、P4 或 v1 成品继续训练。
与 v1 保持 2000/200 数据、seed11、样本顺序、lr5e-5、单组wd0.01、micro1×累积8、
3epochs、750更新、warmup38、线性调度、clip1。仅增加上述训练目标。
为避免两侧 dropout 噪声，正负参考表示都另做 eval/no_grad 前向；参考模型仍随训练变化，
不是冻结教师。这与 P4 复用训练模式参考前向不同，不能将 P5/P4 比较声称为仅替换公式。
每个有配对的样本增加两次参考前向，所以更新预算相同，不等于 FLOPs 或训练时间相同。
额外前向关闭dropout且保存恢复RNG及模块状态，原训练前向不改变。

负例规则（正式训练不读取验证或测试内容来构造负例）
仅使用 e3/data/train.jsonl 的前2000条。train_negative_pairs.json 存储冻结配对与数据哈希。
每个anchor固定选一个不同题干的训练donor；独立随机种子405105+anchor索引。
仅比较相同槽位；两侧算式必须可解析、精确成立、算式数量相等且结果集合不相交。
这样保守排除了结果相同的不同表达，包括已发现的五折替代；也会排除部分本可使用的负例。
多算式尾槽沿用原映射；slot4只读出，不称为第五次递推。PAD、不可靠近似等不参与新增项。
1984个anchor有配对，3744个有效配对槽；16个anchor只使用原损失，不删除训练题。
所有6000次anchor呈现保持原顺序；donor额外呈现需在报告披露。
数值筛选不是完整语义审核，不能保证消除所有题目歧义；保留负例清单供复核。
启动训练时按实际tokenizer重新生成并检查配对ID/槽位，遇差异立即停止。

放置方式
把整个 P5 文件夹放到服务器项目根目录，与 e3、e5、assets 同级。
依赖项目已有 e5/p4/p4_alignment.py、p4_audit.py、p1_schedule.py、protocol.py。
请使用含B最新E5文件的仓库版本。不得覆盖 e3/common.py 或官方 model.py。

新实例环境（已有环境跳过创建）
mkdir -p ~/experiment
conda create -y --override-channels -c conda-forge -p "$HOME/experiment/.conda312" python=3.12 pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/experiment/.conda312"
cd ~/experiment/NLP_Final_Project-main
python -m pip install --prefer-binary -r SIM-CoT/CODI/requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"

必备文件
assets/gpt2/ 六个原文件；保留仓库原 assets/manifest.json，不用download.py重写它。
e3_runs/shared_init.pt：使用原E3发布权重。可在本机下载再上传：
python e3/download_checkpoints.py --names shared_init
同时保留 v1 用于比较；P5训练不需要P3/P4权重。

先做检查和2次更新pilot
python -m unittest P5.test_rules
python P5/preflight.py
python -u P5/train.py --init e3_runs/shared_init.pt --pilot-updates 2 --output P5/runs/pilot2
python e3/evaluate.py --checkpoint P5/runs/pilot2/last.pt --split validation --limit 4 --explain --output P5/results/pilot2_reload
确认loss有限、valid_slots非零、正负相似度有记录、成功保存重载、输出5个解释槽。
pilot不用于效果判断；首次真实GPU验证仍需完成，不能把本地CPU检查当作训练成功。

正式训练：再次从shared_init开始，不续训pilot
python -u P5/train.py --init e3_runs/shared_init.pt --output P5/runs/formal
中断恢复（同目录同代码同配置）：
python -u P5/train.py --init e3_runs/shared_init.pt --output P5/runs/formal --resume
损失日志、negative_pairs、sample_order、contract、环境、session耗时和last.pt全部保存。

正式验证
python e3/evaluate.py --checkpoint P5/runs/formal/last.pt --split validation --explain --output P5/results/validation
如项目中已放入我们的检索诊断脚本：
python e5/check_reference_retrieval.py --checkpoint P5/runs/formal/last.pt --output P5/results/retrieval
检查消耗槽与readout槽分别的检索结果，结合解码解释，不只看对比损失下降。

配置与最终750步checkpoint固定后，正式测试只运行一遍并保留负结果：
python e3/evaluate.py --checkpoint P5/runs/formal/last.pt --split test --explain --output P5/results/test
测试集已经参与过历史研究，属于探索性比较；不能用测试结果反复调系数或挑checkpoint。
P5仍保持K=4，不声称提高结构压缩率。沿用E4步骤评分、严格答案解析和分题型统计。
历史延迟不能直接用来证明加速；额外训练耗时需计入剩余50 GPU-hour总预算。

返回材料
完整保存 P5/runs/formal 和 P5/results。先取回文本输出分析，last.pt另行备份。
不要删除服务器checkpoint直到本地完整备份并核验。

本地验证状态
已做：语法、数值解析/等价结果排除、训练集配对确定性与边界检查、间隔损失梯度测试。
未做：真实模型CUDA前向反向、pilot重载、正式训练。首次需先跑pilot并检查输出。
