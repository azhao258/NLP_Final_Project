> 历史运行指南。当前最终基线与交接规则以 [HANDOFF_TO_B.md](HANDOFF_TO_B.md) 和 active_baseline.json 为准。

# Control v2 / v3 新实例运行

v1不改动。v2仅lr=1e-4，所有可训练参数仍统一weight_decay=0.01。v3仅使用Transformers 4.52.4 Trainer的参数分组，bias/LayerNorm等weight_decay=0，其余0.01，lr仍5e-5。

两个入口使用原shared_init.pt的权重和随机数状态，校验原模型及训练代码哈希；原config.json保持不变。均为2000训练、200验证、4次latent、3epochs、750次更新、batch1×累积8、warmup38、相同线性调度与clip1。样本顺序写入sample_order.json和哈希，优化器成员写入optimizer_groups.json。不得从旧control或8题模型继续训练。

## 1 上传（本机PowerShell）

先更新SSH别名NLP为新实例IP/SSH端口。建议使用已预装NVIDIA驱动的A6000实例。

```powershell
$base = "C:\Users\83946\Desktop\NLP\AIAA4051_Projects\experiment"
scp "$base\control_variants_code.zip" NLP:~/
ssh NLP "mkdir -p ~/experiment/assets ~/experiment/e3_runs"
scp -r "$base\assets\gpt2" NLP:~/experiment/assets/
scp "$base\returned_results\e3_runs\shared_init.pt" NLP:~/experiment/e3_runs/
```

包只包含代码、配置和冻结数据，不含权重。GPT-2文件约0.55GB，共同初始化约1.08GB。若同一路径已有且校验通过，无需重传。已有实例解压前先保留自己的修改；本包是本次冻结的训练代码。不要重新运行prepare_data或download.py覆盖清单。

## 2 解压与环境（远程终端）

```bash
cd ~/experiment
python3 -m zipfile -e ~/control_variants_code.zip .
```

若原环境仍在，直接激活或将其bin加入PATH。若不存在，用conda新建，避免系统Python 3.14：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y --override-channels -c conda-forge -p "$HOME/experiment/.conda312" python=3.12 pip
conda activate "$HOME/experiment/.conda312"
python -m pip install --prefer-binary -r SIM-CoT/CODI/requirements.txt
```

如果conda命令不存在，先配置Python3.12环境，不要直接用系统3.14安装。随后：

```bash
cd ~/experiment
export PATH="$HOME/experiment/.conda312/bin:$PATH"
python --version
python e3/check_variants.py
```

只有PASS后再训练。GPU不可用时先解决实例驱动/挂载，不是靠重新安装torch解决。

## 3 顺序训练

```bash
python -u e3/train_variants.py --variant v2 --output e3_runs/control_v2
python -u e3/train_variants.py --variant v3 --output e3_runs/control_v3
```

逐条执行，确认上一条COMPLETE再启动下一条。建议在tmux会话中运行。每50步保存last.pt；中断恢复使用原命令附加--resume。首次checkpoint前失败须用新目录从初始化重新开始。默认750步，--pilot-updates只用于独立目录的技术测试，不能当正式结果。

## 4 仅验证集统一评测

```bash
python -u e3/evaluate.py --checkpoint e3_runs/control_v2/last.pt --split validation --explain --output e3_runs/control_v2_validation
python -u e3/evaluate.py --checkpoint e3_runs/control_v3/last.pt --split validation --explain --output e3_runs/control_v3_validation
```

这两项统一为BF16权重+autocast、greedy、每集200题，并输出解释。不用训练日志中FP32 master的验证准确率与此混比。v1可用原control_diagnostics结果（已为同一独立评测设置），若需复测先上传原control/last.pt，再用同一evaluate.py输出新目录。

不要在配置选择前跑测试集。根据固定最终checkpoint的验证准确率及解释诊断选择公共配置；保留所有结果，不根据测试集挑选，不宣称单种子微小差异稳定。冻结后再评测选中的正式基线测试集，并让改进组使用相同公共配置和原初始权重。

## 5 下载

本机PowerShell：

```powershell
$dest = "C:\Users\83946\Desktop\NLP\AIAA4051_Projects\experiment\returned_results\e3_runs"
scp -r NLP:~/experiment/e3_runs/control_v2 NLP:~/experiment/e3_runs/control_v3 NLP:~/experiment/e3_runs/control_v2_validation NLP:~/experiment/e3_runs/control_v3_validation "$dest"
```

checkpoint含优化器，每份数GB。请回传整个目录用于核对初始化、顺序、优化器分组、训练曲线和解释。训练各约此前72分钟只能作估计；新环境会变化，记录实际GPU租用时间，并保持总预算低于50小时。

本地只验证了配置差异、参数分组和代码语法，没有执行新的A6000训练。
