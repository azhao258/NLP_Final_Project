# E3 与官方实现核对（2026-11-17）

核对对象：InternLM/SIM-CoT 固定提交 cd3291dac9fe1e495da9fe00e61a9b96a884b2ed 的 CODI 路线。下载原始 train.py、test.py、src/model.py 与 scripts/train_gpt2_gsm8k-aug-decoder-2.sh，文件保存在本目录。不是 Coconut 路线，也没有把 LLaMA 示例当作 GPT-2 配置。

## 结论

模型结构和损失直接来自官方，当前训练与自行实现的推理/解码路径没有发现本次测试能识别的错位。完整训练管线和默认超参数并不等同官方示例。不能把低准确率确定归因于代码错误，也不能说已证明只有学习率问题。

## 已验证的一致项

- 当前 src/model.py 与固定官方版本去除换行格式差异后全文一致，四项损失、投影器、decoder 均未改写。
- 提取并执行官方实际 preprocess 函数，用当前同一 tokenizer 对全部 2000 条训练数据比较。保留最后算式时，encoder_input_ids、decoder_input_ids、ref_input_ids、labels、ref_labels、两个答案位置全部一致。
- 同样的 remove_eos=True 设置，问题后加入 BOT，答案前加入 EOT，答案结尾为 EOS。问题不计入参考路径 CE。蒸馏对齐数值答案前一位置。
- 官方 decoder 训练本来就使用推理器 embedding 编码正确前缀，不是我们意外接错模块。当前自由解码也使用该 embedding；不要未经实验改成 decoder 自身 embedding。
- 官方先监督 z0，再监督 K 次递推后的 z1..zK。当前 K=4 导出5个解释位置，实际输入推理器4个latent，计数一致。
- CPU tiny GPT-2 实测：训练 forward 捕获的5个投影向量与独立 latent 提取一致；decoder 的有效输入前缀一致；训练完整答案前向与缓存推理的首答案 token logits 一致。
- 以上功能测试不是完整 GPT-2 GPU 的逐位等价证明。真实 GPU 梯度和答案 CE 对齐证据另见用户返回的 overfit8_audit_v2。

## 实际差异

1. 最后算式：官方 ModelArguments/TrainingArguments 默认 include_last_cot=False，GPT-2 decoder 示例没有覆盖该选项；icot 分支按空格切分后删除最后一个片段。当前完整保留，相当于该数据开关为 True。在这2000条中，默认删除行为会改变1999条参考输入/标签/位置，并使空链从1条增加到345条。它不是一个无关紧要的默认值。建议保留完整链以提供步骤监督，但明确声明数据适配。
2. 官方 icot 分支有答案首字符必须为数字的过滤，当前为满足前2000条要求不丢弃样本；该过滤会拒绝其中14条。不能静默套用过滤再声称训练了原始前2000条。
3. 官方 train.py 当前固定提交在 Dataset 内覆盖 raw_data，从作者机器上的 gsm_train_clean.json 读取，而不是实际执行 Hugging Face GSM8k-Aug 加载。官方脚本含作者本地模型路径；无法据这些脚本证明发布 checkpoint 当年使用的全部训练资产。
4. 示例与E3：学习率3e-3 vs 5e-5；latent6 vs4；40 epochs vs3；micro64×accum2 vs1×8；clip2 vs1；weight decay0.1 vs0.01；cosine vslinear；warmup ratio0.03 vs38/750。官方 exp_mode=False，因此脚本中的 exp_data_num=2000 并未实际限制训练集。不能直接运行这个脚本满足课程限制。
5. 优化器参数分组：官方使用 Transformers 4.52.4 Trainer，排除bias和LayerNorm等参数的weight decay；当前 E3 的 AdamW 单组对全部可训练参数使用0.01。是已确认的公共训练差异，不是已证明的低分根因。
6. 权重精度：官方构造器BF16加载，E3额外转换为FP32 master参数并采用BF16 autocast；独立评测又将权重转换为BF16。之前精度诊断已证明答案会变化，应统一报告评测精度。
7. 当前micro batch=1无批内padding；官方collator支持padding。该差异在当前单样本训练无需修复，但不能不经检查直接增大micro batch。
8. 官方 test.py 在生成循环保留 pdb.set_trace()；probe_latent_token.py 主要通过推理器LM head探测，不等于当前辅助decoder自由解释工具。我们自行实现解码，已按训练所用embedding、投影位置、EOS和前缀核对，而不是宣称直接使用了官方解释脚本。

## 推荐处理

- 暂不改模型拓扑、decoder embedding来源或K+1监督位置。没有本次审计证据支持这些是接错的地方。
- 保留control_v1和所有诊断，不覆盖代码或checkpoint。
- 报告称“基于官方SIM-CoT/CODI模型实现的课程规模适配control”，列明完整算式、固定2000条、4步、3轮、精度和优化器设置。
- 下一版公共基线若要提高与官方训练入口的一致性，优先把weight decay分组明确化；保留课程要求的数据、预算和声明过的完整链策略。新配置必须由control和改进组共同使用。
- 如果继续做“仅改变学习率”的实验，就必须保持旧分组不变，并准确标注这一变量；如果同时修正分组，就不能把效果变化只归因于学习率。当前不应盲目跳到3e-3。
- 用户本次请求是核对，因此本轮没有修改现有训练、推理代码，也没有启动GPU训练。

## 复查

在仓库根目录运行 python audit_official/check_alignment.py。需要本地tokenizer及现有依赖，不需要GPU。机器可读证据见 alignment_report.json。该脚本只执行从官方AST中提取的预处理函数，不执行官方训练入口。

官方源码：https://github.com/InternLM/SIM-CoT/tree/cd3291dac9fe1e495da9fe00e61a9b96a884b2ed/CODI

Trainer分组：https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/trainer.py （get_decay_parameter_names/create_optimizer）
