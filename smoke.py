"""Run one mode per process on Linux/CUDA. All weights and data are local."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from transformers import set_seed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "SIM-CoT/CODI"))
from src.model import CODI, ModelArguments, TrainingArguments

OUT = ROOT / "outputs"
ASSETS = ROOT / "assets"
IDS = {"codi": "zen-E/CODI-gpt2", "sim": "internlm/SIM_COT-GPT2-CODI"}
STEPS = {"codi": 6, "sim": 6, "control": 4}


def build(mode, decoder):
    k = STEPS[mode]
    args = ModelArguments(model_name_or_path=str(ASSETS / "gpt2"), train=False,
                          use_decoder=decoder, lora_r=128, lora_alpha=32, lora_init=True)
    training = TrainingArguments(output_dir=str(OUT), bf16=True, use_prj=True, prj_dim=768,
                                 distill_loss_div_std=True, remove_eos=True, num_latent=k,
                                 inf_latent_iterations=k, report_to=[])
    adapter = LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                         lora_dropout=0.1, target_modules=["c_attn", "c_proj", "c_fc"])
    return CODI(args, training, adapter)


def tensor(seq):
    return torch.tensor([seq], device="cuda", dtype=torch.long)


def released(mode):
    if mode == "codi":
        weights = torch.load(ASSETS / "codi/pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        weights = load_file(str(ASSETS / "sim/model-00001-of-00001.safetensors"))
    model = build(mode, any(k.startswith("decoder.") for k in weights))
    missing, extra = model.load_state_dict(weights, strict=False)
    # Safetensors can omit one copy of a tied embedding; require its alias in the file.
    pointers = {v.data_ptr() for k,v in model.state_dict().items() if k in weights}
    unresolved = [k for k in missing if model.state_dict()[k].data_ptr() not in pointers]
    if unresolved or extra:
        raise ValueError(dict(unresolved=unresolved, unexpected=extra))
    model.codi.tie_weights()
    return model.cuda().to(torch.bfloat16).eval(), list(missing)


@torch.no_grad()
def infer(model, question, k, explain=False):
    tok = model.tokenizer
    emb = model.get_embd(model.codi, model.model_name)
    out = model.codi(input_ids=tensor(tok.encode(question.strip())+[model.bot_id]),
                     output_hidden_states=True, use_cache=True)
    for idx in range(k):
        z = model.prj(out.hidden_states[-1][:,-1:])
        if explain and idx == 0:
            decoded = []
            x = z
            for _ in range(32):
                next_id = model.decoder(inputs_embeds=x).logits[0,-1].argmax().item()
                if next_id == tok.eos_token_id:
                    break
                decoded.append(next_id)
                x = torch.cat([x, emb(tensor([next_id]))], dim=1)
            return decoded, tok.decode(decoded)
        out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values,
                         output_hidden_states=True, use_cache=True)
    next_ids = tensor([model.eot_id])
    answer = []
    for _ in range(64):
        out = model.codi(input_ids=next_ids, past_key_values=out.past_key_values, use_cache=True)
        token = out.logits[0,-1,:model.codi.config.vocab_size-1].argmax().item()
        if token == tok.eos_token_id:
            break
        answer.append(token)
        next_ids = tensor([token])
    return tok.decode(answer)


def one_update(model, row):
    tok = model.tokenizer
    chain = row["cot"]
    equations = re.findall(r"<<[^<>]+>>", chain)
    if not equations:
        raise ValueError(f"No arithmetic targets in training row: {chain}")
    q = tok.encode(row["question"])
    steps = tok.encode(" ".join(equations))
    prefix = tok.encode("The answer is:")
    a = tok.encode("The answer is: " + str(row["answer"]).split()[-1])+[tok.eos_token_id]
    ref = q + steps + a
    dec = [model.eot_id]+a
    inputs = dict(encoder_input_ids=tensor(q+[model.bot_id]), decoder_input_ids=tensor(dec),
                  ref_input_ids=tensor(ref), labels=tensor(dec), ref_labels=tensor([-100]*len(q)+ref[len(q):]),
                  ref_answer_position=tensor([len(q)+len(steps)+len(prefix)]).flatten(),
                  model_answer_position=tensor([1+len(prefix)]).flatten(),
                  encoder_attention_mask=torch.ones_like(tensor(q+[model.bot_id]),dtype=torch.bool),
                  ref_attention_mask=torch.ones_like(tensor(ref),dtype=torch.bool),step=0,step_ratio=0.)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-3)
    model.train()
    result = model(**inputs)
    loss = result["loss"]
    assert loss.isfinite() and result["explain_loss"].item()>0
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
    assert grad.item()>0
    optimizer.step()
    return dict(loss=loss.item(),decoder_loss=result["explain_loss"].item(),gradient_norm=grad.item())


def first_row(filename):
    columns = json.loads((ASSETS/"data"/filename).read_text(encoding="utf-8"))
    return {name: values[0] for name, values in columns.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=STEPS)
    mode = parser.parse_args().mode
    OUT.mkdir(exist_ok=True)
    set_seed(11)
    test = first_row("gsm8k_test.json")
    report = dict(mode=mode, model_id=IDS.get(mode,"self-trained-control"),base="openai-community/gpt2",
                  latent_steps=STEPS[mode],question=test["question"],gold=test["answer"],
                  torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),seed=11)
    if mode == "control":
        model = build(mode,True).cuda()
        row = first_row("gsm8k_aug_train.json")
        report.update(one_update(model,row))
        model.eval()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            before = infer(model,test["question"],4,True)
        checkpoint = OUT/"control.pt"
        torch.save(model.state_dict(),checkpoint)
        del model
        model = build(mode,True)
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
        model.cuda().eval()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            ids, explanation = infer(model,test["question"],4,True)
        assert (ids,explanation)==before
        report.update(train_row=0,batch_size=1,updates=1,decoded_latent_index=0,token_ids=ids,
                      text=explanation,reload_identical=True,nonempty_text=bool(explanation.strip()))
    else:
        model, aliases = released(mode)
        report.update(tied_aliases=aliases,loading="CODI GPT-2 LoRA wrapper + audited state_dict",
                      answer=infer(model,test["question"],STEPS[mode]))
    report["total_parameters"]=sum(p.numel() for p in model.parameters())
    report["answer_model_parameters"]=sum(p.numel() for p in model.codi.parameters())+sum(p.numel() for p in model.prj.parameters())
    (OUT/f"{mode}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

