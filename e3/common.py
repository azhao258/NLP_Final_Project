"""E3 uses the vendored SIM-CoT CODI implementation without changing its loss."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from peft import LoraConfig, TaskType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SIM-CoT/CODI"))
from src.model import CODI, ModelArguments, TrainingArguments, get_steps


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8", newline="\n")


def build(cfg):
    args = ModelArguments(model_name_or_path=str(ROOT / "assets/gpt2"), train=False,
                          use_decoder=True, lora_init=True)
    training = TrainingArguments(output_dir=str(ROOT / "e3_runs"), bf16=True,
        use_prj=True, prj_dim=cfg["projection_dim"], distill_loss_div_std=True,
        remove_eos=True, num_latent=cfg["latent_steps"],
        inf_latent_iterations=cfg["latent_steps"], report_to=[], print_loss=False,
        explain_loss_factor=cfg["explain_loss_factor"],
        distill_loss_factor=cfg["distill_loss_factor"], ref_loss_factor=cfg["ref_loss_factor"])
    adapter = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["c_attn", "c_proj", "c_fc"])
    # FP32 master parameters and optimizer states, BF16 CUDA autocast for forward passes.
    return CODI(args, training, adapter).float()


def pack(row, tokenizer, bot_id=50258, eot_id=50259):
    q = tokenizer.encode(row["question"].strip())
    cot = tokenizer.encode(row["cot"].strip())
    prefix = tokenizer.encode("The answer is:")
    answer = tokenizer.encode("The answer is: " + str(row["answer"]).strip()) + [tokenizer.eos_token_id]
    assert answer[:len(prefix)] == prefix
    ref = q + cot + answer
    dec = [eot_id] + answer
    return dict(encoder_input_ids=[q + [bot_id]], decoder_input_ids=[dec],
        ref_input_ids=[ref], labels=[dec], ref_labels=[[-100] * len(q) + cot + answer],
        ref_answer_position=[len(q) + len(cot) + len(prefix)],
        model_answer_position=[1 + len(prefix)],
        encoder_attention_mask=[[1] * (len(q) + 1)], ref_attention_mask=[[1] * len(ref)])


def tensor_batch(row, model, device):
    return {k: torch.tensor(v, device=device, dtype=torch.long)
            for k, v in pack(row, model.tokenizer, model.bot_id, model.eot_id).items()}


def targets(row, tokenizer, k):
    # Upstream supervises z0 before the loop and z1..zK after each update.
    ref = pack(row, tokenizer)["ref_input_ids"]
    return get_steps(ref, k + 1, start_ids=(16791, 9959), end_id=4211,
        eot_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id,
        stop_ids=(tokenizer.eos_token_id, tokenizer.pad_token_id))[0]


def load_checkpoint(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


@torch.inference_mode()
def generate(model, question, k, max_tokens=64, explanations=False, timed=False):
    device = next(model.parameters()).device
    tok = model.tokenizer
    ids = torch.tensor([tok.encode(question.strip()) + [model.bot_id]], device=device)
    if timed:
        torch.cuda.synchronize()
        start = time.perf_counter()
    out = model.codi(input_ids=ids, output_hidden_states=True, use_cache=True)
    latent = model.prj(out.hidden_states[-1][:, -1:])
    vectors = [latent]
    for index in range(k):
        out = model.codi(inputs_embeds=latent, past_key_values=out.past_key_values,
                         output_hidden_states=True, use_cache=True)
        if index < k - 1 or explanations:
            latent = model.prj(out.hidden_states[-1][:, -1:])
            vectors.append(latent)
    next_ids = torch.tensor([[model.eot_id]], device=device)
    answer = []
    stopped = False
    for _ in range(max_tokens):
        out = model.codi(input_ids=next_ids, past_key_values=out.past_key_values, use_cache=True)
        value = out.logits[0, -1, :model.codi.config.vocab_size - 1].argmax().item()
        answer.append(value)
        if value == tok.eos_token_id:
            stopped = True
            break
        next_ids = torch.tensor([[value]], device=device)
    if timed:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    visible = answer[:-1] if stopped else answer
    result = dict(raw_output=tok.decode(visible), token_ids=answer,
        answer_tokens=len(visible), generated_tokens=len(answer), stop_reason="eos" if stopped else "length")
    if timed:
        result["latency_seconds"] = elapsed
    if explanations:
        decoded = []
        emb = model.get_embd(model.codi, model.model_name)
        for index, z in enumerate(vectors):
            x, tokens, eos = z, [], False
            for _ in range(max_tokens):
                value = model.decoder(inputs_embeds=x).logits[0, -1].argmax().item()
                if value == tok.eos_token_id:
                    eos = True
                    break
                tokens.append(value)
                x = torch.cat([x, emb(torch.tensor([[value]], device=device))], dim=1)
            decoded.append(dict(slot=index, consumed_by_reasoner=index < k,
                text=tok.decode(tokens), token_ids=tokens, capped=not eos))
        result["decoded_steps"] = decoded
    return result
