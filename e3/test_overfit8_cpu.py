"""Tiny CPU integration test, not a real GPT-2 overfit run."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import common
from overfit8 import INDICES, interface_audit, backward_batch, gradient_report, evaluate


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    cfg = json.loads((common.ROOT / "e3/config.json").read_text())
    cfg.update(projection_dim=32, lora_r=4, answer_max_tokens=2, decode_max_tokens=2)
    original = common.TrainingArguments
    def cpu_args(**kw):
        kw.update(bf16=False, use_cpu=True)
        return original(**kw)
    def tiny(*args, **kwargs):
        return GPT2LMHeadModel(GPT2Config(vocab_size=50257,n_embd=32,n_layer=1,n_head=2,n_positions=1024))
    all_rows = common.read_rows(common.ROOT / "e3/data/train.jsonl")
    rows = [all_rows[i] for i in INDICES]
    with patch.object(common,"TrainingArguments",cpu_args), patch("src.model.AutoModelForCausalLM.from_pretrained",tiny):
        model = common.build(cfg).train()
        report = interface_audit(model,rows,4)
        assert [x["arithmetic_steps"] for x in report] == [1,1,1,2,2,2,3,3]
        original_params = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        losses = backward_batch(model,rows,4)
        groups = gradient_report(model)
        assert all(x["finite"] and x["norm_before_clip"]>0 for x in groups.values())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=5e-5)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        from overfit8 import group_name
        changed = {g:False for g in groups}
        for n,p in model.named_parameters():
            if p.requires_grad and not torch.equal(original_params[n],p): changed[group_name(n)] = True
        assert all(changed.values())
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluate(model,rows,cfg,1,Path(directory))
            assert summary["active_slots"] == 15 and summary["n"] == 8
            assert len(common.read_rows(Path(directory)/"predictions.jsonl")) == 8
    common.write_json(common.ROOT / "e3/overfit8_cpu_results.json",dict(status="PASS",losses=losses,
        gradient_groups=groups,optimizer_changed_groups=changed,
        tests=["8 label/answer positions", "15 arithmetic target alignments", "manual answer CE equality",
               "finite nonzero gradients in all 3 groups", "optimizer changes all 3 groups", "free and teacher-forced evaluation export"],
        limitation="Tiny CPU model only; GPU audit and overfit run not executed"))
    print("PASS: overfit8 tiny CPU integration")


if __name__ == "__main__":
    main()
