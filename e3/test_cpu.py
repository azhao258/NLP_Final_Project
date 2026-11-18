"""Structural integration test with a tiny random GPT-2, not a GPU training claim."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import common
from common import ROOT, build, generate, read_rows, tensor_batch, targets, write_json
from metrics import summarize


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    cfg = json.loads((ROOT / "e3/config.json").read_text())
    cfg.update(projection_dim=32, lora_r=4)
    original_args = common.TrainingArguments
    def cpu_args(**kwargs):
        kwargs.update(bf16=False, use_cpu=True)
        return original_args(**kwargs)
    def tiny_model(*args, **kwargs):
        return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_embd=32, n_layer=1, n_head=2, n_positions=1024))
    with patch.object(common, "TrainingArguments", cpu_args), patch("src.model.AutoModelForCausalLM.from_pretrained", tiny_model):
        model = build(cfg).train()
        row = read_rows(ROOT / "e3/data/train.jsonl")[0]
        result = model(**tensor_batch(row, model, "cpu"), step=0, step_ratio=0.)
        assert torch.isfinite(result["loss"]) and result["explain_loss"] > 0
        result["loss"].backward()
        assert model.prj[1].weight.grad.abs().sum() > 0
        before = model.prj[1].weight.detach().clone()
        torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4).step()
        assert not torch.equal(before, model.prj[1].weight)
        model.eval()
        before_reload = generate(model, row["question"], cfg["latent_steps"], 2, explanations=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save(model.state_dict(), path)
            restored = build(cfg)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            restored.eval()
            after_reload = generate(restored, row["question"], cfg["latent_steps"], 2, explanations=True)
            assert before_reload == after_reload
        assert len(before_reload["decoded_steps"]) == 5
        assert [x["consumed_by_reasoner"] for x in before_reload["decoded_steps"]] == [True]*4+[False]
        empty = read_rows(ROOT / "e3/data/train.jsonl")[67]
        assert all(x == [model.tokenizer.pad_token_id] for x in targets(empty, model.tokenizer, 4))
        no_steps = model(**tensor_batch(empty, model, "cpu"), step=0, step_ratio=0.)
        assert torch.isfinite(no_steps["loss"]) and no_steps["explain_loss"] == 0
        with tempfile.TemporaryDirectory() as directory:
            row_metric = dict(dataset="tiny", reference_tokens=20, latent_steps=4, reference_type="equation",
                correct=True, prediction="1", latency_seconds=.1, answer_tokens=1, stop_reason="eos")
            (Path(directory)/"tiny.predictions.jsonl").write_text(json.dumps(row_metric)+"\n")
            assert summarize(directory)[0]["length_ratio"] == 5
    report = dict(status="PASS", tests=["official forward/backward and optimizer update on tiny GPT-2",
        "strict checkpoint save/load reproduces answer and all five decoded slots",
        "empty arithmetic chain gives finite loss without explanation supervision",
        "four-step compression denominator"], limitation="CPU tiny-model integration only; A6000 pilot required")
    write_json(ROOT / "e3/cpu_test_results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
