"""Tiny CPU model: weighting parity and optimizer/RNG resume across ramp end."""
import copy
import json
import unittest
from unittest.mock import patch
from e5.p1 import train_p1 as train
import common
import torch
from transformers import GPT2Config, GPT2LMHeadModel


class ModelTests(unittest.TestCase):
    def test_weighting_and_resume(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        cfg = json.loads((common.ROOT / "e3/config.json").read_text())
        cfg.update(projection_dim=32, lora_r=4)
        original = common.TrainingArguments
        def cpu_args(**kw):
            return original(**dict(kw, bf16=False, use_cpu=True))
        def tiny(*args, **kwargs):
            return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_embd=32,
                n_layer=1, n_head=2, n_positions=1024))
        with patch.object(common, "TrainingArguments", cpu_args), patch("src.model.AutoModelForCausalLM.from_pretrained", tiny):
            model = common.build(cfg).eval()
        row = common.read_rows(common.ROOT / "e3/data/train.jsonl")[0]
        batch = common.tensor_batch(row, model, "cpu")
        with torch.no_grad():
            baseline = model(**batch, step=0, step_ratio=0.)
            model.explain_loss_factor = train.explain_factor(0)
            ramp = model(**batch, step=0, step_ratio=0.)
        for key in ("ce_loss", "distill_loss", "ref_ce_loss"):
            torch.testing.assert_close(baseline[key], ramp[key], rtol=0, atol=0)
        torch.testing.assert_close(ramp["explain_loss"], baseline["explain_loss"] * .1)
        torch.testing.assert_close(ramp["loss"], baseline["loss"] - .9 * baseline["explain_loss"])
        self.assertFalse(model.decoder._forward_pre_hooks)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1-n/750)
        def update(step):
            optimizer.zero_grad(set_to_none=True)
            model.explain_loss_factor = train.explain_factor(step)
            result = model(**batch, step=step, step_ratio=step/750)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        update(247)
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), rng=train.rng_state(), done=248))
        for step in range(state["done"], 251):
            update(step)
        expected = {n:p.detach().clone() for n,p in model.named_parameters()}
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        train.restore_rng(state["rng"])
        for step in range(state["done"], 251):
            update(step)
        self.assertEqual(model.explain_loss_factor, 1.)
        self.assertEqual(scheduler.state_dict(), expected_scheduler)
        for n,p in model.named_parameters():
            torch.testing.assert_close(p, expected[n], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
