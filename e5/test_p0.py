"""Tiny-model integration checks for P0, without training or checkpoint writes."""
import json
import unittest
from unittest.mock import patch
import p0_diagnose as p0
import common
import torch
from transformers import GPT2Config, GPT2LMHeadModel


class P0Tests(unittest.TestCase):
    def test_loss_graph_and_interventions(self):
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
        rows = common.read_rows(common.ROOT / "e3/data/train.jsonl")
        row = rows[0]
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        batch = common.tensor_batch(row, model, "cpu")
        original_out = model(**batch, step=0, step_ratio=0.)
        forward = p0.differentiable_forward(model)
        out = forward(**batch, step=0, step_ratio=0.)
        for key in ("loss",) + p0.LOSSES:
            torch.testing.assert_close(out[key], original_out[key], rtol=0, atol=0)
        named = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith(("codi.", "prj."))]
        params = [p for _,p in named]
        total = torch.autograd.grad(out["loss"], params, retain_graph=True, allow_unused=True)
        parts = [torch.autograd.grad(out[key], params, retain_graph=True, allow_unused=True) for key in p0.LOSSES]
        for i, g in enumerate(total):
            if g is not None:
                summed = sum(part[i] for part in parts if part[i] is not None)
                torch.testing.assert_close(g, summed, rtol=2e-4, atol=2e-6)
        stats = p0.gradient_stats(out, named)
        self.assertGreater(stats["shared"]["norms"]["explain_loss"], 0)
        self.assertEqual(stats["projector"]["norms"]["ref_ce_loss"], 0)
        z = p0.latent_vectors(model, row["question"], 4)
        normal = p0.reasoner_intervention(model, row, 4, 3)
        self.assertEqual(normal, p0.reasoner_intervention(model, row, 4, 3, z))
        self.assertEqual(normal["token_ids"], common.generate(model, row["question"], 4, 3)["token_ids"])
        donor = p0.latent_vectors(model, rows[1]["question"], 4)
        changed = p0.reasoner_intervention(model, row, 4, 3, donor)
        self.assertNotEqual(normal["answer_nll"], changed["answer_nll"])
        empty = dict(row, cot="There is no equation here.")
        no_steps = forward(**common.tensor_batch(empty, model, "cpu"), step=0, step_ratio=0.)
        self.assertEqual(float(no_steps["explain_loss"]), 0)
        self.assertEqual(p0.gradient_stats(no_steps, named)["shared"]["norms"]["explain_loss"], 0)
        for n, param in model.named_parameters():
            self.assertTrue(torch.equal(before[n], param))
            self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()
