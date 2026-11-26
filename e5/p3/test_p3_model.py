"""Run in the frozen server environment; tiny real CODI model, no GPU needed."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'e3'))
import common
from transformers import GPT2Config, GPT2LMHeadModel
from e5.p3.p3_stage import training_forward


class Integration(unittest.TestCase):
    def test_real_reference_parity_and_joint(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        cfg = json.loads((common.ROOT/'e3/config.json').read_text())
        cfg.update(projection_dim=32, lora_r=4)
        original = common.TrainingArguments
        def cpu_args(**kw):
            return original(**dict(kw, bf16=False, use_cpu=True))
        def tiny(*args, **kw):
            return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_embd=32,
                n_layer=1, n_head=2, n_positions=1024))
        with patch.object(common, 'TrainingArguments', cpu_args), patch('src.model.AutoModelForCausalLM.from_pretrained', tiny):
            m = common.build(cfg).eval()
        batch = common.tensor_batch(common.read_rows(common.ROOT/'e3/data/train.jsonl')[0], m, 'cpu')
        with torch.no_grad():
            baseline = m(**batch, step=149, step_ratio=149/750)
        result = training_forward(m, batch, 149)
        torch.testing.assert_close(result['loss'], baseline['ref_ce_loss'])
        self.assertTrue(result['loss'].requires_grad)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-5, weight_decay=.01)
        before = {n:p.detach().clone() for n,p in m.named_parameters() if n.startswith(('prj.', 'decoder.'))}
        opt.zero_grad(set_to_none=True)
        result['loss'].backward(); opt.step()
        for n,p in m.named_parameters():
            if n in before:
                self.assertIsNone(p.grad)
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        with torch.no_grad():
            expected = m(**batch, step=150, step_ratio=150/750)
            actual = training_forward(m, batch, 150)
        for key in ('loss', 'ce_loss', 'distill_loss', 'ref_ce_loss', 'explain_loss'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        opt.zero_grad(set_to_none=True)
        training_forward(m, batch, 150)['loss'].backward()
        for module in (m.prj, m.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()
