"""CPU checks only: synthetic tensors and tiny random GPT-2, no E5 training run."""
import io
import unittest
from unittest.mock import patch

import torch

from e5.prefix_mask.prefix_mask import PrefixMask, attach_prefix_mask


class MaskTests(unittest.TestCase):
    def test_only_text_tokens_are_masked_and_labels_are_untouched(self):
        ids = torch.tensor([[10, 11, 12, 99, 0]])
        values = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
        before, labels = values.detach().clone(), ids.clone()
        masker = PrefixMask(1, 7, (10, 99, 0))
        out = masker.corrupt(values, ids)
        self.assertTrue(torch.equal(values, before))
        self.assertTrue(torch.equal(ids, labels))
        self.assertTrue(torch.equal(out[:, 0], before[:, 0]))
        self.assertTrue(torch.equal(out[:, [1, 4, 5]], before[:, [1, 4, 5]]))
        self.assertEqual(float(out[:, 2:4].detach().abs().sum()), 0)
        out.sum().backward()
        self.assertEqual(float(values.grad[:, 2:4].sum()), 0)
        self.assertEqual(float(values.grad[:, 0].sum()), 4)

    def test_rng_isolation_zero_probability_and_save_reload(self):
        ids = torch.arange(100).reshape(1, 100)
        values = torch.ones(1, 101, 2)
        zero = PrefixMask(0, 7, ())
        before = zero.state_dict()["generator"]
        self.assertIs(zero.corrupt(values, ids), values)
        self.assertTrue(torch.equal(before, zero.state_dict()["generator"]))
        masker = PrefixMask(.3, 7, ())
        global_before = torch.get_rng_state().clone()
        masker.corrupt(values, ids)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        stream = io.BytesIO()
        torch.save(masker.state_dict(), stream)
        expected = masker.corrupt(values, ids)
        stream.seek(0)
        restored = PrefixMask(.3, 99, ())
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(torch.equal(expected, restored.corrupt(values, ids)))
        with self.assertRaises(ValueError):
            PrefixMask(.4, 99, ()).load_state_dict(masker.state_dict())


class OfficialForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import e3.common as common
        from transformers import GPT2Config, GPT2LMHeadModel
        from e5.protocol import ROOT, read_json
        torch.set_num_threads(2)
        cfg = read_json(ROOT / "e3/config.json")
        cfg.update(projection_dim=16, lora_r=2)
        original_args = common.TrainingArguments

        def cpu_args(**kwargs):
            kwargs.update(bf16=False, use_cpu=True)
            return original_args(**kwargs)

        def tiny(*args, **kwargs):
            return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_embd=16, n_layer=1, n_head=2, n_positions=256))

        with patch.object(common, "TrainingArguments", cpu_args), patch("src.model.AutoModelForCausalLM.from_pretrained", tiny):
            cls.model = common.build(cfg)
        cls.common = common
        cls.row = dict(question="Start with 1 and add 1 five times.",
                       cot="<<1+1=2>> <<2+1=3>> <<3+1=4>> <<4+1=5>> <<5+1=6>>", answer="6")

    def run_forward(self, probability, backward=False):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        method = dict(probability=probability, mask_seed=405103)
        original_state_keys = set(model.state_dict())
        masker = attach_prefix_mask(model, method)
        labels, codi_inputs, decoder_inputs = [], [], []

        def record_labels(module, args, output):
            labels.append(args[1].detach().clone())

        def record_codi(module, args, kwargs):
            codi_inputs.append({key: value.detach().clone() for key, value in kwargs.items()
                                if key in ("input_ids", "inputs_embeds", "attention_mask") and value is not None})

        def record_decoder(module, args, kwargs):
            decoder_inputs.append(kwargs["inputs_embeds"].detach().clone())

        hooks = [model.loss_fct.register_forward_hook(record_labels),
                 model.codi.register_forward_pre_hook(record_codi, with_kwargs=True),
                 model.decoder.register_forward_pre_hook(record_decoder, with_kwargs=True)]
        try:
            result = model(**self.common.tensor_batch(self.row, model, "cpu"), step=0, step_ratio=0.)
            if backward:
                result["loss"].backward()
            self.assertEqual(set(model.state_dict()), original_state_keys)
            return {key: value.detach().clone() for key, value in result.items() if torch.is_tensor(value)}, labels, codi_inputs, decoder_inputs, masker.calls
        finally:
            masker.remove()
            for hook in hooks:
                hook.remove()

    def test_zero_probability_matches_unmodified_forward_and_gradients(self):
        model = self.model.train()
        batch = self.common.tensor_batch(self.row, model, "cpu")
        model.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        control = model(**batch, step=0, step_ratio=0.)
        control["loss"].backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        modified, _, _, _, calls = self.run_forward(0, backward=True)
        self.assertEqual(calls, 0)
        for key in ("loss", "ce_loss", "ref_ce_loss", "distill_loss", "explain_loss"):
            self.assertTrue(torch.equal(control[key].detach(), modified[key]), key)
        for name, parameter in model.named_parameters():
            if name in grads:
                self.assertTrue(torch.equal(parameter.grad, grads[name]), name)

    def test_all_five_slots_masked_without_changing_other_paths_or_labels(self):
        control, labels_a, inputs_a, decoder_a, _ = self.run_forward(0)
        modified, labels_b, inputs_b, decoder_b, calls = self.run_forward(1, backward=True)
        self.assertEqual(calls, 5)
        self.assertEqual(len(decoder_a), 5)
        self.assertEqual(len(labels_a), len(labels_b))
        for a, b in zip(labels_a, labels_b):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(inputs_a, inputs_b):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)
        for a, b in zip(decoder_a, decoder_b):
            self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
            self.assertFalse(torch.equal(a, b))
        for key in ("ce_loss", "ref_ce_loss", "distill_loss"):
            self.assertTrue(torch.equal(control[key], modified[key]), key)
        self.assertFalse(torch.equal(control["explain_loss"], modified["explain_loss"]))
        self.assertTrue(torch.isfinite(modified["loss"]))
        self.assertGreater(float(self.model.prj[1].weight.grad.abs().sum()), 0)

    def test_eval_generation_and_no_step_example_are_inert(self):
        model = self.model.eval()
        control = self.common.generate(model, self.row["question"], 4, 2, explanations=True)
        masker = attach_prefix_mask(model, dict(probability=.3, mask_seed=7))
        try:
            before = masker.state_dict()["generator"]
            modified = self.common.generate(model, self.row["question"], 4, 2, explanations=True)
            self.assertEqual(control, modified)
            self.assertTrue(torch.equal(before, masker.state_dict()["generator"]))
            model.train()
            row = dict(question="What is one?", cot="", answer="1")
            result = model(**self.common.tensor_batch(row, model, "cpu"), step=0, step_ratio=0.)
            self.assertEqual(float(result["explain_loss"]), 0)
            self.assertEqual(masker.calls, 0)
            self.assertTrue(torch.equal(before, masker.state_dict()["generator"]))
        finally:
            masker.remove()

    def test_checkpoint_restores_global_and_mask_rng_for_next_forward(self):
        model = self.model.train()
        masker = attach_prefix_mask(model, dict(probability=.3, mask_seed=7))
        batch = self.common.tensor_batch(self.row, model, "cpu")
        try:
            torch.manual_seed(11)
            model(**batch, step=0, step_ratio=0.)
            saved = io.BytesIO()
            torch.save(dict(model=model.state_dict(), rng=torch.get_rng_state(), mask=masker.state_dict()), saved)
            expected = model(**batch, step=1, step_ratio=1/750)["loss"].detach()
            saved.seek(0)
            state = torch.load(saved, weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            torch.set_rng_state(state["rng"])
            masker.load_state_dict(state["mask"])
            actual = model(**batch, step=1, step_ratio=1/750)["loss"].detach()
            self.assertTrue(torch.equal(expected, actual))
        finally:
            masker.remove()


if __name__ == "__main__":
    unittest.main()
