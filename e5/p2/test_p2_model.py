"""Run in frozen E3 environment: real CODI forward with tiny random GPT-2."""
import unittest
from unittest.mock import patch
import torch
import e3.common as common
from transformers import GPT2Config, GPT2LMHeadModel
from e5.p2.p2_operand import OperandWeight
from e5.protocol import read_json


class ModelTests(unittest.TestCase):
    def test_forward_isolation_gradients_and_reload(self):
        torch.set_num_threads(2)
        cfg = read_json(common.ROOT/'e3/config.json')
        cfg.update(projection_dim=16,lora_r=2)
        original = common.TrainingArguments
        def cpu_args(**kw):
            return original(**dict(kw,bf16=False,use_cpu=True))
        def tiny(*args,**kwargs):
            return GPT2LMHeadModel(GPT2Config(vocab_size=50257,n_embd=16,n_layer=1,n_head=2,n_positions=256))
        with patch.object(common,'TrainingArguments',cpu_args), patch('src.model.AutoModelForCausalLM.from_pretrained',tiny):
            model = common.build(cfg).train()
        row = dict(question='Compute the result.',cot='<<-123.45+124.45=1>> <<1+1=2>> <<2+1=3>> <<3+1=4>> <<4+1=5>> <<5+1=6>>',answer='6')
        batch = common.tensor_batch(row,model,'cpu')
        state_keys = set(model.state_dict())
        def run(factor):
            model.zero_grad(set_to_none=True)
            policy = OperandWeight(model.tokenizer,factor).attach(model) if factor else None
            records = []
            h = model.decoder.register_forward_pre_hook(lambda m,a,k: records.append(k['inputs_embeds'].detach().clone()),with_kwargs=True)
            try:
                torch.manual_seed(31)
                result = model(**batch,step=0,step_ratio=0.)
                result['loss'].backward()
                grads = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                if policy:
                    self.assertEqual(policy.calls,5)
                return result,grads,records
            finally:
                h.remove()
                if policy:
                    policy.remove()
        control,cg,ci = run(0)
        uniform,ug,ui = run(1)
        for key in ('loss','explain_loss','ce_loss','ref_ce_loss','distill_loss'):
            torch.testing.assert_close(control[key],uniform[key])
        for name in cg:
            torch.testing.assert_close(cg[name],ug[name])
        weighted,wg,wi = run(2)
        for a,b in zip(ci,wi):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for key in ('ce_loss','ref_ce_loss','distill_loss'):
            torch.testing.assert_close(control[key],weighted[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(control['explain_loss'],weighted['explain_loss']))
        self.assertTrue(any(not torch.equal(cg[n],wg[n]) for n in cg if n.startswith('codi.')))
        self.assertEqual(set(model.state_dict()),state_keys)
        model.load_state_dict(model.state_dict(),strict=True)
        repeated,rg,_ = run(2)
        torch.testing.assert_close(weighted['loss'],repeated['loss'],rtol=0,atol=0)
        for n in wg:
            torch.testing.assert_close(wg[n],rg[n],rtol=0,atol=0)
        # Empty supervision must not arm a loss hook; eval generation stays inert.
        policy = OperandWeight(model.tokenizer).attach(model)
        try:
            empty = dict(question='What is one?',cot='',answer='1')
            result = model(**common.tensor_batch(empty,model,'cpu'),step=0,step_ratio=0.)
            self.assertEqual(float(result['explain_loss']),0)
            self.assertEqual(policy.calls,0)
            model.eval()
            a = common.generate(model,'One?',4,2,explanations=True)
            self.assertEqual(policy.calls,0)
        finally:
            policy.remove()
        b = common.generate(model,'One?',4,2,explanations=True)
        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()
