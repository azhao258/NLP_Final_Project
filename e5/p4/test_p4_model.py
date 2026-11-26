"""Tiny real CODI integration: unchanged base losses/RNG, live student gradients."""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import torch
import e3.common as common
from transformers import GPT2Config, GPT2LMHeadModel
from e5.p4.p4_alignment import training_forward
from e5.p4.p4_audit import audit_rows
from e5.protocol import read_json


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cfg=read_json(common.ROOT/'e3/config.json')
        cfg.update(projection_dim=16,lora_r=2)
        original=common.TrainingArguments
        def cpu_args(**kw):
            return original(**dict(kw,bf16=False,use_cpu=True))
        def tiny(*args,**kw):
            return GPT2LMHeadModel(GPT2Config(vocab_size=50257,n_embd=16,n_layer=1,n_head=2,n_positions=256))
        # Vendored CODI uses explicit CUDA autocast, so disable it for CPU parity.
        cls.autocast_patch=patch('src.model.autocast',lambda **kw:contextlib.nullcontext())
        cls.autocast_patch.start()
        cls.addClassCleanup(cls.autocast_patch.stop)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(common,'TrainingArguments',cpu_args), patch('src.model.AutoModelForCausalLM.from_pretrained',tiny):
            cls.model=common.build(cfg).train()
        cls.row=dict(id='synthetic',question='Compute.',cot=' '.join(f'<<{i}+1={i+1}>>' for i in range(1,8)),answer='8')
        cls.batch=common.tensor_batch(cls.row,cls.model,'cpu')

    def test_full_forward_parity_gradients_and_rng(self):
        m=self.model.train()
        keys=set(m.state_dict())
        def run(factor):
            m.zero_grad(set_to_none=True)
            torch.manual_seed(31)
            result=m(**self.batch,step=0,step_ratio=0) if factor is None else training_forward(m,self.batch,0,factor)
            result['loss'].backward()
            grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
            return result,grads,torch.get_rng_state()
        control,cg,cr=run(None)
        zero,zg,zr=run(0)
        aligned,ag,ar=run(.1)
        for name in ('loss','ce_loss','ref_ce_loss','distill_loss','explain_loss'):
            torch.testing.assert_close(control[name],zero[name],rtol=0,atol=0)
        for name in cg: torch.testing.assert_close(cg[name],zg[name],rtol=0,atol=0)
        for name in ('ce_loss','ref_ce_loss','distill_loss','explain_loss'):
            torch.testing.assert_close(control[name],aligned[name],rtol=0,atol=0)
        torch.testing.assert_close(cr,zr,rtol=0,atol=0)
        torch.testing.assert_close(cr,ar,rtol=0,atol=0)
        torch.testing.assert_close(aligned['loss'],control['loss']+.1*aligned['alignment_loss'])
        self.assertEqual(aligned['alignment_valid_slots'].item(),5)
        for prefix in ('prj.','codi.'):
            self.assertTrue(any(not torch.equal(cg[n],ag[n]) for n in cg if n.startswith(prefix)))
        for n in cg:
            if n.startswith('decoder.'): torch.testing.assert_close(cg[n],ag[n],rtol=0,atol=0)
        self.assertEqual(keys,set(m.state_dict()))
        m.load_state_dict(m.state_dict(),strict=True)
        self.assertEqual(len(m.codi._forward_hooks),0)
        self.assertEqual(len(m.prj._forward_hooks),0)

    def test_audit_and_no_steps(self):
        self.assertEqual(audit_rows([self.row],self.model.tokenizer)['merged_tail_rows'],1)
        row=dict(question='Compute one.',cot='',answer='1')
        result=training_forward(self.model,common.tensor_batch(row,self.model,'cpu'),0)
        self.assertEqual(result['alignment_loss'].item(),0)
        self.assertEqual(result['alignment_valid_slots'].item(),0)

    def test_cleanup_on_exception(self):
        with patch.object(self.model,'forward',side_effect=RuntimeError('intentional')):
            with self.assertRaises(RuntimeError): training_forward(self.model,self.batch,0)
        self.assertEqual(len(self.model.codi._forward_hooks),0)
        self.assertEqual(len(self.model.prj._forward_hooks),0)

    def test_resume_reproducibility(self):
        m=self.model.train()
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/38)
        def step(i):
            opt.zero_grad(set_to_none=True)
            training_forward(m,self.batch,i)['loss'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step(); sched.step()
        step(0)
        snapshot=copy.deepcopy((m.state_dict(),opt.state_dict(),sched.state_dict(),torch.get_rng_state()))
        step(1)
        expected=copy.deepcopy(m.state_dict())
        expected_schedule=sched.state_dict()
        m.load_state_dict(snapshot[0]); opt.load_state_dict(snapshot[1]); sched.load_state_dict(snapshot[2]); torch.set_rng_state(snapshot[3])
        step(1)
        for n,p in m.state_dict().items(): torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
        self.assertEqual(sched.state_dict(),expected_schedule)


if __name__ == '__main__':
    unittest.main()
