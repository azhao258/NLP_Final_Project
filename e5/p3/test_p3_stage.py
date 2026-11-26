import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e5.p3.p3_stage import stage, training_forward, preflight


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.codi = Core()
        self.prj = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 8)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.ref_loss_factor = 1.0

    def forward(self, **batch):
        x = self.codi.emb(batch['ref_input_ids'])
        loss = self.decoder(self.prj(x)).square().mean()
        return dict(loss=loss, ce_loss=loss.detach())


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)
        self.drop = torch.nn.Dropout(.2)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.drop(self.emb(input_ids))))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Toy()
        self.batch = dict(ref_input_ids=torch.tensor([[1, 2, 3, 4]]),
                          ref_labels=torch.tensor([[-100, -100, 3, 4]]),
                          ref_attention_mask=torch.ones(1, 4, dtype=torch.long))

    def test_boundaries(self):
        self.assertEqual([stage(i) for i in (0, 149, 150, 749)],
                         ['reference_only', 'reference_only', 'joint', 'joint'])
        for i in (-1, 750, 1.5, True):
            with self.assertRaises(ValueError): stage(i)

    def test_reference_mask_shift_and_inactive_adamw(self):
        m = self.model.eval()
        before = {n: p.detach().clone() for n,p in m.named_parameters()}
        logits = m.codi(**{k: self.batch['ref_'+k] for k in ('input_ids', 'attention_mask')}).logits
        expected = torch.nn.functional.cross_entropy(logits[0, 1:3], torch.tensor([3,4]))
        result = training_forward(m, self.batch, 149)
        torch.testing.assert_close(result['loss'], expected)
        opt = torch.optim.AdamW(m.parameters(), lr=.01, weight_decay=.1)
        opt.zero_grad(set_to_none=True)
        result['loss'].backward()
        opt.step()
        for n,p in m.named_parameters():
            if n.startswith(('prj.', 'decoder.')):
                self.assertIsNone(p.grad)
                self.assertNotIn(p, opt.state)
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        self.assertFalse(torch.equal(before['codi.head.weight'], m.codi.head.weight))

    def test_joint_and_resume_across_transition(self):
        m = self.model
        opt = torch.optim.AdamW(m.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1-n/750)
        def update(i):
            opt.zero_grad(set_to_none=True)
            training_forward(m, self.batch, i)['loss'].backward()
            opt.step(); sched.step()
        update(148)
        state = copy.deepcopy((m.state_dict(), opt.state_dict(), sched.state_dict(), torch.get_rng_state()))
        update(149); update(150)
        expected = copy.deepcopy(m.state_dict())
        end_sched = copy.deepcopy(sched.state_dict())
        m.load_state_dict(state[0]); opt.load_state_dict(state[1]); sched.load_state_dict(state[2]); torch.set_rng_state(state[3])
        update(149); update(150)
        for n,p in m.state_dict().items(): torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
        self.assertEqual(sched.state_dict(), end_sched)
        self.assertIsNotNone(m.prj.weight.grad)
        m.eval()
        torch.testing.assert_close(training_forward(m, self.batch, 150)['loss'], m(**self.batch)['loss'])

    def test_frozen_protocol(self):
        result = preflight()
        self.assertEqual(result['full_budget_updates'], 750)
        self.assertEqual(result['variant']['reference_updates'], 150)


if __name__ == '__main__':
    unittest.main()
