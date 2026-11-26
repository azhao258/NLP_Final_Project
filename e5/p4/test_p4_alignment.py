import unittest
import torch
from types import SimpleNamespace
from e5.p4.p4_alignment import step_spans, batch_positions, alignment_loss


class MappingTests(unittest.TestCase):
    def test_offsets_padding_merge_and_stop(self):
        ids = [9] + [16791, 2, 4211] * 7 + [50256, 16791, 3, 4211]
        groups = step_spans(ids)
        self.assertEqual(groups, [[(1,3)],[(4,6)],[(7,9)],[(10,12)],[(13,15),(16,18),(19,21)]])
        self.assertEqual(step_spans([9959,8,4211]), [[(0,2)],[],[],[],[]])
        self.assertEqual(step_spans([16791,8,50257,4211]), [[]]*5)
        self.assertEqual(step_spans([9,8]), [[]]*5)

    def test_mask_and_answer_boundaries(self):
        tok = SimpleNamespace(encode=lambda s:[1,2], eos_token_id=50256, pad_token_id=50257)
        batch = dict(ref_input_ids=torch.tensor([[9,16791,2,4211,7,8,6,50256]]),
            ref_labels=torch.tensor([[-100,16791,2,4211,7,8,6,50256]]), ref_answer_position=torch.tensor([6]))
        self.assertEqual(batch_positions(batch,tok).tolist(), [[3,-1,-1,-1,-1]])
        batch['ref_labels'][0,1] = -100
        with self.assertRaises(ValueError): batch_positions(batch,tok)
        batch['ref_labels'][0,1] = 16791
        batch['ref_answer_position'][0] = 5
        with self.assertRaises(ValueError): batch_positions(batch,tok)

    def test_gradients_detach_mask_and_scale(self):
        torch.manual_seed(5)
        student = torch.randn(2,5,8,requires_grad=True)
        teacher = torch.randn(2,12,8,requires_grad=True)
        positions = torch.tensor([[1,5,9,-1,-1],[2,-1,-1,-1,-1]])
        loss = alignment_loss(student,teacher,positions)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(),0)
        self.assertLessEqual(loss.item(),2)
        torch.testing.assert_close(loss,alignment_loss(student*3,teacher*7,positions))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[positions<0].abs().sum(),0)
        self.assertGreater(student.grad[positions>=0].abs().sum(),0)
        changed = teacher.detach().clone()
        changed[:,11] += 10000
        torch.testing.assert_close(loss,alignment_loss(student,changed,positions))

    def test_no_valid_and_bounds(self):
        student = torch.randn(1,5,8,requires_grad=True)
        teacher = torch.randn(1,10,8,requires_grad=True)
        positions = torch.full((1,5),-1)
        loss = alignment_loss(student,teacher,positions)
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertEqual(student.grad.abs().sum(),0)
        self.assertIsNone(teacher.grad)
        positions[0,0]=10
        with self.assertRaises(ValueError): alignment_loss(student,teacher,positions)


if __name__ == '__main__':
    unittest.main()
