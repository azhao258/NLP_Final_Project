"""Real GPT-2 token spans and independently calculated weighted CE."""
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from e5.p2.p2_operand import OperandWeight, operand_spans
from e5.p2.p2_protocol import preflight

ROOT = Path(__file__).resolve().parents[2]


class OperandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / 'assets/gpt2', local_files_only=True)

    def test_full_spans_and_merged_tail(self):
        text = '<<-12345.67+2=-12343.67>><<.25*4=1>><<(+1,234)/2=617>>'
        self.assertEqual([text[a:b] for a,b in operand_spans(text)], ['-12345.67', '.25', '+1,234'])
        ids = self.tok.encode(text) + [self.tok.eos_token_id, -100]
        weights = OperandWeight(self.tok).token_weights(tuple(ids))
        # Independently locate character overlaps using the fast tokenizer offsets.
        offsets = self.tok(text, return_offsets_mapping=True)['offset_mapping']
        spans = [(text.index(x), text.index(x)+len(x)) for x in ['-12345.67','.25','+1,234']]
        expected = [2. if any(a<d and b>c for c,d in spans) else 1. for a,b in offsets] + [1.,0.]
        self.assertEqual(list(weights), expected)
        self.assertGreater(sum(w==2 for w in weights), 3)

    def test_weighted_value_and_gradient(self):
        ids = self.tok.encode('<<-12345.67+2=-12343.67>>') + [-100]
        labels = torch.tensor(ids)
        logits = torch.randn(len(ids), 50257, requires_grad=True)
        policy = OperandWeight(self.tok)
        weights = torch.tensor(policy.token_weights(tuple(ids)))
        actual = policy.weighted_ce(logits, labels)
        ce = -logits.log_softmax(-1)[torch.arange(len(ids)-1), labels[:-1]]
        expected = (ce*weights[:-1]).sum()/weights.sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected_grad = logits.detach().softmax(-1)
        expected_grad[torch.arange(len(ids)-1), labels[:-1]] -= 1
        expected_grad *= weights[:,None]/weights.sum()
        torch.testing.assert_close(logits.grad, expected_grad)
        uniform = OperandWeight(self.tok, 1).weighted_ce(logits, labels)
        torch.testing.assert_close(uniform, F.cross_entropy(logits, labels))

    def test_protocol(self):
        result = preflight()
        self.assertEqual(result['full_budget_updates'],750)
        self.assertEqual(result['variant']['operand_weight'],2)
        self.assertEqual(result['config']['explain_loss_factor'],1)

    def test_reject_unsupported(self):
        with self.assertRaises(ValueError):
            operand_spans('<<unknown+2=3>>')
        for text in ('<<x=5>>','<<X/2=0.5X>>'):
            self.assertEqual(operand_spans(text),[])
            self.assertTrue(all(w==1 for w in OperandWeight(self.tok).token_weights(tuple(self.tok.encode(text)))))

    def test_unicode_byte_bpe(self):
        text = '<<12×2=24>><<3÷2=1.5>>'
        policy = OperandWeight(self.tok)
        ids = tuple(self.tok.encode(text))
        weights = policy.token_weights(ids)
        selected = [i for i,w in zip(ids,weights) if w==2]
        self.assertEqual(self.tok.decode(selected),'123')


if __name__ == '__main__':
    unittest.main()
