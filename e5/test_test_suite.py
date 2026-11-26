"""Reject corrupted generic exports without running model inference."""
import copy
import json
import unittest

from e5.analyze_test_suite import ROOT, check_rows
from e5.protocol import read_rows


class GenericExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = read_rows(ROOT/'e1/data/gsm8k.jsonl')
        directory = ROOT/'server_backup_eval_20261126/e5_runs/p1_test_bf16_20261126'
        cls.files = {f'gsm8k.{kind}.jsonl': (directory/f'gsm8k.{kind}.jsonl').read_bytes()
                     for kind in ('predictions', 'decoded')}

    def modified(self, kind, edit):
        files = copy.copy(self.files)
        name = f'gsm8k.{kind}.jsonl'
        rows = [json.loads(line) for line in files[name].splitlines()]
        edit(rows)
        files[name] = '\n'.join(json.dumps(r) for r in rows).encode('utf8')
        return files

    def test_reordered_predictions_rejected(self):
        files = self.modified('predictions', lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, 'IDs'):
            check_rows('p1', 'gsm8k', files, self.gold)

    def test_wrong_answer_score_rejected(self):
        files = self.modified('predictions', lambda rows: rows[0].update(correct=not rows[0]['correct']))
        with self.assertRaisesRegex(ValueError, 'answer mismatch'):
            check_rows('p1', 'gsm8k', files, self.gold)

    def test_missing_explanation_slot_rejected(self):
        files = self.modified('decoded', lambda rows: rows[0]['decoded_steps'].pop())
        with self.assertRaisesRegex(ValueError, 'slots mismatch'):
            check_rows('p1', 'gsm8k', files, self.gold)

    def test_wrong_reference_rejected(self):
        files = self.modified('decoded', lambda rows: rows[0].update(reference='<<1+1=2>>'))
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            check_rows('p1', 'gsm8k', files, self.gold)


if __name__ == '__main__':
    unittest.main()
