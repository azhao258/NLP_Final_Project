"""Metric edge cases and complete frozen-export regression checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze import compare, equation, normalize


class EquationTests(unittest.TestCase):
    def test_commutative_not_strict(self):
        result = compare(' <<3*2=6>>', '<<2*3=6>><|endoftext|>')
        self.assertFalse(result['strict_match'])
        self.assertTrue(result['conservative_equivalent'])

    def test_same_result_different_reasoning(self):
        result = compare('<<2*15=30>>', '<<300/10=30>>')
        self.assertTrue(result['result_match'])
        self.assertFalse(result['conservative_equivalent'])

    def test_noncommutative(self):
        self.assertFalse(compare('<<2-3=-1>>', '<<3-2=1>>')['conservative_equivalent'])

    def test_decimal_exactness(self):
        self.assertTrue(equation('<<.1+.2=.3>>')['arithmetic_valid'])
        self.assertFalse(equation('<<40/3=13.33>>')['arithmetic_valid'])

    def test_repeated_operands(self):
        self.assertFalse(compare('<<2+3=5>>', '<<2+2+3=7>>')['operand_match'])

    def test_merged_slot(self):
        result = compare('<<3*2=6>>', '<<58+175=233>> <<233/4=58.25>>')
        self.assertTrue(result['reference_equation']['parsed'])
        self.assertEqual(result['missing_equation_count'], 1)
        self.assertFalse(result['conservative_equivalent'])

    def test_reject_unsafe_and_undefined(self):
        for text in ["<<__import__('os').system('echo no')=0>>", '<<1/0=1>>', '[PAD]', '', '<<2**100=2>>']:
            self.assertFalse(equation(text)['parsed'])

    def test_normalization(self):
        self.assertEqual(normalize(' <<2 * 3=6>><|endoftext|>'), '<<2*3=6>>')


class FrozenExportTests(unittest.TestCase):
    def test_full_analysis_and_reproducibility(self):
        script = Path(__file__).with_name('analyze.py')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            command = [sys.executable, str(script), '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertFalse((output / 'REPORT.md').exists())
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual((summary['n'], summary['correct'], summary['valid_reference_slots'], summary['strict_matches']), (200, 6, 478, 6))
            self.assertEqual(summary['conservative_matches'], 9)
            rows = [json.loads(s) for s in (output / 'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1000)
            self.assertEqual(sum(not r['valid_reference'] for r in rows), 522)
            self.assertTrue(all('strict_match' not in r and not r['labels'] for r in rows if not r['valid_reference']))
            self.assertEqual(summary['missing_equations_in_merged_slots'], 5)
            self.assertEqual(summary['question_label_counts']['correct_answer_explanation_mismatch'], 6)
            self.assertEqual(summary['slot_label_counts']['reference_arithmetic_inconsistency'], 7)
            # Existing reports are maintained separately and must survive reruns.
            (output / 'REPORT.md').write_bytes(b'# Final report\nUser-edited text.\n')
            before = {p.name:p.read_bytes() for p in output.iterdir()}
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(before, {p.name:p.read_bytes() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
