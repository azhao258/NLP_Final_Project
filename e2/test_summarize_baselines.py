"""Check current data selection, reproducibility, and refusal of misaligned IDs."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from summarize_baselines import ROOT, collect, percentile


class BaselineTests(unittest.TestCase):
    def test_quantile_definition(self):
        self.assertAlmostEqual(percentile([0, 10, 20, 30], .95), 28.5)

    def test_current_sources_and_repeatability(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            command = [sys.executable, str(ROOT/'e2/summarize_baselines.py'), '--output', name]
            subprocess.run(command, check=True, capture_output=True)
            self.assertFalse((output/'PART1_REPORT.md').exists())
            with (output/'part1_baseline_table.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            svamp = [r for r in rows if r['dataset']=='svamp']
            self.assertEqual([int(r['correct']) for r in svamp], [88,85,2])
            self.assertTrue(all(float(r['mean_reference_tokens'])==10.855 for r in svamp))
            self.assertEqual(svamp[0]['source_dir'], 'e1_results/codi_svamp_course_test')
            self.assertEqual(svamp[1]['source_dir'], 'e1_results/sim_svamp_course_test')
            with (output/'paired_comparison.csv').open(encoding='utf-8-sig', newline='') as f:
                paired = list(csv.DictReader(f))[-1]
            self.assertEqual([int(paired[k]) for k in ['both_correct','codi_wrong_sim_correct','codi_correct_sim_wrong','both_wrong']], [76,9,12,103])
            # Final reports are maintained independently and must survive reruns.
            (output/'PART1_REPORT.md').write_bytes(b'# Final report\nUser-edited text.\n')
            before = {p.name:p.read_bytes() for p in output.iterdir()}
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(before, {p.name:p.read_bytes() for p in output.iterdir()})

    def test_reject_wrong_prediction_id(self):
        _,_,provenance = collect()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in provenance['inputs']:
                target = root/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            path = root/'e1_results/codi_full/gsm8k.predictions.jsonl'
            lines = path.read_text(encoding='utf8').splitlines()
            row = json.loads(lines[0]);row['id']='wrong-id'
            lines[0] = json.dumps(row)
            path.write_text('\n'.join(lines)+'\n',encoding='utf8')
            with self.assertRaises(AssertionError):
                collect(root)


if __name__ == '__main__':
    unittest.main()
