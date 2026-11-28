"""Synthetic exports live only in temporary directories, never experimental results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from e5.protocol import ROOT, DATASETS, preflight, read_json, read_rows, sample_order, sha, validate_checkpoint, write_json
from e5.problem_types import classify, make_types, reference_slots
from E7.scoring import POLICY, analyze, inspect_steps
from e1.metrics import extract


def decoded(texts):
    return [dict(slot=i, consumed_by_reasoner=i < 4, text=texts[i] if i < len(texts) else "<<1+1=2>>", capped=False)
            for i in range(5)]


class ProtocolTests(unittest.TestCase):
    def test_budget_and_original_sources(self):
        p = preflight()
        order = sample_order(p["config"])
        self.assertEqual(len(order), 750)
        for start in (0, 250, 500):
            self.assertEqual(sorted(i for batch in order[start:start+250] for i in batch), list(range(2000)))
        self.assertEqual(len({tuple(batch) for batch in order}), 750)
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake.pt"
            fake.write_bytes(b"wrong initialization")
            with self.assertRaisesRegex(ValueError, "original shared initialization"):
                preflight(fake, require_weights=True)
            with self.assertRaisesRegex(ValueError, "Restore"):
                preflight(Path(directory) / "missing.pt", require_weights=True)

    def test_incomplete_wrong_init_and_variant_are_rejected(self):
        p = preflight()
        state = dict(config=p["config"], updates=750, init_sha256=p["initialization_sha256"], prefix_mask_rng={},
                     contract=dict(pilot_updates=0, full_budget_updates=750, variant=p["variant"],
                                   order_sha256=p["order_sha256"], data_files_sha256=p["data_files_sha256"]))
        validate_checkpoint(state, "test", "modified", p)
        for key, value in (("updates", 749), ("init_sha256", "other")):
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, "test", "modified", p)
        bad = copy.deepcopy(state)
        bad["contract"]["variant"]["probability"] = .9
        with self.assertRaises(ValueError):
            validate_checkpoint(bad, "test", "modified", p)

    def test_frozen_type_manifest_and_surface_labels(self):
        self.assertEqual(make_types(), read_json(ROOT / "e5/problem_types.json"))
        labels = classify(dict(dataset="gsm_hard", question="Use 1,234.5 items and 2."))
        self.assertIn(dict(axis="max_written_number", label="over_1000"), labels)
        self.assertNotIn("operation_family", [x["axis"] for x in labels])


class StepTests(unittest.TestCase):
    def test_merged_gsm_steps_and_missing_reference(self):
        row = dict(dataset="gsm8k", reference=" ".join(f"<<{i}+1={i+1}>>" for i in range(6)))
        refs = reference_slots(row)
        self.assertEqual(len(refs), 5)
        values = inspect_steps(row, decoded(refs))
        self.assertTrue(all(s["strict_match"] for s in values))
        missing = inspect_steps(dict(dataset="gsm_hard", reference=None), decoded([]))
        self.assertTrue(all(s["strict_match"] is None and s["reference"] is None for s in missing))
        self.assertTrue(all(s["arithmetic_valid"] for s in missing))

    def test_svamp_whole_expression_is_not_a_step_chain(self):
        row = dict(dataset="svamp", reference="((7 + 12) - 15)", answer="4")
        result = inspect_steps(row, decoded(["<<7+12-15=4>>", "<<7+12=19>>"]))
        self.assertTrue(result[0]["whole_expression_compatible"])
        self.assertFalse(result[1]["whole_expression_compatible"])
        self.assertTrue(all(s["reference"] is None and s["strict_match"] is None for s in result))


def fixture(parent):
    p = preflight()
    for role in ("control", "modified"):
        directory = parent / role
        directory.mkdir()
        config = dict(role=role, pairing_id="synthetic-test-only", policy=POLICY, train_updates=750,
            common_config=p["config"], initialization_sha256=p["initialization_sha256"],
            data_files_sha256=p["data_files_sha256"], types_sha256=sha(ROOT / "e5/problem_types.json"),
            checkpoint_sha256=read_json(ROOT / "e3/frozen_v1.json")["checkpoint_sha256"] if role == "control" else "synthetic-modified",
            training_contract=dict(pilot_updates=0, full_budget_updates=750, variant=p["variant"],
                order_sha256=p["order_sha256"], data_files_sha256=p["data_files_sha256"]),
            variant=p["variant"] if role == "modified" else None,
            source_sha256={"synthetic": "only"}, gpu="synthetic", torch="synthetic", transformers="synthetic", peft="synthetic")
        write_json(directory / "config.json", config)
        for name in DATASETS:
            predictions, explanations = [], []
            for index, row in enumerate(read_rows(ROOT / f"e1/data/{name}.jsonl")):
                correct = index != (0 if role == "control" else 1)
                raw = "The answer is: " + (str(row["answer"]) if correct else "999999999999999")
                answer = extract(raw)
                pred = dict(row, raw_output=raw, prediction=answer, correct=correct, latent_steps=4,
                    latency_seconds=.1 if role == "control" else .11, reference_tokens=20 if row.get("reference") else None,
                    answer_tokens=2, generated_tokens=3, stop_reason="eos")
                predictions.append(pred)
                explanations.append(dict(id=row["id"], dataset=name, question=row["question"], gold=row["answer"],
                    reference=row.get("reference"), prediction=answer, correct=correct,
                    decoded_steps=decoded(reference_slots(row))))
            for suffix, rows in (("predictions", predictions), ("decoded", explanations)):
                (directory / f"{name}.{suffix}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf8")
        (directory / "summary.csv").write_text("synthetic fixture; not results\n")
        seal(directory)


def seal(directory):
    write_json(directory / "complete.json", dict(n=600, pairing_id="synthetic-test-only",
        files={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != "complete.json"}))


class ExportTests(unittest.TestCase):
    def test_paired_metrics_full_600_without_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            fixture(parent)
            output = parent / "analysis"
            summary = analyze(parent / "control", parent / "modified", output)
            self.assertEqual(len(summary), 6)
            self.assertTrue(all(s["n"] == 200 and s["correct"] == 199 for s in summary))
            groups = read_rows(output / "problem_types.jsonl")
            self.assertTrue(all(s["gained"] == s["lost"] == 1 for s in groups if s["axis"] == "dataset"))
            self.assertEqual(len(read_rows(output / "steps.jsonl")), 6000)
            self.assertTrue(all(s["conservative_equivalent_rate"] is None for s in summary if s["dataset"] == "gsm_hard"))
            self.assertFalse((output / "REPORT.md").exists())
            with self.assertRaisesRegex(ValueError, "new analysis"):
                analyze(parent / "control", parent / "modified", output)

    def test_tampered_ids_and_pairing_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            fixture(parent)
            path = parent / "modified/gsm8k.predictions.jsonl"
            original = path.read_text(encoding="utf8")
            rows = read_rows(path)
            rows[0]["id"] = rows[1]["id"]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "modified after completion"):
                analyze(parent / "control", parent / "modified", parent / "out")
            seal(parent / "modified")
            with self.assertRaisesRegex(ValueError, "evaluation IDs"):
                analyze(parent / "control", parent / "modified", parent / "out")
            path.write_text(original, encoding="utf8")
            cfg_path = parent / "modified/config.json"
            cfg = read_json(cfg_path)
            cfg["gpu"] = "other-device"
            write_json(cfg_path, cfg)
            seal(parent / "modified")
            with self.assertRaisesRegex(ValueError, "Unmatched evaluation"):
                analyze(parent / "control", parent / "modified", parent / "out")


if __name__ == "__main__":
    unittest.main()
