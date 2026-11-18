"""Recompute handoff accuracy and strict decoded-step counts without a GPU."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r"\s+", "", text.replace("<|endoftext|>", ""))


def main():
    results = {}
    for version, directory in [("v1", "control_diagnostics"),
                               ("v2", "control_v2_validation"),
                               ("v3", "control_v3_validation")]:
        rows = [json.loads(line) for line in
                (ROOT / "e3_runs" / directory / "validation.decoded.jsonl")
                .read_text(encoding="utf8").splitlines()]
        matched = total = all_matched = 0
        for row in rows:
            checks = [normalize(step["text"]) == normalize(reference)
                      for step, reference in zip(row["decoded_steps"], row["supervision_slots"])
                      if reference != "[PAD]"]
            matched += sum(checks)
            total += len(checks)
            all_matched += bool(checks) and all(checks)
        results[version] = dict(n=len(rows), correct=sum(row["correct"] for row in rows),
                                matched_steps=matched, evaluated_steps=total,
                                all_steps_matched=all_matched)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
