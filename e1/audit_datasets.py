"""CPU-only comparison of legacy E1 data and pinned course sources."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def main():
    audit = HERE / "audit"
    hard = jsonl(audit / "course_hard.jsonl")
    hard_index = {x["input"]: x for x in hard}
    old_hard = pd.read_parquet(HERE / "raw/gsm_hard.parquet").to_dict("records")
    assert len(hard_index) == len(hard) == len(old_hard) == 1319
    for row in old_hard:
        official = hard_index[row["instruction"]]
        assert Fraction(str(row["response"])) == Fraction(str(official["target"]))

    splits = {name: pd.read_parquet(audit / f"svamp_{name}.parquet").to_dict("records")
              for name in ("train", "test")}
    split_ids = {name: {x["ID"] for x in rows} for name, rows in splits.items()}
    assert len(split_ids["train"]) == 700 and len(split_ids["test"]) == 300
    assert not split_ids["train"] & split_ids["test"]
    svamp_index = {x["ID"]: x for rows in splits.values() for x in rows}
    old_svamp = json.loads((HERE / "raw/svamp.json").read_text(encoding="utf8"))
    assert len(old_svamp) == len(svamp_index) == 1000
    for row in old_svamp:
        official = svamp_index[row["ID"]]
        for key in ("Body", "Question", "Equation", "Type"):
            assert row[key] == official[key], (row["ID"], key)
        assert Fraction(str(row["Answer"])) == Fraction(str(official["Answer"]))
    old_selected = jsonl(audit / "legacy_data/svamp.jsonl")
    selected = jsonl(HERE / "data/svamp.jsonl")
    assert len(selected) == len({x["id"] for x in selected}) == 200
    assert all(x["id"] in split_ids["test"] for x in selected)
    for row in selected:
        official = svamp_index[row["id"]]
        assert row["question"] == official["question_concat"]
        assert Fraction(row["answer"]) == Fraction(str(official["Answer"]))
    unchanged = {}
    for name in ("gsm8k", "gsm_hard"):
        data = (HERE / f"data/{name}.jsonl").read_bytes()
        assert data == (audit / f"legacy_data/{name}.jsonl").read_bytes()
        unchanged[name] = hashlib.sha256(data).hexdigest()
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    for name, info in manifest["datasets"].items():
        assert hashlib.sha256((HERE / f"data/{name}.jsonl").read_bytes()).hexdigest() == info["sha256"]
    for name, expected in manifest["course_raw_sha256"].items():
        assert hashlib.sha256((audit / name).read_bytes()).hexdigest() == expected
    report = dict(
        status="PASS", hard_question_and_numeric_gold_matches=1319,
        svamp_full_union_field_matches=1000,
        legacy_svamp_split_counts={name: sum(x["id"] in ids for x in old_selected)
                                   for name, ids in split_ids.items()},
        corrected_svamp_test_count=len(selected),
        svamp_old_new_overlap=len({x["id"] for x in selected} & {x["id"] for x in old_selected}),
        unchanged_dataset_sha256=unchanged,
        note="SVAMP test-only is our evaluation policy; the assignment specifies fixed subsets without explicitly naming this split.")
    (audit / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
