"""Freeze prediction-independent problem strata for all 600 test questions."""
import argparse
from collections import Counter
from decimal import Decimal
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.protocol import ROOT, DATASETS, preflight, read_rows, require, sha, write_json
from e4.analyze import equation, expression

POLICY = "v1: reference operation families / GSM8K equation count / SVAMP native Type / question numeric magnitude"


def reference_slots(row):
    if row["dataset"] != "gsm8k":
        return []
    refs = re.findall(r"<<[^<>]*>>", row.get("reference") or "")
    return refs[:4] + ["".join(refs[4:])] if len(refs) > 5 else refs


def classify(row):
    labels = []
    name = row["dataset"]
    if name == "svamp":
        labels.append(dict(axis="native_type", label=row.get("problem_type") or "unknown"))
    operators = []
    parsed = True
    if name == "gsm8k":
        refs = re.findall(r"<<[^<>]*>>", row.get("reference") or "")
        labels.append(dict(axis="reference_steps", label="missing" if not refs else "one" if len(refs) == 1 else "two" if len(refs) == 2 else "three_or_more"))
        for ref in refs:
            value = equation(ref)
            parsed &= value["parsed"]
            if value["parsed"]:
                operators.extend(value["lhs"]["operators"])
        parsed &= bool(refs)
    elif name == "svamp":
        try:
            operators = expression(row["reference"].strip())["operators"]
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
            parsed = False
    else:
        parsed = False
    if name != "gsm_hard":
        has_as, has_md = bool(set(operators) & {"+", "-"}), bool(set(operators) & {"*", "/"})
        family = "unparsed" if not parsed else "mixed" if has_as and has_md else "add_subtract" if has_as else "multiply_divide" if has_md else "no_operation"
        labels.append(dict(axis="operation_family", label=family))
    # This is a lexical problem characteristic, NOT a semantic operation label.
    numbers = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", row["question"])
    magnitudes = [abs(Decimal(x.replace(",", ""))) for x in numbers]
    maximum = max(magnitudes) if magnitudes else None
    size = "no_digits" if maximum is None else "at_most_100" if maximum <= 100 else "101_to_1000" if maximum <= 1000 else "over_1000"
    labels.append(dict(axis="max_written_number", label=size))
    labels.append(dict(axis="decimal_notation", label="present" if any("." in x for x in numbers) else "absent"))
    return labels


def make_types():
    protocol = preflight()
    rows = []
    for name in DATASETS:
        for row in read_rows(ROOT / f"e1/data/{name}.jsonl"):
            rows.append(dict(dataset=name, id=row["id"], strata=classify(row)))
    return dict(policy=POLICY, source_sha256=sha(Path(__file__)),
                data_sha256={name: protocol["data_files_sha256"][name] for name in DATASETS},
                note="Frozen without reading either model's outputs. GSM-Hard groups are numeric surface properties, not manually verified semantic types; written years/percentages also count.",
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "e5/problem_types.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new type-manifest path; do not silently overwrite frozen labels")
    result = make_types()
    write_json(args.output, result)
    print(dict(n=len(result["rows"]), strata=dict(Counter((r["dataset"], s["axis"], s["label"])
          for r in result["rows"] for s in r["strata"]))))


if __name__ == "__main__":
    main()
