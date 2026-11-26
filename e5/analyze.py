"""Paired Part 3 test metrics. CPU-only; exports structured data without Markdown reports."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.protocol import ROOT, DATASETS, preflight, read_json, read_rows, require, sha, write_json
from e5.prefix_mask.evaluate import POLICY
from e5.problem_types import make_types, reference_slots
from e4.analyze import SLOT_LABELS, compare, equation, expression, normalize
from e1.metrics import extract


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def inspect_steps(row, decoded):
    """Return evidence with N/A preserved; never manufacture reference chains."""
    name = row["dataset"]
    refs = reference_slots(row)
    steps = []
    for index, generated in enumerate(decoded):
        value = equation(generated["text"])
        reference = refs[index] if index < len(refs) else None
        match = compare(generated["text"], reference) if reference is not None else {}
        whole = None
        if name == "svamp":
            try:
                target = expression(row["reference"].strip())
                # Expression compatibility is existential at a slot, never a gold step label.
                whole = bool(value["parsed"] and value["equation_count"] == 1 and value["arithmetic_valid"]
                             and value["lhs"] == target and value["rhs"]["value"] == target["value"])
            except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
                whole = None
        labels = []
        if reference is not None:
            if not normalize(generated["text"]):
                labels.append("missing_output")
            elif not value["parsed"]:
                labels.append("unparseable_output")
            for key, label in (("operator_match", "operator_mismatch"), ("operand_match", "operand_mismatch"),
                               ("result_match", "intermediate_result_mismatch")):
                if match[key] is False:
                    labels.append(label)
            if match["reference_equation"]["arithmetic_valid"] is False:
                labels.append("reference_arithmetic_inconsistency")
            if match["conservative_equivalent"] is False:
                labels.append("reference_step_not_reproduced")
            if match["missing_equation_count"]:
                labels.append("fewer_equations_than_reference")
        if value["arithmetic_valid"] is False:
            labels.append("arithmetic_inconsistency")
        steps.append(dict(slot=index, consumed_by_reasoner=index < 4, generated=generated["text"],
            capped=generated["capped"], reference=reference, reference_available=reference is not None,
            parsed=value["parsed"], arithmetic_valid=value["arithmetic_valid"], labels=labels,
            strict_match=match.get("strict_match"), operator_match=match.get("operator_match"),
            operand_match=match.get("operand_match"), result_match=match.get("result_match"),
            conservative_equivalent=match.get("conservative_equivalent"),
            whole_expression_compatible=whole, evidence=match or value))
    return steps


def summarize(rows, steps):
    valid = [s for s in steps if s["reference_available"]]
    refs = [r for r in rows if r["reference_tokens"] is not None]
    latencies = [r["latency_seconds"] * 1000 for r in rows]
    result = dict(n=len(rows), correct=sum(r["correct"] for r in rows), em=rate(sum(r["correct"] for r in rows), len(rows)),
        latent_steps=4, mean_answer_tokens=float(np.mean([r["answer_tokens"] for r in rows])),
        mean_generated_tokens=float(np.mean([r["generated_tokens"] for r in rows])),
        reference_count=len(refs), reference_length_ratio=rate(sum(r["reference_tokens"] for r in refs), len(refs) * 4),
        latency_mean_ms=float(np.mean(latencies)), latency_median_ms=float(np.median(latencies)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        answer_parse_failures=sum(r["prediction"] is None for r in rows), answer_truncated=sum(r["stop_reason"] == "length" for r in rows),
        valid_reference_slots=len(valid), reference_covered_questions=sum(bool(reference_slots(r)) for r in rows),
        generated_slots=len(steps), generated_parseable=sum(s["parsed"] for s in steps),
        generated_arithmetic_valid=sum(s["arithmetic_valid"] is True for s in steps),
        explanation_truncated=sum(s["capped"] for s in steps))
    for key in ("strict_match", "operator_match", "operand_match", "result_match", "conservative_equivalent"):
        count = sum(s[key] is True for s in valid)
        result[key + "_count"] = count if valid else None
        result[key + "_rate"] = rate(count, len(valid))
    by_id = defaultdict(list)
    for s in valid:
        by_id[s["id"]].append(s)
    correct = {r["id"]: r["correct"] for r in rows}
    result["complete_reference_compatible"] = sum(all(s["conservative_equivalent"] is True for s in group) for group in by_id.values()) if by_id else None
    result["answer_and_complete_reference_correct"] = sum(correct[key] and all(s["conservative_equivalent"] is True for s in group)
                                                               for key, group in by_id.items()) if by_id else None
    result["correct_answer_explanation_mismatch"] = sum(correct[key] and not all(s["conservative_equivalent"] is True for s in group)
                                                           for key, group in by_id.items()) if by_id else None
    whole_by_id = defaultdict(list)
    for s in steps:
        if s["whole_expression_compatible"] is not None:
            whole_by_id[s["id"]].append(s)
    result["whole_expression_reference_count"] = len(whole_by_id)
    result["whole_expression_slot0_rate"] = rate(sum(group[0]["whole_expression_compatible"] for group in whole_by_id.values()), len(whole_by_id))
    result["whole_expression_any_slot_rate"] = rate(sum(any(s["whole_expression_compatible"] for s in group) for group in whole_by_id.values()), len(whole_by_id))
    return result


def load_export(directory, role, protocol, types_hash):
    config = read_json(directory / "config.json")
    complete = read_json(directory / "complete.json")
    require(complete["n"] == 600 and complete["pairing_id"] == config["pairing_id"], "Incomplete evaluation")
    expected_files = {"config.json", "summary.csv"} | {f"{name}.{kind}.jsonl" for name in DATASETS for kind in ("predictions", "decoded")}
    require(set(complete["files"]) == expected_files, "Export inventory differs")
    for name, digest in complete["files"].items():
        require(sha(directory / name) == digest, f"Export modified after completion: {name}")
    require(config["role"] == role and config["policy"] == POLICY, "Different evaluation role/procedure")
    require(config["common_config"] == protocol["config"] and config["train_updates"] == 750, "Different training settings")
    require(config["initialization_sha256"] == protocol["initialization_sha256"], "Different initialization")
    require(config["data_files_sha256"] == protocol["data_files_sha256"] and config["types_sha256"] == types_hash, "Data/type manifest mismatch")
    if role == "control":
        require(config["checkpoint_sha256"] == read_json(ROOT / "e3/frozen_v1.json")["checkpoint_sha256"], "Not frozen v1")
    else:
        require(config["variant"] == protocol["variant"], "Wrong improvement")
        require(config["training_contract"]["variant"] == protocol["variant"] and
                config["training_contract"]["order_sha256"] == protocol["order_sha256"] and
                config["training_contract"]["data_files_sha256"] == protocol["data_files_sha256"], "Unmatched training contract")
    require(config["training_contract"]["pilot_updates"] == 0 and config["training_contract"]["full_budget_updates"] == 750, "Incomplete training budget")
    results = {}
    for name in DATASETS:
        golds = read_rows(ROOT / f"e1/data/{name}.jsonl")
        predictions = read_rows(directory / f"{name}.predictions.jsonl")
        decoded = read_rows(directory / f"{name}.decoded.jsonl")
        ids = [r["id"] for r in golds]
        require([r["id"] for r in predictions] == ids == [r["id"] for r in decoded], "Duplicate/missing/reordered evaluation IDs")
        steps = []
        for gold, pred, dec in zip(golds, predictions, decoded):
            require(all(pred.get(key) == value for key, value in gold.items()), f"Prediction source mismatch: {gold['id']}")
            require(dec["question"] == gold["question"] and dec["reference"] == gold.get("reference")
                    and str(dec["gold"]) == str(gold["answer"]) and dec["dataset"] == name, "Explanation source mismatch")
            answer = extract(pred["raw_output"])
            correct = answer is not None and answer == extract(str(gold["answer"]))
            require(pred["prediction"] == dec["prediction"] == answer and pred["correct"] == dec["correct"] == correct, "Answer score mismatch")
            require(pred["latent_steps"] == 4 and np.isfinite(pred["latency_seconds"]) and pred["latency_seconds"] >= 0, "Invalid budget/timing")
            require(len(dec["decoded_steps"]) == 5, "Expected 5 explanation slots")
            require([s["slot"] for s in dec["decoded_steps"]] == list(range(5)) and
                    [s["consumed_by_reasoner"] for s in dec["decoded_steps"]] == [True] * 4 + [False], "Incorrect latent slot identities")
            require((pred["reference_tokens"] is None) == (gold.get("reference") is None), "Reference length availability differs")
            steps.extend(dict(role=role, dataset=name, id=gold["id"], **s) for s in inspect_steps(gold, dec["decoded_steps"]))
        results[name] = (predictions, steps)
    return config, results


def analyze(control, modified, output, types_path=ROOT / "e5/problem_types.json"):
    protocol = preflight()
    frozen_types = read_json(types_path)
    require(frozen_types == make_types(), "Type manifest differs from the frozen prediction-independent rules")
    a_cfg, a = load_export(control, "control", protocol, sha(types_path))
    b_cfg, b = load_export(modified, "modified", protocol, sha(types_path))
    for key in ("pairing_id", "policy", "source_sha256", "gpu", "torch", "transformers", "peft"):
        require(a_cfg[key] == b_cfg[key], f"Unmatched evaluation: {key}")
    require(a_cfg["checkpoint_sha256"] != b_cfg["checkpoint_sha256"], "Both inputs are the same checkpoint")
    require(not output.exists(), "Use a new analysis directory")
    groups = {(r["dataset"], r["id"]): r["strata"] for r in frozen_types["rows"]}
    summary, paired, strata, all_steps, slots, categories = [], [], [], [], [], []
    for name in DATASETS:
        for role, source in (("control", a), ("modified", b)):
            rows, steps = source[name]
            summary.append(dict(role=role, dataset=name, **summarize(rows, steps)))
            all_steps.extend(steps)
            labels = Counter(label for s in steps if s["reference_available"] for label in s["labels"])
            denominator = sum(s["reference_available"] for s in steps)
            categories.extend(dict(role=role, dataset=name, label=label, count=labels[label] if denominator else None,
                                   denominator=denominator) for label in SLOT_LABELS)
            for i in range(5):
                local = [s for s in steps if s["slot"] == i]
                frequencies = Counter(normalize(s["generated"]) for s in local)
                text, count = frequencies.most_common(1)[0]
                slots.append(dict(role=role, dataset=name, slot=i, consumed_by_reasoner=i < 4,
                    outputs=len(local), valid_reference_slots=sum(s["reference_available"] for s in local),
                    unique_outputs=len(frequencies), most_common=text, most_common_count=count, template_share=count/len(local)))
        a_rows, a_steps = a[name]
        b_rows, b_steps = b[name]
        for left, right in zip(a_rows, b_rows):
            require(left["reference_tokens"] == right["reference_tokens"], "Different reference tokenizer lengths")
            paired.append(dict(dataset=name, id=left["id"], question=left["question"], gold=left["answer"],
                control_prediction=left["prediction"], modified_prediction=right["prediction"],
                control_correct=left["correct"], modified_correct=right["correct"],
                outcome="gained" if right["correct"] and not left["correct"] else "lost" if left["correct"] and not right["correct"] else "both_correct" if left["correct"] else "both_wrong",
                control_explanations=[s["generated"] for s in a_steps if s["id"] == left["id"]],
                modified_explanations=[s["generated"] for s in b_steps if s["id"] == left["id"]],
                strata=groups[name, left["id"]]))
        buckets = defaultdict(set)
        for row in a_rows:
            buckets["dataset", "all"].add(row["id"])
            for group in groups[name, row["id"]]:
                buckets[group["axis"], group["label"]].add(row["id"])
        for (axis, label), ids in sorted(buckets.items()):
            left = [r for r in a_rows if r["id"] in ids]
            right = [r for r in b_rows if r["id"] in ids]
            sa = summarize(left, [s for s in a_steps if s["id"] in ids])
            sb = summarize(right, [s for s in b_steps if s["id"] in ids])
            strata.append(dict(dataset=name, axis=axis, label=label, n=len(ids),
                gained=sum(y["correct"] and not x["correct"] for x, y in zip(left, right)),
                lost=sum(x["correct"] and not y["correct"] for x, y in zip(left, right)),
                em_delta_pp=100*(sb["em"]-sa["em"]), small_group=len(ids) < 20,
                **{"control_"+k: v for k, v in sa.items()}, **{"modified_"+k: v for k, v in sb.items()}))
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    for name, rows in (("summary", summary), ("paired_questions", paired), ("problem_types", strata),
                       ("steps", all_steps), ("slots", slots), ("categories", categories)):
        csv_write(output / f"{name}.csv", rows)
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf8")
    write_json(output / "provenance.json", dict(pairing_id=a_cfg["pairing_id"], preflight=protocol,
        input_configs={"control": a_cfg, "modified": b_cfg}, types_sha256=sha(types_path),
        analysis_sources={name: sha(ROOT / name) for name in ("e5/analyze.py", "e5/problem_types.py", "e4/analyze.py", "e1/metrics.py")},
        no_training=True, limitations=["One seed; no claim of stable gains from a few questions.",
        "v1 selection used test results.", "GSM-Hard has no gold reasoning chain.",
        "SVAMP whole-expression compatibility is not step-by-step alignment or general equivalence.",
        "GSM-Hard numeric strata are surface properties, not semantic operation labels."]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.evaluation / "control", args.evaluation / "modified", args.output)


if __name__ == "__main__":
    main()
