"""Freeze the literal first 2000 training rows and 200 disjoint validation questions."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from common import ROOT, sha, write_json, read_rows, pack, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "e3/config.json").read_text())
    source = args.assets / "data/gsm8k_aug_train.json"
    provenance = json.loads((ROOT / "assets/manifest.json").read_text())
    entry = next(x for x in provenance if x["file"].replace("\\", "/").endswith("gsm8k_aug_train.json"))
    assert sha(source) == entry["sha256"], "Training asset differs from pinned GSM8k-Aug"
    columns = json.loads(source.read_text(encoding="utf8"))
    key = lambda q: " ".join(q.split()).casefold()
    def row(i):
        return dict(id=f"gsm8k-aug-train-{i}", source_index=i,
                    **{name: columns[name][i] for name in ("question", "cot", "answer")})
    train = [row(i) for i in range(2000)]
    seen = {key(x["question"]) for x in train}
    val, skipped = [], []
    for i in range(2000, len(columns["question"])):
        item = row(i)
        q = key(item["question"])
        if q in seen:
            skipped.append(i)
            continue
        seen.add(q)
        val.append(item)
        if len(val) == 200:
            break
    assert len(val) == 200
    tok = AutoTokenizer.from_pretrained(args.assets / "gpt2", use_fast=False, local_files_only=True)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    report = {}
    for name, rows in (("train", train), ("validation", val)):
        lengths, counts, zero = [], [], []
        for item in rows:
            inputs = pack(item, tok)
            ref_len = len(inputs["ref_input_ids"][0])
            total = len(inputs["encoder_input_ids"][0]) + cfg["latent_steps"] + len(inputs["decoder_input_ids"][0])
            assert max(ref_len, total) <= 1024, (item["id"], ref_len, total)
            extracted = re.findall(r"<<[^<>]+>>", item["cot"])
            slots = targets(item, tok, cfg["latent_steps"])
            active = [s for s in slots if s != [tok.pad_token_id]]
            assert len(active) == min(len(extracted), cfg["latent_steps"] + 1), item["id"]
            if not active:
                zero.append(item["id"])
            lengths.append(max(ref_len, total))
            counts.append(len(extracted))
        report[name] = dict(n=len(rows), max_context=max(lengths), arithmetic_step_counts=dict(Counter(counts)),
                            no_step_ids=zero, duplicate_questions=len(rows)-len({key(x["question"]) for x in rows}))
    evaluation = {p.stem: read_rows(p) for p in (ROOT / "e1/data").glob("*.jsonl")}
    overlaps = {name: [x["id"] for x in rows if key(x["question"]) in {key(t["question"]) for t in train}]
                for name, rows in evaluation.items()}
    output = ROOT / "e3/data"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf8", newline="\n")
    write_json(output / "manifest.json", dict(source=entry, train_policy="source rows 0..1999 unchanged",
        validation_policy="scan from row 2000; exclude normalized questions in train or previous validation",
        skipped_validation_indices=skipped, splits={n: dict(sha256=sha(output/f"{n}.jsonl"), ids=[r["id"] for r in rows])
        for n, rows in (("train", train), ("validation", val))}, report=report,
        exact_normalized_train_test_overlap=overlaps,
        note="Exact question check does not detect paraphrases or shared original problems. No rows silently removed from first 2000."))
    print(json.dumps(dict(report=report, test_overlap=overlaps, skipped_validation=skipped), indent=2))


if __name__ == "__main__":
    main()
