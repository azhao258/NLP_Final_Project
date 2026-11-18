"""Evaluate a self-trained checkpoint and export independent latent explanations."""
import argparse
import json
import time
from pathlib import Path
import torch
from common import ROOT, generate, load_checkpoint, read_rows, sha, targets, write_json
from e1.metrics import extract, numeric
from metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.limit <= 200
    args.output.mkdir(parents=True, exist_ok=True)
    assert not list(args.output.iterdir()), "Use a fresh output directory; do not mix checkpoints"
    model, checkpoint = load_checkpoint(args.checkpoint)
    model.to(torch.bfloat16).eval()
    cfg = checkpoint["config"]
    if args.split == "test":
        assert checkpoint["contract"]["pilot_updates"] == 0, "Pilot is not the formal test baseline"
        assert checkpoint["updates"] == checkpoint["contract"]["full_budget_updates"], "Finish the frozen training budget before test evaluation"
    dataset_manifest = ROOT / ("e1/data/manifest.json" if args.split == "test" else "e3/data/manifest.json")
    manifest = json.loads(dataset_manifest.read_text())
    if args.split == "test":
        for name, info in manifest["datasets"].items():
            assert sha(ROOT / f"e1/data/{name}.jsonl") == info["sha256"]
    else:
        assert sha(ROOT / "e3/data/validation.jsonl") == manifest["splits"]["validation"]["sha256"]
    write_json(args.output / "config.json", dict(checkpoint_sha256=sha(args.checkpoint),
        train_updates=checkpoint["updates"], config=cfg, split=args.split, limit=args.limit,
        explain=args.explain, gpu=torch.cuda.get_device_name(0),
        data_manifest_sha256=sha(ROOT / ("e1/data/manifest.json" if args.split=="test" else "e3/data/manifest.json")),
        inference_source_sha256=sha(ROOT / "e3/common.py"),
        precision="BF16 weights and autocast", batch_size=1, greedy=True, warmups=5,
        timing="CUDA synchronized, excludes tokenization, text decoding, explanation export and model loading"))
    if args.split == "test":
        groups = {name: read_rows(ROOT / f"e1/data/{name}.jsonl") for name in ("gsm8k", "gsm_hard", "svamp")}
    else:
        groups = {"validation": [dict(r, dataset="validation", reference=r["cot"], reference_type="equation", problem_type=None)
                                 for r in read_rows(ROOT / "e3/data/validation.jsonl")]}
    for name, rows in groups.items():
        rows = rows[:args.limit]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                generate(model, rows[0]["question"], cfg["latent_steps"], cfg["answer_max_tokens"])
            with (args.output / f"{name}.predictions.jsonl").open("w", encoding="utf8") as stream:
                for row in rows:
                    # generate() records synchronized time around model operations only.
                    result = generate(model, row["question"], cfg["latent_steps"], cfg["answer_max_tokens"], timed=True)
                    prediction = extract(result["raw_output"])
                    gold = extract(str(row["answer"]))
                    assert gold is not None, row["id"]
                    result.update(row, prediction=prediction, gold_normalized=gold,
                        correct=prediction==gold, latent_steps=cfg["latent_steps"],
                        reference_tokens=len(model.tokenizer.encode(row["reference"])) if row.get("reference") else None)
                    stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                    stream.flush()
                    if args.explain:
                        decoded = generate(model, row["question"], cfg["latent_steps"], cfg["decode_max_tokens"], explanations=True)
                        example = dict(id=row["id"], question=row["question"], gold=row["answer"],
                            prediction=prediction, correct=result["correct"], decoded_steps=decoded["decoded_steps"],
                            reference=row.get("reference"),
                            note="Decoded text is a diagnostic, not proof of a causal internal calculation.")
                        if "cot" in row:
                            example["supervision_slots"] = [model.tokenizer.decode(s) for s in targets(row, model.tokenizer, cfg["latent_steps"])]
                        with (args.output / f"{name}.decoded.jsonl").open("a", encoding="utf8") as f:
                            f.write(json.dumps(example, ensure_ascii=False)+"\n")
                    print(name, row["id"], result["correct"], flush=True)
    print(json.dumps(summarize(args.output), indent=2))
    print("COMPLETE", args.output)


if __name__ == "__main__":
    main()
