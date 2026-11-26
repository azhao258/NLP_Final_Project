"""Evaluate both completed checkpoints in one process; no optimizer or training."""
import argparse
import gc
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e5.protocol import ROOT, DATASETS, preflight, read_json, read_rows, require, sha, validate_checkpoint, write_json

POLICY = dict(precision="BF16 weights and autocast", greedy=True, batch_size=1,
              warmups=5, answer_max_tokens=64, decode_max_tokens=64, latent_steps=4,
              timing="CUDA synchronized; question + latent + answer; excludes tokenization, model loading and explanation decoding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=ROOT / "e3_runs/control/last.pt")
    parser.add_argument("--modified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = preflight()
    require(not args.output.exists(), "Use a new evaluation directory")
    require(read_json(ROOT / "e5/problem_types.json")["data_sha256"] ==
            {name: protocol["data_files_sha256"][name] for name in DATASETS}, "Problem strata are not frozen for these test data")
    from e5.problem_types import make_types
    require(read_json(ROOT / "e5/problem_types.json") == make_types(), "Type definitions changed before evaluation")
    import torch
    import transformers
    import peft
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "A BF16 CUDA GPU is required")
    require(torch.__version__.split("+")[0] == "2.7.1" and transformers.__version__ == "4.52.4"
            and peft.__version__ == "0.15.2", "Use frozen E3 runtime versions")
    sys.path.insert(0, str(ROOT / "e3"))
    from common import build, generate
    from e1.metrics import extract
    from e3.metrics import summarize
    hashes = {}
    for role, path in (("control", args.control), ("modified", args.modified)):
        hashes[role] = sha(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(state, hashes[role], role, protocol)
        del state
        gc.collect()
    args.output.mkdir(parents=True)
    pairing_id = uuid.uuid4().hex
    sources = {name: sha(ROOT / name) for name in ("e3/common.py", "e5/prefix_mask/evaluate.py", "e5/protocol.py")}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(protocol["config"])
    for role, path in (("control", args.control), ("modified", args.modified)):
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        contract = state["contract"]
        del state
        gc.collect()
        model.to(device="cuda", dtype=torch.bfloat16).eval()
        directory = args.output / role
        directory.mkdir()
        write_json(directory / "config.json", dict(role=role, pairing_id=pairing_id, policy=POLICY,
            checkpoint_sha256=hashes[role], train_updates=750, initialization_sha256=protocol["initialization_sha256"],
            training_contract=contract, common_config=protocol["config"], variant=protocol["variant"] if role == "modified" else None,
            data_files_sha256=protocol["data_files_sha256"], source_sha256=sources,
            types_sha256=sha(ROOT / "e5/problem_types.json"), gpu=torch.cuda.get_device_name(0),
            torch=str(torch.__version__), transformers=transformers.__version__, peft=peft.__version__,
            order="control then modified; same process and hardware, not simultaneous timing",
            selection_used_test_results=True))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name in DATASETS:
                rows = read_rows(ROOT / f"e1/data/{name}.jsonl")
                for _ in range(POLICY["warmups"]):
                    generate(model, rows[0]["question"], 4, 64)
                with (directory / f"{name}.predictions.jsonl").open("w", encoding="utf8") as pf, \
                     (directory / f"{name}.decoded.jsonl").open("w", encoding="utf8") as df:
                    for row in rows:
                        result = generate(model, row["question"], 4, 64, timed=True)
                        prediction, gold = extract(result["raw_output"]), extract(str(row["answer"]))
                        require(gold is not None, f"Unparseable gold: {row['id']}")
                        result.update(row, prediction=prediction, gold_normalized=gold,
                            correct=prediction == gold, latent_steps=4,
                            reference_tokens=len(model.tokenizer.encode(row["reference"])) if row.get("reference") else None)
                        pf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        explanation = generate(model, row["question"], 4, 64, explanations=True)
                        # This second generation is outside latency. A changed answer is an error, not silently hidden.
                        require(explanation["token_ids"] == result["token_ids"], "Answer differs during explanation export")
                        export = dict(id=row["id"], dataset=name, question=row["question"], gold=row["answer"],
                                      reference=row.get("reference"), prediction=prediction, correct=result["correct"],
                                      decoded_steps=explanation["decoded_steps"])
                        df.write(json.dumps(export, ensure_ascii=False) + "\n")
                print(role, name, "200 answers and explanations exported", flush=True)
        summarize(directory)
        write_json(directory / "complete.json", dict(pairing_id=pairing_id, n=600,
                   files={p.name: sha(p) for p in directory.iterdir() if p.is_file()}))
    write_json(args.output / "complete.json", dict(pairing_id=pairing_id, checkpoints=hashes))


if __name__ == "__main__":
    main()
