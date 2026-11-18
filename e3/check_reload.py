"""A6000 checkpoint gate: changed parameters, strict reload, deterministic decoding."""
import argparse
import gc
import json
from pathlib import Path
import torch
from common import ROOT, load_checkpoint, generate, read_rows, write_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=ROOT / "e3_runs/shared_init.pt")
    args = parser.parse_args()
    row = read_rows(ROOT / "e3/data/validation.jsonl")[0]
    model, state = load_checkpoint(args.checkpoint)
    assert state["updates"] > 0
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    assert sha(args.init) == state["init_sha256"]
    assert not torch.equal(initial["model"]["prj.1.weight"], state["model"]["prj.1.weight"])
    updates = state["updates"]
    k = state["config"]["latent_steps"]
    del initial, state
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        before = generate(model, row["question"], k, 32, explanations=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored, state = load_checkpoint(args.checkpoint)
    restored.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        after = generate(restored, row["question"], k, 32, explanations=True)
    assert before == after, "Reload changed deterministic answer or latent decoding"
    report = dict(status="PASS", updates=updates, changed_projector=True, strict_reload_identical=True,
                  sample_id=row["id"], outputs=after,
                  note="A pilot decoder may emit empty or incorrect text; this gate tests mechanics, not reasoning quality.")
    write_json(args.checkpoint.parent / "reload_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
