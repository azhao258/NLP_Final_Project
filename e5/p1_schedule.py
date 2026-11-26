"""Frozen P1 schedule and training/validation-only provenance checks."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.protocol import ROOT, read_json, read_rows, require, sample_order, sha

METHOD = dict(variant="explain_ramp_v1", start_factor=0.1, end_factor=1.0,
              ramp_updates=250, prefix_mask=False,
              convention="update 1 uses 0.1; update 250 uses 1.0; updates 251-750 use 1.0")


def explain_factor(completed_updates):
    """Coefficient for the NEXT optimizer update; identical across its microbatches."""
    require(isinstance(completed_updates, int) and completed_updates >= 0,
            "completed_updates must be a nonnegative integer")
    return 0.1 + 0.9 * min(completed_updates, 249) / 249


def preflight(init=None, require_weights=False):
    frozen = read_json(ROOT / "e3/frozen_v1.json")
    cfg = read_json(ROOT / "e3/config.json")
    require(read_json(ROOT / "e3/active_baseline.json")["active"] == "v1", "Control must remain v1")
    require(cfg == frozen["training_contract"]["config"], "Common configuration differs from v1")
    for name, expected in frozen["training_contract"]["source_sha256"].items():
        require(sha(ROOT / name) == expected, f"Frozen source changed: {name}")
    require(sha(ROOT / "assets/manifest.json") == frozen["training_contract"]["assets_manifest_sha256"],
            "Base assets manifest changed")
    manifest = read_json(ROOT / "e3/data/manifest.json")
    data = {}
    for split, count in (("train", 2000), ("validation", 200)):
        path = ROOT / f"e3/data/{split}.jsonl"
        rows = read_rows(path)
        require(sha(path) == manifest["splits"][split]["sha256"], f"Changed {split} bytes")
        require(len(rows) == count and [r["id"] for r in rows] == manifest["splits"][split]["ids"],
                f"Changed {split} IDs")
        if split == "train":
            require([r["source_index"] for r in rows] == list(range(2000)), "Not the first 2000 rows")
        data[split] = sha(path)
    order = sample_order(cfg)
    require(len(order) == 750 and sum(map(len, order)) == 6000, "Training budget changed")
    init = Path(init) if init is not None else ROOT / "e3_runs/shared_init.pt"
    if init.exists():
        require(sha(init) == frozen["initialization_sha256"], "Not the original shared initialization")
    else:
        require(not require_weights, "Restore original shared_init.pt; never create a replacement")
    return dict(config=cfg, variant=METHOD, data_files_sha256=data,
        order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        full_budget_updates=750, sample_presentations=6000,
        initialization_sha256=frozen["initialization_sha256"], initialization_available=init.is_file(),
        historical_manifest_sha256=frozen["training_contract"]["data_sha256"],
        current_manifest_sha256=sha(ROOT / "e3/data/manifest.json"),
        historical_manifest_mismatch=frozen["training_contract"]["data_sha256"] != sha(ROOT / "e3/data/manifest.json"),
        note="Train/validation bytes checked. Historical manifest discrepancy retained. No test data read.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(require_weights=args.require_weights), indent=2))
