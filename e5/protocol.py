"""Frozen E5 contract and CPU-only preflight. Never constructs or trains a model."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "gsm_hard", "svamp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def sample_order(cfg, n=2000):
    order = []
    for epoch in range(cfg["epochs"]):
        indices = list(range(n))
        random.Random(cfg["seed"] + epoch).shuffle(indices)
        order.extend(indices[i:i + cfg["gradient_accumulation"]]
                     for i in range(0, n, cfg["gradient_accumulation"]))
    return order


def validate_variant(value):
    expected = {"variant": "prefix_mask_v1", "probability": 0.3, "mask_seed": 405103,
                "replacement": "zero_embedding", "protect_equation_boundaries": True}
    require(value == expected, "E5 method is frozen; changed settings require a separately documented experiment")
    return value


def preflight(init=None, require_weights=False):
    frozen = read_json(ROOT / "e3/frozen_v1.json")
    cfg = read_json(ROOT / "e3/config.json")
    require(read_json(ROOT / "e3/active_baseline.json")["active"] == "v1", "Control must remain v1")
    require(cfg == frozen["training_contract"]["config"], "Common configuration differs from frozen v1")
    variant = validate_variant(read_json(ROOT / "e5/prefix_mask/config.json"))
    source = frozen["training_contract"]["source_sha256"]
    for name, expected in source.items():
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
    test_manifest = read_json(ROOT / "e1/data/manifest.json")
    for name in DATASETS:
        path = ROOT / f"e1/data/{name}.jsonl"
        rows = read_rows(path)
        require(sha(path) == test_manifest["datasets"][name]["sha256"], f"Changed {name} bytes")
        require(len(rows) == 200 and [r["id"] for r in rows] == test_manifest["datasets"][name]["ids"],
                f"Changed {name} IDs")
        data[name] = sha(path)
    order = sample_order(cfg)
    require(len(order) == 750 and sum(map(len, order)) == 6000, "Training budget changed")
    init = Path(init) if init is not None else ROOT / "e3_runs/shared_init.pt"
    if init.exists():
        require(sha(init) == frozen["initialization_sha256"], "Not the original shared initialization")
    else:
        require(not require_weights, "Restore the original shared_init.pt before training; never create a replacement")
    return dict(config=cfg, variant=variant, data_files_sha256=data,
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                full_budget_updates=750, sample_presentations=6000,
                initialization_sha256=frozen["initialization_sha256"], initialization_available=init.is_file(),
                historical_manifest_sha256=frozen["training_contract"]["data_sha256"],
                current_manifest_sha256=sha(ROOT / "e3/data/manifest.json"),
                historical_manifest_mismatch=frozen["training_contract"]["data_sha256"] != sha(ROOT / "e3/data/manifest.json"),
                note="Current split bytes verified; historical manifest discrepancy remains disclosed.")


def validate_checkpoint(state, checkpoint_hash, role, protocol):
    frozen = read_json(ROOT / "e3/frozen_v1.json")
    require(state["config"] == protocol["config"], "Checkpoint common configuration mismatch")
    require(state["updates"] == 750 and state["contract"]["full_budget_updates"] == 750
            and state["contract"]["pilot_updates"] == 0, "Only completed formal checkpoints may enter test comparison")
    require(state["init_sha256"] == protocol["initialization_sha256"], "Checkpoint initialization mismatch")
    if role == "control":
        require(checkpoint_hash == frozen["checkpoint_sha256"], "Control is not frozen v1")
    else:
        require(role == "modified", "Unknown model role")
        require(state["contract"]["variant"] == protocol["variant"], "Incorrect E5 variant")
        require(state["contract"]["order_sha256"] == protocol["order_sha256"], "Training order mismatch")
        require(state["contract"]["data_files_sha256"] == protocol["data_files_sha256"], "Training/evaluation data mismatch")
        require("prefix_mask_rng" in state, "Missing masking RNG checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight(args.init, args.require_weights)
    if args.output:
        require(not args.output.exists(), "Preflight output already exists")
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
