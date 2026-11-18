"""Single-GPU SIM-CoT control; exact optimizer-boundary resume, no test selection."""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from transformers import set_seed
from common import ROOT, build, generate, read_rows, sha, tensor_batch, write_json
from e1.metrics import extract, numeric
from variant_support import variant_config, make_optimizer
import hashlib
import transformers


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, rows, cfg):
    model.eval()
    correct = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            out = generate(model, row["question"], cfg["latent_steps"], cfg["answer_max_tokens"])
            gold = extract(str(row["answer"]))
            assert gold is not None, row["id"]
            correct += extract(out["raw_output"]) == gold
    model.train()
    return dict(n=len(rows), correct=correct, em=correct / len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=ROOT / "e3_runs/shared_init.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "e3/config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", choices=["v2", "v3"], required=True)
    parser.add_argument("--pilot-updates", type=int, default=0)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    assert torch.cuda.is_bf16_supported(), "BF16 GPU required"
    assert transformers.__version__ == "4.52.4", "Use the pinned requirements (Transformers 4.52.4)"
    baseline_cfg = json.loads(args.config.read_text())
    cfg = variant_config(baseline_cfg, args.variant)
    assert args.init.is_file(), "Original shared_init.pt is required; do not generate a new initialization"
    assert cfg["micro_batch_size"] == 1 and 1 <= cfg["epochs"] <= 3
    manifest = json.loads((ROOT / "e3/data/manifest.json").read_text())
    for name, info in manifest["splits"].items():
        assert sha(ROOT / f"e3/data/{name}.jsonl") == info["sha256"]
    rows = read_rows(ROOT / "e3/data/train.jsonl")
    val = read_rows(ROOT / "e3/data/validation.jsonl")
    assert len(rows) == 2000 and len(val) == 200
    accumulation = cfg["gradient_accumulation"]
    per_epoch = math.ceil(len(rows) / accumulation)
    total = cfg["epochs"] * per_epoch
    assert 0 <= args.pilot_updates <= total
    stop = args.pilot_updates or total
    order = []
    for epoch in range(cfg["epochs"]):
        indices = list(range(len(rows)))
        random.Random(cfg["seed"] + epoch).shuffle(indices)
        order.extend([indices[i:i+accumulation] for i in range(0, len(indices), accumulation)])
    contract = dict(variant=args.variant, config=cfg,
        order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), data_sha256=sha(ROOT / "e3/data/manifest.json"),
        pilot_updates=args.pilot_updates, updates=stop, full_budget_updates=total,
        source_sha256={name: sha(ROOT / name) for name in
            ("e3/common.py", "e3/train.py", "e3/train_variants.py", "e3/variant_support.py", "SIM-CoT/CODI/src/model.py")},
        assets_manifest_sha256=sha(ROOT / "assets/manifest.json"),
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__))
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / "contract.json"
    if contract_path.exists():
        assert args.resume and json.loads(contract_path.read_text()) == contract, "Run differs; choose a new output directory"
    else:
        assert not list(args.output.iterdir()), "Output must be empty for a new run"
        write_json(contract_path, contract)
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(cfg)
    if args.init.exists():
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        assert initial["config"] == baseline_cfg, "Shared initialization must match original v1 config"
        assert all(sha(ROOT/name) == value for name,value in initial["source_sha256"].items()), "Original initialization/model code differs"
        model.load_state_dict(initial["model"], strict=True)
        initial_rng = initial["rng"]
        del initial
    else:
        args.init.parent.mkdir(parents=True, exist_ok=True)
        initial_rng = rng_state()
        save(args.init, dict(model=model.state_dict(), config=cfg, rng=initial_rng,
                            source_sha256=contract["source_sha256"]))
    write_json(args.output / "initialization.json", dict(path=str(args.init), sha256=sha(args.init),
        base="openai-community/gpt2", released_checkpoint_used=False))
    model.cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer, group_report = make_optimizer(model, cfg, args.variant)
    write_json(args.output / "optimizer_groups.json", group_report)
    write_json(args.output / "sample_order.json", order)
    def scale(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        return max(0.0, (total - step) / (total - cfg["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    checkpoint = args.output / "last.pt"
    done = 0
    restore_rng(initial_rng)
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["contract"] == contract and state["init_sha256"] == sha(args.init)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        done = state["updates"]
        restore_rng(state["rng"])
        del state
    elif args.resume:
        raise RuntimeError("No saved checkpoint; use a new directory to restart")
    log_path = args.output / "train_log.jsonl"
    if args.resume and log_path.exists():
        # A crash may leave logs newer than the last atomic checkpoint.
        log_rows = read_rows(log_path)
        log_path.write_text("".join(json.dumps(x)+"\n" for x in log_rows if x["update"] <= done))
    write_json(args.output / "parameters.json", dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in parameters),
        groups={name: sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith(name))
                for name in ("codi.", "prj.", "decoder.")}))
    (args.output / "environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True))
    start = time.perf_counter()
    start_update = done
    torch.cuda.reset_peak_memory_stats()
    for step in range(done, stop):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        indices = order[step]
        for index in indices:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**tensor_batch(rows[index], model, "cuda"), step=step, step_ratio=step/total)
                loss = result["loss"]
            assert torch.isfinite(loss), (step, rows[index]["id"])
            (loss / len(indices)).backward()
            for name in ("loss", "ce_loss", "distill_loss", "ref_ce_loss", "explain_loss"):
                losses[name] = losses.get(name, 0.0) + float(result[name].detach()) / len(indices)
            del result, loss
        grad = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        done = step + 1
        record = dict(update=done, epoch=(done-1)//per_epoch+1, losses=losses,
            lr=lr, grad_norm=float(grad), elapsed_seconds=time.perf_counter()-start)
        if done % per_epoch == 0 and not args.pilot_updates:
            record["validation"] = validate(model, val, cfg)
        with (args.output / "train_log.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if done % cfg["save_every"] == 0 or done == stop or done % per_epoch == 0:
            save(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng_state(), updates=done, config=cfg,
                contract=contract, init_sha256=sha(args.init)))
    torch.cuda.synchronize()
    session = dict(start_update=start_update,
        completed_updates=done, seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        note="Training loop including validation/checkpoint I/O; excludes model loading. Count rental time separately.")
    with (args.output / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(session)+"\n")
    print("COMPLETE", checkpoint, flush=True)


if __name__ == "__main__":
    main()
