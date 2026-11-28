"""P5 train-only cross-question margin alignment; frozen v1 joint training."""
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
import transformers
import peft
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e3"))
from P5.method import preflight, training_forward, prepare_pairs, set_pair
from e5.p4.p4_audit import audit_rows
from e5.protocol import read_json, require, sample_order
from transformers import set_seed
from common import ROOT, build, generate, read_rows, sha, tensor_batch, write_json
from e1.metrics import extract, numeric


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
    parser.add_argument("--pilot-updates", type=int, default=0)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    assert torch.cuda.is_bf16_supported(), "BF16 GPU required"
    require(transformers.__version__ == "4.52.4" and peft.__version__ == "0.15.2"
            and torch.__version__.split("+")[0] == "2.7.1", "Use the frozen E3 runtime versions")
    protocol = preflight(args.init, require_weights=True)
    cfg = json.loads(args.config.read_text())
    require(cfg == protocol["config"], "Common configuration must equal frozen v1")
    method = protocol["variant"]
    assert cfg["micro_batch_size"] == 1 and 1 <= cfg["epochs"] <= 3
    manifest = json.loads((ROOT / "e3/data/manifest.json").read_text())
    for name, info in ((n, manifest["splits"][n]) for n in ("train", "validation")):
        assert sha(ROOT / f"e3/data/{name}.jsonl") == info["sha256"]
    rows = read_rows(ROOT / "e3/data/train.jsonl")
    val = read_rows(ROOT / "e3/data/validation.jsonl")
    assert len(rows) == 2000 and len(val) == 200
    accumulation = cfg["gradient_accumulation"]
    per_epoch = math.ceil(len(rows) / accumulation)
    total = cfg["epochs"] * per_epoch
    assert 0 <= args.pilot_updates <= total
    stop = args.pilot_updates or total
    order = sample_order(cfg)
    contract = dict(config=cfg, variant=method, order_sha256=protocol["order_sha256"],
        data_files_sha256=protocol["data_files_sha256"],
        initialization_sha256=protocol["initialization_sha256"], data_sha256=sha(ROOT / "e3/data/manifest.json"),
        pilot_updates=args.pilot_updates, updates=stop, full_budget_updates=total,
        source_sha256={name: sha(ROOT / name) for name in
            ("e3/common.py", "e3/train.py", "SIM-CoT/CODI/src/model.py",
             "P5/train.py", "P5/method.py", "P5/negatives.py", "P5/train_negative_pairs.json", "e5/p4/p4_alignment.py", "e5/p4/p4_audit.py",
             "e5/p1_schedule.py", "e5/protocol.py")},
        assets_manifest_sha256=sha(ROOT / "assets/manifest.json"),
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__))
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / "contract.json"
    if contract_path.exists():
        assert args.resume and json.loads(contract_path.read_text()) == contract, "Run differs; choose a new output directory"
    else:
        assert not list(args.output.iterdir()), "Output must be empty for a new run"
        write_json(contract_path, contract)
    write_json(args.output / "sample_order.json", order)
    write_json(args.output / "preflight.json", protocol)
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(cfg)
    write_json(args.output / "alignment_audit.json",
               {"train": audit_rows(rows, model.tokenizer), "validation": audit_rows(val, model.tokenizer)})
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    require(initial["config"] == cfg, "Shared initialization configuration differs")
    expected_sources = read_json(ROOT / "e3/frozen_v1.json")["training_contract"]["source_sha256"]
    require(initial["source_sha256"] == expected_sources, "Original initialization source identity differs")
    model.load_state_dict(initial["model"], strict=True)
    initial_rng = initial["rng"]
    del initial
    write_json(args.output / "initialization.json", dict(path=str(args.init), sha256=sha(args.init),
        base="openai-community/gpt2", released_checkpoint_used=False))
    pairs = prepare_pairs(rows, model.tokenizer)
    write_json(args.output / "negative_pairs.json", pairs)
    model.cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    write_json(args.output / "optimizer_groups.json", [dict(weight_decay=cfg["weight_decay"],
        names=[name for name, parameter in model.named_parameters() if parameter.requires_grad])])
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
        require(0 <= done <= stop, "Resume update lies outside this run's budget")
        restore_rng(state["rng"])
        require(state["alignment_factor"] == method["alignment_factor"], "Saved alignment factor mismatch")
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
        model.explain_loss_factor = 1.0
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        indices = order[step]
        for index in indices:
            set_pair(rows, index, pairs[index])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = training_forward(model, tensor_batch(rows[index], model, "cuda"), step, method["alignment_factor"])
                loss = result["loss"]
            assert torch.isfinite(loss), (step, rows[index]["id"])
            (loss / len(indices)).backward()
            for name in ("loss", "ce_loss", "distill_loss", "ref_ce_loss", "explain_loss",
                         "base_loss", "alignment_loss", "weighted_alignment_loss", "alignment_valid_slots", "positive_cosine", "negative_cosine", "active_fraction"):
                if result[name] is None:
                    losses[name] = None
                else:
                    losses[name] = losses.get(name, 0.0) + float(result[name].detach()) / len(indices)
            del result, loss
        grad = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        done = step + 1
        record = dict(update=done, epoch=(done-1)//per_epoch+1, losses=losses,
            lr=lr, explain_loss_factor=model.explain_loss_factor,
            alignment_factor=method["alignment_factor"],
            grad_norm=float(grad), elapsed_seconds=time.perf_counter()-start)
        if done % per_epoch == 0 and not args.pilot_updates:
            record["validation"] = validate(model, val, cfg)
        with (args.output / "train_log.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if done % cfg["save_every"] == 0 or done == stop or done % per_epoch == 0:
            save(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng_state(), updates=done, config=cfg,
                contract=contract, init_sha256=protocol["initialization_sha256"],
                explain_loss_factor=model.explain_loss_factor, alignment_factor=method["alignment_factor"]))
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
