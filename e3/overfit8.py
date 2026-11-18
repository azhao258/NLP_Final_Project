"""Bounded eight-example debugging run; never a formal course control checkpoint."""
import argparse
import importlib.util
import ast
from contextlib import nullcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
from common import ROOT, build, generate, pack, read_rows, sha, targets, tensor_batch, write_json
from diagnose import latent_vectors, teacher_score
# common.py adds the upstream CODI directory to sys.path; that repository also
# contains train.py. Load our helpers by absolute path, never by the bare name.
_train_spec = importlib.util.spec_from_file_location("e3_control_train_helpers", Path(__file__).resolve().with_name("train.py"))
_train_helpers = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_helpers)
rng_state = _train_helpers.rng_state
restore_rng = _train_helpers.restore_rng
save = _train_helpers.save
from e1.metrics import extract

INDICES = [2, 3, 18, 7, 9, 11, 23, 36]
LOSS_NAMES = ("loss", "ce_loss", "distill_loss", "ref_ce_loss", "explain_loss")


def amp(model):
    return torch.autocast("cuda", dtype=torch.bfloat16) if next(model.parameters()).is_cuda else nullcontext()


def append(path, obj):
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def arithmetic(text):
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError("Unsupported expression in arithmetic audit")
    return visit(ast.parse(text, mode="eval").body)


def interface_audit(model, rows, k):
    result = []
    tok = model.tokenizer
    for row in rows:
        x = pack(row, tok, model.bot_id, model.eot_id)
        q = tok.encode(row["question"].strip())
        cot = tok.encode(row["cot"].strip())
        a = tok.encode("The answer is: " + str(row["answer"]).strip()) + [tok.eos_token_id]
        prefix = tok.encode("The answer is:")
        dec, ref = x["decoder_input_ids"][0], x["ref_input_ids"][0]
        assert dec == [model.eot_id] + a and ref == q + cot + a
        assert x["labels"][0][1:] == a
        assert x["ref_labels"][0] == [-100] * len(q) + cot + a
        # Upstream subtracts one before gathering the hidden state predicting the first answer token.
        sp, tp = x["model_answer_position"][0], x["ref_answer_position"][0]
        assert sp == 1 + len(prefix) and tp == len(q) + len(cot) + len(prefix)
        assert dec[sp:] == ref[tp:] == a[len(prefix):]
        assert dec[sp-1] == ref[tp-1] == prefix[-1]
        slots = targets(row, tok, k)
        active = [s for s in slots if s != [tok.pad_token_id]]
        equations = re.findall(r"<<([^<>]+)>>", row["cot"])
        assert 1 <= len(equations) <= 3
        assert len(slots) == k+1 and len(active) == len(equations)
        for target, equation in zip(active, equations):
            assert target[-1] == tok.eos_token_id
            assert re.sub(r"\s+", "", tok.decode(target[:-1])) == "<<" + equation + ">>"
            left, right = equation.rsplit("=", 1)
            assert arithmetic(left) == arithmetic(right), row["id"]
        assert arithmetic(equations[-1].rsplit("=",1)[1]) == Fraction(extract(str(row["answer"])))
        result.append(dict(id=row["id"], answer=row["answer"], arithmetic_steps=len(equations),
            student_predicting_state_position=sp-1, teacher_predicting_state_position=tp-1,
            first_numeric_target_token=tok.decode([dec[sp]]),
            student_supervised_text=tok.decode(x["labels"][0][1:]),
            reference_supervised_text=tok.decode(cot+a),
            supervision_slots=[tok.decode(s) for s in slots], status="PASS"))
    return result


def group_name(name):
    if name.startswith("codi.") and "lora_" in name: return "lora"
    if name.startswith("prj."): return "projector"
    if name.startswith("decoder."): return "decoder"
    return "other"


def gradient_report(model):
    groups = {g: dict(parameter_tensors=0, with_gradient=0, nonzero_gradient=0,
                     squared_norm=0.0, finite=True) for g in ("lora", "projector", "decoder")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"Frozen parameter has gradient: {name}"
            continue
        g = group_name(name)
        assert g in groups, f"Unexpected trainable parameter: {name}"
        d = groups[g]
        d["parameter_tensors"] += 1
        if p.grad is not None:
            d["with_gradient"] += 1
            norm = float(torch.linalg.vector_norm(p.grad.detach().float()))
            d["finite"] &= math.isfinite(norm)
            d["nonzero_gradient"] += norm > 0
            d["squared_norm"] += norm**2
    for d in groups.values():
        d["norm_before_clip"] = math.sqrt(d.pop("squared_norm"))
    return groups


def backward_batch(model, rows, k, step=0, trace_path=None):
    device = next(model.parameters()).device
    totals = {n: 0.0 for n in LOSS_NAMES}
    for row in rows:
        batch = tensor_batch(row, model, device)
        observed = []
        expected_labels = batch["labels"][:, 1:].reshape(-1)
        def observe_ce(module, inputs, output):
            logits_in, labels_in = inputs
            if labels_in.shape == expected_labels.shape and torch.equal(labels_in, expected_labels):
                observed.append(dict(value=float(output.detach().float()),
                    logits_dtype=str(logits_in.dtype), loss_dtype=str(output.dtype)))
        hook = model.loss_fct.register_forward_hook(observe_ce)
        try:
            with amp(model):
                out = model(**batch, step=step, step_ratio=step/750)
        finally:
            hook.remove()
        assert torch.isfinite(out["loss"]), row["id"]
        # Independently verify the answer CE uses logits before each next-token label.
        flat_logits = out["logits"][:, :-1].reshape(-1, out["logits"].shape[-1])
        with torch.no_grad():
            manual = F.cross_entropy(flat_logits.float(), expected_labels)
            with amp(model):
                same_context = F.cross_entropy(flat_logits, expected_labels)
        passed = torch.allclose(same_context.float(), out["ce_loss"].float(), rtol=1e-4, atol=1e-5)
        trace = dict(id=row["id"], update=step, returned_ce=float(out["ce_loss"]),
            recomputed_fp32=float(manual), recomputed_matching_autocast=float(same_context),
            absolute_difference=float((same_context.float()-out["ce_loss"].float()).abs()),
            loss_module_observations=observed, logits_dtype=str(out["logits"].dtype),
            target_ids=expected_labels.detach().cpu().tolist(), passed=bool(passed))
        if trace_path is not None and (step == 0 or not passed):
            append(trace_path, trace)
        assert passed, "Answer CE mismatch; no optimizer update for this batch. Details: " + json.dumps(trace)
        (out["loss"] / len(rows)).backward()
        for n in LOSS_NAMES:
            totals[n] += float(out[n].detach()) / len(rows)
        del manual, out
    return totals


@torch.inference_mode()
def evaluate(model, rows, cfg, update, output):
    model.eval()
    correct, exact, active, all_exact, tf_nll, tf_correct, tf_tokens = 0, 0, 0, 0, 0., 0, 0
    with amp(model):
        for row in rows:
            out = generate(model, row["question"], cfg["latent_steps"], cfg["decode_max_tokens"], explanations=True)
            gold = extract(str(row["answer"]))
            out.update(id=row["id"], update=update, question=row["question"], reference=row["cot"],
                       gold=gold, prediction=extract(out["raw_output"]))
            out["correct"] = out["prediction"] == gold
            correct += out["correct"]
            zs = latent_vectors(model, row["question"], cfg["latent_steps"])
            details = []
            for slot, target in enumerate(targets(row, model.tokenizer, cfg["latent_steps"])):
                if target == [model.tokenizer.pad_token_id]: continue
                scores = teacher_score(model, zs[slot], target)
                eq = out["decoded_steps"][slot]["token_ids"] == target[:-1]
                active += 1
                exact += eq
                tf_nll += scores["nll_sum"]
                tf_correct += scores["correct_tokens"]
                tf_tokens += scores["n_tokens"]
                details.append(dict(slot=slot, target=model.tokenizer.decode(target), free_exact=eq, teacher_forcing=scores))
            all_exact += all(d["free_exact"] for d in details)
            out["active_step_checks"] = details
            append(output / "predictions.jsonl", out)
    summary = dict(update=update, n=len(rows), answer_correct=correct, answer_em=correct/len(rows),
        free_exact_slots=exact, active_slots=active, all_active_steps_exact_examples=all_exact,
        teacher_token_accuracy=tf_correct/tf_tokens, teacher_mean_nll=tf_nll/tf_tokens,
        precision="FP32 master weights + BF16 autocast; no weight casting between evaluations")
    append(output / "evaluations.jsonl", summary)
    model.train()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", type=Path, default=ROOT / "e3_runs/shared_init.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    assert 1 <= args.updates <= 200, "Debug hard cap: 200 optimizer updates"
    assert not (args.audit_only and args.resume)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    initial = torch.load(args.init, map_location="cpu", weights_only=False, mmap=True)
    assert "updates" not in initial, "Use shared_init.pt, NOT trained control or pilot"
    cfg = initial["config"]
    assert cfg == json.loads((ROOT / "e3/config.json").read_text()), "Training config changed"
    assert cfg["gradient_accumulation"] == 8 and cfg["micro_batch_size"] == 1 and cfg["latent_steps"] == 4
    for name, expected in initial["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Original implementation changed: {name}"
    train_path = ROOT / "e3/data/train.jsonl"
    manifest = json.loads((ROOT / "e3/data/manifest.json").read_text())
    assert sha(train_path) == manifest["splits"]["train"]["sha256"]
    all_rows = read_rows(train_path)
    rows = [all_rows[i] for i in INDICES]
    assert [r["source_index"] for r in rows] == INDICES
    assert len({extract(str(r["answer"])) for r in rows}) == 8
    contract = dict(kind="debug_overfit8_not_formal_control", config=cfg, updates=args.updates,
        audit_only=args.audit_only, init_sha256=sha(args.init), train_sha256=sha(train_path),
        source_sha256={n:sha(ROOT / f"e3/{n}") for n in ("overfit8.py", "diagnose.py", "common.py", "train.py")},
        sample_indices=INDICES, scheduler="original 750-update schedule, truncated at debug cap",
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert json.loads((args.output / "contract.json").read_text()) == contract
    else:
        assert not list(args.output.iterdir()), "Choose an empty, independent debug directory"
        write_json(args.output / "contract.json", contract)
        write_json(args.output / "samples.json", rows)
    model = build(cfg).cuda()
    model.load_state_dict(initial["model"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    restore_rng(initial["rng"])
    start = time.perf_counter()
    write_json(args.output / "interface_audit.json", interface_audit(model, rows, cfg["latent_steps"]))
    params = [v for v in model.parameters() if v.requires_grad]
    if not args.resume:
        model.train()
        model.zero_grad(set_to_none=True)
        losses = backward_batch(model, rows, cfg["latent_steps"], trace_path=args.output / "ce_audit.jsonl")
        groups = gradient_report(model)
        write_json(args.output / "gradient_audit.json", dict(losses=losses, groups=groups,
            note="Group-level nonzero gradients required. Some LoRA A gradients can be zero at zero-B initialization."))
        assert all(g["finite"] and g["norm_before_clip"] > 0 for g in groups.values()), "Invalid/missing group gradient; inspect gradient_audit.json"
        model.zero_grad(set_to_none=True)
        restore_rng(initial["rng"])
        print("PASS: interface, answer CE alignment, and all three gradient groups", flush=True)
        if args.audit_only:
            write_json(args.output / "status.json", dict(status="AUDIT_PASS", optimizer_updates=0,
                seconds=time.perf_counter()-start))
            return
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    def scale(step):
        if step < cfg["warmup_steps"]: return (step+1)/cfg["warmup_steps"]
        return max(0.,(750-step)/(750-cfg["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    done = 0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        assert state["contract"] == contract
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        done = state["updates"]
        restore_rng(state["rng"])
        del state
        # Preserve complete checkpoint-boundary records only, including after a partial final log write.
        for name in ("train_log.jsonl", "predictions.jsonl", "evaluations.jsonl"):
            path = args.output / name
            if path.exists():
                lines = path.read_text(encoding="utf8").splitlines()
                keep = []
                for i, line in enumerate(lines):
                    try: item = json.loads(line)
                    except json.JSONDecodeError:
                        if i == len(lines)-1: break
                        raise
                    if item["update"] <= done: keep.append(item)
                path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in keep),encoding="utf8")
    else:
        print(json.dumps(evaluate(model,rows,cfg,0,args.output)),flush=True)
    (args.output / "environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),encoding="utf8")
    # Snapshot one representative trainable tensor per group to verify actual optimizer changes.
    probe_names = {}
    for name, value in model.named_parameters():
        g = group_name(name)
        if value.requires_grad and g != "other" and g not in probe_names:
            if g != "lora" or "lora_B" in name: probe_names[g] = name
    start_update = done
    for step in range(done, args.updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = list(rows)
        random.Random(cfg["seed"]+step).shuffle(batch)
        losses = backward_batch(model,batch,cfg["latent_steps"],step,trace_path=args.output / "ce_audit.jsonl")
        groups = gradient_report(model) if step == start_update or (step+1)%20 == 0 else None
        if groups is not None:
            assert all(g["finite"] and g["norm_before_clip"]>0 for g in groups.values())
        grad = torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"],error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        done = step+1
        record = dict(update=done,losses=losses,lr=lr,grad_norm=float(grad),gradient_groups=groups)
        append(args.output / "train_log.jsonl",record)
        if done%10 == 0: print(json.dumps(record),flush=True)
        if done%20 == 0 or done == args.updates:
            summary = evaluate(model,rows,cfg,done,args.output)
            named = dict(model.named_parameters())
            changes = {g: float((named[name].detach().cpu()-initial["model"][name]).norm()) for g,name in probe_names.items()}
            write_json(args.output / "parameter_changes.json",dict(update=done,probe_names=probe_names,norm_delta_from_shared_init=changes))
            assert all(v>0 for v in changes.values()), "A representative parameter failed to update"
            save(args.output / "last.pt",dict(kind="debug_only_not_formal_control",config=cfg,
                contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),rng=rng_state(),updates=done))
            write_json(args.output / "summary.json",summary)
            print(json.dumps(summary),flush=True)
    with (args.output / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(dict(start_update=start_update,end_update=done,seconds=time.perf_counter()-start,
            note="Includes local audit/evaluation/save after model load; rental hours tracked separately"))+"\n")
    write_json(args.output / "status.json",dict(status="COMPLETE",updates=done,kind="debug_not_formal"))
    print("COMPLETE",args.output,flush=True)


if __name__ == "__main__":
    main()
