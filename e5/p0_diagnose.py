"""P0: loss gradients and causal latent interventions; never updates parameters."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e3"))
import torch
import torch.nn.functional as F
from common import ROOT, build, generate, read_rows, sha, targets, tensor_batch, write_json
from diagnose import latent_vectors, teacher_score, append
from e1.metrics import extract

LOSSES = ("ce_loss", "distill_loss", "ref_ce_loss", "explain_loss")


def differentiable_forward(model):
    """Expose upstream loss tensors in memory, without editing upstream source."""
    original = type(model).forward
    source = textwrap.dedent(inspect.getsource(original))
    for name in ("ce_loss_total", "distill_loss_total", "ref_ce_loss", "explain_loss_total"):
        old = f"{name} = {name}.detach()"
        assert source.count(old) == 1, f"Upstream changed: {old}"
        source = source.replace(old, "pass  # P0 retains the existing loss graph")
    namespace = dict(original.__globals__)
    exec(compile(source, "<p0_upstream_forward>", "exec"), namespace)
    return namespace["forward"].__get__(model, type(model))


def gradient_stats(output, named):
    params = [p for _, p in named]
    gradients = {}
    for name in LOSSES:
        loss = output[name]
        grads = (torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                 if torch.is_tensor(loss) and loss.requires_grad else [None] * len(params))
        # CPU storage bounds GPU memory; unused coordinates are exactly zero.
        gradients[name] = [g.detach().float().cpu() if g is not None else None for g in grads]
        assert all(g is None or torch.isfinite(g).all() for g in gradients[name]), f"Nonfinite {name} gradients"
    result = {}
    for group in ("shared", "reasoner", "projector"):
        indices = [i for i, (n, _) in enumerate(named)
                   if group == "shared" or n.startswith("codi." if group == "reasoner" else "prj.")]
        def dot(a, b):
            return sum(float((x * y).sum(dtype=torch.float64)) for i in indices
                       if (x := gradients[a][i]) is not None and (y := gradients[b][i]) is not None)
        norms = {name: dot(name, name) ** .5 for name in LOSSES}
        pairs = {}
        for a, b in itertools.combinations(LOSSES, 2):
            denom = norms[a] * norms[b]
            pairs[a + "__" + b] = dict(dot=dot(a, b), cosine=dot(a, b)/denom if denom else None)
        result[group] = dict(norms=norms, pairs=pairs)
    return result


@torch.inference_mode()
def decode_step(model, z, limit):
    emb = model.get_embd(model.codi, model.model_name)
    x, ids = z, []
    for _ in range(limit):
        token = model.decoder(inputs_embeds=x).logits[0, -1].argmax().item()
        if token == model.tokenizer.eos_token_id:
            return dict(text=model.tokenizer.decode(ids), token_ids=ids, capped=False)
        ids.append(token)
        x = torch.cat([x, emb(torch.tensor([[token]], device=z.device))], 1)
    return dict(text=model.tokenizer.decode(ids), token_ids=ids, capped=True)


@torch.inference_mode()
def reasoner_intervention(model, row, k, limit, donor=None):
    """Replace z0..z(K-1) where consumed; retain recipient question KV only.

    Every latent call rebuilds its KV under the intervention. zK is a readout,
    not an additional input. Donor KV and donor question are never transplanted.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([model.tokenizer.encode(row["question"].strip()) + [model.bot_id]], device=device)
    out = model.codi(input_ids=ids, output_hidden_states=True, use_cache=True)
    z = model.prj(out.hidden_states[-1][:, -1:])
    for slot in range(k):
        out = model.codi(inputs_embeds=z if donor is None else donor[slot],
                         past_key_values=out.past_key_values, output_hidden_states=True, use_cache=True)
        z = model.prj(out.hidden_states[-1][:, -1:])
    cache = out.past_key_values
    assert isinstance(cache, tuple), "Review cache cloning before using a different model"
    # Gold answer NLL: recompute from the intervened cache, including the same prefix as training.
    gold_ids = model.tokenizer.encode("The answer is: " + str(row["answer"]).strip()) + [model.tokenizer.eos_token_id]
    teacher_ids = torch.tensor([[model.eot_id] + gold_ids[:-1]], device=device)
    scores = model.codi(input_ids=teacher_ids, past_key_values=cache, use_cache=True).logits.float()
    losses = F.cross_entropy(scores[0], torch.tensor(gold_ids, device=device), reduction="none")
    prefix_len = len(model.tokenizer.encode("The answer is:"))
    answer_nll = float(losses[prefix_len:].mean())
    # GPT-2 legacy tuple cache is immutable; prohibit mutable cache implementations.
    ids = torch.tensor([[model.eot_id]], device=device)
    tokens = []
    for _ in range(limit):
        out = model.codi(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = out.logits[0, -1, :model.codi.config.vocab_size - 1].argmax().item()
        tokens.append(token)
        if token == model.tokenizer.eos_token_id:
            break
        ids = torch.tensor([[token]], device=device)
    raw = model.tokenizer.decode(tokens[:-1] if tokens[-1] == model.tokenizer.eos_token_id else tokens)
    return dict(raw_output=raw, token_ids=tokens, prediction=extract(raw),
                correct=extract(raw) == extract(str(row["answer"])), answer_nll=answer_nll)


def summarize(records, grads):
    result = {}
    for split in ("train", "validation"):
        rows = [r for r in records if r["split"] == split]
        slots = [s for r in rows for s in r["slots"]]
        entry = dict(n=len(rows), normal_correct=sum(r["normal"]["correct"] for r in rows),
            swapped_correct=sum(r["swapped"]["correct"] for r in rows),
            changed_answer_tokens=sum(r["normal"]["token_ids"] != r["swapped"]["token_ids"] for r in rows),
            changed_numeric_answers=sum(r["normal"]["prediction"] != r["swapped"]["prediction"] for r in rows),
            mean_answer_nll_delta=statistics.mean(r["swapped"]["answer_nll"]-r["normal"]["answer_nll"] for r in rows),
            valid_explanation_slots=len(slots),
            normal_exact_slots=sum(s["normal_exact"] for s in slots),
            swapped_exact_slots=sum(s["swapped_exact"] for s in slots),
            changed_explanations=sum(s["normal_free"]["token_ids"] != s["swapped_free"]["token_ids"] for s in slots),
            mean_explanation_nll_delta=statistics.mean(s["nll_delta"] for s in slots) if slots else None)
        entry["gradients"] = {}
        selected = [g for g in grads if g["split"] == split]
        for group in ("shared", "reasoner", "projector"):
            entry["gradients"][group] = dict(
                mean_norms={name: statistics.mean(g["stats"][group]["norms"][name] for g in selected) for name in LOSSES},
                pairs={})
            for a, b in itertools.combinations(LOSSES, 2):
                key = a + "__" + b
                vals = [g["stats"][group]["pairs"][key]["cosine"] for g in selected]
                vals = [v for v in vals if v is not None]
                entry["gradients"][group]["pairs"][key] = dict(n=len(vals),
                    mean_cosine=statistics.mean(vals) if vals else None,
                    negative_fraction=sum(v < 0 for v in vals)/len(vals) if vals else None)
        result[split] = entry
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--smoke", action="store_true", help="Allow 2 samples for an integration smoke check")
    args = p.parse_args()
    assert args.n in (32, 64) or (args.smoke and args.n == 2)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert not args.output.exists(), "Choose a new output directory"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = checkpoint["config"]
    manifest_path = ROOT / "e3/data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    samples = {}
    for split in ("train", "validation"):
        path = ROOT / f"e3/data/{split}.jsonl"
        assert sha(path) == manifest["splits"][split]["sha256"]
        rows = read_rows(path)
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.n))
        samples[split] = [rows[i] for i in indices]
        assert len({r["id"] for r in samples[split]}) == args.n
    args.output.mkdir(parents=True)
    write_json(args.output / "samples.json", samples)
    write_json(args.output / "config.json", dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        updates=checkpoint.get("updates"), config=cfg, n=args.n, seed=args.seed, smoke=args.smoke,
        data_manifest_sha256=sha(manifest_path), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        source_sha256={f:sha(ROOT/f) for f in ("e5/p0_diagnose.py", "e3/common.py", "e3/diagnose.py", "SIM-CoT/CODI/src/model.py")},
        gradient_mode="FP32 master + BF16 autocast, eval mode (dropout off), weighted upstream losses",
        intervention_mode="FP32 master + BF16 autocast; cyclic next-question donor within split",
        scope="Diagnostic only. No optimizer, no test data, no parameter updates."))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(cfg).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    forward = differentiable_forward(model)
    # Includes all trainable reasoner and projection coordinates, excludes auxiliary
    # decoder-only parameters. A task's unused coordinates contribute zero.
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith(("codi.", "prj."))]
    assert named
    write_json(args.output / "parameters.json", {n:list(p.shape) for n,p in named})
    versions = {n:p._version for n,p in model.named_parameters()}
    grad_records, records = [], []
    k = cfg["latent_steps"]
    assert k == 4
    for split, rows in samples.items():
        for i, row in enumerate(rows):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = forward(**tensor_batch(row, model, "cuda"), step=0, step_ratio=0.)
            assert torch.isfinite(output["loss"])
            assert torch.allclose(output["loss"], sum(output[x] for x in LOSSES), atol=1e-5)
            record = dict(split=split, id=row["id"], losses={n:float(output[n].detach()) if torch.is_tensor(output[n]) else output[n] for n in LOSSES},
                          stats=gradient_stats(output, named))
            grad_records.append(record)
            append(args.output / "gradients.jsonl", record)
            del output
            print(f"gradients {split} {i+1}/{args.n}", flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vectors = [latent_vectors(model, row["question"], k) for row in rows]
            for i, row in enumerate(rows):
                donor = (i + 1) % len(rows)
                normal = reasoner_intervention(model, row, k, cfg["answer_max_tokens"])
                identity = reasoner_intervention(model, row, k, cfg["answer_max_tokens"], vectors[i])
                upstream = generate(model, row["question"], k, cfg["answer_max_tokens"])
                assert normal == identity, "Self-swap must be exactly identical"
                assert normal["token_ids"] == upstream["token_ids"], "Normal path differs from existing inference"
                swapped = reasoner_intervention(model, row, k, cfg["answer_max_tokens"], vectors[donor])
                slots = []
                for slot, target in enumerate(targets(row, model.tokenizer, k)):
                    if target == [model.tokenizer.pad_token_id]:
                        continue
                    expected = target[:-1] if target[-1] == model.tokenizer.eos_token_id else target
                    nf = decode_step(model, vectors[i][slot], cfg["decode_max_tokens"])
                    sf = decode_step(model, vectors[donor][slot], cfg["decode_max_tokens"])
                    ns, ss = teacher_score(model, vectors[i][slot], target), teacher_score(model, vectors[donor][slot], target)
                    slots.append(dict(slot=slot, consumed_by_reasoner=slot < k, target=model.tokenizer.decode(target),
                        normal_free=nf, swapped_free=sf, normal_score=ns, swapped_score=ss,
                        normal_exact=nf["token_ids"] == expected, swapped_exact=sf["token_ids"] == expected,
                        nll_delta=ss["mean_nll"]-ns["mean_nll"]))
                record = dict(split=split, id=row["id"], donor_id=rows[donor]["id"], normal=normal, swapped=swapped, slots=slots)
                records.append(record)
                append(args.output / "interventions.jsonl", record)
                print(f"interventions {split} {i+1}/{args.n}", flush=True)
    assert all(p._version == versions[n] and p.grad is None for n,p in model.named_parameters())
    summary = summarize(records, grad_records)
    summary["checks"] = dict(no_parameter_updates=True, self_swap_identity=True, upstream_generation_parity=True)
    summary["limitations"] = "Single checkpoint, small fixed subsets, dropout disabled. Negative cosine is local conflict, not proof of training harm. Swap sensitivity is not proof of faithful reasoning; low sensitivity does not prove absence of latent information. z4 is readout-only. No test-set selection."
    write_json(args.output / "summary.json", summary)
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
