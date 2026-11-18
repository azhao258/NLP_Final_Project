"""Read-only checkpoint diagnostics. No optimizer, backward, or training updates."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import random
import statistics
import torch
import torch.nn.functional as F
from common import ROOT, build, generate, read_rows, sha, targets, write_json
from e1.metrics import extract


def append(path, item):
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


@torch.inference_mode()
def latent_vectors(model, question, k):
    device = next(model.parameters()).device
    ids = torch.tensor([model.tokenizer.encode(question.strip())+[model.bot_id]], device=device)
    out = model.codi(input_ids=ids, output_hidden_states=True, use_cache=True)
    z = model.prj(out.hidden_states[-1][:, -1:])
    vectors = [z]
    for _ in range(k):
        out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values,
                         output_hidden_states=True, use_cache=True)
        z = model.prj(out.hidden_states[-1][:, -1:])
        vectors.append(z)
    return vectors


@torch.inference_mode()
def teacher_score(model, z, target):
    """First target token predicted by z, subsequent tokens by correct preceding tokens."""
    labels = torch.tensor([target], device=z.device)
    embedding = model.get_embd(model.codi, model.model_name)(labels)
    x = torch.cat([z, embedding], dim=1)
    logits = model.decoder(inputs_embeds=x).logits[:, :-1].float()
    assert logits.shape[1] == labels.shape[1]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    assert torch.isfinite(losses).all()
    match = logits.argmax(-1).eq(labels).flatten()
    return dict(n_tokens=len(target), nll_sum=float(losses.sum()), mean_nll=float(losses.mean()),
                correct_tokens=int(match.sum()), first_token_correct=bool(match[0]),
                first_token_nll=float(losses[0]))


def context(mode):
    return nullcontext() if mode == "fp32_full" else torch.autocast("cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Run on a BF16-capable CUDA GPU"
    assert 2 <= args.n <= 200
    args.output.mkdir(parents=True, exist_ok=True)
    assert not list(args.output.iterdir()), "Choose a NEW output directory"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = checkpoint["config"]
    manifest_path = ROOT / "e3/data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    samples = {}
    for name in ("train", "validation"):
        path = ROOT / f"e3/data/{name}.jsonl"
        assert sha(path) == manifest["splits"][name]["sha256"]
        rows = read_rows(path)
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.n))
        samples[name] = [rows[i] for i in indices]
    write_json(args.output / "samples.json", samples)
    write_json(args.output / "config.json", dict(checkpoint_sha256=sha(args.checkpoint),
        updates=checkpoint["updates"], training_config=cfg, seed=args.seed, n=args.n,
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        data_manifest_sha256=sha(manifest_path),
        training_manifest_hash_matches=sha(manifest_path)==checkpoint["contract"]["data_sha256"],
        source_sha256={n:sha(ROOT/f"e3/{n}") for n in ("diagnose.py","common.py")},
        modes={"fp32_full":"FP32 weights, no autocast, TF32 disabled",
               "fp32_amp":"FP32 weights, BF16 autocast (training-validation convention)",
               "bf16_amp":"BF16 weights, BF16 autocast (formal evaluation convention)"},
        note="Diagnostic samples only; no training and no formal latency benchmark."))
    model = build(cfg).cuda().eval()
    k = cfg["latent_steps"]
    answers = {}
    summary = {}
    for mode in ("fp32_full", "fp32_amp", "bf16_amp"):
        # Restore original checkpoint before each precision condition; never re-expand rounded BF16 weights.
        model.float()
        model.load_state_dict(checkpoint["model"], strict=True)
        if mode == "bf16_amp":
            model.to(torch.bfloat16)
        mode_rows = []
        with context(mode):
            for row in samples["validation"]:
                out = generate(model, row["question"], k, cfg["answer_max_tokens"])
                gold = extract(str(row["answer"]))
                assert gold is not None
                out.update(id=row["id"], gold=gold, prediction=extract(out["raw_output"]), mode=mode)
                out["correct"] = out["prediction"] == gold
                mode_rows.append(out)
                append(args.output/"precision.jsonl", out)
        answers[mode] = mode_rows
        summary[mode] = dict(n=len(mode_rows), correct=sum(x["correct"] for x in mode_rows),
                            parse_failures=sum(x["prediction"] is None for x in mode_rows))
        print(mode, summary[mode], flush=True)
    summary["precision_pairs"] = []
    for left, right in (("fp32_full","fp32_amp"),("fp32_amp","bf16_amp"),("fp32_full","bf16_amp")):
        pairs = list(zip(answers[left],answers[right]))
        summary["precision_pairs"].append(dict(left=left,right=right,
            changed_answers=sum(a["prediction"]!=b["prediction"] for a,b in pairs),
            changed_token_sequences=sum(a["token_ids"]!=b["token_ids"] for a,b in pairs)))
    # Model remains in the formal BF16 evaluation condition for checks 1 and 2.
    train_outputs = []
    comparisons = []
    with context("bf16_amp"):
        vectors = [latent_vectors(model,r["question"],k) for r in samples["train"]]
        for i,row in enumerate(samples["train"]):
            out = generate(model,row["question"],k,cfg["decode_max_tokens"],explanations=True)
            gold = extract(str(row["answer"]))
            assert gold is not None
            out.update(id=row["id"], question=row["question"], reference=row["cot"], gold=gold,
                       prediction=extract(out["raw_output"]))
            out["correct"] = out["prediction"] == gold
            train_outputs.append(out)
            append(args.output/"train_free.jsonl",out)
            donor = (i+1) % args.n
            for slot,target in enumerate(targets(row,model.tokenizer,k)):
                if target == [model.tokenizer.pad_token_id]:
                    continue
                normal = teacher_score(model,vectors[i][slot],target)
                swapped = teacher_score(model,vectors[donor][slot],target)
                expected = target[:-1] if target[-1]==model.tokenizer.eos_token_id else target
                record = dict(id=row["id"],donor_id=samples["train"][donor]["id"],slot=slot,
                    target=model.tokenizer.decode(target),normal=normal,swapped=swapped,
                    swapped_minus_normal_nll=swapped["mean_nll"]-normal["mean_nll"],
                    free_exact=out["decoded_steps"][slot]["token_ids"]==expected,
                    free_text=out["decoded_steps"][slot]["text"])
                comparisons.append(record)
                append(args.output/"teacher_forcing.jsonl",record)
            print("train diagnostic",i+1,"/",args.n,flush=True)
    summary["train_free"] = dict(n=args.n,correct=sum(x["correct"] for x in train_outputs),
        top_answers=Counter(x["prediction"] for x in train_outputs).most_common(5),
        unique_decoded_texts_by_slot=[len({x["decoded_steps"][s]["text"].strip() for x in train_outputs}) for s in range(k+1)])
    summary["decoder"] = dict(active_slots=len(comparisons),
        free_exact_slots=sum(x["free_exact"] for x in comparisons),
        mean_swapped_minus_normal_nll=statistics.mean(x["swapped_minus_normal_nll"] for x in comparisons),
        median_swapped_minus_normal_nll=statistics.median(x["swapped_minus_normal_nll"] for x in comparisons),
        normal_better_slots=sum(x["swapped_minus_normal_nll"]>0 for x in comparisons))
    for condition in ("normal","swapped"):
        n=sum(x[condition]["n_tokens"] for x in comparisons)
        summary["decoder"][condition]=dict(token_accuracy=sum(x[condition]["correct_tokens"] for x in comparisons)/n,
            mean_nll=sum(x[condition]["nll_sum"] for x in comparisons)/n)
    summary["interpretation"] = "Small sample diagnostic. Teacher-forcing accuracy benefits from true prefixes. A small swap effect supports insensitivity but does not prove latent information is absent. No significance threshold is imposed."
    write_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2))
    print("COMPLETE",args.output)


if __name__ == "__main__":
    main()
