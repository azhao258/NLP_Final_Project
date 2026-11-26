"""Paired validation-only free generation; no gold CoT in either prompt."""
import argparse
import json
import random
import re
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e3'))
from common import ROOT, load_checkpoint, generate, read_rows, sha, write_json
from e1.metrics import extract


@torch.inference_mode()
def reference_generate(model, question, max_tokens):
    device = next(model.parameters()).device
    ids = torch.tensor([model.tokenizer.encode(question.strip())], device=device)
    capacity = model.codi.config.max_position_embeddings
    budget = min(max_tokens, capacity - ids.shape[1])
    if budget <= 0:
        raise ValueError('Question exhausts model context')
    past, tokens, stopped = None, [], False
    for _ in range(budget):
        out = model.codi(input_ids=ids, past_key_values=past, use_cache=True)
        value = out.logits[0, -1, :model.codi.config.vocab_size - 1].argmax().item()
        past = out.past_key_values
        if value == model.tokenizer.eos_token_id:
            stopped = True
            break
        tokens.append(value)
        ids = torch.tensor([[value]], device=device)
    return dict(raw_output=model.tokenizer.decode(tokens), token_ids=tokens,
                stop_reason='eos' if stopped else 'length', budget=budget)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, choices=(32, 64, 200), default=32)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--max-reference-tokens', type=int, default=512)
    args = p.parse_args()
    assert args.max_reference_tokens > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest = json.loads((ROOT/'e3/data/manifest.json').read_text(encoding='utf8'))
    path = ROOT/'e3/data/validation.jsonl'
    assert sha(path) == manifest['splits']['validation']['sha256']
    rows = read_rows(path)
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.n))
    selected = [rows[i] for i in indices]
    args.output.mkdir(parents=True, exist_ok=False)
    model, ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt['config']
    model.to(torch.bfloat16).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(args.output/'config.json', dict(checkpoint_sha256=sha(args.checkpoint),
        updates=ckpt.get('updates'), config=cfg, seed=args.seed, ids=[r['id'] for r in selected],
        data_sha256=sha(path), dtype='BF16 weights + BF16 autocast',
        max_reference_tokens=args.max_reference_tokens, latent_steps=cfg['latent_steps'],
        source_sha256={n: sha(ROOT/n) for n in ('e5/p3_reference_probe.py', 'e3/common.py', 'e1/metrics.py')},
        note='Question-only explicit prompt; no bot token, gold CoT, or answer. Different token budgets; not a latency comparison.'))
    totals = dict(n=len(selected), reference_correct=0, latent_correct=0,
                  reference_only_correct=0, latent_only_correct=0,
                  reference_capped=0, reference_parse_failures=0)
    for row in selected:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ref = reference_generate(model, row['question'], args.max_reference_tokens)
            latent = generate(model, row['question'], cfg['latent_steps'], cfg['answer_max_tokens'])
        gold = extract(str(row['answer']))
        assert gold is not None
        # Require an answer delimiter so a standalone intermediate number is not scored as an answer.
        rp = extract(ref['raw_output']) if re.search(r'The answer is\s*:|####', ref['raw_output'], re.I) else None
        lp = extract(latent['raw_output'])
        rc, lc = rp == gold, lp == gold
        for key, value in dict(reference_correct=rc, latent_correct=lc,
                              reference_only_correct=rc and not lc, latent_only_correct=lc and not rc,
                              reference_capped=ref['stop_reason']=='length', reference_parse_failures=rp is None).items():
            totals[key] += int(value)
        with (args.output/'pairs.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(dict(id=row['id'], gold=gold, reference=ref, latent=latent,
                reference_prediction=rp, latent_prediction=lp, reference_correct=rc, latent_correct=lc))+'\n')
        print(row['id'], rc, lc, flush=True)
    write_json(args.output/'summary.json', totals)
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
