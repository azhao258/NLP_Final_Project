"""Validation-only reference teacher-forcing and step retrieval diagnostics."""
import argparse
import json
import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3.common import ROOT, load_checkpoint, tensor_batch, targets, read_rows, sha, write_json
from e5.p4.p4_alignment import batch_positions


def retrieval(z, h, metadata):
    scores = F.normalize(z.float(), dim=-1) @ F.normalize(h.float(), dim=-1).T
    results = []
    for i, item in enumerate(metadata):
        for scope in ('same_question', 'other_question_same_slot'):
            candidates = [j for j, m in enumerate(metadata) if j != i
                and m['reference'] != item['reference'] and
                ((scope == 'same_question' and m['id'] == item['id']) or
                 (scope == 'other_question_same_slot' and m['id'] != item['id'] and m['slot'] == item['slot']))]
            if not candidates:
                continue
            positive = scores[i, i].item()
            negatives = scores[i, candidates]
            best = candidates[negatives.argmax().item()]
            # Ties do not count as unique retrieval successes.
            results.append(dict(**item, scope=scope, candidates=len(candidates)+1,
                positive_cosine=positive, hardest_negative_cosine=scores[i,best].item(),
                margin=positive-scores[i,best].item(),
                unique_top1=bool((scores[i,i] > negatives + 1e-6).all()),
                rank=1+int((negatives >= scores[i,i]-1e-6).sum()),
                negative_id=metadata[best]['id'], negative_slot=metadata[best]['slot']))
    return results


@torch.inference_mode()
def probe(model, row, k):
    b = tensor_batch(row, model, 'cuda')
    out = model.codi(input_ids=b['ref_input_ids'], output_hidden_states=True)
    logits = out.logits[0,:-1].float()
    labels = b['ref_labels'][0,1:]
    ce = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    # Each logit at t predicts token t+1. Exclude fixed answer prefix and EOS.
    indices = torch.arange(1, b['ref_input_ids'].shape[1], device='cuda')
    answer_start = b['ref_answer_position'].item()
    prefix_len = len(model.tokenizer.encode('The answer is:'))
    masks = dict(cot=(indices < answer_start-prefix_len) & (labels != -100),
                 answer=(indices >= answer_start) & (labels != model.tokenizer.eos_token_id))
    stats = {'id': row['id']}
    for name, mask in masks.items():
        stats[name+'_tokens'] = int(mask.sum())
        stats[name+'_nll_sum'] = ce[mask].sum().item()
        stats[name+'_token_correct'] = int((logits.argmax(-1)[mask] == labels[mask]).sum())
    positions = batch_positions(b, model.tokenizer, k+1)[0]
    teacher = out.hidden_states[-1][0].detach()
    q = model.codi(input_ids=b['encoder_input_ids'], output_hidden_states=True, use_cache=True)
    latents = [model.prj(q.hidden_states[-1][:,-1:])]
    for _ in range(k):
        q = model.codi(inputs_embeds=latents[-1], past_key_values=q.past_key_values,
                       output_hidden_states=True, use_cache=True)
        latents.append(model.prj(q.hidden_states[-1][:,-1:]))
    refs = targets(row, model.tokenizer, k)
    z, h, meta = [], [], []
    for slot, position in enumerate(positions.tolist()):
        if position < 0:
            continue
        z.append(latents[slot][0,0].float().cpu())
        h.append(teacher[position].float().cpu())
        ref = ''.join(model.tokenizer.decode(refs[slot]).replace('<|endoftext|>', '').split())
        meta.append(dict(id=row['id'], slot=slot, consumed=slot<k, reference=ref))
    return stats, z, h, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, choices=[32,64,200], default=200)
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a CUDA GPU supporting BF16')
    path = ROOT/'e3/data/validation.jsonl'
    manifest = json.loads((ROOT/'e3/data/manifest.json').read_text())
    assert sha(path) == manifest['splits']['validation']['sha256']
    rows = read_rows(path)
    rows = [rows[i] for i in sorted(random.Random(11).sample(range(len(rows)),args.n))]
    args.output.mkdir(parents=True, exist_ok=False)
    model, ckpt = load_checkpoint(args.checkpoint)
    model.to(torch.bfloat16).eval()
    assert ckpt['config']['latent_steps'] == 4
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(args.output/'config.json', dict(checkpoint_sha256=sha(args.checkpoint),
        updates=ckpt.get('updates'), data_sha256=sha(path), ids=[r['id'] for r in rows],
        precision='BF16 weights/autocast; FP32 CE and cosine', seed=11,
        source_sha256={n:sha(ROOT/n) for n in ['e5/check_reference_retrieval.py','e3/common.py','e5/p4/p4_alignment.py']},
        warning='Gold-conditioned diagnostic, not free-generation accuracy or proof of causal faithfulness.'))
    stats, zs, hs, metadata = [], [], [], []
    for row in rows:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result,z,h,m = probe(model,row,4)
        stats.append(result);zs.extend(z);hs.extend(h);metadata.extend(m)
        print(row['id'], flush=True)
    z,h = torch.stack(zs),torch.stack(hs)
    scores = retrieval(z,h,metadata)
    summary = dict(n=len(rows), valid_slots=len(metadata), reference={}, retrieval={})
    for name in ('cot','answer'):
        count=sum(s[name+'_tokens'] for s in stats)
        summary['reference'][name]=dict(tokens=count,
            mean_nll=sum(s[name+'_nll_sum'] for s in stats)/count if count else None,
            token_accuracy=sum(s[name+'_token_correct'] for s in stats)/count if count else None)
    for scope in ('same_question','other_question_same_slot'):
        for subset in ('all','consumed','readout'):
            selected=[s for s in scores if s['scope']==scope and
                      (subset=='all' or s['consumed']==(subset=='consumed'))]
            summary['retrieval'][scope+'/'+subset]=dict(n=len(selected),
                unique_top1=sum(s['unique_top1'] for s in selected)/len(selected) if selected else None,
                mean_margin=sum(s['margin'] for s in selected)/len(selected) if selected else None)
    for name,data in [('reference',stats),('retrieval',scores)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf8')
    torch.save(dict(latents=z,reference=h,metadata=metadata),args.output/'representations.pt')
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
