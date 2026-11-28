"""No-update v1 answer sampling diagnostic. Run from project root."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e3.common import load_checkpoint, generate, read_rows, sha, write_json
from e1.metrics import extract


def groups_summary(rows):
    n=len(rows);g=len(rows[0]['samples']);flat=[s for r in rows for s in r['samples']]
    counts=[sum(s['correct'] for s in r['samples']) for r in rows]
    return dict(n=n,group_size=g,total_samples=n*g,
        greedy_correct=sum(r['greedy']['correct'] for r in rows),
        sampled_correct=sum(counts),sample_accuracy=sum(counts)/(n*g),
        groups_any_correct=sum(k>0 for k in counts),
        groups_mixed_rewards=sum(0<k<g for k in counts),
        groups_all_wrong=sum(k==0 for k in counts),groups_all_correct=sum(k==g for k in counts),
        reward_variance_mean=sum((k/g)*(1-k/g) for k in counts)/n,
        parse_failures=sum(s['prediction'] is None for s in flat),
        capped=sum(s['stop_reason']=='length' for s in flat),
        mean_unique_texts=sum(len({s['raw_output'] for s in r['samples']}) for r in rows)/n,
        mean_unique_parsed_answers=sum(len({s['prediction'] for s in r['samples'] if s['prediction'] is not None}) for r in rows)/n,
        note='Training-subset diagnostic, not test accuracy. Any-correct is sampled coverage, not deployable greedy accuracy.')


def weight_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def sample(model,question,generator,temperature,max_tokens=64):
    device=next(model.parameters()).device;tok=model.tokenizer
    ids=torch.tensor([tok.encode(question.strip())+[model.bot_id]],device=device)
    assert ids.shape[1]+4+max_tokens <= model.codi.config.max_position_embeddings
    out=model.codi(input_ids=ids,output_hidden_states=True,use_cache=True)
    z=model.prj(out.hidden_states[-1][:,-1:])
    for i in range(4):
        out=model.codi(inputs_embeds=z,past_key_values=out.past_key_values,output_hidden_states=True,use_cache=True)
        if i<3:z=model.prj(out.hidden_states[-1][:,-1:])
    ids=torch.tensor([[model.eot_id]],device=device);tokens=[];stopped=False
    for _ in range(max_tokens):
        out=model.codi(input_ids=ids,past_key_values=out.past_key_values,use_cache=True)
        # Match original evaluator's vocabulary restriction, including its special-token policy.
        logits=out.logits[0,-1,:model.codi.config.vocab_size-1].float()/temperature
        value=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator).item()
        tokens.append(value)
        if value==tok.eos_token_id:stopped=True;break
        ids=torch.tensor([[value]],device=device)
    visible=tokens[:-1] if stopped else tokens
    return dict(raw_output=tok.decode(visible),token_ids=tokens,stop_reason='eos' if stopped else 'length')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'e3_runs/control/last.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[4,64],default=64)
    args=p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Requires CUDA with BF16'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    path=ROOT/'e3/data/train.jsonl'
    manifest=json.loads((ROOT/'e3/data/manifest.json').read_text())
    assert sha(path)==manifest['splits']['train']['sha256']
    frozen=json.loads((ROOT/'e3/frozen_v1.json').read_text())
    assert sha(args.checkpoint)==frozen['checkpoint_sha256'],'Use frozen v1 checkpoint'
    source=read_rows(path);assert len(source)==2000
    indices=sorted(random.Random(11).sample(range(2000),64))[:args.n]
    rows=[source[i] for i in indices]
    assert all(extract(str(r['answer'])) is not None for r in rows)
    args.output.mkdir(parents=True,exist_ok=False)
    model,ckpt=load_checkpoint(args.checkpoint)
    assert ckpt['config']['latent_steps']==4
    model.to(torch.bfloat16).eval().requires_grad_(False)
    before=weight_hash(model)
    write_json(args.output/'config.json',dict(n=args.n,ids=[r['id'] for r in rows],indices=indices,
        group_size=8,temperature=1.0,top_p=1.0,top_k=None,max_tokens=64,latent_steps=4,
        split_seed=11,sampling_seed_rule='110000+source_index*8+candidate_index',
        checkpoint_sha256=sha(args.checkpoint),data_sha256=sha(path),
        precision='BF16 weights + autocast; FP32 sampling probabilities',
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__).resolve(),ROOT/'e3/common.py',ROOT/'e1/metrics.py']},
        no_update=True))
    results=[]
    for index,row in zip(indices,rows):
        gold=extract(str(row['answer']))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            greedy=generate(model,row['question'],4,64)
            candidates=[]
            for j in range(8):
                gen=torch.Generator(device='cuda').manual_seed(110000+index*8+j)
                s=sample(model,row['question'],gen,1.0)
                prediction=extract(s['raw_output'])
                s.update(prediction=prediction,correct=prediction==gold,reward=int(prediction==gold),candidate=j)
                candidates.append(s)
        prediction=extract(greedy['raw_output']);greedy.update(prediction=prediction,correct=prediction==gold)
        record=dict(id=row['id'],question=row['question'],gold=gold,greedy=greedy,samples=candidates)
        results.append(record)
        with (args.output/'groups.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
        print(row['id'],'sampled correct',sum(s['correct'] for s in candidates),'/8',flush=True)
    after=weight_hash(model);assert before==after,'Unexpected parameter/buffer change'
    assert all(p.grad is None for p in model.parameters())
    summary=groups_summary(results)
    summary.update(weights_unchanged=True,state_sha256_before=before,state_sha256_after=after)
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
