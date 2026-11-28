import json
from pathlib import Path
import torch
import torch.nn.functional as F
from e5.p1_schedule import preflight as baseline_preflight
from e5.p4.p4_alignment import batch_positions
from e3.common import tensor_batch,targets,sha,ROOT
from P5.negatives import make_pairs

METHOD=dict(variant='p5_cross_question_margin_v1',alignment_factor=0.1,margin=0.02,
    seed=405105,negative='one deterministic train-only donor per anchor; same slot',
    filter='exact arithmetic valid, equal equation counts, disjoint numeric result sets',
    teacher='current reference network, eval mode, stop-gradient on positive and negative',
    slots='z0-z4; z4 readout-only; skip slots without eligible pair',
    extra_reference_forward=2,total_updates=750,
    note='Single added contrastive objective vs v1; not equal FLOPs. Not identical teacher sampling to P4.')
_pair=None

def preflight(init=None,require_weights=False):
    p=baseline_preflight(init,require_weights);p['variant']=METHOD.copy();return p

def prepare_pairs(rows,tokenizer):
    slots=[]
    for row in rows:
        seq=targets(row,tokenizer,4)
        slots.append(['' if s==[tokenizer.pad_token_id] else tokenizer.decode(s).replace('<|endoftext|>','').strip() for s in seq])
    pairs=make_pairs(rows,slots)
    frozen=json.loads(Path(__file__).with_name('train_negative_pairs.json').read_text(encoding='utf8'))
    assert frozen['training_sha256']==sha(ROOT/'e3/data/train.jsonl')
    assert len(frozen['pairs'])==len(pairs)
    for a,b in zip(pairs,frozen['pairs']):
        for key in ['anchor_id','donor_id','donor_index','slots']:
            assert a[key]==b[key],('Tokenizer pair mapping changed',a['anchor_id'],key)
    return pairs

def set_pair(rows,index,pair):
    global _pair
    assert pair['anchor_id']==rows[index]['id']
    _pair=(rows[pair['donor_index']] if pair['donor_index'] is not None else None,pair['slots'])

def margin_loss(z,pos,neg,margin=0.02):
    pos=pos.detach().float();neg=neg.detach().float();z=z.float()
    cp=F.cosine_similarity(z,pos,dim=-1);cn=F.cosine_similarity(z,neg,dim=-1)
    penalties=F.relu(margin+cn-cp)
    return penalties.mean(),cp.mean(),cn.mean(),(penalties>0).float().mean()

def training_forward(model,batch,completed_updates,factor=0.1):
    assert _pair is not None and model.num_latent==4
    latents=[]
    hook=model.prj.register_forward_hook(lambda m,a,o:latents.append(o))
    try:result=model(**batch,step=completed_updates,step_ratio=completed_updates/750)
    finally:hook.remove()
    assert len(latents)==5
    donor,slots=_pair
    base=result['loss'];zero=base*0
    raw=cp=cn=active=zero
    if slots and factor:
        nb=tensor_batch(donor,model,batch['ref_input_ids'].device)
        pp=batch_positions(batch,model.tokenizer)[0];np=batch_positions(nb,model.tokenizer)[0]
        assert all(pp[s]>=0 and np[s]>=0 for s in slots)
        # Both teacher sides use deterministic eval forwards; preserve flags and RNG.
        states=[(m,m.training) for m in model.codi.modules()]
        device=batch['ref_input_ids'].device
        try:
            model.codi.eval()
            with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []),torch.no_grad():
                pos=model.codi(input_ids=batch['ref_input_ids'],output_hidden_states=True,use_cache=False).hidden_states[-1][0,pp[slots]].detach()
                neg=model.codi(input_ids=nb['ref_input_ids'],output_hidden_states=True,use_cache=False).hidden_states[-1][0,np[slots]].detach()
        finally:
            for m,state in states:m.training=state
        z=torch.cat(latents,dim=1)[0,slots]
        raw,cp,cn,active=margin_loss(z,pos,neg,METHOD['margin'])
    result['loss']=base+factor*raw if factor else base
    result.update(base_loss=base.detach(),alignment_loss=raw.detach(),weighted_alignment_loss=(factor*raw).detach(),
        alignment_valid_slots=torch.tensor(len(slots),device=base.device),positive_cosine=cp.detach(),negative_cosine=cn.detach(),active_fraction=active.detach())
    return result
