"""Paired validation-only P1 audit from downloaded exports; no torch or model load."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e4.analyze import compare, equation, normalize, read_rows, sha, write_csv, write_json
from e1.metrics import extract
from e5.source_history import verify_recorded_source

MATCHES = ('strict_match', 'conservative_equivalent', 'operand_match', 'operator_match', 'result_match')


def validation(directory, gold, manifest):
    cfg = json.loads((directory/'config.json').read_text(encoding='utf8'))
    assert cfg['split'] == 'validation' and cfg['limit'] == 200 and cfg['explain']
    assert cfg['train_updates'] == 750 and cfg['precision'] == 'BF16 weights and autocast'
    assert cfg['data_manifest_sha256'] == sha(ROOT/'e3/data/manifest.json')
    assert cfg['inference_source_sha256'] == sha(ROOT/'e3/common.py')
    pred, dec = [read_rows(directory/name) for name in ('validation.predictions.jsonl','validation.decoded.jsonl')]
    ids = [r['id'] for r in gold]
    assert len(ids) == len(set(ids)) == 200
    assert ids == manifest['splits']['validation']['ids'] == [r['id'] for r in pred] == [r['id'] for r in dec]
    slots = []
    for g,p,d in zip(gold,pred,dec):
        assert g['question'] == p['question'] == d['question']
        assert g['cot'] == p['cot'] == d['reference']
        assert Fraction(str(g['answer'])) == Fraction(str(d['gold']))
        assert p['prediction'] == d['prediction'] == extract(p['raw_output'])
        assert p['correct'] == d['correct'] == (p['prediction'] == extract(str(g['answer'])))
        assert len(d['decoded_steps']) == len(d['supervision_slots']) == 5
        refs = re.findall(r'<<.*?>>', g['cot'])
        refs = refs if len(refs)<=5 else refs[:4]+[''.join(refs[4:])]
        assert [normalize(x) for x in d['supervision_slots'] if x!='[PAD]'] == [normalize(x) for x in refs]
        for i,(step,ref) in enumerate(zip(d['decoded_steps'],d['supervision_slots'])):
            assert step['slot']==i and step['consumed_by_reasoner']==(i<4)
            match = compare(step['text'],ref) if ref!='[PAD]' else {}
            slots.append(dict(id=g['id'],slot=i,valid=ref!='[PAD]',reference=ref,text=step['text'],
                arithmetic_valid=equation(step['text'])['arithmetic_valid'],capped=step['capped'],
                **{key:match.get(key) for key in MATCHES}))
    valid = [s for s in slots if s['valid']]
    summary = dict(n=200,correct=sum(p['correct'] for p in pred),valid_slots=len(valid),
        generated_slots=len(slots),arithmetic_valid=sum(s['arithmetic_valid'] is True for s in slots),
        arithmetic_invalid=sum(s['arithmetic_valid'] is False for s in slots),
        arithmetic_unparsed=sum(s['arithmetic_valid'] is None for s in slots),
        explanation_capped=sum(s['capped'] for s in slots),
        **{key:sum(s[key] is True for s in valid) for key in MATCHES})
    summary['slots'] = []
    for i in range(5):
        subset=[s for s in slots if s['slot']==i]
        frequency=Counter(normalize(s['text']) for s in subset)
        summary['slots'].append(dict(slot=i,unique_outputs=len(frequency),top=frequency.most_common(3),
            valid=sum(s['valid'] for s in subset),
            **{key:sum(s[key] is True for s in subset) for key in MATCHES}))
    return cfg,pred,slots,summary


def diagnostics(directory, eval_cfg):
    cfg=json.loads((directory/'config.json').read_text(encoding='utf8'))
    samples=json.loads((directory/'samples.json').read_text(encoding='utf8'))
    summary=json.loads((directory/'summary.json').read_text(encoding='utf8'))
    rows=read_rows(directory/'interventions.jsonl')
    grads=read_rows(directory/'gradients.jsonl')
    assert cfg['checkpoint_sha256']==eval_cfg['checkpoint_sha256'] and cfg['config']==eval_cfg['config']
    assert cfg['updates']==750 and cfg['n']==32 and cfg['seed']==11 and not cfg['smoke']
    assert cfg['data_manifest_sha256']==eval_cfg['data_manifest_sha256']
    assert all(summary['checks'].values())
    for name,expected in cfg['source_sha256'].items():
        verify_recorded_source(name, expected)
    assert len(rows)==len(grads)==64
    result={}
    for split in ('train','validation'):
        group=[r for r in rows if r['split']==split]
        gg=[r for r in grads if r['split']==split]
        ids=[r['id'] for r in samples[split]]
        assert len(ids)==len(set(ids))==32
        assert ids==[r['id'] for r in group]==[r['id'] for r in gg]
        source={r['id']:r for r in read_rows(ROOT/f'e3/data/{split}.jsonl')}
        assert all(source[r['id']]==r for r in samples[split])
        assert [r['donor_id'] for r in group]==ids[1:]+ids[:1]
        answer_deltas=[r['swapped']['answer_nll']-r['normal']['answer_nll'] for r in group]
        ss=[s for r in group for s in r['slots']]
        for r in group:
            for s in r['slots']:
                assert abs(s['nll_delta']-(s['swapped_score']['mean_nll']-s['normal_score']['mean_nll']))<1e-8
        computed=dict(n=32,normal_correct=sum(r['normal']['correct'] for r in group),
            swapped_correct=sum(r['swapped']['correct'] for r in group),
            changed_numeric_answers=sum(r['normal']['prediction']!=r['swapped']['prediction'] for r in group),
            mean_answer_nll_delta=st.mean(answer_deltas),valid_explanation_slots=len(ss),
            changed_explanations=sum(s['normal_free']['token_ids']!=s['swapped_free']['token_ids'] for s in ss),
            normal_exact_slots=sum(s['normal_exact'] for s in ss),swapped_exact_slots=sum(s['swapped_exact'] for s in ss),
            mean_explanation_nll_delta=st.mean(s['nll_delta'] for s in ss))
        for key,value in computed.items():
            assert abs(value-summary[split][key])<1e-8, key
        for groupname in ('shared','reasoner','projector'):
            for key,value in summary[split]['gradients'][groupname]['mean_norms'].items():
                assert abs(value-st.mean(g['stats'][groupname]['norms'][key] for g in gg))<1e-8
        computed.update(answer_nll_delta_median=st.median(answer_deltas),
            original_answer_nll_better=sum(d>0 for d in answer_deltas),
            original_explanation_nll_better=sum(s['nll_delta']>0 for s in ss),
            mean_normal_answer_nll=st.mean(r['normal']['answer_nll'] for r in group),
            mean_normal_explanation_nll=st.mean(s['normal_score']['mean_nll'] for s in ss),
            gradients=summary[split]['gradients'])
        changed=[s for s in ss if s['normal_free']['token_ids']!=s['swapped_free']['token_ids']]
        computed['changed_explanations_equivalent']=sum(compare(s['normal_free']['text'],s['swapped_free']['text'])['conservative_equivalent'] is True for s in changed)
        computed['changed_explanations_not_equivalent']=len(changed)-computed['changed_explanations_equivalent']
        result[split]=computed
    return cfg,samples,rows,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'server_backup_p1')
    parser.add_argument('--output',type=Path,default=ROOT/'e5/p1_analysis')
    args=parser.parse_args()
    assert not args.output.exists(), 'Choose a new output directory'
    manifest=json.loads((ROOT/'e3/data/manifest.json').read_text(encoding='utf8'))
    assert sha(ROOT/'e3/data/validation.jsonl')==manifest['splits']['validation']['sha256']
    gold=read_rows(ROOT/'e3/data/validation.jsonl')
    configs,predictions,steps,samples,interventions={},{},{},{},{}
    result=dict(validation={},diagnostics={})
    for role,ev,diag in [('control','p1_control_validation','p0_control_v1'),('p1','p1_validation','p0_p1_v1')]:
        configs[role],predictions[role],steps[role],result['validation'][role]=validation(args.input/ev,gold,manifest)
        dc,samples[role],interventions[role],result['diagnostics'][role]=diagnostics(args.input/diag,configs[role])
        if role=='control':
            diag_config=dc
        else:
            for key in ('gradient_mode','intervention_mode','source_sha256','data_manifest_sha256','gpu','torch'):
                assert dc[key]==diag_config[key], key
    assert samples['control']==samples['p1']
    frozen=json.loads((ROOT/'e3/frozen_v1.json').read_text(encoding='utf8'))
    assert configs['control']['checkpoint_sha256']==frozen['checkpoint_sha256']
    assert configs['p1']['checkpoint_sha256']!=configs['control']['checkpoint_sha256']
    for key in configs['control']:
        if key!='checkpoint_sha256':
            assert configs['control'][key]==configs['p1'][key], key
    pairs=[]
    for a,b in zip(predictions['control'],predictions['p1']):
        pairs.append(dict(id=a['id'],gold=a['answer'],control=a['prediction'],p1=b['prediction'],
            control_correct=a['correct'],p1_correct=b['correct']))
    result['answer_pairs']=dict(both_correct=sum(p['control_correct'] and p['p1_correct'] for p in pairs),
        control_only=sum(p['control_correct'] and not p['p1_correct'] for p in pairs),
        p1_only=sum(not p['control_correct'] and p['p1_correct'] for p in pairs),
        changed_numeric_answers=sum(p['control']!=p['p1'] for p in pairs))
    slot_pairs=[]
    for a,b in zip(steps['control'],steps['p1']):
        assert (a['id'],a['slot'],a['reference'])==(b['id'],b['slot'],b['reference'])
        slot_pairs.append(dict(id=a['id'],slot=a['slot'],reference=a['reference'],valid=a['valid'],
            control=a['text'],p1=b['text'],control_strict=a['strict_match'],p1_strict=b['strict_match'],
            control_arithmetic=a['arithmetic_valid'],p1_arithmetic=b['arithmetic_valid']))
    result['slot_pairs']=dict(changed=sum(normalize(p['control'])!=normalize(p['p1']) for p in slot_pairs),
        changed_but_equivalent=sum(normalize(p['control'])!=normalize(p['p1']) and compare(p['control'],p['p1'])['conservative_equivalent'] is True for p in slot_pairs),
        strict_gained=sum(p['p1_strict'] is True and p['control_strict'] is not True for p in slot_pairs),
        strict_lost=sum(p['control_strict'] is True and p['p1_strict'] is not True for p in slot_pairs))
    result['limitations']=['Single seed; exploratory validation comparison.',
        'Evaluation BF16 weights; P0 FP32 master + BF16 autocast: compare within each protocol.',
        'Weights/training logs not downloaded: checkpoint hashes cross-checked between exports, not rehashed from model files.',
        'No new test evaluation; historic baseline selection used test results.',
        'P1 changes timing and cumulative explanation weight (mean coefficient 0.85).',
        'Cross-question swap sensitivity is not evidence of faithful reasoning; no significance claim.']
    args.output.mkdir(parents=True)
    write_json(args.output/'summary.json',result)
    write_csv(args.output/'answer_pairs.csv',pairs)
    write_csv(args.output/'step_pairs.csv',slot_pairs)
    write_json(args.output/'provenance.json',dict(
        inputs={str(p.relative_to(ROOT)):sha(p) for p in args.input.rglob('*') if p.is_file()},
        source={n:sha(ROOT/n) for n in ('e5/analyze_p1.py','e4/analyze.py','e1/metrics.py')},
        checkpoints={role:cfg['checkpoint_sha256'] for role,cfg in configs.items()},
        validation_sha256=sha(ROOT/'e3/data/validation.jsonl'),checks='All alignment and recomputation assertions passed'))
    print(json.dumps(dict(validation=result['validation'],answer_pairs=result['answer_pairs'],slot_pairs=result['slot_pairs']),indent=2))
    print('COMPLETE',args.output)


if __name__=='__main__':
    main()
