"""Recompute paired P2 validation/P0 results and audit downloaded training logs."""
import argparse
import json
import math
import re
import statistics as st
from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.analyze_p1 import ROOT, validation, diagnostics, read_rows, sha, write_json, write_csv
from e5.protocol import sample_order
from e5.source_history import verify_recorded_source
import hashlib


def first_operands(text):
    """Numeric first operands in equation order; a missing merged equation fails."""
    chunks = re.findall(r'<<([^<>]*)>>',text)
    values = []
    for chunk in chunks:
        m = re.match(r'\s*\(*\s*([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))',chunk)
        if not m:
            return None
        values.append(str(Fraction(m.group(1).replace(',',''))))
    return values or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'e5/p2_analysis')
    args = parser.parse_args()
    assert not args.output.exists(), 'Choose a fresh output directory'
    def read(path):
        return json.loads(path.read_text(encoding='utf8'))
    manifest = read(ROOT/'e3/data/manifest.json')
    gold = read_rows(ROOT/'e3/data/validation.jsonl')
    assert sha(ROOT/'e3/data/validation.jsonl')==manifest['splits']['validation']['sha256']
    frozen = read(ROOT/'e3/frozen_v1.json')
    result = dict(validation={},diagnostics={},pairs={})
    configs, samples, predictions, slots, diag_configs = {},{},{},{},{}
    inputs = []
    for role,base,ev,diag in [('control','server_backup_p1','p1_control_validation','p0_control_v1'),
                             ('p1','server_backup_p1','p1_validation','p0_p1_v1'),
                             ('p2','server_backup_p2','p2_validation','p0_p2_v1')]:
        evpath, dgpath = ROOT/base/ev, ROOT/base/diag
        inputs.extend(p for d in (evpath,dgpath) for p in d.iterdir() if p.is_file())
        configs[role],predictions[role],slots[role],result['validation'][role] = validation(evpath,gold,manifest)
        dc,samples[role],_,result['diagnostics'][role] = diagnostics(dgpath,configs[role])
        diag_configs[role]=dc
        valid = [s for s in slots[role] if s['valid']]
        assert all(first_operands(s['reference']) is not None for s in valid)
        result['validation'][role]['ordered_first_operand_match'] = sum(first_operands(s['text'])==first_operands(s['reference']) for s in valid)
        # Recompute gradient cosines as well as norms, rather than trust the summary.
        grads=read_rows(dgpath/'gradients.jsonl')
        for split in ('train','validation'):
            gg=[g for g in grads if g['split']==split]
            for group in ('shared','reasoner','projector'):
                for pair,stats in result['diagnostics'][role][split]['gradients'][group]['pairs'].items():
                    values=[g['stats'][group]['pairs'][pair]['cosine'] for g in gg]
                    values=[v for v in values if v is not None]
                    assert stats['n']==len(values)
                    if values:
                        assert abs(stats['mean_cosine']-st.mean(values))<1e-8
                        assert abs(stats['negative_fraction']-sum(v<0 for v in values)/len(values))<1e-8
        if role!='control':
            assert samples[role]==samples['control']
            for key in configs['control']:
                if key!='checkpoint_sha256':
                    assert configs[role][key]==configs['control'][key],key
            for key in ('gradient_mode','intervention_mode','source_sha256','data_manifest_sha256','gpu','torch'):
                assert dc[key]==diag_configs['control'][key],key
    assert configs['control']['checkpoint_sha256']==frozen['checkpoint_sha256']
    contract=read(ROOT/'server_backup_p2/training/contract.json')
    assert contract['config']==configs['p2']['config']==frozen['training_contract']['config']
    assert contract['initialization_sha256']==frozen['initialization_sha256']
    assert contract['pilot_updates']==0 and contract['updates']==contract['full_budget_updates']==750
    assert contract['variant']['operand_weight']==2 and contract['variant']['explain_loss_factor']==1
    assert not contract['variant']['prefix_mask']
    for name,expected in contract['source_sha256'].items():
        verify_recorded_source(name, expected)
    for split,expected in contract['data_files_sha256'].items():
        assert sha(ROOT/f'e3/data/{split}.jsonl')==expected
    assert sha(ROOT/'assets/manifest.json')==contract['assets_manifest_sha256']
    assert contract['order_sha256']==hashlib.sha256(json.dumps(sample_order(contract['config'])).encode()).hexdigest()
    logs=read_rows(ROOT/'server_backup_p2/training/train_log.jsonl')
    sessions=read_rows(ROOT/'server_backup_p2/training/sessions.jsonl')
    assert [r['update'] for r in logs]==list(range(1,751))
    for r in logs:
        assert r['explain_loss_factor']==1 and r['operand_weight']==2
        assert all(math.isfinite(v) for v in r['losses'].values()) and math.isfinite(r['grad_norm'])
        assert abs(r['losses']['loss']-sum(v for k,v in r['losses'].items() if k!='loss'))<1e-4
        assert r['weighted_explain_loss']==r['losses']['explain_loss']
        step=r['update']-1
        expected_lr=5e-5*((step+1)/38 if step<38 else (750-step)/(750-38))
        assert abs(r['lr']-expected_lr)<1e-12
    result['training']=dict(updates=len(logs),epoch_validation=[dict(update=r['update'],**r['validation']) for r in logs if 'validation' in r],
        sessions=sessions,final_log=logs[-1],source_hashes_match=True,
        precision_note='Training validation uses FP32 parameters + BF16 autocast; standalone exports use BF16 parameters + autocast. No paired 200-row precision rerun performed.')
    pair_rows=[]
    for baseline in ('control','p1'):
        pairs=[dict(id=a['id'],baseline=baseline,gold=a['answer'],baseline_prediction=a['prediction'],p2_prediction=b['prediction'],baseline_correct=a['correct'],p2_correct=b['correct']) for a,b in zip(predictions[baseline],predictions['p2'])]
        pair_rows.extend(pairs)
        gained=[p['id'] for p in pairs if p['p2_correct'] and not p['baseline_correct']]
        lost=[p['id'] for p in pairs if p['baseline_correct'] and not p['p2_correct']]
        result['pairs'][baseline]=dict(both_correct=sum(p['baseline_correct'] and p['p2_correct'] for p in pairs),gained_ids=gained,lost_ids=lost,
            changed_answers=sum(p['baseline_prediction']!=p['p2_prediction'] for p in pairs),
            strict_gained=sum(b['strict_match'] is True and a['strict_match'] is not True for a,b in zip(slots[baseline],slots['p2'])),
            strict_lost=sum(a['strict_match'] is True and b['strict_match'] is not True for a,b in zip(slots[baseline],slots['p2'])))
    step_rows=[]
    for triplet in zip(slots['control'],slots['p1'],slots['p2']):
        a=triplet[0]
        record=dict(id=a['id'],slot=a['slot'],valid=a['valid'],reference=a['reference'])
        for name,s in zip(('control','p1','p2'),triplet):
            assert (s['id'],s['slot'],s['reference'])==(a['id'],a['slot'],a['reference'])
            record.update({name:s['text'],name+'_strict':s['strict_match'],name+'_first_operand':first_operands(s['text'])==first_operands(s['reference']) if s['valid'] else None})
        step_rows.append(record)
    result['limitations']=['One seed; validation exploration only; historical test-influenced baseline selection remains.',
        'Checkpoint identity cross-checked between evaluation and P0 exports; model weights not downloaded/rehashed.',
        'P0 uses common UNWEIGHTED explanation loss, not P2 training loss.',
        'Mean NLL change and swap sensitivity do not prove faithful reasoning.',
        'First-operand metric added post hoc: ordered numeric list equality per valid slot (including merged tails).']
    args.output.mkdir(parents=True)
    write_json(args.output/'summary.json',result)
    write_csv(args.output/'answer_pairs.csv',pair_rows)
    write_csv(args.output/'step_pairs.csv',step_rows)
    inputs.extend((ROOT/'server_backup_p2/training').iterdir())
    write_json(args.output/'provenance.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        sources={n:sha(ROOT/n) for n in ('e5/analyze_p2.py','e5/analyze_p1.py','e4/analyze.py','e1/metrics.py')},
        checkpoints={k:v['checkpoint_sha256'] for k,v in configs.items()},checks='All assertions passed'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics',)},indent=2))


if __name__=='__main__':
    main()
