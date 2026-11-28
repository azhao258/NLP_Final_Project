"""CPU-only E6/E7 extension. Reuses original scoring; never edits original exports."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--p5', type=Path, required=True, help='P5 directory containing runs/ and results/')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo, p5, out = args.repo.resolve(), args.p5.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    sys.path.insert(0, str(repo))
    from E6.analyze_test_suite import check_rows, delta
    from E7.scoring import summarize, csv_write
    from e5.problem_types import classify
    from e5.protocol import sha, read_json, read_rows, DATASETS
    from e4.analyze import normalize, SLOT_LABELS

    models = ['v1', 'prefix_mask', 'p1', 'p2', 'p3', 'p4', 'p5']
    paths = {m: repo/'E6/exports'/m for m in models}
    types = read_json(repo/'e5/problem_types.json')
    labels = {(r['dataset'], r['id']): r['strata'] for r in types['rows']}
    configs, data, input_hashes = {}, {}, {}
    base_cfg = read_json(paths['v1']/'config.json')
    source = base_cfg['source_sha256']['e3/common.py']
    for m, d in paths.items():
        cfg = read_json(d/'config.json'); configs[m] = cfg
        assert cfg['train_updates'] == 750
        policy = cfg.get('policy', cfg)
        for key in ['precision', 'batch_size', 'greedy', 'warmups']:
            assert policy[key] == base_cfg['policy'][key], (m, key)
        assert cfg.get('inference_source_sha256', cfg.get('source_sha256', {}).get('e3/common.py')) == source
        budget = cfg.get('policy', cfg.get('config'))
        assert budget['latent_steps'] == 4 and budget['answer_max_tokens'] == budget['decode_max_tokens'] == 64
        files = {f.name: f.read_bytes() for f in d.glob('*') if f.is_file()}
        input_hashes[m] = {f.name: sha(f) for f in d.glob('*') if f.is_file()}
        data[m] = {}
        for ds in DATASETS:
            golds = read_rows(repo/f'e1/data/{ds}.jsonl')
            assert sha(repo/f'e1/data/{ds}.jsonl') == types['data_sha256'][ds]
            assert all(classify(g) == labels[ds,g['id']] for g in golds)
            data[m][ds] = check_rows(m, ds, files, golds)
    assert configs['v1']['checkpoint_sha256'] == read_json(repo/'e3/frozen_v1.json')['checkpoint_sha256']
    for stage in ['validation', 'retrieval']:
        assert read_json((p5/'results/validation' if stage=='validation' else repo/'E7/p5_retrieval')/'config.json')['checkpoint_sha256'] == configs['p5']['checkpoint_sha256']

    tables = {k: [] for k in ['summary', 'steps', 'paired_questions', 'problem_types', 'slots', 'categories']}
    for ds in DATASETS:
        br, bs = data['v1'][ds]
        bg = defaultdict(list)
        for s in bs: bg[s['id']].append(s)
        buckets = defaultdict(set)
        for r in br:
            buckets['dataset', 'all'].add(r['id'])
            for g in labels[ds,r['id']]: buckets[g['axis'],g['label']].add(r['id'])
        for m in models:
            rows, steps = data[m][ds]
            tables['summary'].append(dict(model=m, dataset=ds, **summarize(rows,steps)))
            tables['steps'].extend(steps)
            sg = defaultdict(list)
            for s in steps: sg[s['id']].append(s)
            valid = [s for s in steps if s['reference_available']]
            counts = Counter(l for s in valid for l in s['labels'])
            tables['categories'].extend(dict(model=m,dataset=ds,label=l,count=counts[l] if valid else None,denominator=len(valid)) for l in SLOT_LABELS)
            for slot in range(5):
                ss = [s for s in steps if s['slot']==slot]
                freq = Counter(normalize(s['generated']) for s in ss)
                tables['slots'].append(dict(model=m,dataset=ds,slot=slot,consumed=slot<4,unique_outputs=len(freq),most_common=freq.most_common(1)[0][0],most_common_count=freq.most_common(1)[0][1],outputs=len(ss)))
            if m=='v1': continue
            for a,b in zip(br,rows):
                assert a['reference_tokens']==b['reference_tokens']
                sa,sb = summarize([a],bg[a['id']]),summarize([b],sg[b['id']])
                pair = dict(model=m,dataset=ds,id=a['id'],question=a['question'],reference=a.get('reference'),gold=a['answer'],v1_prediction=a['prediction'],prediction=b['prediction'],v1_correct=a['correct'],correct=b['correct'],answer_outcome='gained' if b['correct'] and not a['correct'] else 'lost' if a['correct'] and not b['correct'] else 'both_correct' if a['correct'] else 'both_wrong',v1_explanations=[s['generated'] for s in bg[a['id']]],explanations=[s['generated'] for s in sg[b['id']]],strata=labels[ds,a['id']])
                for k in ['strict_match_count','conservative_equivalent_count','whole_expression_any_slot_rate']:
                    pair['v1_'+k],pair[k],pair[k+'_delta']=sa[k],sb[k],delta(sa[k],sb[k])
                tables['paired_questions'].append(pair)
            for (axis,label),ids in sorted(buckets.items()):
                aa=[r for r in br if r['id'] in ids];bb=[r for r in rows if r['id'] in ids]
                sa=summarize(aa,[s for s in bs if s['id'] in ids]);sb=summarize(bb,[s for s in steps if s['id'] in ids])
                tables['problem_types'].append(dict(model=m,dataset=ds,axis=axis,label=label,small_group=len(ids)<20,gained=sum(not a['correct'] and b['correct'] for a,b in zip(aa,bb)),lost=sum(a['correct'] and not b['correct'] for a,b in zip(aa,bb)),em_delta_pp=100*(sb['em']-sa['em']),**{'v1_'+k:v for k,v in sa.items()},**sb))
    # Regression against B's six-model analysis: every original summary field is unchanged.
    old = read_rows(repo/'E6/legacy_six_model_analysis/summary.jsonl')
    for r in old:
        new=next(x for x in tables['summary'] if (x['model'],x['dataset'])==(r['model'],r['dataset']))
        assert all(new[k]==r[k] for k in new), (r['model'],r['dataset'])
    out.mkdir(parents=True)
    for name, rows in tables.items():
        csv_write(out/(name+'.csv'),rows)
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')

    # Purposive cases: changes in answers and steps, plus a disagreement/control case.
    candidates=[]
    for m in models[1:]:
        pool=[r for r in tables['paired_questions'] if r['model']==m]
        selected={}
        rules={
            'answer_gain':lambda r:r['answer_outcome']=='gained',
            'answer_loss':lambda r:r['answer_outcome']=='lost',
            'step_gain':lambda r:(r['conservative_equivalent_count_delta'] or 0)>0,
            'step_loss':lambda r:(r['conservative_equivalent_count_delta'] or 0)<0,
            'correct_answer_incomplete_explanation':lambda r:r['dataset']=='gsm8k' and r['correct'],
        }
        for tag,fn in rules.items():
            found=next((r for r in pool if fn(r)),None)
            if found:
                key=(found['dataset'],found['id'])
                if key not in selected:selected[key]=dict(found,selection_reasons=[])
                selected[key]['selection_reasons'].append(tag)
        if m=='p5':
            for r in pool:
                if r['answer_outcome'] in ['gained','lost']:
                    key=(r['dataset'],r['id'])
                    if key not in selected:selected[key]=dict(r,selection_reasons=['all_p5_answer_changes'])
        candidates.extend(selected.values())
    for i,r in enumerate(candidates):
        r.update(case_id=f'C{i+1:02}',review_status='AI_PRELIMINARY_HUMAN_PENDING',human_reviewer='',human_agrees='',human_notes='')
    csv_write(out/'case_review.csv',candidates)
    (out/'case_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidates),encoding='utf8')
    provenance=dict(models=models,no_training=True,no_model_inference=True,original_six_model_summary_regression='PASS',input_configs=configs,input_files_sha256=input_hashes,types_sha256=sha(repo/'e5/problem_types.json'),analysis_sources={s:sha(repo/s) for s in ['E7/scoring.py','E6/analyze_test_suite.py','e4/analyze.py','e5/problem_types.py','e1/metrics.py']},script_sha256=sha(Path(__file__)),checkpoint_verification='Identities from export configs; weight files not loaded/rehashed',case_sampling='Purposive, not random/blinded; all P5 answer gains/losses included',limitations=['Single seed; v1 selection used test results','Historical latency; no same-session seven-model speedup claim','GSM-Hard step alignment N/A; SVAMP whole expression only','K=4; fifth decoded slot is readout-only','Cases are AI preliminary review, human sign-off pending','Group axes overlap; small groups flagged'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(json.dumps({k:len(v) for k,v in tables.items()}));print('cases',len(candidates));print('PASS: frozen golds, settings, scores, six-model regression')


if __name__=='__main__':main()

