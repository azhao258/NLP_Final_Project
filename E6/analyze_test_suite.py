"""Offline CPU analysis of frozen v1, prefix masking and P1-P4 test exports.

Uses the original Part 2/3 scoring functions; never loads model weights.
Reads selected archive members in memory without extracting the archives.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.protocol import ROOT, DATASETS, read_json, read_rows, require, sha, write_json
from e5.problem_types import make_types
from E7.scoring import inspect_steps, summarize, csv_write
from e4.analyze import normalize, SLOT_LABELS
from e1.metrics import extract

MODELS = ('v1', 'prefix_mask', 'p1', 'p2', 'p3', 'p4')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_members(path, wanted):
    with tarfile.open(path) as archive:
        return {name: archive.extractfile(name).read() for name in wanted}


def export_names():
    return ['config.json', 'summary.csv'] + [
        f'{name}.{kind}.jsonl' for name in DATASETS for kind in ('predictions', 'decoded')]


def check_rows(model, name, files, golds):
    rows = [json.loads(line) for line in files[f'{name}.predictions.jsonl'].splitlines()]
    decoded = [json.loads(line) for line in files[f'{name}.decoded.jsonl'].splitlines()]
    ids = [r['id'] for r in golds]
    require(len(ids) == len(set(ids)) == 200, 'Expected 200 unique frozen gold IDs')
    require(ids == [r['id'] for r in rows] == [r['id'] for r in decoded],
            f'{model}/{name}: missing, duplicate or reordered IDs')
    steps = []
    for gold, pred, dec in zip(golds, rows, decoded):
        require(all(pred.get(k) == v for k, v in gold.items()), f'{model}: source mismatch')
        require(dec['question'] == gold['question'] and dec['reference'] == gold.get('reference')
                and str(dec['gold']) == str(gold['answer']), f'{model}: decoded source mismatch')
        answer = extract(pred['raw_output'])
        correct = answer is not None and answer == extract(str(gold['answer']))
        require(pred['prediction'] == dec['prediction'] == answer and
                pred['correct'] == dec['correct'] == correct, f'{model}: answer mismatch')
        require(pred['latent_steps'] == 4 and np.isfinite(pred['latency_seconds']) and
                pred['latency_seconds'] >= 0, f'{model}: budget/timing mismatch')
        require((pred['reference_tokens'] is None) == (gold.get('reference') is None),
                f'{model}: reference availability mismatch')
        ds = dec['decoded_steps']
        require([s['slot'] for s in ds] == list(range(5)) and
                [s['consumed_by_reasoner'] for s in ds] == [True]*4+[False],
                f'{model}: explanation slots mismatch')
        steps.extend(dict(model=model, dataset=name, id=gold['id'], **s)
                     for s in inspect_steps(gold, ds))
    return rows, steps


def load_inputs(args, types):
    names = export_names()
    prefixes = {'v1': 'e5_runs/paired_test_v1/control/',
                'prefix_mask': 'e5_runs/paired_test_v1/modified/'}
    archived = archive_members(args.paired_archive,
        [p+n for p in prefixes.values() for n in names+['complete.json']] +
        ['e5_runs/paired_analysis_v1/summary.json'])
    backfill_hash = sha(args.backfill_archive)
    require(backfill_hash == args.backfill_archive.with_suffix(args.backfill_archive.suffix+'.sha256')
            .read_text(encoding='utf8').split()[0], 'Backfill archive checksum mismatch')
    wanted = [f'e5_runs/{m}_test_bf16_20261126/{n}' for m in MODELS[2:] for n in names]
    backfill = archive_members(args.backfill_archive, wanted)
    audit = read_json(args.input / 'VERIFICATION.json')
    require(audit['status'] == 'PASS' and audit['archive_sha256'] == backfill_hash,
            'Local backup audit does not match archive')
    golds = {n: read_rows(ROOT / f'e1/data/{n}.jsonl') for n in DATASETS}
    configs, data, hashes = {}, {}, {}
    for model in MODELS:
        if model in prefixes:
            prefix = prefixes[model]
            files = {n: archived[prefix+n] for n in names}
            complete = json.loads(archived[prefix+'complete.json'])
            cfg = json.loads(files['config.json'])
            require(complete['n'] == 600 and complete['pairing_id'] == cfg['pairing_id'],
                    'Incomplete paired export')
            require(complete['files'] == {n: digest(b) for n, b in files.items()},
                    'Paired export checksum mismatch')
            require(cfg['types_sha256'] == sha(ROOT/'e5/problem_types.json'), 'Types hash mismatch')
            require(all(cfg['data_files_sha256'][n] == types['data_sha256'][n] for n in DATASETS),
                    'Paired source data hash mismatch')
            expected = read_json(ROOT/'e3/frozen_v1.json')['checkpoint_sha256'] if model == 'v1' else audit['checkpoint_sha256_verified'][model]
            require(cfg['checkpoint_sha256'] == expected, 'Wrong paired checkpoint')
        else:
            prefix = f'e5_runs/{model}_test_bf16_20261126/'
            files = {n: (args.input/prefix/n).read_bytes() for n in names}
            require(all(b == backfill[prefix+n] for n, b in files.items()),
                    f'{model}: local export differs from verified archive')
            cfg = json.loads(files['config.json'])
            require(cfg['checkpoint_sha256'] == audit['checkpoint_sha256_verified'][model],
                    f'{model}: checkpoint identity mismatch')
            require(cfg['split'] == 'test' and cfg['limit'] == 200 and cfg['explain'],
                    f'{model}: wrong evaluation split')
            require(cfg['data_manifest_sha256'] == sha(ROOT/'e1/data/manifest.json'),
                    f'{model}: dataset manifest mismatch')
            require(cfg['inference_source_sha256'] == configs['v1']['source_sha256']['e3/common.py'],
                    f'{model}: inference source mismatch')
            for key in ('precision', 'batch_size', 'greedy', 'warmups'):
                require(cfg[key] == configs['v1']['policy'][key], f'{model}: different {key}')
            for key in ('latent_steps', 'answer_max_tokens', 'decode_max_tokens'):
                require(cfg['config'][key] == configs['v1']['policy'][key], f'{model}: different {key}')
        require(cfg['train_updates'] == 750, f'{model}: unfinished training')
        configs[model] = cfg
        hashes[model] = {n: digest(b) for n, b in files.items()}
        data[model] = {n: check_rows(model, n, files, golds[n]) for n in DATASETS}
    require(configs['v1']['pairing_id'] == configs['prefix_mask']['pairing_id'], 'Paired session mismatch')
    # Reproduce every original summary metric, not merely the answer counts.
    for old in json.loads(archived['e5_runs/paired_analysis_v1/summary.json']):
        model = 'v1' if old['role'] == 'control' else 'prefix_mask'
        new = summarize(*data[model][old['dataset']])
        require(all(new[k] == v for k, v in old.items() if k not in ('role', 'dataset')),
                f'Original scoring regression: {model}/{old["dataset"]}')
    return configs, data, hashes


def delta(a, b):
    return None if a is None or b is None else b-a


def outcome(a, b):
    return 'improved' if b > a else 'degraded' if b < a else 'unchanged'


def run(args):
    require(not args.output.exists(), 'Use a new output directory; existing results are never overwritten')
    types = read_json(ROOT/'e5/problem_types.json')
    require(types == make_types(), 'Frozen problem labels or source data changed')
    configs, data, hashes = load_inputs(args, types)
    labels = {(r['dataset'], r['id']): r['strata'] for r in types['rows']}
    summaries, all_steps, pairs, strata, slots, categories = [], [], [], [], [], []
    for name in DATASETS:
        base_rows, base_steps = data['v1'][name]
        buckets = defaultdict(set)
        for row in base_rows:
            buckets['dataset', 'all'].add(row['id'])
            for group in labels[name, row['id']]:
                buckets[group['axis'], group['label']].add(row['id'])
        for model in MODELS:
            rows, steps = data[model][name]
            session = configs[model].get('pairing_id', f'separate_export:{model}')
            summaries.append(dict(model=model, dataset=name, timing_session=session,
                six_model_same_session=False,
                mean_reference_tokens=(sum(r['reference_tokens'] for r in rows if r['reference_tokens'] is not None) /
                    sum(r['reference_tokens'] is not None for r in rows)) if any(r['reference_tokens'] is not None for r in rows) else None,
                **summarize(rows, steps)))
            all_steps.extend(steps)
            counts = Counter(label for s in steps if s['reference_available'] for label in s['labels'])
            denom = sum(s['reference_available'] for s in steps)
            categories.extend(dict(model=model, dataset=name, label=label,
                count=counts[label] if denom else None, denominator=denom) for label in SLOT_LABELS)
            for i in range(5):
                local = [s for s in steps if s['slot'] == i]
                freq = Counter(normalize(s['generated']) for s in local)
                text, count = freq.most_common(1)[0]
                slots.append(dict(model=model, dataset=name, slot=i, consumed_by_reasoner=i<4,
                    unique_outputs=len(freq), most_common=text, most_common_count=count, template_share=count/len(local)))
            if model == 'v1':
                continue
            for left, right in zip(base_rows, rows):
                require(left['reference_tokens'] == right['reference_tokens'], 'Reference token counts differ')
                ls = [s for s in base_steps if s['id'] == left['id']]
                rs = [s for s in steps if s['id'] == left['id']]
                sa, sb = summarize([left], ls), summarize([right], rs)
                pair = dict(model=model, dataset=name, id=left['id'], question=left['question'],
                    gold=left['answer'], v1_prediction=left['prediction'], prediction=right['prediction'],
                    v1_correct=left['correct'], correct=right['correct'],
                    answer_outcome='gained' if right['correct'] and not left['correct'] else
                        'lost' if left['correct'] and not right['correct'] else
                        'both_correct' if left['correct'] else 'both_wrong',
                    strata=labels[name,left['id']], v1_explanations=[s['generated'] for s in ls],
                    explanations=[s['generated'] for s in rs])
                for key in ('strict_match_count', 'conservative_equivalent_count', 'whole_expression_any_slot_rate'):
                    pair['v1_'+key], pair[key] = sa[key], sb[key]
                    pair[key+'_delta'] = delta(sa[key], sb[key])
                pairs.append(pair)
            for (axis, label), ids in sorted(buckets.items()):
                left, right = [r for r in base_rows if r['id'] in ids], [r for r in rows if r['id'] in ids]
                sa = summarize(left, [s for s in base_steps if s['id'] in ids])
                sb = summarize(right, [s for s in steps if s['id'] in ids])
                group = dict(model=model, dataset=name, axis=axis, label=label,
                    gained=sum(b['correct'] and not a['correct'] for a,b in zip(left,right)),
                    lost=sum(a['correct'] and not b['correct'] for a,b in zip(left,right)),
                    em_delta_pp=100*(sb['em']-sa['em']), answer_change=outcome(sa['correct'],sb['correct']),
                    small_group=len(ids)<20, **{'v1_'+k:v for k,v in sa.items()}, **sb)
                for key in ('strict_match', 'operator_match', 'operand_match', 'result_match', 'conservative_equivalent'):
                    group[key+'_delta_count'] = delta(sa[key+'_count'], sb[key+'_count'])
                    d = delta(sa[key+'_rate'], sb[key+'_rate'])
                    group[key+'_delta_pp'] = None if d is None else 100*d
                    group[key+'_change'] = None if d is None else outcome(0,d)
                for key in ('whole_expression_slot0_rate','whole_expression_any_slot_rate'):
                    d = delta(sa[key],sb[key])
                    group[key+'_delta_pp'] = None if d is None else 100*d
                strata.append(group)
    args.output.mkdir(parents=True)
    for name, rows in [('summary',summaries),('steps',all_steps),('paired_questions',pairs),
                       ('problem_types',strata),('slots',slots),('categories',categories)]:
        csv_write(args.output/f'{name}.csv',rows)
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    write_json(args.output/'summary.json',summaries)
    limitations = [
        'Historical latency only: no complete six-model same-session benchmark and no cross-session speedup claim.',
        'GSM8K uses original strict and conservative step alignment; SVAMP has whole-expression compatibility only.',
        'GSM-Hard has no reference chain: reference-aligned metrics are N/A, not zero.',
        'K=4; five decoded slots include a final diagnostic slot not consumed by the reasoner.',
        'Overlapping grouping axes must not be added together; small groups are flagged.',
        'Single seed; v1 selection previously used test results; exploratory comparisons only.',
        'Decoded explanation matches and arithmetic validity do not establish causal latent faithfulness.',
        'Weights are not loaded or rehashed here; checkpoint identities use the previous verified backup audit.'
    ]
    write_json(args.output/'provenance.json',dict(input_configs=configs,input_files_sha256=hashes,
        types_sha256=sha(ROOT/'e5/problem_types.json'),
        source_sha256={p:sha(ROOT/p) for p in ['E6/analyze_test_suite.py','E7/scoring.py',
            'e5/problem_types.py','e4/analyze.py','e1/metrics.py']},
        original_summary_regression='PASS: all metrics for v1 and prefix_mask exactly reproduced',
        no_training=True,no_model_inference=True,limitations=limitations))
    lines = ['# Six-model fixed-test analysis','',
        'Offline rescoring of saved predictions; 200 questions per dataset, K=4.','',
        '| Dataset | Model | Correct /200 | Strict / reference slots | Compatible / reference slots |',
        '|---|---|---:|---:|---:|']
    for r in summaries:
        strict = 'N/A' if r['strict_match_count'] is None else f"{r['strict_match_count']}/{r['valid_reference_slots']}"
        compatible = 'N/A' if r['conservative_equivalent_count'] is None else f"{r['conservative_equivalent_count']}/{r['valid_reference_slots']}"
        lines.append(f"| {r['dataset']} | {r['model']} | {r['correct']} | {strict} | {compatible} |")
    lines += ['','## Paired answer changes versus v1','',
        '| Dataset | Model | Gained | Lost | Net correct |','|---|---|---:|---:|---:|']
    for r in strata:
        if r['axis']=='dataset':
            lines.append(f"| {r['dataset']} | {r['model']} | {r['gained']} | {r['lost']} | {r['correct']-r['v1_correct']:+d} |")
    lines += ['','## Problem groups','',
        'Full answer and explanation changes, including unchanged and degraded groups, are in problem_types.csv.',
        'Each axis is evaluated separately. Counts below overlap across axes; they are not counts of unique questions.','',
        '| Dataset | Model | EM improved / unchanged / degraded groups |','|---|---|---|']
    for name in DATASETS:
        for model in MODELS[1:]:
            c=Counter(r['answer_change'] for r in strata if r['dataset']==name and r['model']==model and r['axis']!='dataset')
            lines.append(f"| {name} | {model} | {c['improved']} / {c['unchanged']} / {c['degraded']} |")
    lines += ['','## Scope and limitations','']+['- '+s for s in limitations]
    lines += ['','summary.csv contains answer/reference lengths and mean, median and P95 historical latency.',
              'steps.csv contains all 18,000 decoded slot scores; paired_questions.csv covers all 3,000 model-versus-v1 question pairs.','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(args.output/'complete.json',dict(status='PASS',models=6,questions_per_model=600,
        scored_slots=len(all_steps),paired_questions=len(pairs),
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    print('PASS: frozen IDs, golds, answers, archive bytes, checkpoint identities, labels and original summary regression')
    print(f'COMPLETE {args.output}')
    for r in summaries:
        print(f"{r['dataset']:9} {r['model']:11} correct={r['correct']:2}/200 strict={r['strict_match_count']} compatible={r['conservative_equivalent_count']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'server_backup_eval_20261126')
    parser.add_argument('--paired-archive',type=Path,default=ROOT/'server_backup_part3/nlp_part3_backup.tar')
    parser.add_argument('--backfill-archive',type=Path,default=ROOT/'e5_eval_backfill_20261126.tar.gz')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
