"""Audit downloaded P3 exports and recompute paired validation-only results.

No model loading, training, test-set access, or modification of frozen sources.
Run without -O so alignment and provenance assertions remain active.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5.analyze_p1 import ROOT, validation, diagnostics, read_rows, sha, write_json, write_csv
from e5.analyze_p2 import first_operands
from e5.protocol import sample_order
from e5.source_history import verify_recorded_source
from e1.metrics import extract
from e4.analyze import equation


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def exact_p(gained, lost):
    """Two-sided exact McNemar/binomial p-value for discordant answer pairs."""
    n = gained + lost
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost)+1)) / 2**n) if n else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'e5/p3_analysis')
    args = parser.parse_args()
    assert __debug__, 'Do not run with -O'
    assert not args.output.exists(), 'Choose a fresh output directory'
    manifest = read(ROOT/'e3/data/manifest.json')
    frozen = read(ROOT/'e3/frozen_v1.json')
    gold = read_rows(ROOT/'e3/data/validation.jsonl')
    assert sha(ROOT/'e3/data/validation.jsonl') == manifest['splits']['validation']['sha256']
    cfgs, preds, slots, samples, dcfgs = {}, {}, {}, {}, {}
    result = dict(validation={}, diagnostics={}, pairs={}, reference_probes={})
    inputs = []
    for role, base, ev, dg in [
        ('control', 'server_backup_p1', 'p1_control_validation', 'p0_control_v1'),
        ('p1', 'server_backup_p1', 'p1_validation', 'p0_p1_v1'),
        ('p2', 'server_backup_p2', 'p2_validation', 'p0_p2_v1'),
        ('p3', 'server_backup_p3', 'p3_validation', 'p0_p3_v1')]:
        evpath, dgpath = ROOT/base/ev, ROOT/base/dg
        inputs.extend(p for d in (evpath, dgpath) for p in d.iterdir() if p.is_file())
        cfgs[role], preds[role], slots[role], result['validation'][role] = validation(evpath, gold, manifest)
        dcfgs[role], samples[role], _, result['diagnostics'][role] = diagnostics(dgpath, cfgs[role])
        valid = [s for s in slots[role] if s['valid']]
        assert all(first_operands(s['reference']) is not None for s in valid)
        v = result['validation'][role]
        v['ordered_first_operand_match'] = sum(first_operands(s['text']) == first_operands(s['reference']) for s in valid)
        v['answer_parse_failures'] = sum(p['prediction'] is None for p in preds[role])
        v['answer_capped'] = sum(p['stop_reason'] == 'length' for p in preds[role])
        with (evpath/'summary.csv').open(encoding='utf8', newline='') as f:
            tab = list(csv.DictReader(f))
        assert len(tab) == 1 and tab[0]['dataset'] == 'validation'
        assert int(tab[0]['n']) == v['n'] and abs(float(tab[0]['em'])-v['correct']/v['n']) < 1e-12
        assert int(tab[0]['parse_failures']) == v['answer_parse_failures']
        assert int(tab[0]['capped']) == v['answer_capped']
        grads = read_rows(dgpath/'gradients.jsonl')
        for split in ('train', 'validation'):
            gg = [g for g in grads if g['split'] == split]
            for group in ('shared', 'reasoner', 'projector'):
                for pair, stats in result['diagnostics'][role][split]['gradients'][group]['pairs'].items():
                    values = [g['stats'][group]['pairs'][pair]['cosine'] for g in gg]
                    values = [x for x in values if x is not None]
                    assert stats['n'] == len(values)
                    if values:
                        assert abs(stats['mean_cosine']-st.mean(values)) < 1e-8
                        assert abs(stats['negative_fraction']-sum(x < 0 for x in values)/len(values)) < 1e-8
        if role != 'control':
            assert samples[role] == samples['control']
            for key in cfgs['control']:
                if key != 'checkpoint_sha256':
                    assert cfgs[role][key] == cfgs['control'][key], key
            for key in ('gradient_mode', 'intervention_mode', 'source_sha256', 'data_manifest_sha256', 'gpu', 'torch'):
                assert dcfgs[role][key] == dcfgs['control'][key], key
    assert cfgs['control']['checkpoint_sha256'] == frozen['checkpoint_sha256']

    directory = ROOT/'server_backup_p3/training'
    inputs.extend(p for p in directory.iterdir() if p.is_file())
    contract = read(directory/'contract.json')
    assert contract['config'] == cfgs['p3']['config'] == frozen['training_contract']['config']
    assert contract['initialization_sha256'] == frozen['initialization_sha256']
    assert contract['pilot_updates'] == 0 and contract['updates'] == contract['full_budget_updates'] == 750
    variant = contract['variant']
    assert (variant['reference_updates'], variant['joint_updates'], variant['total_updates']) == (150,600,750)
    assert variant['variant'] == 'reference_warmstart_v1'
    assert variant['operand_weight'] == variant['explain_loss_factor'] == 1 and not variant['prefix_mask']
    for name, expected in contract['source_sha256'].items():
        verify_recorded_source(name, expected)
    for split, expected in contract['data_files_sha256'].items():
        assert sha(ROOT/f'e3/data/{split}.jsonl') == expected
    assert sha(ROOT/'assets/manifest.json') == contract['assets_manifest_sha256']
    assert sha(ROOT/'e3/data/manifest.json') == contract['data_sha256']
    assert contract['order_sha256'] == hashlib.sha256(json.dumps(sample_order(contract['config'])).encode()).hexdigest()
    logs = read_rows(directory/'train_log.jsonl')
    sessions = read_rows(directory/'sessions.jsonl')
    assert [r['update'] for r in logs] == list(range(1,751))
    for r in logs:
        u, loss = r['update'], r['losses']
        assert r['epoch'] == (u-1)//250+1 and r['explain_loss_factor'] == 1
        assert math.isfinite(r['grad_norm']) and all(math.isfinite(v) for v in loss.values() if v is not None)
        if u <= 150:
            assert r['stage'] == 'reference_only' and loss['loss'] == loss['ref_ce_loss']
            assert all(loss[k] is None for k in ('ce_loss','distill_loss','explain_loss'))
        else:
            assert r['stage'] == 'joint' and all(v is not None for v in loss.values())
            assert abs(loss['loss']-sum(v for k,v in loss.items() if k != 'loss')) < 1e-4
        step = u-1
        expected_lr = 5e-5*((step+1)/38 if step < 38 else (750-step)/(750-38))
        assert abs(r['lr']-expected_lr) < 1e-12
    assert [r['update'] for r in logs if 'validation' in r] == [250,500,750]
    assert sessions[-1]['completed_updates'] == 750
    result['training'] = dict(updates=len(logs), variant=variant, sessions=sessions,
        session_seconds=sum(s['seconds'] for s in sessions),
        peak_allocated_gib=max(s['peak_allocated_gib'] for s in sessions),
        epoch_validation=[dict(update=r['update'], **r['validation']) for r in logs if 'validation' in r],
        transition_logs=logs[149:151], final_log=logs[-1], source_hashes_match=True,
        gradient_note='Logged norm is BEFORE clipping; source clips to max_grad_norm=1.0.',
        precision_note='Training validation: FP32 master + BF16 autocast. Standalone evaluation: BF16 weights + autocast.')

    pair_rows = []
    for baseline in ('control','p1','p2'):
        pairs = [dict(id=a['id'], baseline=baseline, gold=a['answer'], baseline_prediction=a['prediction'],
                      p3_prediction=b['prediction'], baseline_correct=a['correct'], p3_correct=b['correct'])
                 for a,b in zip(preds[baseline], preds['p3'])]
        pair_rows.extend(pairs)
        gained = [p['id'] for p in pairs if p['p3_correct'] and not p['baseline_correct']]
        lost = [p['id'] for p in pairs if p['baseline_correct'] and not p['p3_correct']]
        result['pairs'][baseline] = dict(both_correct=sum(p['baseline_correct'] and p['p3_correct'] for p in pairs),
            gained_ids=gained, lost_ids=lost, changed_answers=sum(p['baseline_prediction'] != p['p3_prediction'] for p in pairs),
            exact_mcnemar_p=exact_p(len(gained), len(lost)),
            strict_gained=sum(b['strict_match'] is True and a['strict_match'] is not True for a,b in zip(slots[baseline], slots['p3'])),
            strict_lost=sum(a['strict_match'] is True and b['strict_match'] is not True for a,b in zip(slots[baseline], slots['p3'])))
    step_rows = []
    roles = ('control','p1','p2','p3')
    for group in zip(*(slots[k] for k in roles)):
        a = group[0]
        record = {k:a[k] for k in ('id','slot','valid','reference')}
        for role, s in zip(roles, group):
            assert (s['id'],s['slot'],s['reference']) == (a['id'],a['slot'],a['reference'])
            record.update({role:s['text'], role+'_strict':s['strict_match'],
                           role+'_first_operand':first_operands(s['text']) == first_operands(s['reference']) if s['valid'] else None})
        step_rows.append(record)

    gold_by_id = {g['id']:g for g in gold}
    ids = [gold[i]['id'] for i in sorted(random.Random(11).sample(range(200),32))]
    assert ids == [r['id'] for r in samples['p3']['validation']]
    probe_rows, pcfgs, probedata, reference_arithmetic = [], {}, {}, {}
    for role, folder, updates, evrole in [
        ('control','p3_control_reference',750,'control'),
        ('reference150','p3_reference150',150,None),
        ('p3','p3_final_reference',750,'p3')]:
        directory = ROOT/'server_backup_p3'/folder
        inputs.extend(p for p in directory.iterdir() if p.is_file())
        c = read(directory/'config.json')
        rows = read_rows(directory/'pairs.jsonl')
        assert c['ids'] == ids == [r['id'] for r in rows]
        assert c['updates'] == updates and c['config'] == contract['config']
        assert c['seed'] == 11 and c['max_reference_tokens'] == 512 and c['latent_steps'] == 4
        assert c['dtype'] == 'BF16 weights + BF16 autocast'
        assert c['data_sha256'] == manifest['splits']['validation']['sha256']
        for name, expected in c['source_sha256'].items():
            verify_recorded_source(name, expected)
        if evrole:
            assert c['checkpoint_sha256'] == cfgs[evrole]['checkpoint_sha256']
        if role != 'control':
            assert c['source_sha256'] == pcfgs['control']['source_sha256']
        pcfgs[role], probedata[role] = c, rows
        totals = dict(n=32, reference_correct=0, latent_correct=0, reference_only_correct=0,
                      latent_only_correct=0, reference_capped=0, reference_parse_failures=0)
        arithmetic = dict(complete_equations=0, valid_equations=0, invalid_equations=0,
                          unparsed_equations=0, outputs_with_invalid_equations=0,
                          correct_answers_with_invalid_equations=0)
        byid = {p['id']:p for p in preds[evrole]} if evrole else {}
        for r in rows:
            g = gold_by_id[r['id']]
            assert r['gold'] == extract(str(g['answer']))
            raw = r['reference']['raw_output']
            chunks = re.findall(r'<<[^<>]*>>', raw)
            statuses = [equation(chunk)['arithmetic_valid'] for chunk in chunks]
            arithmetic['complete_equations'] += len(chunks)
            arithmetic['valid_equations'] += sum(s is True for s in statuses)
            arithmetic['invalid_equations'] += sum(s is False for s in statuses)
            arithmetic['unparsed_equations'] += sum(s is None for s in statuses)
            arithmetic['outputs_with_invalid_equations'] += any(s is False for s in statuses)
            rp = extract(raw) if re.search(r'The answer is\s*:|####', raw, re.I) else None
            lp = extract(r['latent']['raw_output'])
            rc, lc = rp == r['gold'], lp == r['gold']
            arithmetic['correct_answers_with_invalid_equations'] += rc and any(s is False for s in statuses)
            assert (r['reference_prediction'],r['latent_prediction'],r['reference_correct'],r['latent_correct']) == (rp,lp,rc,lc)
            assert r['reference']['stop_reason'] in ('eos','length')
            assert len(r['reference']['token_ids']) <= r['reference']['budget'] <= 512
            if r['reference']['stop_reason'] == 'length':
                assert len(r['reference']['token_ids']) == r['reference']['budget']
            if evrole:
                assert lp == byid[r['id']]['prediction'] and r['latent']['raw_output'] == byid[r['id']]['raw_output']
            for k,v in dict(reference_correct=rc, latent_correct=lc, reference_only_correct=rc and not lc,
                            latent_only_correct=lc and not rc, reference_capped=r['reference']['stop_reason']=='length',
                            reference_parse_failures=rp is None).items():
                totals[k] += int(v)
            probe_rows.append(dict(role=role, id=r['id'], question=g['question'], gold=r['gold'], gold_cot=g['cot'],
                reference_output=raw, reference_prediction=rp, reference_correct=rc,
                reference_stop=r['reference']['stop_reason'], reference_tokens=len(r['reference']['token_ids']),
                complete_equations=len(chunks), invalid_equations=sum(s is False for s in statuses),
                latent_output=r['latent']['raw_output'], latent_prediction=lp, latent_correct=lc))
        assert totals == read(directory/'summary.json')
        totals['capped_and_parse_failed'] = sum(r['reference']['stop_reason']=='length' and r['reference_prediction'] is None for r in rows)
        totals['reference_correct_ids'] = [r['id'] for r in rows if r['reference_correct']]
        totals['both_correct'] = sum(r['reference_correct'] and r['latent_correct'] for r in rows)
        result['reference_probes'][role] = totals
        reference_arithmetic[role] = arithmetic
    result['reference_arithmetic'] = reference_arithmetic
    result['reference_arithmetic_note'] = 'Post-hoc arithmetic check of complete <<...>> chunks only; no semantic validity claim; truncated partial equations excluded.'
    refg = [b['id'] for a,b in zip(probedata['control'], probedata['p3']) if b['reference_correct'] and not a['reference_correct']]
    refl = [a['id'] for a,b in zip(probedata['control'], probedata['p3']) if a['reference_correct'] and not b['reference_correct']]
    result['reference_pairs'] = dict(gained_ids=refg, lost_ids=refl, exact_mcnemar_p=exact_p(len(refg),len(refl)))
    result['limitations'] = [
        'Single seed, small validation set, exploratory comparisons; previous test-influenced baseline selection remains disclosed.',
        'P3 has 150 reference-only + 600 joint updates; control has 750 joint updates. Joint exposure, LR exposure and dropout RNG differ.',
        'Checkpoint hashes match across exports, but weights and checkpoint-embedded contract were not downloaded/reverified.',
        'Reference150 is diagnostic only; no shared_init or control-step150 probe, so no claim of preheat-induced degradation.',
        'Reference free generation allows 512 tokens; latent answers allow 64. These are not matched inference budgets.',
        'Reference final-answer correctness does not establish correct intermediate reasoning.',
        'P0 is FP32 master + BF16 autocast; compare P0 within that protocol, not to BF16 standalone subsets.',
        'Cross-question swaps and NLL differences do not establish faithful reasoning or a causal training failure mechanism.',
        'Ordered first-operand metric and exact McNemar tests are post-hoc descriptive analyses; no multiple-comparison adjustment.',
        'Historical manifest mismatch remains; current data/source hashes checked against downloaded contract and current frozen records.'
    ]
    args.output.mkdir(parents=True)
    write_json(args.output/'summary.json', result)
    write_csv(args.output/'answer_pairs.csv', pair_rows)
    write_csv(args.output/'step_pairs.csv', step_rows)
    write_csv(args.output/'reference_generations.csv', probe_rows)
    write_json(args.output/'cases.json', dict(
        correct_reference_outputs=[r for r in probe_rows if r['reference_correct']],
        p3_correct_answers=[r for r in read_rows(ROOT/'server_backup_p3/p3_validation/validation.decoded.jsonl') if r['correct']],
        p3_arithmetic_errors=[s for s in slots['p3'] if s['arithmetic_valid'] is False]))
    sources = ('e5/analyze_p3.py','e5/analyze_p2.py','e5/analyze_p1.py','e5/protocol.py',
               'e4/analyze.py','e2/summarize_baselines.py','e1/metrics.py')
    inputs.extend(ROOT/p for p in ('e3/data/manifest.json','e3/data/train.jsonl','e3/data/validation.jsonl','e3/frozen_v1.json'))
    write_json(args.output/'provenance.json', dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        sources={p:sha(ROOT/p) for p in sources}, checkpoints={k:v['checkpoint_sha256'] for k,v in cfgs.items()},
        reference_checkpoints={k:v['checkpoint_sha256'] for k,v in pcfgs.items()}, checks='All alignment, source, log and recomputation assertions passed'))
    print(json.dumps(dict(validation=result['validation'], pairs=result['pairs'],
                         reference_probes=result['reference_probes'], reference_pairs=result['reference_pairs'],
                         training=result['training']), indent=2))
    print('COMPLETE', args.output)


if __name__ == '__main__':
    main()
