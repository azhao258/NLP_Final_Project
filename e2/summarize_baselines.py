"""Rebuild Part 1 CSV/JSON statistics from frozen predictions; no report generation or model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1.metrics import extract  # Reuse the actual evaluation scorer (requires numpy).

DATASETS = ('gsm8k', 'gsm_hard', 'svamp')
MODELS = ('CODI_released', 'SIM-CoT_released', 'control_v1')


def source_dir(model, dataset):
    if model == 'control_v1':
        return 'e3_runs/control_test'
    prefix = 'codi' if model == 'CODI_released' else 'sim'
    return f'e1_results/{prefix}_svamp_course_test' if dataset == 'svamp' else f'e1_results/{prefix}_full'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root=ROOT):
    inputs, tables, paired = {}, [], []
    def record(path):
        inputs[path.relative_to(root).as_posix()] = sha(path)
    manifest_path = root/'e1/data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    record(manifest_path)
    record(root/'e1/metrics.py')
    frozen_path = root/'e3/frozen_v1.json'
    record(frozen_path)
    frozen = json.loads(frozen_path.read_text(encoding='utf8'))
    warnings = []
    for dataset in DATASETS:
        fixed_path = root/f'e1/data/{dataset}.jsonl'
        record(fixed_path)
        assert sha(fixed_path) == manifest['datasets'][dataset]['sha256']
        fixed = read_rows(fixed_path)
        ids = [r['id'] for r in fixed]
        assert len(ids) == len(set(ids)) == 200
        assert ids == manifest['datasets'][dataset]['ids']
        aligned = {}
        for model in MODELS:
            directory = source_dir(model, dataset)
            prediction_path = root/directory/f'{dataset}.predictions.jsonl'
            config_path = root/directory/'config.json'
            summary_path = root/directory/'summary.csv'
            for p in (prediction_path, config_path, summary_path):
                record(p)
            config = json.loads(config_path.read_text(encoding='utf8'))
            rows = read_rows(prediction_path)
            assert len(rows) == 200 and len({r['id'] for r in rows}) == 200
            by_id = {r['id']:r for r in rows}
            assert set(by_id) == set(ids), (model,dataset,'IDs differ')
            rows = [by_id[i] for i in ids]
            assert config['gpu'] == 'NVIDIA RTX A6000'
            assert config['batch_size'] == 1 and config['warmups'] == 5
            if model == 'control_v1':
                steps = config['config']['latent_steps']
                assert steps == 4 and config['greedy'] and config['config']['answer_max_tokens'] == 64
                assert config['precision'] == 'BF16 weights and autocast'
                assert config['checkpoint_sha256'] == frozen['checkpoint_sha256']
            else:
                steps = config['latent_steps']
                assert steps == 6 and config['precision'] == 'bf16' and config['max_new_tokens'] == 64
                assert config['seed'] == 11
                expected_id = 'zen-E/CODI-gpt2' if model == 'CODI_released' else 'internlm/SIM_COT-GPT2-CODI'
                assert config['model_id'] == expected_id
            if config['data_manifest_sha256'] != sha(manifest_path):
                warnings.append(dict(model=model, dataset=dataset,
                    message='Historical whole-manifest hash differs; selected dataset records checked against current frozen rows.',
                    recorded=config['data_manifest_sha256'], current=sha(manifest_path)))
            for row, gold in zip(rows, fixed):
                assert row['question'] == gold['question'] and row['reference'] == gold['reference']
                assert row['reference_type'] == gold['reference_type']
                assert extract(str(row['answer'])) == extract(str(gold['answer']))
                assert row['gold_normalized'] == extract(str(gold['answer']))
                assert row['prediction'] == extract(row['raw_output'])
                assert row['correct'] == (row['prediction'] is not None and row['prediction'] == row['gold_normalized'])
                assert row['latent_steps'] == steps
                assert math.isfinite(row['latency_seconds']) and row['latency_seconds'] >= 0
                if row.get('question_sha256'):
                    assert row['question_sha256'] == hashlib.sha256(row['question'].encode()).hexdigest()
                assert (row['reference_tokens'] is None) == (gold['reference'] is None)
            aligned[model] = rows
            available = [r for r in rows if r['reference_tokens'] is not None]
            latency = [r['latency_seconds'] for r in rows]
            result = dict(model=model, dataset=dataset, n=len(rows), correct=sum(r['correct'] for r in rows),
                em=sum(r['correct'] for r in rows)/len(rows), parse_failures=sum(r['prediction'] is None for r in rows),
                latent_steps=steps, reference_type=rows[0]['reference_type'], reference_count=len(available),
                mean_reference_tokens=statistics.mean(r['reference_tokens'] for r in available) if available else None,
                length_ratio=sum(r['reference_tokens'] for r in available)/(steps*len(available)) if available else None,
                mean_answer_tokens=statistics.mean(r['answer_tokens'] for r in rows),
                capped=sum(r['stop_reason']=='length' for r in rows), mean_seconds=statistics.mean(latency),
                median_seconds=statistics.median(latency), p95_seconds=percentile(latency,.95), source_dir=directory)
            with summary_path.open(encoding='utf-8-sig', newline='') as stream:
                archived = next(r for r in csv.DictReader(stream) if r['dataset']==dataset)
            for key, value in archived.items():
                if key not in result or key == 'dataset':
                    continue
                actual = result[key]
                if actual is None:
                    assert value == ''
                elif isinstance(actual, (float,int)):
                    assert math.isclose(float(value),actual,rel_tol=1e-10,abs_tol=1e-12), (directory,key)
                else:
                    assert value == actual
            tables.append(result)
        for index in range(200):
            assert len({aligned[m][index]['reference_tokens'] for m in MODELS}) == 1
        a,b = aligned[MODELS[0]],aligned[MODELS[1]]
        paired.append(dict(dataset=dataset,
            both_correct=sum(x['correct'] and y['correct'] for x,y in zip(a,b)),
            codi_wrong_sim_correct=sum(not x['correct'] and y['correct'] for x,y in zip(a,b)),
            codi_correct_sim_wrong=sum(x['correct'] and not y['correct'] for x,y in zip(a,b)),
            both_wrong=sum(not x['correct'] and not y['correct'] for x,y in zip(a,b))))
    return tables, paired, dict(inputs=inputs, warnings=warnings,
        scorer='e1.metrics.extract; numeric-normalized exact match, no tolerance',
        percentile='linear interpolation, same definition as numpy.percentile default',
        reference_length='Reaggregate recorded GPT-2 reference_tokens; not re-tokenized during this CPU summary.',
        checkpoint_verification='Recorded checkpoint identities only; weights not downloaded in this task.',
        script_sha256=sha(Path(__file__)))


def export_results(output):
    rows, paired, provenance = collect()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'baseline_comparison.csv', [r for r in rows if r['model']!='control_v1'])
    write_csv(output/'part1_baseline_table.csv', rows)
    write_csv(output/'paired_comparison.csv', paired)
    (output/'baseline_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Generated 9 baseline rows, 6 released-model rows and 3 paired comparisons in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'e2')
    export_results(parser.parse_args().output)
