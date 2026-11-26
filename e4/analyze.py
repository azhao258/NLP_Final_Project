"""CPU-only Part 2 scoring from frozen exports; writes CSV/JSON, not REPORT.md."""
import argparse
import ast
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e2.summarize_baselines import collect
OPS = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
SLOT_LABELS = ['missing_output', 'unparseable_output', 'operator_mismatch', 'operand_mismatch',
               'intermediate_result_mismatch', 'arithmetic_inconsistency',
               'reference_arithmetic_inconsistency', 'reference_step_not_reproduced',
               'fewer_equations_than_reference']


def normalize(text):
    return re.sub(r'\s+', '', text.replace('<|endoftext|>', ''))


def expression(text):
    """Evaluate only bounded numeric +,-,*,/ ASTs, never eval arbitrary input."""
    if len(text) > 500:
        raise ValueError('expression too long')
    tree = ast.parse(text, mode='eval')
    operands, operators = [], []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(ast.get_source_segment(text, node))
            operands.append(str(value))
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            # Signed numeric literals are operands, not subtraction operations.
            if not isinstance(node.operand, ast.Constant):
                raise ValueError('nonliteral unary operator')
            value = visit(node.operand)
            value = -value if isinstance(node.op, ast.USub) else value
            operands[-1] = str(value)
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left), visit(node.right)
            operators.append(OPS[type(node.op)])
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError('unsupported expression')

    value = visit(tree.body)
    return dict(value=str(value), operands=sorted(operands), operators=sorted(operators))


def equation(text):
    cleaned = normalize(text)
    chunks = re.findall(r'<<[^<>]*>>', cleaned)
    if len(chunks) > 1 and ''.join(chunks) == cleaned:
        parts = [equation(chunk) for chunk in chunks]
        if not all(p['parsed'] for p in parts):
            return dict(parsed=False, arithmetic_valid=None)
        return dict(parsed=True, equation_count=len(parts), parts=parts,
            arithmetic_valid=all(p['arithmetic_valid'] for p in parts),
            lhs=dict(value=[p['lhs']['value'] for p in parts],
                operands=sorted(v for p in parts for v in p['lhs']['operands']),
                operators=sorted(v for p in parts for v in p['lhs']['operators'])),
            rhs=dict(value=[p['rhs']['value'] for p in parts]))
    match = re.fullmatch(r'<<(.*?)>>', cleaned)
    if match:
        cleaned = match.group(1)
    try:
        left, right = cleaned.split('=')
        lhs, rhs = expression(left), expression(right)
        return dict(parsed=True, equation_count=1, lhs=lhs, rhs=rhs,
                    arithmetic_valid=lhs['value'] == rhs['value'])
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return dict(parsed=False, arithmetic_valid=None)


def compare(generated, reference):
    g, r = equation(generated), equation(reference)
    both = g['parsed'] and r['parsed']
    result = dict(strict_match=normalize(generated) == normalize(reference),
                  generated_equation=g, reference_equation=r,
                  operator_match=None, operand_match=None, result_match=None,
                  conservative_equivalent=None, missing_equation_count=None)
    if both:
        result.update(operator_match=g['lhs']['operators'] == r['lhs']['operators'],
                      operand_match=g['lhs']['operands'] == r['lhs']['operands'],
                      result_match=g['rhs']['value'] == r['rhs']['value'],
                      missing_equation_count=max(0, r['equation_count']-g['equation_count']))
        # This is a restricted compatibility check, not general semantic equivalence.
        result['conservative_equivalent'] = all((g['arithmetic_valid'], r['arithmetic_valid'],
            result['operator_match'], result['operand_match'], result['result_match'],
            g['lhs']['value'] == r['lhs']['value'], g['equation_count'] == r['equation_count']))
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def analyze(root, output):
    baseline_rows, _, baseline_provenance = collect(root)
    source = root / 'e3_runs/control_diagnostics'
    paths = [source / 'validation.decoded.jsonl', source / 'validation.predictions.jsonl',
             source / 'config.json', root / 'e3/data/validation.jsonl',
             root / 'e3/data/manifest.json', root / 'e3/active_baseline.json', root / 'e3/frozen_v1.json']
    decoded, predictions, validation = (read_rows(p) for p in (paths[0], paths[1], paths[3]))
    manifest, active, frozen, config = (json.loads(p.read_text(encoding='utf8'))
                                     for p in (paths[4], paths[5], paths[6], paths[2]))
    assert active['active'] == 'v1'
    assert config['checkpoint_sha256'] == frozen['checkpoint_sha256']
    assert config['config']['latent_steps'] == 4 and config['precision'] == 'BF16 weights and autocast'
    assert sha(paths[3]) == manifest['splits']['validation']['sha256'], 'Frozen validation bytes changed'
    # The archived v1 config names an older manifest hash. Preserve this discrepancy;
    # verify the actual validation bytes and all exported rows independently above/below.
    manifest_hash_matches_export = sha(paths[4]) == config['data_manifest_sha256']
    ids = [r['id'] for r in decoded]
    assert len(ids) == len(set(ids)) == 200
    assert ids == manifest['splits']['validation']['ids']
    assert ids == [r['id'] for r in predictions] == [r['id'] for r in validation]
    frequency = [Counter(normalize(r['decoded_steps'][i]['text']) for r in decoded) for i in range(5)]
    steps, questions = [], []
    for row, pred, val in zip(decoded, predictions, validation):
        assert row['question'] == pred['question'] == val['question']
        assert row['reference'] == pred['cot'] == val['cot']
        assert Fraction(str(row['gold'])) == Fraction(str(val['answer']))
        assert row['prediction'] == pred['prediction'] and row['correct'] == pred['correct']
        assert row['correct'] == (row['prediction'] is not None and Fraction(row['prediction']) == Fraction(str(row['gold'])))
        assert len(row['decoded_steps']) == len(row['supervision_slots']) == 5
        refs = re.findall(r'<<.*?>>', row['reference'])
        expected_refs = refs if len(refs) <= 5 else refs[:4] + [''.join(refs[4:])]
        assert [normalize(s) for s in row['supervision_slots'] if s != '[PAD]'] == [normalize(s) for s in expected_refs]
        local = []
        for i, (step, ref) in enumerate(zip(row['decoded_steps'], row['supervision_slots'])):
            assert step['slot'] == i and step['consumed_by_reasoner'] == (i < 4)
            valid = ref != '[PAD]'
            text = step['text']
            comparison = compare(text, ref) if valid else {}
            labels = []
            if valid:
                if not normalize(text):
                    labels.append('missing_output')
                elif not comparison['generated_equation']['parsed']:
                    labels.append('unparseable_output')
                for key, label in [('operator_match', 'operator_mismatch'), ('operand_match', 'operand_mismatch'),
                                   ('result_match', 'intermediate_result_mismatch')]:
                    if comparison[key] is False:
                        labels.append(label)
                if comparison['generated_equation']['arithmetic_valid'] is False:
                    labels.append('arithmetic_inconsistency')
                if comparison['reference_equation']['arithmetic_valid'] is False:
                    labels.append('reference_arithmetic_inconsistency')
                if comparison['conservative_equivalent'] is False:
                    labels.append('reference_step_not_reproduced')
                if comparison['missing_equation_count']:
                    labels.append('fewer_equations_than_reference')
            item = dict(id=row['id'], slot=i, consumed_by_reasoner=i < 4, valid_reference=valid,
                        reference=ref, generated=text, capped=step['capped'], answer_correct=row['correct'],
                        slot_output_frequency=frequency[i][normalize(text)],
                        frequent_template=frequency[i][normalize(text)] >= 100,
                        labels=labels, **comparison)
            steps.append(item)
            local.append(item)
        valid_steps = [s for s in local if s['valid_reference']]
        labels = sorted({label for s in valid_steps for label in s['labels']})
        if any(s['frequent_template'] for s in valid_steps):
            labels.append('frequent_template_on_valid_slot')
        all_strict = all(s['strict_match'] for s in valid_steps)
        all_equiv = all(s['conservative_equivalent'] is True for s in valid_steps)
        if row['correct'] and not all_equiv:
            labels.append('correct_answer_explanation_mismatch')
        questions.append(dict(id=row['id'], question=row['question'], gold=row['gold'],
            prediction=row['prediction'], answer_correct=row['correct'], reference=row['reference'],
            generated=[s['text'] for s in row['decoded_steps']], valid_slots=len(valid_steps),
            strict_matches=sum(s['strict_match'] for s in valid_steps),
            conservative_matches=sum(s['conservative_equivalent'] is True for s in valid_steps),
            all_strict=all_strict, all_conservative=all_equiv,
            reference_equations=len(refs), merged_reference_slot=len(refs)>5, labels=labels))
    valid = [s for s in steps if s['valid_reference']]
    slot_rows = []
    for i in range(5):
        subset = [s for s in valid if s['slot'] == i]
        slot_rows.append(dict(slot=i, consumed_by_reasoner=i < 4, all_outputs=200,
            valid_reference=len(subset), strict_matches=sum(s['strict_match'] for s in subset),
            conservative_matches=sum(s['conservative_equivalent'] is True for s in subset),
            unique_outputs=len(frequency[i]), most_common=frequency[i].most_common(1)[0][0],
            most_common_count=frequency[i].most_common(1)[0][1]))
    slot_counts = Counter(l for s in valid for l in s['labels'])
    question_counts = Counter(l for q in questions for l in q['labels'])
    summary = dict(baseline='v1', n=200, correct=sum(q['answer_correct'] for q in questions),
        manifest_hash_matches_export=manifest_hash_matches_export,
        valid_reference_slots=len(valid), pad_slots=len(steps)-len(valid),
        strict_matches=sum(s['strict_match'] for s in valid),
        conservative_matches=sum(s['conservative_equivalent'] is True for s in valid),
        all_strict_questions=sum(q['all_strict'] for q in questions),
        all_conservative_questions=sum(q['all_conservative'] for q in questions),
        comparable_slots=sum(s['generated_equation']['parsed'] and s['reference_equation']['parsed'] for s in valid),
        generated_arithmetic_valid_all_slots=sum(equation(s['generated'])['arithmetic_valid'] is True for s in steps),
        generated_capped=sum(s['capped'] for s in steps),
        answer_parse_failures=sum(p['prediction'] is None for p in predictions),
        answer_capped=sum(p['stop_reason'] == 'length' for p in predictions),
        merged_reference_questions=sum(q['merged_reference_slot'] for q in questions),
        reference_equations=sum(q['reference_equations'] for q in questions),
        missing_equations_in_merged_slots=sum(s['missing_equation_count'] or 0 for s in valid),
        slot_label_counts={k:slot_counts[k] for k in sorted(SLOT_LABELS)},
        question_label_counts={k:question_counts[k] for k in sorted(set(SLOT_LABELS) | set(question_counts))},
        slots=slot_rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'compression_comparison.csv', baseline_rows)
    for name, rows in [('steps', steps), ('questions', questions)]:
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf8')
        # Fixed columns for PAD as well as valid rows.
        if name == 'steps':
            keys = list(dict.fromkeys(k for r in rows for k in r))
            rows = [{k:r.get(k) for k in keys} for r in rows]
        write_csv(output / f'{name}.csv', rows)
    write_csv(output / 'slots.csv', slot_rows)
    labels = sorted(set(summary['slot_label_counts']) | set(summary['question_label_counts']))
    write_csv(output / 'categories.csv', [dict(label=l, valid_slot_count=summary['slot_label_counts'].get(l),
        question_count=summary['question_label_counts'].get(l, 0)) for l in labels])
    write_json(output / 'summary.json', summary)
    write_json(output / 'provenance.json', dict(source_commit='8040d3a',
        checkpoint_sha256=config['checkpoint_sha256'], inputs={str(p.relative_to(root)):sha(p) for p in paths},
        script_sha256=sha(Path(__file__)), rules_version=2,
        compression_sources=baseline_provenance,
        frequent_template_threshold='same normalized text at same slot in >=100/200 questions',
        manifest_hash_matches_export=manifest_hash_matches_export,
        export_manifest_sha256=config['data_manifest_sha256'],
        current_manifest_sha256=sha(paths[4]),
        validation_bytes_match_current_manifest=True,
        warning=None if manifest_hash_matches_export else 'Archived v1 manifest hash differs from current manifest; validation bytes and row alignment verified, historical manifest identity not established.',
        no_training=True, no_checkpoint_download=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'e4/results')
    args = parser.parse_args()
    analyze(ROOT, args.output)
