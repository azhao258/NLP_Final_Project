"""Audit every train/validation equation with real GPT-2 tokenization; no test read."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer
from e5.p2.p2_operand import OperandWeight, operand_spans, EQUATION
from e5.p2.p2_protocol import preflight
from e5.protocol import ROOT, read_rows


def main():
    protocol = preflight()
    tok = AutoTokenizer.from_pretrained(ROOT/'assets/gpt2', local_files_only=True)
    policy = OperandWeight(tok)
    result = {'protocol': protocol, 'splits': {}}
    for split in ('train','validation'):
        rows = read_rows(ROOT/f'e3/data/{split}.jsonl')
        equations = weighted = merged = numeric_spans = 0
        for row in rows:
            eqs = EQUATION.findall(row['cot'])
            texts = ['<<'+eq+'>>' for eq in eqs]
            slots = texts if len(texts)<=5 else texts[:4]+[''.join(texts[4:])]
            equations += len(eqs)
            merged += len(texts)>5
            for text in slots:
                ids = tok.encode(text)+[tok.eos_token_id]
                w = policy.token_weights(tuple(ids))
                assert w[-1] == 1
                numeric_spans += len(operand_spans(text))
                weighted += sum(x==2 for x in w)
        result['splits'][split] = dict(rows=len(rows),equations=equations,numeric_operand_spans=numeric_spans,uniform_symbolic_equations=equations-numeric_spans,weighted_tokens=weighted,merged_tail_rows=merged)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
