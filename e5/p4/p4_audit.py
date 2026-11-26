"""Audit all frozen train/validation slots against upstream get_steps."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from e3.common import ROOT, pack, get_steps, read_rows
from e5.p4.p4_alignment import preflight, step_spans, batch_positions


def audit_rows(rows, tokenizer):
    counts = dict(rows=len(rows), valid_slots=0, consumed_slots=0,
                  readout_only_slots=0, merged_tail_rows=0, no_step_rows=0,
                  maximum_reference_tokens=0)
    for row in rows:
        packed = pack(row, tokenizer)
        ids = packed["ref_input_ids"][0]
        groups = step_spans(ids, stop_ids=(tokenizer.eos_token_id, tokenizer.pad_token_id))
        expected = get_steps([ids], 5, start_ids=(16791, 9959), end_id=4211,
            eot_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id,
            stop_ids=(tokenizer.eos_token_id, tokenizer.pad_token_id))[0]
        actual = [sum((ids[a:b+1] for a,b in group), []) + [tokenizer.eos_token_id]
                  if group else [tokenizer.pad_token_id] for group in groups]
        if actual != expected:
            raise ValueError(f"Upstream target mismatch: {row['id']}")
        batch = {k:torch.tensor(v) for k,v in packed.items()}
        positions = batch_positions(batch, tokenizer)[0].tolist()
        if positions != [g[-1][1] if g else -1 for g in groups]:
            raise ValueError(f"Alignment offset mismatch: {row['id']}")
        counts['valid_slots'] += sum(p >= 0 for p in positions)
        counts['consumed_slots'] += sum(p >= 0 for p in positions[:4])
        counts['readout_only_slots'] += positions[4] >= 0
        counts['merged_tail_rows'] += len(groups[4]) > 1
        counts['no_step_rows'] += not any(groups)
        counts['maximum_reference_tokens'] = max(counts['maximum_reference_tokens'], len(ids))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-weights', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT/'assets/gpt2', local_files_only=True, use_fast=False)
    tok.add_special_tokens({'pad_token': '[PAD]'})
    result = dict(protocol=preflight(require_weights=args.require_weights),
                  splits={s:audit_rows(read_rows(ROOT/f'e3/data/{s}.jsonl'),tok)
                          for s in ('train','validation')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
