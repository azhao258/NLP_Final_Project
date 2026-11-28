import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from P5.method import preflight,prepare_pairs
from e3.common import ROOT,read_rows
from transformers import AutoTokenizer
p=preflight(require_weights=True)
t=AutoTokenizer.from_pretrained(ROOT/'assets/gpt2',local_files_only=True,use_fast=False)
pairs=prepare_pairs(read_rows(ROOT/'e3/data/train.jsonl'),t)
print(json.dumps(dict(protocol=p,paired_anchors=sum(bool(r['slots']) for r in pairs),paired_slots=sum(len(r['slots']) for r in pairs)),indent=2))
