"""Run locally or remotely before renting/using GPU."""
import json
from pathlib import Path
from metrics import extract,numeric
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    checks={"The answer is: 18":"18","18.0":"18","The answer is: -9,867,630.0":"-9867630",
            "The answer is: 1/2":"1/2","The answer is: $18":"18",
            "The answer is: 18 dollars.":"18","12 + 6 = 18":None,
            "The answer is: 18 or 19":None,"":None,"1/0":None}
    for text,expected in checks.items():
        assert extract(text)==expected,(text,extract(text),expected)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"assets/gpt2",local_files_only=True,use_fast=False)
    report={}
    for p in sorted((HERE/"data").glob("*.jsonl")):
        rows=[json.loads(x) for x in p.read_text(encoding="utf8").splitlines()]
        assert len(rows)==200
        assert len({x["id"] for x in rows})==200
        assert all(numeric(x["answer"]) is not None for x in rows)
        lengths=[len(tokenizer.encode(x["question"].strip()))+1 for x in rows]
        assert max(lengths)+6+1+64<=1024
        refs=[len(tokenizer.encode(x["reference"])) for x in rows if x["reference"] is not None]
        report[p.stem]=dict(n=200,max_prompt_tokens=max(lengths),reference_count=len(refs),
            mean_reference_tokens=sum(refs)/len(refs) if refs else None)
    report["status"]="PASS: parser, all 600 gold labels, unique IDs, tokenizer, context budget"
    report["gpu_execution"]="Not performed locally"
    (HERE/"preflight_results.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
