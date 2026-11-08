import hashlib
import json
import random
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    raw = HERE/"raw"
    gsm = [json.loads(line) for line in (raw/"gsm8k_test.jsonl").read_text(encoding="utf8").splitlines()]
    records = {}
    records["gsm8k"] = [
        dict(id=f"gsm8k-test-{i}", source_index=i, question=x["question"],
             answer=x["answer"].split("####")[-1].strip(),
             reference=x["answer"].split("####")[0].strip(), reference_type="natural_language",
             problem_type=None) for i,x in enumerate(gsm)]
    hard = pd.read_parquet(raw/"gsm_hard.parquet").to_dict("records")
    records["gsm_hard"] = [
        dict(id=f"gsm-hard-train-{i}",source_index=i,question=x["instruction"],
             answer=x["response"],reference=None,reference_type="missing",problem_type=None)
        for i,x in enumerate(hard)]
    svamp = json.loads((raw/"svamp.json").read_text(encoding="utf8"))
    records["svamp"] = [
        dict(id=x["ID"], source_index=i,question=(x["Body"].strip()+" "+x["Question"].strip()),
             answer=str(x["Answer"]),reference=x["Equation"],reference_type="equation",
             problem_type=x["Type"]) for i,x in enumerate(svamp)]
    output = HERE/"data"
    output.mkdir(exist_ok=True)
    manifest = dict(seed=4051, sample_size=200, datasets={},
                    note="GSM-Hard is a GSM8K derivative. Datasets are reported separately; no pooled independence assumption.",
                    sources={
                        "gsm8k":"https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl",
                        "gsm_hard":"https://huggingface.co/datasets/juyoung-trl/gsm-hard",
                        "svamp":"https://github.com/arkilpatel/SVAMP/blob/master/SVAMP.json"})
    for name, rows in records.items():
        chosen = sorted(random.Random(4051).sample(range(len(rows)),200))
        selected = []
        for i in chosen:
            row = dict(rows[i],dataset=name)
            row["question_sha256"]=hashlib.sha256(row["question"].encode()).hexdigest()
            selected.append(row)
        p = output/f"{name}.jsonl"
        p.write_text("".join(json.dumps(x,ensure_ascii=False)+chr(10) for x in selected),encoding="utf8")
        manifest["datasets"][name]=dict(population=len(rows),count=len(selected),sha256=digest(p),
            ids=[x["id"] for x in selected],duplicates=len(selected)-len({x["question_sha256"] for x in selected}))
    manifest["raw_sha256"]={p.name:digest(p) for p in raw.iterdir() if p.is_file()}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print({k:v["population"] for k,v in manifest["datasets"].items()})

if __name__=="__main__":
    prepare()

