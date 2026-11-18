import hashlib
import json
import random
from fractions import Fraction
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
    hard = [json.loads(line) for line in (HERE/"audit/course_hard.jsonl").read_text(encoding="utf8").splitlines()]
    official_hard = {row["input"]: row for row in hard}
    legacy_hard = pd.read_parquet(raw/"gsm_hard.parquet").to_dict("records")
    hard = [official_hard[row["instruction"]] for row in legacy_hard]
    assert all(Fraction(str(old["response"])) == Fraction(str(new["target"]))
               for old, new in zip(legacy_hard, hard)), "GSM-Hard gold mismatch"
    records["gsm_hard"] = [
        dict(id=f"gsm-hard-train-{i}",source_index=i,question=x["input"],
             answer=legacy_hard[i]["response"],reference=None,reference_type="missing",problem_type=None)
        for i,x in enumerate(hard)]
    svamp = pd.read_parquet(HERE/"audit/svamp_test.parquet").to_dict("records")
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
                        "gsm_hard":"https://huggingface.co/datasets/reasoning-machines/gsm-hard",
                        "svamp":"https://huggingface.co/datasets/ChilleD/SVAMP"})
    for name, rows in records.items():
        chosen = sorted(random.Random(4051).sample(range(len(rows)),200))
        selected = []
        for i in chosen:
            row = dict(rows[i],dataset=name)
            row["question_sha256"]=hashlib.sha256(row["question"].encode()).hexdigest()
            selected.append(row)
        p = output/f"{name}.jsonl"
        # Freeze CRLF explicitly to reproduce the original Windows artifacts on Linux too.
        p.write_bytes("".join(json.dumps(x,ensure_ascii=False)+"\r\n" for x in selected).encode("utf8"))
        manifest["datasets"][name]=dict(population=len(rows),count=len(selected),sha256=digest(p),
            ids=[x["id"] for x in selected],duplicates=len(selected)-len({x["question_sha256"] for x in selected}))
    manifest["course_revisions"]={"gsm_hard":"960448f73503112d4226baeb8eb41d3fb5ae2506","svamp":"5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"}
    manifest["splits"]={"gsm8k":"test","gsm_hard":"train (evaluation only; legacy IDs retained via exact question matching)","svamp":"test"}
    manifest["course_raw_sha256"]={name:digest(HERE/"audit"/name) for name in ["course_hard.jsonl","svamp_test.parquet","svamp_train.parquet"]}
    manifest["raw_sha256"]={p.name:digest(p) for p in raw.iterdir() if p.is_file()}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8",newline="\n")
    print({k:v["population"] for k,v in manifest["datasets"].items()})

if __name__=="__main__":
    prepare()

