"""Strict numeric answer extraction; no GPU required."""
import csv
import json
import re
from fractions import Fraction
from pathlib import Path
import numpy as np

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
def numeric(text):
    if not re.fullmatch(NUMBER,str(text).strip()):
        return None
    try:
        parts=str(text).replace(",","").replace(" ","").split("/")
        value=Fraction(parts[0])
        if len(parts)==2:
            value/=Fraction(parts[1])
        return str(value)
    except (ValueError,ZeroDivisionError):
        return None

def extract(text):
    markers=list(re.finditer(r"The answer is\s*:|####",text,re.I))
    value=text[markers[-1].end():].strip() if markers else text.strip()
    value=value.strip().strip("$").strip()
    if value.endswith("."):
        value=value[:-1].rstrip()
    match=re.fullmatch(r"("+NUMBER+r")(?:\s+(?:dollars?|USD))?",value,re.I)
    return numeric(match.group(1)) if match else None

def summarize(directory):
    directory=Path(directory)
    rows=[]
    for p in sorted(directory.glob("*.predictions.jsonl")):
        data=[json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x]
        if not data:
            continue
        available=[x for x in data if x["reference_tokens"] is not None]
        latency=[x["latency_seconds"] for x in data]
        rows.append(dict(dataset=data[0]["dataset"],n=len(data),
            em=sum(x["correct"] for x in data)/len(data),
            parse_failures=sum(x["prediction"] is None for x in data),
            latent_steps=6,reference_type=data[0]["reference_type"],
            reference_count=len(available),
            mean_reference_tokens=float(np.mean([x["reference_tokens"] for x in available])) if available else None,
            length_ratio=sum(x["reference_tokens"] for x in available)/(6*len(available)) if available else None,
            mean_answer_tokens=float(np.mean([x["answer_tokens"] for x in data])),
            capped=sum(x["stop_reason"]=="length" for x in data),
            mean_seconds=float(np.mean(latency)),median_seconds=float(np.median(latency)),
            p95_seconds=float(np.percentile(latency,95))))
    if rows:
        with (directory/"summary.csv").open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return rows
