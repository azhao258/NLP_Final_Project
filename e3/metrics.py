"""Same E1 reporting definitions, using the recorded self-trained latent budget."""
import csv
import json
from pathlib import Path
import numpy as np


def summarize(directory):
    result = []
    for path in sorted(Path(directory).glob("*.predictions.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        refs = [r for r in rows if r["reference_tokens"] is not None]
        steps = {r["latent_steps"] for r in rows}
        assert len(steps) == 1
        latencies = [r["latency_seconds"] for r in rows]
        result.append(dict(dataset=rows[0]["dataset"], n=len(rows),
            em=sum(r["correct"] for r in rows)/len(rows),
            parse_failures=sum(r["prediction"] is None for r in rows), latent_steps=steps.pop(),
            reference_type=rows[0]["reference_type"], reference_count=len(refs),
            mean_reference_tokens=float(np.mean([r["reference_tokens"] for r in refs])) if refs else None,
            length_ratio=sum(r["reference_tokens"] for r in refs)/sum(r["latent_steps"] for r in refs) if refs else None,
            mean_answer_tokens=float(np.mean([r["answer_tokens"] for r in rows])),
            capped=sum(r["stop_reason"]=="length" for r in rows),
            mean_seconds=float(np.mean(latencies)), median_seconds=float(np.median(latencies)),
            p95_seconds=float(np.percentile(latencies,95))))
    with (Path(directory)/"summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result
