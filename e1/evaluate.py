"""Offline batch evaluation of the released CODI/SIM-CoT checkpoints."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
import torch
from transformers import set_seed

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from smoke import released, IDS
from metrics import extract,numeric,summarize

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def generate(model,encoded):
    """Same six cache updates and answer prefix as the verified upstream path."""
    torch.cuda.synchronize()
    start=time.perf_counter()
    out=model.codi(input_ids=encoded,output_hidden_states=True,use_cache=True)
    for _ in range(6):
        latent=model.prj(out.hidden_states[-1][:,-1:])
        out=model.codi(inputs_embeds=latent,past_key_values=out.past_key_values,
                       output_hidden_states=True,use_cache=True)
    token=torch.tensor([[model.eot_id]],device="cuda")
    ids=[]
    stopped=False
    for _ in range(64):
        out=model.codi(input_ids=token,past_key_values=out.past_key_values,use_cache=True)
        token=out.logits[:,-1,:model.codi.config.vocab_size-1].argmax(-1).reshape(1,1)
        value=token.item()
        ids.append(value)
        if value==model.tokenizer.eos_token_id:
            stopped=True
            break
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    visible=ids[:-1] if stopped else ids
    return dict(raw_output=model.tokenizer.decode(visible),token_ids=ids,
                answer_tokens=len(visible),generated_tokens=len(ids),
                stop_reason="eos" if stopped else "length",latency_seconds=elapsed)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["codi","sim"],default="codi")
    parser.add_argument("--limit",type=int,default=200)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not 1<=args.limit<=200:
        parser.error("--limit must be 1..200")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Verify GPU before starting this job.")
    set_seed(11)
    config=dict(model_id=IDS[args.model],limit=args.limit,latent_steps=6,
                seed=11,batch_size=1,precision="bf16",max_new_tokens=64,warmups=5,
                gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),
                timing="GPU-synchronized prompt + latent + answer generation; excludes tokenization and text decoding",
                data_manifest_sha256=sha(HERE/"data/manifest.json"),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                    [Path(__file__),HERE/"metrics.py",ROOT/"smoke.py",ROOT/"SIM-CoT/CODI/src/model.py"]})
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/"config.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text())!=config:
            raise RuntimeError("Existing run/config mismatch. Use a new output directory or --resume with the same configuration.")
    elif any(dest.glob("*.predictions.jsonl")):
        raise RuntimeError("Predictions exist without config.")
    else:
        path.write_text(json.dumps(config,indent=2),encoding="utf8")
    manifest=json.loads((HERE/"data/manifest.json").read_text())
    datasets={}
    for name,info in manifest["datasets"].items():
        p=HERE/"data"/f"{name}.jsonl"
        assert sha(p)==info["sha256"],name
        datasets[name]=[json.loads(x) for x in p.read_text(encoding="utf8").splitlines()][:args.limit]
    model,aliases=released(args.model)
    (dest/"model.json").write_text(json.dumps(dict(tied_aliases=aliases,
        total_parameters=sum(p.numel() for p in model.parameters())),indent=2))
    (dest/"environment.txt").write_text(subprocess.check_output(
        [sys.executable,"-m","pip","freeze"],text=True),encoding="utf8")
    session=str(uuid.uuid4())
    start_run=time.perf_counter()
    for name,rows in datasets.items():
        target=dest/f"{name}.predictions.jsonl"
        done={}
        if target.exists():
            # A truncated final line is preserved for diagnosis, then removed before resuming.
            raw=target.read_bytes()
            lines=raw.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):
                (dest/f"{name}.truncated-tail").write_bytes(lines.pop())
                target.write_bytes(b"".join(lines))
            for line in lines:
                row=json.loads(line)
                if row["id"] in done:
                    raise RuntimeError("Duplicate prediction ID")
                done[row["id"]]=row
            allowed={r["id"]:r["question_sha256"] for r in rows}
            assert all(k in allowed and v["question_sha256"]==allowed[k] for k,v in done.items())
        if len(done)==len(rows):
            continue
        def encode(question):
            ids=model.tokenizer.encode(question.strip())+[model.bot_id]
            assert len(ids)+6+1+64<=model.codi.config.n_positions,"Prompt exceeds context budget"
            return torch.tensor([ids],device="cuda")
        for _ in range(5):
            generate(model,encode(rows[0]["question"]))
        with target.open("a",encoding="utf8") as stream:
            for i,row in enumerate(rows):
                if row["id"] in done:
                    continue
                result=generate(model,encode(row["question"]))
                prediction=extract(result["raw_output"])
                gold=numeric(row["answer"])
                assert gold is not None,(row["id"],row["answer"])
                result.update(row,prediction=prediction,gold_normalized=gold,
                    correct=prediction==gold,latent_steps=6,session=session,
                    reference_tokens=len(model.tokenizer.encode(row["reference"])) if row["reference"] is not None else None)
                stream.write(json.dumps(result,ensure_ascii=False)+"\n")
                stream.flush()
                print(f"{name} {i+1}/{len(rows)} correct={result['correct']} seconds={result['latency_seconds']:.3f}",flush=True)
        print(json.dumps(summarize(dest),indent=2),flush=True)
    elapsed=time.perf_counter()-start_run
    with (dest/"sessions.jsonl").open("a") as f:
        f.write(json.dumps(dict(session=session,evaluation_seconds=elapsed,
            note="Includes warmups and file I/O after model load; partial sessions require external wall-clock tracking"))+"\n")
    print("COMPLETE",dest)

if __name__=="__main__":
    main()
