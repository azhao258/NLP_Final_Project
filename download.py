from pathlib import Path
import subprocess, hashlib, json, os
root = Path(__file__).resolve().parent
assets = [
 ("openai-community/gpt2","607a30d783dfa663caf39e06633721c8d4cfcd7e","gpt2","config.json merges.txt vocab.json tokenizer.json tokenizer_config.json model.safetensors"),
 ("zen-E/CODI-gpt2","fd641b3d3edc59e4f534b55588e906588c9e36bb","codi","pytorch_model.bin"),
 ("internlm/SIM_COT-GPT2-CODI","8685b42870510ba33f7caae30f7a24a2020d2e50","sim","model.safetensors.index.json model-00001-of-00001.safetensors"),
 ("datasets/zen-E/GSM8k-Aug","c4972fba71a612e2089d8fb10f78caa01c740083","data","gsm8k_aug_train.json gsm8k_test.json")]
manifest=[]
for repo, rev, folder, names in assets:
 for name in names.split():
    p=root/"assets"/folder/name
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        tmp=str(p)+".part"
        if os.name == "nt": subprocess.run(["pwsh","-NoProfile","-Command","Invoke-WebRequest -Uri $env:ASSET_URL -OutFile $env:ASSET_TMP"],env={**os.environ,"ASSET_URL":f"https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true","ASSET_TMP":tmp},check=True)
        else: subprocess.run(["curl","-fLsS","--retry","3","-o",tmp,f"https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true"],check=True)
        os.replace(tmp,p)
    with p.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    manifest.append(dict(repo=repo,revision=rev,file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha))
    print(name,p.stat().st_size,flush=True)
(root/"assets/manifest.json").write_text(json.dumps(manifest,indent=2))

