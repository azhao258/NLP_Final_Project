import json
import torch
import transformers
from common import ROOT, sha

assert transformers.__version__ == "4.52.4", "Install the pinned requirements"
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
expected = json.loads((ROOT/"e3/expected_initialization.json").read_text())
assert sha(ROOT/"e3_runs/shared_init.pt") == expected["sha256"], "Wrong shared initialization"
for entry in json.loads((ROOT/"assets/manifest.json").read_text()):
    name = entry["file"].replace("\\", "/")
    if name.startswith("assets/gpt2/"):
        assert sha(ROOT/name) == entry["sha256"], name
manifest = json.loads((ROOT/"e3/data/manifest.json").read_text())
for name, info in manifest["splits"].items():
    assert sha(ROOT/f"e3/data/{name}.jsonl") == info["sha256"]
print("PASS: pinned dependencies, GPU, original initialization, GPT-2 assets and frozen splits")
print(torch.cuda.get_device_name(0))
