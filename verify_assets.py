import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
published = json.loads((root/"published_hashes.json").read_text(encoding="utf-8-sig"))
for item in json.loads((root/"assets/manifest.json").read_text()):
    path = root/item["file"].replace(chr(92),"/")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream,"sha256").hexdigest()
    if digest != item["sha256"] or digest != published.get(path.name,digest):
        raise RuntimeError(f"Hash mismatch: {path}")
    print("verified",path.name)
