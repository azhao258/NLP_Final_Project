"""Copy original experiment scripts to their required paths without overwriting differences."""
import argparse
import hashlib
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    args=p.parse_args();src=Path(__file__).resolve().parent;root=args.project_root.resolve()
    if not (root/'e3/common.py').is_file() or not (root/'e5/protocol.py').is_file():
        raise SystemExit('Destination must be a full NLP_Final_Project checkout')
    files=[]
    for folder in ['P5','GRPO']:
        for f in (src/folder).iterdir():
            if f.is_file():files.append((f,root/folder/f.name))
    files.append((src/'diagnostics/check_reference_retrieval.py',root/'e5/check_reference_retrieval.py'))
    for a,b in files:
        if b.exists() and a.read_bytes()!=b.read_bytes():
            raise SystemExit(f'Existing different file; no files copied: {b}. Use a clean checkout.')
    for a,b in files:
        b.parent.mkdir(parents=True,exist_ok=True)
        if not b.exists():shutil.copy2(a,b)
        assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
    print(f'PASS: {len(files)} code/config files available. Results and weights were not copied.')

if __name__=='__main__':main()
