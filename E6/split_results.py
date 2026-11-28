"""Split a fresh combined offline analysis into E6/E7 outputs; never overwrite."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

E7_NAMES={'case_coverage.json','case_review.csv','case_review.jsonl','CASE_REVIEW.txt','diagnostic_provenance.json','reference_path.csv','retrieval_comparison.csv','RETRIEVAL_NOTES.txt','categories.csv','categories.jsonl','slots.csv','slots.jsonl','steps.csv','steps.jsonl'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    if a.output_root.exists():raise SystemExit('Use a new output root')
    files=list(a.input.iterdir())
    if not (a.input/'complete.json').is_file():raise SystemExit('Finish analysis and case generation first')
    hashes={}
    for src in files:
        if not src.is_file():continue
        dst=a.output_root/('E7' if src.name in E7_NAMES else 'E6')/'results'/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        hashes[dst.relative_to(a.output_root).as_posix()]=hashlib.sha256(dst.read_bytes()).hexdigest()
    (a.output_root/'MANIFEST_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    print('PASS split',len(hashes),'files; original combined analysis preserved')

if __name__=='__main__':main()
