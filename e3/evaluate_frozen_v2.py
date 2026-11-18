"""Validate the frozen v2 artifacts, then run the existing official project evaluator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"e3_runs/control_v2/last.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"e3_runs/control_v2_test_final")
    args = parser.parse_args()
    frozen = json.loads((ROOT/"e3/frozen_v2.json").read_text(encoding="utf8"))
    assert sha(args.checkpoint) == frozen["checkpoint_sha256"], "Wrong or modified v2 checkpoint"
    for name, expected in frozen["required_file_sha256"].items():
        assert sha(ROOT/name) == expected, f"Frozen file changed: {name}"
    if args.output.exists():
        assert not list(args.output.iterdir()), "Keep previous results; choose a fresh --output directory"
    print("PASS: frozen checkpoint, evaluator, parser and all 600 test questions", flush=True)
    subprocess.run([sys.executable,"-u",str(ROOT/"e3/evaluate.py"),
                    "--checkpoint",str(args.checkpoint),"--split","test","--limit","200",
                    "--output",str(args.output)],check=True,cwd=ROOT)
    write_json(args.output/"frozen_baseline.json",frozen)
    print("FROZEN V2 TEST COMPLETE",args.output,flush=True)


if __name__ == "__main__":
    main()
