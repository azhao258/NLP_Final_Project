"""Verify historical source hashes without rewriting completed-run contracts.

The byte-exact pre-organization snapshot remains the source evidence for old
runs. Current source hashes are separately recorded in ORGANIZATION_MANIFEST.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify_recorded_source(name, expected):
    path = ROOT / name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    manifest = json.loads((ROOT / 'e5/ORGANIZATION_MANIFEST.json').read_text(encoding='utf8'))
    archive = ROOT / manifest['source_snapshot']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['source_snapshot_sha256'], 'Historical source snapshot changed'
    with zipfile.ZipFile(archive) as source:
        assert name in source.namelist(), f'Historical source unavailable: {name}'
        assert hashlib.sha256(source.read(name)).hexdigest() == expected, f'Historical source mismatch: {name}'
