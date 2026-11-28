"""Restore original manifest bytes only after semantic and GPT-2 asset validation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(data):return hashlib.sha256(data).hexdigest()
def entries(data):
    rows=json.loads(data.decode('utf-8-sig'))
    normalized={}
    for row in rows:
        row=dict(row);row['file']=row['file'].replace('\\','/')
        if row['file'] in normalized:raise ValueError('Duplicate manifest file')
        normalized[row['file']]=row
    return normalized

def main():
    original=Path(__file__).with_name('frozen_assets_manifest.json').read_bytes()
    frozen=json.loads((ROOT/'e3/frozen_v1.json').read_text(encoding='utf8'))
    expected=frozen['training_contract']['assets_manifest_sha256']
    assert digest(original)==expected,'Bundled original manifest hash mismatch'
    path=ROOT/'assets/manifest.json';current=path.read_bytes()
    print('Expected:',expected);print('Current:',digest(current))
    reference=entries(original);actual=entries(current)
    assert actual==reference, 'Manifest entries differ beyond formatting/path separators; STOP. Do not overwrite.'
    checked=0
    for filename,item in reference.items():
        if not filename.startswith('assets/gpt2/'):continue
        asset=(ROOT/filename).resolve();assert asset.is_relative_to(ROOT.resolve())
        assert asset.stat().st_size==item['bytes'],f'Wrong size: {filename}'
        with asset.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        assert sha==item['sha256'],f'Changed GPT-2 asset: {filename}'
        checked+=1;print('VERIFIED',filename)
    assert checked==6
    if current!=original:
        backup=path.with_name('manifest.before_restore.'+digest(current)[:12]+'.json')
        if backup.exists():assert backup.read_bytes()==current
        else:backup.write_bytes(current)
        path.write_bytes(original)
        print('RESTORED original bytes; backup:',backup)
    else:print('Manifest already matches')
    print('PASS: semantic equality and six actual GPT-2 hashes verified')

if __name__=='__main__':main()
