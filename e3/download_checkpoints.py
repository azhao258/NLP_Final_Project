"""Download public Release shards, verify hashes, restore original full checkpoints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--names',nargs='+',default=['shared_init','control'])
    p.add_argument('--all',action='store_true')
    args=p.parse_args()
    manifest=json.loads((ROOT/'e3/checkpoints_release.json').read_text())
    names=list(manifest['checkpoints']) if args.all else args.names
    for name in names:
        info=manifest['checkpoints'][name]
        target=(ROOT/info['path']).resolve()
        assert target.is_relative_to(ROOT.resolve())
        if target.exists():
            assert sha(target)==info['sha256'],f'Existing checkpoint differs: {target}'
            print('verified existing',name);continue
        cache=ROOT/'e3_runs/release_cache'/name;cache.mkdir(parents=True,exist_ok=True)
        for part in info['parts']:
            path=cache/part['file']
            if not path.exists():
                url=f"https://github.com/{manifest['repository']}/releases/download/{manifest['release_tag']}/{part['file']}"
                request=urllib.request.Request(url,headers={'User-Agent':'NLP-project-checkpoint-downloader'})
                temp=path.with_suffix(path.suffix+'.download')
                with urllib.request.urlopen(request,timeout=120) as remote,temp.open('wb') as out:
                    shutil.copyfileobj(remote,out,4*1024*1024)
                assert temp.stat().st_size==part['bytes'] and sha(temp)==part['sha256']
                temp.replace(path)
            assert path.stat().st_size==part['bytes'] and sha(path)==part['sha256']
        joined=cache/'checkpoint.gz'
        with joined.open('wb') as out:
            for part in info['parts']:
                with (cache/part['file']).open('rb') as f:shutil.copyfileobj(f,out,4*1024*1024)
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.pt.download')
        with gzip.open(joined,'rb') as source,temp.open('wb') as out:shutil.copyfileobj(source,out,4*1024*1024)
        assert temp.stat().st_size==info['bytes'] and sha(temp)==info['sha256']
        temp.replace(target)
        joined.unlink()
        print('RESTORED',target)
    print('COMPLETE')


if __name__=='__main__':main()
