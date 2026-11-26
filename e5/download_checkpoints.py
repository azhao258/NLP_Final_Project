"""Download E5 Release shards, check hashes, and restore original checkpoints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

TAG = 'e5-checkpoints-2026-11-26'
REPO = 'azhao258/NLP_Final_Project'
MODELS = ('prefix_mask', 'p1', 'p2', 'p3', 'p4')
BASE = f'https://github.com/{REPO}/releases/download/{TAG}'
BLOCK = 8 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(BLOCK), b''):
            h.update(chunk)
    return h.hexdigest()


def get(url, path):
    temporary = path.with_name(path.name+'.download')
    request = urllib.request.Request(url, headers={'User-Agent': 'NLP-E5-checkpoint-downloader'})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as target:
        shutil.copyfileobj(source, target, BLOCK)
    temporary.replace(path)


class PartsReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Only bounded streaming reads are supported')
        output = bytearray()
        while len(output) < size:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file = path.open('rb')
            chunk = self.file.read(size-len(output))
            if chunk:
                output.extend(chunk)
            else:
                self.file.close()
                self.file = None
        return bytes(output)

    def close(self):
        if self.file:
            self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Project root; defaults to current directory')
    parser.add_argument('--names', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    root = args.root.resolve()
    cache = root/'e5/release_cache'
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).resolve().with_name('checkpoints_release.json')
    if not manifest_path.exists():
        manifest_path = cache/'checkpoints_release.json'
        get(BASE+'/checkpoints_release.json', manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    assert manifest['repository'] == REPO and manifest['release_tag'] == TAG
    for name in args.names:
        info = manifest['checkpoints'][name]
        assert info['path'] == f'e5/{name}/checkpoint/last.pt'
        target = (root/info['path']).resolve()
        assert target.is_relative_to(root)
        if target.exists():
            assert target.stat().st_size == info['bytes'] and sha(target) == info['sha256'], f'Existing checkpoint differs: {target}'
            print('VERIFIED existing', name, flush=True)
            continue
        paths = []
        for part in info['parts']:
            assert Path(part['file']).name == part['file']
            path = cache/part['file']
            if not path.exists():
                print('DOWNLOAD', part['file'], flush=True)
                get(BASE+'/'+part['file'], path)
            assert path.stat().st_size == part['bytes'] and sha(path) == part['sha256'], f'Bad shard: {path}'
            paths.append(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.download')
        reader = PartsReader(paths)
        try:
            with gzip.GzipFile(fileobj=reader, mode='rb') as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, BLOCK)
        finally:
            reader.close()
        assert temporary.stat().st_size == info['bytes'] and sha(temporary) == info['sha256']
        temporary.replace(target)
        print('RESTORED', target, flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
