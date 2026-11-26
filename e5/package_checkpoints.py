"""Package the five frozen E5 checkpoints as gzip shards; no model deserialization."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'e5/release_assets'
TAG = 'e5-checkpoints-2026-11-26'
MODELS = ('prefix_mask', 'p1', 'p2', 'p3', 'p4')
BLOCK = 8 * 1024**2
PART = 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(BLOCK), b''):
            h.update(chunk)
    return h.hexdigest()


class Shards:
    def __init__(self, name):
        self.name, self.file, self.size = name, None, 0
        self.parts = []

    def write(self, data):
        total = len(data)
        while data:
            if self.file is None:
                path = OUT / f'{self.name}.pt.gz.part{len(self.parts)+1:03d}'
                self.file = path.open('xb')
                self.parts.append(path)
                self.size = 0
            take = min(len(data), PART - self.size)
            self.file.write(data[:take])
            data = data[take:]
            self.size += take
            if self.size == PART:
                self.file.close()
                self.file = None
        return total

    def flush(self):
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def package(name):
    source = ROOT / f'e5/{name}/checkpoint/last.pt'
    expected = json.loads((ROOT / f'e5/{name}/test/config.json').read_text())['checkpoint_sha256']
    receipt = OUT / f'{name}.verified.json'
    if receipt.exists():
        info = json.loads(receipt.read_text())
        assert digest(source) == expected == info['sha256']
        for part in info['parts']:
            path = OUT / part['file']
            assert path.stat().st_size == part['bytes'] and digest(path) == part['sha256']
        print(f'REUSE {name}', flush=True)
        return name, info
    if list(OUT.glob(f'{name}.pt.gz.part*')):
        raise RuntimeError(f'Incomplete prior package for {name}; inspect before retry')
    print(f'PACK {name}', flush=True)
    writer = Shards(name)
    original = hashlib.sha256()
    try:
        with gzip.GzipFile(filename='', fileobj=writer, mode='wb', compresslevel=6, mtime=0) as archive:
            with source.open('rb') as f:
                for chunk in iter(lambda: f.read(BLOCK), b''):
                    original.update(chunk)
                    archive.write(chunk)
    finally:
        writer.close()
    assert original.hexdigest() == expected, f'Checkpoint identity mismatch: {name}'
    # Verify restoration byte-for-byte by streaming all shards through a gzip decoder.
    import zlib
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    restored, restored_bytes, parts = hashlib.sha256(), 0, []
    for path in writer.parts:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(BLOCK), b''):
                h.update(chunk)
                raw = decoder.decompress(chunk)
                restored.update(raw)
                restored_bytes += len(raw)
        parts.append(dict(file=path.name, bytes=path.stat().st_size, sha256=h.hexdigest()))
    tail = decoder.flush()
    restored.update(tail)
    restored_bytes += len(tail)
    assert decoder.eof and not decoder.unused_data
    assert restored_bytes == source.stat().st_size and restored.hexdigest() == expected
    info = dict(path=source.relative_to(ROOT).as_posix(), bytes=restored_bytes,
                sha256=expected, updates=750, parts=parts)
    receipt.write_text(json.dumps(info, indent=2)+'\n', encoding='utf8')
    print(f'VERIFIED {name}: {len(parts)} shards, {sum(p["bytes"] for p in parts)} bytes', flush=True)
    return name, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        packed = dict(pool.map(package, MODELS))
    manifest = dict(release_tag=TAG, repository='azhao258/NLP_Final_Project',
                    checkpoints={name: packed[name] for name in MODELS})
    for path in (ROOT/'e5/checkpoints_release.json', OUT/'checkpoints_release.json'):
        path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print('COMPLETE: five checkpoints compressed and restoration verified', flush=True)


if __name__ == '__main__':
    main()
