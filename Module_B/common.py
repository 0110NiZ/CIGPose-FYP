"""File hashes and atomic report writes."""
import hashlib
import json
import os
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()



def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



def source_hashes(*names):
    folder = Path(__file__).resolve().parent
    return {name: sha(folder/name) for name in names}
