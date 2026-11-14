"""Provenance utilities for isolated external-dataset studies."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'agent-interact-external-audit'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def freeze_split(path, questions, cfg, *, forced_train=(), strata=None, inherited=None):
    """Assign question groups using metadata only, before outcome diagnostics."""
    path = Path(path)
    questions = sorted(set(questions))
    forced_train = sorted(set(forced_train))
    inherited = {q: r for q, r in (inherited or {}).items() if q in questions}
    if path.exists():
        record = json.loads(path.read_text())
        assert record['questions'] == questions and record['config'] == cfg
        assert record['forced_train'] == forced_train
        assert record.get('inherited', {}) == inherited
        return record
    rng = np.random.default_rng(cfg['seed'])
    groups = {'train': [], 'validation': [], 'assessment': []}
    strata = strata or {q: 'all' for q in questions}
    for level in sorted(set(strata.values())):
        ids = [q for q in questions if strata[q] == level]
        rng.shuffle(ids)
        ntrain = int(len(ids)*cfg['train_fraction'])
        nval = int(len(ids)*cfg['validation_fraction'])
        fixed = {q: r for q, r in inherited.items() if q in ids}
        for q in forced_train:
            if q in ids:
                assert q not in fixed or fixed[q] == 'train', 'Inspected question was already reserved'
                fixed[q] = 'train'
        rest = [q for q in ids if q not in fixed]
        counts = {'train': ntrain, 'validation': nval, 'assessment': len(ids)-ntrain-nval}
        offset = 0
        for role, count in counts.items():
            retained = sorted(q for q, r in fixed.items() if r == role)
            needed = count-len(retained)
            assert needed >= 0, 'Inherited groups exceed split quota'
            groups[role] += retained+rest[offset:offset+needed]
            offset += needed
        assert offset == len(rest)
    assert set.union(*(set(v) for v in groups.values())) == set(questions)
    assert sum(map(len, groups.values())) == len(questions)
    record = {'created_utc': now(), 'questions': questions, 'config': cfg,
              'forced_train': forced_train, 'groups': groups,
              'inherited': inherited,
              'policy': 'Whole-question split. Assessment outcomes are excluded from exploratory diagnostics and training selection.'}
    write_json(path, record)
    return record


def acquire(url, target, *, expected_size=None, expected_sha256=None,
            max_bytes=600_000_000, compress=False):
    """Download public data only; no remote code execution or credential use."""
    target = Path(target)
    manifest = target.with_name(target.name + '.source.json')
    if target.exists():
        if not manifest.exists():
            raise RuntimeError(f'Existing file has no provenance: {target}')
        record = json.loads(manifest.read_text())
        assert record['url'] == url and record['artifact_sha256'] == sha256(target)
        if expected_sha256:
            assert record['upstream_sha256'] == expected_sha256
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.partial')
    req = urllib.request.Request(url, headers={'User-Agent': 'agent-interact-external-audit'})
    h = hashlib.sha256(); size = 0; mark = 0
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open('wb') as disk:
        sink = gzip.GzipFile(filename='', mode='wb', fileobj=disk, mtime=0) if compress else disk
        try:
            while block := r.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise RuntimeError(f'Download exceeded {max_bytes} byte limit')
                h.update(block); sink.write(block)
                if size - mark >= 25_000_000:
                    print(json.dumps({'download': target.name, 'bytes': size}), flush=True)
                    mark = size
        finally:
            if compress:
                sink.close()
    if expected_size is not None:
        assert size == expected_size, (size, expected_size)
    if expected_sha256:
        assert h.hexdigest() == expected_sha256
    os.replace(tmp, target)
    record = {'url': url, 'retrieved_utc': now(), 'upstream_bytes': size,
              'upstream_sha256': h.hexdigest(), 'artifact_sha256': sha256(target),
              'compression': 'gzip' if compress else None}
    write_json(manifest, record)
    return record
