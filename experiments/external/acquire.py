"""Acquire pinned public files for external studies. Run as a module."""
from pathlib import Path
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

from .common import ROOT, acquire, fetch_json, sha256, write_json


def mallm(config_path=None):
    cfg = json.loads((ROOT/(config_path or 'experiments/external/mallm_pilot_v1/config.json')).read_text())
    data = ROOT/cfg['data_root']
    source = ROOT/'runs/dataset_scout_20261109'
    (data/'metadata').mkdir(parents=True, exist_ok=True)
    for old, new in [('debate_meta.json', 'huggingface_repository.json'),
                     ('debate_card.md', 'DATASET_CARD.md')]:
        target = data/'metadata'/new
        if not target.exists():
            shutil.copyfile(source/old, target)
        assert sha256(target) == sha256(source/old)
    url = f"https://huggingface.co/datasets/{cfg['dataset']}/resolve/{cfg['revision']}/{cfg['source_file']}"
    record = acquire(url, data/'raw'/(cfg['configuration']+'.json.gz'),
                     expected_size=cfg['source_bytes'], expected_sha256=cfg['source_sha256'],
                     max_bytes=int(cfg['source_bytes']*1.1), compress=True)
    write_json(data/'metadata'/(cfg['configuration']+'.source.json'), {**record, 'dataset': cfg['dataset'],
               'revision': cfg['revision'], 'configuration': cfg['configuration'],
               'license': 'Apache-2.0 (dataset card)', 'new_llm_calls': 0})
    print(json.dumps({'acquired': cfg['dataset'], 'upstream_bytes': record['upstream_bytes']}), flush=True)


def coevolve():
    cfg = json.loads((ROOT/'experiments/external/coevolve_pilot_v1/config.json').read_text())
    data = ROOT/cfg['data_root']
    graph_index = json.loads((data/'metadata/experts_graphs.json').read_text())
    entries = []
    for graph in cfg['graph_folders']:
        node = next(x for x in graph_index['tree'] if x['path'] == graph)
        url = f"https://api.github.com/repos/{cfg['repository']}/git/trees/{node['sha']}?recursive=1"
        dest = data/'metadata'/(graph+'_tree.json')
        if dest.exists():
            tree = json.loads(dest.read_text())
        else:
            tree = fetch_json(url); write_json(dest, tree)
        assert not tree.get('truncated')
        for item in tree['tree']:
            if item['type'] != 'blob' or Path(item['path']).name not in ['agents_data.json', 'network_edges.json', 'agent_manifest.json']:
                continue
            parts = Path(item['path']).parts
            assert parts[0] == 'wR_L'
            qid = parts[1]
            upstream = f"data/outputs/runs/zeroshot/{cfg['scenario']}/{graph}/{item['path']}"
            local = data/'raw/trajectories'/cfg['scenario']/graph/qid/Path(item['path']).name
            entries.append({'upstream': upstream, 'local': str(local.relative_to(ROOT)),
                            'bytes': item['size'], 'git_blob': item['sha'], 'graph': graph, 'question': qid})
    assert len({x['local'] for x in entries}) == len(entries), 'Duplicate runs require explicit selection'
    assert len(entries) == 120, len(entries)
    write_json(data/'metadata/pilot_files.json', {'revision': cfg['revision'], 'files': entries})
    def one(entry):
        url = f"https://raw.githubusercontent.com/{cfg['repository']}/{cfg['revision']}/{entry['upstream']}"
        record = acquire(url, ROOT/entry['local'], expected_size=entry['bytes'], max_bytes=200_000)
        raw = (ROOT/entry['local']).read_bytes()
        import hashlib
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == entry['git_blob']
        return record
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(one, entries))
    print(json.dumps({'acquired': 'coevolve pilot', 'files': len(records),
                      'upstream_bytes': sum(x['upstream_bytes'] for x in records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mallm', 'coevolve'])
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.dataset == 'mallm':
        mallm(args.config)
    else:
        coevolve()
