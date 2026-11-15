"""Export a fixed, already-completed training prefix; standard-library CPU job.

Source collection is read-only. Atomic per-question outputs permit a restart
without source-model calls or selection based on which questions finish first.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def export_one(base, item, protocol_hash):
    source = base/item['path']; assert sha(source) == item['sha256']
    episode = read(source); meta = episode['meta']; qid = item['qid']
    assert meta['split'] == 'train' and meta['qid'] == qid
    assert meta['graph_id'] == item['graph_id'] and meta['protocol_sha256'] == protocol_hash
    unit = base/'runs/hpc3_scale_3200_v1/units'/item['unit']
    assert (unit/'complete.json').is_file()
    roots = [unit, base/'runs/question_breadth_3200_v1']
    rows, chunks, executions = [], {}, {}
    quality = Counter(); spread = 0.

    def chunk(group, required=True):
        name = hashlib.sha256(group.encode()).hexdigest()+'.json'
        paths = [r/'chunks'/name for r in roots if (r/'chunks'/name).is_file()]
        if not paths:
            assert not required, group
            return None
        path = paths[0]; value = read(path)
        assert value['group_id'] == group and value['protocol_sha256'] == protocol_hash
        chunks[str(path.relative_to(base))] = sha(path)
        for prompt in value['prompts']:
            assert hashlib.sha256(prompt['prompt'].encode()).hexdigest() == prompt['prompt_sha256']
        if 'hpc3_execution' in value:
            execution = value['hpc3_execution']
            executions[hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()] = execution
        return value

    for t in range(9):
        for begin in range(0, 32, 4):
            group = '{}|round{}|opinions{}'.format(meta['name'], t, begin)
            first = chunk(group); attempts = [first]
            for retry in (1, 2):
                extra = chunk(group+'|retry'+str(retry), False)
                if extra is not None:
                    attempts.append(extra); quality['retry_responses'] += len(extra['responses'])
            original = {(r['agent'], r['vote_position']): r for r in first['responses']}
            assert len(original) == 20
            accepted = {}
            for attempt in attempts:
                for response in attempt['responses']:
                    assert response['qid'] == qid and response['round'] == t
                    key = response['agent'], response['vote_position']
                    if key not in accepted or accepted[key]['parsed_sign'] not in (-1, 1): accepted[key] = response
            for agent in range(begin, begin+4):
                current = [accepted[agent, vote] for vote in range(5)]
                initial = [original[agent, vote] for vote in range(5)]
                votes = [r['parsed_sign'] if r['parsed_sign'] in (-1, 1) else 0 for r in current]
                assert votes == episode['spins_raw_history'][t][agent]
                prompt_hashes = {r['prompt_sha256'] for r in current+initial}; assert len(prompt_hashes) == 1
                pa = [float(r['first_literal_A_probability']) for r in initial]
                pb = [float(r['first_literal_B_probability']) for r in initial]
                assert all(0 <= a <= 1.000001 and 0 <= b <= 1.000001 and 0 < a+b <= 1.000002 for a,b in zip(pa,pb))
                spread = max(spread, max(pa)-min(pa), max(pb)-min(pb))
                for response in current:
                    quality['responses'] += 1
                    quality['failed_after_retries'] += response['parsed_sign'] not in (-1, 1)
                    quality['votes_using_retry'] += response['parse_attempt'] > 0
                    quality['strict_responses'] += bool(response['strict_format'])
                    quality['not_single_bare_token_responses'] += response['raw_response'].strip() not in ('A', 'B')
                    quality['non_stop_responses'] += response['finish_reason'] != 'stop'
                    if response['raw_response'].strip() in ('A', 'B'):
                        assert response['parsed_sign'] == (1 if response['raw_response'].strip() == 'A' else -1)
                quality['initial_not_single_bare_token_responses'] += sum(r['raw_response'].strip() not in ('A', 'B') for r in initial)
                rows.append(dict(qid=qid, split='train', origin='expansion_3200', round=t, agent=agent,
                    raw_votes=votes, first_literal_A_probabilities=pa, first_literal_B_probabilities=pb,
                    parse_attempts=[r['parse_attempt'] for r in current],
                    initial_bare=[r['raw_response'].strip() in ('A', 'B') for r in initial],
                    initial_stopped=[r['finish_reason'] == 'stop' for r in initial], prompt_sha256=next(iter(prompt_hashes))))
    assert len(rows) == 288 and quality['responses'] == 1440
    return dict(episode=episode, episode_sha256=item['sha256'], source_path=item['path'], rows=rows,
        quality=dict(quality), maximum_within_prompt_probability_spread=spread,
        chunks=chunks, executions=executions, complete_marker_sha256=sha(unit/'complete.json'))


def main(base, manifest_path, output):
    base, manifest_path, output = Path(base), Path(manifest_path), Path(output)
    manifest = read(manifest_path); selected = manifest['selected_units']
    assert len(selected) == 192 and all(x['complete'] for x in selected)
    unit_path = base/'runs/hpc3_scale_3200_v1/collection_units.json'
    assert sha(unit_path) == manifest['collection_units_sha256']
    units = [u for u in read(unit_path) if u['split'] == 'train'][:192]
    assert [(u['id'], u['qids'][0], u['graph_id']) for u in units] == [(x['unit'], x['qid'], x['graph_id']) for x in selected]
    selection_path = base/'runs/question_breadth_3200_v1/selection.json'
    assert sha(selection_path) == manifest['selection_sha256']
    selection = read(selection_path)
    questions = {q['qid']: q for q in selection['new_questions'] if q['qid'] in {x['qid'] for x in selected}}
    assert len(questions) == 192 and all(q['split'] == 'train' for q in questions.values())
    output.mkdir(parents=True, exist_ok=True); files = {}; quality = Counter(); spread = 0.
    identity = dict(manifest_sha256=sha(manifest_path), export_code_sha256=sha(__file__), source_model_calls=0,
        new_validation_loaded=False, assessment_loaded=False, source_artifacts_modified=False)
    for i, item in enumerate(selected):
        path = output/'questions'/(item['qid']+'.json.gz')
        if path.exists():
            with gzip.open(path, 'rt') as stream: value = json.load(stream)
            assert value['identity'] == identity and value['episode_sha256'] == item['sha256']
        else:
            value = export_one(base, item, manifest['source_protocol_sha256'])
            metadata = {k: questions[item['qid']][k] for k in ('qid', 'split', 'origin', 'type', 'level', 'graph_id', 'correct_answer', 'family_key', 'source_row', 'screen') if k in questions[item['qid']]}
            value.update(identity=identity, question_metadata=metadata)
            path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.tmp')
            with gzip.open(temp, 'wt') as stream: json.dump(value, stream, allow_nan=False)
            temp.replace(path)
        files[str(path.relative_to(output))] = sha(path)
        quality.update(value['quality']); spread = max(spread, value['maximum_within_prompt_probability_spread'])
        write(output/'progress.json', dict(updated_utc=now(), completed=i+1, planned=192))
        if (i+1) % 16 == 0: print(json.dumps({'completed': i+1, 'planned': 192}), flush=True)
    write(output/'complete.json', dict(**identity, completed_utc=now(), questions=192, rows=192*288,
        files=files, quality=dict(quality), maximum_within_prompt_probability_spread=spread))
    archive = output/'export.tar'; temporary = output/'export.tar.tmp'
    with tarfile.open(temporary, 'w') as tar:
        for name in ['complete.json', *files]: tar.add(output/name, arcname=name)
    temporary.replace(archive)
    write(output/'archive.json', dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size))
    print(json.dumps(read(output/'archive.json')), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--base', default='.'); parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.base, args.manifest, args.output)
