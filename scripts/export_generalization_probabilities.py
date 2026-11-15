"""Export probabilities from saved development responses; no model execution.

Standard library only, for a short CPU allocation on HPC3. Never opens assessment
episodes, invokes the source model, or modifies collection artifacts.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())

def export(base, output):
    base, output = Path(base), Path(output)
    index_path = base/'runs/question_breadth_320_v1/episode_index.json'
    index = read(index_path)
    assert Counter(r['split'] for r in index) == {'train': 320, 'validation': 80}
    units = read(base/'runs/hpc3_recovery_v1/collection_units.json')
    lookup = {q: u for u in units for q in u['qids']}
    rows, chunks, episodes = [], {}, {}
    quality = Counter(); max_spread = 0.
    for number, item in enumerate(index):
        assert item['split'] in ('train', 'validation')
        path = base/item['path']; assert sha(path) == item['sha256']
        episode = read(path); meta = episode['meta']; episodes[item['path']] = item['sha256']
        assert meta['split'] == item['split'] and meta['qid'] == item['qid']
        roots = [base/'runs/question_breadth_v1'] if item['origin'] == 'retained' else []
        if item['qid'] in lookup:
            unit = lookup[item['qid']]
            roots += [base/'runs/hpc3_recovery_v1/units'/unit['id'], base/unit['legacy_root']]
        roots += [base/'runs/question_breadth_320_v1']
        for round_number in range(9):
            for begin in range(0, 32, 4):
                group = '{}|round{}|opinions{}'.format(meta['name'], round_number, begin)
                name = hashlib.sha256(group.encode()).hexdigest()+'.json'
                candidates = [r/'chunks'/name for r in roots if (r/'chunks'/name).is_file()]
                assert candidates, (meta['name'], round_number, begin)
                path = candidates[0]; chunk = read(path)
                chunks[str(path.relative_to(base))] = sha(path)
                assert chunk['group_id'] == group
                assert chunk['protocol_sha256'] == meta['protocol_sha256'] == item['source_protocol_sha256']
                prompts = {r['prompt_sha256']: r for r in chunk['prompts']}
                for prompt_hash, prompt in prompts.items():
                    assert hashlib.sha256(prompt['prompt'].encode()).hexdigest() == prompt_hash
                attempts = [chunk]
                for retry in (1, 2):
                    retry_name = hashlib.sha256((group+'|retry'+str(retry)).encode()).hexdigest()+'.json'
                    found = [r/'chunks'/retry_name for r in roots if (r/'chunks'/retry_name).is_file()]
                    if found:
                        extra = read(found[0]); attempts.append(extra)
                        chunks[str(found[0].relative_to(base))] = sha(found[0])
                        quality['retry_responses'] += len(extra['responses'])
                accepted = {}
                original = {(r['agent'], r['vote_position']): r for r in chunk['responses']}
                for attempt in attempts:
                    for response in attempt['responses']:
                        assert response['qid'] == item['qid'] and response['round'] == round_number
                        key = response['agent'], response['vote_position']
                        if key not in accepted or accepted[key]['parsed_sign'] not in (-1, 1): accepted[key] = response
                for agent in range(begin, begin+4):
                    responses = [accepted[agent, vote] for vote in range(5)]
                    initial_responses = [original[agent, vote] for vote in range(5)]
                    # The frozen collector preserves a zero after exhausted retries.
                    # Such cases must remain visible, not silently omitted from the audit.
                    votes = [r['parsed_sign'] if r['parsed_sign'] in (-1, 1) else 0 for r in responses]
                    assert votes == episode['spins_raw_history'][round_number][agent]
                    prompt_hashes = {r['prompt_sha256'] for r in responses}; assert len(prompt_hashes) == 1
                    pa = [float(r['first_literal_A_probability']) for r in initial_responses]
                    pb = [float(r['first_literal_B_probability']) for r in initial_responses]
                    assert all(0 <= a <= 1.000001 and 0 <= b <= 1.000001 and a+b <= 1.000002 for a,b in zip(pa,pb))
                    max_spread = max(max_spread, max(pa)-min(pa), max(pb)-min(pb))
                    for r in responses:
                        quality['responses'] += 1
                        quality['failed_after_retries'] += r['parsed_sign'] not in (-1, 1)
                        quality['votes_using_retry'] += r['parse_attempt'] > 0
                        quality['strict_responses'] += bool(r['strict_format'])
                        quality['not_single_bare_token_responses'] += r['raw_response'].strip() not in ('A', 'B')
                        quality['non_stop_responses'] += r['finish_reason'] != 'stop'
                        if r['raw_response'].strip() in ('A', 'B'):
                            assert r['parsed_sign'] == (1 if r['raw_response'].strip() == 'A' else -1)
                    quality['initial_not_single_bare_token_responses'] += sum(r['raw_response'].strip() not in ('A', 'B') for r in initial_responses)
                    rows.append({'qid': item['qid'], 'split': item['split'], 'origin': item['origin'],
                        'round': round_number, 'agent': agent, 'raw_votes': votes,
                        'first_literal_A_probabilities': pa, 'first_literal_B_probabilities': pb,
                        'parse_attempts': [r['parse_attempt'] for r in responses],
                        'initial_bare': [r['raw_response'].strip() in ('A', 'B') for r in initial_responses],
                        'initial_stopped': [r['finish_reason'] == 'stop' for r in initial_responses],
                        'prompt_sha256': next(iter(prompt_hashes))})
        if (number+1) % 20 == 0: print(json.dumps({'episodes_exported': number+1, 'rows': len(rows)}), flush=True)
    assert len(rows) == 400*9*32 and quality['responses'] == 400*9*32*5
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(), 'rows': rows,
        'quality': dict(quality), 'maximum_within_prompt_probability_spread': max_spread,
        'index_sha256': sha(index_path), 'episode_artifacts': episodes, 'chunk_artifacts': chunks,
        'export_code_sha256': sha(__file__), 'source_model_calls': 0, 'assessment_loaded': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with gzip.open(temporary, 'wt') as stream: json.dump(payload, stream, allow_nan=False)
    temporary.replace(output)
    print(json.dumps({'complete': True, 'episodes': 400, 'rows': len(rows), 'quality': dict(quality),
        'maximum_within_prompt_probability_spread': max_spread, 'output': str(output), 'sha256': sha(output)}), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--base', default='.'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); export(args.base, args.output)
