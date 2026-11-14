"""Build causal next-answer examples; audit development partitions only."""
from __future__ import annotations
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import argparse

import pandas as pd

from ..common import ROOT, freeze_split, now, sha256, write_json


def answer(value):
    if not isinstance(value, str):
        return None
    # Author's extracted solution field, not agreement flags or target prose.
    match = re.match(r'^\s*(?:\*\*)?([AB])\s*[).:\-]\s*(?:\*\*)?(Yes|No)\b', value, re.I)
    if not match:
        return None
    label, word = match.groups()
    if (label.upper(), word.lower()) not in [('A', 'yes'), ('B', 'no')]:
        return None
    return int(label.upper() == 'A')


def question_id(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()[:16]


def build_record(record, partition):
    text = '\n'.join(record['input']); qid = question_id(text)
    agents = [x['agentId'] for x in record['personas']]
    assert len(agents) == 3 and len(set(agents)) == 3
    agent_index = {x: i for i, x in enumerate(agents)}
    messages = record['globalMemory']
    by_id = {m['message_id']: m for m in messages}
    assert len(by_id) == len(messages)
    assert [m['message_id'] for m in messages] == sorted(by_id)
    rows = []; texts = []; rejected = collections.Counter()
    for m in messages:
        mid = m['message_id']; aid = m['agent_id']
        visible = m.get('memory_ids', [])
        assert len(visible) == len(set(visible))
        assert all(i in by_id and i < mid for i in visible), 'Future/missing message in inbox'
        texts.append({'question_id': qid, 'message_id': mid, 'agent_id': aid,
                      'agent_index': agent_index[aid], 'turn': m['turn'],
                      'answer': answer(m.get('solution')), 'text': m['message']})
        history = sorted((by_id[i] for i in visible), key=lambda x: x['message_id'])
        own_history = [h for h in history if h['agent_id'] == aid]
        if not own_history:
            rejected['no_visible_previous_own_answer'] += 1
            continue
        target = answer(m.get('solution')); own = answer(own_history[-1].get('solution'))
        draft = answer(m.get('additional_args', {}).get('current_draft'))
        if target is None or own is None or draft is None:
            rejected['unparsed_target_own_or_draft'] += 1
            continue
        # Sender states are reconstructed exclusively from delivered messages.
        latest = {}
        for h in history:
            latest[h['agent_id']] = h
        if set(latest) != set(agents) or any(answer(h.get('solution')) is None for h in latest.values()):
            rejected['incomplete_visible_agent_states'] += 1
            continue
        states = [answer(latest[a].get('solution')) for a in agents]
        speaker_ids = [latest[a]['message_id'] for a in agents]
        row = {'question_id': qid, 'partition': partition, 'message_id': mid,
               'agent_index': agent_index[aid], 'turn': m['turn'], 'own': own,
               'draft': draft, 'target': target, 'visible_ids': visible,
               'latest_message_ids': speaker_ids, 'states': states,
               'visible_message_count': len(visible), 'own_changed': own != target,
               'draft_changed': draft != target, 'visible_disagreement': len(set(states)) > 1}
        rows.append(row)
    return rows, texts, rejected


def main(config_path=None):
    cfg = json.loads((ROOT/config_path if config_path else Path(__file__).with_name('config.json')).read_text())
    data, run, report = [ROOT/cfg[k] for k in ['data_root', 'run_root', 'report_root']]
    raw = data/'raw'/(cfg['configuration']+'.json.gz')
    with gzip.open(raw, 'rt') as f:
        records = json.load(f)
    qids = [question_id('\n'.join(r['input'])) for r in records]
    assert len(qids) == len(set(qids)), 'Repeated questions need explicit grouping'
    forced = [q for q, r in zip(qids, records) if 'manta rays' in '\n'.join(r['input']).lower()]
    reference = None; inherited = {}
    if cfg.get('split_source'):
        reference = json.loads((ROOT/cfg['split_source']).read_text())
        inherited = {q: role for role, ids in reference['groups'].items() for q in ids if q in qids}
    split = freeze_split(run/'split.json', qids, cfg['split'], forced_train=forced, inherited=inherited)
    mapping = {q: role for role, ids in split['groups'].items() for q in ids}
    # Feasibility gate fixed before computing development outcome diagnostics.
    gate_path = run/'feasibility_policy.json'
    if not gate_path.exists():
        write_json(gate_path, {'created_utc': now(), 'min_train_draft_changes': 20,
                   'min_train_questions_with_draft_change': 5, 'min_validation_draft_changes': 5,
                   'purpose': 'Avoid an argument-model comparison with almost no departures from the answer explicitly supplied in the prompt; not a statistical power guarantee.'})
    gate = json.loads(gate_path.read_text())
    rows = []; messages = []; excluded = collections.Counter()
    for qid, record in zip(qids, records):
        examples, txt, rejected = build_record(record, mapping[qid])
        rows.extend(examples); messages.extend(txt)
        if mapping[qid] != 'assessment':
            excluded.update(rejected)
    derived = data/'derived'; derived.mkdir(parents=True, exist_ok=True)
    write_json(derived/'questions.json', {'questions': [{'id': q, 'text': '\n'.join(r['input']),
               'example_id': r.get('exampleId')} for q, r in zip(qids, records)], 'source_sha256': sha256(raw)})
    frame = pd.DataFrame(rows)
    frame.to_parquet(derived/'transitions.parquet', index=False)
    pd.DataFrame(messages).to_parquet(derived/'messages.parquet', index=False)
    stats = {}
    for role in ['train', 'validation']:
        f = frame[frame.partition == role]
        stats[role] = {'questions': int(f.question_id.nunique()), 'transitions': len(f),
                       'own_changes': int(f.own_changed.sum()), 'draft_changes': int(f.draft_changed.sum()),
                       'questions_with_own_change': int(f.loc[f.own_changed, 'question_id'].nunique()),
                       'questions_with_draft_change': int(f.loc[f.draft_changed, 'question_id'].nunique()),
                       'disagreement_transitions': int(f.visible_disagreement.sum()),
                       'persistence_error': float(f.own_changed.mean()),
                       'copy_draft_error': float(f.draft_changed.mean())}
    feasible = (stats['train']['draft_changes'] >= gate['min_train_draft_changes'] and
                stats['train']['questions_with_draft_change'] >= gate['min_train_questions_with_draft_change'] and
                stats['validation']['draft_changes'] >= gate['min_validation_draft_changes'])
    audit = {'created_utc': now(), 'configuration': cfg['configuration'], 'raw_records': len(records),
             'distinct_questions': len(qids), 'split_counts': {k: len(v) for k, v in split['groups'].items()},
             'development': stats, 'development_exclusions': dict(excluded),
             'argument_comparison_feasible': feasible, 'assessment_outcome_diagnostics': False,
             'inherited_question_partitions': len(inherited),
             'source_sha256': sha256(raw), 'split_sha256': sha256(run/'split.json'),
             'derived_sha256': {p.name: sha256(p) for p in derived.glob('*.parquet')},
             'target': 'Recorded solution on the next utterance, not a fresh independent opinion probe or majority-of-five target.',
             'input': 'Only prior messages explicitly listed in memory_ids; current_draft is a genuine pre-response prompt field. No agreement flags or new response text.'}
    write_json(report/'audit.json', audit); write_json(run/'prepared.json', audit)
    lines = ['# MALLM pilot: data audit', '', f"Configuration: `{cfg['configuration']}`.", '',
             f"100 distinct questions, split into 60 fitting, 20 validation and 20 reserved assessment questions.",
             'Assessment outcomes were not summarized or scored. Every visible-message ID is checked to precede its target.', '',
             '| Partition | Updates | Own-answer changes | Changes from supplied draft | Questions with draft change |',
             '| --- | ---: | ---: | ---: | ---: |']
    for role, s in stats.items():
        lines.append(f"| {role} | {s['transitions']} | {s['own_changes']} | {s['draft_changes']} | {s['questions_with_draft_change']} |")
    lines += ['', f"Argument-comparison feasibility gate: **{'passed' if feasible else 'not passed'}**.",
              'The threshold was recorded before computing these outcome counts. It is a practical screen, not proof of statistical power.', '',
              'The next-answer target is the author-extracted solution field. Agreement with an explanation is not used as a yes/no answer.',
              'All agents see a shared current draft. Copying that draft is therefore a necessary baseline alongside keeping the agent’s own last answer.', '',
              'Source: https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE', '']
    report.mkdir(parents=True, exist_ok=True); (report/'audit.md').write_text('\n'.join(lines))
    print(json.dumps({'study': cfg['study'], 'development': stats, 'argument_comparison_feasible': feasible}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config')
    main(parser.parse_args().config)
