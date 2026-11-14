"""Remove rationales from every inbox in the fully local conversation pilot."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts import local_conversation_pilot as conversation
from scripts.local_argument_source import LocalSource, ROOT as SESSION

ROOT = SESSION / "rationale_ablation"
PARENT = SESSION / "conversations"
STANCE = re.compile(r"\b(?:favor|favour|support|choose|prefer|select|endorse|agree\s+with|believe)\s+"
                    r"(?:option\s+|answer\s+)?[\s*\"'`]*([AB])\b", re.I)


def declared_stance(text):
    """Conservative explicit-stance audit; no inference of argument correctness."""
    matches = {m.upper() for m in STANCE.findall(text[:160])}
    return 1 if matches == {"A"} else -1 if matches == {"B"} else None


def protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    value = {"parent_protocol_sha256": io.sha(PARENT / "protocol.json"), "source_code_sha256": io.sha("scripts/local_argument_source.py"),
             "collector_code_sha256": io.sha(__file__), "recorder_code_sha256": io.sha("scripts/local_conversation_pilot.py"),
             "model_provenance_sha256": io.sha(SESSION / "model_provenance.json"),
             "selection": "Every receiver at both update rounds of all eight planned local episodes, including empty inboxes. No selection by source outcomes or intervention effect.",
             "intervention": "Replace each delivered message with 'I favor option A.' or 'I favor option B.' using that sender's recorded message-prompt stance. Preserve persona, question, choices, graph signs, source association, deterministic inbox order, and opinion instruction.",
             "stance_audit": "Conservatively parse the first 160 characters of the original message for an explicit stance. Flag whether every incoming message explicitly matches its sender's instructed stance. Report this subset alongside all contexts; do not assume instruction compliance.",
             "controls": "Receivers with no neighbors retain byte-identical prompts, providing a deterministic first-token probability negative control.",
             "target": "Five independent full-response opinion votes per ablated prompt, with the same up-to-two parse retries as the original local conversations. Save all attempts. Also compare first literal-A/B token probabilities without renormalizing.",
             "scope": "Within-local-model intervention on reused development questions. Deleting all rationales changes text length and wording; it measures dependence on full message text beyond the recorded stances, not reasoning correctness or neural predictor improvement.",
             "temperature": .7, "max_tokens": 256, "planned_contexts": 512, "paid_api_calls": 0}
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == value
    else:
        io.write_json(path, value)
    return value


def make_inputs():
    protocol()
    status = json.loads((PARENT / "collection_status.json").read_text())
    assert status["status"] == "complete", "Only run after all original conversations are complete"
    for name, digest in status["artifacts"].items():
        assert io.sha(PARENT / name) == digest, name
    questions = {q["qid"]: q for q in json.loads((PARENT / "questions.json").read_text())}
    source = {r["sample_id"]: r for r in io.read_jsonl(PARENT / "source_responses.jsonl")}
    fmt, rows = conversation.formatter(), []
    for path in sorted((PARENT / "episodes").glob("*.json")):
        ep = json.loads(path.read_text())
        name, qid, n = ep["meta"]["name"], ep["meta"]["qid"], ep["meta"]["n"]
        for step, messages in enumerate(ep["messages_history"], 1):
            short, original, compliant = {}, {}, {}
            for key, text in messages.items():
                sender, receiver = map(int, key.split(","))
                stance = ep["communicated_stances_history"][step-1][str(sender)]
                short[sender] = "I favor option " + ("A" if stance == 1 else "B") + "."
                original[sender] = text
                compliant[sender] = declared_stance(text) == stance
            agree, disagree, _ = conversation.deliver(ep["J"], short, name, step)
            old_agree, old_disagree, rebuilt = conversation.deliver(ep["J"], original, name, step)
            assert rebuilt == messages
            for receiver in range(n):
                prompt = conversation.opinion_prompt(fmt, ep["personas"][receiver], questions[qid], agree[receiver], disagree[receiver])
                old_prompt = conversation.opinion_prompt(fmt, ep["personas"][receiver], questions[qid], old_agree[receiver], old_disagree[receiver])
                old_id = f"{name}|round{step}|agent{receiver}|vote0|attempt0"
                assert source[old_id]["prompt_sha256"] == io.text_sha(old_prompt)
                senders = [sender for sender in range(n) if ep["J"][sender][receiver]]
                rows.append({"context_id": f"ablation|{name}|round{step}|agent{receiver}", "episode": name, "qid": qid,
                             "round": step, "agent": receiver, "prompt": prompt, "prompt_sha256": io.text_sha(prompt),
                             "original_prompt_sha256": io.text_sha(old_prompt), "original_first_response_id": old_id,
                             "incoming_senders": senders, "incoming_stances": [ep["communicated_stances_history"][step-1][str(s)] for s in senders],
                             "incoming_relationships": [ep["J"][s][receiver] for s in senders],
                             "all_incoming_stances_explicit_and_consistent": all(compliant[s] for s in senders),
                             "empty_inbox": not senders})
    assert len(rows) == 512
    if (ROOT / "inputs.jsonl").exists():
        assert io.read_jsonl(ROOT / "inputs.jsonl") == rows
    else:
        io.write_jsonl(ROOT / "inputs.jsonl", rows)
        io.write_json(ROOT / "input_manifest.json", {"inputs_sha256": io.sha(ROOT / "inputs.jsonl"),
                      "parent_collection_status_sha256": io.sha(PARENT / "collection_status.json"),
                      "contexts": len(rows), "empty_inbox_controls": sum(r["empty_inbox"] for r in rows)})
    return rows


def collect(max_seconds=1800):
    rows = make_inputs()
    session = json.loads((SESSION / "session.json").read_text())
    deadline = min(time.time() + max_seconds, datetime.fromisoformat(session["deadline_utc"]).timestamp() - 600)
    source = LocalSource()
    recorder = conversation.Recorder(source, ROOT, deadline)
    begin, completed = time.monotonic(), 0
    outcomes = []
    try:
        for row in rows:
            votes = conversation.vote_group(recorder, row["prompt"], row["context_id"],
                                            {k: row[k] for k in ("episode", "qid", "round", "agent")})
            outcomes.append({"context_id": row["context_id"], "votes": votes, "majority_sign": conversation.aggregate(votes)})
            completed += 1
            if completed % 32 == 0:
                print(json.dumps({"contexts": completed, "responses": len(recorder.records),
                                  "elapsed_seconds": time.monotonic() - begin}), flush=True)
                io.write_json(ROOT / "progress.json", {"contexts": completed, "planned_contexts": len(rows), "responses": len(recorder.records)})
    except conversation.DeadlineReached:
        print("Time bound reached; all completed samples are saved.", flush=True)
    finally:
        recorder.close()
    io.write_json(ROOT / "outcomes.json", outcomes)
    result = {"status": "complete" if completed == len(rows) else "partial", "contexts": completed,
              "planned_contexts": len(rows), "source_response_attempts": len(recorder.records),
              "parse_failures": sum(r["parsed_sign"] is None for r in recorder.records.values()),
              "paid_api_calls": 0, "elapsed_seconds_this_invocation": time.monotonic() - begin,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "peak_memory_gb": source.mx.get_peak_memory() / 1e9,
              "artifacts": {str(p.relative_to(ROOT)): io.sha(p) for p in sorted(ROOT.glob("*.json*")) if p.name != "collection_status.json"}}
    io.write_json(ROOT / "collection_status.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("protocol", "prepare", "collect"))
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.stage == "protocol":
        protocol()
        print("Frozen all-512-context rationale-ablation protocol.")
    elif args.stage == "prepare":
        print(len(make_inputs()))
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / ".collector.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(args.max_seconds)
