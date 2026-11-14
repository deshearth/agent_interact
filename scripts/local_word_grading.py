"""Adaptive follow-up: direct VALID/FLAWED verdicts without A/B grading labels."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts import local_argument_grading as parent
from scripts import local_conversation_pilot as conversation
from scripts.local_argument_source import LocalSource, ROOT as SESSION

ROOT = SESSION / "word_argument_grading"


def parse_verdict(text):
    text = text.strip().upper()
    match = re.fullmatch(r"(VALID|FLAWED)[.!]?", text)
    return (match[1] if match else None), text in ("VALID", "FLAWED")


def prepare():
    original = parent.prepare()
    rows = []
    for row in original:
        # Both original label assignments become the two definition orders.
        order = ("VALID", "FLAWED") if row["valid_label"] == "A" else ("FLAWED", "VALID")
        definitions = {"VALID": "The mathematical reasoning is valid.", "FLAWED": "The reasoning contains a mathematical or logical error."}
        prefix = row["prompt"].split("The grading labels below are independent of the problem answer choices:")[0]
        prompt = (prefix + "Use one of these verdict words:\n" + "\n".join(word + ": " + definitions[word] for word in order)
                  + "\n\n# Response Format\nReturn only the single verdict word, VALID or FLAWED, with no explanation or punctuation.")
        rows.append({"context_id": row["context_id"].replace("grade|", "word_grade|", 1), "qid": row["qid"], "persona": row["persona"],
                     "condition": row["condition"], "definition_order": "_then_".join(order),
                     "expected_verdict": "FLAWED" if row["condition"] == "invalid" else "VALID",
                     "prompt": prompt, "prompt_sha256": io.text_sha(prompt)})
    random.Random(704623).shuffle(rows)
    protocol = {"collector_code_sha256": io.sha(__file__), "source_code_sha256": io.sha("scripts/local_argument_source.py"),
                "recorder_code_sha256": io.sha("scripts/local_conversation_pilot.py"), "parent_protocol_sha256": io.sha(parent.ROOT / "protocol.json"),
                "model_provenance_sha256": io.sha(SESSION / "model_provenance.json"),
                "motivation": "Adaptive follow-up after A/B grading scored 51.56% on balanced valid/flawed cases and showed 46.23 pp mean absolute label-swap probability sensitivity. Preserve that result; test direct semantic verdicts as a separate prompt regime.",
                "contexts": 72, "repeats": 20, "planned_responses": 1440, "temperature": .7, "max_tokens": 256,
                "controls": "Same four questions, three personas, three argument versions and grading instruction through the label definitions. Replace artificial A/B grade labels with VALID/FLAWED; reverse the definition order in half of prompts.",
                "primary": "Balanced valid-versus-flawed sampled grading accuracy; valid paraphrases secondary. Score all planned samples, with parse failures incorrect. Twenty independent full responses per prompt; no retries or replacements.",
                "parser": "Case-insensitive full response VALID or FLAWED, optionally ending in one period or exclamation mark. Record strict single-word compliance separately. Truncated responses invalid.",
                "probabilities": "The common source logger still saves first-literal-A/B diagnostics and an A/B parser field. Those fields are irrelevant to word verdicts and will NOT be used as grading probabilities. Analyze complete raw verdicts only.",
                "scope": "Exploratory response-format follow-up on the same four constructed problems, not a new independent test or an intervention on the communication graph."}
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        assert json.loads((ROOT / "protocol.json").read_text()) == protocol
        assert io.read_jsonl(ROOT / "inputs.jsonl") == rows
    else:
        io.write_json(ROOT / "protocol.json", protocol)
        io.write_jsonl(ROOT / "inputs.jsonl", rows)
    return rows


def collect(max_seconds=600):
    rows = prepare()
    session = json.loads((SESSION / "session.json").read_text())
    deadline = min(time.time() + max_seconds, datetime.fromisoformat(session["deadline_utc"]).timestamp() - 600)
    source = LocalSource()
    recorder = conversation.Recorder(source, ROOT, deadline)
    begin, completed = time.monotonic(), 0
    try:
        for row in rows:
            for i in range(20):
                recorder.request(row["prompt"], row["context_id"] + f"|draw{i}",
                    {**{k: row[k] for k in ("context_id", "qid", "persona", "condition", "definition_order", "expected_verdict")},
                     "kind": "word_validity_grade", "draw": i})
            completed += 1
            if completed % 12 == 0:
                print(json.dumps({"prompts": completed, "responses": len(recorder.records), "elapsed_seconds": time.monotonic()-begin}), flush=True)
    except conversation.DeadlineReached:
        print("Time bound reached; all completed raw verdicts are saved.", flush=True)
    finally:
        recorder.close()
    parsed = [parse_verdict(r["raw_response"]) if r["finish_reason"] == "stop" else (None, False) for r in recorder.records.values()]
    result = {"status": "complete" if completed == 72 else "partial", "prompts": completed, "responses": len(recorder.records),
              "planned_responses": 1440, "paid_api_calls": 0, "verdict_parse_failures": sum(v is None for v, _ in parsed),
              "verdict_strict_format_failures": sum(not strict for _, strict in parsed),
              "elapsed_seconds_this_invocation": time.monotonic()-begin, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "artifacts": {str(p.relative_to(ROOT)): io.sha(p) for p in sorted(ROOT.glob("*.json*")) if p.name != "collection_status.json"}}
    io.write_json(ROOT / "collection_status.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect"))
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(f"Prepared {len(prepare())} word-verdict prompts.")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / ".collector.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(args.max_seconds)
