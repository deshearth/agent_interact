"""Counterbalanced direct validity grading of the four intervention arguments."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts import local_conversation_pilot as conversation
from scripts.local_argument_source import LocalSource, ROOT as SESSION

ROOT = SESSION / "direct_argument_grading"


def make_prompt(persona, question, choices, argument, valid_label):
    invalid_label = "B" if valid_label == "A" else "A"
    labels = {valid_label: "The mathematical reasoning is valid.", invalid_label: "The reasoning contains a mathematical or logical error."}
    return (f"# Your Expertise\n{persona}\n\n# Problem\n{question}\n"
            f"Problem answer choices:\nA) {choices['A']}\nB) {choices['B']}\n\n"
            f"# Argument to grade\n{argument}\n\n"
            "# Grading Task\nJudge the mathematical reasoning in the argument. Check its numerical claims, logical steps, and consistency. "
            "A correct final answer does not make incorrect intermediate reasoning valid. Grade the reasoning, not which problem answer you favor.\n"
            f"The grading labels below are independent of the problem answer choices:\nA) {labels['A']}\nB) {labels['B']}\n\n"
            "# Response Format\nReturn only the single grading label, A or B, with no explanation or punctuation.")


def prepare():
    _, _, _ = io.verify_prepared(io.ROOT)
    cases = json.loads((io.ROOT / "cases.json").read_text())
    variants = json.loads((io.ROOT / "argument_variants.json").read_text())
    personas = json.loads(Path("data/raw/upstream/data/obj/personas.json").read_text())
    rows = []
    for qid in sorted({r["qid"] for r in cases}):
        case = next(c for c in cases if c["qid"] == qid)
        for persona_index in (0, 1, 2):
            persona = personas[persona_index]["persona"]
            for condition in ("valid", "invalid", "valid_paraphrase"):
                for valid_label in ("A", "B"):
                    prompt = make_prompt(persona, case["question"], case["choices"], variants[qid][condition], valid_label)
                    expected = valid_label if condition != "invalid" else ("B" if valid_label == "A" else "A")
                    rows.append({"context_id": f"grade|{qid}|persona{persona_index}|{condition}|valid{valid_label}",
                                 "qid": qid, "persona": persona_index, "condition": condition, "valid_label": valid_label,
                                 "expected_label": expected, "expected_sign": 1 if expected == "A" else -1,
                                 "prompt": prompt, "prompt_sha256": io.text_sha(prompt)})
    random.Random(702617).shuffle(rows)
    protocol = {"collector_code_sha256": io.sha(__file__), "source_code_sha256": io.sha("scripts/local_argument_source.py"),
                "recorder_code_sha256": io.sha("scripts/local_conversation_pilot.py"), "model_provenance_sha256": io.sha(SESSION / "model_provenance.json"),
                "parent_variants_sha256": io.sha(io.ROOT / "argument_variants.json"), "parent_cases_sha256": io.sha(io.ROOT / "cases.json"),
                "prompts": 72, "repeats": 20, "planned_responses": 1440, "temperature": .7, "max_tokens": 256,
                "selection": "All four existing intervention questions, valid/flawed/paraphrased arguments, first three canonical personas, both valid=A and valid=B grading label assignments. No outcome-based selection.",
                "purpose": "Manipulation/capability diagnostic: can this local source distinguish the deliberately edited reasoning when explicitly asked to grade it?",
                "scope": "Separate direct-grading task without an interaction inbox. It does not alter the graph in any agent experiment, change original conversation outcomes, or establish why an opinion update did or did not change.",
                "sampling": "Twenty independent full responses per prompt, prompt blocks shuffled. No parse retries or replacement samples. Save raw answers and unrenormalized first-literal-A/B probabilities.",
                "validity_ground_truth": "Hand-constructed mathematical arguments checked by the earlier exact arithmetic audit. Valid paraphrases are valid; invalid variants alter one numerical/variable claim while retaining the correct final conclusion.",
                "interpretation": "Accuracy here is argument-validity grading on four reused problems, not mathematical problem-solving accuracy or dynamics prediction accuracy. Counterbalanced labels probe confusion with the problem's own A/B choices."}
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
                                 {**{k: row[k] for k in ("context_id", "qid", "persona", "condition", "valid_label", "expected_label", "expected_sign")},
                                  "kind": "validity_grade", "draw": i})
            completed += 1
            if completed % 12 == 0:
                print(json.dumps({"prompts": completed, "responses": len(recorder.records),
                                  "elapsed_seconds": time.monotonic() - begin}), flush=True)
    except conversation.DeadlineReached:
        print("Time bound reached; raw samples are saved.", flush=True)
    finally:
        recorder.close()
    result = {"status": "complete" if completed == 72 else "partial", "prompts": completed,
              "responses": len(recorder.records), "planned_responses": 1440, "paid_api_calls": 0,
              "parse_failures": sum(r["parsed_sign"] is None for r in recorder.records.values()),
              "strict_format_failures": sum(not r["strict_format"] for r in recorder.records.values()),
              "elapsed_seconds_this_invocation": time.monotonic() - begin, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "peak_memory_gb": source.mx.get_peak_memory() / 1e9,
              "artifacts": {str(p.relative_to(ROOT)): io.sha(p) for p in sorted(ROOT.glob("*.json*")) if p.name != "collection_status.json"}}
    io.write_json(ROOT / "collection_status.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect"))
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(f"Prepared {len(prepare())} counterbalanced validity-grading prompts.")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / ".collector.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(args.max_seconds)
