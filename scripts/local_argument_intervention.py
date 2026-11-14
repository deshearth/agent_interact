"""Collect independent local-source responses to the fixed four-condition pilot."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as pilot
from scripts.local_argument_source import LocalSource, ROOT as SESSION

ROOT = SESSION / "intervention"


def make_plan(prompts, repeats):
    if repeats < 5 or repeats % 5:
        raise ValueError("Repeat count must be a positive multiple of five")
    order = list(prompts)
    random.Random(912503).shuffle(order)
    plan = []
    for row in order:
        for i in range(repeats):
            plan.append({"sample_id": pilot.text_sha("local-llama-v1|" + row["prompt_id"] + f"|{i}"),
                         "prompt_id": row["prompt_id"], "sample_index": i, "majority_group": i // 5,
                         "vote_position": i % 5, "prompt_sha256": row["prompt_sha256"]})
    return plan


def collect(repeats=20, max_seconds=1200):
    benchmark = json.loads((SESSION / "benchmark.json").read_text())
    assert benchmark["status"] == "complete"
    assert benchmark["source_code_sha256"] == pilot.sha("scripts/local_argument_source.py")
    _, prompts, _ = pilot.verify_prepared(pilot.ROOT)
    provenance = json.loads((SESSION / "model_provenance.json").read_text())
    plan = make_plan(prompts, repeats)
    protocol = {"model_repository": provenance["repository"], "model_revision": provenance["revision"],
                "model_provenance_sha256": pilot.sha(SESSION / "model_provenance.json"),
                "source_code_sha256": pilot.sha("scripts/local_argument_source.py"),
                "collector_code_sha256": pilot.sha(__file__), "parent_prompts_sha256": pilot.sha(pilot.ROOT / "prompts.jsonl"),
                "conditions": list(pilot.CONDITIONS), "cases": 24, "prompts": 96, "repeats": repeats,
                "votes_per_group": 5, "temperature": .7, "top_p": 1., "top_k": 0, "max_tokens": 256,
                "source": "Local MLX 4-bit Llama-3.1-8B-Instruct; not the historical hosted source or GPT-4o-mini.",
                "scope": "New response samples on four reused questions and GPT-derived background inboxes. No full-community rollout or fresh-question performance claim from this intervention set.",
                "conditioning": "Fixed expertise/persona, question, signed inbox and target stance; only the target rationale changes.",
                "source_instruction": "Released snap-judgment instruction; no reasoning tokens requested or forced A/B decoding.",
                "sampling": "Full sequences sampled independently with sample-ID-derived seeds. Prompt blocks are shuffled once; 20 draws reuse only a copied deterministic prefix cache. No decode history is carried between samples.",
                "parse_policy": "Released standalone A/B parser. Retain all raw responses and format failures. No resampling or cross-group replacement. Truncated outputs are invalid.",
                "diagnostic": "Save exact first literal-A/B token probabilities at temperature .7, with other-token mass. These are a separate diagnostic, not parsed-response or majority probabilities.",
                "paid_api_calls": 0}
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        pilot.write_json(protocol_path, protocol)
        pilot.write_jsonl(ROOT / "plan.jsonl", plan)
        pilot.write_jsonl(ROOT / "prompts.jsonl", prompts)
    existing = pilot.read_jsonl(ROOT / "responses.jsonl")
    done = {r["sample_id"] for r in existing}
    assert len(done) == len(existing) and done <= {r["sample_id"] for r in plan}
    source = LocalSource()
    by_prompt = {r["prompt_id"]: r for r in prompts}
    grouped = {}
    for row in plan:
        grouped.setdefault(row["prompt_id"], []).append(row)
    session = json.loads((SESSION / "session.json").read_text())
    deadline = min(time.time() + max_seconds, datetime.fromisoformat(session["deadline_utc"]).timestamp() - 600)
    begin = time.monotonic()
    completed_prompts = 0
    with (ROOT / "responses.jsonl").open("a") as output, (ROOT / "rendered_prompts.jsonl").open("a") as rendered:
        for pid, planned in grouped.items():
            pending = [r for r in planned if r["sample_id"] not in done]
            if not pending:
                completed_prompts += 1
                continue
            if time.time() >= deadline:
                break
            row = by_prompt[pid]
            tokens = source.render(row["prompt"])
            cache, prefill = source.prefill(tokens)
            rendered.write(json.dumps({"prompt_id": pid, "token_ids": tokens,
                                      "token_ids_sha256": pilot.text_sha(json.dumps(tokens)), "prefill_seconds": prefill}) + "\n")
            rendered.flush()
            first = None
            for planned_row in pending:
                if time.time() >= deadline:
                    break
                response = source.sample(tokens, planned_row["sample_id"], cache=cache, max_tokens=256)
                p = [response[k] for k in ("first_literal_A_probability", "first_literal_B_probability")]
                if first is None:
                    first = p
                else:
                    assert max(abs(x - y) for x, y in zip(first, p)) < 1e-6, "Prefix cache changed across draws"
                record = {**planned_row, **{k: row[k] for k in ("case_id", "qid", "condition", "relationship_sign", "message_stance")},
                          **response, "created_utc": datetime.now(timezone.utc).isoformat()}
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                done.add(planned_row["sample_id"])
            del cache
            source.mx.clear_cache()
            completed_prompts += int(all(r["sample_id"] in done for r in planned))
            if completed_prompts % 4 == 0:
                print(json.dumps({"completed_prompts": completed_prompts, "responses": len(done),
                                  "elapsed_seconds": time.monotonic() - begin,
                                  "peak_memory_gb": source.mx.get_peak_memory() / 1e9}), flush=True)
                pilot.write_json(ROOT / "progress.json", {"responses": len(done), "planned": len(plan),
                                 "completed_prompts": completed_prompts, "elapsed_seconds": time.monotonic() - begin})
    files = ("protocol.json", "plan.jsonl", "prompts.jsonl", "responses.jsonl", "rendered_prompts.jsonl")
    result = {"status": "complete" if len(done) == len(plan) else "partial", "responses": len(done),
              "planned": len(plan), "paid_api_calls": 0, "source_model": provenance["repository"],
              "elapsed_seconds_this_invocation": time.monotonic() - begin, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "peak_memory_gb": source.mx.get_peak_memory() / 1e9,
              "artifacts": {f: pilot.sha(ROOT / f) for f in files}}
    pilot.write_json(ROOT / "collection_status.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    collect(args.repeats, args.max_seconds)
