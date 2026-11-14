"""Explicit, bounded source-model execution for the prepared intervention pilot.

Dry-run is the default. No credentials are needed for dry-run. Live calls require
--execute and a locally configured key. A durable attempt ledger prevents silent
retries, duplicate sample collection, or regrouping after invalid outputs.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as pilot


def now():
    return datetime.now(timezone.utc).isoformat()


def append_event(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ledger_state(rows):
    started, finished = {}, {}
    for row in rows:
        sid = row["sample_id"]
        if row["event"] == "started":
            if sid in started:
                raise ValueError("A sample was dispatched more than once")
            started[sid] = row
        elif row["event"] in ("completed", "failed"):
            if sid not in started or sid in finished:
                raise ValueError("Unmatched or duplicate terminal event")
            finished[sid] = row
        else:
            raise ValueError("Unknown ledger event")
    unresolved = set(started) - set(finished)
    charged_or_reserved = sum(
        max(started[sid]["reserved_cost_usd"], finished[sid].get("usage_cost_usd") or 0)
        if sid in finished else started[sid]["reserved_cost_usd"] for sid in started)
    return started, finished, unresolved, charged_or_reserved


class OpenAITransport:
    def __init__(self, key, protocol, timeout=45):
        if not key or "\n" in key or "\r" in key:
            raise ValueError("A valid locally configured API key is required")
        self.key, self.protocol, self.timeout = key, protocol, timeout

    def __call__(self, prompt):
        p = self.protocol
        body = {"model": p["source_model"], "messages": [{"role": "user", "content": prompt}],
                "temperature": p["temperature"], "max_completion_tokens": p["max_completion_tokens"],
                "n": 1, "store": False, "stream": False}
        request = urllib.request.Request(p["endpoint"], data=json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
                                         method="POST")
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            # Do not log response bodies or authorization values.
            return {"event": "failed", "failure_kind": "http_error", "http_status": exc.code,
                    "latency_seconds": time.monotonic() - start}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"event": "failed", "failure_kind": type(exc).__name__,
                    "latency_seconds": time.monotonic() - start, "billing_uncertain": True}
        choices = data.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            return {"event": "failed", "failure_kind": "unexpected_response_schema", "billing_uncertain": True}
        choice = choices[0]
        message = choice["message"]
        text = message.get("content") or ""
        if not isinstance(text, str):
            return {"event": "failed", "failure_kind": "unexpected_content_schema", "billing_uncertain": True}
        parsed, strict = pilot.parse_answer(text)
        finish = choice.get("finish_reason")
        if finish != "stop" or message.get("refusal"):
            parsed, strict = None, False
        usage = data.get("usage", {})
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
        usage_valid = all(isinstance(v, int) and v >= 0 for v in (inputs, outputs))
        cost = ((inputs * p["input_usd_per_million"] + outputs * p["output_usd_per_million"]) / 1e6
                if usage_valid else None)
        return {"event": "completed", "response_id": data.get("id"), "request_id": request_id,
                "response_model": data.get("model"), "system_fingerprint": data.get("system_fingerprint"),
                "raw_response": text, "parsed_sign": parsed, "strict_format": strict,
                "finish_reason": finish, "refusal": bool(message.get("refusal")), "usage": usage,
                "usage_cost_usd": cost, "latency_seconds": time.monotonic() - start,
                "usage_price_note": "Conservatively bill all reported input tokens at the uncached input rate."}


def preflight(root=pilot.ROOT):
    root = Path(root)
    protocol, prompts, schedule = pilot.verify_prepared(root)
    expected = {r["sample_id"]: r for r in schedule}
    rows = pilot.read_jsonl(root / "source_ledger.jsonl")
    started, finished, unresolved, used = ledger_state(rows)
    if not set(started).issubset(expected):
        raise ValueError("Ledger contains samples outside the prepared schedule")
    for sid, event in started.items():
        if event["prompt_sha256"] != expected[sid]["prompt_sha256"]:
            raise ValueError("Prompt hash in ledger disagrees with schedule")
    return {"protocol": protocol, "prompts": prompts, "schedule": schedule,
            "started": started, "finished": finished, "unresolved": unresolved,
            "reserved_cost_usd": used}


def execute(root, transport, *, max_cost_usd=1., max_seconds=1800, workers=4, max_requests=1920):
    root = Path(root)
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0 or not 1 <= workers <= 4 or max_requests < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Invalid execution bounds")
    lock = (root / "source_execution.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another process already holds the pilot execution lock") from None
    with lock:
        state = preflight(root)
        if state["unresolved"]:
            raise RuntimeError("Interrupted attempts require reconciliation; no automatic redispatch")
        protocol = state["protocol"]
        identity = {"protocol_sha256": pilot.sha(root / "protocol.json"), "runner_code_sha256": pilot.sha(__file__),
                    "transport": "OpenAITransport" if isinstance(transport, OpenAITransport) else "synthetic_test_transport"}
        identity_path = root / "execution_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Execution implementation or transport changed; use a new run directory")
        pilot.write_json(identity_path, identity)
        lookup = {r["prompt_id"]: r for r in state["prompts"]}
        remaining = [r for r in state["schedule"] if r["sample_id"] not in state["started"]]
        spent = state["reserved_cost_usd"]
        began = time.monotonic()
        cursor, submitted = 0, 0
        pending = {}
        stop_reason = None
        ledger_path = root / "source_ledger.jsonl"
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while pending or (cursor < len(remaining) and stop_reason is None):
                while len(pending) < workers and cursor < len(remaining) and stop_reason is None:
                    item = remaining[cursor]
                    prompt = lookup[item["prompt_id"]]
                    reserve = pilot.reserve_cost(prompt["prompt"])
                    if submitted >= max_requests:
                        stop_reason = "request_limit"
                    elif time.monotonic() - began >= max_seconds:
                        stop_reason = "time_limit"
                    elif spent + reserve > max_cost_usd:
                        stop_reason = "cost_limit"
                    if stop_reason is not None:
                        break
                    event = {**item, "event": "started", "utc": now(), "reserved_cost_usd": reserve}
                    append_event(ledger_path, event)
                    spent += reserve
                    pending[executor.submit(transport, prompt["prompt"])] = event
                    cursor += 1
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=min(1., max_seconds), return_when=FIRST_COMPLETED)
                for future in done:
                    item = pending.pop(future)
                    try:
                        response = future.result()
                    except Exception as exc:
                        response = {"event": "failed", "failure_kind": type(exc).__name__, "billing_uncertain": True}
                    response = {**response, **{k: item[k] for k in ("sample_id", "prompt_id", "prompt_sha256", "sample_index", "majority_group", "vote_position", "schedule_position")}, "utc": now()}
                    append_event(ledger_path, response)
                    if response["event"] != "completed":
                        stop_reason = "source_error"
                    elif response.get("response_model") != protocol["source_model"]:
                        stop_reason = "unexpected_model"
                    elif response.get("usage_cost_usd") is None:
                        stop_reason = "missing_usage"
                    elif response["usage_cost_usd"] > item["reserved_cost_usd"]:
                        spent += response["usage_cost_usd"] - item["reserved_cost_usd"]
                        stop_reason = "cost_estimate_exceeded"
                if submitted and submitted % 96 == 0 and not pending:
                    print(json.dumps({"new_requests": submitted, "reserved_cost_usd": spent}), flush=True)
        final = preflight(root)
        result = {**identity, "status": "complete" if len(final["finished"]) == len(final["schedule"]) and all(r["event"] == "completed" for r in final["finished"].values()) else "partial",
                  "stop_reason": stop_reason or "schedule_exhausted", "finished_utc": now(),
                  "requests_this_invocation": submitted, "attempts_total": len(final["started"]),
                  "returned_responses_total": sum(r["event"] == "completed" for r in final["finished"].values()),
                  "reserved_cost_usd": final["reserved_cost_usd"], "max_cost_usd": max_cost_usd,
                  "elapsed_seconds": time.monotonic() - began,
                  "budget_note": "Conservative reserves are never released. Billing uncertainty after failures is retained. Time cap stops new dispatches; at most four in-flight requests can finish afterward."}
        pilot.write_json(root / "execution_status.json", result)
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=pilot.ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-cost-usd", type=float, default=1.)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--max-requests", type=int, default=1920)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    state = preflight(args.root)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "network_requests": 0, "source_model": state["protocol"]["source_model"],
                          "prompts": len(state["prompts"]), "planned_requests": len(state["schedule"]),
                          "attempts_already_recorded": len(state["started"]), "unresolved_attempts": len(state["unresolved"]),
                          "cost_plan": json.loads((args.root / "cost_plan.json").read_text())}, indent=2))
        return
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("No API credential configured. Set OPENAI_API_KEY locally or use --api-key-file; do not put a key in chat or command arguments.")
    transport = OpenAITransport(key, state["protocol"])
    print(json.dumps(execute(args.root, transport, max_cost_usd=args.max_cost_usd,
                             max_seconds=args.max_seconds, max_requests=args.max_requests, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
