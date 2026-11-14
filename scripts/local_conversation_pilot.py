"""Generate bounded, fully local Llama conversations using released prompts."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts.local_argument_source import LocalSource, ROOT as SESSION, sample_seed

ROOT = SESSION / "conversations"
UPSTREAM = Path("data/raw/upstream")
FORMATTER = UPSTREAM / "lib/datagen/samplers.py"
FIXED_QUESTIONS = ("math_train_133", "math_train_162", "math_train_359", "math_train_497")


def formatter():
    spec = importlib.util.spec_from_file_location("frozen_local_prompt_formatter", FORMATTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate(votes):
    total = sum(votes)
    return 1 if total > 0 else -1 if total < 0 else votes[0] if votes else 0


def deliver(graph, sender_messages, episode, step):
    """Broadcast a sender's one message, then apply released inbox ordering."""
    n = len(graph)
    agree, disagree = [[] for _ in range(n)], [[] for _ in range(n)]
    edges = {(i, j): sender_messages[i] for i in sorted(sender_messages)
             for j in range(n) if graph[i][j] != 0}
    for receiver in range(n):
        incoming = [(i, text) for (i, j), text in sorted(edges.items()) if j == receiver]
        random.Random(f"{episode}|{step}|{receiver}").shuffle(incoming)
        for sender, text in incoming:
            (agree if graph[sender][receiver] == 1 else disagree)[receiver].append(text)
    return agree, disagree, {f"{i},{j}": text for (i, j), text in sorted(edges.items())}


def opinion_prompt(fmt, persona, question, agree, disagree):
    return fmt._objective_context(persona, question["question"], question["choices"], agree, disagree) + fmt.OBJECTIVE_SPIN_INSTRUCTION


def message_prompt(fmt, persona, question, agree, disagree, spin, identity):
    stance = spin if spin else random.Random(sample_seed(identity + "|tie")).choice((-1, 1))
    prompts = []
    fmt.message_sampler(persona, question["question"], agree, disagree, stance,
                        lambda p: prompts.append(p) or "captured", mode="objective", choices=question["choices"])
    assert len(prompts) == 1
    return prompts[0], stance


class DeadlineReached(Exception):
    pass


class Recorder:
    def __init__(self, source, root, deadline):
        self.source, self.root, self.deadline = source, Path(root), deadline
        self.root.mkdir(parents=True, exist_ok=True)
        records = io.read_jsonl(self.root / "source_responses.jsonl")
        self.records = {r["sample_id"]: r for r in records}
        assert len(records) == len(self.records)
        prompts = io.read_jsonl(self.root / "source_prompts.jsonl")
        self.prompts = {r["prompt_sha256"]: r for r in prompts}
        assert len(prompts) == len(self.prompts)
        self.samples = (self.root / "source_responses.jsonl").open("a")
        self.prompt_file = (self.root / "source_prompts.jsonl").open("a")
        self.cached_hash = self.tokens = self.cache = None

    def request(self, prompt, identity, metadata):
        digest = io.text_sha(prompt)
        if identity in self.records:
            row = self.records[identity]
            assert row["prompt_sha256"] == digest
            assert all(row[k] == v for k, v in metadata.items())
            return row
        if time.time() >= self.deadline:
            raise DeadlineReached()
        if digest != self.cached_hash:
            self.cache = None
            self.source.mx.clear_cache()
            self.tokens = self.source.render(prompt)
            self.cache, seconds = self.source.prefill(self.tokens)
            self.cached_hash = digest
            if digest not in self.prompts:
                row = {"prompt_sha256": digest, "prompt": prompt, "token_ids": self.tokens,
                       "token_ids_sha256": io.text_sha(json.dumps(self.tokens)), "prefill_seconds": seconds}
                self.prompt_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.prompt_file.flush()
                self.prompts[digest] = row
        result = self.source.sample(self.tokens, identity, cache=self.cache, max_tokens=256)
        row = {"sample_id": identity, "prompt_sha256": digest, **metadata, **result,
               "created_utc": datetime.now(timezone.utc).isoformat()}
        self.samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.samples.flush()
        self.records[identity] = row
        return row

    def close(self):
        self.samples.close()
        self.prompt_file.close()
        self.cache = None


def vote_group(recorder, prompt, identity, metadata):
    votes = []
    for position in range(5):
        vote = 0
        for attempt in range(3):
            row = recorder.request(prompt, f"{identity}|vote{position}|attempt{attempt}",
                                   {**metadata, "kind": "opinion", "vote_position": position, "parse_attempt": attempt})
            if row["parsed_sign"] in (-1, 1):
                vote = row["parsed_sign"]
                break
        votes.append(vote)
    return votes


def run_episode(question, personas, graph, steps, recorder, root, fmt):
    name = "local_llama_" + question["qid"] + "_J0_r0"
    path = Path(root) / "episodes" / (name + ".json")
    if path.exists():
        return json.loads(path.read_text())
    n = len(personas)
    agree, disagree = [[] for _ in personas], [[] for _ in personas]
    spins, raw, messages, sender_stances = [], [], [], []
    for step in range(steps + 1):
        if step:
            # All messages use the same completed prior state and inboxes.
            sender_messages, stances = {}, {}
            for agent in range(n):
                if not any(graph[agent]):
                    continue
                identity = f"{name}|round{step}|message|agent{agent}"
                prompt, stance = message_prompt(fmt, personas[agent], question, agree[agent], disagree[agent], spins[-1][agent], identity)
                row = recorder.request(prompt, identity, {"episode": name, "qid": question["qid"], "round": step,
                                       "kind": "message", "agent": agent, "communicated_stance": stance})
                sender_messages[agent], stances[str(agent)] = row["raw_response"].strip(), stance
            agree, disagree, edges = deliver(graph, sender_messages, name, step)
            messages.append(edges)
            sender_stances.append(stances)
        current = []
        for agent in range(n):
            prompt = opinion_prompt(fmt, personas[agent], question, agree[agent], disagree[agent])
            current.append(vote_group(recorder, prompt, f"{name}|round{step}|agent{agent}",
                                      {"episode": name, "qid": question["qid"], "round": step, "agent": agent}))
        raw.append(current)
        spins.append([aggregate(v) for v in current])
        # Phase checkpoints are reviewable; a restart replays recorded samples.
        io.write_json(Path(root) / "phase_checkpoints" / (name + ".json"),
                      {"name": name, "completed_opinion_round": step, "spins_history": spins,
                       "spins_raw_history": raw, "messages_history": messages})
        print(json.dumps({"episode": name, "completed_opinion_round": step,
                          "source_responses_total": len(recorder.records)}), flush=True)
    payload = {"meta": {"name": name, "qid": question["qid"], "source": "local_4bit_llama_3.1_8b", "n": n,
                         "k": 5, "steps": steps, "protocol_sha256": io.sha(Path(root) / "protocol.json")},
               "mode": "objective", "statement": question["question"], "choices": question["choices"],
               "personas": personas, "J": graph, "spins_history": spins, "spins_raw_history": raw,
               "messages_history": messages, "communicated_stances_history": sender_stances}
    io.write_json(path, payload)
    return payload


def prepare(root=ROOT, steps=2):
    root = Path(root)
    questions = io.read_jsonl(UPSTREAM / "data/obj/train.jsonl")
    qids = list(FIXED_QUESTIONS) + sorted(q["qid"] for q in questions if q["qid"] not in FIXED_QUESTIONS)[:4]
    selected = [next(q for q in questions if q["qid"] == qid) for qid in qids]
    personas = [p["persona"] if isinstance(p, dict) else p for p in json.loads((UPSTREAM / "data/obj/personas.json").read_text())]
    graph = next(g["J"] for g in json.loads((UPSTREAM / "data/J_matrices.json").read_text())["graphs"] if g["id"] == "J0")
    assert len(personas) == len(graph) == 32 and all(len(r) == 32 for r in graph)
    provenance = json.loads((SESSION / "model_provenance.json").read_text())
    files = [UPSTREAM / "data/obj/train.jsonl", UPSTREAM / "data/obj/personas.json", UPSTREAM / "data/J_matrices.json", FORMATTER,
             Path("scripts/local_argument_source.py"), Path(__file__), SESSION / "model_provenance.json"]
    protocol = {"model_repository": provenance["repository"], "revision": provenance["revision"], "quantization": "MLX 4-bit",
                "source_files": {str(p): io.sha(p) for p in files}, "questions": qids, "n": 32, "steps": steps, "k": 5,
                "graph": "J0", "replicas_per_question": 1, "temperature": .7, "top_p": 1., "max_tokens": 256,
                "question_selection": "Four prior intervention questions followed by the first four other development qids in lexicographic order; chosen before examining local outcomes.",
                "scope": "New Llama opinions AND messages on reused objective development questions; a generation pilot, not a fresh-question predictor evaluation.",
                "message_rule": "One sampled message per connected sender per round, broadcast to its neighbors. All senders use completed prior spins/inboxes. Then synchronous opinion updates from the new inboxes.",
                "graph_rule": "J[sender,receiver] selects the released tend-to-agree or tend-to-disagree inbox headings. No learned relationship weights.",
                "opinion_rule": "Five independently seeded full responses; same released snap-judgment prompt and A/B parser. Up to two extra parse attempts per vote, all logged. Persistent failure becomes zero; majority sign, with first vote breaking a tie. Truncated opinions invalid.",
                "cache": "Copied deterministic prefix KV only; no sampled response is carried into another vote. Exact user prompts and rendered token IDs are saved.",
                "changes_from_paper": ["Local quantized MLX serving and deterministic per-sample seeds", "Two update rounds instead of eight", "One graph and eight reused questions with one replica each", "Opinion outputs truncated at 256 tokens are counted as parse failures"],
                "paid_api_calls": 0}
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        assert json.loads((root / "protocol.json").read_text()) == protocol
    else:
        io.write_json(root / "protocol.json", protocol)
        io.write_json(root / "questions.json", selected)
    return selected, personas, graph


def collect(steps=2, max_seconds=3300):
    questions, personas, graph = prepare(steps=steps)
    session = json.loads((SESSION / "session.json").read_text())
    deadline = min(time.time() + max_seconds, datetime.fromisoformat(session["deadline_utc"]).timestamp() - 600)
    benchmark = json.loads((SESSION / "benchmark.json").read_text())
    assert benchmark["source_code_sha256"] == io.sha("scripts/local_argument_source.py")
    begin = time.monotonic()
    source = LocalSource()
    recorder = Recorder(source, ROOT, deadline)
    fmt, episodes = formatter(), []
    try:
        for question in questions:
            episodes.append(run_episode(question, personas, graph, steps, recorder, ROOT, fmt))
    except DeadlineReached:
        print("Time limit reached; all raw responses and complete phases are saved.", flush=True)
    finally:
        recorder.close()
    opinion = [r for r in recorder.records.values() if r["kind"] == "opinion"]
    messages = [r for r in recorder.records.values() if r["kind"] == "message"]
    result = {"status": "complete" if len(episodes) == len(questions) else "partial", "episodes": len(episodes),
              "planned_episodes": len(questions), "completed_transitions": len(episodes) * len(personas) * steps,
              "opinion_response_attempts": len(opinion), "message_responses": len(messages),
              "opinion_parse_failures": sum(r["parsed_sign"] is None for r in opinion),
              "opinion_extra_parse_attempts": sum(r["parse_attempt"] > 0 for r in opinion),
              "truncated_messages": sum(r["finish_reason"] == "length" for r in messages),
              "paid_api_calls": 0, "elapsed_seconds_this_invocation": time.monotonic() - begin,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "peak_memory_gb": source.mx.get_peak_memory() / 1e9,
              "artifacts": {str(p.relative_to(ROOT)): io.sha(p) for p in sorted(ROOT.rglob("*.json*")) if p.name != "collection_status.json"}}
    io.write_json(ROOT / "collection_status.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=3300)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(steps=args.steps)
        print("Prepared eight-question, 32-agent local conversation protocol.")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / ".collector.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(args.steps, args.max_seconds)
