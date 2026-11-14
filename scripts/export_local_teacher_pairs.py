"""Export audited, paired local-teacher examples without inventing a test split."""
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts.local_argument_source import ROOT


def main():
    base, alt = ROOT / "conversations", ROOT / "rationale_ablation"
    for path in (base / "analysis/audit.json", alt / "analysis/audit.json"):
        assert json.loads(path.read_text())["status"] == "complete"
    contexts = io.read_jsonl(alt / "inputs.jsonl")
    outcomes = {r["context_id"]: r for r in json.loads((alt / "outcomes.json").read_text())}
    full_responses = {r["sample_id"]: r for r in io.read_jsonl(base / "source_responses.jsonl")}
    short_responses = {r["sample_id"]: r for r in io.read_jsonl(alt / "source_responses.jsonl")}
    episodes = {p.stem: json.loads(p.read_text()) for p in (base / "episodes").glob("*.json")}
    questions = json.loads((base / "questions.json").read_text())
    first = next(iter(episodes.values()))
    static = {"personas": first["personas"], "graphs": {"J0": first["J"]},
              "questions": [{k: q[k] for k in ("qid", "question", "choices")} for q in questions]}
    rows = []
    for context in contexts:
        ep = episodes[context["episode"]]
        step, receiver = context["round"], context["agent"]
        assert ep["personas"] == static["personas"] and ep["J"] == static["graphs"]["J0"]
        edges = ep["messages_history"][step - 1]
        incoming = [(int(key.split(",")[0]), text) for key, text in edges.items() if int(key.split(",")[1]) == receiver]
        incoming.sort(key=lambda item: item[0])
        random.Random(f"{context['episode']}|{step}|{receiver}").shuffle(incoming)
        grouped = {"agree": [], "disagree": []}
        for sender, text in incoming:
            stance = ep["communicated_stances_history"][step - 1][str(sender)]
            sign = ep["J"][sender][receiver]
            grouped["agree" if sign == 1 else "disagree"].append({"sender": sender, "relationship_sign": sign,
                "prompted_stance": stance, "original_message": text,
                "conclusion_only_message": "I favor option " + ("A" if stance == 1 else "B") + "."})
        full = full_responses[context["original_first_response_id"]]
        short = short_responses[context["context_id"] + "|vote0|attempt0"]
        row = {"context_id": context["context_id"], "episode": context["episode"], "question_group": context["qid"],
            "inputs": {"qid": context["qid"], "graph_id": "J0", "receiver": receiver, "update_round": step,
                       "current_spins": ep["spins_history"][step - 1],
                       "previous_spins": ep["spins_history"][step - 2] if step >= 2 else None,
                       "delivered_messages": grouped},
            "outcomes": {"original": {"raw_votes": ep["spins_raw_history"][step][receiver], "majority_sign": ep["spins_history"][step][receiver],
                                      "first_literal_A_probability": full["first_literal_A_probability"], "first_literal_B_probability": full["first_literal_B_probability"]},
                         "conclusion_only": {"raw_votes": outcomes[context["context_id"]]["votes"], "majority_sign": outcomes[context["context_id"]]["majority_sign"],
                                             "first_literal_A_probability": short["first_literal_A_probability"], "first_literal_B_probability": short["first_literal_B_probability"]}},
            "audit": {"original_prompt_sha256": context["original_prompt_sha256"], "conclusion_prompt_sha256": context["prompt_sha256"],
                      "all_incoming_stances_explicit_and_consistent": context["all_incoming_stances_explicit_and_consistent"],
                      "empty_inbox": context["empty_inbox"]}}
        rows.append(row)
    assert len(rows) == len({r["context_id"] for r in rows}) == 512
    out = ROOT / "teacher_pairs"
    out.mkdir(exist_ok=True)
    io.write_json(out / "static_context.json", static)
    io.write_jsonl(out / "examples.jsonl", rows)
    io.write_json(out / "manifest.json", {"status": "complete", "paired_contexts": len(rows), "source_response_conditions": 2,
        "question_groups": sorted({r["question_group"] for r in rows}), "graph_orientation": "J[sender,receiver]",
        "forecast_boundary": "After messages arrive, before the receiver's next opinion. Current and previous spins are from step-1 and step-2; step labels are retained only under outcomes.",
        "target_distinction": "First-token probabilities and sampled majority-of-five signs are separate supervision choices. The probabilities are not majority probabilities.",
        "split": "Exploratory only. No new training, validation or test split is declared. Never split paired variants or agents from an episode into different folds; a generalization study should split new questions before generation/fitting.",
        "provenance": {str(p): io.sha(p) for p in (base / "analysis/audit.json", alt / "analysis/audit.json", Path(__file__))},
        "artifacts": {name: io.sha(out / name) for name in ("static_context.json", "examples.jsonl")}})
    print("Exported 512 paired contexts with inputs and outcomes separated; no model fitted.")


if __name__ == "__main__":
    main()
