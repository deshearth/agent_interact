"""Audit and summarize the local-source intervention and new conversations."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts import local_conversation_pilot as conversation
from scripts.local_argument_source import ROOT, sample_seed


def verify_collection(root):
    status = json.loads((root / "collection_status.json").read_text())
    for name, digest in status["artifacts"].items():
        assert io.sha(root / name) == digest, name
    assert status["status"] == "complete", status["status"]
    return status


def intervention():
    root = ROOT / "intervention"
    status = verify_collection(root)
    prompts, plan, responses = [io.read_jsonl(root / p) for p in ("prompts.jsonl", "plan.jsonl", "responses.jsonl")]
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["source_code_sha256"] == io.sha("scripts/local_argument_source.py")
    assert protocol["collector_code_sha256"] == io.sha("scripts/local_argument_intervention.py")
    planned = {r["sample_id"]: r for r in plan}
    assert len(planned) == len(plan) == len(responses) == len({r["sample_id"] for r in responses})
    by_prompt = defaultdict(list)
    for r in responses:
        assert all(r[k] == v for k, v in planned[r["sample_id"]].items())
        assert r["sample_seed"] == sample_seed(r["sample_id"])
        parsed, strict = io.parse_answer(r["raw_response"])
        if r["finish_reason"] == "length":
            parsed, strict = None, False
        assert (parsed, strict) == (r["parsed_sign"], r["strict_format"])
        by_prompt[r["prompt_id"]].append(r)
    groups = io.majority_outcomes(plan, responses)
    grouped = defaultdict(list)
    for r in groups:
        grouped[r["prompt_id"]].append(r)
    rows = []
    for prompt in prompts:
        observed = by_prompt[prompt["prompt_id"]]
        assert len(observed) == protocol["repeats"]
        assert io.text_sha(prompt["prompt"]) == prompt["prompt_sha256"]
        pkeys = ("first_literal_A_probability", "first_literal_B_probability", "other_first_token_probability")
        assert all(max(r[k] for r in observed) - min(r[k] for r in observed) < 1e-6 for k in pkeys)
        row = {k: prompt[k] for k in ("prompt_id", "case_id", "qid", "condition", "relationship_sign", "message_stance")}
        row.update({k: observed[0][k] for k in pkeys})
        row["first_literal_stance_probability"] = observed[0][pkeys[0] if prompt["message_stance"] == 1 else pkeys[1]]
        row["parse_failures"] = sum(r["parsed_sign"] is None for r in observed)
        row["strict_format_failures"] = sum(not r["strict_format"] for r in observed)
        for label, values in (("raw", [r["parsed_sign"] for r in observed]),
                              ("majority", [r["majority_sign"] for r in grouped[prompt["prompt_id"]]])):
            valid = [v for v in values if v in (-1, 1)]
            success = sum(v == prompt["message_stance"] for v in valid)
            row[label + "_n"] = len(valid)
            row[label + "_stance_count"] = success
            row[label + "_stance_fraction"] = success / len(valid) if valid else None
            row[label + "_wilson95"] = io.wilson(success, len(valid))
        rows.append(row)
    by_condition = {(r["case_id"], r["condition"]): r for r in rows}
    contrasts = []
    for case in sorted({r["case_id"] for r in rows}):
        for a, b in io.CONTRASTS:
            left, right = by_condition[case, a], by_condition[case, b]
            contrasts.append({"case_id": case, "qid": left["qid"], "relationship_sign": left["relationship_sign"],
                              "contrast": a + "_minus_" + b,
                              **{k: left[k] - right[k] for k in ("first_literal_stance_probability", "raw_stance_fraction", "majority_stance_fraction")}})
    frame = pd.DataFrame(contrasts)
    summaries = []
    for contrast, block in frame.groupby("contrast"):
        for sign in (0, 1, -1):
            subset = block if sign == 0 else block[block.relationship_sign == sign]
            summaries.append({"contrast": contrast, "relationship_sign": sign, "contexts": len(subset),
                              "first_literal_mean_signed_change": float(subset.first_literal_stance_probability.mean()),
                              "first_literal_mean_absolute_change": float(subset.first_literal_stance_probability.abs().mean()),
                              "first_literal_min_change": float(subset.first_literal_stance_probability.min()),
                              "first_literal_max_change": float(subset.first_literal_stance_probability.max()),
                              "raw_mean_change": float(subset.raw_stance_fraction.mean()),
                              "majority_mean_change": float(subset.majority_stance_fraction.mean())})
    out = root / "analysis"
    io.write_json(out / "per_prompt.json", rows)
    io.write_json(out / "majority_groups.json", groups)
    frame.to_csv(out / "paired_contexts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    frame.groupby(["contrast", "qid"]).agg(
        first_literal_mean_change=("first_literal_stance_probability", "mean"),
        first_literal_mean_absolute_change=("first_literal_stance_probability", lambda x: x.abs().mean()),
        raw_mean_change=("raw_stance_fraction", "mean"), majority_mean_change=("majority_stance_fraction", "mean")
    ).reset_index().to_csv(out / "by_question.csv", index=False)
    audit = {"status": "complete", "responses": len(responses), "prompts": len(prompts), "contexts": len(rows) // 4,
             "parse_failures": sum(r["parse_failures"] for r in rows), "strict_format_failures": sum(r["strict_format_failures"] for r in rows),
             "complete_majority_groups": sum(r["majority_sign"] is not None for r in groups),
             "max_other_first_token_mass": max(r["other_first_token_probability"] for r in rows),
             "first_token_probabilities_constant_across_repeats": True, "all_planned_identities_and_response_parses_verified": True,
             "raw_output_counts": dict(Counter(r["raw_response"] for r in responses)),
             "probability_metric": "Temperature-.7 first bare A/B token probabilities computed from full-vocabulary logits. Other mass is retained. Separate from empirical parsed full-response or five-vote majority rates.",
             "inference_scope": "Descriptive intervention effects on 24 selected fixed contexts from four reused questions. No population confidence interval or confirmatory test. The valid-minus-invalid manipulation retains the same conclusion; valid-minus-conclusion-only changes content and length, while paraphrase changes wording and sometimes length.",
             "sampling_caution": "Only 20 raw responses and four disjoint majority outcomes per prompt. Per-prompt Wilson intervals are marginal and exploratory; no independence or question generalization is asserted across contexts.",
             "source": status["source_model"], "collection_status_sha256": io.sha(root / "collection_status.json"),
             "analysis_code_sha256": io.sha(__file__), "created_utc": datetime.now(timezone.utc).isoformat()}
    # This check concerns the actual first-token draws, not majority labels.
    # All observed outputs here consist of one literal A/B token followed by EOS.
    if all(r["raw_response"] in ("A", "B") and len(r["generated_token_ids"]) == 1 for r in responses):
        p = np.array([r["first_literal_A_probability"] for r in rows])
        n = np.array([r["raw_n"] for r in rows])
        k = np.array([r["raw_stance_count"] if r["message_stance"] == 1 else r["raw_n"] - r["raw_stance_count"] for r in rows])
        variance = n * p * (1 - p)
        statistic = float(np.sum((k - n * p) ** 2 / variance))
        draws = np.random.default_rng(93177).binomial(n, p, size=(20000, len(p)))
        simulated = np.sum((draws - n * p) ** 2 / variance, axis=1)
        audit["sampling_calibration_check"] = {"observed_A": int(k.sum()), "expected_A_from_logits": float((n * p).sum()),
            "pooled_standardized_residual": float((k - n * p).sum() / np.sqrt(variance.sum())),
            "pearson_statistic": statistic, "parametric_monte_carlo_tail_fraction": float((1 + (simulated >= statistic).sum()) / 20001),
            "simulation_seed": 93177, "simulations": 20000,
            "interpretation": "Diagnostic against the saved first-token sampling probabilities; a non-extreme result does not establish independence or model validity."}
    io.write_json(out / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    return rows, summaries


def conversations():
    root = ROOT / "conversations"
    status = verify_collection(root)
    protocol = json.loads((root / "protocol.json").read_text())
    for name, digest in protocol["source_files"].items():
        assert io.sha(name) == digest, name
    prompts = {r["prompt_sha256"]: r for r in io.read_jsonl(root / "source_prompts.jsonl")}
    records = io.read_jsonl(root / "source_responses.jsonl")
    by_id = {r["sample_id"]: r for r in records}
    assert len(by_id) == len(records)
    for record in records:
        assert record["sample_seed"] == sample_seed(record["sample_id"])
        prompt = prompts[record["prompt_sha256"]]
        assert io.text_sha(prompt["prompt"]) == record["prompt_sha256"]
        assert io.text_sha(json.dumps(prompt["token_ids"])) == prompt["token_ids_sha256"]
    questions = {q["qid"]: q for q in json.loads((root / "questions.json").read_text())}
    fmt, rows = conversation.formatter(), []
    for path in sorted((root / "episodes").glob("*.json")):
        ep = json.loads(path.read_text())
        name, qid, n = ep["meta"]["name"], ep["meta"]["qid"], ep["meta"]["n"]
        question, graph = questions[qid], ep["J"]
        agree, disagree = [[] for _ in range(n)], [[] for _ in range(n)]
        for step, raw in enumerate(ep["spins_raw_history"]):
            if step:
                messages = {}
                for agent in range(n):
                    if not any(graph[agent]):
                        continue
                    identity = f"{name}|round{step}|message|agent{agent}"
                    prompt, stance = conversation.message_prompt(fmt, ep["personas"][agent], question, agree[agent], disagree[agent], ep["spins_history"][step-1][agent], identity)
                    r = by_id[identity]
                    assert r["prompt_sha256"] == io.text_sha(prompt)
                    assert stance == r["communicated_stance"] == ep["communicated_stances_history"][step-1][str(agent)]
                    messages[agent] = r["raw_response"].strip()
                agree, disagree, edges = conversation.deliver(graph, messages, name, step)
                assert edges == ep["messages_history"][step-1]
            for agent, votes in enumerate(raw):
                prompt = conversation.opinion_prompt(fmt, ep["personas"][agent], question, agree[agent], disagree[agent])
                first = None
                for position, vote in enumerate(votes):
                    found = 0
                    for attempt in range(3):
                        identity = f"{name}|round{step}|agent{agent}|vote{position}|attempt{attempt}"
                        r = by_id[identity]
                        assert r["prompt_sha256"] == io.text_sha(prompt)
                        parsed, strict = io.parse_answer(r["raw_response"])
                        if r["finish_reason"] == "length":
                            parsed, strict = None, False
                        assert (parsed, strict) == (r["parsed_sign"], r["strict_format"])
                        probability = r["first_literal_A_probability"]
                        if first is None:
                            first = probability
                        assert abs(first - probability) < 1e-6
                        if parsed in (-1, 1):
                            found = parsed
                            break
                    assert found == vote
                spin = conversation.aggregate(votes)
                assert spin == ep["spins_history"][step][agent]
                previous = ep["spins_history"][step-1][agent] if step else None
                rows.append({"episode": name, "qid": qid, "round": step, "agent": agent, "spin": spin,
                             "previous_spin": previous, "changed": int(spin != previous) if step else None,
                             "unanimous": len(set(votes)) == 1, "first_literal_A_probability": first,
                             "raw_A_fraction": sum(v == 1 for v in votes) / 5,
                             "agree_messages": len(agree[agent]), "disagree_messages": len(disagree[agent]),
                             "prompt_sha256": io.text_sha(prompt)})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "agent_states.csv", index=False)
    frame.groupby(["qid", "round"]).agg(A_fraction=("spin", lambda x: (x == 1).mean()),
        change_fraction=("changed", "mean"), unanimous_fraction=("unanimous", "mean"),
        mean_first_literal_A_probability=("first_literal_A_probability", "mean")
    ).reset_index().to_csv(out / "by_question_round.csv", index=False)
    updated = frame[frame["round"] > 0]
    audit = {"status": "complete", "episodes": frame.episode.nunique(), "agent_states": len(frame),
             "transitions": len(updated), "changed_transitions": int(updated.changed.sum()),
             "change_fraction": float(updated.changed.mean()), "unanimous_update_vote_fraction": float(updated.unanimous.mean()),
             "all_spins_reconstructed_from_raw_attempts": True, "all_original_prompts_reconstructed_exactly": True,
             "signed_delivery_and_synchronous_phase_order_verified": True,
             "all_messages_are_logged_local_source_outputs": True,
             "opinion_response_attempts": status["opinion_response_attempts"], "message_responses": status["message_responses"],
             "opinion_parse_failures": status["opinion_parse_failures"], "truncated_messages": status["truncated_messages"],
             "opinion_strict_format_failures": sum(r["kind"] == "opinion" and not r["strict_format"] for r in records),
             "max_opinion_other_first_token_mass": max(r["other_first_token_probability"] for r in records if r["kind"] == "opinion"),
             "scope": "Fully new local-Llama trajectories on reused development questions; two updates, one graph, one replica. No predictor was fitted and no held-out accuracy is claimed.",
             "collection_status_sha256": io.sha(root / "collection_status.json"), "analysis_code_sha256": io.sha(__file__)}
    io.write_json(out / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    return audit


def ablation():
    from scripts import local_rationale_ablation as study
    root = study.ROOT
    status = verify_collection(root)
    protocol = study.protocol()
    inputs = study.make_inputs()
    original = {r["sample_id"]: r for r in io.read_jsonl(study.PARENT / "source_responses.jsonl")}
    responses = io.read_jsonl(root / "source_responses.jsonl")
    by_id = {r["sample_id"]: r for r in responses}
    assert len(by_id) == len(responses)
    outcomes = {r["context_id"]: r for r in json.loads((root / "outcomes.json").read_text())}
    source_prompts = {r["prompt_sha256"]: r for r in io.read_jsonl(root / "source_prompts.jsonl")}
    rows, seen = [], set()
    for row in inputs:
        base_id = row["original_first_response_id"].removesuffix("|vote0|attempt0")
        first_old = original[row["original_first_response_id"]]
        first_new = by_id[row["context_id"] + "|vote0|attempt0"]
        prompt = source_prompts[row["prompt_sha256"]]
        assert prompt["prompt"] == row["prompt"]
        assert io.text_sha(json.dumps(prompt["token_ids"])) == prompt["token_ids_sha256"]
        values = {}
        for label, records, identity, digest in (("full", original, base_id, row["original_prompt_sha256"]),
                                               ("conclusion", by_id, row["context_id"], row["prompt_sha256"])):
            votes, p = [], None
            for position in range(5):
                sign = 0
                for attempt in range(3):
                    sid = f"{identity}|vote{position}|attempt{attempt}"
                    r = records[sid]
                    assert r["prompt_sha256"] == digest and r["sample_seed"] == sample_seed(sid)
                    if label == "conclusion":
                        seen.add(sid)
                    parsed, strict = io.parse_answer(r["raw_response"])
                    if r["finish_reason"] == "length":
                        parsed, strict = None, False
                    assert (parsed, strict) == (r["parsed_sign"], r["strict_format"])
                    if p is None:
                        p = r["first_literal_A_probability"]
                    assert abs(r["first_literal_A_probability"] - p) < 1e-6
                    if parsed in (-1, 1):
                        sign = parsed
                        break
                votes.append(sign)
            if label == "conclusion":
                assert votes == outcomes[row["context_id"]]["votes"]
                assert conversation.aggregate(votes) == outcomes[row["context_id"]]["majority_sign"]
            values.update({label + "_p_first_A": p, label + "_p_first_B": records[identity + "|vote0|attempt0"]["first_literal_B_probability"],
                           label + "_raw_A_fraction": sum(v == 1 for v in votes) / 5,
                           label + "_majority": conversation.aggregate(votes), label + "_valid_votes": sum(v in (-1, 1) for v in votes)})
        if row["empty_inbox"]:
            assert row["prompt_sha256"] == row["original_prompt_sha256"]
            assert abs(values["full_p_first_A"] - values["conclusion_p_first_A"]) < 1e-6
        rows.append({k: row[k] for k in ("context_id", "episode", "qid", "round", "agent", "empty_inbox", "all_incoming_stances_explicit_and_consistent")} |
                    values | {"first_A_change_full_minus_conclusion": values["full_p_first_A"] - values["conclusion_p_first_A"],
                              "first_A_dominant_token_flip": (values["full_p_first_A"] >= values["full_p_first_B"]) != (values["conclusion_p_first_A"] >= values["conclusion_p_first_B"]),
                              "raw_A_fraction_change": values["full_raw_A_fraction"] - values["conclusion_raw_A_fraction"],
                              "sampled_majority_flip": values["full_majority"] != values["conclusion_majority"],
                              "max_other_first_token_mass": max(first_old["other_first_token_probability"], first_new["other_first_token_probability"])})
    assert seen == set(by_id)
    frame = pd.DataFrame(rows)
    subsets = {"all": frame, "nonempty": frame[~frame.empty_inbox],
               "explicit_consistent_stances_nonempty": frame[(~frame.empty_inbox) & frame.all_incoming_stances_explicit_and_consistent],
               "empty_inbox_negative_controls": frame[frame.empty_inbox]}
    summaries = []
    for name, data in subsets.items():
        summaries.append({"subset": name, "contexts": len(data),
            "mean_absolute_first_A_change": float(data.first_A_change_full_minus_conclusion.abs().mean()),
            "mean_signed_first_A_change": float(data.first_A_change_full_minus_conclusion.mean()),
            "median_absolute_first_A_change": float(data.first_A_change_full_minus_conclusion.abs().median()),
            "max_absolute_first_A_change": float(data.first_A_change_full_minus_conclusion.abs().max()),
            "first_A_dominant_token_flips": int(data.first_A_dominant_token_flip.sum()),
            "sampled_majority_flips": int(data.sampled_majority_flip.sum()),
            "mean_raw_A_fraction_change": float(data.raw_A_fraction_change.mean()),
            "balanced_pair_state_only_probability_mse_lower_bound": float((data.first_A_change_full_minus_conclusion ** 2).mean() / 4),
            "max_other_first_token_mass": float(data.max_other_first_token_mass.max())})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "paired_contexts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    frame.groupby("qid").agg(contexts=("context_id", "count"),
        mean_absolute_first_A_change=("first_A_change_full_minus_conclusion", lambda x: x.abs().mean()),
        mean_signed_first_A_change=("first_A_change_full_minus_conclusion", "mean"),
        first_A_dominant_token_flips=("first_A_dominant_token_flip", "sum"),
        sampled_majority_flips=("sampled_majority_flip", "sum")
    ).reset_index().to_csv(out / "by_question.csv", index=False)
    audit = {"status": "complete", "contexts": len(frame), "response_attempts": len(responses),
             "parse_failures": status["parse_failures"], "empty_inbox_controls": int(frame.empty_inbox.sum()),
             "empty_inbox_probability_negative_controls_passed": True,
             "all_sample_identities_and_raw_votes_reconstructed": True,
             "interpretation": protocol["scope"], "sampling_caution": "A sampled-majority flip includes ordinary Monte Carlo variability: each condition has only one independent five-vote group. Dominant first-token flips compare the computed distributions and avoid that sampling noise. Neither is predictor accuracy.",
             "protocol_sha256": io.sha(root / "protocol.json"), "analysis_code_sha256": io.sha(__file__)}
    io.write_json(out / "audit.json", audit)
    print(json.dumps({"audit": audit, "summaries": summaries}, indent=2))
    return audit, summaries


def grading():
    from scripts import local_argument_grading as study
    root = study.ROOT
    status = verify_collection(root)
    inputs = study.prepare()
    responses = io.read_jsonl(root / "source_responses.jsonl")
    by_id = {r["sample_id"]: r for r in responses}
    assert len(by_id) == len(responses) == 1440
    rows = []
    for row in inputs:
        observed = [by_id[row["context_id"] + f"|draw{i}"] for i in range(20)]
        for r in observed:
            assert r["sample_seed"] == sample_seed(r["sample_id"]) and r["prompt_sha256"] == row["prompt_sha256"]
            parsed, strict = io.parse_answer(r["raw_response"])
            if r["finish_reason"] == "length":
                parsed, strict = None, False
            assert (parsed, strict) == (r["parsed_sign"], r["strict_format"])
            for label in ("A", "B"):
                assert abs(r[f"first_literal_{label}_probability"] - observed[0][f"first_literal_{label}_probability"]) < 1e-6
        correct_key = "first_literal_" + row["expected_label"] + "_probability"
        incorrect_key = "first_literal_" + ("A" if row["expected_label"] == "B" else "B") + "_probability"
        rows.append({k: row[k] for k in ("context_id", "qid", "persona", "condition", "valid_label", "expected_label")} |
                    {"p_correct_first_literal": observed[0][correct_key],
                     "p_valid_first_literal": observed[0]["first_literal_" + row["valid_label"] + "_probability"],
                     "literal_pair_choice_correct": row["expected_label"] == ("A" if observed[0]["first_literal_A_probability"] >= observed[0]["first_literal_B_probability"] else "B"),
                     "raw_correct_fraction": sum(r["parsed_sign"] == row["expected_sign"] for r in observed) / 20,
                     "parse_failures": sum(r["parsed_sign"] is None for r in observed),
                     "max_other_first_token_mass": max(r["other_first_token_probability"] for r in observed)})
    frame = pd.DataFrame(rows)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_prompt.csv", index=False)
    summary = frame.groupby("condition").agg(prompts=("context_id", "count"), p_correct_first_literal=("p_correct_first_literal", "mean"),
        p_valid_first_literal=("p_valid_first_literal", "mean"), raw_correct_fraction=("raw_correct_fraction", "mean"),
        literal_pair_choice_correct=("literal_pair_choice_correct", "mean"), parse_failures=("parse_failures", "sum"))
    summary.to_csv(out / "summary.csv")
    frame.groupby(["qid", "condition"]).agg(p_correct_first_literal=("p_correct_first_literal", "mean"),
        raw_correct_fraction=("raw_correct_fraction", "mean"), literal_pair_choice_correct=("literal_pair_choice_correct", "mean")
    ).reset_index().to_csv(out / "by_question.csv", index=False)
    label_swap = frame.pivot(index=["qid", "persona", "condition"], columns="valid_label", values="p_valid_first_literal")
    label_swap["absolute_probability_change"] = (label_swap.A - label_swap.B).abs()
    label_swap.to_csv(out / "label_swap.csv")
    primary = frame[frame.condition.isin(("valid", "invalid"))]
    audit = {"status": "complete", "prompts": len(frame), "responses": len(responses),
        "parse_failures": status["parse_failures"], "strict_format_failures": status["strict_format_failures"],
        "max_other_first_token_mass": float(frame.max_other_first_token_mass.max()),
        "primary_valid_flawed_prompts": len(primary), "primary_raw_grading_accuracy": float(primary.raw_correct_fraction.mean()),
        "primary_mean_correct_first_literal_probability": float(primary.p_correct_first_literal.mean()),
        "primary_literal_pair_choice_accuracy": float(primary.literal_pair_choice_correct.mean()),
        "mean_absolute_label_swap_probability_change": float(label_swap.absolute_probability_change.mean()),
        "primary_baseline": "50% for balanced valid versus flawed conditions. The valid paraphrase is secondary; always saying valid would score 2/3 over all three conditions, so the all-condition aggregate is not the primary accuracy.",
        "scope": "Direct validity-grading diagnostic on four reused questions and three fixed personas, with both grading-label assignments. Not an interaction update, mathematical-answer accuracy, predictor accuracy, or broad proof-verification benchmark.",
        "analysis_code_sha256": io.sha(__file__), "collection_status_sha256": io.sha(root / "collection_status.json")}
    io.write_json(out / "audit.json", audit)
    print(json.dumps({"audit": audit, "summary": summary.reset_index().to_dict("records")}, indent=2))
    return audit


def word_grading():
    from scripts import local_word_grading as study
    root = study.ROOT
    status = verify_collection(root)
    inputs = study.prepare()
    responses = io.read_jsonl(root / "source_responses.jsonl")
    by_id = {r["sample_id"]: r for r in responses}
    assert len(by_id) == len(responses) == 1440
    rows, bad = [], Counter()
    for row in inputs:
        observed = [by_id[row["context_id"] + f"|draw{i}"] for i in range(20)]
        verdicts = []
        for r in observed:
            assert r["sample_seed"] == sample_seed(r["sample_id"]) and r["prompt_sha256"] == row["prompt_sha256"]
            verdict, strict = study.parse_verdict(r["raw_response"]) if r["finish_reason"] == "stop" else (None, False)
            if verdict is None:
                bad[r["raw_response"]] += 1
            verdicts.append(verdict)
        rows.append({k: row[k] for k in ("context_id", "qid", "persona", "condition", "definition_order", "expected_verdict")} |
                    {"raw_correct_fraction": sum(v == row["expected_verdict"] for v in verdicts) / 20,
                     "raw_valid_verdict_fraction": sum(v == "VALID" for v in verdicts) / 20,
                     "verdict_parse_failures": sum(v is None for v in verdicts),
                     "valid_parse_correct_fraction": (sum(v == row["expected_verdict"] for v in verdicts) / sum(v is not None for v in verdicts)) if any(v is not None for v in verdicts) else None})
    frame = pd.DataFrame(rows)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_prompt.csv", index=False)
    summary = frame.groupby("condition").agg(prompts=("context_id", "count"), raw_correct_fraction=("raw_correct_fraction", "mean"),
        raw_valid_verdict_fraction=("raw_valid_verdict_fraction", "mean"), verdict_parse_failures=("verdict_parse_failures", "sum"))
    summary.to_csv(out / "summary.csv")
    frame.groupby(["qid", "condition"]).agg(raw_correct_fraction=("raw_correct_fraction", "mean"),
        raw_valid_verdict_fraction=("raw_valid_verdict_fraction", "mean"), verdict_parse_failures=("verdict_parse_failures", "sum")
    ).reset_index().to_csv(out / "by_question.csv", index=False)
    order = frame.pivot(index=["qid", "persona", "condition"], columns="definition_order", values="raw_valid_verdict_fraction")
    order["absolute_sampled_fraction_change"] = (order.VALID_then_FLAWED - order.FLAWED_then_VALID).abs()
    order.to_csv(out / "definition_order.csv")
    primary = frame[frame.condition.isin(("valid", "invalid"))]
    audit = {"status": "complete", "prompts": len(frame), "responses": len(responses),
        "verdict_parse_failures": status["verdict_parse_failures"], "verdict_strict_format_failures": status["verdict_strict_format_failures"],
        "primary_raw_grading_accuracy": float(primary.raw_correct_fraction.mean()),
        "primary_mean_valid_parse_grading_accuracy": float(primary.valid_parse_correct_fraction.mean()) if primary.valid_parse_correct_fraction.notna().any() else None,
        "mean_absolute_definition_order_sampled_fraction_change": float(order.absolute_sampled_fraction_change.mean()),
        "unparsed_responses": dict(bad), "first_literal_A_B_diagnostics_used_for_word_verdicts": False,
        "scope": "Adaptive response-format follow-up after the A/B grader's label sensitivity was observed. Same four problems; no independent generalization claim. Score full word verdicts, not the shared logger's A/B fields.",
        "comparison_caution": "Definition-order sensitivity uses 20-sample fractions and includes sampling noise. It differs from the previous experiment's exact first-token label-swap contrast.",
        "analysis_code_sha256": io.sha(__file__), "collection_status_sha256": io.sha(root / "collection_status.json")}
    io.write_json(out / "audit.json", audit)
    print(json.dumps({"audit": audit, "summary": summary.reset_index().to_dict("records")}, indent=2))
    return audit


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("intervention", "conversations", "ablation", "grading", "word_grading"))
    args = parser.parse_args()
    {"intervention": intervention, "conversations": conversations, "ablation": ablation, "grading": grading, "word_grading": word_grading}[args.stage]()
