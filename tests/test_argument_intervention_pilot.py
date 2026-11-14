import json
from pathlib import Path

import pytest

from scripts import argument_intervention_pilot as pilot
from scripts import run_argument_intervention as runner


@pytest.fixture
def prepared(tmp_path):
    pilot.prepare(tmp_path)
    return tmp_path


def fake_response(text="A"):
    sign, strict = pilot.parse_answer(text)
    return {"event": "completed", "response_model": pilot.MODEL, "raw_response": text,
            "parsed_sign": sign, "strict_format": strict, "finish_reason": "stop",
            "usage_cost_usd": .00001}


def test_single_message_controls_preserve_entire_remaining_prompt_and_parent():
    old_hash = pilot.sha(pilot.PARENT / "manifest.json")
    cases, variants, rows = pilot.make_inputs()
    assert len(rows) == 96 and len(cases) == 24
    for case in cases:
        selected = [r for r in rows if r["case_id"] == case["case_id"]]
        assert {r["condition"] for r in selected} == set(pilot.CONDITIONS)
        assert len({r["prompt"].replace(r["argument"], "<TARGET>") for r in selected}) == 1
        assert all(pilot.parse_answer(r["argument"])[0] == case["message_stance"] for r in selected)
        group = "inbox_agree" if case["relationship_sign"] == 1 else "inbox_disagree"
        for row in selected:
            assert row[group][case["target_position_within_group"]] == row["argument"]
    assert old_hash == pilot.sha(pilot.PARENT / "manifest.json")


def test_fixed_groups_and_balanced_interleaving(prepared):
    _, prompts, schedule = pilot.verify_prepared(prepared)
    assert len(schedule) == 1920
    assert len({r["sample_id"] for r in schedule}) == len(schedule)
    for i in range(20):
        assert {r["prompt_id"] for r in schedule if r["sample_index"] == i} == {r["prompt_id"] for r in prompts}
    groups = pilot.majority_outcomes(schedule, [])
    assert len(groups) == 384 and all(r["majority_sign"] is None for r in groups)


def test_identical_texts_in_distinct_contexts_remain_distinct_samples():
    rows = [{"prompt_id": "case-a", "prompt_sha256": "same"}, {"prompt_id": "case-b", "prompt_sha256": "same"}]
    plan = pilot.make_schedule(rows, repeats=5)
    assert len({r["sample_id"] for r in plan}) == 10


def test_incomplete_group_is_not_repaired_with_votes_from_another_group():
    plan = pilot.make_schedule([{"prompt_id": "one", "prompt_sha256": "digest"}], repeats=10)
    observations = [{"sample_id": r["sample_id"], "parsed_sign": None if r["sample_index"] == 4 else 1} for r in plan]
    groups = pilot.majority_outcomes(plan, observations)
    assert groups[0]["valid_votes"] == 4 and groups[0]["majority_sign"] is None
    assert groups[1]["valid_votes"] == 5 and groups[1]["majority_sign"] == 1


def test_raw_proportion_is_not_majority_probability(prepared):
    _, prompts, schedule = pilot.verify_prepared(prepared)
    target = prompts[0]
    ledger = []
    for r in schedule:
        if r["prompt_id"] == target["prompt_id"]:
            agrees = r["vote_position"] < 3
            sign = target["message_stance"] if agrees else -target["message_stance"]
            ledger.append({**r, **fake_response("A" if sign == 1 else "B")})
    pilot.write_jsonl(prepared / "source_ledger.jsonl", ledger)
    summary = pilot.analyze(prepared)
    rows = json.loads((prepared / "analysis/per_prompt.json").read_text())
    measured = next(r for r in rows if r["prompt_id"] == target["prompt_id"])
    assert measured["raw_p_message_stance"] == .6
    assert measured["majority_p_message_stance"] == 1.
    assert measured["raw_n"] == 20 and measured["majority_n"] == 4
    assert summary["status"] == "partial"
    assert not summary["counterfactual_source_effect_measured"]


def test_no_outcomes_produce_no_effects_or_accuracy(prepared):
    summary = pilot.analyze(prepared)
    assert summary["status"] == "no_source_results"
    assert summary["returned_responses"] == summary["valid_majority_groups"] == 0
    assert not summary["counterfactual_source_effect_measured"]
    rows = json.loads((prepared / "analysis/paired_contexts.json").read_text())
    assert all(r["difference"] is None for r in rows)


def test_budget_stops_before_dispatch_and_persists_across_resume(prepared):
    calls = []
    def send(prompt):
        calls.append(prompt)
        return fake_response()
    _, prompts, schedule = pilot.verify_prepared(prepared)
    lookup = {r["prompt_id"]: r for r in prompts}
    first_cost = pilot.reserve_cost(lookup[schedule[0]["prompt_id"]]["prompt"])
    zero = runner.execute(prepared, send, max_cost_usd=first_cost / 2, workers=1)
    assert zero["requests_this_invocation"] == 0 and not calls
    one = runner.execute(prepared, send, max_requests=1, workers=1)
    assert one["requests_this_invocation"] == 1 and len(calls) == 1
    stopped = runner.execute(prepared, send, max_cost_usd=first_cost, workers=1)
    assert stopped["requests_this_invocation"] == 0 and len(calls) == 1
    two = runner.execute(prepared, send, max_requests=1, workers=1)
    assert two["attempts_total"] == 2 and len(calls) == 2
    starts = [r for r in pilot.read_jsonl(prepared / "source_ledger.jsonl") if r["event"] == "started"]
    assert len({r["sample_id"] for r in starts}) == 2


def test_interrupted_request_is_not_automatically_retried(prepared):
    _, _, schedule = pilot.verify_prepared(prepared)
    runner.append_event(prepared / "source_ledger.jsonl", {**schedule[0], "event": "started", "reserved_cost_usd": .001})
    def forbidden(_):
        raise AssertionError("Must not dispatch")
    with pytest.raises(RuntimeError, match="reconciliation"):
        runner.execute(prepared, forbidden)


def test_transport_failure_is_retained_without_retry(prepared):
    calls = []
    def fail(_):
        calls.append(1)
        raise TimeoutError()
    result = runner.execute(prepared, fail, workers=1)
    assert len(calls) == 1 and result["attempts_total"] == 1
    assert result["stop_reason"] == "source_error"
    events = pilot.read_jsonl(prepared / "source_ledger.jsonl")
    assert events[-1]["event"] == "failed" and events[-1]["billing_uncertain"]
    assert result["reserved_cost_usd"] > 0


def test_model_mismatch_stops_and_is_not_analyzed_as_matching_source(prepared):
    def wrong(_):
        return {**fake_response(), "response_model": "a-different-model"}
    result = runner.execute(prepared, wrong, workers=1)
    assert result["stop_reason"] == "unexpected_model" and result["attempts_total"] == 1
    with pytest.raises(ValueError, match="Unexpected source model"):
        pilot.analyze(prepared)


def test_protocol_changes_are_rejected_before_execution(prepared):
    path = prepared / "prompts.jsonl"
    path.write_text(path.read_text().replace("snap judgement", "careful computation", 1))
    with pytest.raises(ValueError, match="artifact changed"):
        runner.preflight(prepared)


@pytest.mark.parametrize("raw,sign,strict", [("A", 1, True), (" B\n", -1, True),
                                           ("I choose B.", -1, False), ("A or B", None, False),
                                           ("", None, False)])
def test_parser_retains_ambiguity_and_format_failures(raw, sign, strict):
    assert pilot.parse_answer(raw) == (sign, strict)
