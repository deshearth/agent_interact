import json
import time
from types import SimpleNamespace

import pytest

from scripts import argument_intervention_pilot as io
from scripts import local_argument_intervention as intervention
from scripts import local_conversation_pilot as conversation
from scripts import local_rationale_ablation as ablation
from scripts import local_argument_grading as grading
from scripts import local_word_grading as word_grading
from scripts.local_argument_source import sample_seed


def test_local_schedule_independent_seeds_and_five_vote_groups():
    prompts = io.read_jsonl(io.ROOT / "prompts.jsonl")
    plan = intervention.make_plan(prompts, 20)
    assert len(plan) == len({r["sample_id"] for r in plan}) == 1920
    assert len({sample_seed(r["sample_id"]) for r in plan}) == 1920
    assert len(io.majority_outcomes(plan, [])) == 384
    assert plan == intervention.make_plan(prompts, 20)
    with pytest.raises(ValueError):
        intervention.make_plan(prompts, 7)


def test_signed_inboxes_and_single_sender_broadcast():
    graph = [[0, 1, -1], [-1, 0, 1], [1, 0, 0]]
    agree, disagree, edges = conversation.deliver(graph, {0: "zero", 1: "one", 2: "two"}, "ep", 1)
    assert agree == [["two"], ["zero"], ["one"]]
    assert disagree == [["one"], [], ["zero"]]
    assert edges["0,1"] == edges["0,2"] == "zero"
    assert len(edges) == 5
    assert conversation.deliver(graph, {0: "zero", 1: "one", 2: "two"}, "ep", 1) == (agree, disagree, edges)


def test_parse_retries_are_logged_per_vote_and_never_reassign_positions():
    calls = []

    class Failed:
        def request(self, prompt, identity, metadata):
            calls.append((identity, metadata))
            if metadata["vote_position"] == 0:
                return {"parsed_sign": None}
            return {"parsed_sign": 1 if metadata["parse_attempt"] else None}

    votes = conversation.vote_group(Failed(), "prompt", "case", {})
    assert votes == [0, 1, 1, 1, 1]
    assert len(calls) == 11
    assert len({x[0] for x in calls}) == 11
    assert [m["parse_attempt"] for _, m in calls if m["vote_position"] == 0] == [0, 1, 2]
    assert conversation.aggregate(votes) == 1
    assert conversation.aggregate([-1, 1, 0, 0, 0]) == -1
    assert conversation.aggregate([0, 0, 0, 0, 0]) == 0


def test_released_formatter_and_parser_are_used():
    fmt = conversation.formatter()
    question = {"question": "2+2?", "choices": {"A": "4", "B": "5"}}
    expected = []
    fmt.spin_sampler("persona", question["question"], ["yes"], ["no"], lambda p: expected.append(p) or "A", mode="objective", choices=question["choices"])
    assert conversation.opinion_prompt(fmt, "persona", question, ["yes"], ["no"]) == expected[0]
    for response in ("A", "b", "I select A.", "B!", "A or B", "uncertain", "AB"):
        try:
            expected_sign = fmt._parse_spin_objective(response)
        except ValueError:
            expected_sign = None
        assert io.parse_answer(response)[0] == expected_sign


def test_complete_rounds_are_synchronous_and_all_messages_are_new_local_outputs(tmp_path):
    io.write_json(tmp_path / "protocol.json", {})
    question = {"qid": "q", "question": "2+2?", "choices": {"A": "4", "B": "5"}}
    graph = [[0, 1, -1], [1, 0, 1], [-1, 1, 0]]
    calls = []

    class FakeRecorder:
        records = {}

        def request(self, prompt, identity, metadata):
            step = metadata["round"]
            if metadata["kind"] == "message":
                assert not any(m["kind"] == "opinion" and m["round"] == step for _, m in calls)
                if step == 1:
                    assert "local_message_" not in prompt
                else:
                    assert "local_message_1_" in prompt and "local_message_2_" not in prompt
                row = {"raw_response": f"local_message_{step}_{metadata['agent']}", "parsed_sign": None}
            else:
                if step:
                    assert sum(m["kind"] == "message" and m["round"] == step for _, m in calls) == 3
                    assert f"local_message_{step}_" in prompt
                    assert f"local_message_{step-1}_" not in prompt
                row = {"raw_response": "A", "parsed_sign": 1}
            calls.append((prompt, metadata))
            self.records[identity] = row
            return row

    recorder = FakeRecorder()
    result = conversation.run_episode(question, ["p0", "p1", "p2"], graph, 2, recorder, tmp_path, conversation.formatter())
    assert len(calls) == 3 * 3 * 5 + 3 * 2
    assert result["spins_history"] == [[1, 1, 1]] * 3
    assert len(result["messages_history"]) == 2
    assert all(len(m) == 6 for m in result["messages_history"])
    count = len(calls)
    assert conversation.run_episode(question, ["p0", "p1", "p2"], graph, 2, recorder, tmp_path, conversation.formatter()) == result
    assert len(calls) == count


def test_recorder_resume_uses_exact_identity_and_prompt(tmp_path):
    class FakeSource:
        mx = SimpleNamespace(clear_cache=lambda: None)
        calls = 0

        def render(self, prompt):
            return [1, 2, 3]

        def prefill(self, tokens):
            return "prefix", 0.

        def sample(self, tokens, identity, **kwargs):
            self.calls += 1
            return {"raw_response": "A", "parsed_sign": 1}

    source = FakeSource()
    recorder = conversation.Recorder(source, tmp_path, time.time() + 10)
    first = recorder.request("same prompt", "identity", {"kind": "opinion"})
    recorder.close()
    second = conversation.Recorder(source, tmp_path, time.time() + 10)
    assert second.request("same prompt", "identity", {"kind": "opinion"}) == first
    assert source.calls == 1
    with pytest.raises(AssertionError):
        second.request("changed prompt", "identity", {"kind": "opinion"})
    second.deadline = 0
    with pytest.raises(conversation.DeadlineReached):
        second.request("same prompt", "new identity", {"kind": "opinion"})
    second.close()
    assert len(io.read_jsonl(tmp_path / "source_responses.jsonl")) == 1


def test_ablation_preserves_signed_context_and_flags_stance_mismatches(tmp_path, monkeypatch):
    parent, root = tmp_path / "conversations", tmp_path / "ablation"
    monkeypatch.setattr(ablation, "PARENT", parent)
    monkeypatch.setattr(ablation, "ROOT", root)
    io.write_json(parent / "protocol.json", {"fixture": True})
    fmt, questions, source = conversation.formatter(), [], []
    graph = [[0] * 32 for _ in range(32)]
    graph[0][1], graph[0][2] = 1, -1
    for i in range(8):
        question = {"qid": f"q{i}", "question": f"Problem {i}", "choices": {"A": "one", "B": "two"}}
        questions.append(question)
        name = f"episode_{i}"
        personas = [f"persona_{a}" for a in range(32)]
        histories = []
        for step in (1, 2):
            # The second round intentionally contradicts the instructed B stance.
            text = "I favor option " + ("B" if step == 1 else "A") + ". Here is my reason."
            agree, disagree, edges = conversation.deliver(graph, {0: text}, name, step)
            histories.append(edges)
            for agent in range(32):
                prompt = conversation.opinion_prompt(fmt, personas[agent], question, agree[agent], disagree[agent])
                source.append({"sample_id": f"{name}|round{step}|agent{agent}|vote0|attempt0", "prompt_sha256": io.text_sha(prompt)})
        io.write_json(parent / "episodes" / f"{name}.json", {"meta": {"name": name, "qid": question["qid"], "n": 32},
                      "personas": personas, "J": graph, "messages_history": histories,
                      "communicated_stances_history": [{"0": -1}, {"0": -1}]})
    io.write_json(parent / "questions.json", questions)
    io.write_jsonl(parent / "source_responses.jsonl", source)
    io.write_json(parent / "collection_status.json", {"status": "complete", "artifacts": {}})
    rows = ablation.make_inputs()
    assert len(rows) == 512
    assert sum(r["empty_inbox"] for r in rows) == 480
    for r in rows:
        if r["empty_inbox"]:
            assert r["prompt_sha256"] == r["original_prompt_sha256"]
        else:
            assert r["incoming_stances"] == [-1]
            assert r["incoming_relationships"] == [1 if r["agent"] == 1 else -1]
            assert r["all_incoming_stances_explicit_and_consistent"] == (r["round"] == 1)
            assert "I favor option B." in r["prompt"] and "Here is my reason." not in r["prompt"]
    assert ablation.make_inputs() == rows


@pytest.mark.parametrize("text,stance", [("I favor option A. Reason.", 1), ("I agree with option B, because...", -1),
    ("I believe option A is correct.", 1), ("I favor A but I prefer B.", None), ("I cannot decide.", None)])
def test_explicit_stance_audit_does_not_infer_missing_or_ambiguous_choices(text, stance):
    assert ablation.declared_stance(text) == stance


def test_validity_grading_balances_labels_without_changing_the_argument(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "ROOT", tmp_path)
    rows = grading.prepare()
    assert len(rows) == 72 and len({r["context_id"] for r in rows}) == 72
    assert sum(r["expected_label"] == "A" for r in rows) == 36
    assert sum(r["expected_label"] == "B" for r in rows) == 36
    for r in rows:
        mate = next(x for x in rows if x["qid"] == r["qid"] and x["persona"] == r["persona"] and x["condition"] == r["condition"] and x["valid_label"] != r["valid_label"])
        assert r["prompt"].split("# Grading Task")[0] == mate["prompt"].split("# Grading Task")[0]
        assert r["expected_sign"] == -mate["expected_sign"]
    assert grading.prepare() == rows


def test_word_verdicts_keep_argument_and_semantics_fixed_when_order_reverses(tmp_path, monkeypatch):
    monkeypatch.setattr(word_grading, "ROOT", tmp_path)
    rows = word_grading.prepare()
    assert len(rows) == 72 and len({r["context_id"] for r in rows}) == 72
    for r in rows:
        mate = next(x for x in rows if x["qid"] == r["qid"] and x["persona"] == r["persona"] and x["condition"] == r["condition"] and x["definition_order"] != r["definition_order"])
        assert r["prompt"].split("Use one of these verdict words:")[0] == mate["prompt"].split("Use one of these verdict words:")[0]
        assert r["expected_verdict"] == mate["expected_verdict"]
    assert word_grading.parse_verdict("VALID") == ("VALID", True)
    assert word_grading.parse_verdict(" flawed. ") == ("FLAWED", False)
    assert word_grading.parse_verdict("INVALID") == (None, False)
    assert word_grading.parse_verdict("Not VALID") == (None, False)
    assert word_grading.parse_verdict("VALID or FLAWED") == (None, False)
