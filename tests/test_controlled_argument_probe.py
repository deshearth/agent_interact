import json
import pytest

from scripts import prepare_controlled_argument_probe as original
from scripts import prepare_controlled_argument_probe_uncertain as uncertain
from scripts.argument_features import parse_stance


@pytest.mark.parametrize("study", [original, uncertain])
def test_only_target_argument_changes_and_all_stances_agree(study):
    rows = [json.loads(line) for line in (study.ROOT/"prompts.jsonl").read_text().splitlines()]
    manifest = json.loads((study.ROOT/"manifest.json").read_text())
    assert manifest["api_calls"] == 0 and manifest["status"] == "draft_not_executed"
    assert len(rows) == 72 and len({r["case_id"] for r in rows}) == 24
    for case_id in {r["case_id"] for r in rows}:
        variants = [r for r in rows if r["case_id"] == case_id]
        assert {r["condition"] for r in variants} == set(study.CONDITIONS)
        prompts = []
        for row in variants:
            assert parse_stance(row["argument"]) == row["message_stance"]
            assert row["prompt"].count(row["argument"]) == 1
            prompts.append(row["prompt"].replace(row["argument"], "<TARGET ARGUMENT>"))
        assert prompts[0] == prompts[1] == prompts[2]
    counts = {}
    for r in rows:
        key = (r["qid"], r["relationship_sign"], r["condition"])
        counts[key] = counts.get(key, 0)+1
    assert set(counts.values()) == {3}


@pytest.mark.parametrize("study", [original, uncertain])
def test_authored_math_is_checked_and_triangle_coordinates_satisfy_both_areas(study):
    study.verify_arithmetic()
    b, c, d, e = (0,0), (223,0), (680,380), (689,389)
    def twice_area(a,b,c):
        return abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    solutions = [(300+y+sign*1556,y) for y in (-18,18) for sign in (-1,1)]
    assert all(twice_area(a,b,c) == 4014 and twice_area(a,d,e) == 14004 for a in solutions)
    assert sum(x for x,y in solutions) == 1200


def test_refined_cases_are_unique_and_uncertain_by_frozen_baseline():
    cases = json.loads((uncertain.ROOT/"cases.json").read_text())
    assert len({(c["episode_id"],c["round"],c["receiver"]) for c in cases}) == 24
    assert all(.4 <= c["selection_no_text_p_A"] <= .6 for c in cases)
