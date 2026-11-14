"""Prepare and analyze a bounded argument intervention pilot; no API calls here.

The four-condition pilot extends, rather than modifies, the frozen three-condition
draft. Source responses must be collected separately. Empty results stay empty.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import NormalDist
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("runs/argument_intervention_v1")
PARENT = Path("runs/argument_investigation_20261108/controlled_probe_uncertain")
CONDITIONS = ("valid", "invalid", "valid_paraphrase", "conclusion_only")
CONTRASTS = (("valid", "invalid"), ("valid", "conclusion_only"), ("valid_paraphrase", "valid"))
MODEL = "gpt-4o-mini-2024-07-18"
INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60
MAX_OUTPUT_TOKENS = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def reserve_cost(prompt):
    # Conservative for byte-based text tokenization, with a separate allowance
    # for one user message's framing. Actual API usage must still be checked.
    upper_input = len(prompt.encode("utf-8")) + 64
    return (upper_input * INPUT_USD_PER_MILLION + MAX_OUTPUT_TOKENS * OUTPUT_USD_PER_MILLION) / 1e6


def parse_answer(text):
    """Match the released objective parser, retaining strict-format status too."""
    text = text.strip().upper()
    tokens = set(re.findall(r"\b([AB])\b", text))
    sign = (1 if tokens == {"A"} else -1 if tokens == {"B"} else None)
    return sign, text in ("A", "B")


def make_inputs(parent=PARENT):
    parent = Path(parent)
    saved = json.loads((parent / "manifest.json").read_text())
    for file, checksum in saved["artifacts"].items():
        if sha(parent / file) != checksum:
            raise ValueError(f"Frozen parent changed: {file}")
    cases = json.loads((parent / "cases.json").read_text())
    variants = json.loads((parent / "argument_variants.json").read_text())
    old_rows = read_jsonl(parent / "prompts.jsonl")
    by_case = defaultdict(dict)
    for row in old_rows:
        by_case[row["case_id"]][row["condition"]] = row
    rows = []
    for case in cases:
        original = by_case[case["case_id"]]
        if set(original) != set(CONDITIONS[:3]):
            raise ValueError("Incomplete parent conditions")
        letter = "A" if case["message_stance"] == 1 else "B"
        sentence = f"I favor option {letter}."
        variants[case["qid"]]["conclusion_only"] = sentence
        valid = original["valid"]
        if valid["prompt"].count(valid["argument"]) != 1:
            raise ValueError("Target argument is not uniquely identified")
        control = {**valid, "condition": "conclusion_only", "argument": sentence,
                   "prompt": valid["prompt"].replace(valid["argument"], sentence)}
        for group in ("inbox_agree", "inbox_disagree"):
            control[group] = list(valid[group])
        group = "inbox_agree" if case["relationship_sign"] == 1 else "inbox_disagree"
        control[group][case["target_position_within_group"]] = sentence
        originals = [original[c] for c in CONDITIONS[:3]] + [control]
        normalized = []
        for row in originals:
            normalized.append(row["prompt"].replace(row["argument"], "<TARGET ARGUMENT>"))
            sign, _ = parse_answer(row["argument"])
            if sign != case["message_stance"]:
                raise ValueError("Intervention changed stance")
            rows.append({**row, "prompt_id": case["case_id"] + "/" + row["condition"],
                         "prompt_sha256": text_sha(row["prompt"])})
        if len(set(normalized)) != 1:
            raise ValueError("A nontarget part of the prompt changed")
    return cases, variants, rows


def make_schedule(rows, repeats=20, seed=804211):
    if repeats <= 0 or repeats % 5:
        raise ValueError("Repeat count must be a positive multiple of five")
    rng = random.Random(seed)
    schedule = []
    for sample in range(repeats):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            row = rows[i]
            identity = f"{row['prompt_id']}|{row['prompt_sha256']}|{sample}"
            schedule.append({"sample_id": text_sha(identity), "prompt_id": row["prompt_id"],
                             "prompt_sha256": row["prompt_sha256"], "sample_index": sample,
                             "majority_group": sample // 5, "vote_position": sample % 5,
                             "schedule_position": len(schedule)})
    return schedule


def planning_precision(cases=24, repeats=20):
    z = NormalDist().inv_cdf(.975)
    power_z = NormalDist().inv_cdf(.8)
    rows = []
    for label, ncases in (("one_context", 1), ("one_relationship_sign", cases // 2), ("all_contexts", cases)):
        for target, n in (("raw_response", ncases * repeats), ("majority_of_five", ncases * repeats // 5)):
            worst_se = math.sqrt(.5 / n)
            rows.append({"scope": label, "target": target, "outcomes_per_condition": n,
                         "worst_case_95_half_width_pp": 100 * z * worst_se,
                         "approximate_80_percent_power_effect_pp": 100 * (z + power_z) * worst_se})
    return {"assumptions": "Planning approximation: independent Bernoulli outcomes, equal allocation, two-sided alpha=.05, no multiplicity adjustment. Fixed selected contexts only; not uncertainty over new questions. Normal approximations are poor for four majority outcomes per context.", "rows": rows}


def prepare(root=ROOT, repeats=20):
    root = Path(root)
    cases, variants, rows = make_inputs()
    schedule = make_schedule(rows, repeats)
    assert len(cases) == 24 and len(rows) == 96
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": 1, "purpose": "Measure source-LLM response to rationale edits at fixed stance and observed social context.",
                "parent_manifest_sha256": sha(PARENT / "manifest.json"),
                "preparation_code_sha256": sha(__file__), "conditions": list(CONDITIONS),
                "primary_contrast": ["valid", "invalid"],
                "secondary_contrasts": [["valid", "conclusion_only"], ["valid_paraphrase", "valid"]],
                "cases": 24, "prompts": 96, "raw_samples_per_prompt": repeats,
                "majority_group_size": 5, "groups_per_prompt": repeats // 5,
                "source_model": MODEL, "temperature": .7, "max_completion_tokens": MAX_OUTPUT_TOKENS,
                "endpoint": "https://api.openai.com/v1/chat/completions", "n": 1, "store": False,
                "schedule_seed": 804211, "generation_seed": None,
                "request_style": "One independent user-message request per raw sample; no system message, chat history, tools or extra instructions.",
                "instruction_regime": "Released snap-judgment prompt; exact historical full prompts and historical model snapshot were not logged.",
                "matching_limit": "Same source-model family and recorded temperature, but a new controlled study, not exact historical replay. Current receiver opinion/history remain fixed in the predictor; the released source formatter does not explicitly insert them as separate fields.",
                "case_selection": "Reuse fixed uncertain no-text contexts; no source outcomes select cases, arguments or sample counts.",
                "controls_limit": "All target messages favor the correct option. Independent solving can produce a null quality effect. Conclusion-only and paraphrase controls change length; contrasts include that change, not pure logical validity alone.",
                "analysis": "Raw probabilities and observed disjoint majority-of-five rates are separate. Average paired context effects equally and report each relationship sign separately. No predictor-accuracy claim from raw response fractions. No generalization claim from four reused questions.",
                "parse_policy": "Released standalone-A/B parser, plus strict-format flag. Do not retry, substitute or reassign invalid/truncated answers; report them and missing majority groups.",
                "retry_policy": "No automatic retries; preserve attempt ledger and reserve uncertain charges. Interrupted attempts require reconciliation before resumption.",
                "prices_checked_date": "2026-11-08", "pricing_url": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
                "input_usd_per_million": INPUT_USD_PER_MILLION, "output_usd_per_million": OUTPUT_USD_PER_MILLION,
                "model_shutdown_date": "2027-01-14", "deprecation_url": "https://developers.openai.com/api/docs/deprecations"}
    path = root / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != protocol:
            raise ValueError("Prepared protocol differs; use a new run directory")
        manifest = json.loads((root / "manifest.json").read_text())
        for name, checksum in manifest["artifacts"].items():
            if sha(root / name) != checksum:
                raise ValueError(f"Prepared artifact changed: {name}")
        return manifest
    write_json(path, protocol)
    write_json(root / "cases.json", cases)
    write_json(root / "argument_variants.json", variants)
    write_jsonl(root / "prompts.jsonl", rows)
    write_jsonl(root / "schedule.jsonl", schedule)
    write_json(root / "precision_plan.json", planning_precision(len(cases), repeats))
    cost = {"requests": len(schedule), "raw_samples": len(schedule),
            "output_token_cap_total": len(schedule) * MAX_OUTPUT_TOKENS,
            "conservative_input_token_allowance": repeats * sum(len(r["prompt"].encode("utf-8")) + 64 for r in rows),
            "conservative_cost_allowance_usd": repeats * sum(reserve_cost(r["prompt"]) for r in rows),
            "method": "UTF-8 byte count plus 64 framing tokens per single-message request, full output cap, no cache discount. Conservative planning allowance, not an exact tokenizer count or provider billing guarantee. Excludes tax/account funding requirements.",
            "price_source": protocol["pricing_url"], "price_checked_date": protocol["prices_checked_date"]}
    write_json(root / "cost_plan.json", cost)
    files = ("protocol.json", "cases.json", "argument_variants.json", "prompts.jsonl", "schedule.jsonl", "precision_plan.json", "cost_plan.json")
    manifest = {"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                "preparation_source_calls": 0, "cases": 24, "prompts": 96, "planned_raw_samples": len(schedule),
                "artifacts": {name: sha(root / name) for name in files}}
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["artifacts"].items():
        if sha(root / name) != checksum:
            raise ValueError(f"Prepared artifact changed: {name}")
    return json.loads((root / "protocol.json").read_text()), read_jsonl(root / "prompts.jsonl"), read_jsonl(root / "schedule.jsonl")


def majority_outcomes(schedule, observations):
    by_id = {r["sample_id"]: r for r in observations}
    if len(by_id) != len(observations):
        raise ValueError("Duplicate sample results")
    expected = {r["sample_id"] for r in schedule}
    if not set(by_id).issubset(expected):
        raise ValueError("Result contains an unplanned sample")
    groups = defaultdict(list)
    for planned in schedule:
        groups[(planned["prompt_id"], planned["majority_group"])].append(planned)
    output = []
    for (prompt_id, group), members in sorted(groups.items()):
        if len(members) != 5 or {r["vote_position"] for r in members} != set(range(5)):
            raise ValueError("Majority groups must contain exactly five preassigned positions")
        values = [by_id.get(r["sample_id"], {}).get("parsed_sign") for r in members]
        complete = all(v in (-1, 1) for v in values)
        output.append({"prompt_id": prompt_id, "majority_group": group,
                       "valid_votes": sum(v in (-1, 1) for v in values),
                       "majority_sign": (1 if sum(values) > 0 else -1) if complete else None})
    return output


def wilson(successes, n):
    if not n:
        return None
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - half), min(1., center + half)]


def analyze(root=ROOT):
    root = Path(root)
    protocol, prompts, schedule = verify_prepared(root)
    ledger = read_jsonl(root / "source_ledger.jsonl")
    observations = [r for r in ledger if r["event"] == "completed"]
    planned = {r["sample_id"]: r for r in schedule}
    for row in observations:
        expected = planned.get(row["sample_id"])
        if expected is None or any(row[k] != expected[k] for k in ("prompt_id", "prompt_sha256", "majority_group", "vote_position")):
            raise ValueError("Observation identity differs from the frozen schedule")
        if row.get("response_model") != protocol["source_model"]:
            raise ValueError("Unexpected source model; do not pool it with the prepared model")
    groups = majority_outcomes(schedule, observations)
    by_prompt = defaultdict(list)
    for row in observations:
        by_prompt[row["prompt_id"]].append(row)
    grouped = defaultdict(list)
    for row in groups:
        grouped[row["prompt_id"]].append(row)
    results = []
    for prompt in prompts:
        observed = by_prompt[prompt["prompt_id"]]
        raw = [r["parsed_sign"] for r in observed if r.get("parsed_sign") in (-1, 1)]
        votes = [r["majority_sign"] for r in grouped[prompt["prompt_id"]] if r["majority_sign"] is not None]
        row = {k: prompt[k] for k in ("prompt_id", "case_id", "qid", "condition", "relationship_sign", "message_stance")}
        for label, values in (("raw", raw), ("majority", votes)):
            successes = sum(v == prompt["message_stance"] for v in values)
            row[label + "_n"] = len(values)
            row[label + "_successes"] = successes
            row[label + "_p_message_stance"] = successes / len(values) if values else None
            row[label + "_wilson95"] = wilson(successes, len(values))
        row["returned_responses"] = len(observed)
        row["parse_failures"] = len(observed) - len(raw)
        row["strict_format_failures"] = sum(not r.get("strict_format", False) for r in observed)
        results.append(row)
    lookup = {(r["case_id"], r["condition"]): r for r in results}
    contrasts = []
    for case in json.loads((root / "cases.json").read_text()):
        for a, b in CONTRASTS:
            left, right = lookup[case["case_id"], a], lookup[case["case_id"], b]
            for label in ("raw", "majority"):
                pa, pb = left[label + "_p_message_stance"], right[label + "_p_message_stance"]
                contrasts.append({"case_id": case["case_id"], "qid": case["qid"], "relationship_sign": case["relationship_sign"],
                                  "contrast": a + "_minus_" + b, "target": label,
                                  "left_n": left[label + "_n"], "right_n": right[label + "_n"],
                                  "difference": pa - pb if pa is not None and pb is not None else None})
    complete = len(observations) == len(schedule) and all(r.get("parsed_sign") in (-1, 1) for r in observations)
    summary = {"status": "complete" if complete else "partial" if observations else "no_source_results",
               "source_calls_attempted": sum(r["event"] == "started" for r in ledger),
               "returned_responses": len(observations), "planned_responses": len(schedule),
               "valid_majority_groups": sum(r["majority_sign"] is not None for r in groups),
               "planned_majority_groups": len(groups), "counterfactual_source_effect_measured": any(r["difference"] is not None for r in contrasts),
               "interpretation": "Pilot on selected fixed contexts, not fresh-question generalization. Invalid outputs are reported and never replaced or regrouped. Raw fractions cannot be substituted for majority probabilities.",
               "protocol_sha256": sha(root / "protocol.json")}
    output = root / "analysis"
    write_json(output / "summary.json", summary)
    write_json(output / "per_prompt.json", results)
    write_json(output / "paired_contexts.json", contrasts)
    write_json(output / "majority_groups.json", groups)
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    result = prepare(args.root, args.repeats) if args.stage == "prepare" else analyze(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
