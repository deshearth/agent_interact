"""Prepare, but do not execute, a controlled source-LLM argument-quality pilot.

There is intentionally no API client, sampling command, or model invocation here.
Existing questions make this a manipulation pilot, not a new generalization test.
"""
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts.argument_features import load_cache, parse_stance
from scripts import joint_argument_encoding as session

ROOT = session.SESSION/"controlled_probe_draft"
SOURCE = Path("data/raw/upstream/lib/datagen/samplers.py")
CONDITIONS = ("valid", "invalid", "valid_paraphrase")
VARIANTS = {
    "math_train_133": {
        "valid": "I favor option A because six faucets have a combined rate of 100/3 gallons per minute. Filling 25 gallons therefore takes 3/4 of a minute, or 45 seconds.",
        "invalid": "I favor option A because six faucets have a combined rate of 100 gallons per minute. Filling 25 gallons therefore takes 3/4 of a minute, or 45 seconds.",
        "valid_paraphrase": "I favor option A because doubling the faucets doubles their filling rate, while the smaller tub holds one quarter of the water. The time is therefore 360 seconds divided by eight, or 45 seconds.",
        "audit": "Correct combined rate: (100/6)*(6/3)=100/3 gallons/minute; time 25/(100/3)*60=45 seconds. Invalid version changes only the rate to 100, contradicting its stated filling time."},
    "math_train_162": {
        "valid": "I favor option A because there are six mixed-sign unordered pairs among ten possible pairs of distinct integers. The probability is therefore 3/5.",
        "invalid": "I favor option A because there are six mixed-sign unordered pairs among fifteen possible pairs of distinct integers. The probability is therefore 3/5.",
        "valid_paraphrase": "I favor option A because choosing one of the three negative integers and one of the two positive integers gives six favorable pairs. Dividing by the ten distinct unordered pairs gives 3/5.",
        "audit": "There are C(5,2)=10 distinct unordered pairs and 3*2=6 mixed-sign pairs. Invalid version uses 15 total pairs while retaining the same conclusion."},
    "math_train_359": {
        "valid": "I favor option A because the area of ABC gives |y_A| = 18, and the area of ADE gives |300 - x_A + y_A| = 1556. The four resulting x_A values sum to 1200.",
        "invalid": "I favor option A because the area of ABC gives |x_A| = 18, and the area of ADE gives |300 - x_A + y_A| = 1556. The four resulting x_A values sum to 1200.",
        "valid_paraphrase": "I favor option A because y_A is either 18 or -18, while x_A = 300 + y_A + 1556 or x_A = 300 + y_A - 1556. Adding these four possibilities gives 1200.",
        "audit": "ABC has a horizontal base of length 223, so |y_A|=2*2007/223=18. ADE has determinant 9*(300-x_A+y_A), hence absolute value 1556 after dividing 2*7002 by 9. Invalid version confuses the first constraint with |x_A|."},
    "math_train_497": {
        "valid": "I favor option B because the consecutive page numbers 135 and 136 have product 18,360. Their sum is 271.",
        "invalid": "I favor option B because the consecutive page numbers 134 and 137 have product 18,360. Their sum is 271.",
        "valid_paraphrase": "I favor option B because solving n(n+1)=18,360 gives the consecutive positive integers 135 and 136. Adding them gives 271.",
        "audit": "135*136=18360 and 135+136=271. The invalid pair also sums to 271, but is not consecutive and its product is 18358."},
}


def verify_arithmetic():
    assert Fraction(100,6)*2 == Fraction(100,3)
    assert Fraction(25,1)/Fraction(100,3)*60 == 45
    assert Fraction(3*2, 5*4//2) == Fraction(3,5)
    assert Fraction(2*2007,223) == 18 and Fraction(2*7002,9) == 1556
    assert sum(300+y+sgn*1556 for y in (-18,18) for sgn in (-1,1)) == 1200
    assert 135*136 == 18360 and 134*137 == 18358


def main():
    verify_arithmetic()
    cfg = config("configs/argument_mechanisms.yaml")
    arrays, index, _, _ = load_cache(cfg, verify=True)
    manifest = json.loads(Path("data/data_manifest.json").read_text())["files"]
    assert digest(SOURCE) == manifest[str(SOURCE.relative_to("data/raw/upstream"))]["sha256"]
    spec = importlib.util.spec_from_file_location("released_prompt_formatter", SOURCE)
    formatter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter)
    cases, required = [], set()
    for qid in sorted(VARIANTS):
        expected_stance = parse_stance(VARIANTS[qid]["valid"])
        assert all(parse_stance(VARIANTS[qid][c]) == expected_stance for c in CONDITIONS)
        es = sorted([e for e,r in enumerate(index) if r["qid"] == qid], key=lambda e:(index[e]["replica"],index[e]["name"]))
        for sign in (1,-1):
            receivers = set()
            for e in es:
                for receiver in range(32):
                    if receiver in receivers:
                        continue
                    slots = np.flatnonzero((arrays["signs"][e, receiver] == sign)&(arrays["stance"][e, 0, receiver] == expected_stance))
                    if not len(slots):
                        continue
                    slot = int(slots[0])
                    active = np.flatnonzero(arrays["signs"][e,receiver]).tolist()
                    active.sort(key=lambda k:int(arrays["neighbors"][e,receiver,k]))
                    random.Random(f"{index[e]['name']}|1|{receiver}").shuffle(active)
                    grouped = {sgn:[k for k in active if arrays["signs"][e,receiver,k] == sgn] for sgn in (1,-1)}
                    ids = {sgn:[int(arrays["text_ids"][e,0,receiver,k]) for k in grouped[sgn]] for sgn in (1,-1)}
                    for items in ids.values():
                        required.update(items)
                    case = {"case_id": f"{qid}-{'friendly' if sign == 1 else 'unfriendly'}-{len(receivers):02d}",
                        "qid": qid, "episode_id": index[e]["episode_id"], "raw_path": index[e]["raw_path"],
                        "round": 0, "receiver": receiver, "sender": int(arrays["neighbors"][e,receiver,slot]),
                        "relationship_sign": sign, "message_stance": int(expected_stance),
                        "current_receiver_state": int(arrays["spins"][e,0,receiver]), "grouped_text_ids": ids,
                        "target_position_within_group": grouped[sign].index(slot), "target_original_text_id": int(arrays["text_ids"][e,0,receiver,slot]),
                        "question": index[e]["statement"], "choices": index[e]["choices"]}
                    cases.append(case); receivers.add(receiver)
                    if len(receivers) == 3:
                        break
                if len(receivers) == 3:
                    break
            assert len(receivers) == 3
    texts = {}
    for line in (session.AUDIT/"raw_texts.jsonl").open():
        record = json.loads(line)
        if record["id"] in required:
            texts[record["id"]] = record["text"]
    rows = []
    for case in cases:
        raw = json.loads(Path(case["raw_path"]).read_text())
        case["receiver_persona"] = raw["personas"][case["receiver"]]
        case["raw_sha256"] = digest(case["raw_path"])
        for condition in CONDITIONS:
            inbox = {sgn:[texts[t] for t in case["grouped_text_ids"][sgn]] for sgn in (1,-1)}
            inbox[case["relationship_sign"]][case["target_position_within_group"]] = VARIANTS[case["qid"]][condition]
            prompt = formatter._objective_context(case["receiver_persona"],case["question"],case["choices"],inbox[1],inbox[-1])+formatter.OBJECTIVE_SPIN_INSTRUCTION
            rows.append({"case_id":case["case_id"], "condition":condition, "qid":case["qid"],
                "relationship_sign":case["relationship_sign"], "message_stance":case["message_stance"],
                "argument":VARIANTS[case["qid"]][condition], "inbox_agree":inbox[1], "inbox_disagree":inbox[-1], "prompt":prompt})
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256":digest(__file__), "formatter_sha256":digest(SOURCE),
        "feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json"),
        "raw_texts_sha256":digest(session.AUDIT/"raw_texts.jsonl")}
    if (ROOT/"manifest.json").exists():
        saved = json.loads((ROOT/"manifest.json").read_text())
        assert all(saved[k] == v for k,v in identity.items())
        for file,sha in saved["artifacts"].items():
            assert digest(ROOT/file) == sha
        print(json.dumps({"prepared":True,"reused":True,"api_calls":0}),flush=True)
        return
    json_write(ROOT/"cases.json", cases)
    json_write(ROOT/"argument_variants.json", VARIANTS)
    with (ROOT/"prompts.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    json_write(ROOT/"manifest.json", {**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"draft_not_executed", "api_calls":0, "cases":len(cases),"prompts":len(rows),
        "proposed_repeats_per_prompt":20,"proposed_total_raw_samples":20*len(rows),
        "scope":"Existing four development questions, three receivers per signed relationship category. Manipulation pilot only; this cannot be called a fresh generalization test.",
        "primary_contrast":"Valid versus deliberately invalid rationale with the explicit A/B conclusion, receiver, question, background inbox, relationship cue and presentation order held fixed.",
        "control":"Valid paraphrase estimates sensitivity to rewording. Invalid variants are minimal numerical/variable edits; paraphrases are not length-matched, so they are a secondary diagnostic.",
        "selection":"First eligible round-0 inboxes ordered by replica/name, with three distinct receivers per question/sign. Eligibility uses current stance and sign, never the next opinion. No source LLM is invoked.",
        "historical_order":"Reconstructed source order is explicitly fixed for NEW prompts; no claim is made that these match the legacy prompts exactly.",
        "analysis":"For each case estimate P(next answer equals target-message stance); compare conditions within case and summarize across questions/signs. Retain every individual sample, parse failure and exact prompt. Twenty repeats is a pilot count, not a powered confirmatory design; choose a fresh-question study from pilot variability.",
        "interpretation":"A source-LLM response to rationale quality without a predictor response would isolate a representation/learning limitation. No source response despite a successful manipulation would support low incremental argument signal under that prompting. A null on four reused questions remains inconclusive.",
        "review_needed_before_future_execution":"Check that edited rationales are natural and that validity is the intended changed property; verify current model access and an explicit sampling budget. This local preparation performs no calls and makes no cost estimate.",
        "artifacts":{f:digest(ROOT/f) for f in ("cases.json","argument_variants.json","prompts.jsonl")}})
    print(json.dumps({"prepared":True,"cases":len(cases),"prompts":len(rows),"api_calls":0}),flush=True)


if __name__ == "__main__":
    main()
