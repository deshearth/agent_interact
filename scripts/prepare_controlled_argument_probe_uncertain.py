"""Refine the unexecuted pilot toward mixed inboxes with uncertain no-text predictions."""
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from scripts.argument_features import load_cache,parse_stance
from scripts import prepare_controlled_argument_probe as first
from scripts import joint_argument_encoding as session

ROOT = session.SESSION/"controlled_probe_uncertain"
VARIANTS,CONDITIONS,SOURCE = first.VARIANTS,first.CONDITIONS,first.SOURCE
verify_arithmetic = first.verify_arithmetic
KEYS = ["episode_id","round","agent"]


def main():
    verify_arithmetic()
    cfg = config("configs/argument_mechanisms.yaml")
    arrays,index,_,_ = load_cache(cfg,verify=True)
    ep = {r["episode_id"]:e for e,r in enumerate(index)}
    frames,hashes = [],{}
    for seed in (42,43,44):
        path = Path("runs/question_context_v1/evaluation")/f"full_embedding-stance_only-s{seed}"/"unseen_predictions.parquet"
        identity = json.loads((path.parent/"identity.json").read_text())
        assert digest(path) == identity["artifacts"][path.name]
        # Do not load next-opinion outcomes for selection.
        frame = pd.read_parquet(path,columns=KEYS+["qid","p_positive"]).sort_values(KEYS).reset_index(drop=True)
        if frames:pd.testing.assert_frame_equal(frames[0][KEYS+["qid"]],frame[KEYS+["qid"]])
        frames.append(frame);hashes[str(path)] = digest(path)
    frame = frames[0].copy();frame["p_A"] = np.stack([f.p_positive.to_numpy() for f in frames]).mean(0)
    frame["uncertainty_distance"] = abs(frame.p_A-.5)
    eligible = frame[frame.p_A.between(.4,.6)].sort_values(["uncertainty_distance",*KEYS],kind="stable")
    cases,required,used_contexts = [],set(),set()
    for qid in sorted(VARIANTS):
        stance = parse_stance(VARIANTS[qid]["valid"])
        for sign in (1,-1):
            receivers = set()
            for row in eligible[eligible.qid == qid].itertuples(index=False):
                e,t,r = ep[row.episode_id],int(row.round),int(row.agent)
                if r in receivers or (e,t,r) in used_contexts:continue
                active = np.flatnonzero(arrays["signs"][e,r])
                if set(arrays["stance"][e,t,r,active]) != {-1,1}:continue
                slots = np.flatnonzero((arrays["signs"][e,r] == sign)&(arrays["stance"][e,t,r] == stance))
                if not len(slots):continue
                slot = int(slots[0]);order = sorted(active.tolist(),key=lambda k:int(arrays["neighbors"][e,r,k]))
                random.Random(f"{index[e]['name']}|{t+1}|{r}").shuffle(order)
                groups = {s:[k for k in order if arrays["signs"][e,r,k] == s] for s in (1,-1)}
                ids = {s:[int(arrays["text_ids"][e,t,r,k]) for k in groups[s]] for s in (1,-1)}
                for v in ids.values():required.update(v)
                cases.append({"case_id":f"{qid}-{'friendly' if sign == 1 else 'unfriendly'}-{len(receivers):02d}",
                    "qid":qid,"episode_id":row.episode_id,"raw_path":index[e]["raw_path"],"round":t,"receiver":r,
                    "sender":int(arrays["neighbors"][e,r,slot]),"relationship_sign":sign,"message_stance":int(stance),
                    "current_receiver_state":int(arrays["spins"][e,t,r]),"grouped_text_ids":ids,
                    "target_position_within_group":groups[sign].index(slot),"target_original_text_id":int(arrays["text_ids"][e,t,r,slot]),
                    "question":index[e]["statement"],"choices":index[e]["choices"],"selection_no_text_p_A":float(row.p_A)})
                receivers.add(r);used_contexts.add((e,t,r))
                if len(receivers) == 3:break
            assert len(receivers) == 3,(qid,sign,len(receivers))
    assert len(used_contexts) == len(cases) == 24
    texts = {}
    for line in (session.AUDIT/"raw_texts.jsonl").open():
        row = json.loads(line)
        if row["id"] in required:texts[row["id"]] = row["text"]
    spec = importlib.util.spec_from_file_location("released_prompt_formatter",SOURCE)
    formatter = importlib.util.module_from_spec(spec);spec.loader.exec_module(formatter)
    rows = []
    for case in cases:
        raw = json.loads(Path(case["raw_path"]).read_text())
        case["receiver_persona"] = raw["personas"][case["receiver"]];case["raw_sha256"] = digest(case["raw_path"])
        for condition in CONDITIONS:
            inbox = {s:[texts[i] for i in case["grouped_text_ids"][s]] for s in (1,-1)}
            inbox[case["relationship_sign"]][case["target_position_within_group"]] = VARIANTS[case["qid"]][condition]
            prompt = formatter._objective_context(case["receiver_persona"],case["question"],case["choices"],inbox[1],inbox[-1])+formatter.OBJECTIVE_SPIN_INSTRUCTION
            rows.append({"case_id":case["case_id"],"condition":condition,"qid":case["qid"],"relationship_sign":case["relationship_sign"],
                "message_stance":case["message_stance"],"argument":VARIANTS[case["qid"]][condition],
                "inbox_agree":inbox[1],"inbox_disagree":inbox[-1],"prompt":prompt})
    ROOT.mkdir(parents=True,exist_ok=True)
    identity = {"code_sha256":digest(__file__),"original_draft_manifest_sha256":digest(first.ROOT/"manifest.json"),
        "formatter_sha256":digest(SOURCE),"selection_prediction_sha256":hashes,
        "feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json")}
    if (ROOT/"manifest.json").exists():
        saved = json.loads((ROOT/"manifest.json").read_text());assert all(saved[k] == v for k,v in identity.items())
        for file,sha in saved["artifacts"].items():assert digest(ROOT/file) == sha
        print(json.dumps({"prepared":True,"reused":True,"api_calls":0}));return
    json_write(ROOT/"cases.json",cases);json_write(ROOT/"argument_variants.json",VARIANTS)
    with (ROOT/"prompts.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    json_write(ROOT/"manifest.json",{**identity,"status":"draft_not_executed","created_utc":datetime.now(timezone.utc).isoformat(),
        "api_calls":0,"cases":24,"prompts":72,"proposed_repeats_per_prompt":20,"proposed_total_raw_samples":1440,
        "refinement_reason":"The first fixed-order draft often produced confident predictor outputs. Focus this manipulation pilot on mixed-opinion inboxes with uncertain frozen no-text ensemble predictions, using no next-state outcomes for selection.",
        "selection":"P(A) between 0.4 and 0.6 from the three frozen full-question no-text models; mixed A/B inbox; target sender favors the correct option. Sort by closeness to 0.5, then stable case keys. Three distinct receivers per question/sign, 24 unique base inboxes. No source outcomes or intervention predictions choose cases.",
        "primary_contrast":"Valid versus minimally edited invalid rationale, same correct A/B conclusion, graph cue and all remaining prompt text. Valid paraphrase is a secondary, larger-rewording control.",
        "scope":"Four reused development questions and deliberately selected difficult contexts: manipulation pilot only, not representative performance or confirmation. The source LLM may still be confident; model uncertainty is only a proxy.",
        "limits":"All target messages favor the correct answer; an independently solving agent may retain that answer even with an invalid rationale. A null result would require manipulation/power checks and cannot by itself prove argument irrelevance. Paraphrases are not length-matched. Twenty repeats is a pilot count, not a power calculation.",
        "next_step":"Run only under a separate source-model execution budget; record every exact prompt, sampled answer/parse failure, model version and sampling settings. Compare within-context probability of adopting the target stance, then plan a fresh-question study.",
        "artifacts":{f:digest(ROOT/f) for f in ("cases.json","argument_variants.json","prompts.jsonl")}})
    print(json.dumps({"prepared":True,"cases":24,"prompts":72,"api_calls":0,
        "maximum_no_text_confidence":max(max(c["selection_no_text_p_A"],1-c["selection_no_text_p_A"]) for c in cases)}),flush=True)


if __name__ == "__main__":main()
