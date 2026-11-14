"""Read-only predictor responses to draft controlled rationale edits.

No source LLM outcomes exist for these edits. Report sensitivity, never accuracy
or a demonstrated causal effect on the agents. All checkpoints remain frozen.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_context as joint
from scripts import joint_argument_encoding as source
from scripts import joint_argument_encoding_windowed as encoding
from scripts import prepare_controlled_argument_probe as draft
from scripts.argument_features import load_cache

ROOT = source.SESSION/"quality_predictor_probe"


def standalone(tokenizer, model, messages):
    tokens = tokenizer(messages, padding=True, truncation=False, return_tensors="pt")
    assert tokens["input_ids"].shape[1] <= 256
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(hidden.dtype)[..., None]
        pooled = (hidden*mask).sum(1)/mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.numpy()


def main():
    torch.set_num_threads(2)
    manifest = json.loads((draft.ROOT/"manifest.json").read_text())
    assert manifest["status"] == "draft_not_executed" and manifest["api_calls"] == 0
    for file, sha in manifest["artifacts"].items():
        assert digest(draft.ROOT/file) == sha
    cfg = config(question.CONFIG)
    arrays, index, old_vectors, metadata = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    episodes = {r["episode_id"]: e for e,r in enumerate(index)}
    cases = {r["case_id"]: r for r in json.loads((draft.ROOT/"cases.json").read_text())}
    inputs = [json.loads(line) for line in (draft.ROOT/"prompts.jsonl").read_text().splitlines()]
    unique = sorted({(r["qid"],r["condition"]) for r in inputs})
    texts = [draft.VARIANTS[q][condition] for q,condition in unique]
    lookup = {key:i for i,key in enumerate(unique)}
    tokenizer, encoder = source.load_encoder("cpu")
    basic = standalone(tokenizer, encoder, texts)
    qstrings = json.loads((source.ROOT/"questions.json").read_text())
    qt = [tokenizer(qstrings[q],add_special_tokens=False,truncation=False,verbose=False)["input_ids"] for q,_ in unique]
    conditioned, _ = encoding.encode_records(tokenizer,encoder,qt,texts,"cpu",microbatch=12)
    # Independently reproduce cached standalone vectors on real messages.
    check_ids = sorted({int(r["target_original_text_id"]) for r in cases.values()})[:8]
    cache_text = {}
    for line in (source.AUDIT/"raw_texts.jsonl").open():
        r = json.loads(line)
        if r["id"] in check_ids:
            cache_text[r["id"]] = r["text"]
    check = standalone(tokenizer,encoder,[cache_text[i] for i in check_ids])
    np.testing.assert_allclose(check,old_vectors[check_ids],rtol=0,atol=1e-5)
    del encoder
    models = [("original_pca3",previous.ROOT,"A2",previous.make_model),
              ("full_question",question.ROOT,"full_embedding",question.make_model),
              ("question_conditioned",joint.ROOT,"question_conditioned",joint.make_model)]
    provenance = {}
    for label,root,kind,factory in models:
        for seed in (42,43,44):
            run = root/f"{kind}-full_arguments-s{seed}"
            saved = json.loads((run/"manifest.json").read_text())
            assert saved["status"] == "complete"
            assert digest(run/"best.pt") == saved["artifact_sha256"]["best.pt"]
            provenance[f"{label}/s{seed}"] = {"best_sha256":digest(run/"best.pt"),"manifest_sha256":digest(run/"manifest.json")}
    ROOT.mkdir(parents=True,exist_ok=True)
    identity = {"code_sha256":digest(__file__),"draft_manifest_sha256":digest(draft.ROOT/"manifest.json"),
        "encoding_protocol_sha256":digest(encoding.ROOT/"protocol.json"),"models":provenance}
    if (ROOT/"protocol.json").exists():
        p = json.loads((ROOT/"protocol.json").read_text())
        assert all(p[k] == v for k,v in identity.items())
    else:
        json_write(ROOT/"protocol.json",{**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"Read-only predictor responses to one edited incoming rationale at fixed stance and all other recorded state. Nine frozen models, 24 contexts, three authored conditions.",
            "scope":"Illustrative controlled inputs on four reused development questions. No source LLM calls/outcomes and no predictive accuracy claims. Mathematical validity labels authored for the draft never train a model.",
            "checks":"Standalone encoder pooling independently reproduces eight cached real-message embeddings within 1e-5. Target edge/stance/sign match recorded state; only that message vector changes within each case."})
    np.savez(ROOT/"new_argument_vectors.npz", keys=np.array([q+"/"+c for q,c in unique]), standalone=basic, question_conditioned=conditioned)
    geometry = []
    for label,vectors in (("standalone",basic),("question_conditioned",conditioned)):
        for qid in draft.VARIANTS:
            a,b,c = [vectors[lookup[qid,k]] for k in draft.CONDITIONS]
            geometry.append({"encoding":label,"qid":qid,"valid_invalid_cosine":float(a@b),
                "valid_paraphrase_cosine":float(a@c),"valid_invalid_l2":float(np.linalg.norm(a-b)),
                "valid_paraphrase_l2":float(np.linalg.norm(a-c))})
    pd.DataFrame(geometry).to_csv(ROOT/"embedding_geometry.csv",index=False)
    output = []
    for label,root,kind,factory in models:
        a, vec, new = arrays, old_vectors, basic
        if label != "original_pca3":
            a = question.augmented_arrays(arrays,index,question.ROOT/"question_features.npz","full_embedding")
        if label == "question_conditioned":
            with np.load(source.ROOT/"ids.npz",allow_pickle=False) as mapping:
                a = {**a,"text_ids":mapping["text_ids"]}
            vec = np.load(encoding.ROOT/"question_conditioned.npy",mmap_mode="r",allow_pickle=False)
            new = conditioned
        donors = np.load(previous.ROOT/"donors_s42.npy",allow_pickle=False)
        ds = previous.SplitSnapshots(a,index,vec,"unseen","full_arguments",donors)
        positions = {p:i for i,p in enumerate(ds.pairs)}
        batches, receivers = [], []
        for row in inputs:
            case = cases[row["case_id"]]
            e, t, receiver = episodes[case["episode_id"]],case["round"],case["receiver"]
            assert index[e]["diagnostic_role"] == "unseen"
            batch = ds[positions[e,t]]
            slots = np.flatnonzero((a["neighbors"][e,receiver] == case["sender"]) & (a["signs"][e,receiver] == case["relationship_sign"]))
            assert len(slots) == 1
            slot = int(slots[0])
            assert a["stance"][e,t,receiver,slot] == case["message_stance"]
            batch["text"][receiver,slot] = torch.from_numpy(new[lookup[row["qid"],row["condition"]]])
            batches.append(batch); receivers.append(receiver)
        batch = default_collate(batches)
        # True next-state targets are discarded: no counterfactual targets exist.
        batch.pop("target")
        for seed in (42,43,44):
            run = root/f"{kind}-full_arguments-s{seed}"
            model = factory(cfg,kind).eval()
            model.load_state_dict(torch.load(run/"best.pt",map_location="cpu",weights_only=False)["model"])
            with torch.inference_mode():
                logits = model(batch)[torch.arange(len(inputs)),torch.tensor(receivers)]
                probabilities = logits.softmax(-1)[:,1].numpy()
            for row,p in zip(inputs,probabilities):
                output.append({"model":label,"seed":seed,"case_id":row["case_id"],"qid":row["qid"],
                    "condition":row["condition"],"relationship_sign":row["relationship_sign"],
                    "message_stance":row["message_stance"],"p_A":float(p),
                    "p_message_stance":float(p if row["message_stance"] == 1 else 1-p)})
    frame = pd.DataFrame(output)
    frame.to_csv(ROOT/"predictions.csv",index=False)
    pairs = frame.pivot(index=["model","seed","case_id","qid","relationship_sign"],columns="condition",values="p_message_stance").reset_index()
    pairs["valid_minus_invalid"] = pairs.valid-pairs.invalid
    pairs["paraphrase_minus_valid"] = pairs.valid_paraphrase-pairs.valid
    pairs["abs_valid_invalid"] = pairs.valid_minus_invalid.abs()
    pairs["abs_paraphrase_valid"] = pairs.paraphrase_minus_valid.abs()
    pairs["valid_invalid_choice_change"] = (pairs.valid >= .5) != (pairs.invalid >= .5)
    pairs["paraphrase_choice_change"] = (pairs.valid_paraphrase >= .5) != (pairs.valid >= .5)
    pairs.to_csv(ROOT/"paired_predictions.csv",index=False)
    metrics = ["valid_minus_invalid","paraphrase_minus_valid","abs_valid_invalid","abs_paraphrase_valid", "valid_invalid_choice_change","paraphrase_choice_change"]
    summary = pairs.groupby(["model","relationship_sign"],sort=True)[metrics].mean().reset_index()
    summary.to_csv(ROOT/"summary.csv",index=False)
    files = ("new_argument_vectors.npz","embedding_geometry.csv","predictions.csv","paired_predictions.csv","summary.csv")
    json_write(ROOT/"complete.json",{"protocol_sha256":digest(ROOT/"protocol.json"),"api_calls":0,
        "counterfactual_source_outcomes":0,"accuracy_evaluations":0,"models":9,"contexts":24,
        "cached_encoder_max_absolute_difference":float(np.abs(check-old_vectors[check_ids]).max()),
        "completed_utc":datetime.now(timezone.utc).isoformat(),"artifacts":{f:digest(ROOT/f) for f in files}})
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
