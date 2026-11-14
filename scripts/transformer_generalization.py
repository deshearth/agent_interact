"""Nested whole-question comparison of regularized graph predictors.

Only the released GPT development questions and familiar fit graphs are used.
All outer assessment is deferred until tuning, selection and refitting finish.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write, seed_all
from mac_agents.data.prepare import load_prepared, pca_fit_transform
from mac_agents.models.baselines import baseline_logp, fit_baseline
from mac_agents.models.graph import GraphUpdate
from mac_agents.train import restore_rng, rng_state, save_checkpoint

CONFIG = Path("configs/transformer_generalization.yaml")
CORE_SHA = "92c500581c6d74e0d5effa4b89f9c402bb76f7800151ac6dbb3b3afcae939dda"
Q_EMBEDDINGS = Path("data/raw/upstream/data/obj/question_embeddings_choices.json")
P_EMBEDDINGS = Path("data/raw/upstream/data/obj/persona_embeddings.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def make_folds(qids, seed, outer_count=4, inner_count=3):
    qids = sorted(qids)
    assert len(qids) == len(set(qids)) == 20
    assert outer_count == 4 and inner_count == 3
    rng = np.random.default_rng(seed)
    blocks = [sorted(x.tolist()) for x in np.array_split(rng.permutation(qids), outer_count)]
    result = []
    for outer, assess in enumerate(blocks):
        fitting = sorted(set(qids)-set(assess))
        inner_blocks = [sorted(x.tolist()) for x in np.array_split(rng.permutation(fitting), inner_count)]
        result.append({"id": f"o{outer}_refit", "outer": outer, "phase": "refit",
                       "train_questions": fitting, "validation_questions": [], "assessment_questions": assess})
        for inner, validation in enumerate(inner_blocks):
            result.append({"id": f"o{outer}_i{inner}", "outer": outer, "inner": inner, "phase": "tune",
                           "train_questions": sorted(set(fitting)-set(validation)),
                           "validation_questions": validation, "assessment_questions": assess})
    return result


def fit_context(question_embeddings, persona_embeddings, train_questions, episode_qids):
    qids = sorted(question_embeddings)
    values = np.asarray([question_embeddings[q] for q in qids], dtype=np.float64)
    features, qmean, qcomponents = pca_fit_transform(values, [qids.index(q) for q in train_questions])
    persona, pmean, pcomponents = pca_fit_transform(persona_embeddings, np.arange(len(persona_embeddings)))
    qmap = dict(zip(qids, features))
    q = np.stack([qmap[x] for x in episode_qids])
    e, n = len(q), len(persona)
    context = np.concatenate([np.ones((e,n,1)), np.broadcast_to(persona, (e,n,3)),
        np.broadcast_to(q[:,None], (e,n,3)),
        (persona[None,:,:,None]*q[:,None,None,:]).reshape(e,n,9)], -1).astype(np.float32)
    return {"context": context, "question_mean": qmean, "question_components": qcomponents,
            "persona_mean": pmean, "persona_components": pcomponents}


def fold_dataset(common, context, train_questions):
    qids = common["episode_qids"]
    episodes = np.flatnonzero(np.isin(qids, train_questions))
    pairs = [(int(e), int(t)) for e in episodes for t in range(8) if common["valid"][e,t]]
    arrays = {"spins": common["spins"], "J": common["J"], "context": context}
    return SimpleNamespace(arrays=arrays, pairs=pairs)


def model_weights_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_model(cfg, kind):
    model = GraphUpdate(kind="A2", context_dim=16, extra_dim=0,
                        dropout=cfg["training"]["dropout"], **cfg["models"][kind])
    # Shared policy: exact zero correction; ordinary models start at P(A)=1/2.
    torch.nn.init.zeros_(model.classifier[-1].weight)
    torch.nn.init.zeros_(model.classifier[-1].bias)
    return model


def logits(model, batch, residual):
    z = model(batch["s"], batch["context"], batch["J"], batch["extra"])
    delta = z[...,1]-z[...,0]
    score = delta + batch["prior"] if residual else delta
    return score, delta


def loss_for(score, correction, target, penalty):
    return F.binary_cross_entropy_with_logits(score, target) + penalty*correction.square().mean()


def metrics(logp, target, qids):
    lp = np.asarray(logp, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)
    assert lp.shape == (*y.shape, 2) and len(qids) == len(y)
    assert np.isfinite(lp).all()
    nll = -np.take_along_axis(lp, y[...,None], -1)[...,0]
    error = lp.argmax(-1) != y
    brier = 2*(np.exp(lp[...,1])-y)**2
    rows = []
    for q in sorted(set(qids)):
        mask = qids == q
        rows.append({"qid": str(q), "nll": float(nll[mask].mean()),
                     "error": float(error[mask].mean()), "brier_two_class": float(brier[mask].mean()),
                     "transitions": int(y[mask].size)})
    aggregate = {k: float(np.mean([r[k] for r in rows])) for k in ("nll", "error", "brier_two_class")}
    aggregate.update(questions=len(rows), transitions=int(y.size))
    return aggregate, rows


class FoldData:
    def __init__(self, common, features, device):
        self.common, self.device = common, device
        e, t = common["pairs"].T
        s = common["spins"][e,t].astype(np.float32)
        self.y = (common["spins"][e,t+1] == 1).astype(np.int64)
        self.qids = common["episode_qids"][e]
        self.values = {"s": torch.as_tensor(s, device=device),
            "context": torch.as_tensor(features["context"][e], device=device),
            "J": torch.as_tensor(common["J"][e], device=device),
            "prior": torch.as_tensor(features["prior_logit"], dtype=torch.float32, device=device),
            "target": torch.as_tensor(self.y, dtype=torch.float32, device=device)}

    def ids(self, questions):
        return np.flatnonzero(np.isin(self.qids, questions))

    def batch(self, ids):
        result = {k: v[ids] for k,v in self.values.items()}
        result["extra"] = torch.empty((*result["s"].shape,0), device=self.device)
        return result


def predictions(model, data, ids, residual, batch_size=256):
    model.eval()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(ids), batch_size):
            score, _ = logits(model, data.batch(ids[offset:offset+batch_size]), residual)
            values.append(torch.stack([-F.softplus(score), -F.softplus(-score)], -1).cpu().numpy())
    result = np.concatenate(values)
    assert np.isfinite(result).all()
    return result


def candidates(cfg):
    return [{"id": f"c{i}", "learning_rate": lr, "penalty": penalty}
            for i, (lr, penalty) in enumerate((lr,p) for lr in cfg["training"]["learning_rates"]
                                              for p in cfg["training"]["output_penalties"])]


def freeze_previous(cfg):
    old = json.loads(Path(cfg["prior_audit"]).read_text())
    excluded = {"README.md", "STATUS.md", "mac_experiment_work_mode.md"}
    result = {name: sha for name,sha in old["artifacts"].items() if name not in excluded}
    result.update(json.loads(Path("runs/ising_transformer_capacity_v1/prior_artifacts.json").read_text()))
    result[cfg["prior_audit"]] = digest(cfg["prior_audit"])
    for name, sha in result.items():
        assert digest(name) == sha, name
    return result


def prepare(cfg):
    root = Path(cfg["output"])
    if (root/"prepared.json").exists():
        validate(cfg)
        print(json.dumps({"prepared": True, "reused": True})); return
    assert code_digest() == CORE_SHA
    root.mkdir(parents=True, exist_ok=True)
    original = config(cfg["source_config"])
    arrays, index, directory = load_prepared(original)
    ids = [i for i,r in enumerate(index) if r["official_split"] == "train" and r["graph_role"] == "fit_graph"]
    rows = [index[i] for i in ids]
    assert len(ids) == 640 and all(r["question_split"] in ("train","validation") for r in rows)
    qids = sorted({r["qid"] for r in rows})
    assert len(qids) == 20
    common = {k: arrays[k][ids] for k in ("spins", "J", "valid")}
    common["episode_qids"] = np.array([r["qid"] for r in rows])
    common["pairs"] = np.array([(e,t) for e in range(len(rows)) for t in range(8) if common["valid"][e,t]])
    assert len(common["pairs"]) == 5120
    assert np.isin(common["spins"], [-1,1]).all()
    embeddings = {x["qid"]: x["embedding"] for x in json.loads(Q_EMBEDDINGS.read_text())["items"] if x["qid"] in qids}
    pitems = sorted(json.loads(P_EMBEDDINGS.read_text())["items"], key=lambda x: x["idx"])
    personas = np.asarray([x["embedding"] for x in pitems], dtype=np.float64)
    assert len(personas) == 32
    folds = make_folds(qids, cfg["split"]["seed"], cfg["split"]["outer_folds"], cfg["split"]["inner_folds"])
    cohort = [{"phase": "tune", "fold": f["id"], "outer": f["outer"], "model": model,
               "candidate": c["id"], "seed": cfg["training"]["tuning_seed"]}
              for f in folds if f["phase"] == "tune" for model in cfg["models"] for c in candidates(cfg)]
    assert len(cohort) == 216
    prior = freeze_previous(cfg)
    protocol = {"created_utc": now(), "config": cfg, "config_sha256": digest(CONFIG),
        "implementation_sha256": digest(__file__), "core_sha256": CORE_SHA,
        "source_config_sha256": digest(cfg["source_config"]), "source_config": original,
        "source": {str(directory/name): digest(directory/name) for name in ("episodes.npz","index.json","splits.json")},
        "raw_embeddings": {str(p): digest(p) for p in (Q_EMBEDDINGS,P_EMBEDDINGS)},
        "questions": qids, "folds": folds, "candidates": candidates(cfg), "tuning_cohort": cohort,
        "parameters": {k: sum(p.numel() for p in make_model(cfg,k).parameters()) for k in cfg["models"]},
        "design": "Four outer folds of five questions; three inner folds within each remaining fifteen (ten fitting/five selection). All episodes, graphs, replicas, rounds and agents for a question stay together. Only familiar fit graphs and the twenty official training/development questions are used. The twenty consumed official test questions are excluded.",
        "models": "Full original A2, smaller A2, and same-size small A2 added to a frozen fold-fitted Ising logit. All use current opinions, 16-dimensional persona/question context and signed graph/degrees; no text, history or IDs.",
        "preprocessing": "Refit question PCA3 and the author's two-stage Ising coefficients independently on each inner training set and each outer refit set. Persona PCA is refitted on the same 32 training personas present in every fold. No held-out labels fit either preprocessing or Ising; held-out embeddings only transform.",
        "matched_regularization": "All neural models have dropout .1, AdamW decay .001, and the same six choices (two learning rates by three output-magnitude penalties). Penalty is lambda times mean squared neural logit difference. For ordinary models it shrinks toward P(A)=.5; for the hybrid it shrinks toward Ising. This matches generic confidence regularization against a learned prior. All final output layers start at zero; paired small models have identical initial weights and minibatch orders.",
        "tuning": "One fixed tuning seed, raw equal-question inner validation NLL, checks every128 updates including initialization, six-check patience, min_delta .0001, cap3072 updates. Checkpoints are selected on inner validation NLL, not penalized loss. All216 inner fits finish before settings are selected.",
        "selection": "Within each outer fold/model, choose candidate by mean of its three inner question-mean NLL scores (equal five-question folds). Tie-break by candidate ID. Refit duration is the median selected inner number of data passes, scaled to fifteen training questions and rounded up to whole optimizer updates; initialization may win. No outer assessment outcome selects a setting or duration.",
        "refit": "Refit each chosen outer-fold/model from scratch on all fifteen outer training questions at three new seeds142/143/144. Thirty-six refits use the duration frozen by inner selection. Outer labels are never scored during tuning or refitting. Freeze all selected models before one outer assessment stage.",
        "assessment": "Equal-question mean NLL primary, error and Brier secondary. Average scores over seeds, not predictions. Paired question bootstrap is descriptive; overlapping training sets, only four outer folds, prior reuse of these questions and multiple comparisons preclude fresh confirmatory claims. Report all tuning scores and fold choices, plus zero-correction baseline and direct small-vs-hybrid comparisons.",
        "source_model_calls": 0, "new_official_test_evaluations": 0, "autonomous_rollouts": False,
        "planned_neural_fits": 252, "planned_ising_fits": 16}
    path = root/"protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        assert all(existing[k] == v for k,v in protocol.items() if k != "created_utc")
    else:
        json_write(path, protocol)
    np.savez_compressed(root/"development.npz", **common)
    json_write(root/"episode_index.json", rows)
    json_write(root/"prior_artifacts.json", prior)
    artifacts = {str(p.relative_to(root)): digest(p) for p in (root/"development.npz",root/"episode_index.json",root/"prior_artifacts.json")}
    for fold in folds:
        out = root/"folds"/fold["id"]
        out.mkdir(parents=True, exist_ok=True)
        feat = fit_context(embeddings, personas, fold["train_questions"], common["episode_qids"])
        ds = fold_dataset(common, feat["context"], fold["train_questions"])
        assert len(ds.pairs) == len(fold["train_questions"])*256
        with threadpool_limits(limits=cfg["training"]["torch_threads"]):
            baseline = fit_baseline("I0", ds, original)
        e,t = common["pairs"].T
        lp = baseline_logp(baseline, common["spins"][e,t], feat["context"][e], common["J"][e])
        feat["prior_logit"] = lp[...,1]-lp[...,0]
        # Alignment only: no validation or assessment labels are scored here.
        np.savez_compressed(out/"features.npz", **feat)
        json_write(out/"ising.json", baseline)
        json_write(out/"manifest.json", {**fold,"fit_snapshots":len(ds.pairs),
                   "fit_transitions":len(ds.pairs)*32, "features_sha256":digest(out/"features.npz"),
                   "ising_sha256":digest(out/"ising.json"),"protocol_sha256":digest(path)})
        for p in out.iterdir():
            artifacts[str(p.relative_to(root))] = digest(p)
        print(json.dumps({"prepared_fold":fold["id"],"fit_questions":len(fold["train_questions"])}), flush=True)
    json_write(root/"prepared.json", {"created_utc":now(),"protocol_sha256":digest(path),"artifacts":artifacts})
    print(json.dumps({"prepared":True,"tuning_fits":len(cohort),"refits":36,"ising_fits":16}), flush=True)


def validate(cfg):
    root = Path(cfg["output"])
    protocol = json.loads((root/"protocol.json").read_text())
    assert digest(__file__) == protocol["implementation_sha256"]
    assert digest(CONFIG) == protocol["config_sha256"]
    assert code_digest() == protocol["core_sha256"] == CORE_SHA
    prepared = json.loads((root/"prepared.json").read_text())
    assert prepared["protocol_sha256"] == digest(root/"protocol.json")
    for file,sha in prepared["artifacts"].items():
        assert digest(root/file) == sha, file
    for file,sha in {**protocol["source"],**protocol["raw_embeddings"]}.items():
        assert digest(file) == sha, file
    return protocol


def session(root, cfg):
    p = root/"session.json"
    if not p.exists():
        json_write(p, {"started_utc":now(),"deadline":time.time()+60*cfg["training"]["aggregate_minutes"]})
    return json.loads(p.read_text())


def train_one(cfg, protocol, data, fold, item, device, deadline, fixed_steps=None):
    root, tc = Path(cfg["output"]), cfg["training"]
    phase, kind, seed = item["phase"], item["model"], item["seed"]
    candidate = next(c for c in protocol["candidates"] if c["id"] == item["candidate"])
    name = f"{kind}-{item['candidate']}-s{seed}"
    out = root/phase/fold["id"]/name
    out.mkdir(parents=True, exist_ok=True)
    identity = {**item,"device":device,"protocol_sha256":digest(root/"protocol.json"),
                "fold_sha256":digest(root/"folds"/fold["id"]/"manifest.json"),"fixed_steps":fixed_steps}
    if phase == "refit":
        identity["selection_sha256"] = digest(root/"selection.json")
    if (out/"manifest.json").exists():
        old = json.loads((out/"manifest.json").read_text())
        assert all(old[k] == v for k,v in identity.items())
        if old["status"] == "complete":
            for file,sha in old["artifacts"].items():
                assert digest(out/file) == sha
            return out,old
    fitting = data.ids(fold["train_questions"])
    validation = data.ids(fold["validation_questions"])
    assert not set(fitting)&set(validation)
    assert not set(data.qids[fitting])&set(fold["assessment_questions"])
    residual = kind == "ising_small"
    seed_all(seed)
    model = make_model(cfg,kind).to(device)
    initial_sha = model_weights_sha(model)
    opt = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=tc["weight_decay"])
    state = {"step":0,"epoch":0,"position":0,"best":None,"bad":0,"history":[],"seconds":0.,"created_utc":now()}
    if (out/"resume.pt").exists():
        ck = torch.load(out/"resume.pt",map_location=device,weights_only=False)
        assert ck["identity"] == identity
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        state = ck["state"]; restore_rng(ck["rng"])
    started, prior_seconds = time.monotonic(),state["seconds"]
    base = {**identity,"created_utc":state["created_utc"],"initial_weights_sha256":initial_sha,
            "parameters":sum(p.numel() for p in model.parameters()),"candidate_settings":candidate,
            "training_questions":fold["train_questions"],"validation_questions":fold["validation_questions"],
            "training_snapshots":len(fitting),"zero_initialized_output":True}
    json_write(out/"manifest.json",{**base,"status":"running"})

    def checkpoint():
        state["seconds"] = prior_seconds+time.monotonic()-started
        save_checkpoint(out/"resume.pt",{"identity":identity,"model":model.state_dict(),
            "optimizer":opt.state_dict(),"state":state,"rng":rng_state(device)})
        json_write(out/"history.json",state["history"])

    def assess_inner():
        assert phase == "tune" and len(validation) and not set(data.qids[validation])&set(fold["assessment_questions"])
        lp = predictions(model,data,validation,residual,tc["evaluation_batch_size"])
        m,_ = metrics(lp,data.y[validation],data.qids[validation])
        record = {"step":state["step"],"epoch":state["epoch"],"validation":m}
        state["history"].append(record)
        if state["best"] is None or m["nll"] < state["best"]-tc["min_delta"]:
            state["best"],state["bad"] = m["nll"],0
            save_checkpoint(out/"best.pt",{"identity":identity,"model":model.state_dict(),
                "step":state["step"],"validation":m})
        else:
            state["bad"] += 1
        json_write(root/"progress.json",{**item,"updated_utc":now(),**record})

    if phase == "tune" and not state["history"]:
        assess_inner(); checkpoint()
    maximum = tc["max_steps"] if phase == "tune" else fixed_steps
    reason = "step_cap" if phase == "tune" else "inner_selected_duration"
    while state["step"] < maximum:
        if phase == "tune" and state["bad"] >= tc["patience"]:
            reason = "early_stopping"; break
        if time.time() >= deadline:
            checkpoint()
            json_write(out/"manifest.json",{**base,"status":"bounded","step":state["step"]})
            return out,{"status":"bounded"}
        order = np.random.default_rng(seed+state["epoch"]).permutation(fitting)
        ids = order[state["position"]:state["position"]+tc["batch_size"]]
        batch = data.batch(ids)
        model.train(); opt.zero_grad(set_to_none=True)
        score,delta = logits(model,batch,residual)
        loss = loss_for(score,delta,batch["target"],candidate["penalty"])
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc["gradient_clip"],error_if_nonfinite=True)
        opt.step()
        state["step"] += 1; state["position"] += len(ids)
        if state["position"] == len(fitting):
            state["epoch"] += 1; state["position"] = 0
        if state["step"] % tc["validation_every_steps"] == 0 or state["step"] == maximum:
            if phase == "tune": assess_inner()
            else: json_write(root/"progress.json",{**item,"updated_utc":now(),"step":state["step"],"fixed_steps":fixed_steps})
            checkpoint()
    checkpoint()
    terminal_sha = model_weights_sha(model)
    if phase == "tune":
        chosen = torch.load(out/"best.pt",map_location=device,weights_only=False)
        model.load_state_dict(chosen["model"])
        lp = predictions(model,data,validation,residual,tc["evaluation_batch_size"])
        val_metrics,val_rows = metrics(lp,data.y[validation],data.qids[validation])
        np.testing.assert_allclose(val_metrics["nll"],chosen["validation"]["nll"],atol=1e-7,rtol=1e-6)
        np.save(out/"validation_logp.npy",lp,allow_pickle=False)
        selected_step = chosen["step"]
    else:
        selected_step = state["step"]
        val_metrics,val_rows = None,[]
        save_checkpoint(out/"best.pt",{"identity":identity,"model":model.state_dict(),"step":selected_step})
    train_lp = predictions(model,data,fitting,residual,tc["evaluation_batch_size"])
    train_metrics,train_rows = metrics(train_lp,data.y[fitting],data.qids[fitting])
    np.save(out/"train_logp.npy",train_lp,allow_pickle=False)
    json_write(out/"by_question.json",{"train":train_rows,"validation":val_rows})
    result = {**base,"status":"complete","completed_utc":now(),"steps":state["step"],
        "selected_step":selected_step,"selected_weights_sha256":model_weights_sha(model),
        "terminal_weights_sha256":terminal_sha,"train":train_metrics,"validation":val_metrics,
        "seconds":prior_seconds+time.monotonic()-started,"stopping_reason":reason,
        "outer_outcomes_scored":False,
        "artifacts":{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}}
    json_write(out/"manifest.json",result)
    print(json.dumps({"completed":str(out.relative_to(root)),"steps":state["step"],
        "selected_step":selected_step,"inner_nll":None if val_metrics is None else val_metrics["nll"],
        "seconds":result["seconds"]}),flush=True)
    return out,result


def run_stage(cfg, phase, device):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    common = dict(np.load(root/"development.npz",allow_pickle=False))
    if phase == "tune":
        cohort = protocol["tuning_cohort"]
    else:
        selected = json.loads((root/"selection.json").read_text())
        assert selected["protocol_sha256"] == digest(root/"protocol.json")
        assert selected["tuning_complete_sha256"] == digest(root/"tune_complete.json")
        cohort = [{**entry,"phase":"refit","seed":seed} for entry in selected["choices"]
                  for seed in cfg["training"]["refit_seeds"]]
    timer = session(root,cfg)
    cache_key,data,completed = None,None,{}
    for item in cohort:
        fold = next(f for f in protocol["folds"] if f["id"] == item["fold"])
        if cache_key != fold["id"]:
            if data is not None: del data
            features = dict(np.load(root/"folds"/fold["id"]/"features.npz",allow_pickle=False))
            data = FoldData(common,features,device); cache_key = fold["id"]
        out,m = train_one(cfg,protocol,data,fold,item,device,timer["deadline"],item.get("refit_steps"))
        if m["status"] != "complete":
            print(json.dumps({"bounded":True,"phase":phase}),flush=True); return
        completed[str(out.relative_to(root))] = digest(out/"manifest.json")
    expected = 216 if phase == "tune" else 36
    assert len(completed) == expected
    json_write(root/f"{phase}_complete.json",{"completed_utc":now(),"fits":expected,
        "protocol_sha256":digest(root/"protocol.json"),"manifests":completed,"environment":environment()})


def completed_runs(root, phase):
    done = json.loads((root/f"{phase}_complete.json").read_text())
    assert done["protocol_sha256"] == digest(root/"protocol.json")
    result = []
    for name,sha in done["manifests"].items():
        path = root/name
        assert digest(path/"manifest.json") == sha
        m = json.loads((path/"manifest.json").read_text())
        assert m["status"] == "complete" and not m["outer_outcomes_scored"]
        for file,sha2 in m["artifacts"].items(): assert digest(path/file) == sha2
        result.append((path,m))
    assert len(result) == done["fits"]
    return result


def select(cfg):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    runs = completed_runs(root,"tune")
    assert len(runs) == 216
    choices,details = [],[]
    for outer in range(cfg["split"]["outer_folds"]):
        for kind in cfg["models"]:
            options = []
            for c in protocol["candidates"]:
                matched = [(p,m) for p,m in runs if m["outer"] == outer and m["model"] == kind and m["candidate"] == c["id"]]
                assert len(matched) == 3 and len({m["fold"] for _,m in matched}) == 3
                mean = float(np.mean([m["validation"]["nll"] for _,m in matched]))
                steps = [m["selected_step"] for _,m in matched]
                row = {"outer":outer,"model":kind,"candidate":c["id"],"inner_mean_nll":mean,
                       "inner_selected_steps":steps,"runs":[str(p.relative_to(root)) for p,_ in matched]}
                options.append(row); details.append(row)
            chosen = min(options,key=lambda x:(x["inner_mean_nll"],x["candidate"]))
            # 15/10 scales inner optimizer steps to the same median data passes.
            duration = int(math.ceil(float(np.median(chosen["inner_selected_steps"]))*15/10))
            choices.append({"fold":f"o{outer}_refit","outer":outer,"model":kind,
                            "candidate":chosen["candidate"],"refit_steps":duration})
    payload = {"protocol_sha256":digest(root/"protocol.json"),"tuning_complete_sha256":digest(root/"tune_complete.json"),
               "choices":choices,"all_candidates":details,"selection_uses_outer_outcomes":False}
    if (root/"selection.json").exists():
        old = json.loads((root/"selection.json").read_text())
        assert all(old[k] == v for k,v in payload.items())
    else:
        json_write(root/"selection.json",{**payload,"frozen_utc":now()})
    print(json.dumps({"frozen_choices":choices},indent=2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","tune","select","refit"))
    parser.add_argument("--device",choices=("mps","cpu"),default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage == "prepare": prepare(cfg)
    elif args.stage == "select": select(cfg)
    else: run_stage(cfg,args.stage,args.device)


if __name__ == "__main__": main()
