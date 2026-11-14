"""Train-only PCA study with immutable implementation and gated test evaluation.

This entry point is required for its runs, including availability-only controls.
The composite source hash deliberately prevents the ordinary CLI from evaluating
these runs without the declared adapter. Core model/training source is unchanged.
"""
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mac_agents import train as training
from mac_agents import evaluate as evaluation
from mac_agents.common import code_digest as base_code_digest, config, digest, json_write, new_run
from mac_agents.data.features import Snapshots, history_features, partition_indices
from mac_agents.data.prepare import load_prepared, pca_fit_transform

CONFIG = "configs/paper_binary_inductive.yaml"
ROOT = Path("runs/inductive_study")
CONFIGURATIONS = [("A0", "opinion"), ("A1", "opinion"), ("A2", "opinion"),
                  ("A0", "history"), ("A2", "history"),
                  ("A0", "availability_only"), ("A2", "availability_only")]


def implementation():
    return {"core_sha256": base_code_digest(), "driver_sha256": digest(__file__)}


def study_code_digest():
    return hashlib.sha256(json.dumps(implementation(), sort_keys=True).encode()).hexdigest()


def mask_previous_values(extra):
    if extra.shape[-1] != 6:
        raise ValueError("Availability-only control requires a two-snapshot history")
    result = extra.clone() if isinstance(extra, torch.Tensor) else extra.copy()
    result[..., :2] = 0
    return result


class AvailabilitySnapshots(Snapshots):
    def __getitem__(self, item):
        batch = super().__getitem__(item)
        batch["extra"] = mask_previous_values(batch["extra"])
        return batch


def availability_history(spins, t, window):
    return mask_previous_values(history_features(spins, t, window))


@contextmanager
def runtime(condition, allocator=None):
    if condition not in ("opinion", "history", "availability_only"):
        raise ValueError("Unknown study condition")
    previous = [(training, "code_digest", training.code_digest),
                (evaluation, "code_digest", evaluation.code_digest),
                (training, "Snapshots", training.Snapshots),
                (evaluation, "Snapshots", evaluation.Snapshots),
                (evaluation, "history_features", evaluation.history_features),
                (training, "config", training.config),
                (training, "new_run", training.new_run)]

    def study_config(path):
        cfg = config(path)
        cfg["study"] = {"name": "inductive_study", "condition": condition,
                        "implementation": implementation(), "entry_point": "scripts/inductive_study.py"}
        return cfg

    training.code_digest = evaluation.code_digest = study_code_digest
    training.config = study_config
    if allocator is not None:
        training.new_run = allocator
    if condition == "availability_only":
        training.Snapshots = evaluation.Snapshots = AvailabilitySnapshots
        evaluation.history_features = availability_history
    try:
        yield
    finally:
        for module, name, original in previous:
            setattr(module, name, original)


def audit_preprocessing():
    cfg = config(CONFIG)
    arrays, index, directory = load_prepared(cfg)
    old_cfg = config("configs/paper_binary.yaml")
    old, old_index, old_directory = load_prepared(old_cfg)
    split = json.loads((directory / "splits.json").read_text())
    old_split = json.loads((old_directory / "splits.json").read_text())
    if split["questions"] != old_split["questions"] or index != old_index:
        raise ValueError("Example identities or question assignments changed")
    for key in ("spins", "votes", "J", "valid", "message_ok"):
        np.testing.assert_array_equal(arrays[key], old[key])
    source = Path("data/raw/upstream/data/obj/question_embeddings_choices.json")
    items = json.loads(source.read_text())["items"]
    embeddings = {item["qid"]: item["embedding"] for item in items}
    qids = sorted(embeddings)
    values = np.array([embeddings[q] for q in qids], dtype=np.float64)
    fit_indices = [qids.index(q) for q in split["questions"]["train"]]
    transformed, mean, components = pca_fit_transform(values, fit_indices)
    with np.load(directory / "question_pca.npz", allow_pickle=False) as pca:
        np.testing.assert_allclose(mean, pca["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(components, pca["components"], atol=1e-12, rtol=0)
    changed = values.copy()
    heldout = sorted(set(range(len(qids))) - set(fit_indices))
    changed[heldout] *= 1000
    _, other_mean, other_components = pca_fit_transform(changed, fit_indices)
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(components, other_components)
    persona_items = sorted(json.loads(Path("data/raw/upstream/data/obj/persona_embeddings.json").read_text())["items"], key=lambda x:x["idx"])
    persona, _, _ = pca_fit_transform([x["embedding"] for x in persona_items], np.arange(32))
    q_lookup = dict(zip(qids, transformed))
    for e, row in enumerate(index):
        q = q_lookup[row["qid"]]
        expected = np.concatenate([np.ones((32,1)), persona, np.tile(q,(32,1)),
                                   (persona[:,:,None]*q[None,None,:]).reshape(32,9)], axis=-1).astype(np.float32)
        np.testing.assert_allclose(arrays["context"][e], expected, rtol=0, atol=1e-6)
    train = Snapshots(arrays,index,cfg,"train","opinion")
    val = Snapshots(arrays,index,cfg,"validation","opinion")
    if (len(train),len(val)) != (4096,1024):
        raise ValueError("Unexpected development sample counts")
    files = ["episodes.npz","index.json","splits.json","question_pca.npz","persona_pca.npz"]
    return {"context": cfg["data"]["context"], "training_questions": split["questions"]["train"],
            "validation_questions": split["questions"]["validation"],
            "training_only_fit_reproduced": True, "heldout_embedding_perturbation_invariant": True,
            "all_context_arrays_reconstructed": True, "same_examples_targets_and_graphs_as_author_pca": True,
            "train_snapshots": len(train), "validation_snapshots": len(val),
            "test_episodes": len(partition_indices(index,"test")),
            "files_sha256": {name:digest(directory/name) for name in files},
            "raw_question_embeddings_sha256": digest(source)}


def protocol_record(audit):
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": implementation(), "config_sha256": digest(CONFIG),
            "preprocessing": audit,
            "cohort": [{"model":k,"condition":"opinion","seed":42} for k in ("P0","P1","I0")]+[
                {"model":k,"condition":s,"seed":seed} for seed in (42,43,44) for k,s in CONFIGURATIONS],
            "training": "Unchanged architecture, optimizer, caps, and validation early stopping; all declared controls retained",
            "primary_test_role": "test/fit_graph/random",
            "primary_comparison": "A2/opinion minus I0/opinion",
            "primary_metric": "Question-mean NLL; next-opinion accuracy always reported alongside it",
            "secondary_comparisons": ["A2/opinion vs A0/opinion", "A2/opinion vs A1/opinion",
                "A2/history vs A2/opinion", "A0/history vs A0/opinion",
                "A2/history vs A2/availability_only", "A0/history vs A0/availability_only"],
            "reporting": "Keep new-question/familiar-graph, new-question/new-graph, and known-question/new-graph roles separate",
            "uncertainty": "Average metrics over neural seeds within question, then 1000 paired question bootstrap draws; pointwise 95% intervals, no multiplicity-adjusted superiority claims",
            "rollouts": "Eight rounds from s0; deterministic plus 32 common-random-number samples per initialization; compare round-8 accuracy and conditional final-fraction Wasserstein by role",
            "control_transform": "availability_only zeros extra[..., :2] in both observed and generated history, retaining six-feature architecture",
            "budget_seconds": 5400,
            "test_policy": "Freeze every completed declared candidate and checkpoint before test evaluation; no model or preprocessing changes after inspecting test outcomes"}


def validate_run(item):
    path = Path(item["run"])
    meta = json.loads((path / "manifest.json").read_text())
    if meta["code_sha256"] != study_code_digest() or meta["config"]["study"]["condition"] != item["condition"]:
        raise ValueError("Run implementation/condition mismatch")
    if meta["smoke"] or meta["config"]["data"]["context"] != "train_only_pca":
        raise ValueError("Ineligible run")
    if meta["stopping_reason"] == "wall_time_limit":
        raise ValueError("Unfinished run")
    return meta


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation","freeze","test"])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path, plan_path = ROOT/"protocol.json", ROOT/"plan.json"
    audit = audit_preprocessing()
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["implementation"] != implementation() or protocol["config_sha256"] != digest(CONFIG) or protocol["preprocessing"] != audit:
            raise ValueError("Study implementation, config, or preprocessing changed")
    else:
        if args.stage != "validation":
            raise ValueError("Run validation before freezing or testing")
        if list(Path("runs").rglob("test_*metrics.json")):
            raise ValueError("Controlled test outcomes already exist; review reuse before defining a new study")
        protocol = protocol_record(audit)
        json_write(protocol_path, protocol)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["protocol_sha256"] != digest(protocol_path):
            raise ValueError("Protocol changed")
    else:
        plan = {"protocol_sha256": digest(protocol_path), "elapsed_seconds":0,
                "runs":[dict(x,status="pending",validation_rollout=False,test_one_step=False,test_closed_loop=False) for x in protocol["cohort"]]}
    started, prior_seconds = time.monotonic(), plan["elapsed_seconds"]

    def remaining():
        return protocol["budget_seconds"] - prior_seconds - (time.monotonic()-started)

    def save():
        plan["elapsed_seconds"] = prior_seconds + time.monotonic()-started
        json_write(plan_path,plan)

    selection = ROOT/"selection.json"
    frozen_protocol = ROOT/"frozen_protocol.json"
    try:
        if args.stage == "validation":
            if selection.exists():
                raise ValueError("Frozen study cannot be retrained")
            for item in plan["runs"]:
                if item["status"]=="complete":
                    continue
                if remaining()<=0:
                    print("Study budget exhausted; pending training saved",flush=True)
                    return
                if item.get("run") and (Path(item["run"])/"manifest.json").exists():
                    existing=json.loads((Path(item["run"])/"manifest.json").read_text())
                    if existing.get("stopping_reason") in ("early_stopping","epoch_limit","baseline_fitted","step_limit"):
                        validate_run(item)
                        item["status"]="complete"
                        save()
                        continue

                def allocate(label):
                    directory=new_run(label)
                    suffix="-availability_only" if item["condition"]=="availability_only" else ""
                    destination=ROOT/(directory.name+suffix)
                    directory.rename(destination)
                    item["run"]=str(destination)
                    save()
                    return destination

                item["status"]="running"
                save()
                base_state="history" if item["condition"]=="availability_only" else item["condition"]
                resume=Path(item["run"]) if item.get("run") and (Path(item["run"])/"manifest.json").exists() else None
                if resume is not None and not item["model"].startswith("A"):
                    item.setdefault("interrupted_baseline_attempts",[]).append(str(resume))
                    resume=None
                with runtime(item["condition"],allocate):
                    result=training.train(Namespace(config=CONFIG,model=item["model"],state=base_state,
                        seed=item["seed"],device="mps",max_steps=None,wall_minutes=remaining()/60,
                        limit_episodes=None,match_state=None,resume=resume,smoke=False))
                item["run"]=str(result)
                meta=json.loads((result/"manifest.json").read_text())
                item["status"]="pending" if meta["stopping_reason"]=="wall_time_limit" else "complete"
                save()
                if item["status"]!="complete":
                    return
            for item in plan["runs"]:
                if item["validation_rollout"]:
                    continue
                if remaining()<=0:
                    print("Study budget exhausted; pending validation rollouts saved",flush=True)
                    return
                validate_run(item)
                with runtime(item["condition"]):
                    evaluation.evaluate(Namespace(run=Path(item["run"]),partition="validation",mode="closed_loop",selection=None,rollouts=32))
                item["validation_rollout"]=True
                save()
            plan["validation_completed_utc"]=datetime.now(timezone.utc).isoformat()
            print("Inductive validation complete; controlled test remains locked",flush=True)
        elif args.stage=="freeze":
            if not all(x["status"]=="complete" and x["validation_rollout"] for x in plan["runs"]):
                raise ValueError("Complete the declared validation cohort before freezing")
            for item in plan["runs"]:
                validate_run(item)
            if not selection.exists():
                with runtime("opinion"):
                    evaluation.freeze([Path(x["run"]) for x in plan["runs"]],selection,
                        "Training-only PCA audited; retain all predeclared baseline, routing, history, and availability controls at all seeds. No test outcomes used. Primary/secondary comparisons are defined in protocol.json.")
            for item in plan["runs"]:
                evaluation.check_selection(item["run"],selection)
            record={"protocol_sha256":digest(protocol_path),"selection_sha256":digest(selection),
                    "implementation":implementation(),"preprocessing":audit}
            if frozen_protocol.exists() and json.loads(frozen_protocol.read_text())!=record:
                raise ValueError("Frozen protocol changed")
            if not frozen_protocol.exists():
                json_write(frozen_protocol,record)
            plan["frozen_utc"]=json.loads(selection.read_text())["frozen_utc"]
            print("All 24 candidates frozen before controlled test evaluation",flush=True)
        else:
            if not frozen_protocol.exists():
                raise ValueError("Test evaluation requires the explicit freeze stage")
            record=json.loads(frozen_protocol.read_text())
            if record!={"protocol_sha256":digest(protocol_path),"selection_sha256":digest(selection),"implementation":implementation(),"preprocessing":audit}:
                raise ValueError("Frozen protocol or selection changed")
            for item in plan["runs"]:
                validate_run(item)
                for mode in ("one_step","closed_loop"):
                    key="test_"+mode
                    if item[key]:
                        continue
                    if remaining()<=0:
                        print("Study budget exhausted; pending frozen test evaluations saved",flush=True)
                        return
                    with runtime(item["condition"]):
                        evaluation.evaluate(Namespace(run=Path(item["run"]),partition="test",mode=mode,
                                                      selection=selection,rollouts=None))
                    item[key]=True
                    save()
            plan["test_completed_utc"]=datetime.now(timezone.utc).isoformat()
            print("Frozen inductive test evaluation complete; no retraining or reselection",flush=True)
    finally:
        save()


if __name__=="__main__":
    main()
