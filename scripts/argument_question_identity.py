"""Question-ID-only control for the familiar-question argument benefit.

No language representation is supplied to the argument branch: its 384-dimensional
input is a padded one-hot question ID, optionally crossed with the observed stance.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import argument_mechanisms as previous
from scripts import argument_residual_diagnostic as features
from scripts import joint_argument_context as driver
from scripts import joint_argument_encoding as session

ROOT = session.SESSION/"question_identity_diagnostic"
CONFIG = "configs/argument_question_identity.yaml"
VARIANTS = ("question_id", "question_stance_id")


def implementation():
    return {**previous.implementation(), "question_identity_sha256":digest(__file__),
        "id_mapping_source_sha256":digest(features.__file__), "generic_driver_sha256":digest(driver.__file__)}


def make_model(cfg,kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    return previous.make_model(cfg,"A2")


def prepare(cfg):
    caches = features.prepare(config(features.CONFIG))
    parent = json.loads((previous.ROOT/"protocol.json").read_text())
    assert parent["implementation"] == previous.implementation()
    for field in ("data","model","training","evaluation","diagnostic"):
        assert cfg[field] == parent["config"][field]
    ROOT.mkdir(parents=True,exist_ok=True)
    identity = {"implementation":implementation(),"config":cfg,"config_sha256":digest(CONFIG),
        "id_mapping_protocol_sha256":digest(features.ROOT/"protocol.json"),
        "parent_protocol_sha256":digest(previous.ROOT/"protocol.json")}
    path = ROOT/"protocol.json"
    if path.exists():
        p = json.loads(path.read_text());assert all(p[k] == v for k,v in identity.items())
        for file,sha in p["prepared_sha256"].items():
            assert digest(ROOT/file) == sha
    else:
        for kind,size in (("question_id",16),("question_stance_id",32)):
            vectors = np.zeros((size+1,384),dtype=np.float32)
            vectors[np.arange(1,size+1),np.arange(size)] = 1.
            np.save(ROOT/f"{kind}.npy",vectors)
        json_write(path,{**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
            "prepared_sha256":{f"{k}.npy":digest(ROOT/f"{k}.npy") for k in VARIANTS},
            "purpose":"Can explicit familiar-question identity reproduce the earlier text benefit without any message language?",
            "inputs":"One-hot IDs for the 16 fitting questions (or 32 question/current-stance combinations), padded to 384 dimensions and delivered at every active incoming edge. Original context16, opinions/history, signs and explicit stance remain unchanged. No text, answer key or next-state target is used to assign IDs.",
            "mapping":"Reuse only the fitting-question ID grids from the centroid diagnostic; discard its actual centroid vectors. Unseen questions have no assigned ID and are not evaluated.",
            "architecture":"Original 185890-parameter A2 model, same initialization/training/stopping. Active text-input rank is 16 or 32, rather than 384; nominal parameters match but effective capacity and geometry do not. Only 2048/4096 text projection weights can receive data gradients.",
            "interpretation":"A positive result would show that language semantics is unnecessary for some familiar-question gain, not that question IDs generalize or that agents ignore reasoning. This is a mechanism control, not an unseen-question model."})
    result = {}
    for kind,source_kind in (("question_id","question_mean"),("question_stance_id","question_stance_mean")):
        a,index,_,meta = caches[source_kind]
        result[kind] = ({**a,"context":a["context"][...,:16]},index,np.load(ROOT/f"{kind}.npy",allow_pickle=False),meta)
    return result


def evaluate_run(*args,**kwargs):
    old_root,old_factory = features.ROOT,features.make_model
    features.ROOT,features.make_model = ROOT,make_model
    try:
        return features.evaluate_run(*args,**kwargs)
    finally:
        features.ROOT,features.make_model = old_root,old_factory


@contextmanager
def adapted_driver():
    changes = {"CONFIG":CONFIG,"ROOT":ROOT,"VARIANTS":VARIANTS,"CONDITIONS":("full_arguments",),
        "implementation":implementation,"make_model":make_model,"prepare":prepare,"evaluate_run":evaluate_run}
    before = {k:getattr(driver,k) for k in changes}
    for k,v in changes.items():setattr(driver,k,v)
    try:
        yield
    finally:
        for k,v in before.items():setattr(driver,k,v)


if __name__ == "__main__":
    with adapted_driver():driver.main()
