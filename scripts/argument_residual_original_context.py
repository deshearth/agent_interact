"""Repeat the centroid/residual mechanism check with the original PCA3 predictor.

This addresses the earlier 0.54-point familiar-question text benefit in that exact
architecture, reusing the new fitting-only features without modifying their study.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import argument_model
from scripts import argument_mechanisms as previous
from scripts import argument_residual_diagnostic as features
from scripts import joint_argument_context as driver
from scripts import joint_argument_encoding as session

ROOT = session.SESSION/"residual_original_context"
CONFIG = "configs/argument_residual_original_context.yaml"
VARIANTS = tuple("pca3_"+kind for kind in features.VARIANTS)


def implementation():
    return {**previous.implementation(), "residual_feature_code_sha256": digest(features.__file__),
            "residual_original_context_sha256": digest(__file__), "generic_driver_sha256": digest(driver.__file__)}


def make_model(cfg, kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    return argument_model.ResponseUpdate(kind="A2", text_dim=cfg["data"]["encoder_dimension"],
        **{k:v for k,v in cfg["model"].items() if k != "max_parameters"})


def prepare(cfg):
    source = features.prepare(config(features.CONFIG))
    parent = json.loads((previous.ROOT/"protocol.json").read_text())
    assert parent["implementation"] == previous.implementation()
    for field in ("data", "model", "training", "evaluation", "diagnostic"):
        assert cfg[field] == parent["config"][field]
    assert cfg["local_context"] == "original_pca3"
    identity = {"implementation": implementation(), "config": cfg, "config_sha256": digest(CONFIG),
        "feature_protocol_sha256": digest(features.ROOT/"protocol.json"),
        "parent_protocol_sha256": digest(previous.ROOT/"protocol.json")}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == v for k,v in identity.items())
    else:
        json_write(path, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Retest the earlier familiar-question text gain using only shared content versus isolated individual variation, in its original 185890-parameter architecture.",
            "features": "Reuse the immutable fitting-only means and normalized residuals from residual_diagnostic. Original context16 is restored by dropping the appended full question vector. No feature fitting is repeated or changed.",
            "comparison": "Each condition versus frozen A2 full-argument and stance/history references in argument_mechanisms_v1; same training, selection and familiar evaluation episodes, three seeds.",
            "scope": "Familiar questions only: no unseen-question centroid is available. Predictive decomposition, not a causal LLM intervention or proof that common arguments lack reasoning."})
    return {"pca3_"+kind: ({**a, "context": a["context"][..., :16]}, index, vec, meta)
            for kind, (a,index,vec,meta) in source.items()}


def evaluate_run(*args, **kwargs):
    old_root, old_factory = features.ROOT, features.make_model
    features.ROOT, features.make_model = ROOT, make_model
    try:
        return features.evaluate_run(*args, **kwargs)
    finally:
        features.ROOT, features.make_model = old_root, old_factory


@contextmanager
def adapted_driver():
    changes = {"CONFIG": CONFIG, "ROOT": ROOT, "VARIANTS": VARIANTS, "CONDITIONS": ("full_arguments",),
               "implementation": implementation, "make_model": make_model, "prepare": prepare, "evaluate_run": evaluate_run}
    before = {name:getattr(driver,name) for name in changes}
    for name,value in changes.items():
        setattr(driver,name,value)
    try:
        yield
    finally:
        for name,value in before.items():
            setattr(driver,name,value)


if __name__ == "__main__":
    with adapted_driver():
        driver.main()
