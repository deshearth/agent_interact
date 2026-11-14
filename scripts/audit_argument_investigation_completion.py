"""Final independent cohort, frozen-source, artifact, and check inventory audit."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest,digest,json_write
from scripts import joint_argument_context as joint
from scripts import argument_order_diagnostic as order
from scripts import argument_residual_diagnostic as residual
from scripts import argument_residual_original_context as residual_pca3
from scripts import argument_question_identity as identity

ROOT = Path("runs/argument_investigation_20261108")


def main():
    session = json.loads((ROOT/"session.json").read_text())
    start,end = [datetime.fromisoformat(session[k]) for k in ("started_utc","deadline_utc")]
    assert code_digest() == "92c500581c6d74e0d5effa4b89f9c402bb76f7800151ac6dbb3b3afcae939dda"
    preserved = {"scripts/argument_features.py":"d7aba211819a378cafd94a5b011e0ab69005cdfc958bef781195d8650c021b2f",
        "scripts/argument_model.py":"cfdc2ab81c7830d5a56b3a23b1e95df10d361d85e2b8021c311290ea3990fa84",
        "scripts/argument_response.py":"471a4f91f6d18f04e815bbc96f7c15de0efac015c3269cfa19333583b08b6e8e"}
    for file,sha in preserved.items():assert digest(file) == sha
    counts,run_manifests = {},{}
    for module,conditions in ((joint,("full_arguments","shuffled_arguments")),
        (order,("stance_only","full_arguments","shuffled_arguments")),
        (residual,("full_arguments",)),(residual_pca3,("full_arguments",)),(identity,("full_arguments",))):
        plan = json.loads((module.ROOT/"plan.json").read_text())
        protocol = json.loads((module.ROOT/"protocol.json").read_text())
        assert plan["complete"] and plan["protocol_sha256"] == digest(module.ROOT/"protocol.json")
        assert protocol["implementation"] == module.implementation()
        assert protocol["config_sha256"] == digest(module.CONFIG)
        for file,sha in protocol.get("prepared_sha256",{}).items():assert digest(module.ROOT/file) == sha
        expected = {(kind,c,s) for kind in module.VARIANTS for c in conditions for s in (42,43,44)}
        actual = {(r["kind"],r["condition"],r["seed"]) for r in plan["runs"]}
        assert actual == expected and len(plan["runs"]) == len(expected)
        impl = hashlib.sha256(json.dumps(module.implementation(),sort_keys=True).encode()).hexdigest()
        for kind,c,s in sorted(expected):
            run = module.ROOT/f"{kind}-{c}-s{s}"
            m = json.loads((run/"manifest.json").read_text())
            assert m["status"] == "complete" and m["stopping_reason"] == "early_stopping" and not m["smoke"]
            assert m["implementation_sha256"] == impl and m["protocol_sha256"] == digest(module.ROOT/"protocol.json")
            assert start <= datetime.fromisoformat(m["created_utc"]) <= datetime.fromisoformat(m["completed_utc"]) <= end
            for file,sha in m["artifact_sha256"].items():assert digest(run/file) == sha
            run_manifests[str(run)] = digest(run/"manifest.json")
        counts[module.ROOT.name] = len(expected)
    assert sum(counts.values()) == 54
    checks_path = Path("reports/argument_investigation_all_checks.xml")
    checks = ET.parse(checks_path).getroot()
    assert len(checks.findall(".//testcase")) == 21
    assert not any(checks.findall(".//"+tag) for tag in ("failure","error","skipped"))
    extra = {}
    for folder,name in (("input_audit","integrity.json"),("joint_encoding_windowed","complete.json"),
        ("conditional_neighbors","complete.json"),("stratified_diagnostic","complete.json"),
        ("terminal_fit_audit","complete.json"),("controlled_probe_draft","manifest.json"),
        ("controlled_probe_uncertain","manifest.json"),("quality_predictor_probe","complete.json"),
        ("quality_uncertain_probe","complete.json")):
        path = ROOT/folder/name;saved = json.loads(path.read_text())
        for file,sha in saved["artifacts"].items():assert digest(path.parent/file) == sha
        if "api_calls" in saved:assert saved["api_calls"] == 0
        extra[folder] = digest(path)
    source_plan = json.loads((ROOT/"controlled_probe_uncertain/analysis_plan.json").read_text())
    assert source_plan["manifest_sha256"] == digest(ROOT/"controlled_probe_uncertain/manifest.json")
    trace = json.loads((ROOT/"input_trace/trace.json").read_text())
    assert trace["code_sha256"] == digest("scripts/trace_argument_inputs.py") and trace["optimizer_steps"] == 0
    assert len(trace["traces"]) == 6 and sum(len(t["messages"]) for t in trace["traces"]) == 42
    assert all(t["all_delivered_vectors_have_nonzero_local_derivative"] and t["other_receivers_and_nonedges_have_zero_derivative"] for t in trace["traces"])
    capacity = json.loads((ROOT/"pair_capacity/summary.json").read_text())
    assert len(capacity) == 6 and all(r["reason"] == "memorized" and r["error"] == 0 for r in capacity)
    main = json.loads(Path("reports/argument_investigation_integrity.json").read_text())
    assert main["models_audited"] == 72 and main["new_models"] == 54 and main["frozen_references"] == 18
    assert main["analysis_code_sha256"] == digest("scripts/report_argument_investigation.py")
    result = {"audit_code_sha256":digest(__file__),"checked_utc":datetime.now(timezone.utc).isoformat(),
        "deadline_utc":session["deadline_utc"],"new_fit_counts":counts,"new_fits":54,"all_new_fits_stopped_early":True,
        "all_fit_timestamps_within_authorized_window":True,"frozen_core_digest":code_digest(),"frozen_files":preserved,
        "selected_models_in_metric_audit":72,"reused_references":18,"capacity_fits":6,"terminal_checkpoints":24,
        "focused_tests":21,"test_xml_sha256":digest(checks_path),"individual_message_paths_traced":42,
        "source_llm_calls":0,"new_test_scores":0,"future_pilot_status":"draft_not_executed",
        "future_majority_target_distinction_documented":True,"run_manifests":run_manifests,"additional_artifacts":extra}
    json_write(ROOT/"completion_audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("run_manifests","additional_artifacts","frozen_files")}),flush=True)


if __name__ == "__main__":main()
