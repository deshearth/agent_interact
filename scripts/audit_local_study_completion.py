"""Final bounded-source-study inventory and frozen-artifact integrity check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts.local_argument_source import ROOT


def main():
    session = json.loads((ROOT / "session.json").read_text())
    start = datetime.fromisoformat(session["parent_investigation_start_utc"])
    deadline = datetime.fromisoformat(session["deadline_utc"])
    stages = {}
    for name in ("intervention", "conversations", "rationale_ablation", "direct_argument_grading", "word_argument_grading"):
        root = ROOT / name
        status = json.loads((root / "collection_status.json").read_text())
        assert status["status"] == "complete"
        assert start <= datetime.fromisoformat(status["completed_utc"]) <= deadline
        assert status["paid_api_calls"] == 0
        for file, digest in status["artifacts"].items():
            assert io.sha(root / file) == digest, (name, file)
        analysis = json.loads((root / "analysis/audit.json").read_text())
        assert analysis["status"] == "complete" and analysis["analysis_code_sha256"] == io.sha("scripts/analyze_local_source.py")
        stages[name] = {"collection_sha256": io.sha(root / "collection_status.json"),
                       "analysis_sha256": io.sha(root / "analysis/audit.json"), "completed_utc": status["completed_utc"]}
    previous = json.loads(Path("runs/argument_investigation_20261108/completion_audit.json").read_text())
    h = hashlib.sha256()
    for p in sorted(Path("src/mac_agents").rglob("*.py")):
        h.update(str(p).encode()); h.update(p.read_bytes())
    assert h.hexdigest() == previous["frozen_core_digest"]
    for path, digest in previous["frozen_files"].items():
        assert io.sha(path) == digest
    for directory, digest in previous["run_manifests"].items():
        path = Path(directory) / "manifest.json"
        assert io.sha(path) == digest
        manifest = json.loads(Path(path).read_text())
        for file, artifact_sha in manifest["artifact_sha256"].items():
            assert io.sha(Path(path).parent / file) == artifact_sha
    checked = {}
    for file, count in (("reports/local_source_checks.xml", 14), ("reports/argument_intervention_checks.xml", 16),
                        ("reports/argument_intervention_parent_checks.xml", 5)):
        xml = ET.parse(file).getroot()
        assert len(xml.findall(".//testcase")) == count
        assert not any(xml.findall(".//" + tag) for tag in ("failure", "error", "skipped"))
        checked[file] = {"passed": count, "sha256": io.sha(file)}
    teacher = json.loads((ROOT / "teacher_pairs/manifest.json").read_text())
    assert teacher["status"] == "complete" and teacher["paired_contexts"] == 512
    for file, digest in teacher["artifacts"].items():
        assert io.sha(ROOT / "teacher_pairs" / file) == digest
    quality = json.loads((ROOT / "quality_audit/audit.json").read_text())
    assert quality["status"] == "complete" and quality["primary_contexts_failing_prefix_article_guard"] == 0
    assert quality["analysis_code_sha256"] == io.sha("scripts/local_source_quality_audit.py")
    assert quality["paired_contexts_sha256"] == io.sha(ROOT / "rationale_ablation/analysis/paired_contexts.csv")
    probe = json.loads(Path("runs/argument_intervention_v1/predictor_probe/complete.json").read_text())
    assert probe["prediction_rows"] == 864 and probe["optimizer_steps"] == probe["counterfactual_accuracy_evaluations"] == 0
    for file, digest in probe["artifacts"].items():
        assert io.sha(Path("runs/argument_intervention_v1/predictor_probe") / file) == digest
    for file in ("reports/local_open_model_pilot.md", "reports/local_open_model_pilot.png", "reports/local_open_model_pilot.pdf"):
        assert Path(file).stat().st_size > 1000
    now = datetime.now(timezone.utc)
    assert now <= deadline, "Final audit must finish within the current authorized bound"
    result = {"status": "complete", "started_utc": start.isoformat(), "completed_utc": now.isoformat(),
              "deadline_utc": deadline.isoformat(), "elapsed_minutes": (now-start).total_seconds()/60,
              "paid_api_calls": 0, "local_source_responses_including_benchmark": 12 + 1920 + 3840 + 496 + 2560,
              "new_complete_conversations": 8, "new_agent_transitions": 512, "paired_rationale_ablation_contexts": 512,
              "single_message_intervention_contexts": 24, "new_predictor_fits": 0, "new_test_scores": 0,
              "direct_validity_grading_prompts": 72,
              "word_validity_grading_prompts": 72,
              "frozen_core_sha256": h.hexdigest(), "frozen_previous_fit_artifacts_verified": len(previous["run_manifests"]),
              "stages": stages, "checks": checked, "audit_code_sha256": io.sha(__file__),
              "report_artifacts": {p: io.sha(p) for p in ("reports/local_open_model_pilot.md", "reports/local_open_model_pilot.png", "reports/local_open_model_pilot.pdf")},
              "teacher_pairs_manifest_sha256": io.sha(ROOT / "teacher_pairs/manifest.json")}
    result["posthoc_quality_audit_sha256"] = io.sha(ROOT / "quality_audit/audit.json")
    # Count actual requests rather than silently assuming no parse retry occurred.
    result["local_source_responses_including_benchmark"] = 12 + sum(len(io.read_jsonl(ROOT / folder / name))
        for folder, name in (("intervention", "responses.jsonl"), ("conversations", "source_responses.jsonl"),
                             ("rationale_ablation", "source_responses.jsonl"), ("direct_argument_grading", "source_responses.jsonl"),
                             ("word_argument_grading", "source_responses.jsonl")))
    all_draws = [r for folder, name in (("intervention", "responses.jsonl"), ("conversations", "source_responses.jsonl"),
                                      ("rationale_ablation", "source_responses.jsonl"), ("direct_argument_grading", "source_responses.jsonl"),
                                      ("word_argument_grading", "source_responses.jsonl"))
                 for r in io.read_jsonl(ROOT / folder / name)]
    assert len({r["sample_id"] for r in all_draws}) == len(all_draws)
    assert len({(r["prompt_sha256"], r["sample_seed"]) for r in all_draws}) == len(all_draws), "Repeated seed on the same prompt"
    result["cross_prompt_seed_collisions"] = len(all_draws) - len({r["sample_seed"] for r in all_draws})
    result["all_draw_ids_unique_and_no_same_prompt_seed_reuse"] = True
    io.write_json(ROOT / "completion_audit.json", result)
    io.write_json("reports/local_open_model_integrity.json", result)
    session.update({"status": "complete", "completed_utc": now.isoformat(), "completion_audit_sha256": io.sha(ROOT / "completion_audit.json")})
    io.write_json(ROOT / "session.json", session)
    print(json.dumps({k: v for k, v in result.items() if k not in ("stages", "checks", "report_artifacts")}, indent=2))


if __name__ == "__main__":
    main()
