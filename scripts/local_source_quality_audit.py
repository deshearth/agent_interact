"""Post-hoc input-quality sensitivity checks for the local rationale ablation."""
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts.local_argument_source import ROOT
from scripts.local_rationale_ablation import STANCE


def main():
    source = {r["sample_id"]: r for r in io.read_jsonl(ROOT / "conversations/source_responses.jsonl")}
    inputs = io.read_jsonl(ROOT / "rationale_ablation/inputs.jsonl")
    contrasts = pd.read_csv(ROOT / "rationale_ablation/analysis/paired_contexts.csv").set_index("context_id")
    rows = []
    for row in inputs:
        truncated, article_ambiguities, whole_consistent, prefix_consistent = 0, 0, True, True
        for sender, stance in zip(row["incoming_senders"], row["incoming_stances"]):
            r = source[f"{row['episode']}|round{row['round']}|message|agent{sender}"]
            truncated += r["finish_reason"] == "length"
            matches = list(STANCE.finditer(r["raw_response"]))
            ambiguous = [m for m in matches if m[1].islower() and "option" not in m[0].lower() and "answer" not in m[0].lower()]
            article_ambiguities += len(ambiguous)
            labels = {m[1].upper() for m in matches if m not in ambiguous}
            whole_consistent &= labels == ({"A"} if stance == 1 else {"B"})
            prefix = list(STANCE.finditer(r["raw_response"][:160]))
            safe_prefix = {m[1].upper() for m in prefix if not (m[1].islower() and "option" not in m[0].lower() and "answer" not in m[0].lower())}
            prefix_consistent &= safe_prefix == ({"A"} if stance == 1 else {"B"})
        rows.append({"context_id": row["context_id"], "empty_inbox": row["empty_inbox"],
                     "planned_explicit_stance_screen": row["all_incoming_stances_explicit_and_consistent"],
                     "truncated_incoming_messages": truncated, "possible_article_label_ambiguities": article_ambiguities,
                     "whole_message_explicit_stances_consistent": whole_consistent,
                     "prefix_stances_consistent_after_article_guard": prefix_consistent,
                     "absolute_first_A_probability_change": abs(float(contrasts.loc[row["context_id"], "first_A_change_full_minus_conclusion"])),
                     "dominant_A_B_choice_changed": bool(contrasts.loc[row["context_id"], "first_A_dominant_token_flip"])})
    frame = pd.DataFrame(rows)
    selected = frame[(~frame.empty_inbox) & frame.planned_explicit_stance_screen & (frame.truncated_incoming_messages == 0)
                     & (frame.possible_article_label_ambiguities == 0) & frame.whole_message_explicit_stances_consistent]
    out = ROOT / "quality_audit"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_context.csv", index=False)
    result = {"status": "complete", "analysis_type": "Post-hoc input-quality sensitivity; original primary cohort and result retained.",
              "contexts": len(frame), "contexts_with_truncated_incoming_message": int((frame.truncated_incoming_messages > 0).sum()),
              "contexts_with_possible_article_label_ambiguity": int((frame.possible_article_label_ambiguities > 0).sum()),
              "primary_contexts_failing_prefix_article_guard": int(((~frame.empty_inbox) & frame.planned_explicit_stance_screen & ~frame.prefix_stances_consistent_after_article_guard).sum()),
              "strict_untruncated_consistent_contexts": len(selected),
              "strict_subset_mean_absolute_first_A_change": float(selected.absolute_first_A_probability_change.mean()),
              "strict_subset_dominant_A_B_choice_changes": int(selected.dominant_A_B_choice_changed.sum()),
              "limits": "Regex screens explicit stance mentions; it does not certify mathematical correctness or complete semantic consistency. The subset is chosen using inputs and termination flags, not outcome magnitude, but was examined after seeing the primary effect.",
              "analysis_code_sha256": io.sha(__file__), "source_responses_sha256": io.sha(ROOT / "conversations/source_responses.jsonl"),
              "paired_contexts_sha256": io.sha(ROOT / "rationale_ablation/analysis/paired_contexts.csv")}
    io.write_json(out / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
