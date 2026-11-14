"""Combine frozen-run diagnostics and the single scale control, development only."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from mac_agents.metrics import paired_bootstrap,score
from scripts.argument_response import CONFIG,ROOT
from scripts.argument_scale_diagnostic import ROOT as SCALE_ROOT,runtime
from scripts import argument_response as study


def ci(record,percent=False):
    scale=100 if percent else 1
    places=2 if percent else 5
    return f"{record['difference']*scale:+.{places}f} [{record['ci95'][0]*scale:+.{places}f}, {record['ci95'][1]*scale:+.{places}f}]"


def main():
    cfg=config(CONFIG)
    integrity=json.loads(Path("reports/argument_diagnostic_integrity.json").read_text())
    if not integrity["frozen_run_artifacts_unchanged"] or not integrity["gradient_paths_nonzero"]:
        raise ValueError("Incomplete checkpoint audit")
    plan=json.loads((SCALE_ROOT/"plan.json").read_text())
    if not plan.get("complete") or len(plan["runs"])!=6:
        raise ValueError("Complete the six scale-control fits before reporting")
    protocol=json.loads((SCALE_ROOT/"protocol.json").read_text())
    if plan["protocol_sha256"]!=digest(SCALE_ROOT/"protocol.json"):
        raise ValueError("Scale protocol changed")
    with runtime():
        expected_scaled_implementation=study.implementation_digest()
    model_rows,question_rows,frames=[],[],{}
    reference=None
    keys=["episode_id","round","agent"]
    run_hashes={}
    for scale,root,conditions in ((1,ROOT,("stance_only","full_arguments","shuffled_arguments")),
                                 (32,SCALE_ROOT,("full_arguments","shuffled_arguments"))):
        for seed in (42,43,44):
            for condition in conditions:
                run=root/f"A2-{condition}-s{seed}"
                meta=json.loads((run/"manifest.json").read_text())
                expected=(study.implementation_digest() if scale==1 else expected_scaled_implementation)
                if meta["implementation_sha256"]!=expected or meta["status"]!="complete" or meta["smoke"]:
                    raise ValueError("Source run implementation mismatch")
                for name,expected_sha in meta["artifact_sha256"].items():
                    if digest(run/name)!=expected_sha:raise ValueError("Source artifact changed")
                if scale==1:
                    frozen=protocol["baselines"][str(run)]
                    for name,path in (("manifest_sha256",run/"manifest.json"),("best_sha256",run/"best.pt"),
                                      ("prediction_sha256",run/"validation_predictions.parquet")):
                        if digest(path)!=frozen[name]:raise ValueError("Original baseline changed during scale test")
                frame=pd.read_parquet(run/"validation_predictions.parquet").sort_values(keys).reset_index(drop=True)
                truth=frame[keys+["qid","target","current_state"]]
                if reference is None:reference=truth
                else:pd.testing.assert_frame_equal(reference,truth)
                metrics,q=score(frame)
                saved=json.loads((run/"validation_metrics.json").read_text())
                for metric in ("nll","accuracy","brier"):
                    np.testing.assert_allclose(metrics[metric],saved[metric],atol=1e-12,rtol=0)
                model_rows.append({"scale":scale,"condition":condition,"seed":seed,**metrics})
                question_rows.extend(q.assign(scale=scale,condition=condition,seed=seed).to_dict("records"))
                frames[(scale,condition,seed)]=frame
                run_hashes[str(run)]={"manifest_sha256":digest(run/"manifest.json"),**meta["artifact_sha256"]}
    models=pd.DataFrame(model_rows)
    qframe=pd.DataFrame(question_rows)
    models.to_csv("reports/argument_scale_runs.csv",index=False)
    qframe.to_csv("reports/argument_scale_by_question.csv",index=False)
    means=qframe.groupby(["scale","condition","qid"])[["nll","accuracy"]].mean()
    comparisons=[]
    for first,second in (((32,"full_arguments"),(1,"full_arguments")),
                         ((32,"full_arguments"),(1,"stance_only")),
                         ((32,"full_arguments"),(32,"shuffled_arguments"))):
        for metric in ("nll","accuracy"):
            paired=paired_bootstrap(means.loc[first,metric],means.loc[second,metric])
            comparisons.append({"first_scale":first[0],"first":first[1],"second_scale":second[0],"second":second[1],
                                "metric":metric,**paired})
    json_write("reports/argument_scale_paired.json",comparisons)
    # Vote agreement is an outcome diagnostic only. It is never passed to a model.
    source=Path(cfg["data"]["prepared"])
    all_index=json.loads((source/"index.json").read_text())
    lookup={row["episode_id"]:i for i,row in enumerate(all_index)}
    with np.load(source/"episodes.npz",allow_pickle=False) as stored:
        votes=stored["votes"]
    vote_rows=[]
    for condition in ("stance_only","full_arguments"):
        for seed in (42,43,44):
            frame=frames[(1,condition,seed)].copy()
            e=frame.episode_id.map(lookup).to_numpy()
            sampled=votes[e,frame["round"].to_numpy()+1,frame.agent.to_numpy()]
            assert np.isin(sampled,[-1,1]).all()
            count=(sampled==1).sum(-1)
            np.testing.assert_array_equal(np.where(count>=3,1,-1),frame.target)
            frame["majority_count"]=np.maximum(count,5-count)
            frame["wrong"]=frame.prediction!=frame.target
            all_errors=int(frame.wrong.sum())
            for majority,group in frame.groupby("majority_count"):
                vote_rows.append({"condition":condition,"seed":seed,"majority_count":int(majority),
                                  "transitions":len(group),"fraction_transitions":len(group)/len(frame),
                                  "accuracy":float(1-group.wrong.mean()),
                                  "fraction_of_errors":float(group.wrong.sum()/all_errors)})
    vote_frame=pd.DataFrame(vote_rows)
    vote_frame.to_csv("reports/argument_diagnostic_vote_agreement.csv",index=False)
    scores=pd.read_csv("reports/argument_diagnostic_scores.csv")
    average=scores.groupby(["condition","stage","partition"])[["nll","accuracy"]].mean()
    branches=pd.read_csv("reports/argument_diagnostic_branch_scales.csv")
    branch_means=branches.groupby(["stage","feature"])["mean"].mean()
    sensitivity=pd.read_csv("reports/argument_diagnostic_sensitivity.csv")
    sensitivity_means=sensitivity.groupby(["partition","perturbation"])[
        ["nll_change","accuracy_change","mean_absolute_probability_change","fraction_choices_changed"]].mean()
    geometry=pd.read_csv("reports/argument_diagnostic_embedding_geometry.csv")
    results=models.groupby(["scale","condition"])[["nll","accuracy"]].mean()
    primary={r["metric"]:r for r in comparisons if r["second_scale"]==1 and r["second"]=="full_arguments"}
    comparison_text=("The larger text scale did not improve mean validation NLL or accuracy over the original full-text model. Simply increasing text magnitude did not rescue performance at this tested scale; other scaling or optimization choices remain untested."
                     if primary["nll"]["difference"]>=0 and primary["accuracy"]["difference"]<=0 else
                     "The scale change altered validation performance. The table and paired intervals below quantify whether it improved each metric; this post-hoc control does not provide fresh confirmation.")
    text=["# Investigating the argument-response transformer", "",
          "**Yes, the small graph transformer was trained.** Its 185,890 parameters include the message projection, attention query/key/value matrices, signed aggregation and output layers. AdamW updated them from random initialization. The 22,713,216-parameter MiniLM text encoder remained frozen; this experiment learned how to combine fixed message vectors, without fine-tuning token-level argument representations.","",
          "All analyses here are post-hoc and use the existing 16 fitting and four validation questions. No test set or source-LLM intervention is involved. The original experiment and its checkpoints remain unchanged.","",
          "## What the checkpoints establish", "",
          "| Seed | Training epochs completed | Optimizer updates | Selected epoch |", "| ---: | ---: | ---: | ---: |"]
    for seed in (42,43,44):
        meta=json.loads((ROOT/f"A2-full_arguments-s{seed}"/"manifest.json").read_text())
        text.append(f"| {seed} | {meta['epochs']} | {meta['steps']} | {meta['best_step']/128:g} |")
    text += ["", "Every full-text run has optimizer state for all 28 parameter tensors. Text-projection and Q/K/V weights changed, and a representative-batch gradient check found finite, nonzero gradients through each branch. The trained full-text model improves validation accuracy from 47.14% at initialization to 89.33% at the selected checkpoint. A disconnected or untrained argument branch does not explain the result.","",
             "## Continued fitting overfits the development questions", "",
             "Values average the same three seeds. Training scores are reconstructed in evaluation mode from the saved checkpoints; they were not recorded as full training curves during the original run.","",
             "| Input | Checkpoint | Train NLL | Validation NLL | Train accuracy | Validation accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for condition,label in (("stance_only","Stance + history"),("full_arguments","Full argument text")):
        for stage,name in (("initial","Initialization"),("selected","Best validation"),("terminal","End of training")):
            train=average.loc[(condition,stage,"train")];val=average.loc[(condition,stage,"validation")]
            text.append(f"| {label} | {name} | {train.nll:.5f} | {val.nll:.5f} | {100*train.accuracy:.2f}% | {100*val.accuracy:.2f}% |")
    text += ["", "The full-text model keeps improving its training fit while validation NLL worsens. Its validation accuracy also falls after the selected checkpoint. The stance-only model's later NLL worsens even though its thresholded accuracy rises, illustrating that calibration and classification accuracy are different. Checkpoint selection was specified by NLL before these comparisons.","",
             "There are 131,072 fitting transitions but only 16 distinct training questions. The train/validation gap is consistent with limited cross-question generalization. These data do not identify whether the particular cause is the representation, model capacity, optimizer, or limited question diversity.","",
             "![Training and validation diagnostics](argument_diagnostics.png)","",
             "## The text input is small, but its branch learns", "",
             "The model adds a projected unit-normalized text vector to the normalized sender state and a stance projection before normalizing the combined edge representation. Mean text/base vector-norm ratios on validation messages are:","",
             "| Checkpoint | Text/base norm ratio |", "| --- | ---: |"]
    for stage in ("initial","selected","terminal"):
        text.append(f"| {stage} | {100*branch_means.loc[(stage,'text_to_base_norm_ratio')]:.2f}% |")
    text += ["", "This measured scale imbalance is a plausible optimization limitation, not proof that it caused the failed comparison. The text projection's terminal weight change is 1.19–1.32 times its initial weight norm across seeds, so the text branch was actively updated despite the small activation scale.","",
             "### Controlled rescaling", "",
             "A separate six-fit diagnostic multiplies only the message vector by 32, bringing its initial contribution to the same order of magnitude as the other edge features. The multiplier was chosen from the measured 2.75% initial ratio, without a scale sweep. Model size, initial weights, data, batch order, optimizer and stopping rule are unchanged. The zero-text stance-only condition is exactly identical and its frozen runs are reused.","",
             comparison_text,"",
             "| Text scale | Input | Validation NLL | Validation accuracy |", "| ---: | --- | ---: | ---: |"]
    for (scale,condition),r in results.iterrows():
        text.append(f"| {scale} | {condition} | {r.nll:.5f} | {100*r.accuracy:.2f}% |")
    text += ["", "| First minus second | NLL difference [95% interval] | Accuracy difference, points [95% interval] |",
             "| --- | ---: | ---: |"]
    for first_scale,first,second_scale,second in ((32,"full_arguments",1,"full_arguments"),
                                                (32,"full_arguments",1,"stance_only"),
                                                (32,"full_arguments",32,"shuffled_arguments")):
        pair={r["metric"]:r for r in comparisons if (r["first_scale"],r["first"],r["second_scale"],r["second"])==
              (first_scale,first,second_scale,second)}
        text.append(f"| {first_scale}× {first} − {second_scale}× {second} | {ci(pair['nll'])} | {ci(pair['accuracy'],True)} |")
    text += ["", "Intervals average seeds within question and use 1,000 paired question bootstrap draws. With four reused validation questions and stopping selected on their NLL, these intervals are descriptive and pointwise. No confirmatory superiority or equivalence claim follows.","",
             "## What information does the original text branch use?", "",
             "| Partition | Intervention on the trained model | NLL change | Accuracy change, points | Mean absolute probability change, points | Choices changed |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for (partition,perturbation),r in sensitivity_means.iterrows():
        text.append(f"| {partition} | {perturbation} | {r.nll_change:+.5f} | {100*r.accuracy_change:+.2f} | {100*r.mean_absolute_probability_change:.3f} | {100*r.fraction_choices_changed:.3f}% |")
    text += ["", "Deleting the entire text input harms the training fit but slightly improves validation on average. That is consistent with text features that fit the training questions without transferring reliably. Deletion is an input distribution shift; the separately trained stance-only control remains the fair performance comparator.","",
             "Matched argument swaps alter predictions much less. They retain question, communicated stance and source/receiver cues, and may replace a reason with a paraphrase rather than a genuinely different argument. Low sensitivity therefore cannot establish that the source LLM ignores reasoning.","",
             "### Message and embedding inspection", "",
             "The following input-only geometry audit samples 20,000 different-text matched pairs per partition, using seed 48216; it uses no target labels.","",
             "| Partition | Mean cosine similarity | Median | 5th–95th percentiles |", "| --- | ---: | ---: | ---: |"]
    for r in geometry.itertuples():
        text.append(f"| {r.partition} | {r.cosine_mean:.3f} | {r.cosine_median:.3f} | {r.cosine_p05:.3f}–{r.cosine_p95:.3f} |")
    text += ["", "Inspection of pairs selected at fixed cosine quantiles found both paraphrases and reasoning differences. For example, two page-number arguments both use n(n+1)=18,360, n=135 and the sum 271; exchanging them changes wording but retains the mathematical argument. A faucet pair with cosine 0.947 gives the same answer, while one computes a per-faucet flow rate and the other incorrectly describes the new filling time as half of the original. The actual ratio is one eighth: one quarter of the volume at twice the flow rate.","",
             "High cosine similarity is not proof that a downstream model cannot distinguish those claims, and lower similarity is not proof of different reasoning. This small qualitative sample identifies limitations of the perturbation and encoder diagnostics; it is not a corpus-wide quality annotation. The sampled raw pairs are saved in `runs/argument_diagnostics_v1/argument_pair_examples.json`.","",
             "The predictor receives the question/persona context through the earlier compact PCA features (three question components, plus persona and interaction terms). The generic frozen message encoder is not trained to judge an argument against the full problem and choices. A question-conditioned argument representation remains untested.","",
             "## Target-vote disagreement", "",
             "Each source target aggregates five response samples. This retrospective breakdown uses those target votes only for error analysis, never as predictor inputs. Values pool validation transitions within each vote-agreement group, then average over model seeds.","",
             "| Votes supporting target | Fraction of transitions | Full-text accuracy | Fraction of full-text errors |", "| --- | ---: | ---: | ---: |"]
    vote_means=vote_frame[vote_frame.condition=="full_arguments"].groupby("majority_count")[["fraction_transitions","accuracy","fraction_of_errors"]].mean()
    for majority,r in vote_means.iterrows():
        text.append(f"| {majority}/5 | {100*r.fraction_transitions:.2f}% | {100*r.accuracy:.2f}% | {100*r.fraction_of_errors:.2f}% |")
    text += ["", "Vote disagreement indicates response variability for the same prompt, but five samples per prompt do not identify an irreducible accuracy ceiling. Errors on unanimous targets also need to be explained by representation/generalization rather than attributed automatically to noise.","",
             "## What is established and what remains open", "",
             "The transformer was trained and its argument branch receives gradients. Continued full-text fitting overfits these development questions. The original text contribution is small, and the scale control above tests that specific concern; it does not test all possible ways to represent arguments. The frozen encoder, compact question context and limited variation between matched messages remain plausible limitations. None of these results establishes argument irrelevance or opinion-state sufficiency.","",
             "The next focused comparison should make argument meaning explicit relative to the problem and choices, retaining a matched stance control and genuine reasoning contrasts. Any later confirmatory claim needs new questions/outcomes. Do not tune on the previously consumed test set.","",
             "## Reproduce", "", "```bash", ".venv/bin/python -u scripts/diagnose_argument_response.py --device mps",
             ".venv/bin/python -m pytest tests/test_argument_scale_diagnostic.py",
             ".venv/bin/python -u scripts/argument_scale_diagnostic.py --device mps",
             ".venv/bin/python scripts/report_argument_diagnostics.py", "```", "",
             f"The read-only checkpoint/gradient audit took {integrity['elapsed_seconds']:.1f} seconds. The six scale-control fits recorded {plan['elapsed_seconds']:.1f} seconds. No source-LLM generation or paid API was used. The original model/feature/source hashes were verified before and after analysis; the scale control uses a separate run namespace and adapter hash. Two focused tests verify its isolated namespace and text-only transform."]
    Path("reports/argument_training_diagnosis.md").write_text("\n".join(text)+"\n")
    audit={"original_diagnostic":integrity,"scale_protocol_sha256":digest(SCALE_ROOT/"protocol.json"),
           "scale_plan_sha256":digest(SCALE_ROOT/"plan.json"),"source_runs":run_hashes,
           "all_scores_recomputed":True,"all_prediction_keys_targets_identical":True,
           "target_vote_majorities_verified":True,"development_only":True}
    json_write("reports/argument_training_diagnosis_integrity.json",audit)
    print(json.dumps({"report":"reports/argument_training_diagnosis.md","scale_control":primary,
                      "vote_agreement":vote_means.reset_index().to_dict("records")},indent=2))


if __name__=="__main__":main()
