"""Build the report's content and figures from existing project evidence only."""
from pathlib import Path
import hashlib
import json
import re
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

def read(path):
    return json.loads((ROOT / path).read_text())

curve = read('runs/breadth_320_learning_curve_v1/results/summary.json')
sweep = read('runs/breadth_320_size_sweep_v1/results/summary.json')
audit = read('runs/breadth_320_learning_curve_v1/completion_audit.json')

def p(text): return {'type':'p', 'text':text}
def h(text): return {'type':'h1', 'text':text}
def sub(text): return {'type':'h2', 'text':text}
def table(caption, headers, rows, widths):
    return {'type':'table','caption':caption,'headers':headers,'rows':rows,'widths':widths}
def fig(file, caption, height): return {'type':'figure','file':file,'caption':caption,'height':height}

abstract = '''Can a compact predictor capture how language-model agents influence one another, and can its behavior eventually be expressed as an interpretable rule? We compare signed graph transformers with a reproduced Ising baseline, then diagnose the roles of model capacity, question coverage, argument representation and stochastic responses. Experiments use released GPT-4o-mini trajectories and a separate, newly generated Llama-3.1-8B-Instruct corpus. On 320 training and 80 validation questions, an 818-parameter transformer achieves 84.09% next-opinion accuracy versus 82.82% for the 19-coefficient Ising baseline. Expanding the transformer to 533,026 parameters does not improve accuracy. Increasing training coverage from 40 to 320 questions improves accuracy by 0.29 percentage points for the small model and 0.97 points for the largest, which remains behind. Capacity tests show that substantially better training fit is possible but can worsen generalization. Argument features yield little consistent unseen-question gain; however, controlled Llama message interventions demonstrate substantial text dependence at fixed stated opinions. Simple explicit surrogates also fail to closely reproduce the compact transformer's probabilities. These results support useful compression of conditional opinion updates, while leaving state sufficiency, causal interpretation and reliable autonomous simulation unresolved. Most follow-up comparisons reuse development data; a separate 40-question Llama assessment cohort remains ungenerated and unscored.'''

pages = []
pages.append({'label':'Main report · 1 of 4','blocks':[
 {'type':'title','text':'Compact Models of LLM Agent Interaction','subtitle':'Predictive gains, state sufficiency, and the search for an interpretable update rule'},
 {'type':'date','text':'Research report · 14 November 2026'},
 sub('Abstract'),p(abstract),
 h('1. Introduction'),
 p('Interactions among language-model agents can reinforce errors as well as improve answers. A cheap, inspectable predictor of the next opinion could help identify when agreement is informative and when it merely reflects social influence. The scientific question is whether a compact state admits a transferable update rule. A neural approximation is a test of predictive compressibility; it is not, by itself, evidence for a universal or causal law.'),
 p('Our starting hypothesis was that a more expressive graph transformer should predict updates better than an Ising model. The observed gains were modest, motivating three contributions: <b>(1)</b> a controlled comparison across model sizes and training-question counts, including a new 400-question Llama development corpus; <b>(2)</b> diagnostics separating fitting capacity, transfer, stochastic targets and missing message information; and <b>(3)</b> tests of explicit formulas, aggregate-state models, graph transfer and alternative datasets. The report consolidates the completed studies, retaining their different evaluation roles rather than pooling their scores.'),
 h('2. Related Work'),
 p('<i>Physics of Agents</i> [1] motivates the signed-network setting and the statistical-mechanics baseline. We extend that comparison with neural predictors and tests of what the supplied state omits; we do not replicate every result or source model in that paper. MATH [2] supplies objective problems for the new corpus. MiniLM [3,4] supplies frozen semantic features, and Llama 3 [5] supplies the open-weight source agents. CoevolveSim [6] and DEBATE [7] provide separate external checks. Our focus is predicting agent responses, not solving the underlying questions or improving the source LLM through fine-tuning.')
]})

pages.append({'label':'Main report · 2 of 4','blocks':[
 h('3. Methods'),
 p('<b>Task and boundary.</b> Each primary conversation has 32 fixed personas on a prescribed signed graph. An opinion is the majority of five sampled A/B responses; a transition is one agent’s opinion at the next round. The principal predictor observes current opinions, question/persona context and graph, before the next messages are generated. Argument studies instead observe the delivered messages and predict the subsequent response. These are different information conditions. Accuracy measures agreement with the agent’s next opinion, not mathematical correctness.'),
 fig('pipeline.png','Figure 1. Two forecasting boundaries. Source LLM and text encoder weights stay fixed. Only the compact predictor and training-only preprocessing are fitted; argument experiments additionally observe actual delivered messages.',105),
 table('Table 1. Distinct evidence cohorts. Counts are questions, not independent agent updates.',
 ['Cohort','Development / evaluation','Role'],[
 ['Released GPT-4o-mini','16 fit + 4 validation; 20 original test','Frozen original test, then exploratory development follow-ups'],
 ['Local Llama pilot','8 questions; 512 transitions','Argument and intervention diagnostics'],
 ['Expanded Llama','320 fit + 80 validation; 40 reserved','Main size and learning-curve study'],
 ['Llama fresh graphs','20 reused validation questions; 4 new graphs','Paired graph-instance transfer']],[.24,.38,.38]),
 p('<b>Question selection.</b> The Llama corpus uses rational-answer MATH problems, balanced correct A/B placement, seven topics and eight familiar graphs. Candidates are ranked by pooled initial-response entropy from 32 personas × eight draws; selected questions are the top quarter within topic/answer cells. The expansion retains the original 80/20 development questions and adds 240/60 from 1,200 fresh candidates. Screening draws are separate from conversation draws, and predictor errors do not select questions. Each question has one eight-update conversation, giving 81,920 training and 20,480 validation transitions. Appendix A records selection and source limitations.'),
 p('<b>Predictors.</b> The reproduced Ising implementation stores <b>19 fitted coefficients</b>: 16 context coefficients and three social-channel coefficients, not a separate coefficient for each persona. Transformers share weights across agents and use four attention heads, a signed edge mask and one attention block. The expanded-study context contains three persona components, three question components, nine products and a constant. MiniLM’s 22.7 million pretrained parameters and PCA are outside the quoted neural parameter counts. Arguments are absent from the main scaling study.'),
 p('<b>Comparison and uncertainty.</b> At 40/80/160 questions, three balanced nested subset paths are shared across models; the full 320-question endpoint is reused once. Question PCA is fitted on each training subset. Every size/count receives nine learning-rate/penalty choices, validation-NLL stopping, then three fresh-seed fits at frozen settings. Update budgets scale with data size to preserve maximum passes. We average model scores, not ensemble predictions. Paired 95% bootstrap intervals resample questions after seed/subset averaging; they are exploratory, pointwise and conditional on the fitted models. They do not treat agents or rounds as independent observations.')
]})

size_rows=[]
for kind in ['I0','pico','nano','tiny','small','full','wide']:
    v=sweep['summary'][kind]
    size_rows.append(['Ising' if kind=='I0' else 'Transformer',f"{sweep['parameters'][kind]:,}",
        f"{100*(1-v['train']['error']):.2f}%",f"{100*(1-v['validation']['error']):.2f}%",f"{v['validation']['nll']:.5f}"])
pages.append({'label':'Main report · 3 of 4','blocks':[
 h('4. Results'),sub('4.1 More parameters do not ensure better generalization'),
 table('Table 2. Expanded Llama study: 320 training / 80 validation questions. Neural entries average three fresh seeds; NLL is lower when better. No argument inputs.',
 ['Predictor','Parameters','Train accuracy','Validation accuracy','Validation NLL'],size_rows,[.20,.16,.20,.23,.21]),
 p('The 818-parameter model improves validation accuracy over Ising by <b>1.27 percentage points</b> (pp; descriptive interval 0.72–1.86), with NLL 0.35884 versus 0.38751. None of five smaller/larger alternatives improves its mean score. The 533,026-parameter model is 0.28 pp worse, while 7,370 parameters has nearly identical NLL. This is a comparison within one tuned architecture family, not a proof of an optimal parameter count.'),
 sub('4.2 Additional questions help, especially probability estimates'),
 fig('learning_curve_main.png','Figure 2. Same 80 validation questions at every training count. Lines average seeds and subset paths; shading spans subset-path means, not confidence intervals. The 320-question endpoint is shared once. All models receive the same tuning policy.',160),
 p('From 40 to 320 questions, accuracy increases by <b>0.29 pp</b> for 818 parameters (0.10–0.50), <b>0.21 pp</b> for 7,370 (−0.01–0.43), and <b>0.97 pp</b> for 533,026 (0.70–1.26). The largest model gains 0.68 pp more than the smallest (0.33–1.03), mainly recovering its small-data deficit. Ising stays near 82.8%. From 160 to 320, all three accuracy-change intervals include zero; the two larger models nevertheless improve NLL by approximately 0.0044 and 0.0043. No inference about saturation or success beyond 320 follows.'),
 sub('4.3 Fitting capacity is not the main demonstrated bottleneck'),
 p('On the original GPT cohort, a 135,458-parameter transformer imitates Ising probabilities to 0.022 pp training MAE. Training without validation stopping lowers observed-label training error from 8.14% to 4.41%, while validation error rises from 10.59% to 12.22% and NLL worsens. The expanded Llama sweep likewise shows better late training fit with worse validation loss. Similar validation-selected training errors therefore do not establish equal representational capacity (Appendix B).')
]})

pages.append({'label':'Main report · 4 of 4','blocks':[
 sub('4.4 Arguments matter to the source, but current features transfer weakly'),
 p('The GPT argument transformer processes individual delivered messages through a frozen encoder. Yet its familiar-question error reduction, 6.47%→5.93%, can be reproduced by question IDs alone (5.92%). Across unseen questions, richer question context, alternate fusion, joint question-message encoding and order features do not establish a robust argument advantage. In the eight-question Llama pilot, arguments improve held-out accuracy only from 70.64% to 70.90%. These results concern the tested representations and limited cohorts, not the irrelevance of language (Appendix C).'),
 p('Controlled Llama interventions give complementary evidence: replacing rationales with conclusion-only messages changes first-token P(A) by <b>22.40 pp on average</b> across 489 nonempty inboxes with explicitly matching stances; the dominant token choice changes in 130. Thus fixed opinions, personas and graph need not determine the full-prompt response distribution. Minimal validity edits have much smaller effects than rationale deletion, and validity grading is near chance. These experiments establish text dependence, not reliable logical-quality recognition. First-token probabilities and sampled five-vote opinions are different targets (Appendix D).'),
 sub('4.5 Transfer and interpretability remain qualified'),
 p('On four fresh graphs and 20 reused questions, the 320-question transformer reaches <b>84.35%</b> versus Ising’s <b>81.37%</b>, a 2.98 pp advantage (1.18–4.98). Both use the same graph family and known personas; broader graph transfer remains untested in this Llama study. On GPT development folds, an 809-weight aggregate-state network reaches 9.57% error versus 9.12% for graph attention and 10.95% for Ising. Explicit formulas with up to 42 coefficients still miss the compact transformer’s probabilities by about 4.9 pp. Neither result identifies a unique law (Appendices E–F).'),
 h('5. Discussion and Limitations'),
 p('The evidence supports a useful conditional predictor whose gains concentrate on difficult updates. It does not show that opinions are a sufficient state, that attention weights are causal credibility, or that one-step accuracy guarantees reliable simulation. The original frozen GPT test found a modest one-step gain but no consistent autonomous-rollout improvement. Larger models can overfit, and generic frozen text features may fail to preserve useful reasoning distinctions. Noise, omitted information and limited question transfer remain separate contributors. Text interventions do not rule out a calibrated opinion-only forecast that averages over future message generation.'),
 p('<b>Scope.</b> The principal tasks are objective mathematics under snap-judgment prompts, prescribed friendly/unfriendly relations and fixed personas. The new corpus has one trajectory per question and entropy selection favors substantial within-agent randomness. Most follow-ups reuse validation questions; confidence intervals omit full training-resampling and selection uncertainty. Original GPT test results are already consumed; the separate 40-question Llama assessment is still ungenerated. External and in-context pilots are supportive diagnostics, not replications on the same population.'),
 p('<b>Future work.</b> Freeze the next predictor before fresh assessment; separate breadth from transition count; collect repeated full-prompt responses and matched message interventions on new questions; test learned text representations and longer-horizon calibration. Learned mutual credibility, unfamiliar personas and graph families require separately designed experiments.'),
 h('6. Conclusion'),
 p('A compact transformer can improve on an Ising baseline without needing a large update network. More data helps, but greater size has not produced better average accuracy in the tested range. The strongest next step is to improve and test the state representation while preserving independent assessment, rather than interpreting a small predictive gap as a completed explanation of agent influence.'),
 sub('Code and Data'),
 p('Versioned code, source provenance, fitted models and reports are retained in the project workspace. Appendix H maps claims to their artifacts. The configured repository is <link href="https://github.com/deshearth/agent_interact">github.com/deshearth/agent_interact</link>; current public availability is unverified. No public release of the new conversations is claimed here.')
]})

refs=[
('1','El, B., Paeng, J., Dinc, F., Su, S., Erdogan, M., Pappu, A., Ye, H., Zhao, W., Ganguli, S., and Zou, J. (2026). Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents. arXiv:2608.16578v2.','https://arxiv.org/abs/2608.16578v2'),
('2','Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS Datasets and Benchmarks. arXiv:2103.03874.','https://arxiv.org/abs/2103.03874'),
('3','Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv:2002.10957.','https://arxiv.org/abs/2002.10957'),
('4','Sentence Transformers (n.d.). all-MiniLM-L6-v2 model card. Hugging Face. Accessed 14 November 2026.','https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'),
('5','Grattafiori, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.','https://arxiv.org/abs/2407.21783'),
('6','Savcisens, G., Dies, S., Maynard, C., and Eliassi-Rad, T. (2026). Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models. arXiv:2607.27512. Code and data: CoevolveSim.','https://github.com/carlomarxdk/coevolve-sim'),
('7','Multi-Agent-LLMs (n.d.). DEBATE dataset. Hugging Face. Accessed 14 November 2026.','https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE')]
pages.append({'label':'References and disclosure','blocks':[
 h('References'),*[p(f'[{i}] {text}<br/><link href="{url}">{url}</link>') for i,text,url in refs],
 sub('LLM Usage Statement'),
 p('Codex assisted with experiment design discussions, implementation, data checks, computational analyses and drafting this report. The source agents and frozen embedding models are also LLM-derived components of the experiments, as described in Methods. Numerical claims were checked against saved results and computational integrity audits; these checks are not an independent human replication. This document is an AI-assisted synthesis prepared for author review before formal submission.'),
 sub('Report scope and template'),
 p('The structure follows the user-supplied Apart Research hackathon submission template for the November 2026 AI × Epistemics Research Sprint: four main pages, followed by references and appendices. Names and affiliations are intentionally omitted at the author’s request. Use of the template does not assert organizational affiliation or that the report has been submitted. Results are current through 14 November 2026; no new model fitting or source generation was performed to write this report.')
]})

pages.append({'label':'Appendix A · Protocol and state','blocks':[
 h('Appendix A. Data, state and training boundaries'),
 sub('A.1 What is pretrained, fitted, or fixed?'),
 table('Table A1. Component roles. “818 parameters” is the trainable update network, not the complete text-processing system.',
 ['Component','Origin / treatment','Role'],[
 ['Source agents','Released GPT-4o-mini; separately pinned 4-bit Llama-3.1-8B-Instruct','Generate opinions and messages; no source-model fine-tuning'],
 ['Text encoder','Frozen all-MiniLM-L6-v2; 22,713,216 parameters','384-dimensional message vectors and fresh Llama question/persona vectors'],
 ['Released GPT context','Cached upstream question/persona embeddings','Controlled studies refit question PCA only within fitting questions; fuller-context tests use 1,536 cached dimensions'],
 ['PCA and normalization','Fitted within each training subset/fold','No held-out question fitting; persona population is shared'],
 ['Ising / neural predictor','Fitted on project transitions','19 stored coefficients / architecture-specific neural counts'],
 ['Graph and persona population','Prescribed source inputs','Friendliness is disclosed to source agents; mutual credibility is not learned']],[.21,.43,.36]),
 p('The compact head predicts a narrower task than the source LLM performs. It does not generate messages, and end-to-end source-model replacement or inference speedup is not established.'),
 sub('A.2 Baseline and signed attention'),
 p('Let s be the current ±1 opinion vector, J the signed adjacency matrix, and φ the 16-component context. Define F<sub>i</sub> = Σ<sub>j</sub> max(J<sub>ij</sub>,0)s<sub>j</sub> and U<sub>i</sub> = Σ<sub>j</sub> min(J<sub>ij</sub>,0)s<sub>j</sub>. The reproduced baseline uses logit P(A next) = w·φ<sub>i</sub> + aF<sub>i</sub> + bU<sub>i</sub> + c(F<sub>i</sub>−U<sub>i</sub>), fitted with the pinned author two-stage routine. It stores 16+3=19 coefficients; the social columns are linearly dependent, so 19 is not a claim of 19 identifiable degrees of freedom. Persona dependence comes through shared context features, not one free weight per persona.'),
 p('In the transformer, non-neighbors are masked out. Each attention head learns positive/negative edge biases and separate signed message multipliers, plus a degree-dependent aggregation scale. Receiver and sender state/context produce queries, keys and values. These mechanisms use the prescribed relation labels; the attention weights are not estimates of causal influence. The 818 model has hidden width 8, four heads, key width 1 per head, value width 2 and feedforward width 16.'),
 sub('A.3 Source generation and selection'),
 p('The original development bank contains 20 objective questions, with eight graphs, four replicas, eight updates and 32 agents: 163,840 transitions. Initial controlled fits use 16/4 question separation. The 20 original test questions were scored only after the controlled model freeze. Subsequent nested development studies do not restore freshness to those already examined questions.'),
 p('The separate Llama source uses pinned MLX weights, float32 floating computation, temperature 0.7, top-p 1 and a 256-token cap. Generation ran locally and on UCI HPC3; each new GPU allocation had to pass a five-group numerical replay. Request identities, raw votes, messages, prompt records and hashes support resumable collection. The completed 400-question development corpus has 102,400 valid majority transitions; one individual vote is unparseable without invalidating its majority.'),
 p('Initial selection retained 140 of 560 candidates (80 fit, 20 validation, 40 reserved). Expansion selected 300 of 1,200 new candidates, maintaining the top-quarter rule. Near-duplicate filters use number-masked families and character-ngram similarity, not a guarantee of semantic independence. Original distractor construction used a deterministic wrong-option fallback for 249/560 candidates; plausibility was not independently certified. Pooled entropy is a selection property, not an estimate of stable disagreement or predictor difficulty. Evidence: E01, E10, E13–E16.')
]})

pages.append({'label':'Appendix B · GPT prediction and capacity','blocks':[
 h('Appendix B. Original GPT evidence and capacity'),
 sub('B.1 Frozen original test and autonomous rollout'),
 table('Table B1. Already-completed original test, training-only question PCA. A2 is the 135,458-parameter attention model. These are different graph roles within the same released source.',
 ['New-question graph role','Questions','Ising accuracy','A2 accuracy','Ising / A2 NLL'],[
 ['Familiar random',20,'86.81%','88.63%','0.2552 / 0.2423'],
 ['Fresh random',20,'85.36%','87.22%','0.2790 / 0.2663'],
 ['Square',20,'95.19%','95.13%','0.1335 / 0.1204'],
 ['Triangular',20,'93.44%','93.37%','0.1738 / 0.1400'],
 ['Low-rank',10,'97.88%','97.83%','0.1033 / 0.0710']],[.29,.12,.18,.18,.23]),
 p('A2’s familiar-random accuracy gain is 1.82 pp [0.68, 2.93]; its fresh-random gain is 1.86 pp [0.51, 3.25]. The advantage over fixed aggregation A0 is unresolved. Actual history values add 0.60 pp on familiar and 0.88 pp on fresh random graphs relative to the matched history-availability control. Benefits do not transfer uniformly to structured graphs.'),
 p('Autonomous eight-round predictions use generated states after initialization. Familiar-random final-round accuracy is 66.53% for Ising and 66.23% for A2; fresh-random accuracy is 62.84% and 63.49%. A2’s final-fraction Wasserstein distances are worse on average (0.2199 vs 0.2062; 0.2374 vs 0.2269). Paired intervals do not establish a general rollout advantage. Known-question square/triangular graph roles are recorded separately in the original report.'),
 sub('B.2 More capacity can fit better and generalize worse'),
 table('Table B2. Same 16 GPT fitting questions / four reused validation questions. The unregularized diagnostic also changes schedule and duration; it is not a generalization-selected replacement.',
 ['Predictor','Train error','Validation error','Validation NLL'],[
 ['Ising','9.15%','11.91%','0.28469'],
 ['Original selected A2','8.14%','10.59%','0.27943'],
 ['Unregularized A2, 128 epochs','4.41%','12.22%','0.48890']],[.43,.19,.20,.18]),
 p('All nine Ising-probability imitation fits pass the declared closeness criterion; the training-selected rate yields 0.022 pp MAE on fitting states and 0.021 pp on development validation. Exact input collisions force at least 3.41% empirical deterministic training error on the recorded states. This finite-sample conflict bound is not a population Bayes error or a capacity optimum.'),
 sub('B.3 Nested whole-question size comparisons'),
 table('Table B3. Four outer folds on 20 previously studied development questions; three fresh refit seeds. Inner folds choose settings and training-only PCA. No new confirmatory question cohort.',
 ['Model','Parameters','Outer error','Outer NLL'],[
 ['Ising',19,'10.95%','0.22755'],['Ising + own-opinion correction','19 + 3','10.11%','0.22474'],
 ['Transformer',314,'9.22%','0.21622'],['Transformer',818,'9.12%','0.21258'],['Transformer','2,626','9.17%','0.22036'],
 ['Transformer','7,370','9.17%','0.21472'],['Transformer','16,842','9.32%','0.22024'],['Transformer','34,978','9.48%','0.22321'],
 ['Transformer','135,458','9.81%','0.23282'],['Ising + small transformer','19 + 34,978','9.41%','0.22354']],[.43,.19,.19,.19]),
 p('Small-model rankings are uncertain and some early searches hit update caps, so these are finite-budget comparisons rather than fully optimized size optima. The hybrid adds little beyond the small ordinary transformer. These findings motivated the fresh Llama comparison, rather than proving that 818 parameters is universally optimal. Evidence: E01–E04.')
]})

pages.append({'label':'Appendix C · Argument representation','blocks':[
 h('Appendix C. Argument representations and transfer'),
 p('Each delivered message is encoded separately by frozen MiniLM, then processed by trainable graph layers. Gradient, checkpoint, masking and small-memorization checks verify the branch. No encoder fine-tuning was performed; generic semantic vectors need not preserve useful reasoning distinctions.'),
 table('Table C1. Familiar-question controls, GPT replica split. Original 185,890-parameter architecture; no-text still includes opinions/history and question/persona context.',
 ['Argument-branch input','Training error','Familiar-question error'],[
 ['No argument vector','6.11%','6.47%'],['Full individual arguments','5.71%','5.93%'],
 ['One average vector per question','5.84%','6.01%'],['Question ID only','5.93%','5.92%'],
 ['Question/stance ID only','5.81%','5.85%'],['Residual after question/stance mean','6.27%','6.91%']],[.57,.20,.23]),
 p('Question and question/stance centroids explain 83.97% and 86.25% of familiar message-vector variance. ID controls reproduce the gain without individual-message language, but cannot predict new questions. They do not show that the source ignores arguments or residual features are useless.'),
 table('Table C2. Tests that did not establish a transferable remedy. Error is on the specified evaluation cohort; architectures and splits differ between rows.',
 ['Comparison','Result','Interpretation'],[
 ['Additive text vs no text','Familiar 5.93 vs 6.47%; unseen 10.84 vs 9.84%','Familiar benefit does not transfer'],
 ['Separate normalized fusion','Unseen 11.81% vs additive 10.84%','Better fit does not rescue transfer'],
 ['Full 1,536-D question vs matched PCA branch','Unseen argument error 10.36 vs 10.35%','No clear benefit from extra question dimensions'],
 ['Joint question-message vs matched message encoding','Unseen 10.41 vs 10.34%','This frozen joint encoding does not help'],
 ['Reconstructed order / random-order control','Unseen 10.52 / 10.46%','No reliable order-based remedy'],
 ['32× argument-vector scaling','Validation 88.32 vs 89.33% accuracy','Increasing branch scale worsens results']],[.38,.34,.28]),
 table('Table C3. Earlier GPT question-count curve; same four validation questions. This uses fixed architecture and its own protocol, not the expanded Llama experiment.',
 ['Fitting questions','No-text error','Argument error','No-text / argument NLL'],[
 [4,'12.17%','12.40%','0.35016 / 0.35616'],[8,'10.72%','11.02%','0.28783 / 0.29346'],
 [12,'10.31%','10.85%','0.27727 / 0.28727'],[16,'10.47%','10.67%','0.26530 / 0.27133']],[.23,.23,.23,.31]),
 p('The eight-question Llama pilot uses 24 matched 282,146-parameter fits with 4/2/2 fitting/selection/held-out questions per fold. With/without arguments, held-out accuracy is 70.90/70.64%, NLL 0.6179/0.6290, selected training error 20.12/20.18% and terminal error 11.91/12.37%. The control has 49,152 inactive text-projection weights. Four fitting questions and frozen vectors limit this pilot.'),
 p('Mixed-opinion subgroup gains remain exploratory. All 573,440 delivered GPT messages state a sender choice, partly duplicating stance inputs. Historical prompt provenance is incomplete and rationale correctness is not guaranteed. Evidence: E05–E09.')
]})

pages.append({'label':'Appendix D · Text interventions and noise','blocks':[
 h('Appendix D. Source interventions and error structure'),
 sub('D.1 Text dependence differs from validity recognition'),
 p('The local source pilot uses 24 receiver contexts on four reused questions. Four message variants give 96 prompts and 1,920 responses: 20 independent draws per prompt, grouped into four nonoverlapping five-vote outcomes. Each contrast fixes the question, persona, graph cue, background inbox and order, changing one target message.'),
 table('Table D1. Mean absolute change in the source’s first literal-A/B token probability. These are probability effects, not predictor accuracy gains or majority-target errors.',
 ['One-message contrast','Absolute change'],[
 ['Valid vs minimally flawed rationale','2.96 pp'],['Two valid phrasings','6.69 pp'],['Valid rationale vs conclusion only','15.73 pp']],[.74,.26]),
 p('The flawed rationale retains the same correct final choice; an independent solver could keep that choice despite noticing the flaw. Deletion and paraphrase also change wording and length. Friendly/unfriendly cases are different receiver contexts, so their comparison is not an intervention on edge sign.'),
 table('Table D2. Rationale deletion in eight fully new Llama conversations (512 transitions). Incoming stances are kept fixed. Majority changes also include ordinary sampling variation.',
 ['Cohort','Updates','Mean |ΔP(A)|','Dominant-token changes','Sampled-majority changes'],[
 ['All updates',512,'21.87 pp',132,181],['Nonempty, matching explicit stance',489,'22.40 pp',130,169],['Empty-inbox controls',16,'0.00 pp',0,8]],[.36,.12,.17,.18,.17]),
 p('A post-hoc filter excluding truncation and suspicious stance mentions leaves 414 contexts and a 21.35 pp effect. Empty-inbox prompts are byte-identical, yet sampled majorities can disagree. Explicit validity grading on four constructed problems is 51.56% balanced accuracy with counterbalanced A/B labels and 49.90% with VALID/FLAWED words. This small, short-response task does not establish general mathematical-verification ability.'),
 sub('D.2 Stochastic targets and omitted state on the 80/20 Llama corpus'),
 table('Table D3. Retrospective validation error breakdown. Next-vote agreement is not available to the predictor.',
 ['Majority agreement','Share of updates','Transformer error','Share of its errors'],[
 ['3 of 5','19.00%','39.19%','49.29%'],['4 of 5','23.96%','20.18%','32.01%'],['5 of 5','57.03%','4.95%','18.70%']],[.25,.25,.25,.25]),
 p('Split-vote targets make up 42.97% of updates but 81.30% of transformer errors. Normalizing logged first-token probabilities and assuming conditionally independent draws yields the majority approximation q=p³(10−15p+6p²). The mean full-prompt sampling reference min(q,1−q) is 8.58%; the transformer’s expected error under those probabilities is 15.60%. This reference includes newly generated messages unavailable at the original forecasting boundary, and is not an attainable Bayes bound for opinion-only inputs.'),
 p('A verified repeated local opinion state has source P(A) 0.66% in one round and 98.69% in another while both predictors remain unchanged. Other text and ordering change too. Across validation, 15.06% of updates belong to repeated connected-agent states with source majority-probability ranges above 25 pp. These observations identify missing information without isolating logical quality.'),
 p('The transformer corrects 4.75 pp of Ising errors but introduces 3.18 pp, leaving a 1.57 pp net gain. It helps stays (9.56% vs 12.18% error), but not switches (25.91% vs 25.43%). Pooled screening entropy averages 0.994 bits, of which 0.873 is within-agent entropy: selection mainly finds uncertain agents, not necessarily stable disagreement. Evidence: E08, E10–E11.')
]})

pages.append({'label':'Appendix E · Interpretable rules and pooling','blocks':[
 h('Appendix E. Attempts to simplify the learned rule'),
 sub('E.1 Persistence, question bias and the easy majority'),
 p('On the old 20-question GPT development pool, 79.63% of updates have Ising confidence at least 0.8; Ising and the 818-parameter transformer both make about 2.17% errors there. On the remaining 20.37%, error is 45.27% for Ising and 36.28% for the transformer. Thus aggregate accuracy conceals a larger improvement on a small difficult subset, without providing a deployable oracle for selecting correct predictions.'),
 p('Adding a training-fitted own-opinion correction to the frozen Ising logit lowers outer error from 10.95% to 10.11%; initial-response tendency features do not improve that tested linear correction. The correction represents a predictive association, not proven causal self-influence. In a separate same-question adaptation diagnostic, offsets fitted on other replicas lower Ising error to 9.10% and the 34,978-parameter transformer’s error to 8.45%. Those offsets use extra same-question outcomes and are not unseen-question predictors.'),
 sub('E.2 Explicit distillation from the 818-parameter transformer'),
 table('Table E1. Surrogates fitted to frozen transformer probabilities, not observed labels. Old held-out questions; twelve fold/seed teachers and 48 converged fits. Probability MAE is in percentage points.',
 ['Surrogate','Coefficients','Teacher-probability MAE','Actual-opinion error'],[
 ['Ising logit + own opinion','19 retained + 3','5.09','10.13%'],['Pooled linear','21','5.04','10.28%'],
 ['Context-dependent influence','39','4.93','10.18%'],['Context-dependent + quadratic','42','4.87','10.10%'],
 ['Teacher transformer','818','—','9.12%']],[.39,.20,.23,.18]),
 p('The quadratic rule’s training MAE is already 4.84 pp, so its discrepancy is not solely unseen-question transfer. Different full neighborhoods can give different transformer probabilities under identical pooled features. Nevertheless, the finite-sample minimum MAE forced by exact pooled-input collisions is only 0.141 pp on these held-out states, far below the observed 4.87 pp. Pooling loses some information, but the collision evidence does not establish it as the dominant cause of formula failure.'),
 sub('E.3 Nonlinear models of aggregate state'),
 table('Table E2. Same GPT nested-development comparison; targets are observed opinions. Aggregate-state models do not use individual-neighbor attention.',
 ['Predictor','Parameters','Train error','Outer error','Outer NLL'],[
 ['Ising',19,'9.79%','10.95%','0.22755'],['Joint pooled linear',21,'9.49%','10.16%','0.22535'],
 ['Pooled network',809,'7.97%','9.57%','0.22251'],['Pooled, doubled duration',809,'7.76%','9.55%','0.22415'],
 ['Larger pooled network','3,457','7.98%','9.65%','0.21790'],['Graph transformer',818,'8.02%','9.12%','0.21258']],[.35,.15,.17,.17,.16]),
 p('The 809-weight pool recovers most of the observed classification improvement over Ising, while leaving a 0.46 pp gap to graph attention (descriptive interval 0.06–1.02). Doubling its duration lowers training error but does not close the average gap; larger pooling improves NLL without improving error. The neural aggregate result shows that failure of the tested explicit formulas is not equivalent to failure of all aggregate-state predictors.'),
 p('These studies support a modest benefit from neighborhood detail or its inductive bias under the tested procedure. They neither establish that 818 weights are necessary nor rule out a simpler expression. Interpretation should be tested through predictive fidelity and controlled changes to inputs, not attention heatmaps alone. All results reuse previously studied GPT questions. Evidence: E03–E04, E12.')
]})

curve_rows=[]
for n in [40,80,160,320]:
    d=curve['summary'][str(n)]
    curve_rows.append([n]+[f"{100*(1-d[k]['validation']['error']):.2f}% / {d[k]['validation']['nll']:.5f}" for k in ['I0','nano','tiny','wide']])
pages.append({'label':'Appendix F · Llama scaling and graph transfer','blocks':[
 h('Appendix F. Llama learning curves and graph transfer'),
 table('Table F1. Current tuned question-count study. Each cell is validation accuracy / NLL on the same 80 questions. Smaller-count subsets retain 25% original and 75% added questions.',
 ['Train questions','Ising 19','Transformer 818','Transformer 7,370','Transformer 533,026'],curve_rows,[.14,.21,.21,.22,.22]),
 p('The largest model’s 40-question subset means span 82.00–83.32%, with a maximum within-subset seed range of 1.54 pp. No chosen checkpoint reaches its cap. Update caps are 4,096/8,192/16,384/32,768; evaluation cadence 64/128/256/512; patience ten. All sizes use dropout 0.1, AdamW decay 0.001, batch 32 and clipping one.'),
 sub('F.1 Earlier probability-target curve is a different experiment'),
 table('Table F2. Original 80-question training pool and fixed 20-question validation cohort. Same selected settings and 1,280 updates at each count; three nested paths and three seeds. Do not substitute these values for Table F1.',
 ['Train questions','818 hard targets','818 probability targets','Ising hard targets'],[
 [10,'83.52%','83.46%','82.71%'],[20,'83.95%','84.38%','83.29%'],[40,'84.68%','84.59%','83.19%'],[80,'84.89%','85.14%','83.32%']],[.22,.26,.29,.23]),
 p('Logged majority probabilities provide privileged teacher supervision with unchanged inputs. At 80 questions, the accuracy gain is 0.25 pp [−0.05, 0.60]; probability-loss benefits are clearer at small sample sizes. The first-token approximation (Appendix D) does not provide a noise-only explanation.'),
 sub('F.2 Expansion and held-out graph instances'),
 p('The old 80-question transformer scores 83.75% on the expanded 80-question validation cohort, versus 84.09% after training on 320 (+0.34 pp, interval 0.05–0.65). On only the 60 added validation questions, the gain is +0.32 pp (−0.01 to 0.66). These old-80 fits differ from the new random 80-question subsets of Table F1.'),
 table('Table F3. Same 20 validation questions, with paired initial five-vote states. Four new graphs have 32 nodes and 56 undirected signed edges; subsequent stochastic trajectories differ.',
 ['Training count / model','Familiar accuracy','Fresh-graph accuracy','Fresh-graph NLL'],[
 ['80 / Ising','83.32%','81.23%','0.39232'],['80 / 818 transformer','84.89%','83.91%','0.35999'],
 ['320 / Ising','83.05%','81.37%','0.39111'],['320 / 818 transformer','85.31%','84.35%','0.35486']],[.37,.21,.23,.19]),
 p('At 320 questions, graph novelty changes accuracy by −0.96 pp for the transformer and −1.68 pp for Ising; the difference, +0.72 pp [−0.71, 2.43], is unresolved. Training expansion gains +0.44 pp [−0.15, 1.00] on fresh graphs. Jf3 supplies 65% of the transformer–Ising gap, but differing questions confound graph-property attribution.'),
 p('Fresh-graph accuracies are 84.51/84.35/84.21/83.95/84.24/83.68% at 314/818/7,370/34,978/135,458/533,026 parameters. The 314-vs-818 gap is uncertain. Four same-family graphs and reused validation questions do not establish new-question or new-topology transfer. Evidence: E11, E13–E17.')
]})

pages.append({'label':'Appendix G · External and in-context pilots','blocks':[
 h('Appendix G. External data and in-context prediction'),
 sub('G.1 External feasibility audits'),
 p('DEBATE/MALLM [7] was audited before training an argument predictor. In the simple-response/no-persona cohort, copying the supplied draft predicts all 1,440 development updates. In the critical-response/expert-persona cohort, only two of 360 validation updates depart from that draft. Neither audited configuration passes the recorded feasibility gate for an argument comparison, so no such model was trained. This conclusion is about those configurations, not the whole dataset.'),
 sub('G.2 CoevolveSim matched-budget comparison'),
 p('The pilot uses 20 medical-indication statements with matched specialists, two graph realizations, 48 agents and ten updates [6]. Whole-statement splits are 12 fitting, four validation and four reserved. Targets are native true/false/neither beliefs. Inputs are role/belief summaries, current receiver state, degree and round; there are no friendship signs or full argument texts. Its 795-weight attention architecture is an adaptation, not the exact 818 model.'),
 table('Table G1. CoevolveSim validation-selected pilot. Parameter counts for logistic models are stored softmax coefficients. Only four validation statements; accuracy is not comparable to the binary MATH experiments.',
 ['Predictor','Parameters','Train error','Validation error','Validation NLL'],[
 ['Empirical Markov',6,'11.51%','3.28%','0.20979'],['Local-opinion logistic',21,'10.56%','3.41%','0.11129'],
 ['Local + receiver identity',66,'9.11%','3.10%','0.10265'],['Full aggregate logistic',207,'7.99%','2.84%','0.10561'],
 ['Pooled MLP',795,'6.36%','2.52%','0.08359'],['Tiny attention',795,'7.40%','2.58%','0.08102']],[.35,.15,.17,.18,.15]),
 p('Attention has slightly lower NLL but slightly higher error than the equally sized MLP. Validation is substantially easier than training, as shown by the Markov baseline; lower validation error is not evidence of a generalization breakthrough. A preliminary unmatched 895-vs-795 comparison was archived and corrected before assessment. All reserved external groups remain unscored, although source schemas were inspected; they must not be described as unopened.'),
 sub('G.3 Assistant’s committed in-context forecasts'),
 p('The assistant studied 256 labeled updates from eight existing training questions, then manually assigned 128 probabilities for four fresh-graph conversations at one fixed boundary (round 3→4, all 32 agents). The forecast packet excluded future opinions, messages, raw votes, solutions and source probabilities. Forecasts and strategy were committed before new outcomes or competitor predictions were displayed; no optimizer fitted new weights.'),
 table('Table G2. Matched 128-update pilot; frozen 80-question reference predictors. These are fresh trajectories on familiar development questions, not a new question assessment.',
 ['Predictor','Accuracy','Log loss','Binary Brier'],[
 ['Assistant, in context','90.625%','0.23551','0.06699'],['818 transformer, seed mean','87.760%','0.23668','0.07068'],
 ['Ising','85.938%','0.26831','0.08253'],['Keep current opinion','80.469%','—','—']],[.43,.19,.19,.19]),
 p('The assistant gets 116/128 correct. Its +2.86 pp advantage over the transformer has a descriptive four-question interval of −2.60 to +9.64, and probability losses are almost identical. Gains concentrate in one question. Extensive pretraining, raw text access, unequal demonstration/fitting data and procedural rather than access-enforced blinding limit the comparison. A separate fresh-question ICL protocol remains prepared but unexecuted. Evidence: E18–E20.')
]})

# Give the original test and the capacity/size follow-ups separate coherent pages.
bi=next(i for i,page in enumerate(pages) if page['label'].startswith('Appendix B'))
original=pages[bi]['blocks']
cut=next(i for i,b in enumerate(original) if b.get('text','').startswith('B.2 More capacity'))
test_blocks=original[:cut-1]  # Replace the prose rollout summary with its full table.
test_blocks += [
 sub('B.2 History and fixed-aggregation controls'),
 table('Table B2. One-step accuracy on the same original test questions. Availability-only masks past opinion values while preserving the initialization/availability cue.',
 ['Model / state','Parameters','Familiar random','Fresh random'],[
 ['A0 / current opinions','119,066','88.14%','87.23%'],['A0 / history','119,834','89.35%','88.47%'],
 ['A2 / current opinions','135,458','88.63%','87.22%'],['A2 / availability-only','136,226','88.60%','87.02%'],['A2 / history','136,226','89.20%','87.89%']],[.38,.20,.21,.21]),
 sub('B.3 One-step gains do not imply autonomous fidelity'),
 table('Table B3. Eight-round autonomous simulation from observed initialization; later inputs are generated states. Wasserstein compares the final population fraction distribution (lower is better).',
 ['Model / state','Round-8 accuracy: familiar / fresh','Wasserstein: familiar / fresh'],[
 ['Ising','66.53% / 62.84%','0.2062 / 0.2269'],['A0 / opinions','65.35% / 63.51%','0.2267 / 0.2418'],
 ['A2 / opinions','66.23% / 63.49%','0.2199 / 0.2374'],['A2 / history','65.85% / 64.18%','0.2169 / 0.2332']],[.30,.35,.35]),
 p('Paired rollout intervals do not establish a general neural advantage; final-fraction distribution distances are worse on average for A2/opinions in both random-graph roles. Known-training-question square/triangular roles are separately recorded in the original report. Source: E01.')]
capacity_blocks=[h('Appendix B (continued). Capacity and model size')]+original[cut:]
for b in capacity_blocks:
    if b.get('text','').startswith('B.2 More capacity'):b['text']=b['text'].replace('B.2','B.4',1)
    if b.get('text','').startswith('B.3 Nested'):b['text']=b['text'].replace('B.3','B.5',1)
    if b.get('caption','').startswith('Table B2.'):b['caption']=b['caption'].replace('Table B2.','Table B4.',1)
    if b.get('caption','').startswith('Table B3.'):b['caption']=b['caption'].replace('Table B3.','Table B5.',1)
capacity_blocks += [sub('B.6 The expanded Llama sweep shows the same stopping issue'),
 p('For the selected 533,026-parameter tuning candidate (seed 42), continuing from 1,536 to 6,656 updates increases training accuracy from 84.12% to 86.17%, while validation accuracy falls from 83.75% to 83.55%. Training NLL falls 0.35426→0.30955, but validation NLL rises 0.36225→0.38430. This is one tuning-run diagnostic, not the three-fresh-seed table or an exhaustive memorization test. Source: E15.')]
pages[bi:bi+1]=[{'label':'Appendix B · Original frozen test','blocks':test_blocks},{'label':'Appendix B · Capacity and size','blocks':capacity_blocks}]

evidence = [
('E01','Original data, reproduction and frozen test',['reports/data_audit_train_only_pca.md','reports/author_reproduction.md','reports/inductive_validation_results.md','reports/inductive_test_results.md','reports/history_availability_control.md']),
('E02','Ising imitation and capacity',['reports/ising_transformer_capacity.md']),
('E03','Nested generalization and correction',['reports/transformer_generalization.md','reports/ising_gap_diagnosis.md']),
('E04','GPT size and micro sweeps',['reports/transformer_size_sweep.md','reports/transformer_micro_sweep.md']),
('E05','Argument-response training and early curve',['reports/argument_response_results.md','reports/argument_training_diagnosis.md','reports/question_count_learning_curve.md']),
('E06','Fusion and full question context',['reports/argument_mechanisms.md','reports/question_context.md']),
('E07','Argument mechanism controls',['reports/argument_investigation.md']),
('E08','Local source and interventions',['reports/local_open_model_pilot.md']),
('E09','Llama argument predictor',['reports/local_llama_transformer.md']),
('E10','Llama error and probability diagnosis',['reports/question_breadth_error_diagnosis.md','runs/question_breadth_v1/development_results/summary.json']),
('E11','Earlier probability-target curve',['reports/llama_probability_learning_curve.md']),
('E12','Formula distillation and pooled networks',['reports/transformer_rule_distillation.md','reports/pooled_state_comparison.md']),
('E13','New-question selection and corpus',['configs/question_breadth.yaml','configs/question_breadth_320.yaml','runs/question_breadth_320_v1/source_protocol.json','runs/question_breadth_320_v1/episode_index.json']),
('E14','320/80 trained comparison',['runs/question_breadth_320_train_v1/development_results/summary.json']),
('E15','Six-size expanded sweep',['runs/breadth_320_size_sweep_v1/results/summary.json','runs/breadth_320_size_sweep_v1/completion_audit.json']),
('E16','Count × size study',['runs/breadth_320_learning_curve_v1/results/summary.json','runs/breadth_320_learning_curve_v1/completion_audit.json','configs/breadth_320_learning_curve.yaml']),
('E17','Paired fresh-graph transfer',['reports/unseen_graph_generalization.md','runs/unseen_graph_320_v1/results.json']),
('E18','External audits and prediction',['experiments/external/README.md','reports/external/coevolve_pilot_v1/results.md','reports/external/mallm_pilot_v1/audit.md','reports/external/mallm_critical_pilot_v1/audit.md']),
('E19','Completed in-context pilot',['experiments/icl_existing_graph_v1/README.md']),
('E20','Pending fresh-question ICL',['experiments/icl_forecast_v1/README.md'])]

pages.append({'label':'Appendix H · Evidence and reproducibility','blocks':[
 h('Appendix H. Evidence index and reproducibility'),
 p('The table abbreviates artifact paths; evidence_index.json records full project-relative paths and SHA-256 digests for every cited local source. report_content.json preserves the editable report text and tables. Current completed results take precedence over historical submission-status notes. Reserved assessment status remains unchanged.'),
 table('Table H1. Claim-to-artifact index. The machine-readable ledger expands groups to all files.',
 ['ID','Study / finding','Primary report or artifact'],[
 [i,title,paths[0].replace('reports/','').replace('runs/','').replace('experiments/','')] for i,title,paths in evidence],[.08,.39,.53]),
 p('<b>Completed verification.</b> The latest count × size experiment executed 243 new tuning fits and 81 fresh-seed refits, plus nine Ising fits; the full 320 endpoint was reused once. Neural fitting took 125.1 minutes. Six focused tests included a complete synthetic rehearsal. An independent audit reproduced 200 saved prediction arrays and 44 paired contrasts to within 1.12×10<super>-16</super>. All 150 GPU/CPU comparisons agreed on predicted classes; maximum probability difference was 5.67×10<super>-6</super>. Hash checks covered 2,226 new frozen, 1,137 prior and 202 result artifacts.'),
 p('<b>Reproduction.</b> The latest results can be verified without fitting using <font name="Mono">.venv/bin/python scripts/audit_breadth_320_learning_curve.py</font>. Earlier report files provide their respective commands, split manifests, scientific checks and frozen run directories. These are automated implementation checks, not independent external replications. Source probabilities and retrospective vote agreement are never ordinary forecast inputs.'),
 p('<b>Evidence status.</b> The original controlled GPT test was frozen before scoring, but its questions are now observed. Later GPT nested studies, current Llama validation curves, reused-question graph studies, external pilots and the in-context pilot have the limitations stated in their sections. The 40-question Llama assessment and four external CoevolveSim assessment statements remain unscored; emergent credibility, new-persona transfer and an interpretable sufficient update law remain future work.')
]})

def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    counts=[40,80,160,320]
    colors={'I0':'#a7652e','nano':'#23689b','tiny':'#288b72','wide':'#8b579e'}
    labels={'I0':'Ising (19)','nano':'Transformer 818','tiny':'Transformer 7,370','wide':'Transformer 533,026'}
    fig0,axes=plt.subplots(1,2,figsize=(6.5,2.15))
    for k in ['I0','nano','tiny','wide']:
        for ax,metric in zip(axes,['error','nll']):
            vals=[];lo=[];hi=[]
            for n in counts:
                r=curve['summary'][str(n)][k]['validation'];a,b=r['subset_ranges'][metric]
                vals.append(100*(1-r[metric]) if metric=='error' else r[metric]);lo.append(100*(1-b) if metric=='error' else a);hi.append(100*(1-a) if metric=='error' else b)
            ax.plot(counts,vals,color=colors[k],marker='o',markersize=3,lw=1.5,linestyle=':' if k=='I0' else '-',label=labels[k])
            ax.fill_between(counts,lo,hi,color=colors[k],alpha=.11,linewidth=0)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(counts,[str(x) for x in counts]);ax.set_xlabel('Training questions');ax.grid(axis='y',alpha=.17);ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Validation accuracy (%)');axes[1].set_ylabel('Validation NLL')
    handles,labels0=axes[0].get_legend_handles_labels();fig0.legend(handles,labels0,loc='upper center',ncol=2,frameon=False,fontsize=7.6,bbox_to_anchor=(.54,1.04))
    fig0.tight_layout(rect=(0,0,1,.83),w_pad=1.2)
    fig0.savefig(OUT/'figures/learning_curve_main.png',dpi=300,bbox_inches='tight');plt.close(fig0)
    fig0,ax=plt.subplots(figsize=(6.5,1.55));ax.set_xlim(-.08,10);ax.set_ylim(-.34,2.4);ax.axis('off')
    def box(x,y,w,h,text,color='#eff3f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',linewidth=.8,edgecolor='#65727b',facecolor=color))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    box(.05,1.34,2.7,.9,'Observed state\nopinions · graph · context')
    box(3.55,1.34,2.7,.9,'Fixed source LLM\ngenerates delivered messages')
    box(7.05,1.34,2.8,.9,'Source response\nnext five-vote majority')
    box(.05,.05,2.7,.76,'Before-message forecast\nIsing / compact transformer','#e9f1f8')
    box(3.55,.05,2.7,.76,'After-message response model\n+ frozen message vectors','#edf3ee')
    box(7.05,.05,2.8,.76,'Compare predicted probability\nand observed next opinion')
    for start,end in [((2.8,1.8),(3.45,1.8)),((6.3,1.8),(6.95,1.8)),((1.4,1.32),(1.4,.85)),((4.9,1.32),(4.9,.85)),((6.3,.43),(6.95,.43)),((8.45,1.32),(8.45,.85))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,color='#65727b',lw=.8))
    ax.plot([1.4,1.4,8.45],[.0,-.23,-.23],color='#23689b',lw=.8)
    ax.add_patch(FancyArrowPatch((8.45,-.23),(8.45,.0),arrowstyle='-|>',mutation_scale=8,color='#23689b',lw=.8))
    fig0.tight_layout(pad=.2);fig0.savefig(OUT/'figures/pipeline.png',dpi=300,bbox_inches='tight');plt.close(fig0)

def main():
    (OUT/'figures').mkdir(exist_ok=True)
    assert 150 <= len(abstract.split()) <= 250
    ledger=[]
    for code,title,paths in evidence:
        files=[]
        for name in paths:
            b=(ROOT/name).read_bytes();files.append({'path':name,'sha256':hashlib.sha256(b).hexdigest()})
        ledger.append({'id':code,'study':title,'files':files})
    template=Path('/Users/melancholymirth/Downloads/Copy of Apart Research hackathon submission template.pdf')
    record={'title':'Compact Models of LLM Agent Interaction','report_date':'2026-11-14','timezone':'America/Chicago','authors_omitted_by_request':True,'main_pages':4,'pages':pages,'abstract_word_count':len(abstract.split()),'references':[{'id':i,'citation':t,'url':u} for i,t,u in refs]}
    (OUT/'report_content.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (OUT/'evidence_index.json').write_text(json.dumps({'template_sha256':hashlib.sha256(template.read_bytes()).hexdigest(),'template_name':template.name,'evidence':ledger,'source_calls_for_report':0,'new_training_for_report':False},indent=2)+'\n')
    make_figures()
    print(json.dumps({'pages_planned':len(pages),'main_pages':4,'abstract_words':len(abstract.split()),'evidence_groups':len(ledger)}))

if __name__=='__main__': main()
