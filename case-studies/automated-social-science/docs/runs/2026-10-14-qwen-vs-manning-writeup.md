# Qwen as an Automated Social Scientist: Approach, Results, and Comparison with Manning, Zhu, and Horton

## What we did

Manning, Zhu, and Horton’s *Automated Social Science* asks whether a language model can function both as a scientist—selecting hypotheses, variables, treatments, agents, and measurements—and as the simulated subjects who generate experimental data. We followed that basic idea using the same four short scenario seeds: two people bargaining over a mug; a judge setting bail for a defendant accused of $50,000 in tax fraud; a lawyer job interview; and three bidders in an art auction starting at $50. However, we deliberately did **not** give Qwen the structural causal models, treatment variables, or results from the paper. Instead, local `mlx-community/Qwen3-8B-4bit` independently designed a new experiment for each scenario, after which the same model played the agents and measured the outcomes. Everything ran locally through Apple MLX, with no paid API calls.

## What Qwen was shown

For the scientist-design stage, Qwen saw only the one-sentence scenario and a neutral instruction to design a small causal social simulation. It was required to return JSON identifying up to four agents, one measurable outcome, one to three proposed causes, two or three concrete treatment values per cause, predicted causal directions, public and private information, agent goals and constraints, an interaction order, and a maximum of eight turns. The prompt did **not** mention Manning, Zhu, Horton, GPT-4, the paper title, the authors’ chosen variables, or the published findings. Scientist calls used temperature 0.3, disabled Qwen’s extended “thinking” output, used distinct recorded seeds, and accepted the first schema-valid design rather than selecting the most appealing answer.

A second Qwen pass acted only as an execution compiler. It could not change the scenario, outcome, agents, hypotheses, predicted directions, or treatment values; it could only translate each treatment into a public instruction or one agent’s private instruction, select a valid outcome respondent or oracle, and produce an executable turn order. During each simulated interaction, an agent saw its role, goal, constraints, the scenario, the public treatment information, its own private information and private treatment instructions, and the complete public transcript so far; it never received another agent’s private prompt. Agent turns used temperature 0.3, no tools or retrieval, and the design’s fixed turn limit. Afterward, a separate temperature-zero Qwen call received the outcome definition, measurement question, treatment labels, and public transcript and returned strict JSON containing one numerical outcome and supporting transcript evidence.

## Designs Qwen proposed

| Scenario | Qwen’s outcome | Qwen’s proposed causes |
|---|---|---|
| Mug bargaining | Final agreement price | Seller’s initial offer; buyer’s budget; mediator status |
| Bail setting | Exact bail amount | Fraud severity; prior convictions; evidence strength |
| Lawyer interview | Interviewer’s 1–10 suitability rating | Legal knowledge; communication clarity; confidence |
| Art auction | Highest final bid | Bidder risk tolerance; starting price; bidding strategy |

Each study had three causes with three ordered treatment values, producing $3^3=27$ conditions per study and 108 interactions in total. All 108 interactions completed, and 106 produced valid numerical measurements; two lawyer-interview measurement calls repeatedly returned malformed or truncated JSON. We analyzed each generated study by comparing the mean outcome at the highest versus lowest treatment level and by fitting a simple additive regression using treatment indices 0, 1, and 2. These statistics describe this particular model run only: there was one simulated interaction per cell, so they are not estimates of human behavior or stable model-level causal effects.

## Main results

Qwen’s predicted direction matched the simulated result for **9 of 12** generated causes. In mug bargaining, higher initial offers raised the final price, as predicted, but higher buyer budgets also raised the price even though the scientist predicted a decrease; mediator status did not show the predicted non-monotonic relationship. In bail setting, fraud severity, criminal history, and evidence strength all increased bail, matching all three predictions. In the lawyer interview, legal knowledge, communication clarity, and confidence all increased the interviewer’s rating, again matching all predictions. In the auction, starting price and bidding strategy increased the final price, while higher risk tolerance produced a small negative association rather than the predicted positive one.

## Comparison with the paper

The comparison is primarily about **scientific choices**, because our outcomes and treatments intentionally differ from those in the paper. Manning and coauthors’ mug experiment studied whether a deal occurred as buyer budget, seller minimum price, and seller attachment changed; Qwen instead studied the negotiated price and introduced the seller’s opening offer and a mediator. Both studies made budget-related bargaining constraints central, but our Qwen run produced a contradictory budget prediction and also created an ill-defined mediator treatment (“present,” “absent,” and “neutral”). In bail, both Qwen and the paper selected criminal history and found that more convictions increased bail; the paper’s other variables were judge workload and defendant remorse, whereas Qwen chose evidence strength and varied the fraud amount itself—even though the seed fixed the amount at $50,000.

The lawyer and auction comparisons are more revealing because the paper’s authors, rather than GPT-4 alone, selected those original variables. The paper varied bar passage, interviewer friendliness, and applicant height, finding a very large bar-passage effect and no meaningful friendliness or height effects. Qwen chose three directly performance-relevant attributes—knowledge, clarity, and confidence—and then found all three mattered, producing a more obvious and arguably less discriminating study. The paper’s auction varied each bidder’s private budget and found prices closely followed the second-highest valuation, connecting the simulation to auction theory. Qwen instead varied risk tolerance, starting price, and bidding strategy; starting price dominated its result, while risk tolerance moved in the wrong direction. Thus, the paper’s design was better connected to a clean theoretical benchmark, while Qwen’s design emphasized intuitive behavioral labels that were sometimes ambiguous or poorly assigned to agents.

## Interpretation

The strongest result is not that Qwen “confirmed” 9 of 12 hypotheses. The same model proposed the hypotheses, received the treatment descriptions, played all participants, and measured the outcomes, so agreement can partly reflect a shared-model self-fulfilling loop rather than genuine discovery. Qwen nevertheless completed the entire scientist-to-data pipeline cheaply and quickly: it generated four different causal studies, produced executable treatment grids, completed 108 interactions in about 19 minutes, and yielded analyzable outcomes with only two measurement failures. At the same time, its designs exposed important weaknesses—changing facts fixed in the scenario, confusing public and private information, proposing nonparallel treatment categories, and selecting causes whose expected effects were nearly built into their wording. The next informative step is therefore to separate model roles: use a stronger frontier model for study design while retaining free local models as subjects, then cross the scientist model and subject model to determine whether better reasoning improves the proposed studies, the resulting evidence, or both.

## Sources and project artifacts

1. Benjamin S. Manning, Kehang Zhu, and John J. Horton, *Automated Social Science: Language Models as Scientist and Subjects*, arXiv:2404.11794: https://arxiv.org/abs/2404.11794
2. Authors’ code repository: https://github.com/KeHang-Zhu/lm-automated-social-science
3. Qwen design report: `docs/runs/2026-10-14-qwen3-8b-canonical-designs.md`
4. Full local-experiment report: `docs/runs/2026-10-14-qwen3-8b-generated-experiments.md`
5. Raw run artifacts: `outputs/generated-experiments-qwen3-8b-2026-10-14/`
