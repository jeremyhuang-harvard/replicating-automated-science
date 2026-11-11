# Qwen3-8B generated factorial experiments — 2026-10-14

## Scope and interpretation

This run executed the four designs generated earlier by local `mlx-community/Qwen3-8B-4bit`. It is **one-draw local-model evidence, not human behavior**. Each factorial cell contains one stochastic model interaction, so the reported contrasts and regressions describe this bounded run only; they do not estimate human causal effects or sampling uncertainty.

The execution compiler was also Qwen3-8B. It could only (a) assign each existing treatment to public information or one named agent's private information, (b) select a named respondent or an oracle, and (c) produce one executable cyclic turn order. The validator required the scenario, agents, outcome, cause names, hypotheses, predicted directions, treatment values, information scope, interaction cap, and stopping rule to remain exactly equal to the initial design. All four machine-readable diffs report zero substantive changes.

## Frozen generated designs and mechanical finalization

| Study | Outcome | Causes, ordered values, and prediction | Finalized execution |
|---|---|---|---|
| Mug bargaining | `agreement price` | `initial offer`: `$6`, `$8`, `$10` (increase); `buyer's budget`: `$5`, `$7`, `$9` (decrease); `mediator's presence`: `mediator present`, `mediator absent`, `mediator neutral` (non-monotonic) | All treatments public; respondent Alice; cycle Alice → Bob → Charlie → Dana; 8 turns |
| Bail setting | `Bail Amount` | `Severity of Tax Fraud`: `$50,000 fraud`, `$100,000 fraud`, `$200,000 fraud` (increase); `Defendant's Criminal History`: `No prior convictions`, `1 prior conviction`, `3 prior convictions` (increase); `Evidence Strength`: `Weak evidence`, `Moderate evidence`, `Strong evidence` (increase) | All treatments public; respondent Judge; cycle Judge → Defense Attorney → Prosecutor; 3 turns |
| Lawyer interview | `Interview success` | `Legal knowledge`: `Basic legal knowledge`, `Intermediate legal knowledge`, `Advanced legal knowledge` (increase); `Communication clarity`: `Unclear`, `Moderately clear`, `Very clear` (increase); `Confidence level`: `Low confidence`, `Moderate confidence`, `High confidence` (increase) | All treatments public; respondent Interviewer; cycle Interviewer → Candidate → Note-Taker; 3 turns |
| Art auction | `Final Bid Price` | `Bidder Risk Tolerance`: `Low`, `Medium`, `High` (increase); `Starting Price`: `50`, `100`, `150` (increase); `Bidding Strategy`: `Conservative`, `Aggressive`, `Rational` (increase) | Risk tolerance private to Agent A; starting price public; bidding strategy private to Agent B; respondent Agent A; cycle Agent A → Agent B → Agent C; 3 turns |

Every design has three causes with three ordered values. Deterministic Cartesian enumeration produced 27 unique cells per study and 108 unique study-cells total; the runner aborts before inference on any count mismatch or duplicate.

## Execution protocol

- Subject turns: temperature 0.3, thinking disabled, no tools/retrieval, deterministic recorded per-call seeds, frozen cyclic order, and the design's full `max_turns`.
- Measurement: a separate temperature-0 Qwen call after the interaction, strict JSON with exactly numeric `value` and non-empty `evidence`.
- Storage: append-only `raw_calls.jsonl` and `cells.jsonl`; every request includes the complete public transcript and the calling agent's private prompt. Private treatments are included only in the assigned agent's system prompt.
- Resume: a terminal cell record is never rerun. Re-invocation skips recorded cell IDs.
- Retries: finalization allows three attempts for provider/JSON/validation failures; subject turns allow two attempts only for provider error or empty text; measurement allows three attempts for provider/JSON/schema failures. Failed raw calls and failure metadata remain in the artifacts.

## Completion and measurement validity

| Study | Complete cells | Valid measurements | Outcome distribution (valid only) |
|---|---:|---:|---|
| Mug bargaining | 27/27 | 27/27 | min 0; median 7; mean 6.9815; max 9 |
| Bail setting | 27/27 | 27/27 | min 10,000; median 50,000; mean 68,518.52; max 500,000 |
| Lawyer interview | 27/27 | 25/27 | min 5; median 8; mean 7.56; max 9 |
| Art auction | 27/27 | 27/27 | min 70; median 150; mean 153.33; max 250 |

The two invalid lawyer-interview measurements were cells `2-0-0` and `2-0-1`. Each interaction completed, but all three deterministic measurement attempts ended in malformed/truncated JSON. The six malformed calls are retained. No subject-turn call failed. During mechanical finalization, ten rejected calls were retained before valid finalizations: three mug calls kept the invalid multi-agent respondent `Alice and Bob`, and seven auction calls contained malformed JSON (including an unquoted `analog` key); corrective prompting produced the accepted auction finalization.

## Marginal contrasts and additive OLS

Treatment values are encoded by their generated order as indices 0/1/2. A contrast is the marginal mean at index 2 minus the marginal mean at index 0. OLS is `outcome ~ 1 + cause1_index + cause2_index + cause3_index`, using valid measurements only.

### Mug bargaining

| Cause | High − low contrast | OLS coefficient | Qwen prediction | Agreement |
|---|---:|---:|---|---|
| initial offer | +2.1667 | +1.0833 | increase | yes |
| buyer's budget | +1.6111 | +0.8056 | decrease | no |
| mediator's presence | +0.5000 | +0.2500 | non-monotonic | no |

Intercept: 4.8426. The mediator marginal means were 6.6667, 7.1111, and 7.1667, so the middle level was not outside the endpoint interval and did not meet the predeclared non-monotonic agreement rule.

### Bail setting

| Cause | High − low contrast | OLS coefficient | Qwen prediction | Agreement |
|---|---:|---:|---|---|
| Severity of Tax Fraud | +55,555.56 | +27,777.78 | increase | yes |
| Defendant's Criminal History | +80,000.00 | +40,000.00 | increase | yes |
| Evidence Strength | +65,555.56 | +32,777.78 | increase | yes |

Intercept: −32,037.04. All three observed marginal directions matched Qwen's predictions, although the outcome had a large 500,000 maximum and the additive intercept is not substantively meaningful outside the coded grid.

### Lawyer interview

| Cause | High − low contrast | OLS coefficient | Qwen prediction | Agreement |
|---|---:|---:|---|---|
| Legal knowledge | +0.9921 | +0.3277 | increase | yes |
| Communication clarity | +1.2381 | +0.6445 | increase | yes |
| Confidence level | +1.8819 | +0.9600 | increase | yes |

Intercept: 5.5640. All observed signs matched Qwen's predictions, using 25 valid measurements; the two invalid high-knowledge cells make the grid unbalanced.

### Art auction

| Cause | High − low contrast | OLS coefficient | Qwen prediction | Agreement |
|---|---:|---:|---|---|
| Bidder Risk Tolerance | −3.3333 | −1.6667 | increase | no |
| Starting Price | +83.3333 | +41.6667 | increase | yes |
| Bidding Strategy | +8.8889 | +4.4444 | increase | yes |

Intercept: 108.8889. Starting price dominated the additive result. Risk tolerance ran slightly opposite to the prediction. Bidding strategy was strongly non-monotonic in its marginal means (131.11, 188.89, 140.00), despite its generated `increase` prediction.

## Conclusions and caveats

- Qwen's predicted direction agreed with 9 of 12 generated cause-level predictions under the declared rules. The mismatches were mug buyer budget, mug mediator status, and auction risk tolerance.
- The strongest ordered associations were the three bail factors, lawyer confidence, auction starting price, and mug initial offer.
- These are descriptive outputs from one local model draw per cell. The same model helped define treatments, acted as every subject, and measured outcomes, so common-model artifacts, prompt sensitivity, deterministic temperature-0 measurement behavior, and role inconsistencies are central limitations.
- Questionable scientific choices were intentionally preserved, including varying fraud severity away from the seed scenario and varying auction starting price. They are part of the generated-design object being studied.

## Artifacts and runtime

- Run root: `outputs/generated-experiments-qwen3-8b-2026-10-14/`
- Initial/finalized designs and diffs: `finalizations/<study>/{initial.json,finalized.json,diff.json,raw_calls.jsonl}`
- Deterministic factorials: `factorials/<study>.json`
- Cell transcripts, private prompts, raw calls, failures, and measurements: `studies/<study>/{cells.jsonl,raw_calls.jsonl,finalized_design.json}`
- Machine-readable analyses: `studies/<study>/analysis.json`
- Manifest/checkpoints: `manifest.json`, `checkpoint.json`

The successful full pipeline invocation took 1,118.92 seconds (18 minutes 38.92 seconds), including finalization of the remaining auction design and all 108 cell executions. Per-study execution times were 368.52s, 341.29s, 305.60s, and 77.53s in study order. Earlier rejected finalization attempts are retained but are outside that successful-invocation runtime.
