# Data codebook

Every saved output in this repository, what produced it, and what each file holds. Paths are
relative to the case-study folder the code runs from. All model outputs are from
`mlx-community/Qwen3-8B-4bit` run locally through Apple MLX unless stated; none are human data.

The only change to any saved output is path scrubbing: in four Expected Parrot files
(`lasagna-cola-mlx-12*/manifest.json`, `expected-parrot-qwen-replication*/results.json`) the
home-directory prefix of the Python and `mlx_lm.generate` executables was replaced by `~/`,
and the local path of the stimulus image was replaced by its file name `image_09ac93.png`.
No hashed field contains those values. The repeated-seed runs of November 2026 apply the same
scrub automatically (`case-studies/run_repeated_seeds.py` rewrites the image path in each new
`results.json` and the image argument in `provenance.json`, and writes `~` for the home directory);
the stimulus SHA-256 is unchanged.

---

## Automated Social Science — `automated-social-science/outputs/`

Replicates the pipeline of Manning, Zhu & Horton (2024): a *scientist* model chooses causes,
treatments and outcomes for a short scenario; *subject* agents act out each treatment cell; a
*measurement* call reads the transcript and returns the outcome. Four scenario seeds are fixed
(mug bargaining, bail setting, lawyer interview, art auction).

### Run directories, in the order they were run

| directory | stage | produced by | what it is |
|---|---|---|---|
| `mock-smoke/` | pipeline check | `assr-replication smoke` | 3-turn bargaining with the scripted mock provider; no model |
| `local-smoke-verified/` | pipeline check | `assr-replication local-smoke` | the same 3 turns through MLX Qwen, temperature 0 |
| `canonical-designs-qwen3-8b-2026-10-14/` | **Qwen as scientist** | `assr-replication generate-designs` | one schema-valid design per scenario (4 calls, all valid first time) |
| `generated-experiments-qwen3-8b-2026-10-14/` | **Qwen designs × Qwen subjects** | `assr_replication.generated_cli` | the four designs compiled into 27-cell factorials and run once per cell (108 cells) |
| `frontier-dry-run-2026-10-14/` | plan only | `assr-frontier dry-run` | the 108-cell orthogonal-array plan for the GPT-Sol designs; no inference |
| `frontier-mock-verification-2026-10-14/` | pipeline check | `assr-frontier pilot --provider mock` | 4 cells with the mock provider |
| `frontier-pilot-2026-10-14-schema-v1-failed/` | **failed pilot** | `assr-frontier pilot` | 4 real cells; every interaction finished but measurement parser v1 rejected all 4 measurements; kept as-is |
| `frontier-pilot-2026-10-14/` | pilot | `assr-frontier pilot` | the same 4 cells after the parser fix (prompt version `frontier-prompts-v3`) |
| `frontier-planned-qwen-subjects-2026-10-14/` | **GPT-Sol designs × Qwen subjects** | `assr-frontier full-run`, then `analyze` | 108 cells (27 per study, OA(27,5,3,2)), one draw each |

### Repeated-seed runs (November 2026)

Made by `case-studies/run_repeated_seeds.py`, which calls the same entry points with new seeds and
new output directories and changes nothing else. Seeds are spaced so that no two runs of the same
stage share any per-call seed value (`tests/test_results.py` checks this against `raw_calls.jsonl`).

| directory | stage | produced by | what it is |
|---|---|---|---|
| `scientist-design-repeats-qwen3-8b-2026-11-11/base-seed-<seed>/` | **Qwen as scientist, redrawn** | `assr-replication generate-designs --base-seed <seed>` | ten more design draws, base seeds 20261114, 20261214, …, 20262014 (original: 20261014). Same files as `canonical-designs-*`, plus `provenance.json` |
| `generated-experiments-qwen3-8b-2026-11-11-seed-<seed>/` | **Qwen designs × Qwen subjects, redrawn** | `assr_replication.generated_cli --base-seed <seed>` | four more draws of all 108 cells, base seeds 2036101400, 2046101400, 2056101400, 2066101400 (original: 2026101400) |
| `frontier-planned-qwen-subjects-2026-11-11-seed-<seed>/` | **GPT-Sol designs × Qwen subjects, redrawn** | `assr-frontier full-run --base-seed <seed>`, then `analyze` | four more draws of all 108 cells, same base seeds as above, same design file and settings |

In the `generated-experiments-*-2026-11-11-*` directories, `finalizations/<study>/` holds only
`finalized.json`, copied byte-for-byte from the original run before the run started. The runner
treats an existing `finalized.json` as already compiled (its resume path), so the execution design is
held fixed and only the subject and measurement draws change; there are no compiler calls. Measurement
failures are kept as data (cells with `measurement_valid: false` or `status: failed`); unlike the
original frontier run, no failed measurement was recovered by a separate call.

`provenance.json` (every repeated-seed run): stage, seed, the exact command (home directory shown as
`~`), working directory, start and end time (UTC) and runtime in seconds.

The GPT-Sol design bundle itself is `designs/frontier-planned-v1/designs.json`. It was written in a
chat session with GPT Sol, not by code in `src/`, so there are no raw calls for the planning step.
The bail study in that bundle was edited once between the pilot and the full run (commit message:
"preserve fixed bail scenario"): the pilot's `design_bundle.json` still contains the earlier
"Alleged financial harm" factor; the full run's contains "Post-offense cooperation".

### File types

| file | format | contents |
|---|---|---|
| `manifest.json` | JSON | run settings: model, provider configuration, seeds, temperatures, retry policy, prompt hashes, counts, start/end times, runtime |
| `raw_calls.jsonl` | JSON lines | **every model call**, appended in order, including malformed and rejected ones. `request` = messages, model, temperature, `metadata` (seed, task, attempt, and agent/cell/turn where relevant); `response` = `text`, `error`, `raw_response` (CLI provider: stdout, stderr, return code; in-process provider: text, seed, temperature); `prompt_hash` = SHA-256 of model + messages + temperature; `call_hash` = SHA-256 of prompt hash + response |
| `transcript.jsonl` (smoke runs) | JSON lines | one line per turn: `agent`, `content`, `turn`, `call_hash` |
| `validation_report.json` (canonical) | JSON | per scenario: every attempt, its parse/validation status, the accepted design file and hash |
| `designs/NN-*.json` (canonical) | JSON | Qwen's accepted design: agents, outcome and measurement question, 1–3 causes with hypothesis, predicted direction and treatment values, public/private information, turn order, turn cap. Schema: `schemas/scientist-design-v1.schema.json` |
| `finalizations/<study>/initial.json` | JSON | the design as Qwen proposed it (identical to the canonical design) |
| `finalizations/<study>/finalized.json` | JSON | the design plus execution additions from a second Qwen "execution compiler" call: respondent, turn order, and where each treatment goes (public or one agent's private prompt) |
| `finalizations/<study>/diff.json` | JSON | machine diff of initial vs finalized; all four report zero substantive changes |
| `finalizations/<study>/failures.json` | JSON | rejected compiler attempts in the final invocation (mug: 3, auction: 3) with the error message; four earlier rejected auction calls appear only in `raw_calls.jsonl` |
| `finalizations/<study>/raw_calls.jsonl` | JSON lines | the compiler calls, including rejected ones |
| `factorials/<study>.json` | JSON | the 27 treatment cells: `cell_id`, level `indices`, `treatments` |
| `studies/<study>/finalized_design.json` | JSON | copy of the finalized design used for execution |
| `studies/<study>/cells.jsonl` | JSON lines | one line per cell: `cell_id`, `indices` (level 0/1/2 per cause), `treatments`, full `transcript` (each turn's agent, text, seed, call hash), `measurement` (`value` + transcript `evidence`), `measurement_valid`, `measurement_attempts`, `failures` |
| `studies/<study>/analysis.json` | JSON | the runner's own analysis: marginal means per level, high-minus-low contrasts, additive OLS, whether Qwen's predicted direction held |
| `checkpoint.json` | JSON | per-study completion and runtime |
| `design_bundle.json` (frontier) | JSON | the design bundle the run used (five three-level factors per study, each with public/private wording per level and GPT-Sol's textual prediction; outcomes with types) |
| `cell_plan.json` (frontier) | JSON | the 27 OA cells per study with treatment values and cell hashes |
| `prompt_config_snapshot.json` (frontier) | JSON | the exact subject and measurement prompt templates and settings |
| `run_events.jsonl` (frontier) | JSON lines | the latest invocation: cells started/completed/skipped/failed, runtime. The full run was resumed after an interruption, so this file and the manifest show only the last invocation (36 new, 72 skipped); per-cell `runtime_seconds` cover all 108 |
| `studies/<study>/cells.jsonl` (frontier) | JSON lines | as above plus `derived` outcomes (e.g. `deal`, `price`; `bail_amount`; `hire`, `rating`; `final_price`, `winner`, `winner_valuation`, `allocative_efficiency`, `efficient_allocation`), `public_instructions`, `record_hash`, `runtime_seconds` |
| `studies/<study>/analysis.json` (frontier) | JSON | marginal summaries of every outcome by factor level (runner output) |
| `analysis_with_recovery.json`, `studies/*/analysis_with_recovery.json` | JSON | the same analysis after merging the recovered interview measurement |
| `studies/frontier-lawyer-interview/measurement_recoveries.jsonl` | JSON lines | the one recovered measurement (cell `frontier-lawyer-interview-04`), linked to the original record hash |
| `studies/frontier-lawyer-interview/recovery_raw_calls.jsonl` | JSON lines | the raw recovery call (temperature 0, larger token allowance) |
| `exploratory_regressions.json` | JSON | the additive OLS with HC1 errors shown on the regression slide |
| `run-console.log` | text | console output of the last full-run invocation (download progress bars only) |

The recovery call, `analysis_with_recovery.json` and `exploratory_regressions.json` were produced
by one-off commands during the session, not by a command in `src/`. `analysis/make_results.py`
recomputes everything they report from `cells.jsonl` and the recovery record.

### Other folders in this case study

| path | contents |
|---|---|
| `docs/REPLICATION_PLAN.md` | the frozen plan for the first (all-Qwen) stage |
| `docs/runs/*.md` | the write-ups for each run (methods, results, caveats) |
| `docs/runs/*.html`, `*.pdf`, `*.png` | the slides and one-pagers: HTML source plus PDF and PNG renders |
| `docs/runs/2026-10-15-sol-qwen-regressions.csv` | the regression table on the slide; `tests/test_results.py` checks it against the cells |
| `docs/runs/frontier-results-graphic/` | the results graphic and its source notes |
| `designs/frontier-planned-v1/designs.json` | GPT-Sol's design bundle (input to the frontier runner) |
| `schemas/scientist-design-v1.schema.json` | the schema Qwen's designs had to satisfy |
| `vendor/original_scm/` | the four upstream SCM files from Manning et al.'s repository, byte-for-byte, with `LICENSE.txt` (MIT) and `PROVENANCE.md` (commit and SHA-256). Never shown to any model |

---

## Expected Parrot — `expected-parrot/.artifacts/`

Rebuilds Expected Parrot's public "Lasagna Coke" concept test locally: 12 synthetic personas in 3
segments answer 7 questions (4 ratings on 1–5, 3 open text).

### Run directories

| directory | engine | what it is |
|---|---|---|
| `demo/` | demo (rule-based, no model) | the built-in "Lasagna Cola" concept, seed 42 |
| `run-demo/` | demo | a custom concept ("Smoky Maple Seltzer"), seed 7 |
| `web-smoke/<study-id>/` | demo | one study created through the local web app |
| `mlx-smoke/` | MLX Qwen | one persona, pipeline check |
| `lasagna-cola-mlx-12/` | MLX Qwen | **first real panel**: 12 minimal personas. Every one of the 48 ratings came back as the midpoint 3 |
| `lasagna-cola-mlx-12-unanchored/` | MLX Qwen | the same panel after the prompt fix that discourages midpoint answers; this is the "original baseline" in the report |
| `expected-parrot-qwen-replication/` | MLX Qwen (in-process) | first run of `experiments/expected_parrot_qwen_replication.py`: the 12 public Expected Parrot personas under two conditions |
| `expected-parrot-qwen-replication-v2/` | MLX Qwen (in-process) | the full run: the same two conditions (identical outputs, same seeds) plus a third with the persona in the system prompt |
| `expected-parrot-qwen-replication-2026-11-11-seed-<seed>/` | MLX Qwen (in-process) | repeated-seed runs (November 2026): the same script and three conditions with `--seed` 43, 44, 45, 46 (original: 42), made by `case-studies/run_repeated_seeds.py`; plus `provenance.json` |
| `expected-parrot-one-slide/` | — | the summary slide: `comparison.html` source, PDF and PNG renders, `source-notes.md` |

### File types

| file | contents |
|---|---|
| `manifest.json` | study id, UTC time, engine and model, engine settings (temperature 0.7, max tokens, thinking off), runtime metadata, seed, concept, panel (persona attributes), questions, and **the exact prompt sent to every persona** |
| `responses.json` | one record per persona: `persona_id`, `segment`, parsed `answers`, **`raw_output`** (the unmodified model text), `error` |
| `summary.json` | counts, means overall and by segment, 1–5 distributions, representative quotes, warnings, limitations |
| `report.html` | self-contained HTML report of the above |
| `results.json` (replication) | model, seed, stimulus image hash and the text transcription used instead of the image, the cited public Expected Parrot targets, and per condition a `summary` plus `responses` (name, age, segment, persona text, parsed answers, **`raw_outputs`**: one string for the joint condition, one per question for the separate-call conditions) |
| `comparison.json` (replication) | the `summary` blocks side by side: overall and segment means, segment ranges, share of answers at 1 or 5 |

The stimulus image (`image_09ac93.png`, SHA-256 in `results.json`) is not distributed. The local
model is text-only, so the image was never sent to it; the transcription that was sent is stored
in `results.json` and in the script.

The Expected Parrot targets (segment means 4.75 / 1.00 / 1.00 appeal and 4.50 / 3.00 / 1.50
purchase intent; overall means 2.25 / 3.83 / 3.00 / 3.00) were transcribed from Expected Parrot's
public shared run. They are cited, not computed here.
