# Expected Parrot Baseline

A local-first, standard-library-only baseline for AI-persona consumer concept simulation. It is **not** a clone of Expected Parrot's proprietary internals. It separates **hypothesis generation** (cheap, simulated) from **human validation** (required), and records prompts, personas, engine settings, raw responses, analysis and available runtime metadata so runs can be audited and approximately repeated. Exact reproduction can still depend on the model files, MLX version, hardware and platform.

> **Simulated hypotheses, not human evidence.** Every artifact, including the HTML report, carries this label.

The reference case is a novelty beverage concept similar to "Lasagna Coke" (see [RESEARCH.md](RESEARCH.md)).

## Requirements

- Python 3.11+ (stdlib only; no `pip install` needed to run or test).
- Optional: `mlx_lm.generate` on PATH (`pip install mlx-lm`, Apple silicon) and the model `mlx-community/Qwen3-8B-4bit` for the `mlx` engine.

## Commands

Run from the repository root.

```bash
# Full test suite (126 tests, offline)
python3 -m unittest discover

# Offline demo study with the built-in "Lasagna Cola" concept
python3 -m parrot_baseline.cli demo --output-dir .artifacts/demo

# Your own concept (demo engine, offline, deterministic)
python3 -m parrot_baseline.cli run \
  --concept-title "Smoky Maple Seltzer" \
  --concept-description "A sparkling water with smoked maple and a hint of black pepper, no sweeteners." \
  --engine demo --output-dir .artifacts/run-demo --panel-size 12 --seed 7

# Local web app (demo engine only), then open http://127.0.0.1:8765/
python3 -m parrot_baseline.web --port 8765

# Optional: real local LLM via MLX (slow: one generation per persona)
python3 -m parrot_baseline.cli run \
  --concept-title "Smoky Maple Seltzer" \
  --concept-description "A sparkling water with smoked maple and a hint of black pepper, no sweeteners." \
  --engine mlx --output-dir .artifacts/run-mlx --panel-size 12 --seed 7
# optional: --model <other mlx model id>   (default mlx-community/Qwen3-8B-4bit)
```

`python3` must resolve to 3.11+ (on some machines use `python3.11`).

## Output files

Each run writes four files into `--output-dir`:

| File | Contents |
|---|---|
| `manifest.json` | study id, UTC timestamp, engine/model and full engine settings, available software/runtime metadata, seed, concept, panel, questions, and the exact prompt sent to every persona |
| `responses.json` | one record per persona: parsed answers, **raw model output**, and `error` if the output was unusable |
| `summary.json` | counts, means (overall and by segment), 1-5 rating distributions, representative quotes, warnings, limitations |
| `report.html` | self-contained report (inline CSS, HTML/CSS charts, no scripts or external requests) |

The web app stores each study under `.artifacts/web/<study-id>/` and serves only these four files.

## Architecture

```
parrot_baseline/
  schemas.py     Concept, Persona, Question, Response, StudyManifest (frozen dataclasses + validation)
  panel.py       default 12-person panel, 3 segments, attitude-defined (no demographics)
  survey.py      default 7 questions: 4 ratings (1-5) + 3 open text
  prompts.py     one self-contained prompt per persona: full concept + every question
  engines.py     DemoEngine (deterministic rules + seeded noise), EngineError
  mlx_engine.py  MlxEngine: argv-list subprocess to mlx_lm.generate, prompt via stdin, no shell
  parsing.py     extract one JSON object from noisy output; validate/coerce answers
  analysis.py    means, distributions, quotes, warnings, limitations
  report.py      HTML/CSS report, all dynamic text escaped
  runner.py      run_study(): prompts -> engine -> parse -> summarize -> persist
  cli.py         `demo` and `run` commands
  web.py         stdlib http.server UI (GET /, POST /run, whitelist file serving)
tests/           unittest suite, one module per slice
```

Data flow: `Concept + Panel + Questions -> prompts -> Engine.generate (raw text) -> parse_answers -> Responses -> summarize -> artifacts`. Engines only return text; parsing and validation are shared, so demo and MLX outputs are treated identically. A single persona's bad output becomes an error `Response` plus a warning; an engine-level failure (missing executable, non-zero exit, timeout) aborts before anything is written.

## Methodology

- **Panel:** 12 personas, 4 per segment: `adventurous_foodie`, `mainstream_drinker`, `skeptical_health_conscious`. Segments differ by explicit product attitudes (novelty seeking, brand loyalty, health concern, price sensitivity, each 1-5), never by age, gender, ethnicity, religion, income or similar traits.
- **Survey:** `appeal`, `clarity`, `uniqueness`, `purchase_intent` (integers 1-5) and `first_reaction`, `concerns`, `improvement` (free text).
- **Context rule:** every persona gets one prompt containing the complete concept text and all questions, and returns ONE JSON object with all answers. No answer relies on conversational memory (see the reference bug in RESEARCH.md).
- **DemoEngine:** transparent rules. Concept keyword cues (novelty, familiarity, health positioning) interact with persona attitudes; seeded noise (`seed|persona|concept`) adds variation. It demonstrates the pipeline and is repeatable; it is *not* a model of real consumers.
- **MlxEngine:** runs `mlx_lm.generate --model mlx-community/Qwen3-8B-4bit --prompt - --chat-template-config '{"enable_thinking": false}' ...`. Parsing tolerates code fences, `<think>` blocks, banners, trailing commas and chatter; errors include a snippet of the offending output.
- **Aggregation:** means and 1-5 distributions over valid responses only; quotes come from the respondent whose first rating is closest to the segment mean.

## Limitations

- Demo output is rule-generated, so segment differences are built in; they show the report works, not what consumers think.
- Model output reflects prompt wording and training data; expect agreeableness, narrow variance and confident errors.
- Rating levels are uncalibrated; use for relative comparison between concepts, never as forecasts.
- 4 personas per segment is anecdotal; the report warns on segments with n < 5.
- Text-only: no image/vision input yet. Panels larger than 12 repeat base personas (noise differentiates them).
- The web app is for local use: demo engine only, binds to 127.0.0.1 by default, uses a per-server CSRF token, validates Host/Origin, limits concurrent requests and refuses non-loopback binds unless `--allow-network` is supplied. It still has no authentication.

## Validation roadmap

1. **Repeatability:** fixed seeds, recorded prompts and engine/runtime settings; rerun under a matching environment and compare rankings rather than assuming bit-for-bit reproduction.
2. **Prompt sensitivity:** paraphrase questions and persona wording; flag conclusions that flip.
3. **Human pilot:** run the same survey on a small real panel (e.g. 30-100 per segment) for 3+ concepts with known outcomes.
4. **Calibration:** compare simulated vs human means and rank order; fit a monotone mapping per question; report rank correlation and error.
5. **Holdout:** calibrate on some concepts, evaluate on unseen ones before trusting any simulated result.
6. **Diversity checks:** compare response variance and open-text themes against humans; add personas where the simulation is too uniform.
7. **Vision input and structured segments** after the text pipeline validates.

## Compared with Expected Parrot

| | This baseline | Expected Parrot |
|---|---|---|
| Scope | Small, local, auditable concept-test pipeline | Open-source EDSL library plus a product layer (Research Agent) for orchestration, files, approvals, cost estimates, execution, analysis and reporting |
| Models | Offline demo engine; optional local MLX model | Many hosted models, including vision |
| Cost / privacy | Free, nothing leaves the machine | Metered hosted runs |
| Validation | Roadmap only; labels output as hypotheses | Same fundamental need for human calibration |

This project is an independent baseline built from public behavior; it uses no Expected Parrot code. See [RESEARCH.md](RESEARCH.md).
