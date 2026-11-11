# Automated Social Science Replication

A durable, provider-neutral scaffold for a discovery-oriented conceptual replication of *Automated Social Science: Language Models as Scientist and Subjects*. The primary study asks what agents, outcomes, causal hypotheses, treatments, information structures, and measurements a local Qwen model independently constructs from four short canonical social scenarios. Matching the source paper's released designs or findings is not the target.

The upstream source is frozen at commit `c694c28190464a36349638e7d50f9291d081760d`. Its four example SCM JSON files are vendored byte-for-byte under `vendor/original_scm/` as provenance and optional later comparison benchmarks. They are not supplied to the scientist model.

## Frozen first-stage scope

The entire first-stage scenario set is:

1. `two people bargaining over a mug`
2. `a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud`
3. `A person is interviewing for a job as a lawyer`
4. `3 bidders participating in an auction for a piece of art starting at fifty dollars`

The first stage generates one first-valid structured design per scenario. Novel scenarios and repeated scientist generations are later extensions, not part of this run.

## Requirements

- macOS or Linux
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.11
- Optional for local inference: Apple silicon and `mlx-lm`

No API key is required for setup, tests, deterministic smoke, or local inference.

## Setup and tests

```bash
git clone https://github.com/jeremyhuang-harvard/replicating-automated-science.git
cd replicating-automated-science/case-studies/automated-social-science
uv python install 3.11
uv sync --dev
uv run pytest -q
```

## Deterministic smoke

```bash
rm -rf outputs/mock-smoke
uv run assr-replication smoke --output-dir outputs/mock-smoke
```

This writes `manifest.json`, `raw_calls.jsonl`, and `transcript.jsonl` without network calls or model charges.

## Local Qwen scientist-design run

Install the tested MLX CLI and run the four frozen scenarios:

```bash
uv tool install mlx-lm==0.32.0
rm -rf outputs/canonical-designs-qwen3-8b
uv run assr-replication generate-designs \
  --output-dir outputs/canonical-designs-qwen3-8b \
  --model mlx-community/Qwen3-8B-4bit \
  --max-tokens 2200 \
  --base-seed 20241001
```

The command fixes temperature at `0.3`, disables thinking, assigns a distinct seed to every attempt, and allows up to three attempts per scenario only for malformed JSON or schema-invalid output. It preserves every raw attempt, accepts the first valid design, and never ranks valid candidates.

Expected artifacts:

```text
outputs/canonical-designs-qwen3-8b/
  manifest.json
  raw_calls.jsonl
  validation_report.json
  designs/
    01-mug-bargaining.json
    02-bail-setting.json
    03-lawyer-interview.json
    04-art-auction.json
```

The frozen structured contract is `schemas/scientist-design-v1.schema.json`. Each design includes at most four agents; one operationalized outcome and measurement question/respondent; one to three causes with predicted directions and two to three concrete treatment values; public/private information; agent goals and constraints; interaction order; and at most eight turns.

Local inference has no per-call API charge but uses disk, memory, electricity, and time. `Qwen3-8B-4bit` is an exploratory local baseline, not a frontier-equivalent substitute for historical GPT-4 behavior.

## Frontier-planned generalized runner

The fixed bundle at `designs/frontier-planned-v1/designs.json` uses five three-level factors per study and a deterministic finite-field OA(27,5,3,2). The dedicated CLI validates all 108 cells, supports a four-cell pilot (one cell per study), resumes append-only runs, and analyzes typed multi-outcome records:

```bash
uv run assr-frontier validate
uv run assr-frontier dry-run --output-dir outputs/frontier-dry-run-2026-10-14

# Mock pilot (no model load)
uv run assr-frontier pilot \
  --provider mock \
  --output-dir outputs/frontier-mock-verification-2026-10-14

# Real local-Qwen pilot: exactly four cells, isolated from the full-run directory
PYTHONPATH=src ~/.local/share/uv/tools/mlx-lm/bin/python \
  -m assr_replication.frontier_cli pilot \
  --output-dir outputs/frontier-pilot-2026-10-14 \
  --model mlx-community/Qwen3-8B-4bit \
  --wall-clock-budget-seconds 3600

# Full 108-cell run (do not launch until approved)
PYTHONPATH=src ~/.local/share/uv/tools/mlx-lm/bin/python \
  -m assr_replication.frontier_cli full-run \
  --output-dir outputs/frontier-planned-qwen-subjects-2026-10-14 \
  --model mlx-community/Qwen3-8B-4bit \
  --wall-clock-budget-seconds 3600

uv run assr-frontier analyze \
  --output-dir outputs/frontier-planned-qwen-subjects-2026-10-14
```

Subject turns use temperature `0.3`; measurement uses `0`; thinking, tools, and retrieval are disabled. Agent prompts contain only the scenario, that agent's role/goal/constraints, public instructions and transcript, and that agent's private instructions. Raw provider calls and malformed attempts are retained. Re-running the same plan/config skips every terminal cell record.

## Execute the generated factorials

The bounded execution runner mechanically finalizes the four saved designs, verifies 27 unique cells per study and 108 total, and then runs append-only/resumable local interactions plus separate strict-JSON measurements. Because `mlx-lm` is installed as a uv tool, invoke the module with that tool's Python while exposing this repository's `src` directory:

```bash
PYTHONPATH=src ~/.local/share/uv/tools/mlx-lm/bin/python \
  -m assr_replication.generated_cli \
  --initial-dir outputs/canonical-designs-qwen3-8b-2026-10-14/designs \
  --output-dir outputs/generated-experiments-qwen3-8b-2026-10-14 \
  --model mlx-community/Qwen3-8B-4bit \
  --base-seed 2026101400
```

Subject turns use temperature 0.3; measurement uses temperature 0; thinking, tools, and retrieval are disabled. Re-running the same command skips terminal cell records. See `docs/runs/2026-10-14-qwen3-8b-generated-experiments.md` for the completed 108-cell run and its one-draw local-model analysis.

## Package layout

```text
schemas/                    versioned scientist-design JSON schema
docs/                       replication plan and run reports
src/assr_replication/
  models.py                 provider-neutral request/response protocol
  providers.py              deterministic mock and local MLX adapter
  call_log.py               append-only raw provider-call log
  hashing.py                canonical SHA-256 hashes
  scientist_design.py       structured design validator and design hash
  scientist_prompt.py       neutral frozen scientist prompt
  scientist_runner.py       first-valid four-scenario generation workflow
  generated_experiments.py  finalization, factorials, execution, and analysis
  generated_cli.py          persistent-MLX generated-experiment runner
  frontier_runner.py        fixed-bundle OA, privacy-safe execution, typed analysis
  frontier_cli.py           validate/dry-run/pilot/full-run/analyze commands
  smoke.py                  tiny bargaining smoke workflow
  cli.py                    command-line entry points
tests/                      offline behavioral tests
vendor/original_scm/        immutable upstream provenance inputs
```

## Current boundary

The repository now includes one bounded execution of the four generated factorial designs. Its numerical results are explicitly descriptive one-draw local-model evidence, not human behavior or a numerical replication of the source paper.
