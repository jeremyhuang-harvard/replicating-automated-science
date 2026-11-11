# Discovery-Oriented Replication Plan

## 1. Research objective

This project adapts the automated-social-science pipeline to study the **scientific choices made by a local Qwen model**. Given only a short social scenario plus neutral structural and computational constraints, Qwen independently chooses agents, outcomes, causes, treatment values, information boundaries, goals, and interaction protocols. Matching the source paper's released variables, designs, or findings is not the target.

This is a system-level conceptual replication, not an exact historical or coefficient replication. The vendored source designs under `vendor/original_scm/` are provenance and optional later benchmarks; they are never supplied to the scientist prompt.

## 2. Frozen first-stage scenario set

The entire first-stage design-generation set is exactly these four short canonical scenario seeds:

1. `two people bargaining over a mug`
2. `a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud`
3. `A person is interviewing for a job as a lawyer`
4. `3 bidders participating in an auction for a piece of art starting at fifty dollars`

There are no novel scenarios and no repeated scientist generations in the first stage. Novel scenarios, prompt-transfer checks, and repeated generations for stability or diversity analysis are later extensions, to be frozen separately after inspection of this milestone.

## 3. First-stage scientist generation

Run one accepted scientist design per scenario with `mlx-community/Qwen3-8B-4bit` through MLX:

- temperature `0.3`;
- thinking disabled through the chat-template configuration;
- distinct recorded seed for every attempt;
- at most three attempts per scenario;
- retry only when output is malformed JSON or violates the frozen schema;
- stop on provider errors rather than disguising them as formatting retries;
- accept the first schema-valid design without ranking, comparison, repair, or cherry-picking;
- preserve every raw attempt before parsing or validation.

The scientist receives only the exact short scenario and the neutral schema/computational constraints. The prompt must not name the paper, its authors, its released variables, or its findings.

## 4. Frozen structured design contract

The versioned contract is `schemas/scientist-design-v1.schema.json`. A valid record contains:

- the exact scenario string and schema version;
- one to four named agents, each with a role, goal, and constraints;
- one outcome with an operationalization, measurement question, and respondent;
- one to three proposed exogenous causes;
- a hypothesis, predicted direction, and two to three distinct concrete treatment values for each cause;
- public information and agent-specific private information;
- an interaction order, stopping rule, and turn cap from one to eight.

Generated records outside these bounds are retained as invalid attempts and are not silently rewritten.

## 5. Required run artifacts

Each first-stage run directory contains:

- `manifest.json`: model, provider settings, seeds, prompt hashes, scenario-set hash, retry policy, timestamps, counts, and artifact names;
- `raw_calls.jsonl`: append-only provider-neutral requests and unmodified MLX process outputs for every attempt;
- `validation_report.json`: attempt-level parse/validation status and accepted design hashes;
- `designs/01-mug-bargaining.json`;
- `designs/02-bail-setting.json`;
- `designs/03-lawyer-interview.json`;
- `designs/04-art-auction.json`.

The runner refuses to overwrite an existing run directory.

## 6. Primary first-stage results

The first-stage results are descriptive: what study Qwen proposed for each canonical seed, whether each first attempt was valid, what failures occurred, and whether artifact accounting is complete. The four designs are not scored by similarity to the source paper.

No generated factorial experiment is executed in this milestone. Subject simulation, outcome extraction, and effect estimation begin only after the generated designs and methods/results artifact are reviewed.

## 7. Reproducibility and anti-cherry-picking rules

- Freeze scenario strings, prompt, schema, model, temperature, and validation bounds before generation.
- Preserve all raw outputs, including malformed, invalid, refused, and failed attempts.
- Report attempt and validity denominators.
- Select by first-valid run order only.
- Hash exact prompts and accepted designs.
- Do not use source-paper signs, variables, or design similarity to tune prompts or validators.
- Keep the scientist-design stage separate from later subject execution.

## 8. Later extensions

Only after the first-stage artifact is reviewed should the project consider:

- repeated scientist generations for stability and diversity;
- newly written and frozen scenarios;
- execution of the first-valid generated factorial designs;
- repeated subject interactions within cells;
- other local or hosted model conditions;
- frozen execution or descriptive comparison of the vendored source designs.

These extensions require their own frozen sampling and execution rules and must not be retroactively folded into the first-stage denominator.
