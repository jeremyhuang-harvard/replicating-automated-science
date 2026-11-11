# Frontier-planned runner checkpoint and pilot — 2026-10-14

## Scope

Implemented the generalized runner for `designs/frontier-planned-v1/designs.json`. The scientific bundle was committed exactly as received; no scenario, factor, level, prediction, agent, interaction, or outcome content was edited.

The full 108-cell experiment was **not** run. The intended full-run directory, `outputs/frontier-planned-qwen-subjects-2026-10-14`, was not created.

## Deterministic verification

```bash
uv run assr-frontier validate --design-file designs/frontier-planned-v1/designs.json
```

```json
{"bundle_hash":"75555f4232d939f9708f5e8cc572a35b13f640c47d874fbe76b8c27f9b79645c","cells_per_study":27,"studies":4,"total_cells":108,"valid":true}
```

The OA tests verify, for every study: 27 unique cells; every factor level occurs 9 times; every ordered level pair for every pair of factors occurs exactly 3 times; 108 cells total.

Final test/build commands:

```bash
uv run pytest -q
uv run python -m compileall -q src tests
uv build
```

```text
.......................................                                  [100%]
39 passed in 0.04s
Successfully built dist/automated_social_science_replication-0.1.0.tar.gz
Successfully built dist/automated_social_science_replication-0.1.0-py3-none-any.whl
```

Bounded mock command:

```bash
uv run assr-frontier pilot \
  --design-file designs/frontier-planned-v1/designs.json \
  --output-dir outputs/frontier-mock-verification-2026-10-14 \
  --provider mock \
  --base-seed 2026101400 \
  --wall-clock-budget-seconds 60
```

```json
{"cells_completed":4,"cells_failed":0,"cells_skipped":0,"cells_started":4,"runtime_seconds":0.0032103339908644557,"status":"complete"}
```

## Real local-Qwen pilot

Command:

```bash
PYTHONPATH=src ~/.local/share/uv/tools/mlx-lm/bin/python \
  -m assr_replication.frontier_cli pilot \
  --design-file designs/frontier-planned-v1/designs.json \
  --output-dir outputs/frontier-pilot-2026-10-14 \
  --provider mlx \
  --model mlx-community/Qwen3-8B-4bit \
  --max-tokens 384 \
  --base-seed 2026101400 \
  --wall-clock-budget-seconds 3600
```

Final result:

```json
{"cells_completed":4,"cells_failed":0,"cells_skipped":0,"cells_started":4,"runtime_seconds":88.47592620900832,"status":"complete"}
```

Per-study terminal records:

| Study | Cell | Cell runtime (s) | Turns | Measurement attempts | Derived result |
|---|---:|---:|---:|---:|---|
| mug bargaining | 01 | 16.3258 | 8 | 1 | no deal; price null |
| bail setting | 01 | 30.4264 | 8 | 1 | bail $10,000 |
| lawyer interview | 01 | 33.3274 | 6 | 1 | no hire; rating 45 |
| art auction | 01 | 8.3954 | 8 | 1 | price $110; Bidder C; allocative efficiency 1.0 |

Subject calls used temperature `0.3`; measurement calls used `0`; all call metadata recorded thinking, tools, and retrieval disabled. Raw calls and cell records remain under ignored `outputs/` directories and were not committed.

### Pilot-discovered parser defect and fix

The first pilot completed all interactions but rejected all four measurements because Qwen returned a strict two-key object with `evidence` as a global excerpt list, while parser v1 required an outcome-keyed evidence object. All malformed/rejected attempts were preserved in `outputs/frontier-pilot-2026-10-14-schema-v1-failed`.

A failing regression test was added first, then the parser contract was corrected to accept either a non-empty global evidence string/list or an exact outcome-keyed evidence object. Prompt snapshot version advanced to `frontier-prompts-v3`, retries now include validation feedback and the rejected output, and the final pilot above passed all four cells on the first measurement attempt.

### Model-behavior caveat

In the auction pilot all bidders had a private maximum valuation of $100, but the transcript ended with Bidder C bidding $110. This is model noncompliance with an explicit agent constraint, not a runner/parsing failure; the raw transcript is preserved for review.
