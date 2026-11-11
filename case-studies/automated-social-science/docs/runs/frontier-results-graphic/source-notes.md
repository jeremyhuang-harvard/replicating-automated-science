# Source notes — GPT Sol × Qwen comparison graphic

## Current run

Primary data: `outputs/frontier-planned-qwen-subjects-2026-10-14/analysis_with_recovery.json` and per-study `cells.jsonl` files.

- Design: GPT Sol frontier reasoning in Hermes; five three-level factors per study. The planner was not blinded to prior project context, the paper summary, or the earlier all-Qwen results.
- Assignment: pairwise-balanced OA(27,5,3,2), 27 cells per study and 108 total.
- Subjects and measurement: `mlx-community/Qwen3-8B-4bit`; subject temperature 0.3, measurement temperature 0, thinking disabled.
- Completion: 108/108 interactions and 108/108 measurements after one separately logged recovery call.
- Recorded model-cell runtime: 2,412.01 seconds (40.20 minutes).
- Mug: deal rate 51.85%; conditional mean recorded agreement price $14.36.
- Bail: mean bail $22,222.
- Interview: hire rate 62.96%; mean rating 72.41/100.
- Auction: mean final price $171.85; mean allocative efficiency 53.70%; efficient-allocation rate 40.74%; 10/27 no-sale outcomes; three cells named winners at prices above their assigned valuations.

## Earlier all-Qwen run

Primary data: `docs/runs/2026-10-14-qwen3-8b-generated-experiments.md`.

- Qwen designed, enacted, and measured four studies.
- Three factors × three values per study; 108 total cells.
- 106/108 valid measurements.
- Nine of twelve predicted directions agreed with descriptive simulated results.

## Manning, Zhu, and Horton comparison

Primary references: arXiv `2404.11794` and the authors’ repository `KeHang-Zhu/lm-automated-social-science`.

- Four studies: mug bargaining, bail setting, lawyer interview/hiring, and art auction.
- Original released designs contain 1,073 treatment cells in total.
- The paper’s auction design varied bidder valuations and recovered the central second-highest-value logic.
- The hiring design included bar passage, interviewer friendliness, and height; bar passage was consequential while friendliness and height were comparatively weak.

## Interpretation boundary

These outputs describe one local model draw per orthogonal-array cell. They are not estimates of human behavior. Differences across the paper, all-Qwen run, and GPT-Sol-planned run combine changes in scientist model, design, prompts, and simulation procedure; they are descriptive rather than a clean causal model-capability estimate.
