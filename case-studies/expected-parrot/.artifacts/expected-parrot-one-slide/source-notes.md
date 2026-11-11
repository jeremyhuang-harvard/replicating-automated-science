# Source notes

Data are drawn from the repository's verified local artifacts:

- `experiments/expected_parrot_qwen_replication_report.md`
- `.artifacts/expected-parrot-qwen-replication-v2/comparison.json`
- `.artifacts/lasagna-cola-mlx-12-unanchored/summary.json`

Definitions:
- Original baseline: local Qwen3-8B-4bit, minimal personas, seven questions jointly with full text context.
- Rich-persona replication: same 12 public Expected Parrot biographies, Qwen3-8B-4bit, seven questions jointly with full textual stimulus context.
- EP-like separate calls: same personas, seven independent calls, concept context only in Q1.
- Public Expected Parrot: published Lasagna Coke run using Claude Sonnet 5 and image input.

All ratings are on a 1–5 scale. These are synthetic persona outputs, not population estimates.
