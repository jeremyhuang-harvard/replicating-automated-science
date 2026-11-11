# Research notes: the reference workflow

**Provenance.** These notes come from independently inspecting the public shared conversation and its downloadable study files on October 14, 2026, plus Expected Parrot's public product pages, EDSL documentation, and open-source repository. The share page's rendered route briefly showed an error, but its public share API returned the full conversation and files.

## Observed reference workflow

1. **Input:** a product concept, with an image, for a novelty beverage ("Lasagna Coke").
2. **Panel:** 12 synthetic personas split across 3 segments.
3. **Survey:** 7 questions (numeric ratings plus open text).
4. **Model:** one vision-capable LLM answered every question for every persona.
5. **Persistence:** structured results were saved as files.
6. **Analysis:** results were broken out by segment.
7. **Report:** an HTML report summarized the findings.

**Cost observed in the shared run: $0.22.**

This baseline mirrors the shape (concept -> 12 personas / 3 segments -> 7 questions -> one model -> persisted results -> segment analysis -> HTML report) while staying local and text-only.

## Reference design bug: lost concept context

The survey ran with **memory off**, so each question was an independent model call. Only the **first question** repeated the concept and image. Later questions, especially the **free-text** ones, were sent without them. Several later answers therefore said, in effect, that the respondent lacked the concept, or fell back on generic statements that could apply to any product.

Consequences:

- Open-text answers were partly unrelated to the concept, so quotes and themes drawn from them were unreliable.
- Rating items and free-text items were not conditioned on the same information, so they could disagree.
- The failure was silent: it looked like low-quality "opinions," not a pipeline error.

**Fix adopted here.** Each persona gets one self-contained prompt with the full concept text and all questions, and returns one JSON object containing every answer (`parrot_baseline/prompts.py`). Tests assert that every prompt contains the complete concept and every question, both in isolation and for the prompts actually sent by the runner. Future image support must attach the image to that single call too.

## Why simulated panels need human calibration

Simulated respondents are samples from a language model conditioned on a persona description, not samples from a consumer population. So:

- **No ground truth.** Nothing ties a synthetic "4 out of 5 purchase intent" to purchases. Absolute levels are uncalibrated.
- **Model bias.** Models tend toward agreeableness, narrower variance than people, stereotyped or generic answers, and fluent but wrong reasoning.
- **Designer-defined segments.** Segment differences are largely what the prompt author wrote into the personas, so they cannot confirm that real segments differ that way.
- **Prompt and model sensitivity.** Rewording or swapping models can change rankings.
- **Small n.** 12 personas gives 4 per segment; any means are anecdotal.

Hence the position of this project: simulated panels are **hypothesis generators**. They are cheap for ranking ideas, surfacing objections and sharpening questions. Before acting, run a **human calibration** (same survey on real respondents for several concepts, compare rank order and error) and a **holdout validation** (calibrate on some concepts, test on unseen ones). See the validation roadmap in [README.md](README.md).

## Expected Parrot: open-source core vs product layer

- **EDSL** (Expected Parrot's library for surveys, agents, language models and results) is open source, so its behavior can be read directly.
- The **Research Agent / product layer** adds orchestration around it: file handling, approvals, cost estimates, execution, analysis and report generation.

This baseline reimplements only the thin pipeline above. It does not reuse their code or claim parity with the product.

## Why REA is optional here

A research agent (REA) could help with authorized browser, network and static-JavaScript observation of the hosted product. It is unnecessary for this baseline because the public workflow, the share API, the exported files, the documentation and the EDSL source already expose enough behavior to reproduce the pipeline's structure and to identify the context-loss bug. Any such observation should stay within what the account holder is authorized to inspect, and is not needed to build or validate this project.

## Primary sources

- Expected Parrot shared Lasagna Coke run: https://www.expectedparrot.com/research-agent/share/xOkTNsma5I0hWAm67yFYuQ
- Research Agent resources: https://www.expectedparrot.com/products/research-agent/use-cases
- EDSL documentation: https://docs.expectedparrot.com/en/latest/
- EDSL source: https://github.com/expectedparrot/edsl
- REA source: https://github.com/morluto/rea

