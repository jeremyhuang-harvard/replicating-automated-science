# Qwen3-8B Canonical Scientist-Design Run — 2026-10-14

## Methods

The run used `mlx-community/Qwen3-8B-4bit` through `mlx_lm.generate` on the four frozen canonical scenario strings. The scientist prompt contained only each short scenario, the `scientist-design-v1` structure, and neutral computational constraints. It did not contain the paper title, author names, released variables, findings, or target effects.

Generation settings were temperature `0.3`, thinking disabled, maximum 2,200 output tokens, and distinct seeds `20261014` through `20261017`. The runner allowed at most three attempts per scenario only for malformed JSON or schema-invalid output, preserved every raw call, and accepted the first valid design without ranking or repair.

Artifacts are under `outputs/canonical-designs-qwen3-8b-2026-10-14/`. No generated factorial experiment was executed.

## Validation results

- Four of four scenarios produced a schema-valid design.
- All four designs were valid on attempt 1; total attempts: 4.
- No malformed, schema-invalid, or provider-error attempts occurred.
- Saved design hashes were recomputed successfully from the parsed files.
- An inspection of the recorded prompts found none of the forbidden paper-specific names or title/model hints tested by the suite.

## Proposed studies

### 1. Mug bargaining

Qwen proposed four agents: seller Alice, buyer Bob, an observer, and a mediator. The outcome was the final agreement price, measured from the buyer and seller. Proposed causes were the seller's initial offer (`$6`, `$8`, `$10`), buyer budget (`$5`, `$7`, `$9`), and mediator presence/status (`present`, `absent`, `neutral`). It predicted higher initial offers would increase price, lower budgets would decrease price, and mediator status would have a non-monotonic relationship with price. The protocol allowed eight turns.

Inspection note: the design is structurally valid but has substantive ambiguities. Buyer budget is listed as public while maximum willingness to pay is private, the fixed buyer constraint says at most `$7` while a `$9` budget treatment is proposed, and “mediator neutral” is not cleanly parallel to present/absent.

### 2. Bail setting

Qwen proposed a judge, defense attorney, and prosecutor. The outcome was the exact bail amount set by the judge. Proposed causes were tax-fraud severity (`$50,000`, `$100,000`, `$200,000`), prior convictions (`0`, `1`, `3`), and evidence strength (`weak`, `moderate`, `strong`), all predicted to increase bail as their levels rise. The protocol used three ordered turns ending with the judge's decision.

Inspection note: varying fraud severity changes a quantity fixed in the scenario seed, and some information assignments—such as the judge privately holding the defendant's financial status and prior bail decisions—would need clearer provenance before execution.

### 3. Lawyer interview

Qwen proposed an interviewer, candidate, and note-taker. The outcome was interview success, operationalized as the interviewer's 1–10 suitability rating. Proposed causes were legal knowledge, communication clarity, and confidence, each with low/intermediate/high-style levels and a predicted positive direction. The protocol used three ordered turns and ended after answers and transcription.

Inspection note: the proposed causes are plausible candidate attributes but require concrete manipulation text before subject execution. Several private-information assignments are also unusual, including the candidate knowing internal hiring biases and the note-taker privately receiving non-verbal cues while being constrained to record only verbal responses.

### 4. Three-bidder art auction

Qwen proposed three bidders: a conservative collector capped below 100, a speculative investor capped at 200, and a private collector capped at 300. The outcome was the highest final bid. Proposed causes were bidder risk tolerance (`low`, `medium`, `high`), starting price (`50`, `100`, `150`), and bidding strategy (`conservative`, `aggressive`, `rational`), all predicted to increase final price across the stated ordering. The protocol gave each bidder one position in a three-turn order.

Inspection note: the starting-price treatment varies the seed's stated fifty-dollar start, “rational” is not an ordinal high level parallel to conservative/aggressive, and a single three-turn pass is closer to sealed bids than an ascending auction.

## Result boundary

Schema validity established that the four outputs satisfy the frozen structural and computational constraints; it did not establish causal coherence or readiness for experiment execution. The inspection notes above should inform a separately frozen execution validator or protocol-instantiation milestone. No design was edited, replaced, or selected based on these observations.
