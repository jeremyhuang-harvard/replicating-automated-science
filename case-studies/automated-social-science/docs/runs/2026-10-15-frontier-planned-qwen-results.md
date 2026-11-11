# GPT Sol scientist × local-Qwen subjects: four generated experiments

## Design

GPT Sol used the four short scenario settings from Manning, Zhu, and Horton—mug bargaining, bail setting for a defendant charged with $50,000 in tax fraud, a lawyer job interview, and a three-bidder art auction beginning at $50—but independently chose the outcomes, causal factors, treatment values, predictions, agent roles, information boundaries, and analysis plan rather than being constrained to the paper’s released causal models. This planning pass was **not blinded**: GPT Sol had access to the prior project context and summaries of the paper and earlier all-Qwen run, which likely influenced choices such as retaining height as a placebo in hiring. Each study used five three-level factors and a deterministic `OA(27,5,3,2)` orthogonal array, yielding 27 pairwise-balanced conditions per study and 108 interactions total rather than the full 243-cell factorial per study.

Local `mlx-community/Qwen3-8B-4bit` generated all agent turns and measured the resulting public transcripts. Each agent received only the scenario, its role, goal and constraints, public instructions, its own private treatments, and the public transcript; it did not receive GPT Sol’s predictions, analysis plan, the paper’s findings, or another agent’s private information. Agent calls used temperature 0.3, measurement used temperature 0, and model thinking was disabled. There was one model interaction per cell, so all findings are descriptive model behavior rather than human evidence or repeated estimates of a model response distribution.

## Completion and validity

- 108/108 planned interactions completed.
- 107 measurements passed during the main run. One interview measurement returned truncated JSON on three attempts; a separately logged recovery call with a larger token allowance produced a valid result, giving 108/108 measured cells.
- Recorded cell runtime totaled 2,412.01 seconds, or 40.20 minutes, excluding implementation and one recovery call.
- The collection resumed safely after one background-process interruption because cell records and raw calls were append-only.
- All requests, transcripts, treatment assignments, malformed responses, retries, hashes, and recovery calls remain under `outputs/frontier-planned-qwen-subjects-2026-10-14/`.

## Results

### Mug bargaining

The 27 conditions produced a 51.9% agreement rate and a mean recorded agreement price of $14.36. Increasing the buyer’s budget from $6 to $18 raised agreement from 22.2% to 66.7% and the conditional mean price from $8.50 to $18. Increasing the seller’s reservation price from $4 to $16 lowered agreement from 77.8% to 33.3%, while outside options and proposal rights also shifted prices and agreement rates substantially. Seller attachment did not show the predicted monotonic reduction in agreement or increase in price.

### Bail setting

Mean bail was $22,222. Prior convictions produced a clear increase from $13,889 with no convictions to $29,444 with four convictions; low-to-high flight risk increased mean bail from $17,500 to $31,111; and weak-to-strong evidence increased it from $16,111 to $28,333. Strong community ties reduced mean bail relative to weak ties, as predicted. Cooperation was non-monotonic: extensive cooperation lowered bail, but partial cooperation generated higher mean bail than no cooperation.

### Lawyer interview

The model recommended hiring in 63.0% of conditions, with a mean rating of 72.4 out of 100. Low-to-high communication clarity increased the mean rating from 57.2 to 81.1 and the hire rate from 33.3% to 77.8%. Moving from not having passed the bar to admitted status likewise raised the hire rate from 33.3% to 77.8%. Candidate height—the prespecified placebo—had no monotonic effect, while legal reasoning and experience unexpectedly peaked at their middle rather than highest levels.

### Art auction

Mean measured final price was $171.85, mean allocative efficiency was 53.7%, and only 40.7% of cells allocated the artwork to a bidder tied for the highest valuation. Increasing the minimum bid increment from $10 to $50 raised mean price from $126.67 to $241.67 while lowering efficiency from 70.4% to 27.8%. Exact public disclosure of valuations lowered price and efficiency relative to private values and produced more no-sale outcomes. The auction is the least reliable study: ten cells ended in no sale, and three cells named winners at prices above those bidders’ assigned valuations, demonstrating failure to obey private-value constraints.

## Comparison with the paper and the all-Qwen run

The paper used more theory-anchored designs and 1,073 full-factorial treatment cells, including an auction structured around private valuations and second-highest-value logic. The earlier all-Qwen replication used three factors per study and 108 cells; 106 measurements were valid and 9 of 12 predicted directions matched Qwen’s own simulated evidence. The GPT-Sol-planned studies introduced five factors per setting, multiple outcomes, an explicit placebo, richer information treatments, and a pairwise-balanced orthogonal design, while retaining the same 108-cell execution scale.

The crossed run separates scientific planning from subject simulation, removing the strongest self-confirmation problem in the all-Qwen design: Qwen did not see GPT Sol’s hypotheses or analysis. It nevertheless does not validate the findings as human behavior, and the auction shows that a stronger planner cannot by itself prevent a smaller subject model from violating game rules. The most informative next extension would repeat each condition several times or cross the same frozen GPT-Sol designs with multiple subject-model families.
