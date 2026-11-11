# Qwen replication of the Expected Parrot Lasagna Coke run

## Question

Can the local Qwen3-8B-4bit baseline produce Expected Parrot's stronger polarization if we replace the minimal personas with the 12 public Expected Parrot personas and imitate its elicitation workflow?

## Stimulus and model

- Model: `mlx-community/Qwen3-8B-4bit`
- Seed: `42`, deterministically varied by persona and question
- Supplied image SHA-256: `bfd8e43760c1e8eb646579f92eaf7779679368980556046b5b70bbfa333ac72f`
- Image handling: the local Qwen model is text-only, so the visible can, package copy, Coca-Cola branding, lasagna illustration, and background lasagna were transcribed into the prompt. This is not equivalent to Expected Parrot's vision-model input.

## Conditions

1. **Original baseline:** minimal attitude personas; all seven questions answered together with full concept context.
2. **Rich personas, joint/full context:** the 12 named public Expected Parrot personas; all seven questions answered together with full concept and image-description context.
3. **Expected-Parrot-like separate calls:** the same public personas; seven independent calls per persona; product context supplied only to the first question, matching the public run's context-loss behavior.
4. **System-persona separate calls:** condition 3, but persona identity is placed in a system-level role prompt to approximate EDSL agent conditioning more closely.

All non-baseline experimental conditions use the public personas Maya, Jordan, Priya, Dex, Karen, Marcus, Brittany, Tom, Elena, David, Aisha, and Greg.

## Segment means

### Appeal

| Condition | Adventurous | Mainstream | Health-conscious | Segment range |
|---|---:|---:|---:|---:|
| Original local baseline | 4.25 | 2.50 | 2.00 | 2.25 |
| Rich personas, joint/full context | 4.00 | 1.75 | 1.75 | 2.25 |
| Separate calls, context only Q1 | 3.25 | 1.50 | 2.00 | 1.75 |
| System personas + separate calls | 3.50 | 1.75 | 1.25 | 2.25 |
| Public Expected Parrot | 4.75 | 1.00 | 1.00 | 3.75 |

### Purchase intent

| Condition | Adventurous | Mainstream | Health-conscious | Segment range |
|---|---:|---:|---:|---:|
| Original local baseline | 3.25 | 2.00 | 1.75 | 1.50 |
| Rich personas, joint/full context | 4.00 | 1.25 | 1.00 | 3.00 |
| Separate calls, context only Q1 | 3.75 | 3.75 | 2.00 | 1.75 |
| System personas + separate calls | 3.75 | 3.25 | 2.00 | 1.75 |
| Public Expected Parrot | 4.50 | 3.00 | 1.50 | 3.00 |

## Main result

The rich public personas materially changed Qwen's behavior. Under full shared context, mainstream and health-conscious purchase intent fell from 2.00/1.75 to 1.25/1.00 while adventurous purchase intent rose from 3.25 to 4.00. The resulting purchase-intent range exactly matched Expected Parrot's 3.0-point range.

The appeal result did not fully reproduce. Rich personas pushed the negative segments downward, but Qwen held the adventurous segment at 4.0 rather than Expected Parrot's 4.75. Making personas system-level increased endpoint use and moved health-conscious appeal to 1.25, but adventurous appeal remained only 3.5.

## Important aggregate coincidence

The plain separate-call condition almost exactly reproduced Expected Parrot's **overall** means:

| Measure | Qwen separate-call | Expected Parrot |
|---|---:|---:|
| Appeal | 2.25 | 2.25 |
| Clarity | 3.83 | 3.83 |
| Uniqueness | 3.08 | 3.00 |
| Purchase intent | 3.17 | 3.00 |

However, it did not reproduce the subgroup pattern: Qwen gave adventurous appeal 3.25 rather than 4.75 and mainstream purchase intent 3.75 rather than 3.00. Matching aggregate means therefore concealed meaningful respondent-level and segment-level disagreement.

## Interpretation

1. **Persona selection matters.** The Expected Parrot biographies are more identity-saturated than the baseline's short attitude profiles. They strongly lowered mainstream and health-conscious scores and increased purchase-intent separation.
2. **Qwen remains less extreme on adventurous appeal.** Even with the same personas and stronger role framing, the local 8B model did not consistently convert novelty-seeking identities into 5/5 appeal.
3. **Separate calls create the Expected Parrot appeal–intent inconsistency.** When product context was omitted after Q1, mainstream purchase intent rose to 3.75 even though appeal stayed at 1.5. This resembles Expected Parrot's 1.0 appeal/3.0 intent gap, but it is partly an elicitation artifact: respondents are rating an unspecified “concept” based on identity rather than the Lasagna Coke stimulus.
4. **The image remains an unresolved model difference.** Expected Parrot used a vision-capable Claude model. The local Qwen run received an accurate text transcription, not image pixels.
5. **Model capability is a remaining explanation.** The public Claude run produced stronger identity enactment and endpoint use, especially among adventurous personas.

## Bottom line

We can reproduce a substantial portion of the polarization with the public personas alone—especially for purchase intent—and we can nearly reproduce Expected Parrot's aggregate means by also copying its separate-call/context-loss workflow. We cannot reproduce the full 4.75-versus-1.0 appeal split with Qwen3-8B-4bit without adding stronger, outcome-leading instructions. Doing that would tune toward the published answer rather than faithfully reproduce the method.

Raw outputs and machine-readable comparisons are generated under `.artifacts/expected-parrot-qwen-replication-v2/` by `experiments/expected_parrot_qwen_replication.py`.
