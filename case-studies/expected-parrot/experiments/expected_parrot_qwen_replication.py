"""Compare the baseline with increasingly Expected-Parrot-like Qwen elicitation.

This is an experimental diagnostic, not part of the production CLI. The local
Qwen3 text model cannot ingest the supplied image, so the visible package and
scene are transcribed into the stimulus text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3-8B-4bit"
SEGMENTS = ("adventurous_foodie", "mainstream_drinker", "skeptical_health_conscious")

PERSONAS = [
    ("Maya", 24, "adventurous_foodie", "A 24-year-old food blogger who actively seeks out unusual flavor combinations and limited-edition products. She regularly posts about novelty snacks and drinks on social media and enjoys being the first to try new things."),
    ("Jordan", 29, "adventurous_foodie", "A 29-year-old chef who appreciates creative flavor pairings and experimental cuisine. He is open-minded about savory-sweet crossovers and judges products on execution rather than convention."),
    ("Priya", 22, "adventurous_foodie", "A 22-year-old college student who follows food trends on TikTok and loves trying viral snacks. She treats unusual food products as entertainment and conversation starters."),
    ("Dex", 33, "adventurous_foodie", "A 33-year-old craft beverage enthusiast who collects limited-edition drinks. He appreciates bold marketing and is willing to pay a premium for a unique tasting experience, even if the flavor is polarizing."),
    ("Karen", 45, "mainstream_drinker", "A 45-year-old office manager who drinks Coca-Cola daily. She likes the classic taste and is skeptical of unusual flavors but will occasionally try a new Coke variety if it looks refreshing."),
    ("Marcus", 38, "mainstream_drinker", "A 38-year-old construction worker who grabs a Coke with lunch most days. He has straightforward taste preferences and tends to stick with what he knows."),
    ("Brittany", 27, "mainstream_drinker", "A 27-year-old retail associate who drinks soda regularly. She has tried Cherry Coke and Vanilla Coke but generally prefers familiar sweet flavors over anything too experimental."),
    ("Tom", 52, "mainstream_drinker", "A 52-year-old long-haul truck driver and lifelong Coca-Cola fan. He is loyal to the brand but resistant to novelty flavors, believing the original formula is hard to improve on."),
    ("Elena", 34, "skeptical_health_conscious", "A 34-year-old yoga instructor who rarely drinks soda. She reads ingredient labels carefully and is wary of artificial flavoring. She would need a compelling reason to try any soft drink."),
    ("David", 41, "skeptical_health_conscious", "A 41-year-old software engineer who tracks his macros and limits sugar intake. He views novelty sodas as marketing gimmicks and prioritizes nutrition over novelty."),
    ("Aisha", 28, "skeptical_health_conscious", "A 28-year-old graduate student who avoids processed foods when possible. She is interested in food science but skeptical of savory-flavored sodas from a health and palatability standpoint."),
    ("Greg", 44, "skeptical_health_conscious", "A 44-year-old parent of two who has cut back on soda for health reasons. He occasionally buys novelty items for his kids but is critical of products he sees as nutritionally empty gimmicks."),
]

CONCEPT = (
    "Coca-Cola is releasing a limited-edition flavor: Coca-Cola Lasagna Flavored. "
    "It is described as 'Spiced & Savory' with 'Ricotta & Bolognese Notes.' "
    "The tagline reads: 'Original Taste? No, It's Lasagna!'"
)
IMAGE_DESCRIPTION = (
    "A photorealistic advertisement shows a hand holding a cold red Coca-Cola can in front of a baked lasagna. "
    "The can retains prominent classic Coca-Cola script and says 'SPICED & SAVORY,' 'LIMITED EDITION,' "
    "'LASAGNA FLAVORED,' 'ORIGINAL TASTE? NO, IT'S LASAGNA!,' and 'SODA WITH RICOTTA & BOLOGNESE NOTES.' "
    "A large layered-lasagna illustration appears on the lower half of the can."
)

RATING_QUESTIONS = {
    "overall_appeal": "Overall, how appealing is this product concept to you? (1 = not at all appealing, 5 = extremely appealing)",
    "concept_clarity": "How clear is this concept and what the product would taste like? (1 = not at all clear, 5 = extremely clear)",
    "uniqueness": "How unique or different is this concept from other beverages currently available? (1 = not at all unique, 5 = extremely unique)",
    "purchase_intent": "How likely would you be to purchase this product if it were available? (1 = definitely would not purchase, 5 = definitely would purchase)",
}
TEXT_QUESTIONS = {
    "most_appealing": "What is the most appealing aspect of this concept?",
    "primary_concern": "What is your primary concern or hesitation about this concept?",
    "suggested_improvement": "What one change would most improve this concept or make you more likely to try it?",
}
ALL_QUESTIONS = {**RATING_QUESTIONS, **TEXT_QUESTIONS}


def stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256((str(base) + "|" + "|".join(parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def chat_prompt(tokenizer, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def role_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def complete(model, tokenizer, prompt: str, seed: int, max_tokens: int) -> str:
    mx.random.seed(seed)
    return generate(
        model,
        tokenizer,
        chat_prompt(tokenizer, prompt),
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.7),
        verbose=False,
    ).strip()


def complete_role(model, tokenizer, system_prompt: str, user_prompt: str, seed: int, max_tokens: int) -> str:
    mx.random.seed(seed)
    return generate(
        model,
        tokenizer,
        role_chat_prompt(tokenizer, system_prompt, user_prompt),
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.7),
        verbose=False,
    ).strip()


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def extract_rating(text: str) -> int:
    matches = [int(x) for x in re.findall(r"(?<!\d)([1-5])(?!\d)", text)]
    if not matches:
        raise ValueError(f"No 1-5 rating in output: {text[:200]!r}")
    return matches[-1]


def joint_condition(model, tokenizer, seed: int) -> list[dict]:
    """Swap in the public rich personas while retaining full shared context."""
    rows = []
    question_block = "\n".join(f"- {key}: {text}" for key, text in ALL_QUESTIONS.items())
    for name, age, segment, persona in PERSONAS:
        prompt = f"""You are answering questions as if you were a human. Do not break character.

Your identity: {persona}

Product concept:
{CONCEPT}

Visual stimulus:
{IMAGE_DESCRIPTION}

Answer every survey item below from this person's point of view:
{question_block}

Return one JSON object with exactly these keys: {', '.join(ALL_QUESTIONS)}.
Rating values must be integers from 1 to 5. Text values must be concise strings. Do not default to midpoint ratings."""
        raw = complete(model, tokenizer, prompt, stable_seed(seed, "joint", name), 600)
        parsed = extract_json(raw)
        answers: dict[str, int | str] = {key: int(parsed[key]) for key in RATING_QUESTIONS}
        answers.update({key: str(parsed[key]).strip() for key in TEXT_QUESTIONS})
        rows.append({"name": name, "age": age, "segment": segment, "persona": persona, "answers": answers, "raw_outputs": {"joint": raw}})
        print(f"joint {name}: " + ", ".join(f"{k}={answers[k]}" for k in RATING_QUESTIONS), flush=True)
    return rows


def separate_condition(model, tokenizer, seed: int) -> list[dict]:
    """Mimic the public run: seven independent calls; stimulus only in Q1."""
    rows = []
    for name, age, segment, persona in PERSONAS:
        answers: dict[str, int | str] = {}
        raw_outputs: dict[str, str] = {}
        for index, (key, question) in enumerate(ALL_QUESTIONS.items()):
            stimulus = ""
            if index == 0:
                stimulus = f"\n\nProduct concept:\n{CONCEPT}\n\nVisual stimulus:\n{IMAGE_DESCRIPTION}"
            output_rule = "Respond with one integer from 1 to 5 and nothing else." if key in RATING_QUESTIONS else "Respond in character with a concise answer."
            prompt = f"""You are answering questions as if you were a human. Do not break character.

Your identity: {persona}{stimulus}

Survey question: {question}
{output_rule}"""
            raw = complete(model, tokenizer, prompt, stable_seed(seed, "separate", name, key), 20 if key in RATING_QUESTIONS else 220)
            raw_outputs[key] = raw
            answers[key] = extract_rating(raw) if key in RATING_QUESTIONS else raw
        rows.append({"name": name, "age": age, "segment": segment, "persona": persona, "answers": answers, "raw_outputs": raw_outputs})
        print(f"separate {name}: " + ", ".join(f"{k}={answers[k]}" for k in RATING_QUESTIONS), flush=True)
    return rows


def separate_system_role_condition(model, tokenizer, seed: int) -> list[dict]:
    """Closer EDSL analogue: identity is a system-level agent instruction."""
    rows = []
    for name, age, segment, persona in PERSONAS:
        answers: dict[str, int | str] = {}
        raw_outputs: dict[str, str] = {}
        system_prompt = (
            f"You are {name}. {persona} You are answering questions as if you were a human. "
            "Do not break character. Give your own decisive answer rather than a neutral market-analysis answer."
        )
        for index, (key, question) in enumerate(ALL_QUESTIONS.items()):
            stimulus = ""
            if index == 0:
                stimulus = f"Product concept:\n{CONCEPT}\n\nVisual stimulus:\n{IMAGE_DESCRIPTION}\n\n"
            output_rule = "Respond with one integer from 1 to 5 and nothing else." if key in RATING_QUESTIONS else "Respond in character with a concise answer."
            user_prompt = f"{stimulus}Survey question: {question}\n{output_rule}"
            raw = complete_role(
                model, tokenizer, system_prompt, user_prompt,
                stable_seed(seed, "separate-system", name, key),
                20 if key in RATING_QUESTIONS else 220,
            )
            raw_outputs[key] = raw
            answers[key] = extract_rating(raw) if key in RATING_QUESTIONS else raw
        rows.append({"name": name, "age": age, "segment": segment, "persona": persona, "answers": answers, "raw_outputs": raw_outputs})
        print(f"separate-system {name}: " + ", ".join(f"{k}={answers[k]}" for k in RATING_QUESTIONS), flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    by_segment = defaultdict(list)
    for row in rows:
        by_segment[row["segment"]].append(row)
    segment_means = {
        segment: {
            key: round(mean(row["answers"][key] for row in group), 2)
            for key in RATING_QUESTIONS
        }
        for segment, group in by_segment.items()
    }
    overall = {
        key: round(mean(row["answers"][key] for row in rows), 2)
        for key in RATING_QUESTIONS
    }
    appeal_values = [row["answers"]["overall_appeal"] for row in rows]
    intent_values = [row["answers"]["purchase_intent"] for row in rows]
    return {
        "overall": overall,
        "by_segment": segment_means,
        "polarization": {
            "appeal_segment_range": round(max(v["overall_appeal"] for v in segment_means.values()) - min(v["overall_appeal"] for v in segment_means.values()), 2),
            "purchase_segment_range": round(max(v["purchase_intent"] for v in segment_means.values()) - min(v["purchase_intent"] for v in segment_means.values()), 2),
            "appeal_endpoint_share": round(sum(v in (1, 5) for v in appeal_values) / len(appeal_values), 3),
            "purchase_endpoint_share": round(sum(v in (1, 5) for v in intent_values) / len(intent_values), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".artifacts/expected-parrot-qwen-replication")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    image = Path(args.image)
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    print(f"Loading {MODEL} once...", flush=True)
    model, tokenizer = load(MODEL)

    joint = joint_condition(model, tokenizer, args.seed)
    separate = separate_condition(model, tokenizer, args.seed)
    separate_system = separate_system_role_condition(model, tokenizer, args.seed)
    results = {
        "model": MODEL,
        "seed": args.seed,
        "image": {"path": str(image), "sha256": image_sha256, "handling": "text transcription; model is text-only", "description": IMAGE_DESCRIPTION},
        "public_expected_parrot_target": {
            "by_segment": {
                "adventurous_foodie": {"overall_appeal": 4.75, "purchase_intent": 4.5},
                "mainstream_drinker": {"overall_appeal": 1.0, "purchase_intent": 3.0},
                "skeptical_health_conscious": {"overall_appeal": 1.0, "purchase_intent": 1.5},
            },
            "polarization": {"appeal_segment_range": 3.75, "purchase_segment_range": 3.0},
        },
        "rich_personas_joint_full_context": {"summary": summarize(joint), "responses": joint},
        "expected_parrot_like_separate_context_only_q1": {"summary": summarize(separate), "responses": separate},
        "expected_parrot_like_system_personas_separate_context_only_q1": {"summary": summarize(separate_system), "responses": separate_system},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison = {
        "public_expected_parrot": results["public_expected_parrot_target"],
        "rich_personas_joint_full_context": results["rich_personas_joint_full_context"]["summary"],
        "expected_parrot_like_separate_context_only_q1": results["expected_parrot_like_separate_context_only_q1"]["summary"],
        "expected_parrot_like_system_personas_separate_context_only_q1": results["expected_parrot_like_system_personas_separate_context_only_q1"]["summary"],
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
