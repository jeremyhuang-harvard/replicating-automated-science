"""Response engines. An engine turns one persona prompt into raw text; parsing/validation is shared."""

from __future__ import annotations

import json
import random
from typing import Protocol

from .schemas import Concept, Persona, Question

DEMO_MODEL = "demo-rules-v1"


class EngineError(RuntimeError):
    """The engine itself failed (missing executable, non-zero exit, timeout) rather than one bad answer."""


class Engine(Protocol):
    name: str
    model: str

    def configuration(self) -> dict[str, object]:
        """Return serializable settings needed to reproduce this engine invocation."""
        ...

    def generate(self, concept: Concept, persona: Persona, questions: list[Question], prompt: str, seed: int) -> str:
        """Return raw model text expected to contain ONE JSON object of answers."""


# --- Demo engine: transparent rules + seeded noise; no model, fully offline and repeatable. ---

_NOVELTY_CUES = ("bold", "savory", "unusual", "limited", "experimental", "twist", "fusion", "unexpected",
                 "exotic", "spicy", "umami", "lasagna", "pizza", "strange", "surprising")
_FAMILIAR_CUES = ("classic", "familiar", "original", "traditional", "trusted", "same great", "as always", "regular")
_HEALTH_CUES = ("zero sugar", "sugar free", "natural", "organic", "protein", "low calorie", "probiotic", "no artificial")


def concept_signals(concept: Concept) -> dict[str, float]:
    """Crude keyword read of the concept: novelty and health-positioning in 0-1, plus word count."""
    text = " ".join((concept.title + " " + concept.description).lower().replace("-", " ").split())
    count = lambda cues: sum(cue in text for cue in cues)
    novelty = min(1.0, max(0.05, 0.3 + 0.18 * count(_NOVELTY_CUES) - 0.15 * count(_FAMILIAR_CUES)))
    health = min(1.0, 0.4 * count(_HEALTH_CUES))
    return {"novelty": novelty, "health": health, "words": len(concept.description.split())}


def _clamp_round(x: float) -> int:
    return max(1, min(5, int(x + 0.5)))


_REACTION = {
    "adventurous_foodie": {
        "low": ["I like weird drinks, but {title} sounds like it misses the mark.", "Not even my adventurous side is sold on {title}."],
        "mid": ["{title} is interesting enough that I'd taste it once.", "I'm curious about {title} but not convinced yet."],
        "high": ["{title} is exactly the kind of experiment I go looking for!", "Love it: {title} sounds like a fun, bold tasting experience."],
    },
    "mainstream_drinker": {
        "low": ["{title} just sounds strange to me; I'd stay with my usual drink.", "That's not what I expect from a soda, so {title} is a pass."],
        "mid": ["{title} is okay, I guess, if it tastes close to a normal drink.", "I could see trying {title} once if a friend had it."],
        "high": ["{title} sounds fun and easy to enjoy.", "I'd happily pick up {title} next to my regular drinks."],
    },
    "skeptical_health_conscious": {
        "low": ["{title} reads like a marketing gimmick to me.", "I don't see why anyone needs {title}."],
        "mid": ["{title} might be fine, but I'd want to see the ingredients first.", "I'm undecided on {title}; the claims need backing up."],
        "high": ["{title} sounds better than I expected, assuming the label holds up.", "If the ingredients are as simple as claimed, {title} is appealing."],
    },
}
_REACTION_TAIL = ["", " The idea stays with me.", " It's different, for better or worse.", " I'd need to taste it."]

_CONCERNS = {
    "adventurous_foodie": ["I worry it's a gimmick that tastes muddled rather than genuinely crafted.", "The risk is poor execution: the flavors need to balance, not just shock.", "Novelty can fade after one can unless the flavor is actually good."],
    "mainstream_drinker": ["It might not taste like a drink I'd actually want, and the price could be too high for something unproven.", "I'm unsure what it tastes like and don't want to waste money on a weird flavor.", "It feels too far from the sodas I know."],
    "skeptical_health_conscious": ["I'd want the full ingredient list, sweeteners, and sodium before believing any of the claims.", "Flavorings and additives worry me, and the marketing tells me little about real benefits.", "I don't trust the positioning without evidence or a clear nutrition panel."],
}
_IMPROVE = {
    "adventurous_foodie": ["A small-batch sample pack so I can taste it before committing.", "Chef-style tasting notes or a collaboration that shows craft.", "A bolder, more distinctive flavor identity."],
    "mainstream_drinker": ["A familiar flavor base and a lower price or a promotion.", "A single-can trial size at the register.", "A clear description that it tastes like a normal soda."],
    "skeptical_health_conscious": ["A short, transparent ingredient list with no artificial additives.", "Third-party nutrition information on the label.", "Smaller portions or lower sodium and sugar."],
}
_HEALTH_NOTE = " The sugar-free claim helps, but I'd still check the label."


class DemoEngine:
    """Deterministic, offline engine. Segment differences come from persona attitudes x concept cues."""

    name = "demo"
    model = DEMO_MODEL

    def configuration(self) -> dict[str, object]:
        return {"version": DEMO_MODEL}

    def generate(self, concept: Concept, persona: Persona, questions: list[Question], prompt: str, seed: int) -> str:
        rng = random.Random(f"{seed}|{persona.persona_id}|{concept.title}|{concept.description}")
        sig, att = concept_signals(concept), persona.attitudes
        w, h = sig["novelty"], sig["health"]
        ns, bl, hc, ps = (att.get(k, 3) for k in ("novelty_seeking", "brand_loyalty", "health_concern", "price_sensitivity"))
        noise = lambda spread: rng.uniform(-spread, spread)

        appeal_core = 3.0 + (ns - 3) * w + (bl - 3) * 0.3 * (1 - w) + (hc - 3) * 0.5 * h - (hc - 3) * 0.5 * w * (1 - h)
        words = sig["words"]
        clarity_core = 2.0 if words < 6 else 3.2 if words < 12 else 3.5 if words > 120 else 4.4
        ratings = {
            "appeal": _clamp_round(appeal_core + noise(0.6)),
            "clarity": _clamp_round(clarity_core + noise(0.6)),
            "uniqueness": _clamp_round(1.5 + 3.2 * w + 0.15 * (ns - 3) + noise(0.6)),
            "purchase_intent": _clamp_round(appeal_core - 0.3 - 0.25 * (ps - 3) + noise(0.7)),
        }
        bucket = "low" if ratings["appeal"] <= 2 else "mid" if ratings["appeal"] == 3 else "high"
        seg, title = persona.segment, concept.title
        reaction = rng.choice(_REACTION[seg][bucket]).format(title=title) + rng.choice(_REACTION_TAIL)
        concern = rng.choice(_CONCERNS[seg]) + (_HEALTH_NOTE if h > 0 and hc >= 4 else "")
        text = {
            "first_reaction": reaction,
            "concerns": f"About {title}: {concern}",
            "improvement": f"For {title}: {rng.choice(_IMPROVE[seg])}",
        }
        answers = {}
        for q in questions:
            answers[q.question_id] = ratings.get(q.question_id, 3) if q.kind == "rating" else text.get(q.question_id, reaction)
        return json.dumps(answers)
