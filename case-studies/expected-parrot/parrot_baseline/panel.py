"""Default synthetic panel: three segments defined by explicit product attitudes (1-5), never demographics."""

from __future__ import annotations

from .schemas import Persona, SchemaError

MAX_PANEL_SIZE = 1000

# attitude keys: novelty_seeking, brand_loyalty, health_concern, price_sensitivity
SEGMENTS: dict[str, list[tuple[str, dict[str, int]]]] = {
    "adventurous_foodie": [
        ("Seeks out unusual flavor pairings and enjoys describing them to others.",
         {"novelty_seeking": 5, "brand_loyalty": 1, "health_concern": 2, "price_sensitivity": 2}),
        ("Buys limited-edition and international products mainly to see what they taste like.",
         {"novelty_seeking": 5, "brand_loyalty": 2, "health_concern": 2, "price_sensitivity": 3}),
        ("Treats drinks as a tasting experience; bored by anything too familiar.",
         {"novelty_seeking": 4, "brand_loyalty": 1, "health_concern": 3, "price_sensitivity": 2}),
        ("Likes savory-meets-sweet experiments but wants them to be well executed.",
         {"novelty_seeking": 4, "brand_loyalty": 2, "health_concern": 2, "price_sensitivity": 3}),
    ],
    "mainstream_drinker": [
        ("Sticks with a few trusted soft drinks and rarely switches brands.",
         {"novelty_seeking": 2, "brand_loyalty": 5, "health_concern": 3, "price_sensitivity": 3}),
        ("Chooses whatever is familiar, cold, and on promotion at the store.",
         {"novelty_seeking": 2, "brand_loyalty": 3, "health_concern": 2, "price_sensitivity": 5}),
        ("Will try a new flavor occasionally if it sounds like a normal drink.",
         {"novelty_seeking": 3, "brand_loyalty": 4, "health_concern": 3, "price_sensitivity": 4}),
        ("Drinks soda socially and does not think much about ingredients.",
         {"novelty_seeking": 2, "brand_loyalty": 4, "health_concern": 2, "price_sensitivity": 3}),
    ],
    "skeptical_health_conscious": [
        ("Reads labels closely and distrusts marketing claims on new beverages.",
         {"novelty_seeking": 2, "brand_loyalty": 3, "health_concern": 5, "price_sensitivity": 3}),
        ("Avoids artificial additives and wants clear evidence a product is worth it.",
         {"novelty_seeking": 3, "brand_loyalty": 2, "health_concern": 5, "price_sensitivity": 3}),
        ("Open to novelty only when ingredients are simple and the benefit is explained.",
         {"novelty_seeking": 3, "brand_loyalty": 2, "health_concern": 4, "price_sensitivity": 4}),
        ("Sees novelty drinks as gimmicks unless they replace something less healthy.",
         {"novelty_seeking": 2, "brand_loyalty": 3, "health_concern": 4, "price_sensitivity": 4}),
    ],
}

_PREFIX = {
    "adventurous_foodie": ("af", "Adventurous Foodie"),
    "mainstream_drinker": ("md", "Mainstream Drinker"),
    "skeptical_health_conscious": ("sh", "Skeptical Health-Conscious"),
}


def default_panel(size: int = 12) -> list[Persona]:
    """Round-robin across segments so any panel size stays as balanced as possible.

    Templates repeat once a segment runs out, so panels larger than 12 are deterministic copies
    of the base personas; the demo engine's seeded noise differentiates them.
    """
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_PANEL_SIZE:
        raise SchemaError(f"panel size must be an integer 1-{MAX_PANEL_SIZE}, got {size!r}")
    segments = list(SEGMENTS)
    panel = []
    for i in range(size):
        segment = segments[i % len(segments)]
        n = i // len(segments)  # nth persona within the segment
        template = SEGMENTS[segment][n % len(SEGMENTS[segment])]
        prefix, label = _PREFIX[segment]
        panel.append(Persona(
            persona_id=f"{prefix}-{n + 1:02d}",
            name=f"{label} {n + 1}",
            segment=segment,
            description=template[0],
            attitudes=dict(template[1]),
        ))
    return panel
