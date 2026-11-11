"""Neutral prompt for independent scientist-design generation."""

from __future__ import annotations

from .models import ChatMessage


def render_scientist_messages(scenario: str) -> tuple[ChatMessage, ...]:
    """Render the frozen prompt using only a short scenario and neutral constraints."""
    system = (
        "You are designing a small causal social simulation. Return exactly one JSON object "
        "and no markdown, commentary, or code fences. Independently choose the agents, outcome, "
        "causes, treatments, information boundaries, goals, and interaction protocol."
    )
    user = f"""Scenario: {scenario}

Use schema_version \"scientist-design-v1\" and exactly these fields:
{{
  \"schema_version\": \"scientist-design-v1\",
  \"scenario\": \"copy the scenario exactly\",
  \"agents\": [{{\"name\": \"...\", \"role\": \"...\", \"goal\": \"...\", \"constraints\": [\"...\"]}}],
  \"outcome\": {{\"name\": \"...\", \"operationalization\": \"...\", \"measurement_question\": \"...\", \"respondent\": \"...\"}},
  \"causes\": [{{\"name\": \"...\", \"hypothesis\": \"...\", \"predicted_direction\": \"increase|decrease|non_monotonic|uncertain\", \"treatment_values\": [\"...\", \"...\"]}}],
  \"information_scope\": {{\"public\": [\"...\"], \"private\": [{{\"agent\": \"agent name\", \"information\": [\"...\"]}}]}},
  \"interaction\": {{\"order\": [\"agent name\"], \"max_turns\": 1, \"stopping_rule\": \"...\"}}
}}

Computational constraints:
- Include 1 to a maximum 4 agents. Give every agent a role, goal, and at least one constraint.
- Select exactly one outcome and state its operationalization, measurement question, and respondent.
- Include 1 to 3 exogenous causes. For each, state a causal hypothesis, predicted direction, and 2 to 3 distinct concrete treatment values.
- Separate public information from agent-specific private information.
- Name agents in interaction order and use a maximum 8 turns.
- Make the design executable as written without adding fields.
- Output valid JSON only."""
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    )
