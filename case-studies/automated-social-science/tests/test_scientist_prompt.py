from assr_replication.scientist_prompt import render_scientist_messages


SCENARIO = "A person is interviewing for a job as a lawyer"


def test_rendered_scientist_prompt_contains_only_neutral_constraints_and_scenario() -> None:
    messages = render_scientist_messages(SCENARIO)
    rendered = "\n".join(message.content for message in messages)

    assert SCENARIO in rendered
    assert "scientist-design-v1" in rendered
    assert "maximum 4 agents" in rendered
    assert "1 to 3" in rendered
    assert "2 to 3" in rendered
    assert "maximum 8 turns" in rendered
    assert "JSON" in rendered
    for forbidden in (
        "Manning",
        "Horton",
        "KeHang",
        "Automated Social Science",
        "published",
        "released",
        "GPT-4",
        "beauty",
        "race",
        "salary",
    ):
        assert forbidden.casefold() not in rendered.casefold()
