import copy

import pytest

from assr_replication.scientist_design import (
    DesignValidationError,
    design_hash,
    validate_design,
)


def valid_design() -> dict[str, object]:
    return {
        "schema_version": "scientist-design-v1",
        "scenario": "two people bargaining over a mug",
        "agents": [
            {
                "name": "buyer",
                "role": "potential buyer",
                "goal": "buy the mug at an acceptable price",
                "constraints": ["cannot pay more than their available budget"],
            },
            {
                "name": "seller",
                "role": "mug owner",
                "goal": "sell the mug for an acceptable price",
                "constraints": ["may keep the mug instead of selling"],
            },
        ],
        "outcome": {
            "name": "agreement price",
            "operationalization": "the final dollar price if an agreement occurs, otherwise null",
            "measurement_question": "What final price did the parties agree to, if any?",
            "respondent": "neutral recorder",
        },
        "causes": [
            {
                "name": "buyer budget",
                "hypothesis": "A higher buyer budget raises the final agreement price.",
                "predicted_direction": "increase",
                "treatment_values": [40, 70, 100],
            }
        ],
        "information_scope": {
            "public": ["the mug is available for sale"],
            "private": [
                {"agent": "buyer", "information": ["buyer budget"]},
                {"agent": "seller", "information": ["seller reservation price"]},
            ],
        },
        "interaction": {
            "order": ["buyer", "seller"],
            "max_turns": 6,
            "stopping_rule": "stop on agreement, rejection, or the turn cap",
        },
    }


def test_valid_design_is_accepted_and_hash_is_stable() -> None:
    design = valid_design()
    reordered = dict(reversed(list(design.items())))

    assert validate_design(design) == design
    assert design_hash(design) == design_hash(reordered)
    assert len(design_hash(design)) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["causes"][0].update(treatment_values=[1]), "2 to 3"),
        (lambda d: d["causes"][0].update(treatment_values=[1, 2, 3, 4]), "2 to 3"),
        (lambda d: d["interaction"].update(max_turns=9), "max_turns"),
        (
            lambda d: d.update(
                agents=d["agents"]
                + [
                    {"name": f"extra-{i}", "role": "observer", "goal": "observe", "constraints": ["none"]}
                    for i in range(3)
                ]
            ),
            "agents",
        ),
        (
            lambda d: d.update(
                causes=d["causes"]
                + [
                    {
                        "name": f"cause-{i}",
                        "hypothesis": "changes the outcome",
                        "predicted_direction": "uncertain",
                        "treatment_values": ["low", "high"],
                    }
                    for i in range(3)
                ]
            ),
            "causes",
        ),
        (lambda d: d["outcome"].pop("measurement_question"), "measurement_question"),
    ],
)
def test_invalid_design_constraints_are_rejected(mutation: object, message: str) -> None:
    design = copy.deepcopy(valid_design())
    mutation(design)

    with pytest.raises(DesignValidationError, match=message):
        validate_design(design)
