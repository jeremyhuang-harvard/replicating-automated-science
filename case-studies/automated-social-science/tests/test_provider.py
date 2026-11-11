from assr_replication.models import ChatMessage, ChatRequest
from assr_replication.providers import DeterministicMockProvider


def test_mock_provider_is_deterministic_for_the_same_request() -> None:
    provider = DeterministicMockProvider()
    request = ChatRequest(
        model="mock-bargainer-v1",
        messages=(
            ChatMessage(role="system", content="You are the buyer."),
            ChatMessage(role="user", content="Make your opening offer."),
        ),
        metadata={"agent": "buyer", "turn": 0},
    )

    first = provider.complete(request)
    second = provider.complete(request)

    assert first.text == second.text == "I can offer $45 for the mug."
    assert first.provider == "deterministic-mock"
    assert first.raw_response == {
        "agent": "buyer",
        "turn": 0,
        "text": "I can offer $45 for the mug.",
    }
