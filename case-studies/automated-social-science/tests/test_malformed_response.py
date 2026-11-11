from assr_replication.models import ChatResponse


def test_malformed_provider_payload_is_preserved_verbatim() -> None:
    malformed = {"choices": [{"message": {"content": 17}}], "request_id": "abc"}

    response = ChatResponse.from_raw(
        provider="future-provider",
        model="future-model",
        raw_response=malformed,
    )

    assert response.text is None
    assert response.error == "response did not contain string field 'text'"
    assert response.raw_response is malformed
    assert response.to_dict()["raw_response"] == malformed
