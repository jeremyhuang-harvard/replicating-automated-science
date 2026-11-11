from assr_replication.hashing import stable_hash


def test_stable_hash_is_canonical_and_sensitive() -> None:
    left = {"messages": [{"role": "user", "content": "Offer $45"}], "seed": 7}
    reordered = {"seed": 7, "messages": [{"content": "Offer $45", "role": "user"}]}
    changed = {"messages": [{"role": "user", "content": "Offer $46"}], "seed": 7}

    assert stable_hash(left) == stable_hash(reordered)
    assert stable_hash(left) != stable_hash(changed)
    assert len(stable_hash(left)) == 64
