import json

from code_mia.data.manifest import build_manifest, deterministic_stratified_indices


def test_deterministic_stratified_sampling():
    rows = [{"sample_id": f"id-{i:02}", "membership": i % 2, "eligibility_status": "eligible"} for i in range(20)]
    first = deterministic_stratified_indices(rows, 3, 4, 42)
    second = deterministic_stratified_indices(rows, 3, 4, 42)
    assert first == second
    assert sum(rows[i]["membership"] == 1 for i in first) == 3
    assert sum(rows[i]["membership"] == 0 for i in first) == 4


def test_frozen_manifest_is_reused(monkeypatch, tmp_path):
    import code_mia.data.manifest as module
    fake_data = []
    for i in range(10):
        fake_data.append({"content": " ".join(f"w{j}" for j in range(40)) + f" row{i}", "membership": i % 2})
    class FakeDataset(list): column_names = ["content", "membership"]
    monkeypatch.setattr(module, "load_public_test", lambda _: FakeDataset(fake_data))
    tokenizer = lambda text, **kwargs: {"input_ids": list(range(len(text.split())))}
    config = {
        "experiment": {"seed": 42},
        "dataset": {"id": "d", "revision": "r", "config": "Python", "split": "test", "members": 2,
                    "non_members": 2, "text_column": "content", "label_column": "membership"},
        "canonicalization": {"word_units": 32},
        "model": {"tokenizer_id": "t", "tokenizer_revision": "tr"},
    }
    first, meta1 = build_manifest(config, tmp_path, tokenizer=tokenizer)
    monkeypatch.setattr(module, "load_public_test", lambda _: (_ for _ in ()).throw(AssertionError("must not reload")))
    second, meta2 = build_manifest(config, tmp_path, tokenizer=tokenizer)
    assert first == second
    assert meta1["protocol_fingerprint"] == meta2["protocol_fingerprint"]
    assert len({row["sample_id"] for row in first}) == 4

