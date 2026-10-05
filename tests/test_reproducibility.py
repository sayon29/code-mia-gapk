from code_mia.config import config_fingerprint
from code_mia.runner import _compatible


def test_config_fingerprint_stable_and_ignores_output_path():
    a = {"experiment": {"output_dir": "a", "seed": 42}, "x": {"b": 2, "a": 1}}
    b = {"x": {"a": 1, "b": 2}, "experiment": {"seed": 42, "output_dir": "b"}}
    assert config_fingerprint(a) == config_fingerprint(b)
    b["experiment"]["seed"] = 43
    assert config_fingerprint(a) != config_fingerprint(b)


def test_checkpoint_compatibility_checks_every_field():
    expected = {"sample_id": "x", "canonical_text_sha256": "h", "model_revision": "r", "gap_fraction": 0.2}
    assert _compatible({**expected, "status": "success"}, expected)
    assert not _compatible({**expected, "model_revision": "other", "status": "success"}, expected)
    assert not _compatible({**expected, "status": "failure"}, expected)
