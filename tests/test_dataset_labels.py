import pytest

from code_mia.data.poisoned_chalice import membership_label


def test_dataset_label_mapping():
    assert membership_label(1) == 1
    assert membership_label(0) == 0
    assert membership_label("1") == 1
    with pytest.raises(ValueError): membership_label(2)

