import pytest

from code_mia.data.poisoned_chalice import membership_label


def test_dataset_label_mapping():
    assert membership_label(1) == 1
    assert membership_label(0) == 0
    assert membership_label("1") == 1
    with pytest.raises(ValueError): membership_label(2)


def test_numeric_membership_labels():
    assert membership_label(1) == 1
    assert membership_label(0) == 0


def test_boolean_membership_labels():
    assert membership_label(True) == 1
    assert membership_label(False) == 0


def test_numeric_string_membership_labels():
    assert membership_label("1") == 1
    assert membership_label("0") == 0


def test_official_member_label():
    assert membership_label("member") == 1


def test_official_non_member_label():
    assert membership_label("non-member") == 0


def test_official_labels_allow_case_and_whitespace():
    assert membership_label(" Member ") == 1
    assert membership_label(" NON-MEMBER ") == 0


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "",
        "unknown",
        "members",
        "nonmember",
        2,
        -1,
    ],
)
def test_invalid_membership_labels_are_rejected(invalid_value):
    with pytest.raises(ValueError):
        membership_label(invalid_value)