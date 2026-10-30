import pytest
from metrics import entropy, nmi


def test_nmi_perfect_split():
    assert nmi([0, 1, 0, 1], ["iron", "copper", "iron", "copper"]) == pytest.approx(1.0)


def test_nmi_no_specialisation():
    assert nmi([0, 0, 1, 1], ["iron", "copper", "iron", "copper"]) == pytest.approx(0.0)


def test_nmi_constant_choice_is_zero():
    assert nmi([0, 1], ["iron", "iron"]) == 0.0


def test_entropy_uniform_two():
    assert entropy(["iron", "copper"]) == pytest.approx(1.0)
