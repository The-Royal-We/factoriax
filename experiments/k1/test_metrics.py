import pytest
from metrics import entropy, last_per_step, near_patch_rate, nmi


def test_nmi_perfect_split():
    assert nmi([0, 1, 0, 1], ["iron", "copper", "iron", "copper"]) == pytest.approx(1.0)


def test_nmi_no_specialisation():
    assert nmi([0, 0, 1, 1], ["iron", "copper", "iron", "copper"]) == pytest.approx(0.0)


def test_nmi_constant_choice_is_zero():
    assert nmi([0, 1], ["iron", "iron"]) == 0.0


def test_entropy_uniform_two():
    assert entropy(["iron", "copper"]) == pytest.approx(1.0)


def test_last_per_step_drops_replayed_rows():
    rows = [{"global_step": s, "id": i} for i, s in enumerate([0, 0, 1, 1, 1, 1])]
    assert [r["id"] for r in last_per_step(rows, per_step=2)] == [0, 1, 4, 5]


def test_near_patch_rate_tells_oracle_from_fixed_rule():
    fixed = [["iron", 4]], [["copper", 4]]
    rows = [{"variant": v, "plans": fixed} for v in range(4)]
    assert near_patch_rate(rows) == pytest.approx(0.5)
    swapped = [{"variant": 1, "plans": [[["copper", 4]], [["iron", 4]]]}]
    assert near_patch_rate(swapped) == pytest.approx(1.0)
