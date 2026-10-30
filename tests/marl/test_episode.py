import pytest

from factoriax.marl.episode import oracle_score, oracle_texts, rollout, score_plans


def test_split_beats_duplicate():
    split = score_plans(["BUILD iron 4", "BUILD copper 4"], variant=0)
    dup = score_plans(["BUILD iron 4", "BUILD iron 4"], variant=0)
    assert split.team_score > 0 and dup.team_score == 0


def test_same_patch_collision_scores_lower():
    clash = score_plans(
        ["BUILD iron 2\nBUILD copper 2", "BUILD iron 2\nBUILD copper 2"], 0
    )
    split = score_plans(["BUILD iron 4", "BUILD copper 4"], 0)
    assert clash.team_score < split.team_score


def test_oracle_normalises_to_one():
    for v in range(4):
        assert oracle_score(v) > 0
    r = score_plans(oracle_texts(0), 0)
    assert r.normalised == pytest.approx(1.0)


def test_deterministic():
    a = score_plans(["BUILD iron 3", "BUILD copper 4"], 1)
    assert a == score_plans(["BUILD iron 3", "BUILD copper 4"], 1)


def test_rollout_compiles_once():
    score_plans(["BUILD iron 1", ""], 0)
    score_plans(["BUILD copper 4", "BUILD iron 2"], 3)
    assert rollout._cache_size() == 1


def test_unparseable_still_scores():
    r = score_plans(["no idea", "BUILD copper 4"], 0)
    assert r.parsed_ok == [False, True] and r.team_score == 0


def test_swapped_split_scores_below_oracle():
    for v in range(4):
        assert score_plans(oracle_texts(v)[::-1], v).normalised < 0.6
