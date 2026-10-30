import json

import pytest
from reward import make_reward_func


def test_reward_and_log(tmp_path):
    f = make_reward_func(tmp_path / "ep.jsonl", step_fn=lambda: 7)
    r = f(["BUILD iron 4"], ["BUILD copper 4"], batch_items=[{"variant": 0}])
    assert r == [pytest.approx(1.0)]
    row = json.loads((tmp_path / "ep.jsonl").read_text().splitlines()[0])
    assert row["global_step"] == 7 and row["variant"] == 0


def test_format_penalty(tmp_path):
    f = make_reward_func(tmp_path / "ep.jsonl", step_fn=lambda: 0)
    assert f(["??"], ["??"], batch_items=[{"variant": 0}]) == [pytest.approx(-0.2)]


def test_one_reward_per_aligned_pair(tmp_path):
    f = make_reward_func(tmp_path / "ep.jsonl", step_fn=lambda: 0)
    r = f(
        ["BUILD copper 4", "??"],
        ["BUILD iron 4", "BUILD iron 4"],
        batch_items=[{"variant": 1}],
    )
    assert r == [pytest.approx(1.0), pytest.approx(-0.1)]
    assert len((tmp_path / "ep.jsonl").read_text().splitlines()) == 2


def test_comlrl_dispatch_passes_batch_items(tmp_path):
    reward_utils = pytest.importorskip("comlrl.utils.reward_utils")
    f = make_reward_func(tmp_path / "ep.jsonl", step_fn=lambda: 0)
    r = reward_utils.call_reward_function(
        f,
        ["prompt"],
        [["BUILD iron 4"], ["BUILD copper 4"]],
        num_agents=2,
        batch_items=[{"variant": 0}],
    )
    assert r == [pytest.approx(1.0)]
