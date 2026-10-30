from factoriax.marl.macros import Macro
from factoriax.marl.text import parse_plan, render_prompt


def test_prompt_mentions_both_patches_and_distances():
    p = render_prompt(0, 0)
    assert "iron" in p and "copper" in p and "agent 0 of 2" in p
    assert "BUILD" in p


def test_prompt_differs_by_agent():
    assert render_prompt(0, 0) != render_prompt(1, 0)


def test_prompt_hides_variant():
    assert "variant" not in render_prompt(0, 3).lower()


def test_parse_basic():
    assert parse_plan("BUILD iron 3\nBUILD copper 1") == (
        [Macro("iron", 3), Macro("copper", 1)],
        True,
    )


def test_parse_ignores_surrounding_text():
    text = "Sure! Here's my plan:\n```\nbuild Iron 2\n```\nGood luck."
    assert parse_plan(text) == ([Macro("iron", 2)], True)


def test_parse_clamps_and_drops():
    macros, ok = parse_plan(
        "BUILD iron 99\nBUILD gold 2\nBUILD copper 0\nBUILD copper -1"
    )
    assert macros == [Macro("iron", 4)] and ok


def test_parse_garbage():
    assert parse_plan("I will help my teammate.") == ([], False)
