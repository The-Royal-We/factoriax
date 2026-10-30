"""Specialisation metrics over (agent, choice) samples."""

from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence

import numpy as np

from factoriax.marl.episode import oracle_texts
from factoriax.marl.macros import Macro
from factoriax.marl.text import parse_plan


def choice(plan: Sequence[Macro]) -> str:
    """Return the patch with most miners in the plan (first wins ties), or "none"."""
    totals: Counter[str] = Counter()
    for m in plan:
        totals[m.patch] += m.count
    return max(totals, key=totals.__getitem__) if totals else "none"


def entropy(choices: Sequence[Hashable]) -> float:
    """Return the Shannon entropy in bits."""
    p = np.array(list(Counter(choices).values()), dtype=float) / len(choices)
    return float((p * np.log2(1 / p)).sum())


def nmi(agent_ids: Sequence[int], choices: Sequence[str]) -> float:
    """Return I(A;C) / sqrt(H(A) H(C)), and 0 when either entropy is 0."""
    h_a, h_c = entropy(agent_ids), entropy(choices)
    if h_a == 0 or h_c == 0:
        return 0.0
    mutual = h_a + h_c - entropy(list(zip(agent_ids, choices, strict=True)))
    return float(mutual / np.sqrt(h_a * h_c))


def last_per_step(rows: Sequence[dict], per_step: int) -> list[dict]:
    """Keep the last ``per_step`` rows of each step.

    A resumed session replays the steps after its checkpoint, so the log holds
    the abandoned rows first and the kept ones after.
    """
    by_step: defaultdict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_step[r["global_step"]].append(r)
    return [r for step in sorted(by_step) for r in by_step[step][-per_step:]]


def near_patch_rate(rows: Sequence[dict]) -> float:
    """Return how often an agent's choice matches its oracle patch."""
    hits = [
        choice([Macro(*m) for m in plan]) == choice(parse_plan(oracle)[0])
        for r in rows
        for plan, oracle in zip(r["plans"], oracle_texts(r["variant"]), strict=True)
    ]
    return float(np.mean(hits))
