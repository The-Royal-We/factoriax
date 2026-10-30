"""Specialisation metrics over (agent, choice) samples."""

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np

from factoriax.marl.macros import Macro


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
