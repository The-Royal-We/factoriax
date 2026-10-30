"""CoMLRL reward adapter: score each joint completion in Factoriax and log it."""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from factoriax.marl.episode import score_plans
from factoriax.marl.ore_split import NUM_PLAYERS
from factoriax.marl.text import render_prompt

NUM_VARIANTS = 4
FORMAT_PENALTY = 0.1


def make_dataset(n_rows: int) -> Any:
    """Build ``n_rows`` prompt rows, cycling through the variants."""
    from datasets import Dataset

    rows = [
        {
            "variant": v,
            **{f"prompt_{i}": render_prompt(i, v) for i in range(NUM_PLAYERS)},
        }
        for v in (r % NUM_VARIANTS for r in range(n_rows))
    ]
    return Dataset.from_list(rows)


def make_reward_func(
    log_path: Path, step_fn: Callable[[], int]
) -> Callable[..., list[float]]:
    """Return a reward function that scores aligned completion pairs and logs each."""

    def reward_func(
        completions_0: Sequence[str],
        completions_1: Sequence[str],
        *,
        # Keyword-only: CoMLRL counts positional params to decide whether to
        # prepend prompts, so a positional batch_items breaks its dispatch.
        batch_items: Sequence[dict[str, Any]],
    ) -> list[float]:
        variant = int(batch_items[0]["variant"])
        rewards = []
        with open(log_path, "a") as log:
            for texts in zip(completions_0, completions_1, strict=True):
                r = score_plans(texts, variant)
                reward = r.normalised - FORMAT_PENALTY * r.parsed_ok.count(False)
                rewards.append(reward)
                row = {
                    "global_step": step_fn(),
                    "t": time.time(),
                    "variant": variant,
                    "texts": list(texts),
                    "plans": [[list(m) for m in p] for p in r.plans],
                    "parsed_ok": r.parsed_ok,
                    "iron": r.iron,
                    "copper": r.copper,
                    "team_score": r.team_score,
                    "normalised": r.normalised,
                    "reward": reward,
                }
                log.write(json.dumps(row) + "\n")
        return rewards

    return reward_func
