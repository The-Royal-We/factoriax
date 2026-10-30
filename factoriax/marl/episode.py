"""Score two agents' text plans with one jitted Factoriax rollout."""

from collections.abc import Sequence
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp

from factoriax.engine.state import EnvParams, EnvState
from factoriax.marl.macros import MAX_SITES, Macro, compile_joint
from factoriax.marl.ore_split import make_state, ore_totals, sites, spawns
from factoriax.marl.step import multi_step
from factoriax.marl.text import parse_plan


class EpisodeResult(NamedTuple):
    """Outcome of one episode; ``normalised`` is ``team_score`` over the oracle."""

    team_score: float
    iron: int
    copper: int
    normalised: float
    plans: list[list[Macro]]
    parsed_ok: list[bool]


@jax.jit
def rollout(
    key: jax.Array, state: EnvState, actions: jax.Array, params: EnvParams
) -> tuple[jax.Array, jax.Array]:
    """Run the whole horizon and return ``(iron, copper)`` in pallets."""

    def tick(s: EnvState, a: jax.Array) -> tuple[EnvState, None]:
        return multi_step(jax.random.fold_in(key, s.timestep), s, a, params), None

    final, _ = jax.lax.scan(tick, state, actions)
    return ore_totals(final)


def _simulate(plans: Sequence[Sequence[Macro]], variant: int) -> tuple[int, int]:
    state, params = make_state(variant)
    actions = jnp.asarray(compile_joint(plans, variant))
    iron, copper = rollout(jax.random.PRNGKey(variant), state, actions, params)
    return int(iron), int(copper)


def oracle_texts(variant: int) -> list[str]:
    """Return the split plan: the agent nearer iron takes iron, the other copper."""
    iron = sites(variant)["iron"][0].miner
    dists = [abs(x - iron[0]) + abs(y - iron[1]) for x, y in spawns(variant)]
    iron_agent = dists.index(min(dists))
    return [
        f"BUILD {'iron' if i == iron_agent else 'copper'} {MAX_SITES}"
        for i in range(len(dists))
    ]


@cache
def oracle_score(variant: int) -> float:
    """Score the oracle split, the normaliser for ``score_plans``."""
    return float(
        min(_simulate([parse_plan(t)[0] for t in oracle_texts(variant)], variant))
    )


def score_plans(texts: Sequence[str], variant: int) -> EpisodeResult:
    """Parse, compile, and run both agents' texts; unparseable text builds nothing."""
    parsed = [parse_plan(t) for t in texts]
    plans = [macros for macros, _ in parsed]
    iron, copper = _simulate(plans, variant)
    score = float(min(iron, copper))
    return EpisodeResult(
        team_score=score,
        iron=iron,
        copper=copper,
        normalised=score / oracle_score(variant),
        plans=plans,
        parsed_ok=[ok for _, ok in parsed],
    )
