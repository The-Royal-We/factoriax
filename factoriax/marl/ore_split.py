"""OreSplit: two players, two ore patches, team score is min(iron, copper) in pallets.

Neither player carries enough to cover both patches well, so the best team
score needs each player to take one patch.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from factoriax.engine.constants import Action, BlockType, ItemType, Machine
from factoriax.engine.levels import Level, LevelBuilder, build_state
from factoriax.engine.state import EnvParams, EnvState

HORIZON = 400
NUM_PLAYERS = 2
PATCHES = ("iron", "copper")
START_INVENTORY = [(ItemType.MINER, 4), (ItemType.PALLET, 4)]
MAX_MACHINES = 32

_WIDTH, _HEIGHT = 16, 12
_ORE_ROW = 2
_PATCH_XS = (range(2, 6), range(10, 14))
_SPAWNS = ((3, 9), (12, 9))


class MinerSite(NamedTuple):
    """Where a player stands, and where its miner and pallet go, along ``facing``."""

    stand: tuple[int, int]
    miner: tuple[int, int]
    pallet: tuple[int, int]
    facing: Action


def _patch_sites(xs: range) -> tuple[MinerSite, ...]:
    return tuple(
        MinerSite(
            stand=(x, _ORE_ROW + 1),
            miner=(x, _ORE_ROW),
            pallet=(x, _ORE_ROW - 1),
            facing=Action.UP,
        )
        for x in xs
    )


def _patch_xs(variant: int) -> dict[str, range]:
    a, b = _PATCH_XS
    return {"iron": b, "copper": a} if variant & 1 else {"iron": a, "copper": b}


def sites(variant: int) -> dict[str, tuple[MinerSite, ...]]:
    """Map each resource to its sites; bit 0 of ``variant`` swaps the patches."""
    return {name: _patch_sites(xs) for name, xs in _patch_xs(variant).items()}


def spawns(variant: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return both spawns; bit 1 of ``variant`` swaps them."""
    a, b = _SPAWNS
    return (b, a) if variant & 2 else (a, b)


def level_builder(variant: int) -> LevelBuilder:
    """Start a builder for the variant, so a caller can add machines before build."""
    xs = _patch_xs(variant)
    builder = (
        LevelBuilder(_WIDTH, _HEIGHT)
        .fill_rect(xs["iron"].start, _ORE_ROW, len(xs["iron"]), 1, BlockType.IRON)
        .fill_rect(xs["copper"].start, _ORE_ROW, len(xs["copper"]), 1, BlockType.COPPER)
        .set_player_inventory(START_INVENTORY)
    )
    for x, y in spawns(variant):
        builder.set_player_position(x, y)
    return builder


def make_level(variant: int) -> Level:
    """Build the level, named ``ore_split_<variant>``."""
    return level_builder(variant).build(f"ore_split_{variant}")


def make_state(variant: int) -> tuple[EnvState, EnvParams]:
    """Build a two-player state; every variant has the same array shapes."""
    state = build_state(
        make_level(variant), num_players=NUM_PLAYERS, max_machines=MAX_MACHINES
    )
    return state, EnvParams(max_timesteps=HORIZON)


def _ore_in_pallets(state: EnvState, item: ItemType) -> jax.Array:
    in_pallet = (
        (state.ent_y >= 0)
        & (state.ent_type == Machine.PALLET)
        & (state.ent_buf_type == item)
    )
    return jnp.sum(jnp.where(in_pallet, state.ent_buf_count, 0))


def ore_totals(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return iron ore and copper ore held in active pallets."""
    return _ore_in_pallets(state, ItemType.IRON_ORE), _ore_in_pallets(
        state, ItemType.COPPER_ORE
    )


def team_score(state: EnvState) -> jax.Array:
    """Score the min of iron and copper ore in pallets, as float32."""
    return jnp.minimum(*ore_totals(state)).astype(jnp.float32)
