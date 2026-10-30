"""Compile ``BUILD <patch> <count>`` macros into fixed-length action arrays."""

from collections import deque
from collections.abc import Sequence
from functools import cache
from typing import NamedTuple

import numpy as np

from factoriax.engine.constants import Action, BlockType
from factoriax.marl.ore_split import HORIZON, START_INVENTORY, make_level, sites, spawns

Tile = tuple[int, int]

_OFFSETS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}
_FACE = {
    Action.UP: Action.FACE_UP,
    Action.DOWN: Action.FACE_DOWN,
    Action.LEFT: Action.FACE_LEFT,
    Action.RIGHT: Action.FACE_RIGHT,
}
_OPPOSITE = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}
MAX_SITES = min(count for _, count in START_INVENTORY)


class Macro(NamedTuple):
    """Build at the first ``count`` sites of ``patch``."""

    patch: str
    # The plan text calls it count; shadowing tuple.count is harmless here.
    count: int  # type: ignore[assignment]


_level = cache(make_level)


@cache
def _blocked(variant: int) -> frozenset[Tile]:
    """Return static obstacles plus every tile a machine may later stand on."""
    level = _level(variant)
    h, w = level.block_map.shape
    solid = {
        (x, y)
        for y in range(h)
        for x in range(w)
        if level.block_map[y, x] in (BlockType.WATER, BlockType.OUT_OF_BOUNDS)
    }
    machine_tiles = {
        t
        for patch in sites(variant).values()
        for s in patch
        for t in (s.miner, s.pallet)
    }
    return frozenset(solid | machine_tiles)


def _path(start: Tile, goal: Tile, variant: int) -> list[Action]:
    level = _level(variant)
    blocked = _blocked(variant) - {goal}
    came_from: dict[Tile, tuple[Tile, Action] | None] = {start: None}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        for move, (dx, dy) in _OFFSETS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            in_bounds = 0 <= nxt[0] < level.map_width and 0 <= nxt[1] < level.map_height
            if in_bounds and nxt not in blocked and nxt not in came_from:
                came_from[nxt] = (pos, move)
                queue.append(nxt)
    if goal not in came_from:
        raise ValueError(f"No path from {start} to {goal} in variant {variant}")
    moves = []
    while (step := came_from[goal]) is not None:
        goal, move = step
        moves.append(move)
    return moves[::-1]


def compile_plan(macros: Sequence[Macro], player_idx: int, variant: int) -> np.ndarray:
    """Build at the first ``count`` sites of each named patch, in table order.

    Both players share the table order, so two players naming the same patch
    collide on the same sites. The later placement fails silently in the engine.
    """
    pos = spawns(variant)[player_idx]
    budget = MAX_SITES
    actions: list[Action] = []
    for macro in macros:
        if budget == 0 or len(actions) >= HORIZON:
            break
        n = min(macro.count, budget)
        budget -= n
        for site in sites(variant)[macro.patch][:n]:
            actions += _path(pos, site.miner, variant)
            actions += [
                _FACE[site.facing],
                Action.PLACE_PALLET,
                _OPPOSITE[site.facing],
                _FACE[site.facing],
                Action.PLACE_MINER,
            ]
            pos = site.stand
    out = np.full(HORIZON, Action.NOOP, dtype=np.int32)
    actions = actions[:HORIZON]
    out[: len(actions)] = actions
    return out


def compile_joint(plans: Sequence[Sequence[Macro]], variant: int) -> np.ndarray:
    """Stack one compiled plan per player into ``(HORIZON, NUM_PLAYERS)``."""
    return np.stack(
        [compile_plan(plan, i, variant) for i, plan in enumerate(plans)], axis=1
    )
