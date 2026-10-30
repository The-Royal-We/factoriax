"""Render the per-agent prompt and parse ``BUILD`` lines out of a reply."""

import re

from factoriax.marl.macros import MAX_SITES, Macro
from factoriax.marl.ore_split import NUM_PLAYERS, PATCHES, sites, spawns

_BUILD_LINE = re.compile(
    r"^\s*BUILD\s+(\w+)\s+(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE
)


def _direction(src: tuple[int, int], dst: tuple[float, float]) -> str:
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    vertical = "north" if dy <= -2 else "south" if dy >= 2 else ""
    horizontal = "west" if dx <= -2 else "east" if dx >= 2 else ""
    return "-".join(filter(None, (vertical, horizontal))) or "here"


def _distance(src: tuple[int, int], variant: int, patch: str) -> int:
    return min(
        abs(s.miner[0] - src[0]) + abs(s.miner[1] - src[1])
        for s in sites(variant)[patch]
    )


def render_prompt(player_idx: int, variant: int) -> str:
    """Describe both patches from this agent's and its teammate's spawn."""
    me = spawns(variant)[player_idx]
    mate = spawns(variant)[1 - player_idx]
    patch_lines = []
    for patch in PATCHES:
        miners = [s.miner for s in sites(variant)[patch]]
        centre = (
            sum(x for x, _ in miners) / len(miners),
            sum(y for _, y in miners) / len(miners),
        )
        patch_lines.append(
            f"- {patch}: {_direction(me, centre)} of you, "
            f"{_distance(me, variant, patch)} steps from you, "
            f"{_distance(mate, variant, patch)} steps from your teammate"
        )
    return "\n".join(
        [
            f"You are agent {player_idx} of {NUM_PLAYERS} in a factory game. "
            "You and your teammate act at the same time and cannot talk.",
            "Objective: team score = min(iron ore delivered, copper ore delivered). "
            "Both resources are needed.",
            f"Your inventory: {MAX_SITES} miners, {MAX_SITES} pallets. "
            "Each BUILD places one miner and one pallet per count.",
            f"Ore patches ({MAX_SITES} sites each; if both agents build on the same "
            "patch, they compete for the same sites):",
            *patch_lines,
            "Output format: one line per order, BUILD <iron|copper> <count>. Example:",
            "BUILD iron 2",
            "Output only BUILD lines.",
        ]
    )


def parse_plan(text: str) -> tuple[list[Macro], bool]:
    """Pull ``BUILD`` lines out of free text, clamping counts and dropping bad ones.

    The flag is False when no usable line was found.
    """
    macros = []
    for name, count in _BUILD_LINE.findall(text):
        patch, n = name.lower(), int(count)
        if patch in PATCHES and n > 0:
            macros.append(Macro(patch, min(n, MAX_SITES)))
    return macros, bool(macros)
