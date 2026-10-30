from factoriax.engine.constants import ItemType, Machine
from factoriax.engine.levels import build_state
from factoriax.marl.ore_split import (
    level_builder,
    make_state,
    sites,
    spawns,
    team_score,
)


def state_with_pallets(variant, iron, copper):
    level = (
        level_builder(variant)
        .place_machine(0, 0, Machine.PALLET)
        .set_machine_inventory(0, 0, ItemType.IRON_ORE, iron)
        .place_machine(1, 0, Machine.PALLET)
        .set_machine_inventory(1, 0, ItemType.COPPER_ORE, copper)
        .build("pallets")
    )
    return build_state(level, num_players=2, max_machines=32)


def test_variants_swap_resources():
    assert sites(0)["iron"][0].miner != sites(1)["iron"][0].miner


def test_variants_swap_spawns():
    assert spawns(0) == tuple(reversed(spawns(2)))


def test_site_geometry_is_collinear():
    for v in range(4):
        for site in (s for patch in sites(v).values() for s in patch):
            dx, dy = site.miner[0] - site.stand[0], site.miner[1] - site.stand[1]
            assert abs(dx) + abs(dy) == 1
            assert site.pallet == (site.miner[0] + dx, site.miner[1] + dy)


def test_team_score_is_min_of_resources():
    state = state_with_pallets(variant=0, iron=7, copper=3)
    assert float(team_score(state)) == 3.0


def test_empty_factory_scores_zero():
    state, _ = make_state(0)
    assert float(team_score(state)) == 0.0
