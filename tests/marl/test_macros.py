import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import Action, ItemType, Machine
from factoriax.marl.macros import Macro, compile_joint, compile_plan
from factoriax.marl.ore_split import HORIZON, make_state, sites
from factoriax.marl.step import multi_step


def run_actions(state, params, actions):
    def tick(s, a):
        return multi_step(
            jax.random.fold_in(jax.random.PRNGKey(0), s.timestep), s, a, params
        ), None

    final, _ = jax.lax.scan(tick, state, jnp.asarray(actions))
    return final


def test_executed_plan_builds_one_working_site():
    state, params = make_state(0)
    actions = compile_joint([[Macro("iron", 1)], []], variant=0)
    final = run_actions(state, params, actions)
    site = sites(0)["iron"][0]
    assert final.machine_types[site.miner[1], site.miner[0]] == Machine.MINER
    assert final.machine_types[site.pallet[1], site.pallet[0]] == Machine.PALLET
    assert final.player_inventory[0, ItemType.MINER] == 3


def test_compile_truncates_to_horizon():
    actions = compile_plan([Macro("iron", 4), Macro("copper", 4)] * 10, 0, 0)
    assert actions.shape == (HORIZON,)


def test_inventory_cap():
    actions = compile_plan([Macro("iron", 4), Macro("copper", 4)], 0, 0)
    assert int(np.sum(actions == Action.PLACE_MINER)) == 4


def test_empty_plan_is_all_noop():
    assert np.all(compile_plan([], 0, 0) == Action.NOOP)
