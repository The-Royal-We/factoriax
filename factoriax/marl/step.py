"""Multi-player step: every player acts, then the factory runs once."""

import jax
import jax.numpy as jnp

from factoriax.engine.constants import NUM_SCIENCE_PACK_TYPES
from factoriax.engine.machines import update_all_machines
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import _handle_player_action, run_labs


def multi_step(
    key: jax.Array,
    state: EnvState,
    actions: jax.Array,
    params: EnvParams,
) -> EnvState:
    """Every player acts once, in a random order, then the factory runs once.

    The order is shuffled by ``key`` so no player always wins a contested tile.
    """
    state = state.replace(
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
    )
    order = jax.random.permutation(key, actions.shape[0])

    def act(i: int, s: EnvState) -> EnvState:
        p = order[i]
        return _handle_player_action(s, params, actions[p], p)

    state = jax.lax.fori_loop(0, actions.shape[0], act, state)
    state = update_all_machines(state, params)
    state = run_labs(state)
    return state.replace(timestep=state.timestep + 1)
