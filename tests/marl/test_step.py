import jax
import jax.numpy as jnp
import pytest

from factoriax.engine.constants import Action
from factoriax.engine.levels import LevelBuilder, build_state
from factoriax.engine.state import EnvParams
from factoriax.engine.step import factoriax_step
from factoriax.marl.step import multi_step


@pytest.fixture
def two_player_state():
    level = (
        LevelBuilder(8, 8)
        .set_player_position(2, 4)
        .set_player_position(5, 4)
        .build("t")
    )
    return build_state(level, num_players=2)


@pytest.fixture
def params():
    return EnvParams()


def test_both_players_move_in_one_step(two_player_state, params):
    actions = jnp.array([Action.RIGHT, Action.LEFT], dtype=jnp.int32)
    new = multi_step(jax.random.PRNGKey(0), two_player_state, actions, params)
    assert new.player_positions[0, 0] == two_player_state.player_positions[0, 0] + 1
    assert new.player_positions[1, 0] == two_player_state.player_positions[1, 0] - 1
    assert new.timestep == two_player_state.timestep + 1


def test_factory_ticks_once_per_step(two_player_state, params):
    noop = jnp.zeros(2, dtype=jnp.int32)
    new = multi_step(jax.random.PRNGKey(0), two_player_state, noop, params)
    single = factoriax_step(
        jax.random.PRNGKey(0), two_player_state, Action.NOOP, params
    )
    assert jnp.array_equal(new.ent_buf_count, single.ent_buf_count)


def test_multi_step_jits(two_player_state, params):
    jax.jit(multi_step)(
        jax.random.PRNGKey(0), two_player_state, jnp.zeros(2, jnp.int32), params
    )
