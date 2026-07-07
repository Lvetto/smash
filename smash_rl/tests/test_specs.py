"""Unit test dei registry di specs (azioni, osservazioni, reward)."""
import melee

from smash_rl.specs import ACT_SPECS, OBS_SPECS, REWARD_FNS
from smash_rl.specs.actions import STICK_MAP, BUTTONS_FULL, BUTTONS_A_ONLY


def test_full_action_spec_covers_all_combinations():
    space, decode = ACT_SPECS["full"]
    assert space.n == len(STICK_MAP) * len(BUTTONS_FULL)  # 54

    seen = set()
    for action in range(space.n):
        (x, y), button = decode(action)
        assert (x, y) in STICK_MAP.values()
        assert button in BUTTONS_FULL.values()
        seen.add(((x, y), button))
    assert len(seen) == space.n, "ogni azione deve decodificare in una combinazione unica"


def test_a_only_action_spec():
    space, decode = ACT_SPECS["a_only"]
    assert space.n == len(STICK_MAP) * len(BUTTONS_A_ONLY)  # 18

    buttons = {decode(action)[1] for action in range(space.n)}
    assert buttons == {None, melee.Button.BUTTON_A}


def test_neutral_action_is_zero():
    _, decode = ACT_SPECS["full"]
    assert decode(0) == ((0.5, 0.5), None)


def test_registries_expose_default_specs():
    assert "full_v1" in OBS_SPECS
    space, _ = OBS_SPECS["full_v1"]
    assert space.shape == (32,)

    assert "v1" in REWARD_FNS
