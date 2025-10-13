from gymnasium.spaces import Discrete
import numpy as np

from poke_env.battle.battle import Battle


def action_masker(battle: Battle, action_space: Discrete) -> np.ndarray:
    mask = np.zeros(action_space.n)
    # switches
    indices = [
        i
        for i, pokemon in enumerate(battle.team.values())
        if pokemon in battle.available_switches
    ]
    mask[indices] = 1

    # moves
    indices = [
        i + 6
        for i, move in enumerate(battle.available_moves)
        if move.current_pp > 0 and not battle.force_switch
    ]
    if not battle.used_tera:
        indices += [i + 4 for i in indices]

    mask[indices] = 1
    return mask
