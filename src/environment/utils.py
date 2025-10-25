from gymnasium.spaces import Discrete
import numpy as np

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move


def action_masker(battle: Battle, action_space: Discrete = Discrete(14)) -> np.ndarray:
    mask = np.zeros(action_space.n)

    # Safety check: if battle is finished or only default orders available, return all zeros
    # This signals that no actions are valid (environment should reset)
    if battle.finished:
        return mask
    if not battle.valid_orders or (
        len(battle.valid_orders) == 1
        and str(battle.valid_orders[0]) == "/choose default"
    ):
        return mask

    # switches
    team = list(battle.team.values())
    switch_indices = [
        i for i, pokemon in enumerate(team) if pokemon in battle.available_switches
    ]
    # Explicitly ensure the active pokemon is never a switch target, even if available_switches is inconsistent/empty
    # if battle.active_pokemon in team:
    #     active_idx = team.index(battle.active_pokemon)
    #     switch_indices = [i for i in switch_indices if i != active_idx]
    mask[switch_indices] = 1

    if battle.active_pokemon is None or battle.force_switch:
        return mask

    # moves
    move_indices = [
        i + 6
        for i, move in enumerate(battle.active_pokemon.moves.values())
        if move in battle.available_moves
    ]

    if not battle.used_tera:
        move_indices += [i + 4 for i in move_indices]

    mask[move_indices] = 1
    return mask
