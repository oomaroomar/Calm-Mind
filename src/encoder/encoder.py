from typing import Dict, List, Optional, Set

import numpy as np

from poke_env.battle.pokemon import Pokemon
from poke_env.data import GenData
from poke_env.battle import AbstractBattle, Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import ALL_TYPES, STANDARD_TYPES, PokemonType
from poke_env.battle.status import Status

class Encoder:
    def __init__(self):
        pass

    @staticmethod
    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    @staticmethod
    def encode_pkmn_types(battle: AbstractBattle):
        assert isinstance(battle, Battle)
        team_types = []
        opponent_team_types = []
        for pkmn in battle.team.values():
            team_types.append(pkmn.original_types)
        for pkmn in battle.opponent_team.values():
            opponent_team_types.append(pkmn.original_types)
        return team_types, opponent_team_types

    def _encode_move(self, move: Move):
        bp = move.base_power/100
        move_type = move.type

    @staticmethod
    def encode_offensive_type(battle: AbstractBattle):
        assert isinstance(battle, Battle)
        offensive_type = battle.active_pokemon.type_1
        if battle.active_pokemon.type_2:
            offensive_type = battle.active_pokemon.type_2
        return offensive_type
    
    @staticmethod
    def encode_defensive_type(battle: AbstractBattle):
        pass

    @staticmethod
    def type_one_hot(
        types: List[PokemonType] | None = None,
        include_special: bool = False
    ) -> np.ndarray:
        """
        Returns a one-hot encoding for one or more Pokemon types.

        :param types: A list of PokemonType(s) to encode (1 or 2 usually).
        :param include_special: Whether to include ??? in the vector.
        :return: A numpy array with one-hot encoding.
                Length = 19 if include_special=False (incl. Stellar),
                else 20 (incl. ??? as well).
        """

        type_list = ALL_TYPES if include_special else STANDARD_TYPES

        if types is None:
            return np.zeros(len(type_list), dtype=np.int8)

        type_vector = np.array([1 if t in types else 0 for t in type_list], dtype=np.int8)

        return type_vector

    @staticmethod
    def type_effectiveness(
        user_types: List[PokemonType] | None = None,
        type_chart: Dict[str, Dict[str, float]] = GenData.from_gen(9).type_chart,
        include_special: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Returns an effectiveness vector for the given type(s).

        Each entry i corresponds to the damage multiplier of the i-th type

        :param user_types: A list of 1 or 2 PokemonType(s) for the user Pokémon.
        :param type_chart: Nested dict of multipliers, e.g. type_chart[defender][attacker] = float.
        :param include_special: If True, includes ??? type in the vector.
        :param normalize: If True, divides multipliers by 4 so values are in [0,1].
        :return: A numpy array of shape (num_types,), containing multipliers.
        """
        type_list = ALL_TYPES if include_special else STANDARD_TYPES

        if user_types is None:
            return np.ones(len(type_list), dtype=np.float32)

        te_vector = np.array([t.damage_multiplier(user_types[0], user_types[1] if len(user_types) > 1 else None, type_chart=type_chart) for t in type_list], dtype=np.float32)

        if normalize:
            te_vector /= 4.0

        return te_vector


used_items = [
    "Heavy-Duty Boots",
    "Choice Band",
    "Leftovers",
    "Life Orb",
]

ITEM_TO_INDEX = {t: i for i, t in enumerate(used_items)}

def item_one_hot(item: str):
    vec = np.zeros(len(used_items), dtype=np.float32)
    vec[ITEM_TO_INDEX[item]] = 1.0
    return vec


def encode_moves_bp(pkmn: Pokemon):
    moves_base_power = np.zeros(4)
    for i, move in enumerate(pkmn.moves.values()):
        moves_base_power[i] = move.base_power / 120
    return moves_base_power
