import numpy as np

from numpy.typing import NDArray
from poke_env.battle import AbstractBattle, Battle

from encoder.pokemon_encoder import PokemonEncoder

class Encoder:
    MY_TEAM_FEATURES_DIM = PokemonEncoder.MY_POKEMON_FEATURES_DIM * 6 + 1
    OP_TEAM_FEATURES_DIM = PokemonEncoder.OP_POKEMON_FEATURES_DIM * 6 + 1
    BATTLE_FEATURES_DIM = MY_TEAM_FEATURES_DIM + OP_TEAM_FEATURES_DIM
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
    def encode_own_team(battle: AbstractBattle) -> NDArray[np.float32]:
        """
        Encodes the team of the player.

        :param battle: The battle to encode.
        :return: A numpy array of shape (Encoder.MY_TEAM_FEATURES_DIM,) containing the encoded features.
        """
        assert isinstance(battle, Battle)
        return np.concatenate([PokemonEncoder.my_pokemon_encoder(pkmn) for pkmn in battle.team.values()] 
            + [np.array([1.0 if battle.used_tera else 0.0])] )

    @staticmethod
    def encode_opponent_team(battle: AbstractBattle):
        """
        Encodes the team of the opponent.

        :param battle: The battle to encode.
        :return: A numpy array of shape (Encoder.OP_TEAM_FEATURES_DIM,) containing the encoded features.
        """
        assert isinstance(battle, Battle)
        return np.concatenate([PokemonEncoder.opponent_pokemon_encoder(pkmn) for pkmn in battle.opponent_team.values()] 
            + [np.array([1.0 if battle.opponent_used_tera else 0.0])])