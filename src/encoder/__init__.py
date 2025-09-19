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
    def embed_battle(battle: AbstractBattle):
        """
        Encodes the battle state

        :param battle: The battle to encode
        :return: A numpy array of shape (Encoder.BATTLE_FEATURES_DIM, )
        """
        assert isinstance(battle, Battle)
        return np.concatenate(
            [Encoder.encode_own_team(battle), Encoder.encode_opponent_team(battle)]
        )

    @staticmethod
    def encode_own_team(battle: AbstractBattle) -> NDArray[np.float32]:
        """
        Encodes the team of the player.

        :param battle: The battle to encode.
        :return: A numpy array of shape (Encoder.MY_TEAM_FEATURES_DIM,) containing the encoded features.
        """
        assert isinstance(battle, Battle)
        encs = [
            PokemonEncoder.my_pokemon_encoder(pkmn) for pkmn in battle.team.values()
        ]
        # Pad to exactly 6 team members
        if len(encs) < 6:
            encs.extend([PokemonEncoder.zeros_my_features()] * (6 - len(encs)))
        elif len(encs) > 6:
            encs = encs[:6]
        return np.concatenate(
            encs + [np.array([1.0 if battle.used_tera else 0.0], dtype=np.float32)]
        )

    @staticmethod
    def encode_opponent_team(battle: AbstractBattle):
        """
        Encodes the team of the opponent.

        :param battle: The battle to encode.
        :return: A numpy array of shape (Encoder.OP_TEAM_FEATURES_DIM,) containing the encoded features.
        """
        assert isinstance(battle, Battle)
        encs = [
            PokemonEncoder.opponent_pokemon_encoder(pkmn)
            for pkmn in battle.opponent_team.values()
        ]
        # Pad to exactly 6 team members
        if len(encs) < 6:
            encs.extend([PokemonEncoder.zeros_opponent_features()] * (6 - len(encs)))
        elif len(encs) > 6:
            encs = encs[:6]
        return np.concatenate(
            encs
            + [np.array([1.0 if battle.opponent_used_tera else 0.0], dtype=np.float32)]
        )
