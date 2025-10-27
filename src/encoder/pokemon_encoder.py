from numpy.typing import NDArray
from poke_env.battle import Battle
from poke_env.battle.pokemon import Pokemon
import numpy as np
from poke_env.battle.stat import STATS
from poke_env.battle.pokemon_type import STANDARD_TYPES
from poke_env.battle.status import STATUSES
from encoder.move_encoder import MoveEncoder, ENCODED_MOVE_DIM
from encoder.utils import type_one_hot


class PokemonEncoder:
    # Class variables

    ABILITIES = (
        "speedboost",
        "regenerator",
        "grassysurge",
        "vesselofruin",
        "neutralizinggas",
    )
    ITEMS = ("lifeorb", "choiceband", "leftovers", "heavydutyboots")
    HP_FEATURES = ("CURRENT_HP_FRACTION", "FAINTED")
    ENCODED_MOVES_DIM = ENCODED_MOVE_DIM * 4

    OP_POKEMON_FEATURES_DIM = (
        len(STANDARD_TYPES)
        + len(STATS)
        + len(STATUSES)
        # + len(ITEMS)
        # + len(ABILITIES)
        + len(HP_FEATURES)
        + 1  # protect counter
    )
    MY_POKEMON_FEATURES_DIM = OP_POKEMON_FEATURES_DIM + ENCODED_MOVES_DIM

    def __init__(self):
        pass

    @staticmethod
    def _encode_base_stats(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the base stats of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(STATS),) containing the encoded features.
        """
        return (
            np.array(
                [pkmn.base_stats[stat.name.lower()] for stat in STATS], dtype=np.float32
            )
            / 255
        )

    @staticmethod
    def _protect_counter_encoder(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the protect counter of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (1,) containing the encoded features.
        """
        return np.array([1.0 if pkmn.protect_counter > 0 else 0.0], dtype=np.float32)

    @staticmethod
    def _hp_encoder(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the hp of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(HP_FEATURES),) containing the encoded features.
        """
        return np.array(
            [pkmn.current_hp_fraction, 1.0 if pkmn.fainted else 0.0], dtype=np.float32
        )

    @staticmethod
    def _status_encoder(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the status of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(STATUSES),) containing the encoded features.
        """
        return np.array(
            [1 if pkmn.status == sts else 0 for sts in STATUSES], dtype=np.int8
        )

    @classmethod
    def _item_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the item of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(ITEMS),) containing the encoded features.
        """
        return np.array(
            [1 if pkmn.item == item else 0 for item in cls.ITEMS], dtype=np.int8
        )

    @classmethod
    def _ability_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the ability of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(ABILITIES),) containing the encoded features.
        """
        return np.array(
            [1 if pkmn.ability == ability else 0 for ability in cls.ABILITIES],
            dtype=np.int8,
        )

    @staticmethod
    def moves_encoder(pkmn: Pokemon, battle: Battle) -> NDArray[np.float32]:
        """
        Encodes the moves of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (ENCODED_MOVES_DIM) containing the encoded features.
        """
        encoded_moves = [
            MoveEncoder.encode_move(move, pkmn, battle) for move in pkmn.moves.values()
        ]
        # Pad to exactly 4 moves with zeros to ensure fixed length
        num_missing = 4 - len(encoded_moves)
        if num_missing > 0:
            encoded_moves.extend(
                [np.zeros(ENCODED_MOVE_DIM, dtype=np.float32)] * num_missing
            )
        elif num_missing < 0:
            encoded_moves = encoded_moves[:4]
        return np.concatenate(encoded_moves)

    @classmethod
    def opponent_pokemon_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes a opponent pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (OP_POKEMON_FEATURES_DIM,) containing the encoded features.
        """
        return np.concatenate(
            [
                type_one_hot(pkmn.types),
                # type_effectiveness(pkmn.original_types),
                # type_one_hot([pkmn.tera_type] if pkmn.tera_type else None),
                # type_effectiveness([pkmn.tera_type] if pkmn.tera_type else None),
                cls._encode_base_stats(pkmn),
                cls._hp_encoder(pkmn),
                cls._status_encoder(pkmn),
                # cls._item_encoder(pkmn),
                # cls._ability_encoder(pkmn),
                cls._protect_counter_encoder(pkmn),
            ]
        )

    @classmethod
    def my_pokemon_encoder(cls, pkmn: Pokemon, battle: Battle) -> NDArray[np.float32]:
        """
        Encodes a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (MY_POKEMON_FEATURES_DIM,) containing the encoded features.
        """
        return np.concatenate(
            [
                type_one_hot(pkmn.types),
                # type_effectiveness(pkmn.original_types),
                # type_one_hot([pkmn.tera_type] if pkmn.tera_type else None),
                # type_effectiveness([pkmn.tera_type] if pkmn.tera_type else None),
                cls._encode_base_stats(pkmn),
                cls._hp_encoder(pkmn),
                cls._status_encoder(pkmn),
                # cls._item_encoder(pkmn),
                # cls._ability_encoder(pkmn),
                cls._protect_counter_encoder(pkmn),
                cls.moves_encoder(pkmn, battle),
            ]
        )

    @classmethod
    def zeros_opponent_features(cls) -> NDArray[np.float32]:
        """Returns a zero vector matching OP_POKEMON_FEATURES_DIM."""
        return np.zeros(cls.OP_POKEMON_FEATURES_DIM, dtype=np.float32)

    @classmethod
    def zeros_my_features(cls) -> NDArray[np.float32]:
        """Returns a zero vector matching MY_POKEMON_FEATURES_DIM."""
        return np.zeros(cls.MY_POKEMON_FEATURES_DIM, dtype=np.float32)
