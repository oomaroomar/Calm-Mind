from numpy.typing import NDArray
from poke_env.battle.pokemon import Pokemon
import numpy as np
from poke_env.battle.pokemon_type import STANDARD_TYPES
from poke_env.battle.status import STATUSES
from encoder.encoder import Encoder

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
    POKEMON_FEATURES_DIM = (
        len(STANDARD_TYPES) * 4 
        + len(STATUSES) 
        + len(ITEMS) 
        + len(ABILITIES) 
        + len(HP_FEATURES)
    )

    def __init__(self):
        pass
    


    @staticmethod
    def _hp_encoder(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the hp of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(HP_FEATURES),) containing the encoded features.
        """
        return np.array([pkmn.current_hp_fraction, 1.0 if pkmn.fainted else 0.0], dtype=np.float32)

    @staticmethod
    def _status_encoder(pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the status of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(STATUSES),) containing the encoded features.
        """
        return np.array([1 if pkmn.status == sts else 0 for sts in STATUSES], dtype=np.int8)

    @classmethod
    def _item_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the item of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(ITEMS),) containing the encoded features.
        """
        return np.array([1 if pkmn.item == item else 0 for item in cls.ITEMS], dtype=np.int8)

    @classmethod
    def _ability_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes the ability of a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(ABILITIES),) containing the encoded features.
        """
        return np.array([1 if pkmn.ability == ability else 0 for ability in cls.ABILITIES], dtype=np.int8)


    @classmethod
    def pokemon_encoder(cls, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes a pokemon.

        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (POKEMON_FEATURES_DIM,) containing the encoded features.
        """
        return np.concatenate([
            Encoder.type_one_hot(pkmn.original_types),
            Encoder.type_effectiveness(pkmn.original_types),
            Encoder.type_one_hot([pkmn.tera_type] if pkmn.tera_type else None),
            Encoder.type_effectiveness([pkmn.tera_type] if pkmn.tera_type else None),
            cls._hp_encoder(pkmn),
            cls._status_encoder(pkmn),
            cls._item_encoder(pkmn),
            cls._ability_encoder(pkmn),
        ])