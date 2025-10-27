import numpy as np
from typing import Dict, List

from poke_env.battle import Pokemon, PokemonType
from poke_env.battle.pokemon_type import ALL_TYPES, STANDARD_TYPES
from poke_env.data import GenData

MAX_TEAM_SIZE = 6


def type_one_hot(
    types: List[PokemonType] | None = None, include_special: bool = False
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

    te_vector = np.array(
        [
            t.damage_multiplier(
                user_types[0],
                user_types[1] if len(user_types) > 1 else None,
                type_chart=type_chart,
            )
            for t in type_list
        ],
        dtype=np.float32,
    )

    if normalize:
        te_vector /= 4.0

    return te_vector


def pad_to_length(array: np.ndarray, length: int = MAX_TEAM_SIZE) -> np.ndarray:
    """
    Pads an array to the given length with zeros.

    :param array: The array to pad.
    :param length: The length to pad to.
    :return: The padded array.
    """
    if len(array) < length:
        return np.concatenate([array, np.zeros(length - len(array), dtype=np.float32)])
    elif len(array) > length:
        return array[:length]
    return array
