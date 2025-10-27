from __future__ import annotations
from typing import Union

import numpy as np
from numpy.typing import NDArray
from poke_env.battle import Battle, Pokemon
from poke_env.battle.pokemon_type import STANDARD_TYPES
from poke_env.battle.stat import MOVABLE_STATS
from poke_env.data import GenData

from encoder.utils import pad_to_length, MAX_TEAM_SIZE
from poke_env.battle.effect import Effect
from poke_env.battle.status import STATUS_STRINGS, STATUSES, Status
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.move import Move

MOVE_CATEGORIES = (
    MoveCategory.PHYSICAL,
    MoveCategory.SPECIAL,
    MoveCategory.STATUS,
)

VOLATILE_STATUSES = (Effect.CONFUSION,)

BASE_FEATURES = ("BASE_POWER", "ACCURACY", "TARGET", "CURRENT_PP", "0_PP")
STAB_FEATURES = ("STAB_ORIGINAL", "STAB_TERA")

# Curated sets for effects not cleanly tagged in Showdown data
HAZARD_REMOVERS = ("rapidspin", "defog", "mortalspin", "tidyup")
ITEM_REMOVERS = ("knockoff", "thief", "covet", "corrosivegas", "trick", "switcheroo")

HAZARD_REMOVERS_SET = frozenset(HAZARD_REMOVERS)
ITEM_REMOVERS_SET = frozenset(ITEM_REMOVERS)

SPECIAL_CASES: list[Union[str, frozenset[str]]] = [
    ITEM_REMOVERS_SET,
    HAZARD_REMOVERS_SET,
    "stealthrock",
    "ruination",
    "wish",
    "painsplit",
    "grassyglide",
]

MAX_BASE_POWER = 130

CORE_FEATURES_DIM = (
    len(BASE_FEATURES) + len(MOVE_CATEGORIES) + MAX_TEAM_SIZE + len(STAB_FEATURES)
)
SPECIAL_CASES_DIM = len(SPECIAL_CASES)
SECONDARY_EFFECTS_DIM = len(MOVABLE_STATS) * 2 + 6
ENCODED_MOVE_DIM = (
    CORE_FEATURES_DIM
    + len(STATUSES)
    + len(VOLATILE_STATUSES)
    + 1  # protect counter
    + SECONDARY_EFFECTS_DIM
    + SPECIAL_CASES_DIM
)


class MoveEncoder:
    def __init__(self):
        pass

    @staticmethod
    def _status_prob(move: Move) -> NDArray[np.float32]:
        """
        Encodes the status of a move.

        :param move: The move to encode.
        :return: A numpy array of shape (len(STATUSES),) containing the encoded features.
        """
        status, chance = None, 1.0
        if move.category == MoveCategory.STATUS:
            return np.array(
                [1.0 if s == move.status else 0.0 for s in STATUSES], dtype=np.float32
            )
        else:
            secs = move.secondary or []
            sts = next((s for s in secs if s.get("status") in STATUS_STRINGS), None)
            if sts:
                status = Status[sts["status"].upper()]  # could KeyError if mismatched
                chance = sts.get("chance", 100) / 100.0
            else:
                status = None
            return np.array(
                [chance if s == status else 0.0 for s in STATUSES], dtype=np.float32
            )

    @staticmethod
    def _protect_counter_encoder(move: Move) -> NDArray[np.float32]:
        """
        Encodes the protect counter of a move.

        :param move: The move to encode.
        :return: A numpy array of shape (1,) containing the encoded features.
        """
        return np.array([1.0 if move.is_protect_counter else 0.0], dtype=np.float32)

    @staticmethod
    def _volatile_status_prob(move: Move) -> NDArray[np.float32]:
        """
        Encodes the volatile status of a move.

        :param move: The move to encode.
        :return: A numpy array of shape (len(VOLATILE_STATUSES),) containing the encoded features.
        """
        if move.volatile_status_chance:
            return np.array(
                [
                    (
                        move.volatile_status_chance.get("chance", 1.0)
                        if move.volatile_status_chance.get("effect") == s
                        else 0.0
                    )
                    for s in VOLATILE_STATUSES
                ],
                dtype=np.float32,
            )
        return np.array([0.0] * len(VOLATILE_STATUSES), dtype=np.float32)

    @staticmethod
    def _encode_category(move: Move) -> NDArray[np.float32]:
        """
        Encodes the category of a move.
        typestypestypes
        :param move: The move to encode.
        :return: A numpy array of shape (len(MOVE_CATEGORIES),) containing the encoded features.
        """
        # Output shape: (len(MOVE_CATEGORIES),)
        return np.array(
            [1.0 if move.category == c else 0.0 for c in MOVE_CATEGORIES],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_stab(move: Move, pkmn: Pokemon) -> NDArray[np.float32]:
        """
        Encodes whether the move gains STAB from the pokemon's original or tera type.

        :param move: The move to encode.
        :param pkmn: The pokemon to encode.
        :return: A numpy array of shape (len(STAB_FEATURES),) containing the encoded features.
        """
        return np.array(
            [
                1.0 if move.type in pkmn.original_types else 0.0,
                1.0 if move.type == pkmn.tera_type else 0.0,
            ],
            dtype=np.float32,
        )

    # +1's come from: base power, accuracy, target
    @staticmethod
    def _encode_core_features(
        move: Move, pkmn: Pokemon, battle: Battle
    ) -> NDArray[np.float32]:
        """
        Encodes the base features of a move:
        base power (1 dim), accuracy (1 dim), current pp (1 dim), 0 pp (1 dim), category (3 dim), target (1 dim), type (19*2 dim)

        :param move: The move to encode.
        :return: A numpy array of shape (CORE_FEATURES_DIM) containing the encoded features.
        """
        return np.array(
            [
                move.base_power / MAX_BASE_POWER,
                move.accuracy,
                move.current_pp / move.max_pp,
                0.0 if move.current_pp == 0 else 1.0,
                0.0 if move.target == "self" else 1.0,
                *MoveEncoder._encode_category(move),
                *MoveEncoder._encode_type_multipliers(move, battle),
                *MoveEncoder._encode_stab(move, pkmn),
                # *type_one_hot([move.type]),
                # *type_effectiveness([move.type]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_self_stat_change(move: Move) -> NDArray[np.float32]:
        """
        Encodes the self stat change of a move.

        :param move: The move to encode.
        :return: a numpy array of shape (len(MOVABLE_STATS),) containing the encoded features.
        """
        return (
            np.array(
                [move.boosts_self.get(s.name, 0) for s in MOVABLE_STATS],
                dtype=np.float32,
            )
            / 2.0
        )

    @staticmethod
    def _encode_type_multipliers(move: Move, battle: Battle) -> NDArray[np.float32]:
        """
        Encodes the type multipliers of a move vs each enemy pokemon.

        :param move: The move to encode.
        :return: a numpy array of shape (MAX_TEAM_SIZE,) containing the encoded features.
        """

        type_multipliers = np.array(
            [
                move.type.damage_multiplier(
                    foe.type_1, foe.type_2, type_chart=GenData.from_gen(9).type_chart
                )
                / 4.0  # Normalize to [0, 1]
                for foe in battle.opponent_team.values()
            ],
            dtype=np.float32,
        )
        return pad_to_length(type_multipliers)

    @staticmethod
    def _encode_foe_stat_change(move: Move) -> NDArray[np.float32]:
        """
        Encodes the foe stat change of a move.

        :param move: The move to encode.
        :return: a numpy array of shape (len(MOVABLE_STATS),) containing the encoded features.
        """
        return (
            np.array(
                [move.boosts_target.get(s.name, 0) for s in MOVABLE_STATS],
                dtype=np.float32,
            )
            / 2.0
        )

    @staticmethod
    def _encode_special_cases(move: Move) -> NDArray[np.float32]:
        """
        Encodes special cases of a move.

        :param move: The move to encode.
        :return: A numpy array of shape (SPECIAL_CASES_DIM,) containing the encoded features.
        """
        return np.array(
            [
                (
                    1.0
                    if (
                        (isinstance(case, str) and move.id == case)
                        or (not isinstance(case, str) and move.id in case)
                    )
                    else 0.0
                )
                for case in SPECIAL_CASES
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_secondary_effects(move: Move) -> NDArray[np.float32]:
        """
        Encodes secondary effects.

        :param move: The move to encode.
        :return: A numpy array of shape (SECONDARY_EFFECTS_DIM,) containing the encoded features.
        """
        feats = (
            float(move.recoil),
            float(move.thaws_user),
            float(move.is_protect_move),
            float(move.is_side_protect_move),
            float(move.self_switch),
            float(move.force_switch),
        )
        return np.array(
            [
                *MoveEncoder._encode_self_stat_change(move),
                *MoveEncoder._encode_foe_stat_change(move),
                *feats,
            ],
            dtype=np.float32,
        )

    @classmethod
    def encode_move(
        cls, move: Move, pkmn: Pokemon, battle: Battle
    ) -> NDArray[np.float32]:
        """
        Encodes a move into a one-hotish probability vector.

        :param move: The move to encode.
        :return: A numpy array of shape (ENCODED_MOVE_DIM,) containing the encoded features.
        """
        return np.concatenate(
            [
                cls._encode_core_features(move, pkmn, battle),
                cls._status_prob(move),
                cls._volatile_status_prob(move),
                cls._protect_counter_encoder(move),
                cls._encode_secondary_effects(move),
                cls._encode_special_cases(move),
            ]
        )
