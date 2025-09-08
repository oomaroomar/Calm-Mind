from typing import Dict
import numpy as np
from numpy.typing import NDArray
from poke_env.battle.battle import Battle
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition

TERRAINS = (
    Field.GRASSY_TERRAIN, # Only one we use
    # Field.ELECTRIC_TERRAIN,
    # Field.MISTY_TERRAIN,
    # Field.PSYCHIC_TERRAIN,
)

HAZARDS = (
    SideCondition.STEALTH_ROCK, # Only one we use
    # SideCondition.SPIKES,
    # SideCondition.TOXIC_SPIKES,
    # SideCondition.STICKY_WEB,
)

class FieldEncoder:
    ENCODED_FIELD_DIM = len(TERRAINS) + len(HAZARDS)*2
    def __init__(self):
        pass
    @staticmethod
    def encode_field(battle: Battle) -> NDArray[np.float32]:
        """
        Encodes the field effects of a battle.

        :param battle: The battle to encode.
        :return: A numpy array of shape (ENCODED_FIELD_DIM,) containing the encoded features.
        """
        return np.concatenate([
            FieldEncoder._encode_terrain(battle),
            FieldEncoder._encode_hazards(battle.side_conditions),
            FieldEncoder._encode_hazards(battle.opponent_side_conditions),
        ])

    @staticmethod
    def _encode_terrain(battle: Battle) -> NDArray[np.float32]:
        return np.array([1.0 if battle.fields.get(terrain, None) is not None else 0.0 for terrain in TERRAINS], dtype=np.float32)

    @staticmethod
    def _encode_hazards(side_conditions: Dict[SideCondition, int]) -> NDArray[np.float32]:
        return np.array([1.0 if side_conditions.get(hazard, 0) else 0.0 for hazard in HAZARDS], dtype=np.float32)