import numpy as np
from unittest.mock import MagicMock
from encoder import Encoder
from teams import TEAMS
from poke_env.battle import Battle, Pokemon
from poke_env.battle.stat import MOVABLE_STATS, STATS
from poke_env.teambuilder.teambuilder import Teambuilder
from encoder.field_encoder import FieldEncoder
from encoder.move_encoder import (
    CORE_FEATURES_DIM,
    ENCODED_MOVE_DIM,
    MOVE_CATEGORIES,
    SECONDARY_EFFECTS_DIM,
    SPECIAL_CASES_DIM,
    VOLATILE_STATUSES,
    MoveEncoder,
)
from poke_env.battle.status import STATUSES
from poke_env.battle.move import Move
from poke_env.battle.field import Field

from encoder.pokemon_encoder import PokemonEncoder

##############################
### TESTS FOR MOVE ENCODER ###
##############################


def test_status_prob_dim():
    assert MoveEncoder._status_prob(Move("tackle", gen=9)).shape == (len(STATUSES),)


def test_volatile_status_prob_dim():
    assert MoveEncoder._volatile_status_prob(Move("tackle", gen=9)).shape == (
        len(VOLATILE_STATUSES),
    )


def test_encode_category_dim():
    assert MoveEncoder._encode_category(Move("tackle", gen=9)).shape == (
        len(MOVE_CATEGORIES),
    )


def test_encode_core_features():
    assert MoveEncoder._encode_core_features(Move("tackle", gen=9)).shape == (
        CORE_FEATURES_DIM,
    )


def test_encode_self_stat_change_dim():
    assert MoveEncoder._encode_self_stat_change(Move("tackle", gen=9)).shape == (
        len(MOVABLE_STATS),
    )


def test_encode_foe_stat_change_dim():
    assert MoveEncoder._encode_foe_stat_change(Move("tackle", gen=9)).shape == (
        len(MOVABLE_STATS),
    )


def test_encode_secondary_effects_dim():
    assert MoveEncoder._encode_secondary_effects(Move("tackle", gen=9)).shape == (
        SECONDARY_EFFECTS_DIM,
    )


def test_encode_special_cases_dim():
    assert MoveEncoder._encode_special_cases(Move("tackle", gen=9)).shape == (
        SPECIAL_CASES_DIM,
    )


def test_encode_move_dim():
    assert MoveEncoder.encode_move(Move("tackle", gen=9)).shape == (ENCODED_MOVE_DIM,)


##############################
## TESTS FOR FIELD ENCODER ###
##############################


def test_encode_field():
    logger = MagicMock()
    battle = Battle("tag", "username", logger, gen=9)
    assert FieldEncoder.encode_field(battle).shape == (FieldEncoder.ENCODED_FIELD_DIM,)
    battle.parse_message(["", "-fieldstart", "Grassy terrain"])
    assert battle.fields == {Field.GRASSY_TERRAIN: 0}
    assert FieldEncoder.encode_field(battle).shape == (FieldEncoder.ENCODED_FIELD_DIM,)
    assert FieldEncoder.encode_field(battle)[0] == 1.0


##############################
## TESTS FOR POKEMON ENCODER #
##############################


def test_encode_pokemon():
    tb_mons = Teambuilder.parse_showdown_team(TEAMS[0])
    mon = Pokemon(9, teambuilder=tb_mons[0])
    assert PokemonEncoder.opponent_pokemon_encoder(mon).shape == (
        PokemonEncoder.OP_POKEMON_FEATURES_DIM,
    )
    assert PokemonEncoder.my_pokemon_encoder(mon).shape == (
        PokemonEncoder.MY_POKEMON_FEATURES_DIM,
    )


##############################
## TESTS FOR BATTLE ENCODER ##
##############################


def test_encoder():
    logger = MagicMock()
    battle = Battle("tag", "username", logger, gen=9)
    tb_mons = Teambuilder.parse_showdown_team(TEAMS[0])
    team = {f"p1: {mon.nickname}": Pokemon(9, teambuilder=mon) for mon in tb_mons}
    op_team = {f"p2: {mon.nickname}": Pokemon(9, teambuilder=mon) for mon in tb_mons}
    battle.team = team
    assert Encoder.encode_own_team(battle).shape == (Encoder.MY_TEAM_FEATURES_DIM,)
    battle.opponent_team = op_team
    assert Encoder.encode_opponent_team(battle).shape == (Encoder.OP_TEAM_FEATURES_DIM,)
    assert Encoder.embed_battle(battle).shape == (Encoder.BATTLE_FEATURES_DIM,)
    print(f"My team features dim: {Encoder.MY_TEAM_FEATURES_DIM}")
    print(f"Opponent team features dim: {Encoder.OP_TEAM_FEATURES_DIM}")
    print(
        f"Total features dim: {Encoder.MY_TEAM_FEATURES_DIM + Encoder.OP_TEAM_FEATURES_DIM}"
    )
