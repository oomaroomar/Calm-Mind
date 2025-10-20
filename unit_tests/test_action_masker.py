from gymnasium.spaces import Discrete
from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.teambuilder.teambuilder import Teambuilder
import numpy as np

from environment.utils import action_masker
from teams import TEAMS


def test_action_masker():
    """
    Test that action_masker correctly creates action masks for a battle.
    This test creates a battle, sets up two players with pokemon,
    chooses starting pokemon for both players, and runs action_masker.
    """
    # Create battle
    battle = Battle("tag", "test_player", None, gen=9)

    # Parse the first team from TEAMS
    tb_mons = Teambuilder.parse_showdown_team(TEAMS[0])

    # Create pokemon from the parsed team
    team_pokemon = [Pokemon(gen=9, teambuilder=mon) for mon in tb_mons]

    # Set the first pokemon as active
    team_pokemon[1]._active = True

    # Add pokemon to the battle's team
    battle._team = {f"p1: {mon.species}": mon for mon in team_pokemon}

    # Set up opponent's active pokemon
    opponent_charizard = Pokemon(species="charizard", gen=9)
    opponent_charizard._active = True
    battle._opponent_team = {"p2: charizard": opponent_charizard}

    # Set up available moves from the active pokemon (first 4 moves)
    active_pokemon = team_pokemon[1]
    battle._available_moves = list(active_pokemon.moves.values())[:4]

    # Set up available switches (all pokemon except the active one)
    battle._available_switches = [team_pokemon[0]] + team_pokemon[2:]

    # Set battle state flags
    battle._can_tera = True
    battle._force_switch = False

    # Verify the battle state
    assert battle.active_pokemon is not None, "Player should have an active pokemon"
    assert battle.active_pokemon.species == "alomomola", "Alomomola should be active"
    assert (
        battle.opponent_active_pokemon is not None
    ), "Opponent should have an active pokemon"
    assert (
        battle.opponent_active_pokemon.species == "charizard"
    ), "Opponent Charizard should be active"

    # Verify available moves and switches
    assert len(battle.available_moves) == 4, "Should have 4 available moves"
    assert len(battle.available_switches) == 5, "Should have 5 available switches"

    # Run action_masker
    mask = action_masker(battle)

    # Verify the mask
    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.shape == (14,), "Mask should have 14 elements"

    # Check that switches are masked correctly
    # Indices 0-5 are for switches to team pokemon
    # Blaziken is active (index 0), so it shouldn't be available as a switch
    # The other 5 pokemon (indices 1-5) should be available
    assert mask[0] == 1, "Blaziken should be available as switch"
    assert mask[1] == 0, "Active pokemon (Alomomola) should not be available as switch"
    assert mask[2] == 1, "Third pokemon should be available as switch"
    assert mask[3] == 1, "Fourth pokemon should be available as switch"
    assert mask[4] == 1, "Fifth pokemon should be available as switch"
    assert mask[5] == 1, "Sixth pokemon should be available as switch"

    # Check that regular moves are masked correctly (indices 6-9)
    # All 4 moves should be available (they have PP and force_switch is False)
    assert mask[6] == 1, "First move should be available"
    assert mask[7] == 1, "Second move should be available"
    assert mask[8] == 1, "Third move should be available"
    assert mask[9] == 1, "Fourth move should be available"

    # Check that tera moves are masked correctly (indices 10-13)
    # Since battle.used_tera is False, tera moves should be available
    assert not battle.used_tera, "Player should not have used tera yet"
    assert mask[10] == 1, "Tera move 1 should be available"
    assert mask[11] == 1, "Tera move 2 should be available"
    assert mask[12] == 1, "Tera move 3 should be available"
    assert mask[13] == 1, "Tera move 4 should be available"


# def test_action_masker_with_used_tera():
#     """
#     Test that action_masker correctly masks tera moves after tera has been used.
#     """
#     battle = Battle("tag", "test_player", None, gen=9)

#     # Set up minimal battle state
#     pikachu = Pokemon(species="pikachu", gen=9)
#     pikachu._active = True
#     charizard = Pokemon(species="charizard", gen=9)

#     battle._team = {
#         "p1: pikachu": pikachu,
#         "p1: charizard": charizard,
#     }

#     opponent_blastoise = Pokemon(species="blastoise", gen=9)
#     opponent_blastoise._active = True
#     battle._opponent_team = {"p2: blastoise": opponent_blastoise}

#     # Set up available moves and switches
#     battle._available_moves = [
#         Move("thunderbolt", gen=9),
#         Move("quickattack", gen=9),
#     ]

#     battle._available_switches = [charizard]

#     # Simulate that tera has been used
#     battle._used_tera = True
#     battle._force_switch = False

#     # Create action space (2 switches + 2 moves + 2 tera moves = 6 actions)
#     action_space = Discrete(6)

#     # Run action_masker
#     mask = action_masker(battle, action_space)

#     # Verify that tera moves are NOT available after tera has been used
#     assert battle.used_tera, "Battle should show tera has been used"
#     assert mask[4] == 0, "Tera moves should not be available after using tera"
#     assert mask[5] == 0, "Tera moves should not be available after using tera"

#     # Regular moves should still be available
#     assert mask[2] == 1, "Regular moves should still be available"
#     assert mask[3] == 1, "Regular moves should still be available"


# def test_action_masker_with_force_switch():
#     """
#     Test that action_masker correctly masks moves when forced to switch.
#     """
#     battle = Battle("tag", "test_player", None, gen=9)

#     # Set up minimal battle state with a fainted active pokemon
#     pikachu = Pokemon(species="pikachu", gen=9)
#     pikachu._active = True
#     pikachu._current_hp = 0  # Fainted

#     charizard = Pokemon(species="charizard", gen=9)

#     battle._team = {
#         "p1: pikachu": pikachu,
#         "p1: charizard": charizard,
#     }

#     opponent_blastoise = Pokemon(species="blastoise", gen=9)
#     opponent_blastoise._active = True
#     battle._opponent_team = {"p2: blastoise": opponent_blastoise}

#     # When forced to switch, there are no available moves
#     battle._available_moves = []
#     battle._available_switches = [charizard]
#     battle._force_switch = True

#     # Create action space (2 switches + 1 move + 1 tera move = 4 actions)
#     action_space = Discrete(4)

#     # Run action_masker
#     mask = action_masker(battle, action_space)

#     # When forced to switch, moves should not be available
#     assert battle.force_switch, "Battle should be in force switch state"
#     assert mask[2] == 0, "Moves should not be available when forced to switch"
#     assert mask[3] == 0, "Tera moves should not be available when forced to switch"

#     # Only switches should be available
#     assert mask[1] == 1, "Switches should be available when forced to switch"


def test_action_masker_with_no_pp():
    """
    Test that action_masker correctly masks moves with no PP remaining.
    """
    battle = Battle("tag", "test_player", None, gen=9)

    # Set up minimal battle state
    pikachu = Pokemon(species="pikachu", gen=9)
    pikachu._active = True
    charizard = Pokemon(species="charizard", gen=9)

    battle._team = {
        "p1: pikachu": pikachu,
        "p1: charizard": charizard,
    }

    opponent_blastoise = Pokemon(species="blastoise", gen=9)
    opponent_blastoise._active = True
    battle._opponent_team = {"p2: blastoise": opponent_blastoise}

    # Set up moves with one having 0 PP
    move_with_pp = Move("thunderbolt", gen=9)
    move_without_pp = Move("quickattack", gen=9)
    move_without_pp._current_pp = 0  # No PP left

    battle._available_moves = [move_with_pp]
    battle._available_switches = [charizard]
    battle._force_switch = False

    # Create action space (2 switches + 2 moves + 2 tera moves = 6 actions)
    # Run action_masker
    mask = action_masker(battle)

    # Move with PP should be available, move without PP should not
    assert mask[6] == 1, "Move with PP should be available"
    assert mask[7] == 0, "Move without PP should not be available"

    # Tera moves should follow the same pattern (assuming tera isn't used)
    assert mask[10] == 1, "Tera move with PP should be available"
    assert mask[11] == 0, "Tera move without PP should not be available"
