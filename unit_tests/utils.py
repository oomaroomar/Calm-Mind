from typing import Tuple
from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from teams import TEAMS


def create_battle(team=TEAMS[0]) -> Tuple[Battle, Pokemon]:
    battle = Battle("tag", "test_player", None, gen=9)

    # Parse the first team from TEAMS
    tb_mons = Teambuilder.parse_showdown_team(team)

    # Create pokemon from the parsed team
    team_pokemon = [Pokemon(gen=9, teambuilder=mon) for mon in tb_mons]
    opponent_pokemon = [
        Pokemon(gen=9, teambuilder=mon) for mon in tb_mons
    ]  # create separate pokemon objects for opponent

    # Add pokemon to the battle's teams
    battle._team = {f"p1: {mon.species}": mon for mon in team_pokemon}
    battle._opponent_team = {f"p2: {mon.species}": mon for mon in opponent_pokemon}

    active_pokemon = team_pokemon[0]
    # Set the first pokemon as active
    active_pokemon._active = True
    opponent_pokemon[0]._active = True

    # Set available actions
    battle._available_switches = team_pokemon[1:]
    battle._available_moves = list(active_pokemon.moves.values())[:4]

    # Set battle state flags
    battle._can_tera = True
    battle._force_switch = False

    return battle, active_pokemon
