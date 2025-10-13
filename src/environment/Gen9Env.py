from gymnasium.spaces import Discrete
import numpy as np
import numpy.typing as npt

from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player


class Gen9Env(SinglesEnv[npt.NDArray[np.float32]]):
    def __init__(self, *args, **kwargs):
        kwargs["battle_format"] = "gen9ou"
        super().__init__(*args, **kwargs)
        num_switches = 6
        num_moves = 4
        act_size = num_switches + num_moves * 2
        self.action_spaces = {
            agent: Discrete(act_size) for agent in self.possible_agents
        }

    @staticmethod
    def action_to_order(
        action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and terastallize

        :param action: The action to take.
        :param battle: The current battle state
        :param fake: If true, action-order converters will try to avoid returning a default
            output if at all possible, even if the output isn't a legal decision. Defaults
            to False.
        :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder
        """
        try:
            if action == -2:
                return DefaultBattleOrder()
            elif action == -1:
                return ForfeitBattleOrder()
            elif action < 6:
                order = Player.create_order(list(battle.team.values())[action])
            else:
                if battle.active_pokemon is None:
                    raise ValueError(
                        f"Invalid order from player {battle.player_username} "
                        f"in battle {battle.battle_tag} - action specifies a "
                        f"move, but battle.active_pokemon is None!"
                    )
                mvs = (
                    battle.available_moves
                    if len(battle.available_moves) == 1
                    and battle.available_moves[0].id in ["struggle", "recharge"]
                    else list(battle.active_pokemon.moves.values())
                )
                if (action - 6) % 4 not in range(len(mvs)):
                    raise ValueError(
                        f"Invalid action {action} from player {battle.player_username} "
                        f"in battle {battle.battle_tag} - action specifies a move "
                        f"but the move index {(action - 6) % 4} is out of bounds "
                        f"for available moves {mvs}!"
                    )
                order = Player.create_order(
                    mvs[(action - 6) % 4],
                    terastallize=10 <= action.item() < 14,
                )
            if not fake and str(order) not in [str(o) for o in battle.valid_orders]:
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} - converted order {order} "
                    f"not in valid orders {[str(o) for o in battle.valid_orders]}!"
                )
            return order
        except ValueError as e:
            # if strict:
            raise e
            # else:
            #     if battle.logger is not None:
            #         battle.logger.warning(str(e) + "BOOBA Defaulting to random move.")
            #     return Player.choose_random_singles_move(battle)

    @staticmethod
    def order_to_action(
        order: BattleOrder, battle: Battle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        """
        Returns the action relative to the given BattleOrder.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If true, action-order converters will try to avoid returning a default
            output if at all possible, even if the output isn't a legal decision. Defaults
            to False.
        :type fake: bool
        :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
        :type strict: bool

        :return: The action for the given battle order in context of the current battle.
        :rtype: int64
        """
        try:
            if isinstance(order, DefaultBattleOrder):
                action = -2
            elif isinstance(order, ForfeitBattleOrder):
                action = -1
            else:
                assert isinstance(order, SingleBattleOrder)
                assert not isinstance(order.order, str)
                if not fake and str(order) not in [str(o) for o in battle.valid_orders]:
                    raise ValueError(
                        f"Invalid order from player {battle.player_username} "
                        f"in battle {battle.battle_tag} - order {order} "
                        f"not in valid orders {[str(o) for o in battle.valid_orders]}!"
                    )
                if isinstance(order.order, Pokemon):
                    action = [p.base_species for p in battle.team.values()].index(
                        order.order.base_species
                    )
                else:
                    assert battle.active_pokemon is not None
                    mvs = (
                        battle.available_moves
                        if len(battle.available_moves) == 1
                        and battle.available_moves[0].id in ["struggle", "recharge"]
                        else list(battle.active_pokemon.moves.values())
                    )
                    action = [m.id for m in mvs].index(order.order.id)
                    gimmick = 1 if order.terastallize else 0
                    action = 6 + action + 4 * gimmick
            return np.int64(action)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                return SinglesEnv.order_to_action(
                    Player.choose_random_singles_move(battle), battle, fake, strict
                )
