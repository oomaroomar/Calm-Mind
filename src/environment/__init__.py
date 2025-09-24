from typing import Any, Dict
from gymnasium.spaces import Box, Discrete
import numpy as np
import numpy.typing as npt

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.baselines import RandomPlayer
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from encoder import Encoder
from teams import TEAMS


class PokemonEnv(SinglesEnv[npt.NDArray[np.float32]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_switches = 6
        num_moves = 4
        act_size = num_switches + num_moves * 2
        self.action_spaces = {
            agent: Discrete(act_size) for agent in self.possible_agents
        }
        # This is our big 5k-dim vector
        print(
            f"Initializing Pokemon battle environment with {Encoder.BATTLE_FEATURES_DIM}-dimensional feature vector."
        )
        self.observation_spaces = {
            agent: Box(
                -np.inf, np.inf, shape=(Encoder.BATTLE_FEATURES_DIM,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        env = cls(
            account_configuration1=AccountConfiguration("RL agent", None),
            battle_format=config["battle_format"],
            log_level=25,
            open_timeout=None,
            strict=False,
            team=TEAMS[0],
        )
        opponent = RandomPlayer(
            battle_format=config["battle_format"],
            team=TEAMS[0],
            account_configuration=AccountConfiguration("Random agent", None),
        )
        return SingleAgentWrapper(env, opponent)

    def embed_battle(self, battle: AbstractBattle):
        return Encoder.embed_battle(battle)

    def calc_reward(
        self,
        battle: AbstractBattle,
        starting_value: float = 0.0,
        hp_value: float = 1.0,
        fainted_value: float = 1.0,
        status_value: float = 0.2,
        number_of_pokemons: int = 6,
        victory_value: float = 30.0,
    ) -> float:
        """
        reward_buffer is the total return thus far (idk why its named that)
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0.0

        ### REWARD FOR STAT CHANGES ###
        active_mon = battle.active_pokemon
        stat_diff = 0
        if active_mon is not None:
            stat_diff = 0.03 * sum(active_mon.boosts.values())
        if battle.opponent_active_pokemon is not None:
            stat_diff -= 0.03 * sum(battle.opponent_active_pokemon.boosts.values())
        current_value += stat_diff

        ### REWARD FOR TERA ###
        tera_diff = 0.2 if not battle.used_tera else 0.0
        tera_diff -= 0.2 if not battle.opponent_used_tera else 0.0
        current_value += tera_diff

        ### REWARD FOR KO, DAMAGE, AND STATUS ###
        ### MY TEAM
        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value
        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        ### OPPONENT TEAM
        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value
        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        ###  REWARD FOR WINNING OR LOSING ###
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        reward = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return reward

    def valid_action_mask(self) -> np.ndarray:
        battle = self.battle1
        mask = np.zeros(self.action_spaces[self.agents[0]].n)

        # switches
        indices = [
            i
            for i, pokemon in enumerate(battle.team.values())
            if pokemon in battle.available_switches
        ]
        mask[indices] = 1

        # moves
        indices = [
            i + 6
            for i, move in enumerate(battle.available_moves)
            if move.current_pp > 0
        ]
        if not battle.used_tera:
            indices += [i + 4 for i in indices]

        mask[indices] = 1
        return mask

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


def mask_fn(saw: SingleAgentWrapper) -> np.ndarray:
    return saw.env.valid_action_mask()


def single_agent_train(total_timesteps: int = 100000):
    """Train a single agent using Stable Baselines 3 PPO."""

    def make_env():
        return PokemonEnv.create_single_agent_env({"battle_format": "gen9ou"})

    env = make_env()
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("sb3_showdown_ppo_single_agent")

    return model


if __name__ == "__main__":
    # Train single agent with SB3
    model = single_agent_train(total_timesteps=10**5)
    print("Training completed! Model saved as 'sb3_showdown_ppo_single_agent'")
