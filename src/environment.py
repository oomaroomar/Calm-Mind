from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import PokeEnv, SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer


class Gen9RandomBattleEnv(PokeEnv):
    LOW = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    HIGH = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(
                np.array(self.LOW, dtype=np.float32),
                np.array(self.HIGH, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        act_size = 4 * 2 + 6
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
        :type action: int64
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If true, action-order converters will try to avoid returning a default
            output if at all possible, even if the output isn't a legal decision. Defaults
            to False.
        :type fake: bool
        :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
        :type strict: bool

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder
        """
        try:
            match action:
                case -2:
                    return DefaultBattleOrder()
                case -1:
                    return ForfeitBattleOrder()
                case _ if action < 6:
                    order = Player.create_order(list(battle.team.values())[action])
                    if not fake:
                        assert not battle.trapped, "invalid action"
                        assert isinstance(order.order, Pokemon)
                        assert order.order.base_species in [
                            p.base_species for p in battle.available_switches
                        ], "invalid action"
                case _:
                    if not fake:
                        assert not battle.force_switch, "invalid action"
                        assert battle.active_pokemon is not None, "invalid action"
                    elif battle.active_pokemon is None:
                        return DefaultBattleOrder()
                    mvs = (
                        battle.available_moves
                        if len(battle.available_moves) == 1
                        and battle.available_moves[0].id in ["struggle", "recharge"]
                        else list(battle.active_pokemon.moves.values())
                    )
                    if not fake:
                        assert (action - 6) % 4 in range(len(mvs)), "invalid action"
                    elif (action - 6) % 4 not in range(len(mvs)):
                        return DefaultBattleOrder()
                    order = Player.create_order(
                        mvs[(action - 6) % 4],
                        terastallize=10 <= action.item() < 14,
                    )
                    if not fake:
                        assert isinstance(order.order, Move)
                        assert order.order.id in [
                            m.id for m in battle.available_moves
                        ], "invalid action"
                        assert (
                            not order.terastallize or battle.can_tera
                        ), "invalid action"
            return order
        except AssertionError as e:
            if not strict and str(e) == "invalid action":
                return DefaultBattleOrder()
            else:
                raise e

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
            match order:
                case DefaultBattleOrder():
                    action = -2
                case ForfeitBattleOrder():
                    action = -1
                case SingleBattleOrder():
                    assert not isinstance(order.order, str)
                    if isinstance(order.order, Pokemon):
                        if not fake:
                            assert not battle.trapped, "invalid order"
                            assert order.order.base_species in [
                                p.base_species for p in battle.available_switches
                            ], "invalid order"
                        action = [p.base_species for p in battle.team.values()].index(
                            order.order.base_species
                        )
                    else:
                        if not fake:
                            assert not battle.force_switch, "invalid order"
                            assert battle.active_pokemon is not None, "invalid order"
                        elif battle.active_pokemon is None:
                            return np.int64(-2)
                        mvs = (
                            battle.available_moves
                            if len(battle.available_moves) == 1
                            and battle.available_moves[0].id in ["struggle", "recharge"]
                            else list(battle.active_pokemon.moves.values())
                        )
                        if not fake:
                            assert order.order.id in [
                                m.id for m in mvs
                            ], "invalid order"
                        action = [m.id for m in mvs].index(order.order.id)
                        gimmick = 1 if order.terastallize else 0
                        action = 6 + action + 4 * gimmick
                        if not fake:
                            assert order.order.id in [
                                m.id for m in battle.available_moves
                            ], "invalid order"
                            assert (
                                not order.terastallize or battle.can_tera
                            ), "invalid order"
            return np.int64(action)
        except AssertionError as e:
            if not strict and str(e) == "invalid order":
                return np.int64(-2)
            else:
                raise e

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        env = cls(
            battle_format=config["battle_format"],
            log_level=25,
            open_timeout=None,
            strict=False,
        )
        opponent = RandomPlayer()
        return SingleAgentWrapper(env, opponent)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)


def single_agent_train(total_timesteps: int = 100000):
    """Train a single agent using Stable Baselines 3 PPO."""

    def make_env():
        return Gen9RandomBattleEnv.create_single_agent_env(
            {"battle_format": "gen9randombattle"}
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("sb3_showdown_ppo_single_agent")

    return model


# Note: Multi-agent training in SB3 requires different approach than Ray RLLib
# For multi-agent Pokemon battles, consider using PettingZoo environments
# or self-play techniques with alternating training
def multi_agent_train():
    """
    Multi-agent training not directly supported in this SB3 conversion.
    Consider using:
    1. Self-play with single agent
    2. PettingZoo environment wrapper
    3. Alternating training between agents
    """
    raise NotImplementedError("Multi-agent training requires different approach in SB3")


if __name__ == "__main__":
    # Train single agent with SB3
    model = single_agent_train(total_timesteps=10**2)
    print("Training completed! Model saved as 'sb3_showdown_ppo_single_agent'")

    # Multi-agent training commented out - requires different approach in SB3
    # multi_agent_train()
