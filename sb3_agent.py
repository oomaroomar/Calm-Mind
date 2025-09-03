from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper
from poke_env.environment.singles_env import Gen9RandombattleSinglesEnv
from poke_env.player import RandomPlayer


class ExampleEnv(Gen9RandombattleSinglesEnv):
    LOW = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    HIGH = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define observation space per-agent; the single-agent wrapper will expose the right space
        self.observation_spaces = {
            agent: Box(
                np.array(self.LOW, dtype=np.float32),
                np.array(self.HIGH, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        env = cls(
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
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)


def train_single_agent(total_timesteps: int = 10000) -> None:
    def make_env():
        return ExampleEnv.create_single_agent_env({})

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
    model.save("sb3_showdown_ppo")


if __name__ == "__main__":
    train_single_agent()


