from gymnasium.spaces import Box
import numpy as np
import numpy.typing as npt

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.baselines import RandomPlayer
from poke_env.player.player import Player
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO

from encoder import Encoder
from environment.Gen9Env import Gen9Env
from environment.utils import action_masker
from teams import TEAMS


class PokemonEnv(Gen9Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def create_single_agent_env(
        cls, opponent: Player | None = None
    ) -> SingleAgentWrapper:
        env = cls(
            log_level=25,
            open_timeout=None,
            strict=False,
            team=TEAMS[0],
        )
        # Opponent doesn't need to connect to server - it just provides move choices
        opponent = opponent or RandomPlayer(start_listening=False)
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
        reward_buffer is the total return thus far
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

    def action_masks(self) -> np.ndarray:
        return action_masker(self.battle1, self.action_spaces[self.agents[0]])


if __name__ == "__main__":
    gym_env = PokemonEnv.create_single_agent_env()
    vec_env = DummyVecEnv([lambda: gym_env])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
    )
    model.learn(total_timesteps=10)
    model.save("sb3_showdown_ppo_single_agent")
    # Ensure proper cleanup of connections
    gym_env.close()
