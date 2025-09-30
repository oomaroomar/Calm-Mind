import argparse
import asyncio
import numpy as np
from tabulate import tabulate
from os import putenv

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player import cross_evaluate
from poke_env.player.player import Player
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from encoder import Encoder
from environment import PokemonEnv
from teams import TEAMS


### Parameters

STEPS_PER_ITER = 10**4
AGENT_CHECKPOINT = "sb3_showdown_ppo_single_agent"
OPPONENT_CHECKPOINT = "sb3_showdown_ppo_single_agent_opponent"


class ModelPlayer(Player):
    def __init__(self, model_path, **kwargs):
        super().__init__(team=TEAMS[0], **kwargs)
        self.model = MaskablePPO.load(model_path, device="cpu")

    def choose_move(self, battle: AbstractBattle):
        obs = Encoder.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        return PokemonEnv.action_to_order(action, battle)

    def action_masks(self) -> np.ndarray:
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


def create_model():
    gym_env = PokemonEnv.create_single_agent_env({"battle_format": "gen9ou"})
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
    return model, gym_env


def single_agent_train(total_timesteps: int = 100000):
    """Train a single agent using Stable Baselines 3 PPO."""

    model, _ = create_model()

    model.learn(total_timesteps=total_timesteps)
    model.save("sb3_showdown_ppo_single_agent")

    return model


async def evaluate_agent(model_path: str):
    rl_agent = ModelPlayer(model_path=model_path, team=TEAMS[0])
    random_player = RandomPlayer(battle_format="gen9ou", team=TEAMS[0])
    max_base_power_player = MaxBasePowerPlayer(battle_format="gen9ou", team=TEAMS[0])
    simple_heuristics_player = SimpleHeuristicsPlayer(
        battle_format="gen9ou", team=TEAMS[0]
    )
    players = [rl_agent, random_player, max_base_power_player, simple_heuristics_player]
    x_eval = await cross_evaluate(players, n_challenges=100)
    return x_eval


async def train_selfplay(total_timesteps: int = 10**5):
    # initial opponent: a simple random player or a saved baseline
    initial_opponent = None  # poke-env default opponent (or create a scripted Player)
    # create main model
    model, gym_env = create_model()
    total = 0
    iter_no = 0
    while total < total_timesteps:
        iter_no += 1
        print(
            f"=== Iter {iter_no}: training {STEPS_PER_ITER} steps against opponent ==="
        )
        model.learn(total_timesteps=STEPS_PER_ITER)
        total += STEPS_PER_ITER

        model.save(AGENT_CHECKPOINT)

        # reload new opponent player for the env (you may need to recreate envs)
        # opponent_player = ModelPlayer(
        #     model_path=AGENT_CHECKPOINT, name=f"Opponent-{iter_no}"
        # )
        opponent_player = RandomPlayer(battle_format="gen9ou", team=TEAMS[0])
        gym_env.close()
        gym_env = PokemonEnv.create_single_agent_env(
            opponent_player, {"battle_format": "gen9ou"}, iter_no
        )
        vec_env = DummyVecEnv([lambda: gym_env])
        model.set_env(vec_env)

        # Evaluate current agent against a suite of opponents
        # x_eval = await evaluate_agent(AGENT_CHECKPOINT)
        # print(tabulate(x_eval))

    print("Training finished.")
    model.save("final_agent.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_selfplay", action="store_true")
    parser.add_argument("--total_timesteps", type=int, default=10**5)
    args = parser.parse_args()
    putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    if args.train_selfplay:
        print("Training via selfplay...")
        asyncio.run(train_selfplay(total_timesteps=args.total_timesteps))
    else:
        print("Training vs a fixed opponent...")
        model = single_agent_train(total_timesteps=args.total_timesteps)
        print("Training completed! Model saved as 'sb3_showdown_ppo_single_agent'")
