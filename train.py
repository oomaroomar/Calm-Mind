import argparse
import asyncio
from typing import Tuple
import numpy as np
from poke_env.ps_client.account_configuration import AccountConfiguration
from tabulate import tabulate

from gymnasium.spaces import Discrete
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
from environment.utils import action_masker
from teams import TEAMS


### Parameters

STEPS_PER_ITER = 10**4
AGENT_CHECKPOINT = "sb3_showdown_ppo_single_agent"
OPPONENT_CHECKPOINT = "sb3_showdown_ppo_single_agent_opponent"


class ModelPlayer(Player):
    def __init__(self, model_path, **kwargs):
        # Create action space matching Gen9Env (before parent init)
        # Load the model BEFORE calling parent init (which might trigger choose_move)
        print(f"Loading model from {model_path}...")
        self.model = MaskablePPO.load(model_path, device="cpu")
        print(f"Model loaded successfully. Model type: {type(self.model)}")
        # Now initialize the parent Player class
        super().__init__(battle_format="gen9ou", team=TEAMS[0], **kwargs)

    def choose_move(self, battle: AbstractBattle):
        obs = Encoder.embed_battle(battle)
        action_masks = action_masker(battle)
        action, _ = self.model.predict(
            obs, deterministic=True, action_masks=action_masks
        )
        try:
            return PokemonEnv.action_to_order(action, battle)
        except ValueError as e:
            print(f"Invalid action {action}: {e}. Trying next best move.")
            action_masks[action] = 0
            try:
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=action_masks
                )
                return PokemonEnv.action_to_order(action, battle)
            except ValueError as e:
                print(f"Invalid action {action}: {e}. Defaulting to random move.")
                return Player.choose_random_move(battle)


def create_model(opponent: Player | None = None) -> Tuple[MaskablePPO, PokemonEnv]:
    """Create a model and a gym environment.
    :param opponent: The opponent player. If None, a random player is used.
    :return: A tuple containing the model and the gym environment.
    """
    gym_env = PokemonEnv.create_single_agent_env(opponent)
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
    rl_agent = ModelPlayer(
        model_path=model_path,
        account_configuration=AccountConfiguration("Agent", None),
    )
    random_player = RandomPlayer(
        battle_format="gen9ou",
        team=TEAMS[0],
        account_configuration=AccountConfiguration("Random", None),
    )
    max_base_power_player = MaxBasePowerPlayer(
        battle_format="gen9ou",
        team=TEAMS[0],
        account_configuration=AccountConfiguration("MaxBasePower", None),
    )
    simple_heuristics_player = SimpleHeuristicsPlayer(
        battle_format="gen9ou",
        team=TEAMS[0],
        account_configuration=AccountConfiguration("SimpleHeuristics", None),
    )
    players = [rl_agent, random_player, max_base_power_player, simple_heuristics_player]
    x_eval = await cross_evaluate(players, n_challenges=10)
    table = [["-"] + [p.username for p in players]]
    for p_1, results in x_eval.items():
        table.append([p_1] + [x_eval[p_1][p_2] for p_2 in results])
    print(tabulate(table))
    return x_eval


async def train_selfplay(total_timesteps: int = 10**5):
    # initial opponent: a simple random player or a saved baseline
    initial_opponent = None  # poke-env default opponent (or create a scripted Player)
    # create main model
    model, gym_env = create_model(initial_opponent)
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
        opponent_player = ModelPlayer(model_path=AGENT_CHECKPOINT)
        gym_env.close()
        gym_env = PokemonEnv.create_single_agent_env(opponent_player)
        vec_env = DummyVecEnv([lambda: gym_env])
        model.set_env(vec_env)

        # Evaluate current agent against a suite of opponents
        x_eval = await evaluate_agent(AGENT_CHECKPOINT)
        print(tabulate(x_eval))

    print("Training finished.")
    model.save("final_agent.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_selfplay", action="store_true")
    parser.add_argument("--test_eval", action="store_true")
    parser.add_argument("--total_timesteps", type=int, default=10**5)
    args = parser.parse_args()
    if args.train_selfplay:
        print("Training via selfplay...")
        asyncio.run(train_selfplay(total_timesteps=args.total_timesteps))
    elif args.test_eval:
        print("Testing evaluation...")
        # agent = ModelPlayer(model_path=AGENT_CHECKPOINT)
        # print(agent.action_masks())
        asyncio.run(evaluate_agent(AGENT_CHECKPOINT))
    else:
        print("Training vs a fixed opponent...")
        model = single_agent_train(total_timesteps=args.total_timesteps)
        print("Training completed! Model saved as 'sb3_showdown_ppo_single_agent'")
