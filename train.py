import argparse
import asyncio
import random
from typing import Tuple
import numpy as np
import json
import os.path as osp
from datetime import datetime
from pathlib import Path
from poke_env.ps_client.account_configuration import AccountConfiguration
from tabulate import tabulate

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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from encoder import Encoder
from environment import PokemonEnv
from environment.utils import action_masker
from player import ModelPlayer
from teams import TEAMS


### Parameters

STEPS_PER_ITER = 10**5
TOTAL_TIMESTEPS = (
    STEPS_PER_ITER * 5
)  # While running on my computer, over 5 iterations causes the computer to freeze
AGENT_CHECKPOINT = "ppo_with_entropy_coef.zip"


def test():
    model, vec_env = create_model()
    model.learn(total_timesteps=10)
    model.save("test_model.zip")
    vec_env.close()


def create_model(parallel: bool = False) -> Tuple[MaskablePPO, SubprocVecEnv]:
    """Create a model and a vectorized environment.
    :return: A tuple containing the model and the vectorized environment.
    """

    def make_env():
        return PokemonEnv.create_single_agent_env()

    if parallel:
        vec_env = SubprocVecEnv([make_env for _ in range(6)])
    else:
        vec_env = DummyVecEnv([make_env])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        gamma=0.9,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        ent_coef=0.05,
    )
    return model, vec_env


def save_eval_results(x_eval: dict, iteration: int):
    """Save evaluation results to JSON file for tracking over time."""
    eval_dir = Path("eval_results")
    eval_dir.mkdir(exist_ok=True)
    eval_file = eval_dir / "training_history.json"

    # Load existing results
    if eval_file.exists():
        with open(eval_file, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Extract winrates for the agent
    agent_results = x_eval.get(f"RL-Agent-it-{iteration}", {})

    entry = {
        "iteration": iteration,
        "timesteps": iteration * STEPS_PER_ITER,
        "timestamp": datetime.now().isoformat(),
        "winrates": {
            "Random": agent_results.get("Random", 0),
            "MaxBasePower": agent_results.get("MaxBasePower", 0),
            "SimpleHeuristics": agent_results.get("SimpleHeuristics", 0),
        },
    }

    history.append(entry)

    # Save updated history
    with open(eval_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved evaluation results to {eval_file}")


def evaluate_agent_factory():
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

    async def evaluate_agent(
        model_path: str, iteration: int = None, timesteps: int = None
    ):
        rl_agent = ModelPlayer(
            model_path=model_path,
            account_configuration=AccountConfiguration(
                f"RL-Agent-it-{iteration}", None
            ),
        )
        rl_agent2 = ModelPlayer(
            model_path=model_path,
            account_configuration=AccountConfiguration(
                f"RL-Agent-2-it-{iteration}", None
            ),
        )
        players = [
            rl_agent,
            random_player,
            max_base_power_player,
            simple_heuristics_player,
        ]
        x_eval = await cross_evaluate(players, n_challenges=10)
        rl_agent2._save_replays = f"eval_results/replays/it-{iteration}"
        for player in players:
            await rl_agent2.battle_against(player, n_battles=1)
        rl_agent2._save_replays = False
        table = [["-"] + [p.username for p in players]]
        for p_1, results in x_eval.items():
            table.append([p_1] + [x_eval[p_1][p_2] for p_2 in results])
        print(tabulate(table))

        # Save evaluation results
        if iteration is not None or timesteps is not None:
            save_eval_results(x_eval, iteration)
        del rl_agent, rl_agent2
        return x_eval

    return evaluate_agent


## Misleading name for now, we are training against a suite of opponents (including self-play)
async def train_selfplay(total_timesteps: int = TOTAL_TIMESTEPS):
    # initial opponent: a simple random player or a saved baseline
    # create main model
    model, vec_env = create_model(parallel=True)
    if osp.exists(AGENT_CHECKPOINT):
        model.load(AGENT_CHECKPOINT)
    else:
        print(f"Model {AGENT_CHECKPOINT} not found, starting from scratch")
    total = 0
    iter_no = 0
    if osp.exists(f"eval_results/training_history.json"):
        with open(f"eval_results/training_history.json", "r") as f:
            history = json.load(f)
        iter_no = len(history)
    total_iterations = total_timesteps / STEPS_PER_ITER
    evaluate_agent = evaluate_agent_factory()
    print(f"Training for {total_timesteps} steps...")
    while total < total_timesteps:
        iter_no += 1
        print(
            f"=== Iter {iter_no}/{total_iterations}: training {STEPS_PER_ITER} steps against opponent ==="
        )
        model.learn(total_timesteps=STEPS_PER_ITER)
        total += STEPS_PER_ITER

        model.save(AGENT_CHECKPOINT)

        model.get_env().env_method("update_selfplay_opponent", AGENT_CHECKPOINT)
        model.get_env().env_method("change_opponent", "model")

        # Evaluate current agent against a suite of opponents
        x_eval = await evaluate_agent(
            AGENT_CHECKPOINT, iteration=iter_no, timesteps=total
        )
        print(tabulate(x_eval))

    print("Training finished.")
    model.save("final_agent.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_selfplay", action="store_true", help="Train via self-play"
    )
    parser.add_argument(
        "--test_eval",
        action="store_true",
        help="Test evaluation against baseline opponents",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize training results (requires training history)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Total training timesteps",
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.visualize:
        print("Visualizing results...")
        import subprocess

        subprocess.run(["python", "visualize_results.py"])
    elif args.train_selfplay:
        print("Training via selfplay...")
        asyncio.run(train_selfplay(total_timesteps=args.total_timesteps))
    elif args.test:
        test()
