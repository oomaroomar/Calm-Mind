import argparse
import asyncio
import random
from typing import Tuple
import numpy as np
import json
import time
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
from stable_baselines3.common.vec_env import DummyVecEnv
from encoder import Encoder
from environment import PokemonEnv
from environment.utils import action_masker
from teams import TEAMS


### Parameters

STEPS_PER_ITER = 10**5
TOTAL_TIMESTEPS = STEPS_PER_ITER * 47
AGENT_CHECKPOINT = "sb3_showdown_ppo_single_agent"


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
        # Check if battle is finished or if we can't make moves
        if battle.finished:
            return self.choose_default_move()

        # Additional safety check: if there are no valid orders, return default
        if not battle.valid_orders or (
            len(battle.valid_orders) == 1
            and str(battle.valid_orders[0]) == "/choose default"
        ):
            return self.choose_default_move()

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
                return self.choose_random_move(battle)


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


def single_agent_train(total_timesteps: int = 100000):
    """Train a single agent using Stable Baselines 3 PPO."""
    model, _ = create_model()

    model.learn(total_timesteps=total_timesteps)
    model.save("sb3_showdown_ppo_single_agent")

    return model


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
heuristic_non_listening_player = SimpleHeuristicsPlayer(
    battle_format="gen9ou",
    team=TEAMS[0],
    start_listening=False,
)
mbp_non_listening_player = MaxBasePowerPlayer(
    battle_format="gen9ou",
    team=TEAMS[0],
    start_listening=False,
)


async def evaluate_agent(model_path: str, iteration: int = None, timesteps: int = None):
    # Added sleep timer because my computer is overheating
    rl_agent = ModelPlayer(
        model_path=model_path,
        account_configuration=AccountConfiguration(f"RL-Agent-it-{iteration}", None),
    )
    players = [rl_agent, random_player, max_base_power_player, simple_heuristics_player]
    x_eval = await cross_evaluate(players, n_challenges=10)
    rl_agent._save_replays = f"eval_results/replays/it-{iteration}"
    await rl_agent.battle_against(random_player, n_battles=1)
    await rl_agent.battle_against(max_base_power_player, n_battles=1)
    await rl_agent.battle_against(simple_heuristics_player, n_battles=1)
    rl_agent._save_replays = False
    table = [["-"] + [p.username for p in players]]
    for p_1, results in x_eval.items():
        table.append([p_1] + [x_eval[p_1][p_2] for p_2 in results])
    print(tabulate(table))

    # Save evaluation results
    if iteration is not None or timesteps is not None:
        save_eval_results(x_eval, iteration)
    del rl_agent
    return x_eval


## Misleading name for now, we are training against a suite of opponents (including self-play)
async def train_selfplay(total_timesteps: int = TOTAL_TIMESTEPS):
    # initial opponent: a simple random player or a saved baseline
    initial_opponent = heuristic_non_listening_player  # poke-env default opponent (or create a scripted Player)
    # create main model
    model, gym_env = create_model(initial_opponent)
    model.load(AGENT_CHECKPOINT)
    total = 0
    iter_no = 0
    total_iterations = total_timesteps / STEPS_PER_ITER
    print(f"Training for {total_timesteps} steps...")
    while total < total_timesteps:
        iter_no += 1
        print(
            f"=== Iter {iter_no}/{total_iterations}: training {STEPS_PER_ITER} steps against opponent ==="
        )
        model.learn(total_timesteps=STEPS_PER_ITER)
        total += STEPS_PER_ITER

        model.save(AGENT_CHECKPOINT)

        r = random.random()
        opponent_player = None
        if iter_no > 10**2 and r < 0.33:
            opponent_player = ModelPlayer(
                model_path=AGENT_CHECKPOINT,
                start_listening=False,
            )
        elif r < 0.66:
            opponent_player = mbp_non_listening_player
        else:
            opponent_player = heuristic_non_listening_player
        gym_env.update_opponent(opponent_player)
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
    args = parser.parse_args()

    if args.visualize:
        print("Visualizing results...")
        import subprocess

        subprocess.run(["python", "visualize_results.py"])
    elif args.train_selfplay:
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
