import random
from typing import Any, Awaitable, Dict, Literal, Tuple
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import DefaultBattleOrder
import numpy as np

from environment.Gen9Env import Gen9Env
from player import ModelPlayer
from teams import TEAMS


class MaskedSingleAgentWrapper(SingleAgentWrapper):
    def __init__(
        self, env: Gen9Env, model_path: str | None = "ppo_with_entropy_coef.zip"
    ):
        super().__init__(env)
        self.env = env

        self.heuristic_non_listening_player = SimpleHeuristicsPlayer(
            battle_format="gen9ou",
            team=TEAMS[0],
            start_listening=False,
        )
        self.mbp_non_listening_player = MaxBasePowerPlayer(
            battle_format="gen9ou",
            team=TEAMS[0],
            start_listening=False,
        )
        try:
            self.selfplay_opponent = ModelPlayer(
                model_path=model_path, start_listening=False
            )
        except:
            self.selfplay_opponent = None
        self.opponent = self.heuristic_non_listening_player

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def step(
        self, action: np.int64
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Override step to add safety checks for finished battles."""
        assert self.env.battle2 is not None

        # Safety check: if battle2 is finished, something went wrong
        # This shouldn't happen in normal flow, but prevents crashes
        if self.env.battle2.finished:
            # Return default order if battle is finished
            opp_order = DefaultBattleOrder()
        else:
            opp_order = self.opponent.choose_move(self.env.battle2)
            assert not isinstance(opp_order, Awaitable)

        opp_action = self.env.order_to_action(
            opp_order, self.env.battle2, fake=self.env.fake, strict=self.env.strict
        )
        actions = {
            self.env.agent1.username: action,
            self.env.agent2.username: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )

    def update_selfplay_opponent(
        self, model_path: str | None = "ppo_with_entropy_coef.zip"
    ):
        # Clean up old selfplay opponent before creating new one
        if self.selfplay_opponent is not None:
            try:
                # Clear the model to free memory
                if hasattr(self.selfplay_opponent, "model"):
                    del self.selfplay_opponent.model
                # Delete the player object
                del self.selfplay_opponent
            except Exception as e:
                print(f"Warning: Error cleaning up old selfplay opponent: {e}")

        self.selfplay_opponent = ModelPlayer(
            model_path=model_path, start_listening=False
        )

    def change_opponent(
        self, policy: Literal["model", "mbp", "heuristics"] | None = None
    ):
        match policy:
            case "model":
                self.opponent = self.selfplay_opponent
            case "mbp":
                self.opponent = self.mbp_non_listening_player
            case "heuristics":
                self.opponent = self.heuristic_non_listening_player
            case _:
                r = random.random()
                if r < 0.33:
                    self.opponent = self.selfplay_opponent
                elif r < 0.66:
                    self.opponent = self.mbp_non_listening_player
                else:
                    self.opponent = self.heuristic_non_listening_player
