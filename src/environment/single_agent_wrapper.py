from typing import Any, Awaitable, Dict, Tuple
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.battle_order import DefaultBattleOrder
import numpy as np
from poke_env.player.player import Player
import asyncio
from poke_env.concurrency import POKE_LOOP

from environment.Gen9Env import Gen9Env


class MaskedSingleAgentWrapper(SingleAgentWrapper):
    def __init__(self, env: Gen9Env, opponent: Player):
        super().__init__(env, opponent)
        self.env = env

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

    def update_opponent(self, opponent: Player):
        """Update the opponent player for the next set of battles.

        This should be called between training iterations to update
        the opponent with a new model checkpoint.
        """
        self.opponent = opponent
        # Note: The actual battle reset happens in reset(), so new battles
        # will be created for the new opponent
