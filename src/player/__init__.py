from poke_env.battle import AbstractBattle
from poke_env.player import Player
from sb3_contrib.ppo_mask import MaskablePPO

from encoder import Encoder
from environment.Gen9Env import Gen9Env
from environment.utils import action_masker
from teams import TEAMS


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
        action, _ = self.model.predict(obs, action_masks=action_masks)
        try:
            return Gen9Env.action_to_order(action, battle)
        except ValueError as e:
            print(f"Invalid action {action}: {e}. Trying next best move.")
            action_masks[action] = 0
            try:
                action, _ = self.model.predict(obs, action_masks=action_masks)
                return Gen9Env.action_to_order(action, battle)
            except ValueError as e:
                print(f"Invalid action {action}: {e}. Defaulting to random move.")
                return self.choose_random_move(battle)
