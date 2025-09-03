from poke_env.battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer

# Define a simple environment player
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def compute_reward(self, battle: AbstractBattle) -> float:
        # Example: reward is current HP fraction of active Pokémon
        return battle.active_pokemon.current_hp_fraction

# Instantiate environment
env_player = SimpleRLPlayer(battle_format="gen8randombattle")

# Turn it into a gymnasium environment
env = env_player.get_battle_env()

# Now we can use the standard RL loop
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()   # take random action
    obs, reward, done, truncated, info = env.step(action)
    print("Reward:", reward)
