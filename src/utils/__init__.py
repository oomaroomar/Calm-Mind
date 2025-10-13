from poke_env.ps_client import AccountConfiguration
import time


def gen_acc_config(name: str):
    """Generate an account configuration with a timestamp appended to the name."""
    digits = 18 - len(name) - 1
    timestamp = str(int(time.time()))[-digits:]
    return AccountConfiguration(f"{name}_{timestamp}", None)
