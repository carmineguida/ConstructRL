"""Centralized configuration for Construct RL.

Loads defaults from ``config.json`` next to this file (or the working directory).
Values can be overridden at runtime via the handshake from JavaScript.
"""

import json
import os

_DEFAULTS = {
    # --- Logging / diagnostics ---
    "verbose": True,
    "slow_ms": 50,

    # --- WebSocket server ---
    "ws_host": "localhost",
    "ws_port": 8080,
    "ws_slow_recv_ms": 10,

    # --- Environment defaults ---
    "max_steps": 1000,
    "total_timesteps": 100000,
    "num_agents": 1,
    "mode": "train",

    # --- Timeouts (seconds) ---
    "send_timeout": 0.5,
    "response_timeout": 1.0,

    # --- Reward thresholds ---
    "timeout_penalty": -1,
  
    # --- Training ---
    "rl_algorithm": "PPO",
    "learning_rate": -1, # use default
    "gamma": -1, # use default

    # -- MLP Network Settings
    "activation":"",
    "net_actor":[64, 64],
    "net_critic":[64, 64],

    # -- Logging --
    "tensorboard_log":"./tensorboard_log/",
    "train_verbose": 0,

    # --- Polling intervals (seconds) ---
    "poll_interval": 0.01,
    "post_num_agents_delay": 0.2,
}


def _find_config_file():
    """Look for config.json next to this module, then in the cwd."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(pkg_dir, "config.json")
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(os.getcwd(), "config.json")
    if os.path.isfile(candidate):
        return candidate
    return None


def load_config(path=None):
    """Return a dict of config values, merging file overrides onto defaults."""
    cfg = dict(_DEFAULTS)
    config_path = path or _find_config_file()
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        cfg.update(overrides)
        print(f"Config loaded from {config_path}")
    return cfg


# Module-level config dict — imported by other modules.
config = load_config()
#print(config)
