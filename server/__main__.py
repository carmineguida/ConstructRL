"""Entry point for the Construct RL bridge.

Run with:  python -m construct_rl
"""

import torch as th

import time
import threading
import tkinter as tk
from tkinter import filedialog

from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
from stable_baselines3.common.env_checker import check_env

from .config import config
from .diagnostics import dlog
from .hub import hub
from .env import CustomEnv
from .vec_env import BatchedConstructVecEnv
from .server import run_websocket_server
from .custom_logger import SummaryWriterCallback

def main():
    print('=' * 60)
    print('Initializing Custom RL Environment')
    print('=' * 60)

    print('\nStarting WebSocket server thread...')
    server_thread = threading.Thread(target=run_websocket_server, daemon=True)
    server_thread.start()

    print(f"Waiting for WebSocket server to start on ws://{config['ws_host']}:{config['ws_port']}...")
    while not server_thread.is_alive():
        time.sleep(config["poll_interval"])
    print("WebSocket server is ready")

    print('\n' + '=' * 60)
    print('Ready! Waiting for client to connect...')
    print('Connect your Construct 3 game to begin')
    print('=' * 60 + '\n')

    print('Waiting for WebSocket client connection...')
    while hub.websocket is None:
        time.sleep(config["poll_interval"])
    print('\nClient connected!')

    print('Waiting for handshake with environment configuration...')
    hub.handshake_event.wait()
    if hub.closed.is_set():
        print('Connection lost before handshake completed – exiting.')
        return

    print('Handshake complete!')


    # Tell JS how many agents to prepare (even if 1)
    dlog("MAIN", f"Sending set_num_agents={hub.num_agents}", force=True)
    hub.send({"command": "set_num_agents", "num_agents": hub.num_agents})
    time.sleep(config["post_num_agents_delay"])

    # Run check_env with a single-agent env
    dlog("MAIN", "Running check_env…", force=True)
    tc = time.perf_counter()
    primary_env = CustomEnv(agent_id=0)
    primary_env._apply_handshake()
    check_env(primary_env)
    dlog(
        "MAIN", f"check_env OK {(time.perf_counter()-tc)*1000:.0f}ms", force=True)

    num_agents = hub.num_agents
    print(f'\nCreating batched environment with {num_agents} agent(s)...')

    vec_env = BatchedConstructVecEnv(
        num_envs=num_agents,
        observation_space=hub.observation_space,
        action_space=hub.action_space,
        max_steps=hub.max_steps,
    )
    print(f'Batched environment created with {num_agents} agent(s)')
    print('observation_space:', vec_env.observation_space)
    print('action_space:', vec_env.action_space)

    total_ts = hub.total_timesteps
    dlog("MAIN", f"total_timesteps target = {total_ts:,}", force=True)

    if hub.mode == "inference":
        _run_inference(vec_env, total_ts)
    else:
        _run_training(vec_env, total_ts)


def _run_inference(vec_env, total_ts):
    """Load a saved model and run inference."""
    print('\n' + '=' * 60)
    print('INFERENCE MODE – select a trained model to load')
    print('=' * 60 + '\n')

    tk_root = tk.Tk()
    tk_root.lift()
    tk_root.attributes('-topmost', True)
    tk_root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select a trained model",
        filetypes=[("Stable-Baselines3 model", "*.zip"), ("All files", "*.*")],
    )
    tk_root.attributes('-topmost', False)
    tk_root.destroy()

    if not model_path:
        print("No model selected – exiting.")
        return

    try:
        if config["rl_algorithm"] == "A2C":
            model = A2C.load(model_path, env=vec_env)
        elif config["rl_algorithm"] == "SAC":
            model = SAC.load(model_path, env=vec_env)
        elif config["rl_algorithm"] == "DDPG":
            model = DDPG.load(model_path, env=vec_env)
        elif config["rl_algorithm"] == "TD3":
            model = TD3.load(model_path, env=vec_env)
        else:
            model = PPO.load(model_path, env=vec_env)
            
        if config["learning_rate"] > 0: model.learning_rate = config["learning_rate"]
        if config["gamma"] > 0: model.gamma = config["gamma"]
        print(f'Model loaded from "{model_path}"')
    except Exception as e:
        print(f'ERROR: Could not load model from "{model_path}": {e}')
        return

    dlog("MAIN", "Starting inference loop…", force=True)
    obs = vec_env.reset()
    try:
        while not hub.closed.is_set():
            #actions, _states = dict(obs, deterministic=True)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user")
    except Exception as e:
        if hub.closed.is_set():
            print("\n\nClient disconnected – stopping inference.")
        else:
            print(f"\n\nError during inference: {e}")
            import traceback
            traceback.print_exc()
    finally:
        vec_env.close()
        _print_stats("Inference", vec_env)


def _run_training(vec_env, total_ts):
    """Train a new model and save it."""
    print('\nStarting training...\n')
    time.sleep(0.1)
    try:
        dlog("MAIN", "model.learn() starting", force=True)
        t_train = time.perf_counter()
        policy = "MlpPolicy"
        if hub.obs_type == "image": policy = "CnnPolicy"
        if hub.obs_type == "dictionary": policy = "MultiInputPolicy"


        policy_kwargs = None
        if policy == "MlpPolicy" and config["activation"] != "":
            activation_fn = th.nn.ReLU
            config["activation"] = config["activation"].lower()
            if config["activation"] == "tanh": activation_fn = th.nn.Tanh;
            if config["activation"] == "leakyrelu": activation_fn = th.nn.LeakyReLU;
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=dict(pi=config["net_actor"], vf=config["net_critic"]))

        callback = SummaryWriterCallback(vec_env)
        
        if config["rl_algorithm"] == "A2C":
            model = A2C(policy, vec_env, policy_kwargs=policy_kwargs, verbose=config["train_verbose"], tensorboard_log=config["tensorboard_log"])
        elif config["rl_algorithm"] == "SAC":
            model = SAC(policy, vec_env, policy_kwargs=policy_kwargs, verbose=config["train_verbose"], tensorboard_log=config["tensorboard_log"])
        elif config["rl_algorithm"] == "DDPG":
            model = DDPG(policy, vec_env, policy_kwargs=policy_kwargs, verbose=config["train_verbose"], tensorboard_log=config["tensorboard_log"])
        elif config["rl_algorithm"] == "TD3":
            model = TD3(policy, vec_env, policy_kwargs=policy_kwargs, verbose=config["train_verbose"], tensorboard_log=config["tensorboard_log"])
        else:
            model = PPO(policy, vec_env, policy_kwargs=policy_kwargs, verbose=config["train_verbose"], tensorboard_log=config["tensorboard_log"])

        if config["learning_rate"] > 0: model.learning_rate = config["learning_rate"]
        if config["gamma"] > 0: model.gamma = config["gamma"]

        print('model.learning_rate:', model.learning_rate)
        print('model.gamma:', model.gamma)
        #print('model.policy:', model.policy)

        model.learn(total_timesteps=total_ts, callback=callback)
        print(f'\nTraining complete! ({total_ts:,} timesteps in {time.perf_counter()-t_train:.1f}s)')
        dlog(
            "MAIN", f"Training complete in {time.perf_counter()-t_train:.1f}s", force=True)

        tk_root = tk.Tk()
        tk_root.lift()
        tk_root.attributes('-topmost', True)
        tk_root.withdraw()
        save_path = filedialog.asksaveasfilename(
            title="Save trained model",
            defaultextension=".zip",
            filetypes=[("Stable-Baselines3 model", "*.zip"), ("All files", "*.*")],
            initialfile="model_construct_rl.zip",
        )
        tk_root.attributes('-topmost', False)
        tk_root.destroy()

        if save_path:
            if save_path.endswith(".zip"):
                save_path = save_path[:-4]
            model.save(save_path)
            print(f'Model saved to "{save_path}.zip"')
        else:
            print("No save location selected – model not saved.")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        if hub.closed.is_set():
            print("\n\nClient disconnected – stopping training.")
        else:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()
    finally:
        vec_env.close()
        _print_stats("Training", vec_env)


def _print_stats(label, vec_env):
    """Print attempt statistics for all agents."""
    print("\n" + "=" * 60)
    print(f"{label} Statistics:")
    print("=" * 60)
    for i in range(vec_env.num_envs):
        print(
            f'  Agent {i}: attempts={int(vec_env.attempts[i])}, '
            f'timeouts={int(vec_env.timeouts[i])}')
    print("=" * 60)


if __name__ == "__main__":
    main()
