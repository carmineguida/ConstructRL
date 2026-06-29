"""CustomEnv — single-agent Gymnasium environment backed by the WebSocket hub."""

import time

import gymnasium as gym
import numpy as np

from .config import config
from .diagnostics import dlog, dlog_step
from .hub import hub, dump_hub_state


class CustomEnv(gym.Env):

    def __init__(self, agent_id=0):
        super().__init__()
        self.agent_id = agent_id
        hub._ensure_agent(agent_id)

        # Will be set after handshake
        self.action_space = None
        self.observation_space = None

        self.max_steps = config["max_steps"]

        self.attempt = 0
        self.timeouts = 0
        self.steps = 0
        self.total_reward = 0

    def _apply_handshake(self):
        """Copy space definitions from the hub after handshake completes."""
        self.observation_space = hub.observation_space
        self.action_space = hub.action_space
        self.max_steps = hub.max_steps

    # ---- reset ----
    def reset_map(self):
        self.total_reward = 0
        self.steps = 0
        aid = self.agent_id
        t0 = time.perf_counter()
        dlog("ENV-RST", f"agent {aid} reset_map called")
        if hub.closed.is_set():
            dlog("ENV-RST", f"agent {aid} hub closed – skip")
            return
        if hub.websocket is not None and hub.loop is not None:
            try:
                hub.response_events[aid].clear()
                hub.send({"command": "reset", "agent_id": aid})

                tw = time.perf_counter()
                if not hub.response_events[aid].wait(timeout=config["response_timeout"]):
                    dlog(
                        "ENV-RST", f"!! agent {aid} TIMEOUT ({(time.perf_counter()-tw)*1000:.0f}ms)", force=True)
                    dump_hub_state(f"ENV-RST timeout agent {aid}")
                else:
                    ms = (time.perf_counter()-t0)*1000
                    dlog("ENV-RST", f"agent {aid} reset done {ms:.1f}ms")
            except Exception as e:
                dlog("ENV-RST", f"agent {aid} ERROR: {e}", force=True)

    def reset(self, seed=None, options=None):
        if self.attempt > 0:
            print(
                f'[Agent {self.agent_id}] Attempt {self.attempt}  Total Reward: {self.total_reward}')
        super().reset(seed=seed)
        self.attempt += 1
        self.reset_map()
        return self._read_obs(), {}

    # ---- step ----
    def step(self, action):
        self.steps += 1
        truncated = False
        aid = self.agent_id
        t0 = time.perf_counter()

        if hub.closed.is_set():
            dlog("ENV-STEP", f"agent {aid} hub closed")
            return self._read_obs(), 0, True, truncated, {}

        hub.response_events[aid].clear()

        try:
            action_val = int(action) if np.isscalar(action) else action.tolist()
            hub.send({"command": "action", "agent_id": aid, "action": action_val})
        except Exception as e:
            dlog("ENV-STEP", f"agent {aid} send ERROR: {e}", force=True)
            return self._read_obs(), -1, True, truncated, {}

        tw = time.perf_counter()
        if not hub.response_events[aid].wait(timeout=1.0):
            dlog("ENV-STEP", f"!! agent {aid} TIMEOUT wait={(time.perf_counter()-tw)*1000:.0f}ms", force=True)
            dump_hub_state(f"ENV-STEP timeout agent {aid} step#{self.steps}")
            return self._read_obs(), -1, True, truncated, {}

        wait_ms = (time.perf_counter()-tw)*1000
        total_ms = (time.perf_counter()-t0)*1000
        reward = hub.received_rewards[aid]
        done = hub.received_dones[aid]

        if total_ms > config["slow_ms"]:
            dlog("ENV-STEP", f"!! SLOW agent {aid} step#{self.steps} total={total_ms:.1f}ms wait={wait_ms:.1f}ms", force=True)
        else:
            dlog_step("ENV-STEP", f"agent {aid} step#{self.steps} {total_ms:.1f}ms r={reward} d={done}")

        if self.steps >= self.max_steps and not done:
            dlog("ENV-STEP", f"agent {aid} max_steps={self.max_steps} reached", force=True)
            self.timeouts += 1
            done = True
            reward = config["timeout_penalty"]

        self.total_reward += reward
        return self._read_obs(), reward, done, truncated, {}


    def reshape_image(self, data):
        flat_rgba = np.array(data, np.uint8)
        rgba_array = flat_rgba.reshape(hub.image_height, hub.image_width, 4).transpose(2, 0, 1)
        rgb_array = rgba_array[:3, :, :]
        return np.array(rgb_array, dtype=np.uint8)
    
    # ---- observations ----
    def _read_obs(self):
        """Read the last received observation from the hub (no round-trip)."""

        aid = self.agent_id    

        if hub.temp_obs.get(aid) is not None:
            if hub.obs_type == "dictionary":
                image = self.reshape_image(hub.temp_obs[aid]["image"])
                return {
                    "image": np.array(image, dtype=np.uint8),
                    "vector": np.array(hub.temp_obs[aid]["vector"], dtype=np.int64)
                }

            elif hub.obs_type == "image":
                image = self.reshape_image(hub.temp_obs[aid])
                return np.array(image, dtype=np.uint8)
            else:
                return np.array(hub.temp_obs[aid], dtype=np.int64)
            
        
        if hub.obs_type == "dictionary":
            return {
                "image": np.zeros(self.observation_space["image"].shape, dtype=np.uint8),
                "vector": np.zeros(self.observation_space["vector"].shape, dtype=np.int64)
            }
        
        shape = self.observation_space.shape
        if hub.obs_type == "image":
            return np.zeros(shape, dtype=np.uint8)
        else:
            return np.zeros(shape, dtype=np.int64)

    def get_obs(self):
        """Request a fresh observation via round-trip (used by check_env)."""
        aid = self.agent_id
        if hub.closed.is_set():
            return self._read_obs()
        if hub.websocket is not None and hub.loop is not None:
            try:
                hub.response_events[aid].clear()
                hub.send({"command": "observations", "agent_id": aid})

                if hub.response_events[aid].wait(timeout=config["response_timeout"]):
                    hub.response_events[aid].clear()
                else:
                    dlog("GET-OBS", f"!! agent {aid} obs TIMEOUT", force=True)
                    dump_hub_state(f"observations timeout agent {aid}")
            except Exception as e:
                print(f'[Agent {aid}] Error requesting obs: {e}')
        return self._read_obs()
