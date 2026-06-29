"""BatchedConstructVecEnv — batched VecEnv that multiplexes all agents over one websocket."""

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from .config import config
from .diagnostics import dlog, dlog_step, _SLOW_MS
from .hub import hub, dump_hub_state


class BatchedConstructVecEnv(VecEnv):
    """Custom VecEnv that batches all agent actions into a single websocket message.

    Instead of per-agent round-trips, each step/reset exchanges ONE message
    containing data for ALL agents simultaneously.
    """

    def __init__(self, num_envs, observation_space, action_space, max_steps=None):
        super().__init__(num_envs, observation_space, action_space)
        self.max_steps = max_steps if max_steps is not None else config["max_steps"]
        self.steps = np.zeros(num_envs, dtype=int)
        self.lastStepReport = 0
        self.attempts = np.zeros(num_envs, dtype=int)
        self.nextEpisode = 0
        self.episode = np.zeros(num_envs, dtype=int)
        self.total_rewards = np.zeros(num_envs, dtype=float)
        self.terminal_rewards = np.zeros(num_envs, dtype=float)
        self.terminal_steps = np.zeros(num_envs, dtype=int)
        self.timeouts = np.zeros(num_envs, dtype=int)
        self._actions = None

    def _empty_obs(self):
        if isinstance(self.observation_space, spaces.Dict):
            return {
                    "image": np.zeros((self.num_envs, *hub.image_observation_space.shape), dtype=hub.image_observation_space.dtype),
                    "vector": np.zeros((self.num_envs, *hub.vector_observation_space.shape), dtype=hub.vector_observation_space.dtype)
                }

        return np.zeros((self.num_envs, *self.observation_space.shape), dtype=self.observation_space.dtype)

    def next_episode(self):
        next = self.nextEpisode
        self.nextEpisode += 1
        return next
    
    def reshape_image(self, data):
        flat_rgba = np.array(data, np.uint8)
        rgba_array = flat_rgba.reshape(hub.image_height, hub.image_width, 4).transpose(2, 0, 1)
        rgb_array = rgba_array[:3, :, :]
        return np.array(rgb_array, dtype=np.uint8)
    
    def reset(self):
        t0 = time.perf_counter()
        dlog("B-RESET", f"reset_all for {self.num_envs} agents", force=True)
        if hub.closed.is_set():
            dlog("B-RESET", "hub closed – empty obs", force=True)
            return self._empty_obs()

        hub.batch_event.clear()
        hub.send({"command": "reset_all"})
        dlog("B-RESET", "sent reset_all – waiting…")

        tw = time.perf_counter()
        if not hub.batch_event.wait(timeout=config["response_timeout"]) or hub.closed.is_set():
            dlog("B-RESET", f"!! TIMEOUT {(time.perf_counter()-tw)*1000:.0f}ms", force=True)
            dump_hub_state("B-RESET timeout (reset_all)")
            return self._empty_obs()
        dlog("B-RESET", f"response in {(time.perf_counter()-tw)*1000:.1f}ms  entries={len(hub.batch_data)}")

        obs = self._empty_obs()
        for agent_data in hub.batch_data:
            aid = agent_data['agent_id']
            if aid < self.num_envs:
                if hub.obs_type == 'dictionary':
                    obs["image"][aid] = self.reshape_image(agent_data['obs']["image"])
                    obs["vector"][aid] = np.array(agent_data['obs']["vector"], dtype=np.int64)
                elif hub.obs_type == 'image':
                    obs[aid] = self.reshape_image(agent_data['obs'])
                else:
                    obs[aid] = np.array(agent_data['obs'], dtype=np.int64)

                self.episode[aid] = self.next_episode()

        self.steps[:] = 0
        self.total_rewards[:] = 0
        dlog("B-RESET", f"complete {(time.perf_counter()-t0)*1000:.1f}ms", force=True)

        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        t0 = time.perf_counter()
        if hub.closed.is_set():
            dlog("B-STEP", "hub closed", force=True)
            return (self._empty_obs(), np.zeros(self.num_envs),
                    np.ones(self.num_envs, dtype=bool),
                    [{} for _ in range(self.num_envs)])


        #print(type(self._actions[0][0]))
        if hub.action_type == "discrete":
            actions_list = [int(a) for a in self._actions]
        else:
            actions_list = [float(a) for a in self._actions]

        hub.batch_event.clear()
        ts = time.perf_counter()
        hub.send({"command": "step_all", "actions": actions_list})
        send_ms = (time.perf_counter()-ts)*1000

        tw = time.perf_counter()
        if not hub.batch_event.wait(timeout=config["response_timeout"]) or hub.closed.is_set():
            dlog(
                "B-STEP", f"!! step_all TIMEOUT  send={send_ms:.1f}ms  wait={(time.perf_counter()-tw)*1000:.0f}ms", force=True)
            dump_hub_state(
                f"B-STEP timeout (step_all) actions={actions_list}")
            return (self._empty_obs(), np.full(self.num_envs, -1.0),
                    np.ones(self.num_envs, dtype=bool),
                    [{} for _ in range(self.num_envs)])
        wait_ms = (time.perf_counter()-tw)*1000

        # Parse batch response
        tp = time.perf_counter()
        obs = self._empty_obs()
        rewards_arr = np.zeros(self.num_envs, dtype=float)
        dones_arr = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        for agent_data in hub.batch_data:
            aid = agent_data['agent_id']
            if aid < self.num_envs:
                if hub.obs_type == 'dictionary':
                    obs["image"][aid] = self.reshape_image(agent_data['obs']["image"])
                    obs["vector"][aid] = np.array(agent_data['obs']["vector"], dtype=np.int64)
                elif hub.obs_type == 'image':
                    obs[aid] = self.reshape_image(agent_data['obs'])
                else:
                    obs[aid] = np.array(agent_data['obs'], dtype=np.int64)
                    
                rewards_arr[aid] = agent_data['reward']
                dones_arr[aid] = agent_data['done']
        parse_ms = (time.perf_counter()-tp)*1000

        # Update per-agent tracking - updates all agents at once
        self.steps += 1
        self.total_rewards += rewards_arr

        total = sum(self.steps)
        if (total - self.lastStepReport > 1000):
            self.lastStepReport = total
            print('Total Steps: ', f"{total:,}", '/', f"{hub.total_timesteps:,}")

        # Print reward for each agent this step
        #for i in range(self.num_envs):
        #    print(f'  Step reward: Agent {i} = {rewards_arr[i]:.2f}')

        # Max steps check
        for i in range(self.num_envs):
            if self.steps[i] >= self.max_steps and not dones_arr[i]:
                dones_arr[i] = True
                rewards_arr[i] = config["timeout_penalty"]
                self.timeouts[i] += 1
                dlog("B-STEP", f"agent {i} max_steps reached", force=True)

        # Auto-reset done environments (SB3 convention:
        #   store terminal obs in info, replace obs with fresh post-reset obs)
        done_ids = [i for i in range(self.num_envs) if dones_arr[i]]
        reset_ms = 0
        if done_ids:
            for i in done_ids:
                self.attempts[i] += 1
                dlog("B-STEP", f"[Agent {i}] Episode {int(self.episode[i])}  "
                     f"Reward: {self.total_rewards[i]:.2f}", force=True)
                #infos[i]["terminal_observation"] = obs[i].copy()
                self.terminal_steps[i] = self.steps[i]
                self.terminal_rewards[i] = self.total_rewards[i]
                self.total_rewards[i] = 0
                self.steps[i] = 0
                self.episode[i] = self.next_episode()

            hub.batch_event.clear()
            tr = time.perf_counter()
            hub.send({"command": "reset_agents", "agent_ids": done_ids})
            if hub.batch_event.wait(timeout=config["response_timeout"]) and not hub.closed.is_set():
                for agent_data in hub.batch_data:
                    aid = agent_data['agent_id']
                    if aid < self.num_envs:
                        if hub.obs_type == 'dictionary':
                            obs["image"][aid] = self.reshape_image(agent_data['obs']["image"])
                            obs["vector"][aid] = np.array(agent_data['obs']["vector"], dtype=np.int64)
                        elif hub.obs_type == 'image':
                            obs[aid] = self.reshape_image(agent_data['obs'])
                        else:
                            obs[aid] = np.array(agent_data['obs'], dtype=np.int64)

                reset_ms = (time.perf_counter()-tr)*1000
                dlog_step(
                    "B-STEP", f"reset_agents {done_ids} {reset_ms:.1f}ms")
            else:
                reset_ms = (time.perf_counter()-tr)*1000
                dlog(
                    "B-STEP", f"!! reset_agents TIMEOUT {done_ids} {reset_ms:.0f}ms", force=True)
                dump_hub_state(
                    f"B-STEP timeout (reset_agents) ids={done_ids}")

        total_ms = (time.perf_counter()-t0)*1000
        if total_ms > _SLOW_MS:
            dlog("B-STEP", f"!! SLOW total={total_ms:.1f}ms send={send_ms:.1f}ms "
                 f"wait={wait_ms:.1f}ms parse={parse_ms:.1f}ms reset={reset_ms:.1f}ms "
                 f"dones={done_ids}", force=True)
        else:
            dlog_step(
                "B-STEP", f"total={total_ms:.1f}ms send={send_ms:.1f}ms wait={wait_ms:.1f}ms")

        return obs, rewards_arr, dones_arr, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        target = indices if indices is not None else range(self.num_envs)
        return [None for _ in target]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target = indices if indices is not None else range(self.num_envs)
        return [None for _ in target]

    def env_is_wrapped(self, wrapper_class, indices=None):
        target = indices if indices is not None else range(self.num_envs)
        return [False for _ in target]

    def seed(self, seed=None):
        return [None for _ in range(self.num_envs)]
