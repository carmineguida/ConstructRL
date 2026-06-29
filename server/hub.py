"""WebSocketHub — single websocket connection shared by all agent environments."""

import json
import asyncio
import threading
import time

import numpy as np
from gymnasium import spaces

from .config import config
from .diagnostics import _ts, _step_counter, _SLOW_MS, dlog, dlog_step


class WebSocketHub:
    """Manages a single WebSocket connection shared by all agent environments.

    Messages to/from JavaScript carry an ``agent_id`` field so that each
    CustomEnv instance only sees its own responses.
    """

    def __init__(self):
        self.websocket = None
        self.loop = None

        # Per-agent threading events & data slots (keyed by agent_id int)
        self.response_events = {}
        self.received_rewards = {}
        self.received_dones = {}
        self.temp_obs = {}

        # Handshake (global, not per-agent)
        self.handshake_event = threading.Event()
        self.handshake_received = False
        self.handshake_config = None

        # Connection state – set when the client disconnects
        self.closed = threading.Event()

        # Batch response (for batched step/reset)
        self.batch_event = threading.Event()
        self.batch_data = []

        # Parsed spaces (set after handshake)
        self.observation_space = None
        self.action_space = None
        self.max_steps = config["max_steps"]
        self.total_timesteps = config["total_timesteps"]
        self.num_agents = config["num_agents"]
        self.mode = config["mode"]


    # -- helpers to lazily create per-agent slots --
    def _ensure_agent(self, agent_id):
        if agent_id not in self.response_events:
            self.response_events[agent_id] = threading.Event()
            self.received_rewards[agent_id] = 0
            self.received_dones[agent_id] = False
            self.temp_obs[agent_id] = None

    def shutdown(self, reason="connection closed"):
        """Mark the hub as closed and unblock all waiting agent threads."""
        print(f'\nWebSocketHub shutting down: {reason}')
        self.websocket = None
        self.closed.set()
        for evt in self.response_events.values():
            evt.set()
        self.handshake_event.set()
        self.batch_event.set()

    # -- send helper (thread-safe) --
    def send(self, data_dict, timeout=None):
        if timeout is None:
            timeout = config["send_timeout"]
        """Send a JSON message over the websocket from any thread."""
        if self.closed.is_set() or self.websocket is None or self.loop is None:
            dlog("SEND", "SKIP – closed/no ws/no loop")
            return
        msg = json.dumps(data_dict)
        cmd = data_dict.get('command', '???')
        t0 = time.perf_counter()
        dlog_step("SEND", f">> {cmd}  len={len(msg)}")
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.websocket.send(msg), self.loop
            )
            future.result(timeout=timeout)
            ms = (time.perf_counter()-t0)*1000
            if ms > _SLOW_MS:
                dlog("SEND", f"!! SLOW send {ms:.1f}ms  cmd={cmd}", force=True)
            else:
                dlog_step("SEND", f"<< {cmd} OK {ms:.1f}ms")
        except Exception as e:
            ms = (time.perf_counter()-t0)*1000
            dlog("SEND", f"EXCEPTION {ms:.1f}ms: {e}", force=True)
            if '1001' in str(e):
                print("\nClient closed connection (1001 Going Away) – quitting...")
                import os
                os._exit(0)
            raise

    # -- called by websocket handler when a message arrives --
    def dispatch(self, message_data):
        """Route an incoming JSON message to the correct agent slot."""
        if 'handshake' in message_data:
            dlog("DISPATCH", "handshake received", force=True)
            self._handle_handshake(message_data['handshake'])
            return

        if 'ack_num_agents' in message_data:
            dlog("DISPATCH", "ack_num_agents", force=True)
            return
        
        if message_data.get('batch'):
            n = len(message_data.get('agents', []))
            dlog_step("DISPATCH", f"batch response  n_agents={n}")
            self.batch_data = message_data.get('agents', [])
            self.batch_event.set()
            return

        agent_id = message_data.get('agent_id', 0)
        self._ensure_agent(agent_id)

        if 'reward' in message_data:
            self.received_rewards[agent_id] = message_data['reward']
        if 'done' in message_data:
            self.received_dones[agent_id] = message_data['done']
        if 'obs' in message_data:
            self.temp_obs[agent_id] = message_data['obs']

        dlog_step(
            "DISPATCH", f"agent {agent_id}  r={message_data.get('reward', '?')}  d={message_data.get('done', '?')}")
        self.response_events[agent_id].set()

    def find_config(self, config, t):
        for c in config:
            if c["type"] == t: return c
        return None
    
    def parse_image_config(self, image_config):
        self.image_channels = image_config['shape'][0]
        self.image_height = image_config['shape'][1]
        self.image_width = image_config['shape'][2]

    def parse_vector_config(self, vector_config):
        self.obs_low = np.float64(vector_config['low']) if vector_config['low'] != '-inf' else -np.inf
        self.obs_high = np.float64(vector_config['high']) if vector_config['high'] != 'inf' else np.inf
    
    def _handle_handshake(self, hs_config):
        
        obs_config = hs_config['observation_space']
        
        if isinstance(obs_config, list): # multi input

            self.obs_type = "dictionary"
            self.image_config = self.find_config(obs_config, "image")
            self.parse_image_config(self.image_config)
            self.image_observation_space = spaces.Box(low=0, high=255, shape=tuple(self.image_config['shape']), dtype=np.uint8)
            
            self.vector_config = self.find_config(obs_config, "vector")
            self.parse_vector_config(self.vector_config)
            self.vector_observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, shape=tuple(self.vector_config['shape']), dtype=np.int64)

            self.observation_space = spaces.Dict({
                "image":self.image_observation_space,
                "vector":self.vector_observation_space
            })

        else:
            self.obs_type = obs_config.get('type', 'vector')
        
            if self.obs_type == 'image':
                self.parse_image_config(obs_config)
                self.observation_space = spaces.Box(low=0, high=255, shape=tuple(obs_config['shape']), dtype=np.uint8)
            else:
                self.parse_vector_config(obs_config)
                self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, shape=tuple(obs_config['shape']), dtype=np.int64)


        action_config = hs_config['action_space']
        self.action_type = action_config['type']
        
        if action_config['type'] == 'discrete':
            self.action_space = spaces.Discrete(action_config['n'])
        elif action_config['type'] == 'continuous':
            action_low = np.float32(
                action_config['low']) if action_config['low'] != '-inf' else -np.inf
            action_high = np.float32(
                action_config['high']) if action_config['high'] != 'inf' else np.inf
            self.action_space = spaces.Box(
                low=action_low, high=action_high,
                shape=tuple(action_config['shape']), dtype=np.float32
            )
        else:
            raise ValueError(
                f'Unknown action space type: {action_config["type"]}')

        if 'max_steps' in hs_config:
            self.max_steps = hs_config['max_steps']
        if 'total_timesteps' in hs_config:
            self.total_timesteps = hs_config['total_timesteps']
        if 'num_agents' in hs_config:
            self.num_agents = hs_config['num_agents']
        if 'mode' in hs_config:
            self.mode = hs_config['mode']

        for i in range(self.num_agents):
            self._ensure_agent(i)

        self.handshake_config = hs_config
        self.handshake_received = True
        self.handshake_event.set()

        print(f'Handshake received:')
        print(f'  observation_space = {self.observation_space}')
        print(f'  action_space = {self.action_space}')
        print(f'  max_steps = {self.max_steps}')
        print(f'  total_timesteps = {self.total_timesteps}')
        print(f'  num_agents = {self.num_agents}')
        print(f'  mode = {self.mode}')



# Global hub singleton
hub = WebSocketHub()


def dump_hub_state(label=""):
    """Print a snapshot of hub internals — call on every timeout to diagnose hangs."""
    print(f"\n{'='*60}", flush=True)
    print(f"  HUB STATE DUMP  ({label})  at {_ts()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  websocket   = {hub.websocket}", flush=True)
    print(f"  loop        = {hub.loop}", flush=True)
    print(f"  closed      = {hub.closed.is_set()}", flush=True)
    print(
        f"  handshake   = received={hub.handshake_received}  event={hub.handshake_event.is_set()}", flush=True)
    print(
        f"  batch_event = {hub.batch_event.is_set()}  batch_data_len={len(hub.batch_data)}", flush=True)
    print(f"  num_agents  = {hub.num_agents}", flush=True)
    for aid in sorted(hub.response_events.keys()):
        evt = hub.response_events[aid]
        obs = hub.temp_obs.get(aid)
        obs_str = f"len={len(obs)}" if obs is not None else "None"
        print(f"  agent {aid}: resp_event={evt.is_set()}  "
              f"reward={hub.received_rewards.get(aid, '?')}  "
              f"done={hub.received_dones.get(aid, '?')}  "
              f"obs={obs_str}", flush=True)
    print(f"  total dlog steps = {_step_counter}", flush=True)
    print(f"{'='*60}\n", flush=True)
