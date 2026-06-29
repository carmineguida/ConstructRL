"""WebSocket server for bridging Construct 3 and the RL agent."""

import asyncio
import json
import time

import websockets

from .config import config
from .diagnostics import dlog
from .hub import hub


async def connection_handler(websocket):
    dlog("WS", f"Client connected from {websocket.remote_address}", force=True)
    hub.websocket = websocket
    hub.loop = asyncio.get_event_loop()
    n_msg = 0
    try:
        async for message in websocket:
            n_msg += 1
            t0 = time.perf_counter()
         
            message_data = json.loads(message)
            hub.dispatch(message_data)
            #if 'obs' in message_data:
            #    print(f"Received coordinates: {message_data['obs']}")
            ms = (time.perf_counter()-t0)*1000
            if ms > config["ws_slow_recv_ms"]:
                dlog(
                    "WS", f"!! SLOW recv+dispatch {ms:.1f}ms  msg#{n_msg}", force=True)
    except websockets.exceptions.ConnectionClosedOK:
        dlog("WS", f"closed normally after {n_msg} msgs", force=True)
        hub.shutdown("connection closed normally")
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 1001:
            print("\nClient closed connection (1001 Going Away) – quitting...")
            import os
            os._exit(0)
        dlog("WS", f"closed code={e.code} after {n_msg} msgs", force=True)
        hub.shutdown(f"connection closed (code {e.code}): {e}")


async def websocket_server():
    host = config["ws_host"]
    port = config["ws_port"]
    print(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(connection_handler, host, port):
        await asyncio.Future()


def run_websocket_server():
    asyncio.run(websocket_server())
