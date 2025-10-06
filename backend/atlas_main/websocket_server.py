from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from .agent import AtlasAgent
from .ollama import OllamaClient

class AtlasWebSocketServer:
    """Expose the Atlas agent over a lightweight WebSocket protocol."""

    def __init__(self, host: str = "localhost", port: int = 8765, *, agent: Optional[AtlasAgent] = None) -> None:
        self.host = host
        self.port = port
        self.client = OllamaClient()
        self.agent = agent or AtlasAgent(self.client)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

    async def handle_message(self, websocket: WebSocketServerProtocol, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON payload")
            return

        msg_type = data.get("type")
        if msg_type == "command":
            payload = data.get("payload", "")
            if not isinstance(payload, str):
                await self._send_error(websocket, "Command payload must be a string")
                return
            print(f"[ws] <- command: {payload[:120]}")
            result = await self._execute_command(payload)
            print(f"[ws] -> response: {result[:120]}")
            await websocket.send(json.dumps({"type": "response", "payload": result}))
        elif msg_type == "get_metrics":
            metrics = self.agent.get_memory_metrics()
            await websocket.send(json.dumps({"type": "metrics", "payload": metrics}))
        else:
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        client_id = id(websocket)
        print(f"[ws] Client {client_id} connected")
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[ws] Client {client_id} disconnected")
            return
        finally:
            print(f"[ws] Client {client_id} connection closed")

    async def start(self) -> None:
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"ATLAS WebSocket server running on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

    async def _execute_command(self, command: str) -> str:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()

        def run_agent() -> str:
            return self.agent.respond(command)

        try:
            response = await loop.run_in_executor(self._executor, run_agent)
        except Exception as exc:  # noqa: BLE001 - surface agent failures
            response = f"Error: {exc}"
        _ = time.perf_counter() - start  # retains timing hook for future diagnostics
        return response

    async def _send_error(self, websocket: WebSocketServerProtocol, message: str) -> None:
        await websocket.send(json.dumps({"type": "error", "payload": message}))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.agent.close()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass
        self._executor.shutdown(wait=False)


def run(host: str = "localhost", port: int = 8765) -> None:
    server = AtlasWebSocketServer(host=host, port=port)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Stopping Atlas WebSocket server...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    run()
