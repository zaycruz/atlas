import asyncio
import json
from typing import Any, Dict

import websockets

from .agent import AtlasAgent


class AtlasWebSocketServer:
    def __init__(self, host: str = 'localhost', port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.agent = AtlasAgent()

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str) -> None:
        data = json.loads(message)
        msg_type = data.get('type')
        payload = data.get('payload')

        if msg_type == 'command':
            response = await self.agent.process_command(str(payload))
            await websocket.send(json.dumps({
                'type': 'response',
                'payload': response
            }))
        elif msg_type == 'get_metrics':
            metrics = self.agent.get_memory_metrics()
            await websocket.send(json.dumps({
                'type': 'metrics',
                'payload': metrics
            }))
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'payload': f'Unknown message type: {msg_type}'
            }))

    async def handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self) -> None:
        async with websockets.serve(self.handler, self.host, self.port):
            print(f'ATLAS WebSocket server running on ws://{self.host}:{self.port}')
            await asyncio.Future()


def run() -> None:
    server = AtlasWebSocketServer()
    asyncio.run(server.start())


if __name__ == '__main__':
    run()
