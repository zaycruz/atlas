#!/usr/bin/env python3
"""Test script to verify WebSocket communication with Atlas server."""

import asyncio
import json

import websockets


async def _run_test() -> None:
    """Test sending a command and receiving a response."""
    uri = "ws://localhost:8765"

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected!")

        # Test 1: Get metrics
        print("\n[Test 1] Requesting metrics...")
        await websocket.send(json.dumps({"type": "get_metrics"}))
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Received metrics response: {data['type']}")
        print(f"  - Atlas tokens: {data['payload']['atlas']['tokens']}")
        print(f"  - Operations: {data['payload']['atlas']['operations']}")

        # Test 2: Send a simple command
        print("\n[Test 2] Sending command: 'What is 2+2?'...")
        await websocket.send(json.dumps({"type": "command", "payload": "What is 2+2?"}))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Received command response: {data['type']}")
        print(f"  Response: {data['payload'][:200]}...")

        # Might also receive metrics update
        try:
            metrics_response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            metrics_data = json.loads(metrics_response)
            if metrics_data['type'] == 'metrics':
                print(f"✓ Received metrics update after command")
        except asyncio.TimeoutError:
            pass

        print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(_run_test())


def test_websocket() -> None:
    asyncio.run(_run_test())
