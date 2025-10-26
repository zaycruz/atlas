"""Minimal process agent that echoes the requested task."""
from __future__ import annotations

import json
import sys
from typing import Any, Dict


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _emit_failure(f"Invalid payload: {exc}")
        return

    step: Dict[str, Any] = payload.get("step") or {}
    description = step.get("description", "(missing description)")
    response = {
        "status": "succeeded",
        "summary": "Echo agent produced a placeholder plan.",
        "artifacts": [
            {
                "kind": "note",
                "content": f"I would tackle: {description}",
            }
        ],
        "logs": [
            "echo-agent: placeholder agent for orchestrator smoke test",
        ],
        "metadata": {
            "echo": True,
        },
    }
    json.dump(response, sys.stdout)


def _emit_failure(message: str) -> None:
    payload = {
        "status": "failed",
        "summary": message,
        "logs": [message],
        "artifacts": [],
    }
    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
