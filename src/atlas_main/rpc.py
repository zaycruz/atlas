"""Simple JSON-RPC loop for programmatic access to Atlas."""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, Optional

from .agent import AtlasAgent
from .ollama import OllamaClient
from .telemetry import Telemetry


def _write(response: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error(message: str, *, rid: Optional[Any]) -> None:
    _write({"id": rid, "error": message, "result": None})


def _success(result: Any, *, rid: Optional[Any]) -> None:
    _write({"id": rid, "error": None, "result": result})


def _stream_callback(buffer: list[str]) -> None:
    def _append(chunk: str) -> None:
        buffer.append(chunk)
    return _append


def _handle_chat(agent: AtlasAgent, params: Dict[str, Any]) -> str:
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    buffer: list[str] = []
    response = agent.respond(prompt, stream_callback=_stream_callback(buffer))
    return response or "".join(buffer)


def _handle_tool_list(agent: AtlasAgent) -> Dict[str, Any]:
    specs = []
    for tool_name in agent.tools.list_names():
        tool = agent.tools._tools.get(tool_name)  # pylint: disable=protected-access
        specs.append(
            {
                "name": tool_name,
                "description": getattr(tool, "description", ""),
                "capabilities": sorted(getattr(tool, "capabilities", set())),
            }
        )
    return {"tools": specs}


def _handle_tool_run(agent: AtlasAgent, params: Dict[str, Any]) -> str:
    name = params.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return agent.tools.run(name.strip(), agent=agent, arguments=arguments)


def _handle_memory_snapshot(agent: AtlasAgent, params: Dict[str, Any]) -> Dict[str, Any]:
    if not getattr(agent, "layered_memory", None):
        raise RuntimeError("layered memory is disabled")
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    snapshot = agent.layered_memory.build_snapshot(query, client=agent.client)
    return {
        "summary": snapshot.summary,
        "rendered": snapshot.rendered,
        "episodic": snapshot.assembled.episodic,
        "facts": snapshot.assembled.facts,
        "reflections": snapshot.assembled.reflections,
    }


def _dispatch(agent: AtlasAgent, request: Dict[str, Any]) -> None:
    rid = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        _error("params must be an object", rid=rid)
        return
    try:
        if method == "chat":
            result = _handle_chat(agent, params)
            _success({"text": result}, rid=rid)
        elif method == "tools.list":
            _success(_handle_tool_list(agent), rid=rid)
        elif method == "tools.run":
            result = _handle_tool_run(agent, params)
            _success({"output": result}, rid=rid)
        elif method == "memory.snapshot":
            _success(_handle_memory_snapshot(agent, params), rid=rid)
        elif method == "stats.get":
            _success(Telemetry.instance().stats(), rid=rid)
        elif method == "shutdown":
            _success({"status": "ok"}, rid=rid)
            raise SystemExit
        else:
            _error(f"unknown method: {method}", rid=rid)
    except SystemExit:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        _error(f"{type(exc).__name__}: {exc}", rid=rid)
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()


def run_loop(agent: AtlasAgent) -> None:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
        except json.JSONDecodeError:
            _error("invalid_json", rid=None)
            continue
        _dispatch(agent, request)


def main() -> None:
    client = OllamaClient()
    agent = AtlasAgent(client)
    try:
        run_loop(agent)
    except SystemExit:
        pass
    finally:
        try:
            agent.close()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
