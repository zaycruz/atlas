"""Utility client for interacting with the local Ollama server."""
from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional

import requests


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error response."""


class OllamaClient:
    """Small wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        request_timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core HTTP helpers
    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict, *, stream: bool = False, request_timeout: Optional[int] = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        timeout = request_timeout if request_timeout is not None else self.timeout
        response = self._session.post(url, json=payload, timeout=timeout, stream=stream)
        self._raise_for_status(response)
        return response

    def _get(self, path: str, *, request_timeout: Optional[int] = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        timeout = request_timeout if request_timeout is not None else self.timeout
        response = self._session.get(url, timeout=timeout)
        self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network error path
            detail = None
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise OllamaError(f"Ollama request failed: {detail}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict] = None,
    keep_alive: Optional[int] = None,
    request_timeout: Optional[int] = None,
    ) -> Dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        response = self._post("/api/chat", payload, stream=stream, request_timeout=request_timeout)
        if stream:
            return {"stream": self._stream_chat(response)}
        return response.json()

    def chat_stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict] = None,
    keep_alive: Optional[int] = None,
    request_timeout: Optional[int] = None,
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        response = self._post("/api/chat", payload, stream=True, request_timeout=request_timeout)
        yield from self._stream_chat(response)

    def _stream_chat(self, response: requests.Response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("done"):
                break
            message = data.get("message") or {}
            content = message.get("content") or data.get("response")
            if content:
                yield content

    def embed(self, model: str, text: str) -> Optional[List[float]]:
        payload = {"model": model, "prompt": text}
        response = self._post("/api/embeddings", payload)
        body = response.json()
        return body.get("embedding")

    def list_models(self) -> List[str]:
        response = self._get("/api/tags")
        body = response.json()
        items = body.get("models", [])
        return [item.get("name", "") for item in items if item.get("name")]

    def close(self) -> None:
        self._session.close()


__all__ = ["OllamaClient", "OllamaError"]
