from __future__ import annotations

import os
from typing import List

from atlas_main.memory import MemoryRecord, summarize_memories_abstractive


class _FakeClient:
    def __init__(self, response_text: str = "Summary OK") -> None:
        self.response_text = response_text
        self.last_model = None
        self.last_messages = None

    def chat(
        self,
        *,
        model: str,
        messages: List[dict],
        stream: bool = False,
        options=None,
        tools=None,
        context=None,
        keep_alive=None,
        request_timeout=None,
    ):
        self.last_model = model
        self.last_messages = messages
        return {"message": {"content": self.response_text}}


def _mk_records(n: int) -> list[MemoryRecord]:
    recs = []
    for i in range(n):
        recs.append(
            MemoryRecord(
                id=str(i),
                user=f"user {i}",
                assistant=f"assistant {i} did thing {i}",
                timestamp=float(i),
                embedding=None,
            )
        )
    return recs


def test_summarizer_uses_provided_model_and_limits_items(monkeypatch):
    client = _FakeClient(response_text="• A\n• B")
    recs = _mk_records(5)

    out = summarize_memories_abstractive(recs, client, model="phi3:latest", max_items=2, style="bullets")

    assert "•" in out
    assert client.last_model == "phi3:latest"
    # Ensure payload includes our episodes content
    joined = "\n".join([m["content"] for m in client.last_messages if m["role"] == "user"])  # type: ignore[index]
    assert "assistant 0 did thing 0" in joined


def test_summarizer_uses_env_model_when_not_provided(monkeypatch):
    monkeypatch.setenv("ATLAS_SUMMARY_MODEL", "llama3:instruct")
    client = _FakeClient()
    recs = _mk_records(1)

    _ = summarize_memories_abstractive(recs, client)
    assert client.last_model == "llama3:instruct"


def test_summarizer_handles_empty_records():
    client = _FakeClient()
    out = summarize_memories_abstractive([], client)
    assert "no episodes" in out
