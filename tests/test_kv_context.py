from __future__ import annotations

from atlas_main.agent import AtlasAgent


class _FakeClientWithContext:
    def __init__(self):
        self._round = 0
        self.last_received_context = None

    def chat_stream(self, *, model, messages, tools=None, context=None, keep_alive=None, options=None):
        # Record the context passed in by the agent
        self.last_received_context = context
        self._round += 1
        if self._round == 1:
            # First turn returns a final chunk with a context payload
            yield {"content": "First done.", "tool_calls": [], "context": [1, 2, 3]}
        else:
            # Second turn should receive the previous context back
            yield {"content": "Second done.", "tool_calls": []}

    def close(self):
        pass


def test_kv_context_is_propagated_between_turns():
    client = _FakeClientWithContext()
    agent = AtlasAgent(client)
    try:
        # First respond should capture context from final chunk
        out1 = agent.respond("hello")
        assert "First done" in out1

        # Second respond should pass captured context back to client
        out2 = agent.respond("again")
        assert "Second done" in out2

        # Ensure the client saw the propagated context on the second call
        assert client.last_received_context == [1, 2, 3]
    finally:
        agent.close()
