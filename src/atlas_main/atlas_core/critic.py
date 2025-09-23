"""Atlas Critic - Optional self-check stage for reasoning loop."""

from __future__ import annotations

from typing import Optional


class Critic:
    """Optional critic pass to surface contradictions or suggest disagreement."""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
    
    def critique(self, user_input: str, response: str, context: str) -> Optional[str]:
        """Perform optional critique of the response.
        
        Returns suggestion for disagreement or None if response is fine.
        """
        # Placeholder implementation
        prompt = (
            "Review this response for potential issues:\n"
            f"User: {user_input}\n"
            f"Response: {response}\n"
            f"Context: {context}\n\n"
            "If the response might conflict with user goals or show contradictions, "
            "suggest a brief disagreement. Otherwise, say 'OK'."
        )
        
        try:
            critique_response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            critique = critique_response.get("message", {}).get("content", "").strip()
            if critique and critique.lower() != "ok":
                return critique
        except Exception:
            pass
        
        return None
