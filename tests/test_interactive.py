#!/usr/bin/env python3
"""Quick interactive test to verify CLI works."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from atlas_main.ui import ConversationShell
from atlas_main.agent import AtlasAgent
from atlas_main.ollama import OllamaClient
from rich.console import Console

def test_initialization():
    """Test that all components initialize correctly."""
    print("🧪 Testing Atlas CLI Components\n")

    try:
        console = Console()
        print("✓ Console created")

        ui = ConversationShell(console)
        print(f"✓ UI initialized (objective: {ui.objective})")

        client = OllamaClient()
        print("✓ Ollama client created")

        agent = AtlasAgent(client)
        print(f"✓ Agent created (model: {agent.chat_model})")

        # Test that objective can be set
        ui.set_objective("test objective", ["tag1"])
        print(f"✓ Objective setting works: {ui.objective}")

        print("\n✅ All components initialized successfully!")
        print("\nYou can now run: atlas")
        print("Or: poetry run python -m atlas_main.cli")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
