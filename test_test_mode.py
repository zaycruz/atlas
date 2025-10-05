#!/usr/bin/env python3
"""Test script for test mode functionality."""

import sys
import os
sys.path.insert(0, '/Users/master/projects/atlas-main/src')

from atlas_main.agent import AtlasAgent
from atlas_main.ui import ConversationShell
from atlas_main.ollama import OllamaClient
from rich.console import Console

def test_test_mode():
    """Test the test mode functionality."""
    print("Testing Test Mode Functionality")
    print("=" * 40)
    
    # Create mock agent (we won't actually use Ollama)
    console = Console()
    ui = ConversationShell(console)
    
    # Mock agent with test mode
    class MockAgent:
        def __init__(self):
            self.test_mode = False
    
    agent = MockAgent()
    
    # Test initial state
    print(f"\n1. Initial state:")
    print(f"   Test mode: {agent.test_mode}")
    print(f"   UI test mode: {ui.test_mode}")
    
    # Test enabling test mode
    print(f"\n2. Enabling test mode...")
    agent.test_mode = True
    ui.set_test_mode(True)
    print(f"   Agent test mode: {agent.test_mode}")
    print(f"   UI test mode: {ui.test_mode}")
    
    # Test UI rendering with test mode
    print(f"\n3. Testing UI rendering with test mode...")
    ui.update_memory_stats({
        'working_memory': {
            'turns': 5,
            'tokens': 12000,
            'capacity_pct': 25.0,
            'token_pct': 12.5
        },
        'episodic_count': 123,
        'semantic_count': 45,
        'reflections_count': 12
    })
    
    try:
        rendered = ui._render_memory_stats()
        print("   ✅ Memory stats render successfully with test mode indicator")
    except Exception as e:
        print(f"   ❌ Rendering error: {e}")
    
    # Test disabling test mode
    print(f"\n4. Disabling test mode...")
    agent.test_mode = False
    ui.set_test_mode(False)
    print(f"   Agent test mode: {agent.test_mode}")
    print(f"   UI test mode: {ui.test_mode}")
    
    # Test visual output
    print(f"\n5. Visual test with test mode ON:")
    ui.set_test_mode(True)
    console.print("=" * 30)
    console.print(ui._render_memory_stats())
    console.print("=" * 30)
    
    print(f"\n6. Visual test with test mode OFF:")
    ui.set_test_mode(False)
    console.print("=" * 30)
    console.print(ui._render_memory_stats())
    console.print("=" * 30)
    
    print(f"\n✅ Test mode functionality test completed!")

if __name__ == "__main__":
    test_test_mode()
