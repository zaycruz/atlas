#!/usr/bin/env python3
"""Test script for conversation scrolling functionality."""

import sys
import os
sys.path.insert(0, '/Users/master/projects/atlas-main/src')

from atlas_main.ui import ConversationShell
from rich.console import Console

def test_conversation_scrolling():
    """Test conversation scrolling with multiple turns."""
    print("Testing Conversation Scrolling Implementation")
    print("=" * 50)
    
    console = Console()
    ui = ConversationShell(console)
    
    # Add many conversation turns to test scrolling
    print("\n1. Adding multiple conversation turns...")
    for i in range(15):
        turn = ui.add_turn(f"User question {i+1}: This is a test question to create multiple turns for scrolling.")
        turn.assistant_text = f"Assistant response {i+1}: This is a detailed response that demonstrates the scrolling functionality."
        turn.status = "complete"
    
    print(f"   Added {len(ui.turns)} turns")
    print(f"   Max visible turns: {ui.max_visible_turns}")
    
    # Test initial state
    print(f"\n2. Initial scroll state:")
    print(f"   Total turns: {len(ui.turns)}")
    print(f"   Scroll offset: {ui.scroll_offset}")
    print(f"   Auto-scroll: {ui.auto_scroll}")
    
    # Test scroll calculations
    if len(ui.turns) > ui.max_visible_turns:
        start_idx = max(0, len(ui.turns) - ui.max_visible_turns - ui.scroll_offset)
        end_idx = len(ui.turns) - ui.scroll_offset
        print(f"   Visible turn range: {start_idx+1} to {end_idx}")
    
    # Test rendering (just check for errors)
    print(f"\n3. Testing conversation rendering...")
    try:
        rendered = ui._render_conversation()
        print(f"   ✅ Conversation renders successfully")
    except Exception as e:
        print(f"   ❌ Rendering error: {e}")
    
    # Test scroll offset changes
    print(f"\n4. Testing scroll offset manipulation...")
    
    # Simulate scrolling up
    ui.scroll_offset = 5
    ui.auto_scroll = False
    start_idx = max(0, len(ui.turns) - ui.max_visible_turns - ui.scroll_offset)
    end_idx = len(ui.turns) - ui.scroll_offset
    print(f"   After scrolling up (offset=5): showing turns {start_idx+1} to {end_idx}")
    
    # Simulate scrolling to top
    ui.scroll_offset = len(ui.turns) - ui.max_visible_turns
    start_idx = max(0, len(ui.turns) - ui.max_visible_turns - ui.scroll_offset)
    end_idx = len(ui.turns) - ui.scroll_offset
    print(f"   At top (offset={ui.scroll_offset}): showing turns {start_idx+1} to {end_idx}")
    
    # Simulate scrolling to bottom
    ui.scroll_offset = 0
    ui.auto_scroll = True
    start_idx = max(0, len(ui.turns) - ui.max_visible_turns - ui.scroll_offset)
    end_idx = len(ui.turns) - ui.scroll_offset
    print(f"   At bottom (offset=0): showing turns {start_idx+1} to {end_idx}")
    
    print(f"\n✅ Conversation scrolling test completed!")

def test_scroll_indicators():
    """Test scroll indicator rendering."""
    print(f"\n" + "=" * 50)
    print("Testing Scroll Indicators")
    print("=" * 50)
    
    console = Console()
    ui = ConversationShell(console)
    
    # Add enough turns to require scrolling
    for i in range(12):
        turn = ui.add_turn(f"Turn {i+1}")
        turn.assistant_text = f"Response {i+1}"
        turn.status = "complete"
    
    # Test different scroll positions
    test_cases = [
        (0, "Latest messages (bottom)"),
        (3, "Scrolled up 3 turns"),
        (ui.max_visible_turns - 1, "Near top"),
        (len(ui.turns) - ui.max_visible_turns, "Oldest messages (top)"),
    ]
    
    for offset, description in test_cases:
        ui.scroll_offset = offset
        start_idx = max(0, len(ui.turns) - ui.max_visible_turns - ui.scroll_offset)
        end_idx = len(ui.turns) - ui.scroll_offset
        
        hidden_above = start_idx
        hidden_below = ui.scroll_offset
        
        print(f"\n{description}:")
        print(f"   Scroll offset: {offset}")
        print(f"   Visible range: turns {start_idx+1}-{end_idx}")
        print(f"   Hidden above: {hidden_above}")
        print(f"   Hidden below: {hidden_below}")
    
    print(f"\n✅ Scroll indicators test completed!")

if __name__ == "__main__":
    test_conversation_scrolling()
    test_scroll_indicators()
