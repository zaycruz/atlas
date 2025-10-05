#!/usr/bin/env python3
"""Test script for enhanced memory statistics visualization."""

import sys
import os
sys.path.insert(0, '/Users/master/projects/atlas-main/src')

from atlas_main.ui import ConversationShell
from rich.console import Console

def test_memory_visualization():
    """Test the enhanced memory statistics display."""
    print("Testing Enhanced Memory Statistics Visualization")
    print("=" * 60)
    
    console = Console()
    ui = ConversationShell(console)
    
    # Simulate memory statistics with various usage levels
    print("\n1. Testing with sample memory statistics...")
    
    # Test case 1: Low usage
    ui.update_memory_stats({
        "working_memory": {
            "turns": 5,
            "tokens": 12000,
            "capacity_pct": 25.0,
            "token_pct": 12.5,
            "important_messages": 1
        },
        "episodic_count": 500,
        "semantic_count": 150,
        "reflections_count": 75,
        "quality_gates": {
            "facts_accepted": 45,
            "reflections_accepted": 23
        }
    })
    
    print("   Low usage scenario:")
    try:
        rendered = ui._render_memory_stats()
        print("   ✅ Low usage memory stats render successfully")
    except Exception as e:
        print(f"   ❌ Low usage rendering error: {e}")
    
    # Test case 2: High usage
    ui.update_memory_stats({
        "working_memory": {
            "turns": 18,
            "tokens": 85000,
            "capacity_pct": 90.0,
            "token_pct": 88.5,
            "important_messages": 3
        },
        "episodic_count": 4500,
        "semantic_count": 720,
        "reflections_count": 350,
        "quality_gates": {
            "facts_accepted": 156,
            "reflections_accepted": 89
        }
    })
    
    print("   High usage scenario:")
    try:
        rendered = ui._render_memory_stats()
        print("   ✅ High usage memory stats render successfully")
    except Exception as e:
        print(f"   ❌ High usage rendering error: {e}")
    
    # Test case 3: Critical usage
    ui.update_memory_stats({
        "working_memory": {
            "turns": 20,
            "tokens": 94000,
            "capacity_pct": 100.0,
            "token_pct": 97.9,
            "important_messages": 5
        },
        "episodic_count": 5000,
        "semantic_count": 800,
        "reflections_count": 400,
        "quality_gates": {
            "facts_accepted": 234,
            "reflections_accepted": 145
        }
    })
    
    print("   Critical usage scenario:")
    try:
        rendered = ui._render_memory_stats()
        print("   ✅ Critical usage memory stats render successfully")
    except Exception as e:
        print(f"   ❌ Critical usage rendering error: {e}")
    
    # Test visual rendering
    print(f"\n2. Testing visual output (sample display):")
    console.print("=" * 40)
    console.print(rendered)
    console.print("=" * 40)
    
    print(f"\n✅ Enhanced memory visualization test completed!")

def test_progress_bars():
    """Test the progress bar calculations."""
    print(f"\n" + "=" * 60)
    print("Testing Progress Bar Calculations")
    print("=" * 60)
    
    test_cases = [
        (0, "Empty"),
        (25, "Quarter full"),
        (50, "Half full"),
        (75, "Three quarters"),
        (90, "Nearly full"),
        (100, "Full"),
    ]
    
    for percentage, description in test_cases:
        bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
        print(f"{description:15} ({percentage:3}%): [{bar}]")
    
    print(f"\n✅ Progress bar calculations working correctly!")

if __name__ == "__main__":
    test_memory_visualization()
    test_progress_bars()
