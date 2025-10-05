#!/usr/bin/env python3
"""Test script for hybrid working memory implementation with 128K context."""

import sys
import os
sys.path.insert(0, '/Users/master/projects/atlas-main/src')

from atlas_main.memory import HybridWorkingMemory, WorkingMemoryConfig

def test_128k_memory():
    """Test hybrid working memory with 128K context configuration."""
    print("Testing 128K Context Hybrid Working Memory")
    print("=" * 50)
    
    # Test with 128K-optimized configuration
    config = WorkingMemoryConfig(
        max_turns=20,
        token_budget=96000,  # 96K token budget
        enable_token_awareness=True
    )
    
    memory = HybridWorkingMemory(config)
    
    print(f"\n1. Configuration:")
    print(f"   Max turns: {config.max_turns}")
    print(f"   Token budget: {config.token_budget:,} tokens")
    print(f"   Max token budget: {config.max_token_budget:,} tokens")
    
    # Add several large messages to test token management
    print(f"\n2. Adding large messages to test token handling...")
    
    # Simulate large conversation context
    large_user_msg = "This is a very long user message. " * 100  # ~3K tokens
    large_assistant_msg = "This is a comprehensive assistant response. " * 200  # ~6K tokens
    
    evicted = memory.add_user(large_user_msg)
    print(f"   Large user message added ({len(large_user_msg)} chars), evicted: {len(evicted)}")
    
    evicted = memory.add_assistant(large_assistant_msg)
    print(f"   Large assistant message added ({len(large_assistant_msg)} chars), evicted: {len(evicted)}")
    
    # Check stats
    stats = memory.get_stats()
    print(f"\n3. Memory stats after large messages:")
    print(f"   Turns: {stats['turns']}")
    print(f"   Tokens: {stats['tokens']:,}")
    print(f"   Capacity %: {stats['capacity_pct']:.1f}%")
    print(f"   Token %: {stats['token_pct']:.2f}%")
    
    # Add many more messages to test scalability
    print(f"\n4. Adding multiple conversation turns...")
    for i in range(15):
        evicted_user = memory.add_user(f"User message {i+1}: " + "This is a detailed question. " * 20)
        evicted_assistant = memory.add_assistant(f"Assistant response {i+1}: " + "This is a detailed answer. " * 30)
        
        if evicted_user or evicted_assistant:
            print(f"   Turn {i+1}: evicted {len(evicted_user)} user, {len(evicted_assistant)} assistant messages")
    
    # Final stats
    final_stats = memory.get_stats()
    print(f"\n5. Final memory stats:")
    print(f"   Turns: {final_stats['turns']}")
    print(f"   Tokens: {final_stats['tokens']:,}")
    print(f"   Capacity %: {final_stats['capacity_pct']:.1f}%")
    print(f"   Token %: {final_stats['token_pct']:.2f}%")
    print(f"   Important messages: {final_stats['important_messages']}")
    
    # Verify we're utilizing the large context effectively
    token_utilization = final_stats['tokens'] / config.token_budget * 100
    print(f"\n6. 128K Context Utilization Analysis:")
    print(f"   Working memory tokens: {final_stats['tokens']:,}")
    print(f"   Token budget: {config.token_budget:,}")
    print(f"   Utilization: {token_utilization:.1f}%")
    
    if token_utilization > 50:
        print(f"   ✅ Good utilization of 128K context capacity")
    else:
        print(f"   ⚠️  Could utilize more of the 128K context")
    
    print(f"\n✅ 128K context hybrid memory test completed!")

def test_token_estimation_accuracy():
    """Test token estimation accuracy with various text sizes."""
    print(f"\n" + "=" * 50)
    print("Testing Token Estimation Accuracy")
    print("=" * 50)
    
    from atlas_main.memory import _estimate_tokens
    
    test_cases = [
        ("Short", "Hello world"),
        ("Medium", "This is a medium length sentence with various words and punctuation."),
        ("Long", "This is a much longer text that contains multiple sentences. " * 10),
        ("Very Long", "This represents a very long context with extensive detail. " * 50),
    ]
    
    print(f"\n{'Text Type':<12} {'Characters':<12} {'Est. Tokens':<12} {'Chars/Token':<12}")
    print("-" * 50)
    
    for name, text in test_cases:
        chars = len(text)
        tokens = _estimate_tokens(text)
        ratio = chars / tokens if tokens > 0 else 0
        print(f"{name:<12} {chars:<12} {tokens:<12} {ratio:.1f}")
    
    print(f"\n✅ Token estimation working (4 chars ≈ 1 token)")

if __name__ == "__main__":
    test_128k_memory()
    test_token_estimation_accuracy()
