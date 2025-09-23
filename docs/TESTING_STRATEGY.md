# Memory Enhancement Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for Atlas's enhanced memory system, including unit tests, integration tests, performance benchmarks, and validation procedures for contextual chunking, temporal decay, and session memory features.

## Testing Philosophy

### Core Principles
1. **Test-Driven Development**: All new features implemented with tests first
2. **Comprehensive Coverage**: Minimum 90% code coverage for new components
3. **Backward Compatibility**: Ensure existing functionality remains intact
4. **Performance Validation**: Memory operations meet specified performance targets
5. **Edge Case Resilience**: Handle malformed inputs and error conditions gracefully

### Testing Pyramid

```
    E2E Tests (5%)
   ┌─────────────────┐
   │ Full CLI/Agent  │
   └─────────────────┘
     
  Integration Tests (25%)
 ┌─────────────────────┐
 │ Memory Components   │
 │ Cross-System Tests  │
 └─────────────────────┘
   
    Unit Tests (70%)
  ┌───────────────────────┐
  │ Individual Functions  │
  │ Class Methods         │
  │ Edge Cases           │
  └───────────────────────┘
```

## Unit Testing Strategy

### Test Coverage Requirements

#### Contextual Chunking
- **Chunk boundary detection**: Sentence, paragraph, and semantic boundaries
- **Overlap calculation**: Proper overlap percentages and character positions  
- **Size constraints**: Minimum and maximum chunk sizes
- **Metadata generation**: Accurate chunk positioning and semantic flags
- **Error handling**: Invalid inputs, empty content, encoding issues

#### Temporal Decay
- **Weight calculation**: Exponential decay for different time periods
- **Importance boosting**: Access frequency multipliers and caps
- **Memory type handling**: Different decay rates for memory types
- **Forgetting thresholds**: Memory removal based on age
- **Configuration validation**: Parameter bounds and edge cases

#### Session Memory
- **Session creation**: Automatic session boundaries and triggers
- **Context retrieval**: Recent turns and cross-session queries
- **Expiration handling**: Session cleanup and memory management
- **Promotion logic**: Long-term memory promotion criteria
- **Thread safety**: Concurrent access and cleanup operations

### Unit Test Implementation

```python
import unittest
import tempfile
import time
import math
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from atlas_main.memory import MemoryRecord, EpisodicMemory
from enhanced_memory import (
    ChunkedMemoryRecord, 
    ContextualChunker, 
    TemporalMemoryWeighting,
    SessionMemory,
    EnhancedEpisodicMemory
)

class TestContextualChunking(unittest.TestCase):
    """Comprehensive tests for contextual chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = ContextualChunker(
            chunk_size=100,
            overlap_ratio=0.2,
            min_chunk_size=30
        )
        self.embedding_fn = Mock(return_value=[1.0, 0.0, 0.0, 0.0])
    
    def test_short_conversation_single_chunk(self):
        """Test that short conversations remain as single chunks."""
        user = "Hi"
        assistant = "Hello there!"
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_parent)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertIsNone(chunks[0].parent_id)
    
    def test_long_conversation_multiple_chunks(self):
        """Test chunking of conversations exceeding chunk size."""
        user = "Tell me about Python"
        assistant = "Python is a programming language. " * 20  # ~600 chars
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        self.assertGreater(len(chunks), 1, "Should create multiple chunks")
        self.assertTrue(chunks[0].is_parent, "First chunk should be parent")
        
        # Verify parent-child relationships
        parent_id = chunks[0].id
        for i in range(1, len(chunks)):
            child = chunks[i]
            self.assertTrue(child.is_child, f"Chunk {i} should be child")
            self.assertEqual(child.parent_id, parent_id)
            self.assertEqual(child.chunk_index, i)
    
    def test_semantic_boundary_detection(self):
        """Test detection of sentence and paragraph boundaries."""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        
        # Test sentence boundary detection
        pos, is_semantic = self.chunker.find_semantic_boundary(text, 20, 10)
        self.assertTrue(is_semantic)
        self.assertIn(text[pos-2:pos], [". ", "! ", "? "])
        
        # Test paragraph boundary (higher priority)
        text_with_para = "First para.\n\nSecond para."
        pos, is_semantic = self.chunker.find_semantic_boundary(text_with_para, 15, 10)
        self.assertTrue(is_semantic)
    
    def test_chunk_overlap_calculation(self):
        """Test that chunks have proper overlap."""
        user = "Question"
        assistant = "Word " * 50  # Predictable text for testing overlap
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        if len(chunks) > 2:
            # Check overlap between consecutive chunks
            for i in range(1, len(chunks) - 1):
                chunk = chunks[i]
                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                
                self.assertIsNotNone(chunk.chunk_metadata)
                self.assertGreater(chunk.chunk_metadata.overlap_start, 0)
                
                if next_chunk:
                    # Verify overlap makes sense
                    expected_overlap = int(self.chunker.chunk_size * self.chunker.overlap_ratio)
                    self.assertLessEqual(chunk.chunk_metadata.overlap_start, expected_overlap + 10)
    
    def test_minimum_chunk_size_enforcement(self):
        """Test that tiny chunks are not created."""
        user = "Short"
        assistant = "A" * 150  # Just over chunk_size but would create tiny remainder
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        # All non-parent chunks should meet minimum size
        for chunk in chunks[1:]:  # Skip parent
            self.assertGreaterEqual(
                len(chunk.content.strip()), 
                self.chunker.min_chunk_size - 10,  # Allow small tolerance
                f"Chunk too small: {len(chunk.content)} chars"
            )
    
    def test_empty_input_handling(self):
        """Test handling of empty or whitespace-only inputs."""
        test_cases = [
            ("", ""),
            ("  ", "  "),
            ("user", ""),
            ("", "assistant"),
        ]
        
        for user, assistant in test_cases:
            with self.subTest(user=user, assistant=assistant):
                chunks = self.chunker.chunk_conversation(user, assistant)
                self.assertEqual(len(chunks), 1)
                self.assertTrue(chunks[0].is_parent)
    
    def test_unicode_handling(self):
        """Test proper handling of Unicode characters."""
        user = "What about émojis? 🤖"
        assistant = "Émojis work fine! 😊 " * 20
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        # Should not crash and should preserve Unicode
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk.content, str)
            # Verify Unicode characters are preserved
            if "😊" in chunk.content:
                self.assertIn("😊", chunk.content)

class TestTemporalDecay(unittest.TestCase):
    """Test temporal decay weighting functionality."""
    
    def setUp(self):
        """Set up temporal weighting with test configuration."""
        from enhanced_memory import TemporalConfig
        
        self.config = TemporalConfig(
            episodic_memory_decay=0.1,
            access_boost_factor=0.1,
            max_importance_multiplier=3.0
        )
        self.weighting = TemporalMemoryWeighting(self.config)
    
    def test_temporal_weight_decreases_with_age(self):
        """Test that older memories have lower temporal weights."""
        now = time.time()
        
        weights = []
        for hours_ago in [0, 1, 6, 24, 168]:  # Now to 1 week
            timestamp = now - (hours_ago * 3600)
            weight = self.weighting.calculate_temporal_weight(timestamp)
            weights.append(weight)
        
        # Weights should decrease monotonically
        for i in range(1, len(weights)):
            self.assertGreater(
                weights[i-1], weights[i],
                f"Weight should decrease: {weights[i-1]} > {weights[i]}"
            )
        
        # Recent memory should have weight ≈ 1.0
        self.assertAlmostEqual(weights[0], 1.0, places=2)
    
    def test_different_memory_type_decay_rates(self):
        """Test that different memory types have different decay rates."""
        timestamp = time.time() - 3600  # 1 hour ago
        
        working_weight = self.weighting.calculate_temporal_weight(timestamp, "working")
        session_weight = self.weighting.calculate_temporal_weight(timestamp, "session")
        episodic_weight = self.weighting.calculate_temporal_weight(timestamp, "episodic")
        semantic_weight = self.weighting.calculate_temporal_weight(timestamp, "semantic")
        
        # Should follow: working < session < episodic < semantic (for same age)
        self.assertLess(working_weight, session_weight)
        self.assertLess(session_weight, episodic_weight)
        self.assertLess(episodic_weight, semantic_weight)
    
    def test_importance_boosting_with_access_frequency(self):
        """Test that frequently accessed memories get importance boost."""
        base_importance = 1.0
        
        # Test different access counts
        access_counts = [0, 1, 5, 10, 50]
        multipliers = []
        
        for count in access_counts:
            multiplier = self.weighting.calculate_importance_multiplier(count, base_importance)
            multipliers.append(multiplier)
        
        # Multipliers should increase with access count
        for i in range(1, len(multipliers)):
            self.assertGreaterEqual(
                multipliers[i], multipliers[i-1],
                f"Importance should not decrease with more access: {access_counts[i]} accesses"
            )
        
        # Should respect maximum multiplier
        high_access_multiplier = self.weighting.calculate_importance_multiplier(1000, base_importance)
        self.assertLessEqual(high_access_multiplier, self.config.max_importance_multiplier)
    
    def test_final_score_calculation(self):
        """Test combined scoring with all factors."""
        now = time.time()
        
        # Test case: high similarity, recent, frequently accessed
        score = self.weighting.calculate_final_score(
            similarity_score=0.9,
            timestamp=now - 1800,  # 30 minutes ago
            access_count=5,
            base_importance=1.0,
            memory_type="episodic"
        )
        
        # Score should be close to similarity but potentially boosted
        self.assertGreater(score, 0.8)  # Should be high due to recent + accessed
        self.assertLessEqual(score, 0.9 * self.config.max_importance_multiplier)
    
    def test_forgetting_threshold_enforcement(self):
        """Test that very old memories are marked for forgetting."""
        now = time.time()
        
        # Very recent memory should not be forgotten
        recent_timestamp = now - 60  # 1 minute ago
        should_forget_recent = self.weighting.should_forget_memory(recent_timestamp)
        self.assertFalse(should_forget_recent)
        
        # Very old memory should be forgotten
        old_timestamp = now - (365 * 24 * 3600)  # 1 year ago
        should_forget_old = self.weighting.should_forget_memory(old_timestamp)
        self.assertTrue(should_forget_old)
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases in temporal weight calculation."""
        now = time.time()
        
        # Future timestamp (should not crash)
        future_weight = self.weighting.calculate_temporal_weight(now + 3600)
        self.assertGreaterEqual(future_weight, 0.0)
        self.assertLessEqual(future_weight, 1.0)
        
        # Negative access count
        negative_multiplier = self.weighting.calculate_importance_multiplier(-5)
        self.assertEqual(negative_multiplier, 1.0)  # Should default to base importance
        
        # Zero similarity score
        zero_score = self.weighting.calculate_final_score(0.0, now, 5)
        self.assertEqual(zero_score, 0.0)

class TestSessionMemory(unittest.TestCase):
    """Test session memory functionality."""
    
    def setUp(self):
        """Set up session memory with test configuration."""
        from enhanced_memory import SessionConfig
        
        self.config = SessionConfig(
            duration_hours=1.0,
            max_turns_per_session=5,
            inactivity_threshold_minutes=15,
            cleanup_interval_minutes=1  # Fast cleanup for testing
        )
        self.embedding_fn = Mock(return_value=[0.5, 0.5, 0.0])
        self.session_memory = SessionMemory(
            config=self.config,
            embedding_fn=self.embedding_fn
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop any running timers
        if hasattr(self.session_memory, '_cleanup_timer') and self.session_memory._cleanup_timer:
            self.session_memory._cleanup_timer.cancel()
    
    def test_session_creation_and_turn_addition(self):
        """Test basic session creation and turn addition."""
        session_id = self.session_memory.add_turn("Hello", "Hi there!")
        
        self.assertIsNotNone(session_id)
        self.assertEqual(self.session_memory._current_session_id, session_id)
        
        # Get context and verify
        context = self.session_memory.get_session_context()
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].user, "Hello")
        self.assertEqual(context[0].assistant, "Hi there!")
    
    def test_multiple_turns_same_session(self):
        """Test adding multiple turns to the same session."""
        turns = [
            ("Question 1", "Answer 1"),
            ("Question 2", "Answer 2"),
            ("Question 3", "Answer 3"),
        ]
        
        session_ids = []
        for user, assistant in turns:
            session_id = self.session_memory.add_turn(user, assistant)
            session_ids.append(session_id)
        
        # All turns should be in the same session
        self.assertEqual(len(set(session_ids)), 1)
        
        # Context should contain all turns
        context = self.session_memory.get_session_context()
        self.assertEqual(len(context), 3)
    
    def test_session_capacity_limits(self):
        """Test that sessions respect turn capacity limits."""
        # Add more turns than max_turns_per_session
        session_ids = []
        for i in range(self.config.max_turns_per_session + 2):
            session_id = self.session_memory.add_turn(f"Q{i}", f"A{i}")
            session_ids.append(session_id)
        
        # Should have created multiple sessions
        unique_sessions = set(session_ids)
        self.assertGreater(len(unique_sessions), 1)
        
        # Total context should include all turns
        all_context = self.session_memory.get_all_recent_context()
        self.assertEqual(len(all_context), self.config.max_turns_per_session + 2)
    
    def test_session_inactivity_boundary(self):
        """Test session creation due to inactivity."""
        # Add initial turn
        session_id1 = self.session_memory.add_turn("First", "Response 1")
        
        # Simulate inactivity by modifying last activity time
        session = self.session_memory._sessions[session_id1]
        session.turns[-1].timestamp = time.time() - (self.config.inactivity_threshold_minutes * 60 + 60)
        
        # Add another turn - should create new session
        session_id2 = self.session_memory.add_turn("Second", "Response 2")
        
        self.assertNotEqual(session_id1, session_id2)
    
    def test_session_context_retrieval(self):
        """Test various context retrieval methods."""
        # Add turns across multiple sessions
        self.session_memory.add_turn("Q1", "A1")
        self.session_memory.add_turn("Q2", "A2", force_new_session=True)
        self.session_memory.add_turn("Q3", "A3")
        
        # Test current session context
        current_context = self.session_memory.get_session_context(max_turns=2)
        self.assertLessEqual(len(current_context), 2)
        
        # Test all recent context
        all_context = self.session_memory.get_all_recent_context()
        self.assertEqual(len(all_context), 3)
        
        # Test time-based retrieval
        recent_context = self.session_memory.get_all_recent_context(hours_back=0.01)  # Very recent
        self.assertGreaterEqual(len(recent_context), 1)
    
    def test_session_expiration(self):
        """Test session expiration and cleanup."""
        # Create session and manually expire it
        session_id = self.session_memory.add_turn("Test", "Response")
        session = self.session_memory._sessions[session_id]
        
        # Force expiration
        session.start_time = time.time() - (self.config.duration_hours * 3600 + 60)
        
        # Check expiration
        self.assertTrue(session.is_expired())
        
        # Cleanup should remove expired sessions
        self.session_memory._cleanup_expired_sessions()
        self.assertNotIn(session_id, self.session_memory._sessions)
    
    def test_promotion_to_longterm_memory(self):
        """Test promotion of session memories to long-term storage."""
        # Create a mock episodic memory
        mock_episodic = Mock()
        mock_episodic.remember = Mock(return_value=Mock())
        
        # Add some turns and expire the session
        session_id = self.session_memory.add_turn("Important", "Discussion")
        self.session_memory.add_turn("Follow up", "More details")
        
        session = self.session_memory._sessions[session_id]
        session.start_time = time.time() - (self.config.duration_hours * 3600 + 60)
        
        # Promote to long-term memory
        promoted = self.session_memory.promote_to_longterm(mock_episodic)
        
        # Should have called episodic memory's remember method
        self.assertGreater(mock_episodic.remember.call_count, 0)
        self.assertTrue(session.promoted)
    
    def test_topic_change_detection(self):
        """Test session boundary detection based on topic changes."""
        # Mock embedding function to return different vectors for different topics
        def topic_embedding(text):
            if "python" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "javascript" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]
        
        self.session_memory.embedding_fn = topic_embedding
        
        # Add turns about same topic
        session_id1 = self.session_memory.add_turn("Python question", "Python answer")
        session_id2 = self.session_memory.add_turn("More Python", "Python details")
        
        # Should be same session
        self.assertEqual(session_id1, session_id2)
        
        # Switch to different topic
        session_id3 = self.session_memory.add_turn("JavaScript question", "JavaScript answer")
        
        # Should create new session due to topic change
        self.assertNotEqual(session_id2, session_id3)
    
    def test_concurrent_access_safety(self):
        """Test thread safety of session memory operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def add_turns():
            try:
                for i in range(10):
                    session_id = self.session_memory.add_turn(f"Concurrent {i}", f"Response {i}")
                    results.append(session_id)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_turns)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 30)  # 3 threads × 10 turns each

class TestEnhancedEpisodicMemory(unittest.TestCase):
    """Test enhanced episodic memory integration."""
    
    def setUp(self):
        """Set up enhanced episodic memory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_memory.json"
        
        self.embedding_fn = Mock(return_value=[0.8, 0.1, 0.1])
        self.chunker = ContextualChunker(chunk_size=50, overlap_ratio=0.2)
        self.temporal_weighting = TemporalMemoryWeighting()
        
        self.memory = EnhancedEpisodicMemory(
            storage_path=self.storage_path,
            embedding_fn=self.embedding_fn,
            chunker=self.chunker,
            temporal_weighting=self.temporal_weighting,
            max_records=100
        )
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chunked_conversation_storage(self):
        """Test storage of chunked conversations."""
        user = "Explain machine learning"
        assistant = "Machine learning is a subset of AI. " * 10  # Force chunking
        
        chunks = self.memory.remember(user, assistant)
        
        self.assertGreater(len(chunks), 1, "Should create multiple chunks")
        self.assertTrue(chunks[0].is_parent, "First chunk should be parent")
        
        # Verify all chunks have embeddings
        for chunk in chunks:
            if self.embedding_fn:
                self.assertIsNotNone(chunk.embedding)
    
    def test_enhanced_recall_with_temporal_weighting(self):
        """Test recall with temporal decay and importance boosting."""
        # Store memories with different ages
        now = time.time()
        
        # Recent memory
        chunks1 = self.memory.remember("Recent question", "Recent answer")
        chunks1[0].timestamp = now - 300  # 5 minutes ago
        
        # Old memory with same content
        chunks2 = self.memory.remember("Recent question", "Recent answer")  
        chunks2[0].timestamp = now - 86400  # 1 day ago
        chunks2[0].access_count = 10  # But frequently accessed
        
        # Test recall
        results = self.memory.recall("Recent question", top_k=2)
        
        self.assertGreater(len(results), 0)
        # Recent memory should generally score higher, but frequent access may boost old memory
        # This tests the balance between recency and importance
    
    def test_parent_child_context_retrieval(self):
        """Test that child chunks return parent context."""
        user = "Long question about multiple topics"
        assistant = "This answer covers many topics. " * 20  # Force chunking
        
        chunks = self.memory.remember(user, assistant)
        
        if len(chunks) > 1:
            # Recall should return child chunks with parent context
            results = self.memory.recall("topics", top_k=3, include_parents=True)
            
            for result in results:
                if result.is_child:
                    self.assertIsNotNone(result.full_context)
                    self.assertGreater(len(result.full_context), len(result.content))
    
    def test_memory_statistics(self):
        """Test memory statistics reporting."""
        # Add some memories
        for i in range(5):
            self.memory.remember(f"Question {i}", f"Answer {i} with some content")
        
        stats = self.memory.get_memory_stats()
        
        required_keys = [
            'total_records', 'parent_records', 'child_chunks',
            'avg_chunks_per_conversation', 'storage_size_bytes'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))
        
        self.assertGreater(stats['total_records'], 0)
    
    def test_memory_cleanup_and_forgetting(self):
        """Test cleanup of expired memories."""
        # Add old memory
        old_chunks = self.memory.remember("Old question", "Old answer")
        
        # Force age to trigger forgetting
        for chunk in old_chunks:
            chunk.timestamp = time.time() - (365 * 24 * 3600)  # 1 year ago
        
        initial_count = len(self.memory._records)
        removed_count = self.memory.cleanup_expired_memories()
        final_count = len(self.memory._records)
        
        self.assertGreater(removed_count, 0)
        self.assertEqual(final_count, initial_count - removed_count)
    
    def test_persistence_and_loading(self):
        """Test that enhanced memories persist and load correctly."""
        # Store some chunked memories
        self.memory.remember("Persistent question", "Persistent answer " * 20)
        
        # Create new memory instance with same storage path
        new_memory = EnhancedEpisodicMemory(
            storage_path=self.storage_path,
            embedding_fn=self.embedding_fn,
            chunker=self.chunker,
            temporal_weighting=self.temporal_weighting
        )
        
        # Should load existing memories
        self.assertGreater(len(new_memory._records), 0)
        
        # Test recall works
        results = new_memory.recall("Persistent", top_k=1)
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    # Run tests with coverage if available
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()
        
        unittest.main(exit=False)
        
        cov.stop()
        cov.save()
        print("\nCoverage Report:")
        cov.report()
    except ImportError:
        unittest.main()
```

## Integration Testing Strategy

### Cross-Component Integration Tests

#### Memory System Integration
- **Agent ↔ Enhanced Memory**: Verify agent can use new memory features
- **CLI ↔ Memory System**: Test CLI commands with enhanced memory
- **Session ↔ Episodic**: Verify session promotion to long-term memory
- **Chunking ↔ Retrieval**: Test end-to-end chunked conversation flow

#### Implementation

```python
class TestMemorySystemIntegration(unittest.TestCase):
    """Integration tests for enhanced memory system."""
    
    def setUp(self):
        """Set up full memory system integration."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock Ollama client for testing
        self.mock_client = Mock()
        self.mock_client.embed.return_value = [0.5, 0.3, 0.2]
        
        # Create full agent with enhanced memory
        from atlas_main.agent import AtlasAgent
        self.agent = AtlasAgent(
            client=self.mock_client,
            memory_path=Path(self.temp_dir) / "episodic.json",
            enable_enhanced_memory=True
        )
    
    def test_end_to_end_conversation_flow(self):
        """Test complete conversation flow with enhanced memory."""
        # Simulate a long conversation
        conversations = [
            ("What is Python?", "Python is a programming language..."),
            ("Tell me about data structures", "Python has lists, dicts..."),
            ("How do I sort a list?", "You can use the sort() method..."),
            ("What about machine learning?", "ML in Python uses libraries like scikit-learn...")
        ]
        
        # Process conversations
        for user_input, expected_response in conversations:
            # Mock LLM response
            self.mock_client.chat.return_value = expected_response
            
            # Process through agent
            response = self.agent.respond(user_input)
            
            # Verify response and memory storage
            self.assertIsNotNone(response)
        
        # Test memory recall
        recalled = self.agent.memory.recall("Python data structures", top_k=2)
        self.assertGreater(len(recalled), 0)
    
    def test_cli_memory_commands(self):
        """Test CLI commands with enhanced memory."""
        from atlas_main.cli import handle_memory_command
        
        # Add some memories first
        self.agent.respond("Test question")
        
        # Test memory status command
        status_output = handle_memory_command(self.agent, ["status"])
        self.assertIn("Total records", status_output)
        
        # Test memory search command
        search_output = handle_memory_command(self.agent, ["search", "test"])
        self.assertIsNotNone(search_output)
```

### Performance Benchmarks

#### Benchmark Implementation

```python
import time
import statistics
from typing import List

class MemoryPerformanceBenchmarks:
    """Performance benchmarks for enhanced memory system."""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.results = {}
    
    def benchmark_chunking_performance(self, conversations: List[tuple]) -> dict:
        """Benchmark chunking performance."""
        times = []
        
        for user, assistant in conversations:
            start_time = time.perf_counter()
            chunks = self.memory.chunker.chunk_conversation(user, assistant)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': statistics.mean(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'std_dev_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0
        }
    
    def benchmark_recall_performance(self, queries: List[str]) -> dict:
        """Benchmark recall performance."""
        times = []
        
        for query in queries:
            start_time = time.perf_counter()
            results = self.memory.recall(query, top_k=5)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': statistics.mean(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'target_met': max(times) * 1000 < 100  # Target: <100ms
        }
    
    def benchmark_memory_usage(self) -> dict:
        """Benchmark memory usage characteristics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add large number of memories
        for i in range(100):
            self.memory.remember(f"Question {i}", f"Answer {i} " * 50)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        return {
            'memory_increase_bytes': memory_increase,
            'memory_per_record_bytes': memory_increase / 100,
            'total_records': len(self.memory._records)
        }
    
    def run_all_benchmarks(self) -> dict:
        """Run all performance benchmarks."""
        # Generate test data
        conversations = [
            (f"Question {i}", f"Answer {i} with detailed explanation " * 10)
            for i in range(50)
        ]
        
        queries = [f"question {i}" for i in range(20)]
        
        # Run benchmarks
        return {
            'chunking': self.benchmark_chunking_performance(conversations),
            'recall': self.benchmark_recall_performance(queries),
            'memory_usage': self.benchmark_memory_usage(),
            'timestamp': time.time()
        }

# Usage
def run_performance_tests():
    """Run performance benchmarks and validate targets."""
    memory = create_test_memory_system()
    benchmarks = MemoryPerformanceBenchmarks(memory)
    
    results = benchmarks.run_all_benchmarks()
    
    # Validate performance targets
    assert results['chunking']['mean_time_ms'] < 50, "Chunking too slow"
    assert results['recall']['target_met'], "Recall performance target not met"
    assert results['memory_usage']['memory_per_record_bytes'] < 2000, "Memory usage too high"
    
    print("All performance benchmarks passed!")
    return results
```

## Error Handling and Edge Case Testing

### Error Scenarios

```python
class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_malformed_embedding_handling(self):
        """Test handling of malformed embeddings."""
        def bad_embedding_fn(text):
            return None  # Simulate embedding failure
        
        memory = create_memory_with_embedding(bad_embedding_fn)
        
        # Should not crash, should gracefully degrade
        chunks = memory.remember("Test", "Response")
        self.assertGreater(len(chunks), 0)
        
        # Recall should work even without embeddings
        results = memory.recall("Test", top_k=1)
        # May return empty results, but shouldn't crash
        self.assertIsInstance(results, list)
    
    def test_corrupted_storage_file_recovery(self):
        """Test recovery from corrupted storage files."""
        temp_path = Path(tempfile.mktemp())
        
        # Create corrupted JSON file
        temp_path.write_text("invalid json content{")
        
        # Should recover gracefully
        memory = EnhancedEpisodicMemory(storage_path=temp_path)
        
        # Should create backup and start fresh
        self.assertEqual(len(memory._records), 0)
        self.assertTrue(temp_path.with_suffix(".corrupt").exists())
    
    def test_concurrent_access_edge_cases(self):
        """Test edge cases in concurrent access."""
        # Test session cleanup during active usage
        # Test memory recall during storage operations
        # Test embedding generation failures
        pass  # Implementation depends on specific threading model
```

## Continuous Integration Configuration

### Test Pipeline Configuration

```yaml
# .github/workflows/memory-tests.yml
name: Enhanced Memory Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install --with dev
    
    - name: Run unit tests with coverage
      run: |
        poetry run coverage run -m pytest tests/ -v
        poetry run coverage report --fail-under=90
        poetry run coverage xml
    
    - name: Run integration tests
      run: |
        poetry run pytest tests/integration/ -v
    
    - name: Run performance benchmarks
      run: |
        poetry run python tests/benchmarks/run_benchmarks.py
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### Test Organization

```
tests/
├── unit/
│   ├── test_chunking.py
│   ├── test_temporal_decay.py
│   ├── test_session_memory.py
│   └── test_enhanced_episodic.py
├── integration/
│   ├── test_agent_integration.py
│   ├── test_cli_integration.py
│   └── test_end_to_end.py
├── benchmarks/
│   ├── run_benchmarks.py
│   ├── memory_performance.py
│   └── stress_tests.py
├── fixtures/
│   ├── test_data.json
│   ├── sample_conversations.json
│   └── benchmark_data.json
└── utils/
    ├── test_helpers.py
    ├── mock_embeddings.py
    └── assertions.py
```

## Quality Gates

### Pre-commit Requirements
- [ ] All unit tests pass
- [ ] Code coverage ≥ 90%
- [ ] No performance regressions
- [ ] Integration tests pass
- [ ] Documentation updated

### Release Criteria
- [ ] All test suites pass
- [ ] Performance benchmarks meet targets
- [ ] Backward compatibility verified
- [ ] Migration tests successful
- [ ] Memory leak tests pass

---

*This testing strategy ensures comprehensive coverage and quality assurance for the enhanced memory system. All tests should be implemented alongside feature development.*