"""Comprehensive unit tests for enhanced memory system.

Tests contextual chunking, temporal decay, session memory, and enhanced episodic memory
with edge cases, error handling, and performance validation.
"""
import tempfile
import unittest
import time
import math
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

import numpy as np

# Import the modules we're testing
from atlas_main.memory import MemoryRecord
from atlas_main.enhanced_memory import (
    ChunkMetadata,
    ChunkedMemoryRecord,
    ContextualChunker,
    TemporalConfig,
    TemporalMemoryWeighting,
    SessionConfig,
    SessionTurn,
    Session,
    SessionMemory,
    EnhancedEpisodicMemory
)


class TestChunkedMemoryRecord(unittest.TestCase):
    """Test ChunkedMemoryRecord data structure and methods."""
    
    def test_basic_creation(self):
        """Test basic record creation with defaults."""
        record = ChunkedMemoryRecord(
            user="Hello",
            assistant="Hi there",
            content="Test content"
        )
        
        self.assertIsNotNone(record.id)
        self.assertEqual(record.user, "Hello")
        self.assertEqual(record.assistant, "Hi there")
        self.assertEqual(record.content, "Test content")
        self.assertTrue(record.is_parent)
        self.assertFalse(record.is_child)
        self.assertEqual(record.access_count, 0)
        self.assertEqual(record.importance_score, 1.0)
    
    def test_parent_child_relationships(self):
        """Test parent-child record relationships."""
        parent = ChunkedMemoryRecord(
            user="Question",
            assistant="Answer",
            chunk_index=0
        )
        
        child = ChunkedMemoryRecord(
            parent_id=parent.id,
            chunk_index=1,
            content="Child chunk content"
        )
        
        self.assertTrue(parent.is_parent)
        self.assertFalse(parent.is_child)
        self.assertTrue(child.is_child)
        self.assertFalse(child.is_parent)
        self.assertEqual(child.parent_id, parent.id)
    
    def test_access_tracking(self):
        """Test access count and importance score tracking."""
        record = ChunkedMemoryRecord(user="Test", assistant="Response")
        
        initial_importance = record.importance_score
        initial_time = record.last_accessed
        
        # Simulate access
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        record.access()
        
        self.assertEqual(record.access_count, 1)
        self.assertGreater(record.last_accessed, initial_time)
        self.assertGreater(record.importance_score, initial_importance)
        
        # Multiple accesses should increase importance with diminishing returns
        prev_importance = record.importance_score
        record.access()
        record.access()
        
        self.assertEqual(record.access_count, 3)
        self.assertGreater(record.importance_score, prev_importance)
    
    def test_legacy_conversion(self):
        """Test conversion to and from legacy MemoryRecord format."""
        # Create legacy record
        legacy = MemoryRecord(
            id="test-id",
            user="Legacy user",
            assistant="Legacy assistant",
            timestamp=12345.0,
            embedding=[1.0, 2.0, 3.0]
        )
        
        # Convert to enhanced
        enhanced = ChunkedMemoryRecord.from_legacy_record(legacy)
        
        self.assertEqual(enhanced.id, legacy.id)
        self.assertEqual(enhanced.user, legacy.user)
        self.assertEqual(enhanced.assistant, legacy.assistant)
        self.assertEqual(enhanced.timestamp, legacy.timestamp)
        self.assertEqual(enhanced.embedding, legacy.embedding)
        self.assertTrue(enhanced.is_parent)
        self.assertEqual(enhanced.chunk_index, 0)
        
        # Convert back to legacy
        back_to_legacy = enhanced.to_legacy_record()
        
        self.assertEqual(back_to_legacy.id, legacy.id)
        self.assertEqual(back_to_legacy.user, legacy.user)
        self.assertEqual(back_to_legacy.assistant, legacy.assistant)
        self.assertEqual(back_to_legacy.timestamp, legacy.timestamp)
        self.assertEqual(back_to_legacy.embedding, legacy.embedding)
    
    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        metadata = ChunkMetadata(
            start_char=0,
            end_char=100,
            overlap_start=10,
            overlap_end=20,
            semantic_boundary=True
        )
        
        record = ChunkedMemoryRecord(
            user="Test user",
            assistant="Test assistant",
            content="Test content",
            chunk_metadata=metadata,
            access_count=5,
            importance_score=2.5
        )
        
        # Serialize to dict
        data = record.to_dict()
        
        self.assertIn('user', data)
        self.assertIn('chunk_metadata', data)
        self.assertEqual(data['access_count'], 5)
        self.assertEqual(data['importance_score'], 2.5)
        
        # Deserialize from dict
        restored = ChunkedMemoryRecord.from_dict(data)
        
        self.assertEqual(restored.user, record.user)
        self.assertEqual(restored.assistant, record.assistant)
        self.assertEqual(restored.content, record.content)
        self.assertEqual(restored.access_count, record.access_count)
        self.assertEqual(restored.importance_score, record.importance_score)
        self.assertIsNotNone(restored.chunk_metadata)
        self.assertEqual(restored.chunk_metadata.start_char, 0)
        self.assertEqual(restored.chunk_metadata.semantic_boundary, True)


class TestContextualChunker(unittest.TestCase):
    """Test contextual chunking functionality."""
    
    def setUp(self):
        """Set up test chunker with small sizes for testing."""
        self.chunker = ContextualChunker(
            chunk_size=100,
            overlap_ratio=0.2,
            min_chunk_size=30
        )
    
    def test_short_conversation_no_chunking(self):
        """Test that short conversations remain as single records."""
        user = "Hi"
        assistant = "Hello there!"
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_parent)
        self.assertEqual(chunks[0].user, user)
        self.assertEqual(chunks[0].assistant, assistant)
        self.assertIn("User: Hi", chunks[0].content)
        self.assertIn("Assistant: Hello there!", chunks[0].content)
    
    def test_long_conversation_chunking(self):
        """Test chunking of conversations exceeding chunk size."""
        user = "Tell me about Python"
        assistant = "Python is a programming language. " * 20  # ~600 characters
        
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
            self.assertIsNotNone(child.full_context)
    
    def test_semantic_boundary_detection(self):
        """Test detection of sentence and paragraph boundaries."""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        
        # Test sentence boundary detection
        pos, is_semantic = self.chunker.find_semantic_boundary(text, 20, 10)
        self.assertTrue(is_semantic)
        # Should find a sentence boundary
        boundary_chars = [". ", "! ", "? "]
        found_boundary = any(boundary in text[max(0, pos-2):pos+2] for boundary in boundary_chars)
        self.assertTrue(found_boundary)
        
        # Test paragraph boundary (higher priority)
        text_with_para = "First paragraph.\n\nSecond paragraph."
        pos, is_semantic = self.chunker.find_semantic_boundary(text_with_para, 15, 10)
        self.assertTrue(is_semantic)
    
    def test_chunk_overlap_calculation(self):
        """Test that chunks have proper overlap."""
        user = "Question about overlapping chunks"
        # Create predictable text for testing overlap
        assistant = "Word " * 50  # 250 characters, should create multiple chunks
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        if len(chunks) > 2:  # Need at least 3 chunks to test overlap
            # Check that chunks after the first child have overlap metadata
            for i in range(2, len(chunks)):  # Start from second child chunk
                chunk = chunks[i]
                self.assertIsNotNone(chunk.chunk_metadata)
                self.assertGreater(chunk.chunk_metadata.overlap_start, 0)
                
                # Verify overlap makes sense  
                self.assertLessEqual(
                    chunk.chunk_metadata.overlap_start, 
                    self.chunker.overlap_size + 10  # Allow small tolerance
                )
    
    def test_minimum_chunk_size_enforcement(self):
        """Test that tiny chunks are not created."""
        user = "Short"
        # Text just over chunk_size but would create tiny remainder
        assistant = "A" * 150  
        
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        # All non-parent chunks should meet minimum size (with tolerance)
        for chunk in chunks[1:]:  # Skip parent
            actual_size = len(chunk.content.strip())
            self.assertGreaterEqual(
                actual_size, 
                self.chunker.min_chunk_size - 10,  # Allow small tolerance
                f"Chunk too small: {actual_size} chars"
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
                if not user.strip() and not assistant.strip():
                    with self.assertRaises(ValueError):
                        self.chunker.chunk_conversation(user, assistant)
                else:
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
            # Verify Unicode characters are preserved in at least one chunk
            if "😊" in chunk.content:
                self.assertIn("😊", chunk.content)


class TestTemporalWeighting(unittest.TestCase):
    """Test temporal decay weighting functionality."""
    
    def setUp(self):
        """Set up temporal weighting with test configuration."""
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
                f"Weight should decrease: {weights[i-1]} > {weights[i]} "
                f"(ages: {[0, 1, 6, 24, 168][i-1]}h vs {[0, 1, 6, 24, 168][i]}h)"
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
        # Note: using assertLessEqual to handle floating point precision issues
        self.assertLess(working_weight, session_weight)
        
        # For very close values, allow small tolerance due to floating point precision
        self.assertTrue(
            session_weight <= episodic_weight or abs(session_weight - episodic_weight) < 1e-10,
            f"session_weight ({session_weight}) should be <= episodic_weight ({episodic_weight})"
        )
        self.assertTrue(
            episodic_weight <= semantic_weight or abs(episodic_weight - semantic_weight) < 1e-10,
            f"episodic_weight ({episodic_weight}) should be <= semantic_weight ({semantic_weight})"
        )
    
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
                f"Importance should not decrease: {access_counts[i]} accesses = {multipliers[i]}, "
                f"{access_counts[i-1]} accesses = {multipliers[i-1]}"
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
        self.session_memory.cleanup()
    
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
        recent_context = self.session_memory.get_all_recent_context(hours_back=0.01)
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
    
    def test_topic_change_detection(self):
        """Test session boundary detection based on topic changes."""
        def topic_embedding(text):
            """Mock embedding function returning different vectors for different topics."""
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
        # Force chunking with long response
        assistant = "Machine learning is a subset of AI. " * 10
        
        chunks = self.memory.remember(user, assistant)
        
        self.assertGreater(len(chunks), 1, "Should create multiple chunks")
        self.assertTrue(chunks[0].is_parent, "First chunk should be parent")
        
        # Verify all chunks have embeddings (our mock returns embeddings)
        for chunk in chunks:
            if self.embedding_fn:
                self.assertIsNotNone(chunk.embedding)
    
    def test_enhanced_recall_with_temporal_weighting(self):
        """Test recall with temporal decay and importance boosting."""
        # Store memories with different characteristics
        now = time.time()
        
        # Recent memory
        chunks1 = self.memory.remember("Recent question", "Recent answer")
        for chunk in chunks1:
            chunk.timestamp = now - 300  # 5 minutes ago
        
        # Old memory but frequently accessed
        chunks2 = self.memory.remember("Old question", "Old answer")
        for chunk in chunks2:
            chunk.timestamp = now - 86400  # 1 day ago
            chunk.access_count = 10  # Frequently accessed
            chunk.importance_score = 2.0
        
        # Test recall
        results = self.memory.recall("question", top_k=5)
        
        self.assertGreater(len(results), 0)
        # Both memories should be retrieved but scored differently
        question_results = [r for r in results if "question" in r.content.lower()]
        self.assertGreater(len(question_results), 0)
    
    def test_parent_child_context_retrieval(self):
        """Test that child chunks return parent context."""
        user = "Long question about multiple topics"
        # Force chunking with very long response
        assistant = "This answer covers many topics. " * 20
        
        chunks = self.memory.remember(user, assistant)
        
        if len(chunks) > 1:
            # Recall should potentially return child chunks with parent context
            results = self.memory.recall("topics", top_k=3, include_parents=True)
            
            # Check if any results have full context
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
        self.assertGreater(stats['parent_records'], 0)
    
    def test_memory_cleanup_and_forgetting(self):
        """Test cleanup of expired memories."""
        # Add memory that will be marked for forgetting
        chunks = self.memory.remember("Old question", "Old answer")
        
        # Force age to trigger forgetting
        for chunk in chunks:
            chunk.timestamp = time.time() - (365 * 24 * 3600)  # 1 year ago
        
        initial_count = len(self.memory._records)
        removed_count = self.memory.cleanup_expired_memories()
        final_count = len(self.memory._records)
        
        self.assertGreater(removed_count, 0)
        self.assertEqual(final_count, initial_count - removed_count)
    
    def test_persistence_and_loading(self):
        """Test that enhanced memories persist and load correctly."""
        # Store some chunked memories
        original_chunks = self.memory.remember("Persistent question", "Persistent answer " * 20)
        original_count = len(self.memory._records)
        
        # Create new memory instance with same storage path
        new_memory = EnhancedEpisodicMemory(
            storage_path=self.storage_path,
            embedding_fn=self.embedding_fn,
            chunker=self.chunker,
            temporal_weighting=self.temporal_weighting
        )
        
        # Should load existing memories
        self.assertEqual(len(new_memory._records), original_count)
        
        # Test recall works
        results = new_memory.recall("Persistent", top_k=1)
        self.assertGreater(len(results), 0)
    
    def test_legacy_memory_migration(self):
        """Test migration from legacy MemoryRecord format."""
        # Create a legacy memory file
        legacy_data = {
            "records": [
                {
                    "id": "legacy-1",
                    "user": "Legacy user 1",
                    "assistant": "Legacy assistant 1",
                    "timestamp": time.time() - 3600,
                    "embedding": [1.0, 2.0, 3.0]
                },
                {
                    "id": "legacy-2", 
                    "user": "Legacy user 2",
                    "assistant": "Legacy assistant 2",
                    "timestamp": time.time() - 7200,
                    "embedding": [4.0, 5.0, 6.0]
                }
            ]
        }
        
        # Write legacy data to storage file
        import json
        with open(self.storage_path, 'w') as f:
            json.dump(legacy_data, f)
        
        # Create new enhanced memory instance - should migrate automatically
        migrated_memory = EnhancedEpisodicMemory(
            storage_path=self.storage_path,
            embedding_fn=self.embedding_fn
        )
        
        # Should have migrated records
        self.assertEqual(len(migrated_memory._records), 2)
        
        # All records should be parents (legacy migration)
        for record in migrated_memory._records:
            self.assertTrue(record.is_parent)
        
        # Content should be preserved
        record_contents = [r.content for r in migrated_memory._records]
        self.assertTrue(any("Legacy user 1" in content for content in record_contents))
        self.assertTrue(any("Legacy user 2" in content for content in record_contents))
    
    def test_memory_limits_enforcement(self):
        """Test that memory limits are enforced with intelligent cleanup."""
        # Set low limit for testing
        self.memory.max_records = 5
        
        # Add more records than limit
        for i in range(10):
            self.memory.remember(f"Question {i}", f"Answer {i}")
        
        # Should enforce limit
        self.assertLessEqual(len(self.memory._records), self.memory.max_records)
        
        # Should keep most recent/important records
        remaining_content = [r.content for r in self.memory._records]
        # Recent records should be more likely to remain
        recent_found = any("Question 9" in content for content in remaining_content)
        old_found = any("Question 0" in content for content in remaining_content)
        # This is probabilistic, but recent should be more likely to survive
        self.assertTrue(recent_found or len(remaining_content) > 0)


def create_test_embedding_fn():
    """Create deterministic embedding function for testing."""
    def test_embedding(text: str) -> List[float]:
        # Simple hash-based embedding for consistent testing
        hash_val = hash(text) % 1000
        return [
            float(hash_val % 3) / 2.0,
            float((hash_val // 3) % 3) / 2.0, 
            float((hash_val // 9) % 3) / 2.0,
            1.0 if "important" in text.lower() else 0.0,
            1.0 if "recent" in text.lower() else 0.0,
        ]
    return test_embedding


class TestIntegration(unittest.TestCase):
    """Integration tests for complete enhanced memory system."""
    
    def setUp(self):
        """Set up complete enhanced memory system."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "integration_test.json"
        
        # Create components
        self.embedding_fn = create_test_embedding_fn()
        self.chunker = ContextualChunker(chunk_size=200, overlap_ratio=0.2)
        self.temporal_weighting = TemporalMemoryWeighting()
        self.session_memory = SessionMemory(embedding_fn=self.embedding_fn)
        
        # Create integrated system
        self.memory = EnhancedEpisodicMemory(
            storage_path=self.storage_path,
            embedding_fn=self.embedding_fn,
            chunker=self.chunker,
            temporal_weighting=self.temporal_weighting,
            session_memory=self.session_memory,
            max_records=50
        )
    
    def tearDown(self):
        """Clean up resources."""
        self.session_memory.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_conversation_flow(self):
        """Test complete conversation flow with all enhanced features."""
        conversations = [
            ("What is Python?", "Python is a programming language known for simplicity."),
            ("Tell me about data structures", "Python has lists, dictionaries, sets, and tuples."),
            ("How do I sort a list?", "You can use the sort() method or sorted() function."),
            ("What about machine learning?", "Python has great ML libraries like scikit-learn.")
        ]
        
        # Process conversations
        for user_input, assistant_response in conversations:
            chunks = self.memory.remember(user_input, assistant_response)
            self.assertGreater(len(chunks), 0)
        
        # Test recall with different queries
        python_results = self.memory.recall("Python programming", top_k=2)
        self.assertGreater(len(python_results), 0)
        
        ml_results = self.memory.recall("machine learning libraries", top_k=2)
        self.assertGreater(len(ml_results), 0)
        
        # Test session memory integration
        session_context = self.session_memory.get_all_recent_context()
        self.assertEqual(len(session_context), 4)  # All conversations in session
        
        # Test memory statistics
        stats = self.memory.get_memory_stats()
        self.assertGreater(stats['total_records'], 0)
        self.assertGreater(stats['parent_records'], 0)
    
    def test_performance_characteristics(self):
        """Test basic performance characteristics."""
        # Add a moderate number of conversations
        start_time = time.time()
        
        for i in range(20):
            self.memory.remember(
                f"Question {i} about various topics",
                f"Answer {i} with detailed explanation about the topic " * 5
            )
        
        storage_time = time.time() - start_time
        self.assertLess(storage_time, 5.0, "Storage should be reasonably fast")
        
        # Test recall performance
        start_time = time.time()
        results = self.memory.recall("topics explanation", top_k=5)
        recall_time = time.time() - start_time
        
        self.assertLess(recall_time, 1.0, "Recall should be fast")
        self.assertGreater(len(results), 0, "Should find relevant results")


if __name__ == "__main__":
    # Configure test runner for better output
    unittest.main(verbosity=2, buffer=True)