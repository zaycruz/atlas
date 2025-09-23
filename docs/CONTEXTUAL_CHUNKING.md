# Contextual Chunking Implementation Guide

## Overview

Contextual chunking addresses the fundamental limitation of embedding entire conversation turns by splitting long interactions into semantically coherent chunks while preserving context boundaries. This approach significantly improves retrieval accuracy for specific information while maintaining broader conversational context.

## Problem Statement

### Current System Limitations
1. **Coarse Granularity**: Entire conversation turns are embedded as single units
2. **Context Loss**: Specific details buried within long responses are hard to retrieve
3. **Poor Precision**: Similarity search returns entire conversations instead of relevant segments
4. **Memory Inefficiency**: Large embeddings for mostly irrelevant content

### Example Scenario
```
User: "Tell me about Python data structures and also help me debug this SQL query..."
Assistant: [Long response covering both Python and SQL topics]

Current Problem: Searching for "Python lists" later retrieves the entire response
Desired Behavior: Retrieve only the Python-specific segment with surrounding context
```

## Solution: Parent-Child Chunking with Contextual Overlap

Based on research from:
- **Parent-Child Chunking**: [Parent-Child Chunking in LangChain for Advanced RAG](https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a) (2025)
- **Contextual Retrieval**: [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Anthropic's approach reducing failed retrievals by 49%
- **Hierarchical Memory**: [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)
- **Sliding Window Chunking**: [Advanced RAG — Sentence Window Retrieval](https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/) (2025)

### Architecture Design

```
Original Conversation Turn
├── Parent Record (full context)
│   ├── Chunk 1 (characters 0-400 + overlap)
│   ├── Chunk 2 (characters 320-720 + overlap)  ← 20% overlap
│   ├── Chunk 3 (characters 640-1040 + overlap)
│   └── Chunk N (final segment)
└── Metadata (timestamps, relationships)
```

### Key Principles

1. **Semantic Boundaries**: Split at sentence/paragraph boundaries when possible
2. **Contextual Overlap**: 20% overlap between adjacent chunks to preserve context
3. **Parent Preservation**: Keep full conversation for broader context retrieval
4. **Hierarchical Storage**: Both chunks and parents are searchable

## Implementation Details

### Core Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Optional
import time
import uuid

@dataclass
class ChunkMetadata:
    """Metadata for chunk positioning and overlap."""
    start_char: int          # Starting character position in parent
    end_char: int           # Ending character position in parent  
    overlap_start: int      # Characters of overlap at beginning
    overlap_end: int        # Characters of overlap at end
    semantic_boundary: bool # True if chunk ends at sentence/paragraph boundary

@dataclass
class ChunkedMemoryRecord:
    """Enhanced memory record supporting hierarchical chunking."""
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None  # None for parent records
    chunk_index: int = 0             # 0 for parent, 1+ for children
    
    # Content fields
    user: str = ""
    assistant: str = ""
    content: str = ""               # Chunk content (for children) or full content (for parent)
    full_context: Optional[str] = None  # Full parent context for reference
    
    # Memory fields
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    
    # Chunking metadata
    chunk_metadata: Optional[ChunkMetadata] = None
    
    # Enhanced tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance_score: float = 1.0
    
    @property
    def is_parent(self) -> bool:
        """Check if this is a parent record."""
        return self.parent_id is None and self.chunk_index == 0
    
    @property
    def is_child(self) -> bool:
        """Check if this is a child chunk."""
        return self.parent_id is not None and self.chunk_index > 0
    
    def access(self) -> None:
        """Record memory access for reinforcement learning."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Logarithmic importance boost to prevent runaway growth
        self.importance_score *= (1.0 + 0.1 / math.log(self.access_count + 1))
```

### Chunking Algorithm

```python
import re
from typing import Tuple

class ContextualChunker:
    """Intelligent text chunking with semantic boundary detection."""
    
    def __init__(
        self, 
        chunk_size: int = 400,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary patterns (ordered by preference)
        self.sentence_patterns = [
            r'\.[\s\n]+',           # Period followed by whitespace
            r'[!?][\s\n]+',         # Exclamation/question + whitespace
            r';\s+',                # Semicolon + space
            r':\s+',                # Colon + space (lower priority)
        ]
        
        # Paragraph boundary patterns
        self.paragraph_patterns = [
            r'\n\s*\n',             # Double newline (paragraph break)
            r'\n\s*[-*]\s+',        # Newline + bullet point
            r'\n\s*\d+\.\s+',       # Newline + numbered list
        ]
    
    def find_semantic_boundary(self, text: str, ideal_pos: int, search_window: int = 50) -> Tuple[int, bool]:
        """Find the best semantic boundary near the ideal position.
        
        Args:
            text: Text to search within
            ideal_pos: Ideal character position for boundary
            search_window: Characters to search before/after ideal position
            
        Returns:
            Tuple of (boundary_position, is_semantic_boundary)
        """
        start_search = max(0, ideal_pos - search_window)
        end_search = min(len(text), ideal_pos + search_window)
        search_text = text[start_search:end_search]
        
        # Try paragraph boundaries first (highest priority)
        for pattern in self.paragraph_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find match closest to ideal position
                best_match = min(matches, key=lambda m: abs(m.end() + start_search - ideal_pos))
                return start_search + best_match.end(), True
        
        # Try sentence boundaries
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                best_match = min(matches, key=lambda m: abs(m.end() + start_search - ideal_pos))
                return start_search + best_match.end(), True
        
        # Fallback to word boundary
        word_boundaries = list(re.finditer(r'\s+', search_text))
        if word_boundaries:
            best_match = min(word_boundaries, key=lambda m: abs(m.start() + start_search - ideal_pos))
            return start_search + best_match.start(), False
        
        # Last resort: exact position
        return ideal_pos, False
    
    def chunk_conversation(self, user_text: str, assistant_text: str) -> List[ChunkedMemoryRecord]:
        """Split a conversation turn into contextual chunks.
        
        Args:
            user_text: User input text
            assistant_text: Assistant response text
            
        Returns:
            List of ChunkedMemoryRecord objects (parent + children)
        """
        # Combine texts for chunking analysis
        full_conversation = f"User: {user_text}\nAssistant: {assistant_text}"
        
        # If conversation is short enough, return single parent record
        if len(full_conversation) <= self.chunk_size:
            parent = ChunkedMemoryRecord(
                user=user_text,
                assistant=assistant_text,
                content=full_conversation,
                chunk_index=0
            )
            return [parent]
        
        # Create parent record
        parent_id = str(uuid.uuid4())
        parent = ChunkedMemoryRecord(
            id=parent_id,
            user=user_text,
            assistant=assistant_text,
            content=full_conversation,
            chunk_index=0
        )
        
        # Generate child chunks
        chunks = [parent]
        current_pos = 0
        chunk_index = 1
        
        while current_pos < len(full_conversation):
            # Calculate ideal chunk end position
            ideal_end = current_pos + self.chunk_size
            
            if ideal_end >= len(full_conversation):
                # Final chunk - take remaining text
                chunk_end = len(full_conversation)
                semantic_boundary = True
            else:
                # Find semantic boundary near ideal position
                chunk_end, semantic_boundary = self.find_semantic_boundary(
                    full_conversation, ideal_end
                )
            
            # Calculate overlap with previous chunk
            overlap_start = max(0, current_pos - self.overlap_size) if chunk_index > 1 else 0
            actual_start = current_pos - overlap_start
            
            # Extract chunk content with overlap
            chunk_content = full_conversation[actual_start:chunk_end]
            
            # Ensure minimum chunk size (skip tiny chunks)
            if len(chunk_content.strip()) < self.min_chunk_size and chunk_index > 1:
                break
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                start_char=actual_start,
                end_char=chunk_end,
                overlap_start=overlap_start,
                overlap_end=0,  # Calculated for next chunk
                semantic_boundary=semantic_boundary
            )
            
            # Create child chunk record
            child_chunk = ChunkedMemoryRecord(
                parent_id=parent_id,
                chunk_index=chunk_index,
                content=chunk_content,
                full_context=full_conversation,
                chunk_metadata=metadata
            )
            
            chunks.append(child_chunk)
            
            # Move to next chunk position
            current_pos = chunk_end - self.overlap_size
            chunk_index += 1
        
        # Update overlap_end for all chunks except the last
        for i in range(len(chunks) - 1):
            if chunks[i].chunk_metadata:
                next_start = chunks[i + 1].chunk_metadata.start_char if i + 1 < len(chunks) else 0
                chunks[i].chunk_metadata.overlap_end = max(0, chunks[i].chunk_metadata.end_char - next_start)
        
        return chunks

# Usage example
chunker = ContextualChunker(chunk_size=400, overlap_ratio=0.2)
chunks = chunker.chunk_conversation(
    user_text="Tell me about Python data structures",
    assistant_text="Python offers several built-in data structures..."
)
```

### Enhanced Episodic Memory Integration

```python
class EnhancedEpisodicMemory(EpisodicMemory):
    """Extended episodic memory with contextual chunking support."""
    
    def __init__(self, *args, chunker: Optional[ContextualChunker] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunker = chunker or ContextualChunker()
        self._chunk_index: Dict[str, List[str]] = {}  # parent_id -> child_ids
    
    def remember(self, user: str, assistant: str) -> List[ChunkedMemoryRecord]:
        """Store conversation with contextual chunking.
        
        Returns:
            List of stored memory records (parent + children)
        """
        # Generate chunks
        chunks = self.chunker.chunk_conversation(user, assistant)
        
        # Generate embeddings for all chunks
        for chunk in chunks:
            if self.embedding_fn:
                try:
                    # Use appropriate content for embedding
                    embed_text = chunk.content
                    if chunk.is_parent:
                        # For parent, use the full conversation
                        embed_text = f"User: {chunk.user}\nAssistant: {chunk.assistant}"
                    
                    embedding = self.embedding_fn(embed_text)
                    if embedding:
                        chunk.embedding = list(embedding)
                except Exception:
                    # Graceful degradation if embedding fails
                    pass
        
        # Store all chunks
        parent_id = chunks[0].id
        child_ids = [chunk.id for chunk in chunks[1:]]
        self._chunk_index[parent_id] = child_ids
        
        # Add to memory store
        self._records.extend(chunks)
        
        # Enforce memory limits (remove oldest if necessary)
        self._enforce_memory_limits()
        
        # Persist to disk
        self._save()
        
        return chunks
    
    def recall(self, query: str, *, top_k: int = 4, include_parents: bool = True) -> List[ChunkedMemoryRecord]:
        """Enhanced recall with chunk-aware retrieval.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            include_parents: Whether to return parent context for child chunks
            
        Returns:
            List of relevant memory records with context
        """
        if not self.embedding_fn or not self._records:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_fn(query)
        if not query_embedding:
            return []
        
        query_vec = np.asarray(query_embedding, dtype=float)
        scored_records = []
        
        # Score all records (both parents and children)
        for record in self._records:
            if not record.embedding:
                continue
                
            candidate_vec = np.asarray(record.embedding, dtype=float)
            if candidate_vec.shape != query_vec.shape:
                continue
            
            # Calculate base similarity
            denom = np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
            if denom == 0:
                continue
            
            similarity = float(np.dot(query_vec, candidate_vec) / denom)
            if math.isnan(similarity):
                continue
            
            # Apply temporal decay weighting
            temporal_weight = self._calculate_temporal_weight(record.timestamp)
            
            # Apply importance boosting
            importance_weight = math.log(record.importance_score + 1)
            
            # Combined score
            final_score = similarity * temporal_weight * importance_weight
            scored_records.append((final_score, record))
        
        # Sort by score and take top results
        scored_records.sort(key=lambda x: x[0], reverse=True)
        top_records = [record for _, record in scored_records[:top_k]]
        
        # Enhance results with parent context if needed
        if include_parents:
            enhanced_records = []
            for record in top_records:
                if record.is_child and record.parent_id:
                    # Find parent record
                    parent = self._find_record_by_id(record.parent_id)
                    if parent:
                        # Create enhanced record with full context
                        enhanced = ChunkedMemoryRecord(
                            id=record.id,
                            parent_id=record.parent_id,
                            chunk_index=record.chunk_index,
                            content=record.content,
                            full_context=parent.content,  # Include full parent context
                            timestamp=record.timestamp,
                            embedding=record.embedding,
                            chunk_metadata=record.chunk_metadata,
                            access_count=record.access_count,
                            last_accessed=record.last_accessed,
                            importance_score=record.importance_score
                        )
                        enhanced_records.append(enhanced)
                    else:
                        enhanced_records.append(record)
                else:
                    enhanced_records.append(record)
            top_records = enhanced_records
        
        # Record access for reinforcement learning
        for record in top_records:
            record.access()
        
        return top_records
    
    def _calculate_temporal_weight(self, timestamp: float, decay_rate: float = 0.1) -> float:
        """Calculate temporal decay weight for memory relevance."""
        age_hours = (time.time() - timestamp) / 3600
        return math.exp(-decay_rate * age_hours)
    
    def _find_record_by_id(self, record_id: str) -> Optional[ChunkedMemoryRecord]:
        """Find a record by its ID."""
        for record in self._records:
            if record.id == record_id:
                return record
        return None
```

## Testing Strategy

### Unit Test Coverage

```python
import unittest
from unittest.mock import Mock

class TestContextualChunking(unittest.TestCase):
    """Comprehensive tests for contextual chunking functionality."""
    
    def setUp(self):
        self.chunker = ContextualChunker(chunk_size=100, overlap_ratio=0.2)
        self.embedding_fn = Mock(return_value=[1.0, 0.0, 0.0])
    
    def test_short_conversation_no_chunking(self):
        """Test that short conversations remain as single records."""
        chunks = self.chunker.chunk_conversation("Hi", "Hello there!")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_parent)
    
    def test_long_conversation_chunking(self):
        """Test chunking of long conversations."""
        long_response = "This is a very long response. " * 20  # ~600 characters
        chunks = self.chunker.chunk_conversation("Question", long_response)
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        self.assertTrue(chunks[0].is_parent)  # First should be parent
        
        # All subsequent chunks should be children
        for i in range(1, len(chunks)):
            self.assertTrue(chunks[i].is_child)
            self.assertEqual(chunks[i].parent_id, chunks[0].id)
    
    def test_semantic_boundary_detection(self):
        """Test that chunking respects semantic boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        boundary, is_semantic = self.chunker.find_semantic_boundary(text, 20)
        
        self.assertTrue(is_semantic)
        self.assertIn(text[boundary-1:boundary+1], [". ", "! ", "? "])
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        long_text = "Word " * 50  # Predictable text for testing
        chunks = self.chunker.chunk_conversation("Q", long_text)
        
        if len(chunks) > 2:  # Need at least 3 chunks to test overlap
            # Check that middle chunks have overlap
            for i in range(1, len(chunks) - 1):
                metadata = chunks[i].chunk_metadata
                self.assertIsNotNone(metadata)
                self.assertGreater(metadata.overlap_start, 0)
    
    def test_memory_integration(self):
        """Test integration with enhanced episodic memory."""
        memory = EnhancedEpisodicMemory(
            storage_path=Path("/tmp/test_memory.json"),
            embedding_fn=self.embedding_fn,
            chunker=self.chunker
        )
        
        # Store a long conversation
        long_response = "Python has lists. Python has dictionaries. Python has sets."
        chunks = memory.remember("Tell me about Python", long_response)
        
        self.assertGreater(len(chunks), 1)
        
        # Test recall
        results = memory.recall("Python lists", top_k=2)
        self.assertGreater(len(results), 0)
        
        # Verify results have context
        for result in results:
            if result.is_child:
                self.assertIsNotNone(result.full_context)

class TestTemporalDecay(unittest.TestCase):
    """Test temporal decay weighting functionality."""
    
    def test_recent_memory_higher_weight(self):
        """Test that recent memories get higher weights."""
        now = time.time()
        recent_weight = EnhancedEpisodicMemory._calculate_temporal_weight(None, now)
        old_weight = EnhancedEpisodicMemory._calculate_temporal_weight(None, now - 3600)  # 1 hour ago
        
        self.assertGreater(recent_weight, old_weight)
    
    def test_decay_rate_configuration(self):
        """Test configurable decay rates."""
        timestamp = time.time() - 3600  # 1 hour ago
        
        slow_decay = EnhancedEpisodicMemory._calculate_temporal_weight(None, timestamp, decay_rate=0.05)
        fast_decay = EnhancedEpisodicMemory._calculate_temporal_weight(None, timestamp, decay_rate=0.2)
        
        self.assertGreater(slow_decay, fast_decay)

if __name__ == "__main__":
    unittest.main()
```

## Performance Considerations

### Memory Usage
- **Overhead**: Each chunk adds ~200 bytes of metadata
- **Storage**: 20-30% increase in storage for chunked conversations
- **Embedding Cost**: More embeddings generated, but smaller and more focused

### Retrieval Performance
- **Target**: <100ms for typical queries
- **Optimization**: Index chunk embeddings separately from parent embeddings
- **Caching**: Cache frequently accessed parent contexts

### Scalability
- **Chunk Limit**: Maximum 10 chunks per conversation turn
- **Total Records**: Increased from 240 to 1000 total records
- **Cleanup**: Automatic removal of orphaned chunks when parents are deleted

## Configuration Options

```python
# Environment variables for chunking configuration
ATLAS_CHUNK_SIZE = 400                # Characters per chunk
ATLAS_CHUNK_OVERLAP = 0.2            # Overlap ratio (0.0-0.5)
ATLAS_MIN_CHUNK_SIZE = 100           # Minimum chunk size
ATLAS_MAX_CHUNKS_PER_TURN = 10       # Maximum chunks per conversation turn
ATLAS_SEMANTIC_BOUNDARIES = true     # Enable semantic boundary detection
```

## Migration Path

### Phase 1: Backward Compatibility
- Existing `MemoryRecord` objects remain functional
- New `ChunkedMemoryRecord` extends base functionality
- Gradual migration as new conversations are added

### Phase 2: Full Migration
- Convert existing records to chunked format (single chunk per record)
- Add default values for new fields
- Preserve all existing functionality

### Phase 3: Optimization
- Analyze chunk patterns and optimize sizes
- Implement advanced semantic boundary detection
- Add performance monitoring and tuning

---

*This implementation guide provides the foundation for contextual chunking. All code should be thoroughly tested before integration with the main Atlas system.*