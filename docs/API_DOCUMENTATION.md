# Enhanced Memory System API Documentation

## Overview

This document provides comprehensive API documentation for Atlas's enhanced memory system, including contextual chunking, temporal decay, and session memory features.

## Core Interfaces

### Memory Record Types

#### `ChunkedMemoryRecord`

Enhanced memory record supporting hierarchical chunking and temporal awareness.

```python
@dataclass
class ChunkedMemoryRecord:
    """Enhanced memory record with hierarchical chunking support."""
    
    # Core identification
    id: str                                 # Unique record identifier
    parent_id: Optional[str] = None         # Parent record ID (None for parents)
    chunk_index: int = 0                    # Position within parent (0 for parent)
    
    # Content fields
    user: str = ""                          # User input text
    assistant: str = ""                     # Assistant response text  
    content: str = ""                       # Chunk content or full content
    full_context: Optional[str] = None      # Full parent context for reference
    
    # Memory metadata
    timestamp: float                        # Creation timestamp
    embedding: Optional[List[float]] = None # Vector embedding
    
    # Chunking metadata
    chunk_metadata: Optional[ChunkMetadata] = None
    
    # Temporal tracking
    access_count: int = 0                   # Number of times accessed
    last_accessed: float                    # Last access timestamp
    importance_score: float = 1.0           # Importance multiplier
    
    # Properties
    @property
    def is_parent(self) -> bool:
        """Check if this is a parent record."""
        
    @property  
    def is_child(self) -> bool:
        """Check if this is a child chunk."""
        
    # Methods
    def access(self) -> None:
        """Record memory access for reinforcement learning."""
```

**Usage Example:**
```python
# Create a parent record
parent = ChunkedMemoryRecord(
    user="What are Python data structures?",
    assistant="Python has several built-in data structures...",
    content="Full conversation content",
    chunk_index=0
)

# Create a child chunk
child = ChunkedMemoryRecord(
    parent_id=parent.id,
    chunk_index=1,
    content="Python has lists, dictionaries, and sets...",
    full_context=parent.content
)
```

#### `ChunkMetadata`

Metadata for chunk positioning and semantic boundaries.

```python
@dataclass
class ChunkMetadata:
    """Metadata for chunk positioning and overlap."""
    
    start_char: int          # Starting character position in parent
    end_char: int           # Ending character position in parent
    overlap_start: int      # Characters of overlap at beginning
    overlap_end: int        # Characters of overlap at end
    semantic_boundary: bool # True if chunk ends at sentence/paragraph boundary
```

### Chunking Interface

#### `ContextualChunker`

Intelligent text chunking with semantic boundary detection.

```python
class ContextualChunker:
    """Intelligent text chunking with semantic boundary detection."""
    
    def __init__(
        self, 
        chunk_size: int = 400,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 100
    ):
        """Initialize chunker with configuration.
        
        Args:
            chunk_size: Target characters per chunk
            overlap_ratio: Overlap percentage between chunks (0.0-0.5)
            min_chunk_size: Minimum viable chunk size
        """
    
    def chunk_conversation(
        self, 
        user_text: str, 
        assistant_text: str
    ) -> List[ChunkedMemoryRecord]:
        """Split conversation into contextual chunks.
        
        Args:
            user_text: User input text
            assistant_text: Assistant response text
            
        Returns:
            List of ChunkedMemoryRecord objects (parent + children)
            
        Raises:
            ValueError: If inputs are empty or invalid
        """
    
    def find_semantic_boundary(
        self, 
        text: str, 
        ideal_pos: int, 
        search_window: int = 50
    ) -> Tuple[int, bool]:
        """Find optimal semantic boundary near target position.
        
        Args:
            text: Text to search within
            ideal_pos: Ideal character position
            search_window: Characters to search before/after
            
        Returns:
            Tuple of (boundary_position, is_semantic_boundary)
        """
```

**Usage Example:**
```python
# Initialize chunker
chunker = ContextualChunker(
    chunk_size=400,
    overlap_ratio=0.2,
    min_chunk_size=100
)

# Chunk a conversation
chunks = chunker.chunk_conversation(
    user_text="Tell me about Python data structures and algorithms",
    assistant_text="Python provides several data structures... [long response]"
)

# Result: [parent_record, child_chunk_1, child_chunk_2, ...]
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk.content)} chars, parent={chunk.is_parent}")
```

### Temporal Weighting Interface

#### `TemporalMemoryWeighting`

Handles temporal decay and reinforcement for memory systems.

```python
class TemporalMemoryWeighting:
    """Handles temporal decay and reinforcement for memory systems."""
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize with temporal configuration."""
    
    def calculate_temporal_weight(
        self, 
        timestamp: float, 
        memory_type: str = "episodic",
        current_time: Optional[float] = None
    ) -> float:
        """Calculate temporal decay weight for a memory.
        
        Args:
            timestamp: When the memory was created
            memory_type: Type of memory (working, session, episodic, semantic)
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Weight factor between 0.0 and 1.0
        """
    
    def calculate_importance_multiplier(
        self, 
        access_count: int, 
        base_importance: float = 1.0
    ) -> float:
        """Calculate importance multiplier based on access frequency.
        
        Args:
            access_count: Number of times memory has been accessed
            base_importance: Base importance score
            
        Returns:
            Importance multiplier (capped at max_importance_multiplier)
        """
    
    def calculate_final_score(
        self,
        similarity_score: float,
        timestamp: float,
        access_count: int = 0,
        base_importance: float = 1.0,
        memory_type: str = "episodic"
    ) -> float:
        """Calculate final memory relevance score.
        
        Args:
            similarity_score: Base semantic similarity (0.0-1.0)
            timestamp: Memory creation timestamp
            access_count: Number of previous accesses
            base_importance: Base importance score
            memory_type: Type of memory for decay calculation
            
        Returns:
            Final weighted score combining all factors
        """
```

**Usage Example:**
```python
# Initialize temporal weighting
weighting = TemporalMemoryWeighting()

# Calculate weights for different memory ages
now = time.time()
recent_weight = weighting.calculate_temporal_weight(now)           # ≈1.0
hour_ago_weight = weighting.calculate_temporal_weight(now - 3600)  # ≈0.9
day_ago_weight = weighting.calculate_temporal_weight(now - 86400)  # ≈0.4

# Calculate final score with importance boosting
final_score = weighting.calculate_final_score(
    similarity_score=0.8,
    timestamp=now - 3600,
    access_count=5,
    memory_type="episodic"
)
```

#### `TemporalConfig`

Configuration for temporal decay behavior.

```python
@dataclass
class TemporalConfig:
    """Configuration for temporal decay behavior."""
    
    # Decay rates for different memory types (per hour)
    working_memory_decay: float = 0.5     # Fast decay for working memory
    session_memory_decay: float = 0.1     # Medium decay for session memory
    episodic_memory_decay: float = 0.01   # Slow decay for long-term memories
    semantic_memory_decay: float = 0.001  # Very slow decay for facts
    
    # Importance boosting factors
    access_boost_factor: float = 0.1      # How much each access boosts importance
    max_importance_multiplier: float = 5.0 # Cap on importance boosting
    
    # Reinforcement learning
    enable_reinforcement: bool = True      # Enable access-based reinforcement
    forgetting_threshold: float = 0.01    # Weight below which memories are forgotten
```

### Session Memory Interface

#### `SessionMemory`

Intermediate memory layer for recent conversation context.

```python
class SessionMemory:
    """Intermediate memory layer for recent conversation context."""
    
    def __init__(
        self, 
        config: Optional[SessionConfig] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        storage_path: Optional[Path] = None
    ):
        """Initialize session memory.
        
        Args:
            config: Session configuration
            embedding_fn: Function for generating embeddings
            storage_path: Optional persistent storage path
        """
    
    def add_turn(
        self, 
        user: str, 
        assistant: str, 
        force_new_session: bool = False
    ) -> str:
        """Add conversation turn to session memory.
        
        Args:
            user: User input text
            assistant: Assistant response text
            force_new_session: Force creation of new session
            
        Returns:
            Session ID where turn was added
        """
    
    def get_session_context(
        self, 
        session_id: Optional[str] = None, 
        max_turns: int = 20,
        include_expired: bool = False
    ) -> List[MemoryRecord]:
        """Retrieve conversation context from session memory.
        
        Args:
            session_id: Specific session to retrieve (defaults to current)
            max_turns: Maximum number of turns to return
            include_expired: Whether to include expired sessions
            
        Returns:
            List of memory records from session(s)
        """
    
    def get_all_recent_context(
        self, 
        hours_back: float = 2.0, 
        max_turns: int = 50
    ) -> List[MemoryRecord]:
        """Get all recent context across sessions within time window.
        
        Args:
            hours_back: How far back to look (in hours)
            max_turns: Maximum total turns to return
            
        Returns:
            List of recent memory records across all sessions
        """
    
    def promote_to_longterm(self, episodic_memory: 'EpisodicMemory') -> List[MemoryRecord]:
        """Promote important session memories to long-term storage.
        
        Args:
            episodic_memory: Long-term memory system to promote to
            
        Returns:
            List of promoted memory records
        """
```

**Usage Example:**
```python
# Initialize session memory
session_memory = SessionMemory(
    config=SessionConfig(duration_hours=2.0),
    embedding_fn=embedding_function
)

# Add conversation turns
session_id = session_memory.add_turn(
    user="What's the weather like?",
    assistant="I don't have access to real-time weather data..."
)

# Get recent context
context = session_memory.get_session_context(max_turns=10)
print(f"Retrieved {len(context)} recent turns")

# Get context across all sessions
all_context = session_memory.get_all_recent_context(hours_back=1.0)
```

#### `SessionConfig`

Configuration for session memory behavior.

```python
@dataclass
class SessionConfig:
    """Configuration for session memory behavior."""
    
    duration_hours: float = 2.0           # How long to keep sessions active
    max_turns_per_session: int = 50       # Maximum turns per session
    promotion_threshold: float = 0.7      # Similarity threshold for promotion
    cleanup_interval_minutes: int = 15    # How often to clean up expired sessions
    
    # Session boundaries
    inactivity_threshold_minutes: int = 30  # Minutes of inactivity before new session
    topic_change_threshold: float = 0.3     # Embedding similarity threshold for topic change
```

### Enhanced Episodic Memory Interface

#### `EnhancedEpisodicMemory`

Extended episodic memory with contextual chunking support.

```python
class EnhancedEpisodicMemory(EpisodicMemory):
    """Extended episodic memory with contextual chunking support."""
    
    def __init__(
        self, 
        *args, 
        chunker: Optional[ContextualChunker] = None,
        temporal_weighting: Optional[TemporalMemoryWeighting] = None,
        **kwargs
    ):
        """Initialize enhanced episodic memory.
        
        Args:
            chunker: Contextual chunker instance
            temporal_weighting: Temporal weighting instance
            *args, **kwargs: Arguments passed to parent EpisodicMemory
        """
    
    def remember(self, user: str, assistant: str) -> List[ChunkedMemoryRecord]:
        """Store conversation with contextual chunking.
        
        Args:
            user: User input text
            assistant: Assistant response text
            
        Returns:
            List of stored memory records (parent + children)
            
        Raises:
            ValueError: If inputs are invalid
            MemoryError: If storage fails
        """
    
    def recall(
        self, 
        query: str, 
        *, 
        top_k: int = 4, 
        include_parents: bool = True,
        memory_type: str = "episodic"
    ) -> List[ChunkedMemoryRecord]:
        """Enhanced recall with chunk-aware retrieval.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            include_parents: Whether to return parent context for child chunks
            memory_type: Memory type for temporal weighting
            
        Returns:
            List of relevant memory records with context
        """
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.
        
        Returns:
            Dictionary containing:
            - total_records: Total number of records
            - parent_records: Number of parent records
            - child_chunks: Number of child chunks
            - avg_chunks_per_conversation: Average chunks per conversation
            - storage_size_bytes: Estimated storage size
            - oldest_record_age_hours: Age of oldest record
        """
    
    def cleanup_expired_memories(self, current_time: Optional[float] = None) -> int:
        """Remove memories that have decayed below forgetting threshold.
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Number of memories removed
        """
```

**Usage Example:**
```python
# Initialize enhanced episodic memory
memory = EnhancedEpisodicMemory(
    storage_path=Path("~/.local/share/atlas/episodic.json"),
    embedding_fn=embedding_function,
    chunker=ContextualChunker(chunk_size=400),
    temporal_weighting=TemporalMemoryWeighting(),
    max_records=1000
)

# Store a conversation
chunks = memory.remember(
    user="Explain machine learning algorithms",
    assistant="Machine learning algorithms can be categorized into... [long response]"
)
print(f"Stored {len(chunks)} chunks")

# Recall relevant memories
results = memory.recall(
    query="supervised learning",
    top_k=3,
    include_parents=True
)

for result in results:
    print(f"Found: {result.content[:100]}...")
    if result.full_context:
        print(f"Context: {len(result.full_context)} characters")
```

## Error Handling

### Exception Types

```python
class MemoryError(Exception):
    """Base exception for memory system errors."""
    pass

class ChunkingError(MemoryError):
    """Exception raised during text chunking."""
    pass

class TemporalWeightingError(MemoryError):
    """Exception raised during temporal weight calculation."""
    pass

class SessionMemoryError(MemoryError):
    """Exception raised during session memory operations."""
    pass
```

### Error Handling Patterns

```python
try:
    chunks = memory.remember(user_input, assistant_response)
except ChunkingError as e:
    # Fallback to single chunk
    logger.warning(f"Chunking failed: {e}, using single chunk")
    chunks = [create_single_chunk(user_input, assistant_response)]
except MemoryError as e:
    # Graceful degradation
    logger.error(f"Memory storage failed: {e}")
    # Continue without persistence
```

## Performance Characteristics

### Memory Usage

| Component | Memory per Record | Notes |
|-----------|------------------|-------|
| Base MemoryRecord | ~500 bytes | Original record size |
| ChunkedMemoryRecord | ~700 bytes | Additional metadata |
| SessionTurn | ~600 bytes | Session-specific tracking |
| ChunkMetadata | ~100 bytes | Positioning information |

### Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Chunking | <50ms | Per conversation turn |
| Temporal scoring | <10ms | Per memory record |
| Session context retrieval | <20ms | Recent turns only |
| Enhanced recall | <100ms | Including temporal weighting |
| Memory cleanup | <500ms | Expired memory removal |

### Scalability Limits

| Component | Limit | Configuration |
|-----------|-------|---------------|
| Total episodic records | 1000 | `max_records` |
| Chunks per conversation | 10 | `ATLAS_MAX_CHUNKS_PER_TURN` |
| Session duration | 8 hours | `duration_hours` |
| Active sessions | 50 | Memory management |

## Configuration Reference

### Environment Variables

```bash
# Chunking configuration
ATLAS_CHUNK_SIZE=400                    # Characters per chunk
ATLAS_CHUNK_OVERLAP=0.2                # Overlap percentage (0.0-0.5)  
ATLAS_MIN_CHUNK_SIZE=100               # Minimum chunk size
ATLAS_MAX_CHUNKS_PER_TURN=10           # Maximum chunks per conversation

# Temporal decay configuration
ATLAS_TEMPORAL_DECAY_RATE=0.1          # Decay rate per hour
ATLAS_IMPORTANCE_BOOST_FACTOR=0.1       # Access frequency boost
ATLAS_MAX_IMPORTANCE_MULTIPLIER=5.0     # Importance cap
ATLAS_FORGETTING_THRESHOLD=0.01         # Forgetting threshold

# Session memory configuration
ATLAS_SESSION_DURATION_HOURS=2.0        # Session lifetime
ATLAS_SESSION_MAX_TURNS=50              # Max turns per session
ATLAS_SESSION_INACTIVITY_MINUTES=30     # Inactivity threshold
ATLAS_SESSION_CLEANUP_MINUTES=15        # Cleanup interval

# Enhanced memory limits
ATLAS_MAX_EPISODIC_RECORDS=1000         # Total episodic record limit
ATLAS_ENABLE_MEMORY_METRICS=true        # Enable metrics collection
```

### Programmatic Configuration

```python
# Chunker configuration
chunker_config = {
    'chunk_size': 400,
    'overlap_ratio': 0.2,
    'min_chunk_size': 100
}

# Temporal configuration
temporal_config = TemporalConfig(
    episodic_memory_decay=0.01,
    access_boost_factor=0.1,
    max_importance_multiplier=5.0,
    enable_reinforcement=True
)

# Session configuration
session_config = SessionConfig(
    duration_hours=2.0,
    max_turns_per_session=50,
    inactivity_threshold_minutes=30
)
```

## Migration Guide

### From Legacy MemoryRecord

```python
def migrate_legacy_record(legacy_record: MemoryRecord) -> ChunkedMemoryRecord:
    """Migrate legacy MemoryRecord to ChunkedMemoryRecord."""
    return ChunkedMemoryRecord(
        id=legacy_record.id,
        user=legacy_record.user,
        assistant=legacy_record.assistant,
        content=f"User: {legacy_record.user}\nAssistant: {legacy_record.assistant}",
        timestamp=legacy_record.timestamp,
        embedding=legacy_record.embedding,
        chunk_index=0,  # Treat as parent record
        access_count=0,
        importance_score=1.0
    )
```

### Backward Compatibility

The enhanced memory system maintains full backward compatibility:

- Existing `MemoryRecord` objects can be loaded and used
- New `ChunkedMemoryRecord` extends functionality without breaking changes
- Legacy storage formats are automatically migrated on first load
- All existing Atlas functionality continues to work unchanged

---

*This API documentation provides comprehensive coverage of the enhanced memory system. For implementation examples and testing guidance, see the accompanying documentation files.*