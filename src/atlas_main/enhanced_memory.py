"""Enhanced memory system for Atlas with contextual chunking, temporal decay, and session memory.

This module implements sophisticated memory capabilities inspired by MemGPT, MemoriPY, 
and modern RAG techniques, providing human-like memory dynamics while maintaining 
full backward compatibility with existing Atlas memory systems.

Research Foundation:
- MemGPT: OS-inspired virtual context management (arXiv:2310.08560)
- EM-LLM: Human episodic memory patterns (arXiv:2407.09450)  
- Anthropic Contextual Retrieval: 49% improvement in retrieval accuracy
- Parent-Child Chunking: Modern RAG best practices
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from .memory import MemoryRecord, EmbeddingFunction

# Core Data Structures

@dataclass
class ChunkMetadata:
    """Metadata for chunk positioning and semantic boundaries.
    
    Tracks the position of text chunks within their parent conversation,
    including overlap information and whether chunk boundaries align with 
    semantic boundaries (sentences, paragraphs).
    """
    start_char: int          # Starting character position in parent text
    end_char: int           # Ending character position in parent text  
    overlap_start: int      # Characters of overlap at beginning of chunk
    overlap_end: int        # Characters of overlap at end of chunk
    semantic_boundary: bool # True if chunk ends at sentence/paragraph boundary

@dataclass
class ChunkedMemoryRecord:
    """Enhanced memory record supporting hierarchical chunking and temporal dynamics.
    
    Extends the base MemoryRecord with support for parent-child relationships,
    contextual chunking, temporal decay tracking, and access-based reinforcement.
    Maintains full backward compatibility with existing MemoryRecord objects.
    """
    
    # Core identification fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None         # Links to parent conversation record
    chunk_index: int = 0                    # 0 for parent, 1+ for child chunks
    
    # Content fields (maintains MemoryRecord compatibility)
    user: str = ""                          # User input text
    assistant: str = ""                     # Assistant response text
    content: str = ""                       # Chunk content or full conversation
    full_context: Optional[str] = None      # Full parent context for child chunks
    
    # Memory metadata
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None # Vector embedding for similarity search
    
    # Enhanced chunking metadata
    chunk_metadata: Optional[ChunkMetadata] = None
    
    # Temporal dynamics and reinforcement learning
    access_count: int = 0                   # Number of times memory accessed
    last_accessed: float = field(default_factory=time.time)
    importance_score: float = 1.0           # Importance multiplier from access patterns
    
    @property
    def is_parent(self) -> bool:
        """Check if this is a parent record (full conversation)."""
        return self.parent_id is None and self.chunk_index == 0
    
    @property  
    def is_child(self) -> bool:
        """Check if this is a child chunk of a parent conversation."""
        return self.parent_id is not None and self.chunk_index > 0
    
    def access(self) -> None:
        """Record memory access for reinforcement learning.
        
        Updates access count, last accessed timestamp, and applies logarithmic
        importance score boosting to prevent runaway growth while still 
        rewarding frequently accessed memories.
        """
        self.access_count += 1
        self.last_accessed = time.time()
        
        # Logarithmic importance boost prevents runaway growth
        # Formula: importance *= (1 + boost_factor / log(access_count + 1))
        boost_factor = 0.1
        if self.access_count > 0:
            boost = 1.0 + (boost_factor / math.log(self.access_count + 1))
            self.importance_score *= boost
    
    @classmethod
    def from_legacy_record(cls, legacy: MemoryRecord) -> 'ChunkedMemoryRecord':
        """Create enhanced record from legacy MemoryRecord for migration.
        
        Args:
            legacy: Legacy MemoryRecord to migrate
            
        Returns:
            New ChunkedMemoryRecord with preserved data and default enhanced fields
        """
        return cls(
            id=legacy.id,
            user=legacy.user,
            assistant=legacy.assistant,
            content=f"User: {legacy.user}\nAssistant: {legacy.assistant}",
            timestamp=legacy.timestamp,
            embedding=legacy.embedding,
            chunk_index=0,  # Legacy records become parents
            access_count=0,
            importance_score=1.0
        )
    
    def to_legacy_record(self) -> MemoryRecord:
        """Convert to legacy MemoryRecord for backward compatibility.
        
        Returns:
            Legacy MemoryRecord with core fields preserved
        """
        return MemoryRecord(
            id=self.id,
            user=self.user,
            assistant=self.assistant,
            timestamp=self.timestamp,
            embedding=self.embedding
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Handle nested dataclass
        if self.chunk_metadata:
            data['chunk_metadata'] = asdict(self.chunk_metadata)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkedMemoryRecord':
        """Create instance from dictionary (JSON deserialization)."""
        # Handle chunk metadata
        chunk_metadata = data.get('chunk_metadata')
        if chunk_metadata and isinstance(chunk_metadata, dict):
            data['chunk_metadata'] = ChunkMetadata(**chunk_metadata)
        
        # Ensure required fields have defaults
        data.setdefault('id', str(uuid.uuid4()))
        data.setdefault('timestamp', time.time())
        data.setdefault('chunk_index', 0)
        data.setdefault('access_count', 0)
        data.setdefault('last_accessed', time.time())
        data.setdefault('importance_score', 1.0)
        
        return cls(**data)

# Contextual Chunking Implementation

class ContextualChunker:
    """Intelligent text chunking with semantic boundary detection.
    
    Implements parent-child chunking strategy that splits long conversations
    into semantically coherent chunks with configurable overlap, while 
    preserving sentence and paragraph boundaries when possible.
    
    Based on research from:
    - Parent-Child Chunking in LangChain for Advanced RAG
    - Anthropic's Contextual Retrieval (49% retrieval improvement)
    - Sliding window methodologies for context preservation
    """
    
    def __init__(
        self, 
        chunk_size: int = 400,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 100
    ):
        """Initialize contextual chunker with configuration.
        
        Args:
            chunk_size: Target characters per chunk (300-500 recommended)
            overlap_ratio: Overlap percentage between chunks (0.1-0.3 recommended)
            min_chunk_size: Minimum viable chunk size to prevent tiny fragments
        """
        self.chunk_size = max(100, chunk_size)
        self.overlap_size = max(10, int(chunk_size * overlap_ratio))
        self.min_chunk_size = max(30, min_chunk_size)
        
        # Sentence boundary patterns (ordered by preference for splitting)
        self.sentence_patterns = [
            r'\.[\s\n]+',           # Period followed by whitespace (highest priority)
            r'[!?][\s\n]+',         # Exclamation/question + whitespace  
            r';\s+',                # Semicolon + space
            r':\s+(?=[A-Z])',       # Colon + space + capital letter
        ]
        
        # Paragraph boundary patterns (even higher priority)
        self.paragraph_patterns = [
            r'\n\s*\n',             # Double newline (paragraph break)
            r'\n\s*[-*]\s+',        # Newline + bullet point
            r'\n\s*\d+\.\s+',       # Newline + numbered list
            r'\n\s*[A-Z][^.]*:\s*', # Newline + header-like pattern
        ]
    
    def find_semantic_boundary(
        self, 
        text: str, 
        ideal_pos: int, 
        search_window: int = 50
    ) -> Tuple[int, bool]:
        """Find optimal semantic boundary near target position.
        
        Searches for sentence or paragraph boundaries within a window around
        the ideal chunk boundary position. Prioritizes paragraph boundaries
        over sentence boundaries for better semantic coherence.
        
        Args:
            text: Text to search within
            ideal_pos: Ideal character position for boundary
            search_window: Characters to search before/after ideal position
            
        Returns:
            Tuple of (boundary_position, is_semantic_boundary)
        """
        if ideal_pos >= len(text):
            return len(text), True
        
        start_search = max(0, ideal_pos - search_window)
        end_search = min(len(text), ideal_pos + search_window)
        search_text = text[start_search:end_search]
        
        # Try paragraph boundaries first (highest priority)
        for pattern in self.paragraph_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find match closest to ideal position
                best_match = min(
                    matches, 
                    key=lambda m: abs((start_search + m.end()) - ideal_pos)
                )
                boundary_pos = start_search + best_match.end()
                return min(boundary_pos, len(text)), True
        
        # Try sentence boundaries
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                best_match = min(
                    matches,
                    key=lambda m: abs((start_search + m.end()) - ideal_pos)
                )
                boundary_pos = start_search + best_match.end()
                return min(boundary_pos, len(text)), True
        
        # Fallback to word boundary
        word_boundaries = list(re.finditer(r'\s+', search_text))
        if word_boundaries:
            best_match = min(
                word_boundaries,
                key=lambda m: abs((start_search + m.start()) - ideal_pos)
            )
            boundary_pos = start_search + best_match.start()
            return min(boundary_pos, len(text)), False
        
        # Last resort: exact position (avoid splitting within words if possible)
        return min(ideal_pos, len(text)), False
    
    def chunk_conversation(
        self, 
        user_text: str, 
        assistant_text: str
    ) -> List[ChunkedMemoryRecord]:
        """Split conversation into contextual chunks with overlap.
        
        Creates parent record containing full conversation plus child chunks
        for long responses. Implements 20% overlap between chunks and respects
        semantic boundaries when possible.
        
        Args:
            user_text: User input text
            assistant_text: Assistant response text
            
        Returns:
            List of ChunkedMemoryRecord objects (parent + children)
            
        Raises:
            ValueError: If inputs are invalid or empty
        """
        if not user_text.strip() and not assistant_text.strip():
            raise ValueError("Both user and assistant text cannot be empty")
        
        # Combine texts for chunking analysis
        full_conversation = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
        
        # If conversation is short enough, return single parent record
        if len(full_conversation) <= self.chunk_size:
            parent = ChunkedMemoryRecord(
                user=user_text,
                assistant=assistant_text,
                content=full_conversation,
                chunk_index=0
            )
            return [parent]
        
        # Create parent record for full conversation
        parent_id = str(uuid.uuid4())
        parent = ChunkedMemoryRecord(
            id=parent_id,
            user=user_text,
            assistant=assistant_text,
            content=full_conversation,
            chunk_index=0
        )
        
        # Generate child chunks with overlap
        chunks = [parent]
        current_pos = 0
        chunk_index = 1
        max_chunks = 10  # Safety limit to prevent runaway chunking
        
        while current_pos < len(full_conversation) and chunk_index <= max_chunks:
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
            overlap_start = self.overlap_size if chunk_index > 1 else 0
            actual_start = max(0, current_pos - overlap_start)
            
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
                overlap_end=0,  # Will be calculated for next chunk
                semantic_boundary=semantic_boundary
            )
            
            # Create child chunk record
            child_chunk = ChunkedMemoryRecord(
                parent_id=parent_id,
                chunk_index=chunk_index,
                content=chunk_content,
                full_context=full_conversation,
                chunk_metadata=metadata,
                timestamp=parent.timestamp  # Inherit parent timestamp
            )
            
            chunks.append(child_chunk)
            
            # Move to next chunk position (considering overlap)
            current_pos = chunk_end - self.overlap_size
            chunk_index += 1
        
        # Update overlap_end metadata for all chunks except the last
        for i in range(len(chunks) - 1):
            if chunks[i].chunk_metadata and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if next_chunk.chunk_metadata:
                    overlap_end = max(0, 
                        chunks[i].chunk_metadata.end_char - next_chunk.chunk_metadata.start_char
                    )
                    chunks[i].chunk_metadata.overlap_end = overlap_end
        
        return chunks

# Temporal Decay and Reinforcement

@dataclass
class TemporalConfig:
    """Configuration for temporal decay and reinforcement behavior.
    
    Controls how memory importance changes over time and with access patterns.
    Different memory types can have different decay characteristics to match
    human-like forgetting curves.
    """
    
    # Decay rates for different memory types (per hour)
    working_memory_decay: float = 0.5       # Fast decay - minutes to hours
    session_memory_decay: float = 0.1       # Medium decay - hours to days
    episodic_memory_decay: float = 0.01     # Slow decay - days to weeks
    semantic_memory_decay: float = 0.001    # Very slow - weeks to months
    
    # Access-based importance boosting
    access_boost_factor: float = 0.1        # How much each access boosts importance
    max_importance_multiplier: float = 5.0  # Cap on importance boosting
    
    # Forgetting and reinforcement
    enable_reinforcement: bool = True       # Enable access-based reinforcement
    forgetting_threshold: float = 0.01      # Weight below which memories forgotten

class TemporalMemoryWeighting:
    """Handles temporal decay and access-based reinforcement for memory systems.
    
    Implements exponential temporal decay combined with logarithmic importance
    boosting based on access frequency. Supports different decay rates for 
    different types of memories to match human-like forgetting patterns.
    
    Research Foundation:
    - MemoriPY memory dynamics with decay/reinforcement
    - Ebbinghaus forgetting curve principles
    - Human episodic memory temporal dynamics
    """
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize temporal weighting with configuration.
        
        Args:
            config: Temporal configuration, uses defaults if None
        """
        self.config = config or TemporalConfig()
    
    def calculate_temporal_weight(
        self, 
        timestamp: float, 
        memory_type: str = "episodic",
        current_time: Optional[float] = None
    ) -> float:
        """Calculate exponential temporal decay weight for a memory.
        
        Uses formula: weight = exp(-decay_rate * age_hours)
        Different memory types have different decay rates to match human
        forgetting patterns (working memory fades fast, semantic slowly).
        
        Args:
            timestamp: When the memory was created (Unix timestamp)
            memory_type: Type of memory (working, session, episodic, semantic)
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Weight factor between 0.0 and 1.0
        """
        if current_time is None:
            current_time = time.time()
        
        # Handle edge case of future timestamps (shouldn't happen but be robust)
        age_hours = max(0.0, (current_time - timestamp) / 3600)
        
        # Select appropriate decay rate based on memory type
        decay_rates = {
            "working": self.config.working_memory_decay,
            "session": self.config.session_memory_decay,
            "episodic": self.config.episodic_memory_decay,
            "semantic": self.config.semantic_memory_decay
        }
        
        decay_rate = decay_rates.get(memory_type, self.config.episodic_memory_decay)
        
        # Calculate exponential decay: e^(-λt)
        weight = math.exp(-decay_rate * age_hours)
        
        # Ensure weight stays in valid range
        return max(0.0, min(1.0, weight))
    
    def calculate_importance_multiplier(
        self, 
        access_count: int, 
        base_importance: float = 1.0
    ) -> float:
        """Calculate importance multiplier based on access frequency.
        
        Uses logarithmic scaling to boost importance of frequently accessed
        memories while preventing runaway growth. Formula allows memories
        to become more important but caps the maximum boost.
        
        Args:
            access_count: Number of times memory has been accessed
            base_importance: Base importance score
            
        Returns:
            Importance multiplier (capped at max_importance_multiplier)
        """
        if not self.config.enable_reinforcement or access_count <= 0:
            return base_importance
        
        # Logarithmic importance boost: 1 + (boost_factor * log(count + 1))
        # This prevents runaway growth while still rewarding frequent access
        boost = 1.0 + (self.config.access_boost_factor * math.log(access_count + 1))
        multiplier = base_importance * boost
        
        # Apply maximum importance cap
        return min(multiplier, self.config.max_importance_multiplier)
    
    def calculate_final_score(
        self,
        similarity_score: float,
        timestamp: float,
        access_count: int = 0,
        base_importance: float = 1.0,
        memory_type: str = "episodic"
    ) -> float:
        """Calculate final memory relevance score combining all factors.
        
        Combines semantic similarity, temporal decay, and access-based importance
        into unified relevance score for memory retrieval ranking.
        
        Args:
            similarity_score: Base semantic similarity (0.0-1.0)
            timestamp: Memory creation timestamp
            access_count: Number of previous accesses
            base_importance: Base importance score
            memory_type: Type of memory for decay calculation
            
        Returns:
            Final weighted score for retrieval ranking
        """
        # Calculate temporal decay factor
        temporal_weight = self.calculate_temporal_weight(timestamp, memory_type)
        
        # Calculate importance multiplier from access frequency
        importance_multiplier = self.calculate_importance_multiplier(
            access_count, base_importance
        )
        
        # Combine all factors: similarity × temporal_decay × importance
        final_score = similarity_score * temporal_weight * importance_multiplier
        
        return max(0.0, final_score)
    
    def should_forget_memory(
        self, 
        timestamp: float, 
        memory_type: str = "episodic"
    ) -> bool:
        """Determine if memory should be forgotten due to age and low importance.
        
        Memories that have decayed below the forgetting threshold are candidates
        for removal to prevent unlimited memory growth.
        
        Args:
            timestamp: Memory creation timestamp
            memory_type: Type of memory for decay calculation
            
        Returns:
            True if memory should be forgotten (removed from storage)
        """
        weight = self.calculate_temporal_weight(timestamp, memory_type)
        return weight < self.config.forgetting_threshold

# Session Memory Implementation

@dataclass
class SessionConfig:
    """Configuration for session memory behavior.
    
    Controls session boundaries, cleanup, and promotion to long-term memory.
    Sessions bridge the gap between working memory and episodic storage.
    """
    
    duration_hours: float = 2.0             # How long to keep sessions active
    max_turns_per_session: int = 50         # Maximum turns per session
    promotion_threshold: float = 0.7        # Threshold for promoting to long-term
    cleanup_interval_minutes: int = 15      # How often to clean up expired sessions
    
    # Session boundary detection
    inactivity_threshold_minutes: int = 30  # Minutes of inactivity → new session
    topic_change_threshold: float = 0.3     # Embedding similarity threshold

@dataclass
class SessionTurn:
    """Individual conversation turn within a session.
    
    Lighter-weight representation than full ChunkedMemoryRecord for
    intermediate session storage before promotion to long-term memory.
    """
    user: str
    assistant: str
    timestamp: float
    importance_score: float = 1.0
    embedding: Optional[List[float]] = None

class Session:
    """Individual conversation session with automatic management.
    
    Represents a coherent conversation session with topic consistency,
    automatic expiration, and promotion logic for important memories.
    """
    
    def __init__(self, session_id: str, start_time: float, config: SessionConfig):
        """Initialize session with configuration.
        
        Args:
            session_id: Unique identifier for this session
            start_time: Session creation timestamp
            config: Session configuration parameters
        """
        self.session_id = session_id
        self.start_time = start_time
        self.config = config
        self.turns: deque[SessionTurn] = deque(maxlen=config.max_turns_per_session)
        self.promoted = False
        self.topic_embedding: Optional[List[float]] = None
    
    @property
    def turn_count(self) -> int:
        """Get number of turns in this session."""
        return len(self.turns)
    
    def add_turn(self, user: str, assistant: str, timestamp: float) -> None:
        """Add conversation turn to this session.
        
        Args:
            user: User input text
            assistant: Assistant response text
            timestamp: Turn timestamp
        """
        turn = SessionTurn(
            user=user,
            assistant=assistant,
            timestamp=timestamp
        )
        self.turns.append(turn)
    
    def get_recent_turns(self, max_turns: int) -> List[MemoryRecord]:
        """Get recent turns as MemoryRecord objects for compatibility.
        
        Args:
            max_turns: Maximum number of turns to return
            
        Returns:
            List of MemoryRecord objects for recent turns
        """
        recent = list(self.turns)[-max_turns:] if max_turns > 0 else list(self.turns)
        
        records = []
        for i, turn in enumerate(recent):
            record = MemoryRecord(
                id=f"{self.session_id}_{len(records)}",
                user=turn.user,
                assistant=turn.assistant,
                timestamp=turn.timestamp,
                embedding=turn.embedding
            )
            records.append(record)
        
        return records
    
    def get_turns_since(self, cutoff_time: float) -> List[MemoryRecord]:
        """Get all turns since a specific timestamp.
        
        Args:
            cutoff_time: Only return turns after this timestamp
            
        Returns:
            List of MemoryRecord objects for recent turns
        """
        filtered_turns = [turn for turn in self.turns if turn.timestamp >= cutoff_time]
        
        records = []
        for i, turn in enumerate(filtered_turns):
            record = MemoryRecord(
                id=f"{self.session_id}_{len(records)}",
                user=turn.user,
                assistant=turn.assistant,
                timestamp=turn.timestamp,
                embedding=turn.embedding
            )
            records.append(record)
        
        return records
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if session has expired based on age.
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            True if session should be considered expired
        """
        if current_time is None:
            current_time = time.time()
        
        age_hours = (current_time - self.start_time) / 3600
        return age_hours > self.config.duration_hours
    
    def get_last_activity_time(self) -> float:
        """Get timestamp of most recent activity in session.
        
        Returns:
            Timestamp of last turn, or session start time if no turns
        """
        if not self.turns:
            return self.start_time
        return self.turns[-1].timestamp
    
    def should_promote(self, current_time: float, threshold: float) -> bool:
        """Check if session should be promoted to long-term memory.
        
        Args:
            current_time: Current timestamp
            threshold: Promotion threshold (unused in basic implementation)
            
        Returns:
            True if session should be promoted to episodic memory
        """
        return (
            not self.promoted and
            self.is_expired(current_time) and
            len(self.turns) >= 3  # Minimum turns for meaningful promotion
        )
    
    def get_important_turns(self) -> List[SessionTurn]:
        """Get turns that should be promoted to long-term memory.
        
        In basic implementation, returns all turns. Could be enhanced
        with importance scoring and selective promotion.
        
        Returns:
            List of turns to promote to long-term storage
        """
        return list(self.turns)
    
    def mark_promoted(self) -> None:
        """Mark session as promoted to prevent duplicate promotion."""
        self.promoted = True
    
    def calculate_topic_similarity(self, user_input: str, embedding_fn: EmbeddingFunction) -> float:
        """Calculate similarity between new input and session topic.
        
        Uses session topic embedding (derived from conversation content) to
        determine if new input represents topic change requiring new session.
        
        Args:
            user_input: New user input to compare
            embedding_fn: Function to generate embeddings
            
        Returns:
            Similarity score (0.0-1.0), higher means more similar
        """
        if not self.topic_embedding:
            # Initialize topic embedding from session content
            if self.turns:
                session_texts = [f"{turn.user} {turn.assistant}" for turn in self.turns]
                session_content = " ".join(session_texts)
                if session_content.strip():
                    try:
                        self.topic_embedding = embedding_fn(session_content)
                    except Exception:
                        # If embedding fails, assume similarity (conservative)
                        return 1.0
        
        if not self.topic_embedding:
            return 1.0  # Default to high similarity if no embedding available
        
        # Calculate similarity with new input
        try:
            input_embedding = embedding_fn(user_input)
            if not input_embedding:
                return 1.0
            
            # Cosine similarity calculation
            topic_vec = np.asarray(self.topic_embedding, dtype=float)
            input_vec = np.asarray(input_embedding, dtype=float)
            
            if topic_vec.shape != input_vec.shape:
                return 1.0  # Assume similarity if shapes don't match
            
            # Calculate cosine similarity
            denom = np.linalg.norm(topic_vec) * np.linalg.norm(input_vec)
            if denom == 0:
                return 1.0
            
            similarity = float(np.dot(topic_vec, input_vec) / denom)
            return max(0.0, min(1.0, similarity))
        
        except Exception:
            # If any error occurs, assume similarity (conservative)
            return 1.0

class SessionMemory:
    """Intermediate memory layer for recent conversation context.
    
    Bridges the gap between working memory (immediate context) and episodic
    memory (long-term storage) by maintaining session-level conversation
    context with automatic boundaries and cleanup.
    
    Research Foundation:
    - MemGPT hierarchical memory architecture
    - LlamaIndex memory blocks for context-aware conversations
    - Multi-timescale memory organization patterns
    """
    
    def __init__(
        self, 
        config: Optional[SessionConfig] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        storage_path: Optional[Path] = None
    ):
        """Initialize session memory with configuration.
        
        Args:
            config: Session configuration parameters
            embedding_fn: Function for generating embeddings
            storage_path: Optional path for session persistence
        """
        self.config = config or SessionConfig()
        self.embedding_fn = embedding_fn
        self.storage_path = storage_path
        
        # Active session storage
        self._sessions: Dict[str, Session] = {}
        self._current_session_id: Optional[str] = None
        
        # Cleanup management
        self._cleanup_timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()  # Thread safety for cleanup
        
        # Start automatic cleanup
        self._start_cleanup_timer()
        
        # Load persisted sessions if storage configured
        if self.storage_path:
            self._load_sessions()
    
    def add_turn(
        self, 
        user: str, 
        assistant: str, 
        force_new_session: bool = False
    ) -> str:
        """Add conversation turn to session memory with automatic session management.
        
        Automatically creates new sessions based on inactivity, topic changes,
        or capacity limits. Returns session ID for tracking.
        
        Args:
            user: User input text
            assistant: Assistant response text
            force_new_session: Force creation of new session regardless of rules
            
        Returns:
            Session ID where turn was added
        """
        current_time = time.time()
        
        with self._lock:
            # Determine if we need a new session
            should_create_new = (
                force_new_session or
                self._current_session_id is None or
                self._should_start_new_session(user, assistant, current_time)
            )
            
            if should_create_new:
                session_id = self._create_new_session(current_time)
            else:
                session_id = self._current_session_id
            
            # Add turn to appropriate session
            session = self._sessions[session_id]
            session.add_turn(user, assistant, current_time)
            
            # Update current session tracking
            self._current_session_id = session_id
            
            # Persist if storage configured
            if self.storage_path:
                self._save_sessions()
        
        return session_id
    
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
            List of MemoryRecord objects from session(s)
        """
        with self._lock:
            if session_id:
                # Get specific session
                session = self._sessions.get(session_id)
                if session and (include_expired or not session.is_expired()):
                    return session.get_recent_turns(max_turns)
                return []
            
            # Get context from current session
            if self._current_session_id:
                return self.get_session_context(
                    self._current_session_id, max_turns, include_expired
                )
            
            return []
    
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
            List of recent MemoryRecord objects across all sessions
        """
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        all_records = []
        
        with self._lock:
            # Collect records from all active sessions
            for session in self._sessions.values():
                if not session.is_expired(current_time):
                    session_records = session.get_turns_since(cutoff_time)
                    all_records.extend(session_records)
        
        # Sort by timestamp and limit
        all_records.sort(key=lambda r: r.timestamp)
        return all_records[-max_turns:] if max_turns > 0 else all_records
    
    def promote_to_longterm(self, episodic_memory) -> List[MemoryRecord]:
        """Promote important session memories to long-term storage.
        
        Identifies expired sessions with important conversations and promotes
        them to episodic memory for long-term retention.
        
        Args:
            episodic_memory: Enhanced episodic memory system
            
        Returns:
            List of promoted memory records
        """
        promoted_records = []
        current_time = time.time()
        
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.should_promote(current_time, self.config.promotion_threshold):
                    # Get important turns from session
                    important_turns = session.get_important_turns()
                    
                    # Promote each turn to long-term memory
                    for turn in important_turns:
                        try:
                            promoted = episodic_memory.remember(turn.user, turn.assistant)
                            if isinstance(promoted, list):
                                promoted_records.extend(promoted)
                            else:
                                promoted_records.append(promoted)
                        except Exception as e:
                            # Log error but continue with other turns
                            print(f"Warning: Failed to promote turn to long-term memory: {e}")
                    
                    # Mark session as promoted
                    session.mark_promoted()
        
        return promoted_records
    
    def _should_start_new_session(self, user: str, assistant: str, current_time: float) -> bool:
        """Determine if a new session should be started based on heuristics.
        
        Args:
            user: New user input
            assistant: New assistant response
            current_time: Current timestamp
            
        Returns:
            True if new session should be created
        """
        if not self._current_session_id:
            return True
        
        current_session = self._sessions.get(self._current_session_id)
        if not current_session:
            return True
        
        # Check inactivity threshold
        last_activity = current_session.get_last_activity_time()
        inactivity_minutes = (current_time - last_activity) / 60
        if inactivity_minutes > self.config.inactivity_threshold_minutes:
            return True
        
        # Check session capacity
        if current_session.turn_count >= self.config.max_turns_per_session:
            return True
        
        # Check topic change (if embedding function available)
        if self.embedding_fn:
            try:
                topic_similarity = current_session.calculate_topic_similarity(
                    user, self.embedding_fn
                )
                if topic_similarity < self.config.topic_change_threshold:
                    return True
            except Exception:
                # If topic similarity fails, don't force new session
                pass
        
        return False
    
    def _create_new_session(self, start_time: float) -> str:
        """Create a new session and return its ID.
        
        Args:
            start_time: Session creation timestamp
            
        Returns:
            New session ID
        """
        session_id = f"session_{int(start_time)}_{len(self._sessions)}"
        session = Session(
            session_id=session_id,
            start_time=start_time,
            config=self.config
        )
        self._sessions[session_id] = session
        return session_id
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from memory and restart cleanup timer."""
        current_time = time.time()
        expired_sessions = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.is_expired(current_time):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self._sessions[session_id]
                if self._current_session_id == session_id:
                    self._current_session_id = None
        
        # Restart cleanup timer
        self._start_cleanup_timer()
    
    def _start_cleanup_timer(self) -> None:
        """Start the automatic cleanup timer."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        self._cleanup_timer = threading.Timer(
            self.config.cleanup_interval_minutes * 60,
            self._cleanup_expired_sessions
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _load_sessions(self) -> None:
        """Load persisted sessions from storage (placeholder implementation)."""
        # TODO: Implement session persistence if needed
        # For now, sessions are ephemeral and don't persist across restarts
        pass
    
    def _save_sessions(self) -> None:
        """Save sessions to persistent storage (placeholder implementation)."""
        # TODO: Implement session persistence if needed
        # For now, sessions are ephemeral and don't persist across restarts
        pass
    
    def cleanup(self) -> None:
        """Clean up resources and stop timers."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

# Enhanced Episodic Memory

from .memory import MemoryBackend

class EnhancedEpisodicMemory(MemoryBackend):
    """Enhanced episodic memory with contextual chunking and temporal weighting.
    
    Extends the base episodic memory system with sophisticated chunking,
    temporal decay, access-based reinforcement, and session integration.
    Maintains full backward compatibility with existing MemoryRecord objects.
    
    Research Foundation:
    - Parent-child chunking for improved context retrieval
    - Temporal decay and reinforcement learning from MemoriPY
    - MemGPT-inspired hierarchical memory organization
    - Anthropic contextual retrieval techniques
    """
    
    def __init__(
        self,
        storage_path: Path,
        *,
        embedding_fn: Optional[EmbeddingFunction] = None,
        chunker: Optional[ContextualChunker] = None,
        temporal_weighting: Optional[TemporalMemoryWeighting] = None,
        session_memory: Optional[SessionMemory] = None,
        max_records: int = 1000
    ):
        """Initialize enhanced episodic memory system.
        
        Args:
            storage_path: Path to persistent storage file
            embedding_fn: Function for generating vector embeddings
            chunker: Contextual chunker for conversation splitting
            temporal_weighting: Temporal decay and reinforcement system
            session_memory: Session memory for intermediate storage
            max_records: Maximum number of records to maintain
        """
        self.storage_path = storage_path.expanduser()
        self.embedding_fn = embedding_fn
        self.max_records = max(100, max_records)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=str(self.storage_path.parent / "vector_store"))
        self.collection = self.chroma_client.get_or_create_collection(name="episodic_memory")
        
        # Initialize embedder if not provided
        if not self.embedding_fn:
            self.embedder = SentenceTransformer('intfloat/e5-large-v2')
            self.embedding_fn = lambda text: self.embedder.encode(text).tolist()
        
        # Enhanced memory components
        self.chunker = chunker or ContextualChunker()
        self.temporal_weighting = temporal_weighting or TemporalMemoryWeighting()
        self.session_memory = session_memory
        
        # Record storage (keep in memory for now, but persist to Chroma)
        self._records: List[ChunkedMemoryRecord] = []
        self._chunk_index: Dict[str, List[str]] = {}  # parent_id -> child_ids mapping
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load memory records from ChromaDB, with migration from JSON if needed."""
        try:
            results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
            self._records = []
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                # Normalize parent_id: store as None for parents (Chroma persists '' for None)
                raw_parent_id = metadata.get('parent_id')
                parent_id = None if (raw_parent_id is None or str(raw_parent_id).strip() == "") else str(raw_parent_id)
                embedding = results['embeddings'][i]
                record = ChunkedMemoryRecord(
                    id=metadata.get('id', str(uuid.uuid4())),
                    parent_id=parent_id,
                    chunk_index=int(metadata.get('chunk_index', 0)),
                    user=metadata.get('user', ''),
                    assistant=metadata.get('assistant', ''),
                    content=doc,
                    timestamp=float(metadata.get('timestamp', time.time())),
                    embedding=list(embedding) if embedding is not None else None,
                    access_count=int(metadata.get('access_count', 0)),
                    last_accessed=float(metadata.get('last_accessed', time.time())),
                    importance_score=float(metadata.get('importance_score', 1.0))
                )
                self._records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load from Chroma: {e}")
            self._records = []
        
        # If no records in Chroma, try migrating from JSON
        if not self._records and self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and 'records' in raw_data:
                    legacy_records = raw_data['records']
                    for record_dict in legacy_records:
                        try:
                            record = ChunkedMemoryRecord(
                                id=record_dict.get('id', str(uuid.uuid4())),
                                user=record_dict.get('user', ''),
                                assistant=record_dict.get('assistant', ''),
                                content=f"User: {record_dict.get('user', '')}\nAssistant: {record_dict.get('assistant', '')}",
                                timestamp=float(record_dict.get('timestamp', time.time())),
                                embedding=record_dict.get('embedding')
                            )
                            self._records.append(record)
                        except Exception as e:
                            print(f"Warning: Failed to migrate record: {e}")
                            continue
                    
                    # Add migrated records to Chroma
                    if self._records:
                        ids = [r.id for r in self._records]
                        documents = [r.content for r in self._records]
                        metadatas = [{
                            'id': r.id,
                            'parent_id': r.parent_id or '',
                            'chunk_index': r.chunk_index,
                            'user': r.user,
                            'assistant': r.assistant,
                            'timestamp': r.timestamp,
                            'access_count': r.access_count or 0,
                            'last_accessed': r.last_accessed or r.timestamp,
                            'importance_score': r.importance_score or 1.0
                        } for r in self._records]
                        embeddings = [r.embedding for r in self._records if r.embedding]
                        if embeddings:
                            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                        print(f"Migrated {len(self._records)} records from JSON to ChromaDB.")
            except Exception as e:
                print(f"Warning: Migration failed: {e}")
        
        # Rebuild chunk index
        self._rebuild_chunk_index()
    
    def _migrate_legacy_records(self, legacy_data: List[Dict]) -> None:
        """Migrate legacy MemoryRecord format to ChunkedMemoryRecord."""
        migrated_records = []
        
        for record_data in legacy_data:
            try:
                # Create legacy MemoryRecord first
                legacy_record = MemoryRecord.from_dict(record_data)
                
                # Convert to enhanced format
                enhanced_record = ChunkedMemoryRecord.from_legacy_record(legacy_record)
                migrated_records.append(enhanced_record)
                
            except Exception as e:
                print(f"Warning: Failed to migrate legacy record: {e}")
                continue
        
        self._records = migrated_records
        print(f"Migrated {len(migrated_records)} legacy memory records")
    
    def _rebuild_chunk_index(self) -> None:
        """Rebuild the parent-child chunk index from loaded records."""
        self._chunk_index = {}
        
        for record in self._records:
            if record.is_parent:
                # Initialize parent entry
                if record.id not in self._chunk_index:
                    self._chunk_index[record.id] = []
            elif record.is_child and record.parent_id:
                # Add child to parent's index
                if record.parent_id not in self._chunk_index:
                    self._chunk_index[record.parent_id] = []
                self._chunk_index[record.parent_id].append(record.id)
    
    def _save(self) -> None:
        """Save enhanced memory records to persistent storage."""
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced format data
        enhanced_data = {
            'format_version': '2.0',
            'created_timestamp': time.time(),
            'total_records': len(self._records),
            'chunk_index': self._chunk_index,
            'records': [record.to_dict() for record in self._records]
        }
        
        # Atomic write with backup
        temp_path = self.storage_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            temp_path.replace(self.storage_path)
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _enforce_memory_limits(self) -> None:
        """Enforce maximum record limits with intelligent cleanup."""
        if len(self._records) <= self.max_records:
            return
        
        # Calculate how many records to remove
        excess = len(self._records) - self.max_records
        
        # Sort records by combined score (age + importance + access frequency)
        scored_records = []
        current_time = time.time()
        
        for record in self._records:
            # Calculate removal score (lower = more likely to be removed)
            temporal_score = self.temporal_weighting.calculate_temporal_weight(
                record.timestamp, "episodic"
            )
            importance_score = self.temporal_weighting.calculate_importance_multiplier(
                record.access_count, record.importance_score
            )
            
            # Combined score: newer, more important, more accessed = higher score
            removal_score = temporal_score * importance_score
            scored_records.append((removal_score, record))
        
        # Sort by score (ascending - lowest scores removed first)
        scored_records.sort(key=lambda x: x[0])
        
        # Remove lowest-scored records
        records_to_remove = [record for _, record in scored_records[:excess]]
        remaining_records = [record for _, record in scored_records[excess:]]
        
        # Handle parent-child relationships
        # If parent is removed, remove all children too
        # If child is removed but parent remains, keep parent
        parent_ids_to_remove = {r.id for r in records_to_remove if r.is_parent}
        
        final_records = []
        for record in remaining_records:
            # Skip children of removed parents
            if record.is_child and record.parent_id in parent_ids_to_remove:
                continue
            final_records.append(record)
        
        # Also remove children of any parents that were selected for removal
        for record in records_to_remove:
            if record.is_child and record.parent_id not in parent_ids_to_remove:
                # Keep the child if parent is staying
                final_records.append(record)
        
        self._records = final_records
        self._rebuild_chunk_index()
        
        print(f"Cleaned up {len(records_to_remove)} old memory records")
    
    def _find_record_by_id(self, record_id: str) -> Optional[ChunkedMemoryRecord]:
        """Find a record by its ID."""
        for record in self._records:
            if record.id == record_id:
                return record
        return None
    
    # MemoryBackend interface implementation
    
    def get_recent(self, limit: int) -> List[MemoryRecord]:
        """Get recent memory records in legacy format for compatibility.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent MemoryRecord objects (legacy format)
        """
        if limit <= 0:
            return []

        # Get recent parent records when available
        recent_records: List[MemoryRecord] = []
        seen_parent_ids = set()
        for record in reversed(self._records):
            if record.is_parent:
                recent_records.append(record.to_legacy_record())
                seen_parent_ids.add(record.id)
                if len(recent_records) >= limit:
                    return recent_records

        # Fallback: synthesize from child chunks if no parents (or not enough)
        if len(recent_records) < limit:
            for record in reversed(self._records):
                if not record.is_child:
                    continue
                # Skip if we already captured its parent via a parent record
                if record.parent_id and record.parent_id in seen_parent_ids:
                    continue
                # Try to extract the latest user/assistant pair from full_context
                user_text = ""
                assistant_text = ""
                ctx = record.full_context or record.content or ""
                try:
                    a_idx = ctx.rfind("Assistant:")
                    if a_idx != -1:
                        assistant_text = ctx[a_idx + len("Assistant:"):].strip()
                        # Find the preceding user
                        u_idx = ctx.rfind("User:", 0, a_idx)
                        if u_idx != -1:
                            user_text = ctx[u_idx + len("User:"):a_idx].strip()
                    else:
                        # Fallback to chunk content as assistant snippet
                        assistant_text = (record.content or "").strip()
                except Exception:
                    assistant_text = (record.content or "").strip()
                # Build a legacy MemoryRecord view
                legacy = MemoryRecord(
                    id=record.id,
                    user=user_text,
                    assistant=assistant_text,
                    timestamp=record.timestamp,
                    embedding=None,
                )
                recent_records.append(legacy)
                # Mark its parent to avoid duplicates from the same conversation
                if record.parent_id:
                    seen_parent_ids.add(record.parent_id)
                if len(recent_records) >= limit:
                    break

        return recent_records
    
    def recall(
        self, 
        query: str, 
        *, 
        top_k: int = 4,
        include_parents: bool = True,
        memory_type: str = "episodic"
    ) -> List[ChunkedMemoryRecord]:
        """Enhanced recall with chunked retrieval and temporal weighting.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            include_parents: Whether to include parent context for child chunks
            memory_type: Memory type for temporal weighting
            
        Returns:
            List of relevant ChunkedMemoryRecord objects with context
        """
        if top_k <= 0:
            return []
        
        if not self.embedding_fn:
            # Fallback to recent records if no embedding function
            return self._records[-top_k:] if self._records else []
        
        # Use ChromaDB for similarity search (embed query locally to match collection dim)
        try:
            query_vec = None
            if self.embedding_fn:
                try:
                    qemb = self.embedding_fn(query)
                    if qemb is not None:
                        query_vec = list(qemb)
                except Exception as ee:
                    print(f"Warning: Failed to embed query, falling back to recent: {ee}")
            if query_vec is None:
                return self._records[-top_k:] if self._records else []

            results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, include=['metadatas', 'distances'])
            top_ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
        except Exception as e:
            print(f"Warning: Chroma query failed: {e}")
            return self._records[-top_k:] if self._records else []
        
        # Get records by ids
        top_records = []
        for i, rec_id in enumerate(top_ids):
            record = self._find_record_by_id(rec_id)
            if record:
                # Apply temporal weighting to the similarity score
                similarity = 1 - distances[i]  # Chroma returns cosine distance, convert to similarity
                final_score = self.temporal_weighting.calculate_final_score(
                    similarity_score=similarity,
                    timestamp=record.timestamp,
                    access_count=record.access_count,
                    base_importance=record.importance_score,
                    memory_type=memory_type
                )
                top_records.append(record)
        
        # If not enough from Chroma, fallback to recent
        if len(top_records) < top_k and self._records:
            recent = [r for r in self._records[-top_k:] if r not in top_records]
            top_records.extend(recent[:top_k - len(top_records)])
        
        # Enhance results with parent context if requested
        if include_parents:
            enhanced_records = []
            for record in top_records:
                if record.is_child and record.parent_id:
                    # Find parent record for context
                    parent = self._find_record_by_id(record.parent_id)
                    if parent and parent.is_parent:
                        # Create enhanced record with full parent context
                        enhanced_record = ChunkedMemoryRecord(
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
                        enhanced_records.append(enhanced_record)
                    else:
                        enhanced_records.append(record)
                else:
                    enhanced_records.append(record)
            top_records = enhanced_records
        
        # Record access for reinforcement learning
        for record in top_records:
            record.access()
        
        return top_records
    
    def remember(self, user: str, assistant: str) -> List[ChunkedMemoryRecord]:
        """Store conversation with contextual chunking and temporal tracking.
        
        Args:
            user: User input text
            assistant: Assistant response text
            
        Returns:
            List of stored ChunkedMemoryRecord objects (parent + children)
        """
        if not user.strip() and not assistant.strip():
            return []
        
        # Add to session memory if available
        if self.session_memory:
            try:
                self.session_memory.add_turn(user, assistant)
            except Exception as e:
                print(f"Warning: Failed to add to session memory: {e}")
        
        # Generate chunks using contextual chunker
        try:
            chunks = self.chunker.chunk_conversation(user, assistant)
        except Exception as e:
            print(f"Warning: Chunking failed, using single record: {e}")
            # Fallback to single record
            single_record = ChunkedMemoryRecord(
                user=user,
                assistant=assistant,
                content=f"User: {user}\nAssistant: {assistant}",
                chunk_index=0
            )
            chunks = [single_record]
        
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
                except Exception as e:
                    print(f"Warning: Embedding generation failed: {e}")
                    # Continue without embedding
                    pass
        
        # Update chunk index for parent-child relationships
        if len(chunks) > 1:
            parent_id = chunks[0].id
            child_ids = [chunk.id for chunk in chunks[1:]]
            self._chunk_index[parent_id] = child_ids
        
        # Add to record storage
        self._records.extend(chunks)
        
        # Add to ChromaDB
        try:
            # Only add records that have embeddings to keep dimensions consistent
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                ids.append(chunk.id)
                documents.append(chunk.content)
                metadatas.append({
                    'id': chunk.id,
                    'parent_id': chunk.parent_id or '',
                    'chunk_index': chunk.chunk_index,
                    'user': chunk.user,
                    'assistant': chunk.assistant,
                    'timestamp': chunk.timestamp,
                    'access_count': chunk.access_count or 0,
                    'last_accessed': chunk.last_accessed or chunk.timestamp,
                    'importance_score': chunk.importance_score or 1.0
                })
                embeddings.append(chunk.embedding)
            if ids:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        except Exception as e:
            print(f"Warning: Failed to add to Chroma: {e}")
        
        # Enforce memory limits
        self._enforce_memory_limits()
        
        # Persist to storage (now Chroma handles persistence)
        try:
            self._save()
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")
        
        return chunks
    
    def clear(self) -> None:
        """Clear all memory records and persistent storage."""
        self._records = []
        self._chunk_index = {}
        
        # Clear ChromaDB collection
        try:
            self.chroma_client.delete_collection(name="episodic_memory")
            self.collection = self.chroma_client.create_collection(name="episodic_memory")
        except Exception as e:
            print(f"Warning: Failed to clear Chroma: {e}")
        
        if self.storage_path.exists():
            backup_path = self.storage_path.with_suffix('.backup')
            self.storage_path.rename(backup_path)
            print(f"Memory cleared, backup saved to {backup_path}")
    
    # Enhanced functionality

    def reindex(self, batch_size: int = 128) -> Dict[str, Any]:
        """Recompute embeddings for all records with current embedding_fn and rebuild Chroma.

        Args:
            batch_size: number of records to add to Chroma in each batch.

        Returns:
            Summary dict with counts and timing.
        """
        start = time.time()
        if not self.embedding_fn:
            return {"reindexed": 0, "skipped": len(self._records), "error": "No embedding_fn configured"}

        # Recompute embeddings consistently with remember()
        reindexed = 0
        skipped = 0
        for rec in self._records:
            try:
                embed_text = rec.content
                if rec.is_parent:
                    embed_text = f"User: {rec.user}\nAssistant: {rec.assistant}"
                emb = self.embedding_fn(embed_text)
                if emb:
                    rec.embedding = list(emb)
                    reindexed += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        # Rebuild Chroma collection
        try:
            self.chroma_client.delete_collection(name="episodic_memory")
        except Exception:
            # ignore if it doesn't exist
            pass
        self.collection = self.chroma_client.get_or_create_collection(name="episodic_memory")

        # Add in batches
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        def flush_batch():
            if ids:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        for rec in self._records:
            if rec.embedding is None:
                continue
            ids.append(rec.id)
            documents.append(rec.content)
            metadatas.append({
                'id': rec.id,
                'parent_id': rec.parent_id or '',
                'chunk_index': rec.chunk_index,
                'user': rec.user,
                'assistant': rec.assistant,
                'timestamp': rec.timestamp,
                'access_count': rec.access_count or 0,
                'last_accessed': rec.last_accessed or rec.timestamp,
                'importance_score': rec.importance_score or 1.0
            })
            embeddings.append(rec.embedding)
            if len(ids) >= batch_size:
                flush_batch()
                ids.clear(); documents.clear(); metadatas.clear(); embeddings.clear()

        flush_batch()
        elapsed = time.time() - start
        # Persist JSON backup with updated embeddings
        try:
            self._save()
        except Exception:
            pass

        return {"reindexed": reindexed, "skipped": skipped, "elapsed_sec": round(elapsed, 3)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics.
        
        Returns:
            Dictionary with memory usage and performance statistics
        """
        if not self._records:
            return {
                'total_records': 0,
                'parent_records': 0,
                'child_chunks': 0,
                'avg_chunks_per_conversation': 0.0,
                'storage_size_bytes': 0,
                'oldest_record_age_hours': 0.0,
                'newest_record_age_hours': 0.0
            }
        
        current_time = time.time()
        
        # Count record types
        parent_records = sum(1 for r in self._records if r.is_parent)
        child_chunks = sum(1 for r in self._records if r.is_child)
        
        # Calculate average chunks per conversation
        avg_chunks = (len(self._records) / parent_records) if parent_records > 0 else 0.0
        
        # Calculate age statistics
        timestamps = [r.timestamp for r in self._records]
        oldest_age = (current_time - min(timestamps)) / 3600
        newest_age = (current_time - max(timestamps)) / 3600
        
        # Estimate storage size
        try:
            storage_size = self.storage_path.stat().st_size if self.storage_path.exists() else 0
        except:
            storage_size = 0
        
        return {
            'total_records': len(self._records),
            'parent_records': parent_records,
            'child_chunks': child_chunks,
            'avg_chunks_per_conversation': round(avg_chunks, 2),
            'storage_size_bytes': storage_size,
            'oldest_record_age_hours': round(oldest_age, 1),
            'newest_record_age_hours': round(newest_age, 1)
        }
    
    def cleanup_expired_memories(self, current_time: Optional[float] = None) -> int:
        """Remove memories below forgetting threshold.
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Number of memories removed
        """
        if current_time is None:
            current_time = time.time()
        
        initial_count = len(self._records)
        
        # Filter out memories below forgetting threshold
        remaining_records = []
        for record in self._records:
            if not self.temporal_weighting.should_forget_memory(record.timestamp, "episodic"):
                remaining_records.append(record)
        
        self._records = remaining_records
        self._rebuild_chunk_index()
        
        removed_count = initial_count - len(remaining_records)
        
        if removed_count > 0:
            try:
                self._save()
            except Exception as e:
                print(f"Warning: Failed to save after cleanup: {e}")
        
        return removed_count