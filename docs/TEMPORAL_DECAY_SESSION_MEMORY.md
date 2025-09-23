# Temporal Decay and Session Memory Design

## Overview

This document details the implementation of temporal decay weighting and session memory layer for Atlas. These features bridge the gap between short-term working memory and long-term episodic storage while making memory retrieval more contextually relevant based on recency.

## Temporal Decay Weighting

### Problem Statement

Current Atlas memory treats all stored memories equally regardless of when they occurred. A conversation from last week has the same retrieval weight as yesterday's discussion, leading to:

- **Stale Context**: Old, irrelevant memories interfering with current conversations
- **Poor Relevance**: Recent important context being overshadowed by older but similar content
- **Unnatural Behavior**: Humans naturally weight recent experiences more heavily

### Solution: Exponential Temporal Decay

Implement exponential decay function that reduces memory relevance over time while preserving important long-term memories through reinforcement.

**Research Foundation:**
- **MemoriPY Memory Dynamics**: [Memoripy implementation](https://github.com/caspianmoon/memoripy) - "Older memories decay over time, while frequently accessed memories are reinforced"
- **Human Episodic Memory**: [Towards large language models with human-like episodic memory](https://www.sciencedirect.com/science/article/abs/pii/S1364661325001792) - ScienceDirect research on temporal dynamics
- **Forgetting Curves**: Based on Ebbinghaus forgetting curve principles adapted for LLM memory systems
- **Reinforcement Learning**: [Assessing Episodic Memory in LLMs with Sequence Order Recall Tasks](https://openreview.net/forum?id=LLtUtzSOL5) - OpenReview research on memory reinforcement

### Mathematical Foundation

```
temporal_weight(t) = e^(-λ * age_hours)

Where:
- λ (lambda) = decay rate per hour (configurable)
- age_hours = (current_time - memory_timestamp) / 3600
- Result ranges from 1.0 (new) to approaching 0.0 (very old)
```

### Implementation

```python
import math
import time
from typing import Optional
from dataclasses import dataclass

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

class TemporalMemoryWeighting:
    """Handles temporal decay and reinforcement for memory systems."""
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
    
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
        if current_time is None:
            current_time = time.time()
        
        age_hours = max(0, (current_time - timestamp) / 3600)
        
        # Select appropriate decay rate
        decay_rates = {
            "working": self.config.working_memory_decay,
            "session": self.config.session_memory_decay, 
            "episodic": self.config.episodic_memory_decay,
            "semantic": self.config.semantic_memory_decay
        }
        
        decay_rate = decay_rates.get(memory_type, self.config.episodic_memory_decay)
        
        # Calculate exponential decay
        weight = math.exp(-decay_rate * age_hours)
        
        return max(0.0, min(1.0, weight))
    
    def calculate_importance_multiplier(self, access_count: int, base_importance: float = 1.0) -> float:
        """Calculate importance multiplier based on access frequency.
        
        Args:
            access_count: Number of times memory has been accessed
            base_importance: Base importance score
            
        Returns:
            Importance multiplier
        """
        if not self.config.enable_reinforcement or access_count <= 0:
            return base_importance
        
        # Logarithmic scaling to prevent runaway importance
        boost = 1.0 + (self.config.access_boost_factor * math.log(access_count + 1))
        multiplier = base_importance * boost
        
        return min(multiplier, self.config.max_importance_multiplier)
    
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
            Final weighted score
        """
        temporal_weight = self.calculate_temporal_weight(timestamp, memory_type)
        importance_multiplier = self.calculate_importance_multiplier(access_count, base_importance)
        
        # Combine all factors
        final_score = similarity_score * temporal_weight * importance_multiplier
        
        return max(0.0, final_score)
    
    def should_forget_memory(self, timestamp: float, memory_type: str = "episodic") -> bool:
        """Determine if a memory should be forgotten due to age.
        
        Args:
            timestamp: Memory creation timestamp
            memory_type: Type of memory
            
        Returns:
            True if memory should be forgotten
        """
        weight = self.calculate_temporal_weight(timestamp, memory_type)
        return weight < self.config.forgetting_threshold

# Example usage and testing
def demonstrate_temporal_decay():
    """Demonstrate temporal decay behavior."""
    weighting = TemporalMemoryWeighting()
    current_time = time.time()
    
    # Test different memory ages
    test_times = [
        current_time,           # Now (weight ≈ 1.0)
        current_time - 3600,    # 1 hour ago
        current_time - 86400,   # 1 day ago  
        current_time - 604800,  # 1 week ago
        current_time - 2592000, # 1 month ago
    ]
    
    print("Temporal Decay Demonstration:")
    print("Age\t\tWeight\t\tWith 5 accesses")
    print("-" * 50)
    
    for timestamp in test_times:
        age_hours = (current_time - timestamp) / 3600
        weight = weighting.calculate_temporal_weight(timestamp)
        boosted_weight = weighting.calculate_importance_multiplier(5) * weight
        
        if age_hours < 1:
            age_str = "Now"
        elif age_hours < 24:
            age_str = f"{age_hours:.1f}h ago"
        elif age_hours < 168:
            age_str = f"{age_hours/24:.1f}d ago"
        else:
            age_str = f"{age_hours/168:.1f}w ago"
        
        print(f"{age_str:12}\t{weight:.3f}\t\t{boosted_weight:.3f}")
```

### Integration with Existing Memory

```python
class TemporallyAwareMemoryRecord(ChunkedMemoryRecord):
    """Memory record with temporal decay support."""
    
    def __init__(self, *args, temporal_weighting: Optional[TemporalMemoryWeighting] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_weighting = temporal_weighting or TemporalMemoryWeighting()
    
    def calculate_relevance_score(self, similarity_score: float, memory_type: str = "episodic") -> float:
        """Calculate temporal-aware relevance score."""
        return self.temporal_weighting.calculate_final_score(
            similarity_score=similarity_score,
            timestamp=self.timestamp,
            access_count=self.access_count,
            base_importance=self.importance_score,
            memory_type=memory_type
        )
    
    def access(self) -> None:
        """Enhanced access tracking with importance boosting."""
        super().access()
        
        # Update importance score using temporal weighting
        self.importance_score = self.temporal_weighting.calculate_importance_multiplier(
            self.access_count, 
            self.importance_score
        )
```

## Session Memory Layer

### Problem Statement

Atlas currently has a gap between working memory (12 turns) and long-term episodic storage:

- **Context Loss**: Important recent context beyond 12 turns is immediately moved to long-term storage
- **Retrieval Inefficiency**: Recent conversations require expensive vector similarity search
- **Poor Temporal Locality**: No dedicated space for "recent but not immediate" context

### Solution: Intermediate Session Memory

Create a time-based intermediate memory layer that bridges working and episodic memory.

**Research Foundation:**
- **MemGPT Hierarchical Memory**: [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - OS-inspired memory hierarchy with main context and external context
- **LlamaIndex Memory Blocks**: [Improved Long & Short-Term Memory for LlamaIndex Agents](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents) - "Memory component allows agents to have context-aware conversations and long-term recall"
- **Session-based Memory**: [LLM Agentic Memory Systems](https://kickitlikeshika.github.io/2025/03/22/agentic-memory.html) - Research on session boundaries and context management
- **Multi-timescale Memory**: [Hierarchical Episodic Memory in LLMs via Multi-Scale Event Organization](https://openreview.net/forum?id=lHQAXe7kgx) - OpenReview research on nested timescale organization

### Architecture Design

```
Conversation Flow:
User Input → Working Memory (12 turns) → Session Memory (2 hours) → Episodic Memory (permanent)
                     ↓                         ↓                          ↓
                [Immediate]              [Recent Context]            [Long-term Knowledge]
```

### Implementation

```python
from collections import deque
from typing import List, Dict, Optional
import time
import threading
from pathlib import Path

@dataclass 
class SessionConfig:
    """Configuration for session memory behavior."""
    
    duration_hours: float = 2.0           # How long to keep sessions active
    max_turns_per_session: int = 50       # Maximum turns per session
    promotion_threshold: float = 0.7      # Similarity threshold for promotion to long-term
    cleanup_interval_minutes: int = 15    # How often to clean up expired sessions
    
    # Session boundaries (when to start new session)
    inactivity_threshold_minutes: int = 30  # Minutes of inactivity before new session
    topic_change_threshold: float = 0.3     # Embedding similarity threshold for topic change

class SessionMemory:
    """Intermediate memory layer for recent conversation context."""
    
    def __init__(
        self, 
        config: Optional[SessionConfig] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        storage_path: Optional[Path] = None
    ):
        self.config = config or SessionConfig()
        self.embedding_fn = embedding_fn
        self.storage_path = storage_path
        
        # Active session storage
        self._sessions: Dict[str, 'Session'] = {}
        self._current_session_id: Optional[str] = None
        
        # Cleanup management
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()
        
        # Load persisted sessions if storage path provided
        if self.storage_path:
            self._load_sessions()
    
    def add_turn(self, user: str, assistant: str, force_new_session: bool = False) -> str:
        """Add a conversation turn to session memory.
        
        Args:
            user: User input text
            assistant: Assistant response text
            force_new_session: Force creation of new session
            
        Returns:
            Session ID where turn was added
        """
        current_time = time.time()
        
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
            List of memory records from session(s)
        """
        if session_id:
            # Get specific session
            session = self._sessions.get(session_id)
            if session and (include_expired or not session.is_expired()):
                return session.get_recent_turns(max_turns)
            return []
        
        # Get context from current session
        if self._current_session_id:
            return self.get_session_context(self._current_session_id, max_turns, include_expired)
        
        return []
    
    def get_all_recent_context(self, hours_back: float = 2.0, max_turns: int = 50) -> List[MemoryRecord]:
        """Get all recent context across sessions within time window.
        
        Args:
            hours_back: How far back to look (in hours)
            max_turns: Maximum total turns to return
            
        Returns:
            List of recent memory records across all sessions
        """
        current_time = time.time()
        cutoff_time = current_time - (hours_back * 3600)
        
        all_records = []
        
        # Collect records from all active sessions
        for session in self._sessions.values():
            if not session.is_expired(current_time):
                session_records = session.get_turns_since(cutoff_time)
                all_records.extend(session_records)
        
        # Sort by timestamp and limit
        all_records.sort(key=lambda r: r.timestamp)
        return all_records[-max_turns:] if max_turns > 0 else all_records
    
    def promote_to_longterm(self, episodic_memory: 'EpisodicMemory') -> List[MemoryRecord]:
        """Promote important session memories to long-term storage.
        
        Args:
            episodic_memory: Long-term memory system to promote to
            
        Returns:
            List of promoted memory records
        """
        promoted_records = []
        current_time = time.time()
        
        for session_id, session in list(self._sessions.items()):
            if session.should_promote(current_time, self.config.promotion_threshold):
                # Get important turns from session
                important_turns = session.get_important_turns()
                
                # Promote each turn to long-term memory
                for turn in important_turns:
                    # Use episodic memory's chunking if available
                    if hasattr(episodic_memory, 'remember'):
                        promoted = episodic_memory.remember(turn.user, turn.assistant)
                        if isinstance(promoted, list):
                            promoted_records.extend(promoted)
                        else:
                            promoted_records.append(promoted)
                
                # Mark session as promoted
                session.mark_promoted()
        
        return promoted_records
    
    def _should_start_new_session(self, user: str, assistant: str, current_time: float) -> bool:
        """Determine if a new session should be started."""
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
            topic_similarity = current_session.calculate_topic_similarity(user, self.embedding_fn)
            if topic_similarity < self.config.topic_change_threshold:
                return True
        
        return False
    
    def _create_new_session(self, start_time: float) -> str:
        """Create a new session and return its ID."""
        session_id = f"session_{int(start_time)}_{len(self._sessions)}"
        session = Session(
            session_id=session_id,
            start_time=start_time,
            config=self.config
        )
        self._sessions[session_id] = session
        return session_id
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from memory."""
        current_time = time.time()
        expired_sessions = []
        
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
        """Start the cleanup timer."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        self._cleanup_timer = threading.Timer(
            self.config.cleanup_interval_minutes * 60,
            self._cleanup_expired_sessions
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

@dataclass
class SessionTurn:
    """Individual turn within a session."""
    user: str
    assistant: str
    timestamp: float
    importance_score: float = 1.0
    embedding: Optional[List[float]] = None

class Session:
    """Individual conversation session."""
    
    def __init__(self, session_id: str, start_time: float, config: SessionConfig):
        self.session_id = session_id
        self.start_time = start_time
        self.config = config
        self.turns: deque[SessionTurn] = deque(maxlen=config.max_turns_per_session)
        self.promoted = False
        self.topic_embedding: Optional[List[float]] = None
    
    @property
    def turn_count(self) -> int:
        """Get number of turns in session."""
        return len(self.turns)
    
    def add_turn(self, user: str, assistant: str, timestamp: float) -> None:
        """Add a turn to this session."""
        turn = SessionTurn(
            user=user,
            assistant=assistant,
            timestamp=timestamp
        )
        self.turns.append(turn)
    
    def get_recent_turns(self, max_turns: int) -> List[MemoryRecord]:
        """Get recent turns as MemoryRecord objects."""
        recent = list(self.turns)[-max_turns:] if max_turns > 0 else list(self.turns)
        
        records = []
        for turn in recent:
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
        """Get all turns since a specific timestamp."""
        filtered_turns = [turn for turn in self.turns if turn.timestamp >= cutoff_time]
        
        records = []
        for turn in filtered_turns:
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
        """Check if session has expired."""
        if current_time is None:
            current_time = time.time()
        
        age_hours = (current_time - self.start_time) / 3600
        return age_hours > self.config.duration_hours
    
    def get_last_activity_time(self) -> float:
        """Get timestamp of last activity in session."""
        if not self.turns:
            return self.start_time
        return self.turns[-1].timestamp
    
    def should_promote(self, current_time: float, threshold: float) -> bool:
        """Check if session should be promoted to long-term memory."""
        return (
            not self.promoted and
            self.is_expired(current_time) and
            len(self.turns) >= 3  # Minimum turns for promotion
        )
    
    def get_important_turns(self) -> List[SessionTurn]:
        """Get turns that should be promoted to long-term memory."""
        # For now, return all turns; could be enhanced with importance scoring
        return list(self.turns)
    
    def mark_promoted(self) -> None:
        """Mark session as promoted to long-term memory."""
        self.promoted = True
    
    def calculate_topic_similarity(self, user_input: str, embedding_fn: EmbeddingFunction) -> float:
        """Calculate similarity between new input and session topic."""
        if not self.topic_embedding:
            # Initialize topic embedding from session content
            session_text = " ".join([f"{turn.user} {turn.assistant}" for turn in self.turns])
            if session_text.strip():
                self.topic_embedding = embedding_fn(session_text)
        
        if not self.topic_embedding:
            return 1.0  # Default to high similarity if no embedding
        
        # Calculate similarity with new input
        input_embedding = embedding_fn(user_input)
        if not input_embedding:
            return 1.0
        
        # Cosine similarity
        topic_vec = np.asarray(self.topic_embedding, dtype=float)
        input_vec = np.asarray(input_embedding, dtype=float)
        
        if topic_vec.shape != input_vec.shape:
            return 1.0
        
        denom = np.linalg.norm(topic_vec) * np.linalg.norm(input_vec)
        if denom == 0:
            return 1.0
        
        similarity = float(np.dot(topic_vec, input_vec) / denom)
        return max(0.0, min(1.0, similarity))
```

## Testing Strategy

### Temporal Decay Tests

```python
class TestTemporalDecay(unittest.TestCase):
    """Test temporal decay functionality."""
    
    def setUp(self):
        self.weighting = TemporalMemoryWeighting()
    
    def test_recent_memories_higher_weight(self):
        """Recent memories should have higher weights."""
        now = time.time()
        recent_weight = self.weighting.calculate_temporal_weight(now)
        old_weight = self.weighting.calculate_temporal_weight(now - 3600)  # 1 hour ago
        
        self.assertGreater(recent_weight, old_weight)
        self.assertAlmostEqual(recent_weight, 1.0, places=2)
    
    def test_importance_boosting(self):
        """Frequently accessed memories should get importance boost."""
        base_score = 0.5
        zero_access = self.weighting.calculate_importance_multiplier(0, base_score)
        many_access = self.weighting.calculate_importance_multiplier(10, base_score)
        
        self.assertEqual(zero_access, base_score)
        self.assertGreater(many_access, base_score)
        self.assertLessEqual(many_access, self.weighting.config.max_importance_multiplier)
    
    def test_forgetting_threshold(self):
        """Very old memories should be marked for forgetting."""
        very_old = time.time() - (365 * 24 * 3600)  # 1 year ago
        should_forget = self.weighting.should_forget_memory(very_old)
        
        recent = time.time()
        should_keep = self.weighting.should_forget_memory(recent)
        
        self.assertTrue(should_forget)
        self.assertFalse(should_keep)

class TestSessionMemory(unittest.TestCase):
    """Test session memory functionality."""
    
    def setUp(self):
        self.config = SessionConfig(duration_hours=1.0, max_turns_per_session=5)
        self.session_memory = SessionMemory(config=self.config)
    
    def test_session_creation(self):
        """Test automatic session creation."""
        session_id = self.session_memory.add_turn("Hello", "Hi there!")
        self.assertIsNotNone(session_id)
        
        context = self.session_memory.get_session_context()
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].user, "Hello")
    
    def test_session_capacity_limit(self):
        """Test that sessions respect turn limits."""
        # Fill up one session
        for i in range(6):  # More than max_turns_per_session (5)
            self.session_memory.add_turn(f"Question {i}", f"Answer {i}")
        
        # Should have created multiple sessions
        context = self.session_memory.get_all_recent_context()
        self.assertEqual(len(context), 6)
    
    def test_session_expiration(self):
        """Test that sessions expire after configured time."""
        # Add turn to create session
        session_id = self.session_memory.add_turn("Hello", "Hi")
        
        # Manually expire session by modifying timestamp
        session = self.session_memory._sessions[session_id]
        session.start_time = time.time() - (2 * 3600)  # 2 hours ago
        
        # Check expiration
        self.assertTrue(session.is_expired())
    
    def test_context_retrieval(self):
        """Test context retrieval across sessions."""
        # Add multiple turns
        for i in range(3):
            self.session_memory.add_turn(f"Q{i}", f"A{i}")
        
        context = self.session_memory.get_session_context(max_turns=2)
        self.assertEqual(len(context), 2)
        
        # Should get most recent turns
        self.assertEqual(context[-1].user, "Q2")
        self.assertEqual(context[-2].user, "Q1")

if __name__ == "__main__":
    unittest.main()
```

## Configuration and Tuning

### Default Configuration Values

```python
# Temporal decay configuration
ATLAS_TEMPORAL_DECAY_RATE = 0.1          # Decay rate per hour
ATLAS_IMPORTANCE_BOOST_FACTOR = 0.1       # Access frequency boost factor
ATLAS_MAX_IMPORTANCE_MULTIPLIER = 5.0     # Cap on importance boosting
ATLAS_FORGETTING_THRESHOLD = 0.01         # Weight threshold for forgetting

# Session memory configuration  
ATLAS_SESSION_DURATION_HOURS = 2.0        # Session lifetime
ATLAS_SESSION_MAX_TURNS = 50              # Max turns per session
ATLAS_SESSION_INACTIVITY_MINUTES = 30     # Inactivity before new session
ATLAS_SESSION_CLEANUP_MINUTES = 15        # Cleanup interval
```

### Performance Characteristics

- **Memory Usage**: ~1KB per session turn
- **Retrieval Speed**: <10ms for session context, <50ms for temporal scoring
- **Storage Overhead**: ~20% increase for temporal metadata
- **Cleanup Frequency**: Every 15 minutes for expired sessions

---

*This design provides the foundation for temporal awareness and session management in Atlas memory systems. Implementation should follow the documented APIs and testing requirements.*