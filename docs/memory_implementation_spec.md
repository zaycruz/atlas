# Context Management Implementation Specification

## Overview

This document provides the technical specification for implementing hybrid token-aware context management in Atlas, based on research findings and optimal performance recommendations.

## Core Components

### 1. Enhanced Working Memory Manager

**Location**: `src/atlas_main/memory.py`

**New Configuration Class**:
```python
@dataclass
class WorkingMemoryConfig:
    """Configuration for hybrid token-aware working memory"""
    max_turns: int = 20                # Primary limit (conversation flow)
    token_budget: int = 96000          # 128K context - 32K buffer for system/tools
    max_token_budget: int = 120000     # Near full 128K context utilization
    enable_token_awareness: bool = True # Hybrid mode toggle
    preserve_important: bool = True     # Keep pinned/expanded turns
    eviction_strategy: str = "oldest_first"  # "oldest_first" | "least_important"
```

**Token Estimation Integration**:
```python
class HybridWorkingMemory:
    def __init__(self, config: WorkingMemoryConfig):
        self.config = config
        self.turns: List[Turn] = []
        self._token_cache: Dict[str, int] = {}
    
    def _estimate_tokens(self, text: str) -> int:
        """Use Atlas existing token estimation (4 chars ≈ 1 token)"""
        if text in self._token_cache:
            return self._token_cache[text]
        
        tokens = int(len(text) / 4)
        self._token_cache[text] = tokens
        return tokens
    
    def _calculate_total_tokens(self) -> int:
        """Calculate total tokens in working memory"""
        return sum(self._estimate_tokens(turn.content) for turn in self.turns)
    
    def should_evict(self) -> bool:
        """Check if eviction needed based on hybrid limits"""
        if len(self.turns) > self.config.max_turns:
            return True
        
        if self.config.enable_token_awareness:
            total_tokens = self._calculate_total_tokens()
            return total_tokens > self.config.token_budget
        
        return False
    
    def evict_turns(self) -> List[Turn]:
        """Evict turns using hybrid strategy"""
        evicted = []
        
        while self.should_evict() and self.turns:
            # Find oldest non-important turn
            for i, turn in enumerate(self.turns):
                if not (turn.pinned or turn.expanded or turn.recent_tool_use):
                    evicted.append(self.turns.pop(i))
                    break
            else:
                # If all turns are important, evict oldest
                if self.turns:
                    evicted.append(self.turns.pop(0))
        
        return evicted
```

### 2. Enhanced Memory Layer Configuration

**Location**: `src/atlas_main/memory_layers.py`

**Updated LayeredMemoryConfig**:
```python
@dataclass
class LayeredMemoryConfig:
    """Enhanced configuration with research-backed limits"""
    
    # Working Memory (hybrid token-aware)
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    
    # Episodic Layer (conversation history)
    max_episodic_records: int = 5000        # Increased from 2000
    episodic_chunk_size: int = 100          # For batch processing
    
    # Semantic Layer (facts/knowledge)
    prune_semantic_max_items: int = 800     # Increased from 400
    semantic_dedup_threshold: float = 0.9   # Similarity threshold
    
    # Reflection Layer (insights/lessons)
    prune_reflections_max_items: int = 400  # Increased from 200
    reflection_decay_factor: float = 0.95   # Temporal decay
    
    # Retrieval Parameters
    k_ep: int = 5                          # Episodic retrieval (was 3)
    k_facts: int = 5                       # Semantic retrieval (was 3)
    k_reflections: int = 4                 # Reflection retrieval (was 3)
    
    # Quality Gates
    min_fact_confidence: float = 0.7       # Filter low-confidence facts
    min_reflection_quality: float = 0.6    # Filter low-quality reflections
    enable_quality_gates: bool = True      # Quality filtering toggle
    
    # Performance Settings
    batch_size: int = 50                   # Batch processing size
    index_rebuild_threshold: int = 1000    # When to rebuild search indices
    cache_size: int = 100                  # LRU cache for frequent queries
```

### 3. Memory Monitoring System

**Location**: `src/atlas_main/ui.py`

**Enhanced Memory Panel**:
```python
class MemoryMonitor:
    """Real-time memory usage monitoring"""
    
    def __init__(self):
        self.stats = {
            'working_memory': {'turns': 0, 'tokens': 0, 'capacity_pct': 0},
            'episodic': {'records': 0, 'capacity_pct': 0},
            'semantic': {'facts': 0, 'capacity_pct': 0},
            'reflections': {'insights': 0, 'capacity_pct': 0},
            'quality_gates': {'facts_accepted': 0, 'reflections_accepted': 0}
        }
        self.events = deque(maxlen=100)
    
    def update_stats(self, memory_manager):
        """Update memory statistics from manager"""
        wm = memory_manager.working_memory
        config = memory_manager.config
        
        # Working memory stats
        self.stats['working_memory'] = {
            'turns': len(wm.turns),
            'tokens': wm._calculate_total_tokens(),
            'capacity_pct': (len(wm.turns) / config.working_memory.max_turns) * 100
        }
        
        # Layer stats
        self.stats['episodic']['records'] = len(memory_manager.episodic_layer)
        self.stats['episodic']['capacity_pct'] = (
            len(memory_manager.episodic_layer) / config.max_episodic_records
        ) * 100
        
        # Similar for semantic and reflections...
    
    def add_event(self, event_type: str, description: str):
        """Record memory event"""
        self.events.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'type': event_type,
            'description': description
        })
    
    def render_panel(self) -> Panel:
        """Render memory monitoring panel"""
        lines = []
        
        # Working memory
        wm_stats = self.stats['working_memory']
        lines.append(f"Working: {wm_stats['turns']}T/{wm_stats['tokens']}tok ({wm_stats['capacity_pct']:.1f}%)")
        
        # Layer summaries
        for layer in ['episodic', 'semantic', 'reflections']:
            stats = self.stats[layer]
            lines.append(f"{layer.title()}: {stats['records']} ({stats['capacity_pct']:.1f}%)")
        
        # Recent events
        lines.append("\nRecent Events:")
        for event in list(self.events)[-5:]:
            lines.append(f"[dim]{event['timestamp']}[/] {event['description']}")
        
        return Panel(
            "\n".join(lines),
            title="Memory Monitor",
            border_style="blue"
        )
```

### 4. Configuration Management

**Location**: `src/atlas_main/config/memory.yaml`

**Configuration File**:
```yaml
# Atlas Memory Configuration
# Based on research findings and optimal performance recommendations

memory_system:
  # Working Memory (hybrid token-aware)
  working_memory:
    max_turns: 20                    # Primary limit
    token_budget: 8000               # Secondary limit
    max_token_budget: 16000          # For large models
    enable_token_awareness: true     # Hybrid mode
    preserve_important: true         # Keep pinned turns
    eviction_strategy: "oldest_first"

  # Memory Layers
  layers:
    episodic:
      max_records: 5000              # Conversation history
      chunk_size: 100                # Batch processing
      retrieval_k: 5                 # Retrieved items
    
    semantic:
      max_items: 800                 # Facts/knowledge
      dedup_threshold: 0.9           # Similarity cutoff
      retrieval_k: 5                 # Retrieved facts
      min_confidence: 0.7            # Quality gate
    
    reflections:
      max_items: 400                 # Insights/lessons
      decay_factor: 0.95             # Temporal decay
      retrieval_k: 4                 # Retrieved insights
      min_quality: 0.6               # Quality gate

  # Performance Tuning
  performance:
    batch_size: 50                   # Processing batch size
    cache_size: 100                  # LRU cache size
    index_rebuild_threshold: 1000    # Search index rebuild
    enable_quality_gates: true       # Filter low-quality items

  # Monitoring
  monitoring:
    enable_ui_panel: true            # Memory panel in UI
    event_history_size: 100          # Memory event tracking
    stats_update_interval: 2.0      # UI refresh rate (seconds)

# Environment-specific overrides
environments:
  development:
    working_memory:
      max_turns: 15                  # Smaller for development
      token_budget: 6000
    
  production:
    working_memory:
      max_turns: 25                  # Larger for production
      token_budget: 12000
    layers:
      episodic:
        max_records: 10000           # More history in production
```

### 5. Implementation Timeline

**Phase 1: Core Infrastructure (Week 1)**
- [ ] Implement `HybridWorkingMemory` class
- [ ] Update `LayeredMemoryConfig` with new limits
- [ ] Create configuration loading system
- [ ] Add token estimation integration

**Phase 2: Memory Monitoring (Week 2)**
- [ ] Implement `MemoryMonitor` class
- [ ] Integrate monitoring panel into UI
- [ ] Add memory event tracking
- [ ] Create performance metrics collection

**Phase 3: Quality Gates (Week 3)**
- [ ] Implement confidence-based fact filtering
- [ ] Add quality-based reflection filtering
- [ ] Create adaptive threshold learning
- [ ] Add quality gate statistics

**Phase 4: Optimization & Testing (Week 4)**
- [ ] Performance benchmarking
- [ ] Memory usage optimization
- [ ] Configuration tuning
- [ ] Integration testing

### 6. Testing Strategy

**Unit Tests**:
```python
# tests/test_hybrid_memory.py
def test_token_estimation():
    """Test token estimation accuracy"""
    
def test_hybrid_eviction():
    """Test turn vs token eviction logic"""
    
def test_quality_gates():
    """Test confidence/quality filtering"""

def test_memory_monitoring():
    """Test stats collection and UI integration"""
```

**Integration Tests**:
```python
# tests/test_memory_integration.py
def test_end_to_end_memory_flow():
    """Test complete memory pipeline"""
    
def test_configuration_loading():
    """Test YAML config loading and validation"""
    
def test_ui_integration():
    """Test memory panel in UI"""
```

**Performance Tests**:
```python
# tests/test_memory_performance.py
def test_memory_latency():
    """Test retrieval and storage latency"""
    
def test_memory_scalability():
    """Test performance with large memory stores"""
    
def test_token_estimation_performance():
    """Test token calculation speed"""
```

### 7. Migration Strategy

**Backward Compatibility**:
- Existing memory stores remain functional
- New features are opt-in via configuration
- Graceful fallback to turn-based limits if token estimation fails
- Configuration migration utilities

**Data Migration**:
```python
def migrate_memory_config(old_config: dict) -> LayeredMemoryConfig:
    """Migrate old configuration to new format"""
    # Map old settings to new structure
    # Apply research-backed defaults
    # Preserve user customizations
```

### 8. Performance Benchmarks

**Target Metrics for 128K Context Models**:
- Memory retrieval latency: < 50ms p95
- Context assembly time: < 100ms p95
- Token estimation accuracy: > 85%
- Memory usage efficiency: < 500MB for 20K records
- UI responsiveness: < 2s refresh rate
- **Working memory capacity: 96K tokens (75% of 128K context)**

**Monitoring Alerts**:
- Memory usage > 90% capacity
- Retrieval latency > 100ms
- Quality gate rejection rate > 50%
- Token estimation cache miss rate > 30%

---

## Implementation Notes

### Key Design Decisions

1. **Hybrid Approach**: Use turns as primary limit (conversation flow) with tokens as secondary (efficiency)
2. **Backward Compatibility**: Ensure existing setups continue working without changes
3. **Graceful Degradation**: Fall back to turn-based limits if token estimation fails
4. **Research Alignment**: Follow MemGPT/Letta patterns for quality gating and hierarchical memory
5. **Performance Focus**: Optimize for real-time usage with sub-100ms retrieval

### Risk Mitigation

1. **Token Estimation Accuracy**: Use simple but proven 4-char approach with caching
2. **Memory Bloat**: Implement aggressive quality gates and usage-based pruning
3. **UI Performance**: Separate monitoring from core memory operations
4. **Configuration Complexity**: Provide sensible defaults and environment-specific overrides
5. **Migration Issues**: Extensive testing and gradual rollout strategy

---

Last Updated: October 4, 2025  
Document Status: Technical Specification Complete  
Implementation Phase: Ready for Development
