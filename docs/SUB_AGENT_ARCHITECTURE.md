# Sub-Agent Architecture: Multi-Agent Planning & Parallel Execution

**Status**: Design Phase  
**Date**: 2025-10-26  
**Goal**: Enable Atlas to orchestrate complex multi-step tasks by collaborating with a planner agent and spawning multiple coding agents that work in parallel on separate branches.

---

## Executive Summary

Transform Atlas from a single-agent system into a hierarchical multi-agent orchestrator capable of:
1. **Intelligent Planning**: Collaborate with reasoning models (DeepSeek-R1, Sonnet 4.5, GPT-5) to decompose complex objectives
2. **Parallel Execution**: Spawn multiple coding agents working simultaneously on different features/modules
3. **Branch Isolation**: Each agent works on dedicated branches to prevent conflicts
4. **Iterative Refinement**: Feedback loops allow plan revision based on execution results

---

## Current State

### Existing Components
- **Atlas Agent** (`agent.py`): Main conversational agent with tools
- **SimplePlanner** (`orchestrator/planner.py`): Hardcoded 3-step plans (plan → implement → test)
- **Orchestrator** (`orchestrator/engine.py`): Sequential DAG execution, no parallelization
- **AgentFactory** (`agents/factory.py`): Spawns codex, claude-code, droid agents
- **Delegation Tools**:
  - `delegate_task`: Fire-and-forget task execution
  - `agent_session`: Interactive streaming sessions with agents

### Current Limitations
1. Planning is rule-based, not context-aware
2. No parallel agent execution
3. No branch management for concurrent work
4. No feedback-based plan revision
5. No collaboration between Atlas and planner
6. Risk of infinite conversation loops in agent sessions

---

## Proposed Architecture

### High-Level Flow

```
User Request → Atlas (triage)
    ↓
    ├─→ [Simple Task] → Direct execution by Atlas
    ↓
    └─→ [Complex Task] → Planner Collaboration
            ↓
            1. Codebase Analysis (Atlas tools)
            2. Planning Session (Atlas ↔ Planner Agent)
               - Round-limited conversation (max 5-10 rounds)
               - DeepSeek-R1 (reasoning) or Sonnet/GPT-5 (tools+reasoning)
            3. Plan Generation (DAG with parallelization markers)
            ↓
            Enhanced Orchestrator
            ↓
            ├─→ Agent 1 (branch: feature/frontend-auth) → Frontend auth UI
            ├─→ Agent 2 (branch: feature/backend-auth) → Backend auth API
            └─→ Agent 3 (branch: feature/auth-tests) → Integration tests
            ↓
            Parallel Execution Monitor
            ↓
            Results Aggregation & Feedback
            ↓
            [Success] → Merge branches / [Failure] → Replan with feedback
```

---

## Component Design

### 1. Planner Agent Interface

**Purpose**: Reasoning-focused agent that analyzes objectives and creates execution plans.

**Implementation Options**:
- **Option A**: DeepSeek-R1 via Ollama (pure reasoning, no tools)
- **Option B**: Sonnet 4.5 / GPT-5 via API (reasoning + tools)
- **Hybrid**: Start with DeepSeek-R1, fallback to API-based for tool needs

**Integration Point**: 
```python
# New agent adapter in agents/reasoning.py
class ReasoningAgentAdapter(AgentAdapter):
    """Adapter for reasoning models (DeepSeek-R1, Sonnet, GPT-5)"""
    
    def __init__(self, model: str, provider: str = "ollama"):
        self.model = model  # "deepseek-r1", "claude-sonnet-4.5", "gpt-5"
        self.provider = provider  # "ollama", "anthropic", "openai"
```

**Configuration** (`config/agents.yaml`):
```yaml
planner-deepseek:
  description: DeepSeek-R1 reasoning planner via Ollama
  type: reasoning
  provider: ollama
  model: deepseek-r1:latest
  timeout: 300
  
planner-sonnet:
  description: Claude Sonnet 4.5 planner with tools via Anthropic API
  type: reasoning
  provider: anthropic
  model: claude-sonnet-4.5
  timeout: 300
  max_rounds: 8
```

### 2. Planning Session Manager

**Purpose**: Manage bounded conversations between Atlas and planner to prevent infinite loops.

**Key Features**:
- **Round Limiting**: Max 5-10 conversation turns
- **Context Injection**: Codebase structure, git status, file summaries
- **Session Protocol**:
  1. Atlas provides: objective, codebase context, constraints
  2. Planner asks clarifying questions (optional, counted towards rounds)
  3. Planner proposes plan (DAG with steps, agents, dependencies, branch strategy)
  4. Atlas validates plan feasibility
  5. Planner refines (if needed)

**New Class** (`orchestrator/planning_session.py`):
```python
class PlanningSession:
    def __init__(
        self,
        planner_agent_id: str,
        max_rounds: int = 8,
        codebase_context: Optional[Dict] = None
    ):
        self.max_rounds = max_rounds
        self.rounds_used = 0
        # ...
    
    async def collaborate(self, objective: str) -> Plan:
        """Run bounded planning conversation."""
        # Round 1: Atlas sends objective + context
        # Rounds 2-N: Planner asks questions, proposes plan
        # Return: Validated Plan object
```

### 3. Enhanced Orchestrator

**Current**: Sequential execution with DAG dependencies  
**Target**: Parallel execution with branch isolation

**New Capabilities**:
- **Parallel Step Execution**: Run independent steps concurrently
- **Branch Management**: Create/switch/merge branches for each agent
- **Dynamic Agent Spawning**: Scale based on parallelizable work
- **Progress Monitoring**: Real-time status across parallel agents
- **Failure Handling**: Rollback branches, replan on failure

**Enhanced Orchestrator** (`orchestrator/parallel_engine.py`):
```python
class ParallelOrchestrator(Orchestrator):
    async def run_task(self, task: TaskSpec) -> TaskResult:
        # Analyze DAG for parallel opportunities
        parallel_groups = self._identify_parallel_steps(task.steps)
        
        for group in parallel_groups:
            if len(group) > 1:
                # Parallel execution
                results = await self._run_parallel_group(group, task)
            else:
                # Sequential execution
                results = await self._run_step(group[0], task)
    
    async def _run_parallel_group(
        self, 
        steps: List[StepSpec], 
        task: TaskSpec
    ) -> List[StepResult]:
        # 1. Create branches for each step
        # 2. Spawn agents in parallel via asyncio.gather
        # 3. Monitor progress
        # 4. Collect results
```

### 4. Branch Strategy Manager

**Purpose**: Coordinate branch creation, isolation, and merging for parallel agents.

**Strategy**:
- Base branch: `main` or current working branch
- Agent branches: `atlas/task-{task_id}/step-{step_id}`
- Example:
  - `atlas/task-abc123/frontend-auth`
  - `atlas/task-abc123/backend-auth`
  - `atlas/task-abc123/integration-tests`

**New Module** (`orchestrator/branches.py`):
```python
class BranchStrategy:
    async def create_step_branch(
        self, 
        repo_path: Path, 
        task_id: str, 
        step_id: str,
        base_branch: str = "main"
    ) -> str:
        """Create isolated branch for agent work."""
        branch_name = f"atlas/{task_id}/{step_id}"
        # git checkout -b {branch_name} {base_branch}
        return branch_name
    
    async def merge_step_results(
        self,
        repo_path: Path,
        branches: List[str],
        strategy: str = "sequential"  # or "octopus"
    ) -> MergeResult:
        """Merge completed agent branches."""
        # Handle conflicts, create merge commits
```

### 5. Codebase Context Provider

**Purpose**: Provide planner with structural understanding of the codebase.

**Context Elements**:
- Directory structure
- Key file summaries
- Technology stack detection
- Dependency graph
- Recent git history
- Current git status

**New Module** (`orchestrator/context.py`):
```python
class CodebaseContext:
    async def analyze(self, repo_path: Path) -> Dict[str, Any]:
        return {
            "structure": await self._get_structure(repo_path),
            "languages": await self._detect_languages(repo_path),
            "frameworks": await self._detect_frameworks(repo_path),
            "entry_points": await self._find_entry_points(repo_path),
            "test_framework": await self._detect_test_framework(repo_path),
            "git_status": await git_status(str(repo_path)),
            "recent_commits": await self._get_recent_commits(repo_path, limit=10),
        }
```

### 6. Feedback Loop & Plan Revision

**Purpose**: Allow planner to revise plans based on execution outcomes.

**Trigger Conditions**:
- Agent failure (compilation error, test failure)
- Merge conflict
- Unexpected dependency discovered
- Resource constraints (too many parallel agents)

**Revision Flow**:
```
Step Failure → Capture error details → Send to planner
    ↓
Planner analyzes failure context
    ↓
Proposes revised plan (may add steps, change dependencies, adjust agents)
    ↓
User approval (optional) → Re-execute
```

---

## Enhanced Types & Data Structures

### Plan Object (Extended)

```python
@dataclass
class ParallelGroup:
    """Group of steps that can execute in parallel."""
    steps: List[StepSpec]
    merge_strategy: str = "sequential"  # sequential, octopus, manual

@dataclass
class BranchConfig:
    """Branch strategy for a step."""
    base_branch: str = "main"
    step_branch: str = ""
    auto_merge: bool = False
    
@dataclass
class EnhancedStepSpec(StepSpec):
    """Extended step with parallelization metadata."""
    branch_config: Optional[BranchConfig] = None
    estimated_duration: Optional[int] = None  # seconds
    
@dataclass
class EnhancedPlan(Plan):
    """Plan with parallel execution groups."""
    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    codebase_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: str = ""  # Planner's reasoning (from DeepSeek-R1)
```

---

## Workflow Examples

### Example 1: Simple Task (No Planner Needed)

**User**: "Add a TODO comment to the main function"

**Atlas Decision**: Simple task, no planning needed
```python
# Direct execution
atlas.tools.run("write_file", path="main.py", ...)
```

### Example 2: Complex Task (With Planner Collaboration)

**User**: "Add OAuth authentication to the app with Google and GitHub providers"

**Atlas Decision**: Complex, delegate to planner

**Planning Session**:
```
[Round 1]
Atlas → Planner:
  Objective: "Add OAuth authentication with Google and GitHub providers"
  Context: {
    "structure": "Flask backend, React frontend",
    "auth_existing": false,
    "database": "PostgreSQL"
  }

Planner → Atlas:
  Questions:
  - Should we use an existing OAuth library?
  - Do you want session-based or JWT-based auth?

[Round 2]
Atlas → Planner:
  - Use Flask-Dance for OAuth
  - JWT-based auth preferred

Planner → Atlas:
  Proposed Plan:
  Step 1: Backend - Install Flask-Dance, create OAuth routes [Agent: codex]
  Step 2: Backend - Add JWT generation/validation [Agent: codex]
  Step 3: Frontend - Create OAuth login buttons [Agent: codex]
  Step 4: Frontend - Add JWT storage and auth context [Agent: claude]
  Step 5: Integration tests [Agent: droid]
  
  Parallel Groups:
    - Group 1 (parallel): Step 1, Step 2 (both backend, different files)
    - Group 2 (parallel): Step 3, Step 4 (both frontend, different components)
    - Group 3 (sequential): Step 5 (depends on Groups 1 & 2)
  
  Branch Strategy:
    - atlas/task-abc/backend-oauth (Steps 1, 2)
    - atlas/task-abc/frontend-oauth (Steps 3, 4)
    - atlas/task-abc/integration-tests (Step 5)

[Round 3]
Atlas → Planner: Approved

Planner: Plan finalized ✓
```

**Execution**:
```
ParallelOrchestrator:
  Group 1 (parallel):
    ├─ Agent codex-1 (branch: atlas/task-abc/backend-oauth-routes)
    └─ Agent codex-2 (branch: atlas/task-abc/backend-oauth-jwt)
  
  Group 2 (parallel):
    ├─ Agent codex-3 (branch: atlas/task-abc/frontend-oauth-ui)
    └─ Agent claude-1 (branch: atlas/task-abc/frontend-oauth-context)
  
  Group 3 (sequential):
    └─ Agent droid-1 (branch: atlas/task-abc/integration-tests)
  
  Merge Strategy:
    1. Merge backend branches → main
    2. Merge frontend branches → main
    3. Merge test branch → main
    4. Run CI/CD
```

### Example 3: Plan Revision After Failure

**Scenario**: Step 2 (JWT validation) fails due to missing dependency

**Feedback Loop**:
```
Step 2 Failed:
  Error: "ModuleNotFoundError: No module named 'PyJWT'"
  
Atlas → Planner:
  "Step 2 failed. Agent tried to use PyJWT but it's not installed."
  
Planner → Atlas:
  Revised Plan:
    Insert Step 1.5: Install PyJWT via pip/poetry
    Update Step 2 dependencies
  
Atlas: Execute revised plan
```

---

## Implementation Phases

### Phase 1: Planner Agent Foundation (Week 1)
- [ ] Add `ReasoningAgentAdapter` for DeepSeek-R1 via Ollama
- [ ] Implement `PlanningSession` with round limits
- [ ] Create `CodebaseContext` analyzer
- [ ] Update `Plan` types with parallelization metadata
- [ ] Add planner agents to `config/agents.yaml`

**Deliverable**: Atlas can collaborate with a planner to create plans

### Phase 2: Parallel Orchestration (Week 2)
- [ ] Implement `ParallelOrchestrator` with concurrent step execution
- [ ] Add `BranchStrategy` for branch creation/management
- [ ] Update `TaskSpec` to support parallel groups
- [ ] Add progress monitoring for parallel agents
- [ ] Handle agent failures gracefully

**Deliverable**: Orchestrator can run multiple agents in parallel on separate branches

### Phase 3: Hybrid Planning Mode (Week 3)
- [ ] Add complexity heuristic to Atlas for task triage
- [ ] Implement automatic planner invocation for complex tasks
- [ ] Add user preference settings (always-plan, auto-plan, manual-plan)
- [ ] Create fallback logic (if planner fails, use SimplePlanner)

**Deliverable**: Atlas intelligently decides when to use planner

### Phase 4: Feedback & Iteration (Week 4)
- [ ] Implement failure capture and context extraction
- [ ] Add plan revision protocol (planner ↔ orchestrator)
- [ ] Create merge conflict resolution strategies
- [ ] Add execution analytics (success rate, bottlenecks)

**Deliverable**: System can learn from failures and revise plans

### Phase 5: Advanced Features (Future)
- [ ] Multi-planner collaboration (e.g., Sonnet + DeepSeek debate)
- [ ] Agent specialization learning (track which agents excel at what)
- [ ] Dependency graph auto-detection
- [ ] Cost optimization (prefer cheaper models for simple steps)
- [ ] Human-in-the-loop approval gates

---

## Open Questions & Discussion Points

### 1. Coordinator Agent
**Question**: Should we add a dedicated coordinator agent to review and integrate parallel work?

**Pros**:
- Ensures consistency across parallel branches
- Can catch integration issues before merge
- Central point for conflict resolution

**Cons**:
- Adds complexity and latency
- Modern agents are good at following specs
- Git merge handles most conflicts automatically

**Recommendation**: Start without coordinator, add if we see quality issues

### 2. Round Limits
**Question**: What's the optimal max_rounds for planning sessions?

**Options**:
- Conservative: 5 rounds (faster, risk of incomplete plans)
- Moderate: 8 rounds (balanced)
- Generous: 15 rounds (thorough, risk of over-planning)

**Recommendation**: Default to 8, make configurable per task complexity

### 3. Merge Strategy
**Question**: How should we merge parallel branches?

**Options**:
- Sequential: Merge branches one-by-one (safest, slower)
- Octopus: Git octopus merge (fast, complex conflicts)
- Manual: Present to user for manual merge
- Smart: Auto-merge if no conflicts, else manual

**Recommendation**: Start with Smart strategy

### 4. Planner Model Selection
**Question**: Which reasoning model(s) should we prioritize?

**Options**:
- DeepSeek-R1 (local, fast, no tools)
- Sonnet 4.5 (API, tools, expensive)
- GPT-5 (API, advanced reasoning, most expensive)
- Hybrid: Try DeepSeek first, escalate to API if needed

**Recommendation**: Implement hybrid, let user configure preference

### 5. Branch Naming
**Question**: Should branches follow a specific convention?

**Current Proposal**: `atlas/{task-id}/{step-description}`

**Alternative**: `atlas/{date}/{objective-slug}/{agent-id}`

**Recommendation**: Keep proposal, add metadata in branch description

### 6. Agent Reuse vs Fresh Spawn
**Question**: Should we reuse agent processes or spawn fresh for each step?

**Pros of Reuse**:
- Faster (no startup overhead)
- Agent maintains context

**Cons of Reuse**:
- State leakage between steps
- Resource hogging

**Recommendation**: Fresh spawn per step, optimize startup time separately

---

## Success Metrics

1. **Planning Quality**:
   - % of plans executed without revision: >80%
   - Average planning rounds used: <5

2. **Parallel Efficiency**:
   - Speedup vs sequential execution: 2-3x for parallelizable tasks
   - Branch merge success rate: >90%

3. **User Experience**:
   - Atlas correctly triages simple vs complex: >85%
   - User intervention rate: <20%

4. **Reliability**:
   - Task completion rate: >90%
   - Plan revision convergence: <3 iterations

---

## Security & Safety Considerations

1. **Credential Isolation**: Agents must use env vars, never hardcode secrets
2. **Branch Cleanup**: Auto-delete failed attempt branches
3. **Resource Limits**: Cap max parallel agents (default: 4-6)
4. **Timeout Enforcement**: Per-agent and per-task timeouts
5. **Code Review Gates**: Optional human approval before merge
6. **Audit Trail**: Log all planner decisions and agent actions

---

## Migration Path

**Backward Compatibility**:
- Keep existing `delegate_task` and `agent_session` tools
- Add new `plan_and_execute` tool for complex tasks
- SimplePlanner remains as fallback
- Users opt-in to planner via config or explicit tool call

**Rollout Plan**:
1. Deploy Phase 1 (planner foundation) to dev
2. Test with internal tasks (Atlas self-improvement)
3. Beta release Phase 2 (parallel orchestration)
4. Gradual rollout of hybrid mode (Phase 3)
5. Full production with feedback loops (Phase 4)

---

## Next Steps

1. **Review this plan** with user, get feedback on approach
2. **Finalize open questions** (coordinator, merge strategy, model selection)
3. **Set up dev environment** for planner integration (install DeepSeek-R1 via Ollama)
4. **Create branch** `feature/sub-agent-architecture` for implementation
5. **Begin Phase 1 implementation** with planner agent adapter

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-26  
**Authors**: Atlas Team  
**Status**: Awaiting Review & Approval
