# Implementation Guide: Sub-Agent Architecture

**Target**: Complete implementation today  
**Execution**: Hand this to Codex for implementation  
**Reference**: See `SUB_AGENT_ARCHITECTURE.md` and `SUB_AGENT_PRD.md` for context

---

## Quick Start for Codex

This guide provides concrete implementation tasks for the sub-agent architecture. All tasks should be completed in order, with tests added as you go.

---

## Phase 1: Planner Agent Foundation

### Task 1.1: Create Reasoning Agent Adapter

**File**: `src/atlas_main/agents/reasoning.py`

**Requirements**:
- Support DeepSeek-R1 via Ollama (local)
- Support Anthropic Claude Sonnet 4.5 (API)
- Support OpenAI GPT-5 (API)
- Implement `AgentAdapter` protocol from `agents/base.py`
- Handle streaming responses
- Parse reasoning traces from DeepSeek-R1 (thinking blocks)

**Key Methods**:
```python
class ReasoningAgentAdapter(AgentAdapter):
    def __init__(
        self,
        agent_id: str,
        model: str,
        provider: str = "ollama",  # ollama, anthropic, openai
        api_key: Optional[str] = None,
        timeout: float = 300,
        max_tokens: int = 8000,
    ):
        ...
    
    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        """Execute a planning step - analyze objective and create plan."""
        ...
    
    async def open_session(self, *, task: str, step: StepSpec, shared_context: Mapping[str, Any]) -> StreamingAgentSession:
        """Open iterative planning conversation."""
        ...
```

**Integration Points**:
- Use `OllamaClient` from `ollama.py` for DeepSeek-R1
- Use `httpx` or `requests` for Anthropic/OpenAI APIs
- Return `StepResult` with plan in `artifacts` (kind="plan")

---

### Task 1.2: Extend Types for Parallel Execution

**File**: `src/atlas_main/orchestrator/types.py`

**Add These Types**:
```python
@dataclass
class ParallelGroup:
    """Group of steps that can execute concurrently."""
    steps: List[StepSpec]
    merge_strategy: str = "smart"  # smart, sequential, octopus, manual

@dataclass
class BranchConfig:
    """Branch strategy for a step."""
    base_branch: str = "main"
    step_branch: str = ""
    auto_merge: bool = True
    
@dataclass
class EnhancedStepSpec(StepSpec):
    """Extended step with parallelization metadata."""
    branch_config: Optional[BranchConfig] = None
    estimated_duration: Optional[int] = None  # seconds
    parallel_group_id: Optional[str] = None
    
@dataclass
class EnhancedPlan(Plan):
    """Plan with parallel execution groups."""
    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    codebase_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: str = ""
    task_id: str = ""

@dataclass
class PlanningContext:
    """Context provided to planner for plan generation."""
    objective: str
    repo_path: str
    codebase_structure: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    available_agents: List[str] = field(default_factory=list)
```

**Why**: These types support parallel execution planning and branch management.

---

### Task 1.3: Codebase Context Analyzer

**File**: `src/atlas_main/orchestrator/context.py`

**Requirements**:
- Analyze repository structure (directories, files)
- Detect tech stack (languages, frameworks)
- Find entry points (main.py, app.py, index.js, etc.)
- Detect test framework (pytest, jest, unittest)
- Get recent git history (last 10 commits)
- Get current git status

**Key Class**:
```python
class CodebaseContext:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def analyze(self) -> Dict[str, Any]:
        """Generate comprehensive codebase context."""
        return {
            "structure": await self._get_structure(),
            "languages": await self._detect_languages(),
            "frameworks": await self._detect_frameworks(),
            "entry_points": await self._find_entry_points(),
            "test_framework": await self._detect_test_framework(),
            "git_status": await self._get_git_status(),
            "recent_commits": await self._get_recent_commits(limit=10),
            "dependencies": await self._get_dependencies(),
        }
    
    async def _get_structure(self) -> Dict[str, Any]:
        """Return directory tree (max depth 3)."""
        ...
    
    async def _detect_languages(self) -> List[str]:
        """Detect languages by file extensions."""
        ...
    
    async def _detect_frameworks(self) -> List[str]:
        """Detect frameworks from imports and config files."""
        ...
```

**Use Cases**:
- Planner receives this context before generating plan
- Helps planner make architecture-aware decisions

---

### Task 1.4: Planning Session Manager

**File**: `src/atlas_main/orchestrator/planning_session.py`

**Requirements**:
- Manage bounded conversation between Atlas and planner
- Hard limit on max_rounds (default: 8)
- Inject codebase context into initial prompt
- Track rounds used
- Return validated `EnhancedPlan` object
- Handle planner timeouts gracefully

**Key Class**:
```python
class PlanningSession:
    def __init__(
        self,
        planner_agent_id: str,
        agent_factory: AgentFactory,
        max_rounds: int = 8,
        round_timeout: float = 45.0,
    ):
        self.planner_agent_id = planner_agent_id
        self.agent_factory = agent_factory
        self.max_rounds = max_rounds
        self.round_timeout = round_timeout
        self.rounds_used = 0
    
    async def collaborate(
        self,
        objective: str,
        context: PlanningContext,
    ) -> EnhancedPlan:
        """Run bounded planning conversation."""
        # Round 1: Send objective + codebase context
        # Rounds 2-N: Planner asks questions, proposes plan
        # Validate plan structure
        # Return EnhancedPlan
        ...
    
    def _build_initial_prompt(self, objective: str, context: PlanningContext) -> str:
        """Construct prompt with objective and context."""
        ...
    
    def _validate_plan(self, plan_dict: Dict[str, Any]) -> EnhancedPlan:
        """Validate plan structure and convert to EnhancedPlan."""
        ...
```

**Prompt Template**:
```
You are a technical planner creating execution plans for coding agents.

OBJECTIVE:
{objective}

CODEBASE CONTEXT:
- Structure: {structure}
- Languages: {languages}
- Frameworks: {frameworks}
- Entry Points: {entry_points}
- Test Framework: {test_framework}
- Git Status: {git_status}
- Recent Commits: {recent_commits}

AVAILABLE AGENTS:
- codex: GitHub Codex CLI (general coding)
- claude-code: Claude Code CLI (complex refactors)
- droid: Factory Droid (Python specialist)

TASK:
Create a detailed execution plan with:
1. Steps (id, description, agent_id, dependencies)
2. Parallel groups (steps that can run concurrently)
3. Branch strategy (which steps can share branches)
4. Estimated duration per step

RULES:
- Steps should be granular (1 clear objective per step)
- Identify independent steps for parallel execution
- Suggest appropriate agents based on task type and codebase tech stack
- Include dependencies (depends_on: [step_ids])
- Provide reasoning for your decisions

OUTPUT FORMAT (JSON):
{
  "objective": "...",
  "steps": [...],
  "parallel_groups": [...],
  "reasoning": "..."
}
```

---

### Task 1.5: Update Agent Factory

**File**: `src/atlas_main/agents/factory.py`

**Add Support for Reasoning Agents**:
```python
# In AgentFactory.create() method, add:
if agent_type == "reasoning":
    model = str(config.get("model", "deepseek-r1"))
    provider = str(config.get("provider", "ollama"))
    timeout = float(config.get("timeout", 300))
    api_key = config.get("api_key")  # Optional, can use env var
    max_tokens = int(config.get("max_tokens", 8000))
    return ReasoningAgentAdapter(
        agent_id=agent_id,
        model=model,
        provider=provider,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
```

---

### Task 1.6: Update Agent Config

**File**: `src/atlas_main/config/agents.yaml`

**Add Planner Agents**:
```yaml
planner-deepseek:
  description: DeepSeek-R1 reasoning planner via Ollama
  type: reasoning
  provider: ollama
  model: deepseek-r1:latest
  timeout: 300
  max_tokens: 8000

planner-sonnet:
  description: Claude Sonnet 4.5 planner with tools
  type: reasoning
  provider: anthropic
  model: claude-sonnet-4.5-20250514
  timeout: 300
  max_tokens: 8000
  # api_key: ${ANTHROPIC_API_KEY}  # Read from env

planner-gpt5:
  description: OpenAI GPT-5 planner
  type: reasoning
  provider: openai
  model: gpt-5-turbo
  timeout: 300
  max_tokens: 8000
  # api_key: ${OPENAI_API_KEY}  # Read from env
```

---

## Phase 2: Parallel Orchestration

### Task 2.1: Branch Strategy Manager

**File**: `src/atlas_main/orchestrator/branches.py`

**Requirements**:
- Create branches for agent work
- Handle branch cleanup
- Implement merge strategies (smart, sequential, octopus)
- Detect merge conflicts

**Key Class**:
```python
class BranchStrategy:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def create_step_branch(
        self,
        task_id: str,
        step_id: str,
        base_branch: str = "main"
    ) -> str:
        """Create isolated branch for agent work."""
        branch_name = f"atlas/{task_id}/{step_id}"
        await self._git_checkout(branch_name, create=True, base=base_branch)
        return branch_name
    
    async def merge_branches(
        self,
        branches: List[str],
        target_branch: str = "main",
        strategy: str = "smart"
    ) -> MergeResult:
        """Merge agent branches into target."""
        if strategy == "smart":
            return await self._smart_merge(branches, target_branch)
        elif strategy == "sequential":
            return await self._sequential_merge(branches, target_branch)
        elif strategy == "octopus":
            return await self._octopus_merge(branches, target_branch)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    async def cleanup_branches(self, branches: List[str], force: bool = False):
        """Delete temporary agent branches."""
        ...
    
    async def _smart_merge(self, branches: List[str], target: str) -> MergeResult:
        """Try auto-merge, prompt on conflict."""
        ...

@dataclass
class MergeResult:
    success: bool
    conflicts: List[str] = field(default_factory=list)
    merged_branches: List[str] = field(default_factory=list)
    failed_branches: List[str] = field(default_factory=list)
    message: str = ""
```

**Git Operations**:
- Use `asyncio.create_subprocess_exec` for git commands
- Reuse patterns from `agents/common.py` (git_status, git_diff)

---

### Task 2.2: Parallel Orchestrator

**File**: `src/atlas_main/orchestrator/parallel_engine.py`

**Requirements**:
- Extend `Orchestrator` from `engine.py`
- Identify parallel execution groups from DAG
- Execute parallel steps concurrently with `asyncio.gather`
- Handle per-step branch creation
- Monitor progress across agents
- Aggregate results

**Key Class**:
```python
class ParallelOrchestrator(Orchestrator):
    """Orchestrator with parallel execution support."""
    
    def __init__(
        self,
        agent_factory: AgentFactory,
        branch_strategy: Optional[BranchStrategy] = None,
        max_parallel: int = 4,
        event_callback: Optional[EventCallback] = None
    ):
        super().__init__(agent_factory, event_callback=event_callback)
        self.branch_strategy = branch_strategy
        self.max_parallel = max_parallel
    
    async def run_task(self, task: TaskSpec) -> TaskResult:
        """Execute task with parallel step support."""
        # Analyze task.steps for parallelization
        parallel_groups = self._identify_parallel_groups(task.steps)
        
        step_results = []
        for group in parallel_groups:
            if len(group) > 1 and len(group) <= self.max_parallel:
                # Parallel execution
                results = await self._run_parallel_group(group, task)
            else:
                # Sequential execution
                results = [await self._run_single_step(step, task) for step in group]
            step_results.extend(results)
        
        # Merge branches if branch_strategy is configured
        if self.branch_strategy:
            await self._merge_step_branches(step_results)
        
        status = "succeeded" if all(r.succeeded for r in step_results) else "failed"
        return TaskResult(task_id=task.id, status=status, step_results=step_results)
    
    def _identify_parallel_groups(self, steps: List[StepSpec]) -> List[List[StepSpec]]:
        """Group steps by dependency levels for parallel execution."""
        # Build dependency graph
        # Topologically sort
        # Group steps at same level
        ...
    
    async def _run_parallel_group(
        self,
        steps: List[StepSpec],
        task: TaskSpec
    ) -> List[StepResult]:
        """Execute steps in parallel with branch isolation."""
        # Create branches for each step
        # Spawn agents in parallel
        # Use asyncio.gather with timeout
        tasks = [self._run_step_with_branch(step, task) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _run_step_with_branch(
        self,
        step: StepSpec,
        task: TaskSpec
    ) -> StepResult:
        """Run single step with branch isolation."""
        if self.branch_strategy:
            branch = await self.branch_strategy.create_step_branch(
                task.id, step.id
            )
            self._emit("step.branch_created", task, {"step_id": step.id, "branch": branch})
        
        # Run step (reuse parent class logic)
        result = await self._run_single_step(step, task)
        result.metadata["branch"] = branch
        return result
```

**Key Features**:
- Respect `max_parallel` limit (default: 4)
- Handle exceptions in parallel execution gracefully
- Emit events for branch creation, parallel start/end

---

### Task 2.3: Update Planner to Generate Parallel Groups

**File**: `src/atlas_main/orchestrator/planner.py`

**Requirements**:
- Replace `SimplePlanner` with `IntelligentPlanner`
- Use `PlanningSession` for complex tasks
- Fall back to simple heuristics for basic tasks

**Key Class**:
```python
class IntelligentPlanner:
    """Planner that uses reasoning agents for complex tasks."""
    
    def __init__(
        self,
        agent_factory: AgentFactory,
        planner_agent_id: str = "planner-deepseek",
        max_rounds: int = 8,
    ):
        self.agent_factory = agent_factory
        self.planner_agent_id = planner_agent_id
        self.max_rounds = max_rounds
    
    async def plan(
        self,
        objective: str,
        repo_path: str,
        available_agents: Optional[List[str]] = None
    ) -> EnhancedPlan:
        """Generate execution plan for objective."""
        # Analyze codebase
        context_analyzer = CodebaseContext(Path(repo_path))
        codebase_ctx = await context_analyzer.analyze()
        
        # Build planning context
        planning_ctx = PlanningContext(
            objective=objective,
            repo_path=repo_path,
            codebase_structure=codebase_ctx,
            available_agents=available_agents or ["codex", "claude-code", "droid"],
        )
        
        # Start planning session
        session = PlanningSession(
            planner_agent_id=self.planner_agent_id,
            agent_factory=self.agent_factory,
            max_rounds=self.max_rounds,
        )
        
        plan = await session.collaborate(objective, planning_ctx)
        return plan
    
    async def should_plan(self, objective: str) -> bool:
        """Heuristic: should we invoke planner or execute directly?"""
        # Simple heuristics:
        # - Objective length < 50 chars -> likely simple
        # - Contains "add", "create", "refactor", "oauth" -> likely complex
        # - Mentions multiple modules/files -> complex
        ...
```

**Fallback Logic**:
- If planner times out, fall back to `SimplePlanner`
- If planner returns invalid plan, fall back to sequential execution

---

## Phase 3: Hybrid Planning Mode

### Task 3.1: Add Planning Tool to Atlas

**File**: `src/atlas_main/tools.py`

**Add New Tool**:
```python
class PlanAndExecuteTool(Tool):
    """Plan and execute complex tasks with parallel agents."""
    
    name = "plan_and_execute"
    description = (
        "Collaborate with a planner agent to decompose a complex objective, "
        "then execute the plan using multiple coding agents in parallel. "
        "Use this for multi-step features, refactors, or complex changes."
    )
    args_hint = (
        "objective=<text> repo_path=/path/to/repo planner=planner-deepseek "
        "max_parallel=4 allow_dirty=false auto_merge=true"
    )
    capabilities = frozenset({"filesystem", "process"})
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "repo_path": {"type": "string"},
            "planner": {
                "type": "string",
                "description": "Planner agent ID (default: planner-deepseek)",
            },
            "max_parallel": {
                "type": "integer",
                "description": "Max concurrent agents (default: 4)",
            },
            "allow_dirty": {"type": "boolean"},
            "auto_merge": {
                "type": "boolean",
                "description": "Auto-merge branches on success (default: true)",
            },
        },
        "required": ["objective"],
        "additionalProperties": False,
    }
    
    def run(
        self,
        *,
        agent=None,
        objective: str,
        repo_path: Optional[str] = None,
        planner: str = "planner-deepseek",
        max_parallel: int = 4,
        allow_dirty: bool = False,
        auto_merge: bool = True,
    ) -> str:
        repo = Path(repo_path).expanduser() if repo_path else Path.cwd()
        
        # Create planner
        factory = AgentFactory()
        intelligent_planner = IntelligentPlanner(
            agent_factory=factory,
            planner_agent_id=planner,
        )
        
        # Generate plan
        plan = asyncio.run(intelligent_planner.plan(objective, str(repo)))
        
        # Convert EnhancedPlan to TaskSpec
        task_spec = self._plan_to_task_spec(plan, repo, allow_dirty)
        
        # Execute with parallel orchestrator
        branch_strategy = BranchStrategy(repo) if auto_merge else None
        orchestrator = ParallelOrchestrator(
            agent_factory=factory,
            branch_strategy=branch_strategy,
            max_parallel=max_parallel,
        )
        
        result = asyncio.run(orchestrator.run_task(task_spec))
        
        # Format response
        return self._format_result(result, plan)
```

**Register Tool** in `AtlasAgent.__init__`:
```python
self.tools.register(PlanAndExecuteTool())
```

---

### Task 3.2: Update Atlas System Prompt

**File**: `src/atlas_main/agent.py`

**Update `DEFAULT_PROMPT`**:
Add guidance for when to use `plan_and_execute`:
```
Tool guidance:
...
- For complex multi-step tasks (OAuth, new features, major refactors), use plan_and_execute to collaborate with a planner and spawn parallel agents.
- For simple tasks (add comment, fix typo, single file change), use existing tools directly.
...
```

---

## Phase 4: Feedback & Iteration

### Task 4.1: Failure Context Capture

**File**: `src/atlas_main/orchestrator/feedback.py`

**Requirements**:
- Capture failure details from StepResult
- Extract actionable context (error messages, logs, git diff)
- Format for planner consumption

**Key Class**:
```python
@dataclass
class FailureContext:
    step_id: str
    agent_id: str
    error_type: str
    error_message: str
    logs: List[str]
    git_diff: str
    suggested_fixes: List[str] = field(default_factory=list)

class FeedbackCollector:
    def capture_failure(self, step_result: StepResult) -> FailureContext:
        """Extract failure details from step result."""
        ...
    
    def format_for_planner(self, context: FailureContext) -> str:
        """Format failure context for planner revision."""
        return f"""
STEP FAILURE REPORT:
Step: {context.step_id}
Agent: {context.agent_id}
Error Type: {context.error_type}
Error Message: {context.error_message}

Logs:
{chr(10).join(context.logs[-10:])}

Git Diff:
{context.git_diff[:1000]}

Please revise the plan to address this failure.
"""
```

---

### Task 4.2: Plan Revision Protocol

**File**: `src/atlas_main/orchestrator/planning_session.py`

**Add Method to `PlanningSession`**:
```python
async def revise_plan(
    self,
    original_plan: EnhancedPlan,
    failure_contexts: List[FailureContext],
) -> EnhancedPlan:
    """Revise plan based on execution failures."""
    # Send failure context to planner
    # Request revised plan
    # Validate and return
    ...
```

---

### Task 4.3: Update Parallel Orchestrator for Revision

**File**: `src/atlas_main/orchestrator/parallel_engine.py`

**Add to `ParallelOrchestrator`**:
```python
async def run_task_with_revision(
    self,
    task: TaskSpec,
    max_revisions: int = 3,
) -> TaskResult:
    """Execute task with automatic plan revision on failure."""
    attempts = 0
    result = None
    
    while attempts < max_revisions:
        result = await self.run_task(task)
        
        if result.succeeded:
            return result
        
        # Collect failure contexts
        failures = [
            FeedbackCollector().capture_failure(sr)
            for sr in result.step_results
            if sr.status == "failed"
        ]
        
        if not failures:
            break  # No failures to revise
        
        # Revise plan
        session = PlanningSession(...)
        revised_plan = await session.revise_plan(task.metadata["plan"], failures)
        
        # Update task with revised steps
        task = self._update_task_from_plan(task, revised_plan)
        attempts += 1
    
    return result
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/orchestrator/test_planning_session.py`
- Test round limits enforced
- Test plan validation
- Test timeout handling

**File**: `tests/orchestrator/test_parallel_engine.py`
- Test parallel group identification
- Test concurrent execution
- Test branch isolation

**File**: `tests/orchestrator/test_branches.py`
- Test branch creation
- Test merge strategies
- Test conflict detection

**File**: `tests/agents/test_reasoning.py`
- Test DeepSeek-R1 adapter
- Test API-based adapters (mock responses)
- Test streaming sessions

### Integration Tests

**File**: `tests/integration/test_plan_and_execute.py`
- End-to-end test: simple objective → plan → execute
- Test parallel execution with mock agents
- Test failure + revision cycle

---

## Environment Setup

### Install DeepSeek-R1 via Ollama

```bash
# If ollama not installed
curl -fsSL https://ollama.com/install.sh | sh

# Pull DeepSeek-R1
ollama pull deepseek-r1:latest

# Verify
ollama run deepseek-r1:latest "Hello"
```

### API Keys (Optional)

```bash
# ~/.atlas/config.yaml or environment
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

---

## Execution Order

**Hand this to Codex in this order:**

1. ✅ **Phase 1 (Foundation)**: Tasks 1.1 → 1.6
   - "Implement reasoning agent adapter, update types, create context analyzer, planning session, update factory and config"

2. ✅ **Phase 2 (Parallel)**: Tasks 2.1 → 2.3
   - "Implement branch strategy, parallel orchestrator, update planner to use reasoning agents"

3. ✅ **Phase 3 (Integration)**: Tasks 3.1 → 3.2
   - "Add plan_and_execute tool to Atlas, update system prompt"

4. ✅ **Phase 4 (Feedback)**: Tasks 4.1 → 4.3
   - "Implement failure capture, plan revision, update orchestrator for auto-revision"

5. ✅ **Testing**: Add tests as you go, run full test suite at end

---

## Success Criteria

- [ ] `plan_and_execute` tool works end-to-end
- [ ] Can collaborate with DeepSeek-R1 planner
- [ ] Parallel orchestrator runs multiple agents concurrently
- [ ] Branches created/merged automatically
- [ ] Plan revision works on failure
- [ ] Tests pass
- [ ] Can execute complex task (e.g., "Add OAuth") successfully

---

## Notes for Codex

- **Reuse existing patterns** from `engine.py`, `agents/base.py`, `tools.py`
- **Follow existing code style** (type hints, docstrings, dataclasses)
- **Add tests as you implement** (don't save for later)
- **Use async/await** for all I/O operations
- **Handle errors gracefully** (timeouts, invalid plans, git failures)
- **Emit events** for observability (planning started, parallel execution, etc.)
- **Keep git operations safe** (check status before creating branches)

---

## Quick Reference

**Key Files to Modify**:
- `agents/reasoning.py` (NEW)
- `agents/factory.py` (UPDATE)
- `orchestrator/types.py` (UPDATE)
- `orchestrator/context.py` (NEW)
- `orchestrator/planning_session.py` (NEW)
- `orchestrator/branches.py` (NEW)
- `orchestrator/parallel_engine.py` (NEW)
- `orchestrator/planner.py` (UPDATE)
- `orchestrator/feedback.py` (NEW)
- `tools.py` (UPDATE - add PlanAndExecuteTool)
- `agent.py` (UPDATE - register tool, update prompt)
- `config/agents.yaml` (UPDATE)

**Dependencies to Add** (pyproject.toml):
- `httpx` (for API calls to Anthropic/OpenAI)
- Already have: `asyncio`, `yaml`, `ollama client`

**Architecture Docs**:
- `docs/SUB_AGENT_ARCHITECTURE.md` - Full technical design
- `docs/SUB_AGENT_PRD.md` - Product requirements
- `docs/IMPLEMENTATION_GUIDE.md` - This file

---

**Ready to Execute?** Copy tasks to Codex and start with Phase 1!
