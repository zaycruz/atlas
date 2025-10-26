# Product Requirements Document: Multi-Agent Architecture

**Product**: Atlas Sub-Agent System  
**Version**: 1.0  
**Date**: 2025-10-26  
**Status**: Draft - Awaiting Approval

---

## Problem Statement

Atlas currently handles tasks sequentially with limited collaboration capabilities. For complex multi-step projects (e.g., "Add OAuth authentication to the app"), Atlas either:
1. Attempts the entire task alone (risky for complex work)
2. Delegates to a single external agent (slow, no parallelization)
3. Uses a hardcoded 3-step plan that doesn't adapt to context

**Pain Points**:
- No intelligent decomposition of complex tasks
- No parallel execution of independent work
- Limited collaboration between Atlas and specialist agents
- Risk of conflicts when multiple changes are needed
- No feedback-based learning from execution failures

---

## Vision

Transform Atlas into an **intelligent orchestrator** that:
1. Collaborates with a reasoning-focused **planner agent** to decompose complex objectives
2. Spawns multiple **coding agents** that work in parallel on isolated branches
3. Learns from execution feedback to revise and improve plans
4. Handles simple tasks directly while delegating complex ones intelligently

**Key Metaphor**: Atlas becomes a "tech lead" that works with a "solutions architect" (planner) to coordinate a team of "engineers" (coding agents) working on different features simultaneously.

---

## Goals & Success Criteria

### Primary Goals
1. **Intelligent Planning**: 80% of plans execute without revision
2. **Speed**: 2-3x faster execution for parallelizable tasks
3. **Reliability**: 90%+ task completion rate
4. **Autonomy**: 85%+ correct simple/complex task triage

### Secondary Goals
1. Iterative improvement through feedback loops
2. Branch-based isolation prevents conflicts
3. User can audit and understand agent decisions
4. System scales with task complexity

---

## User Stories

### Story 1: Complex Feature Development
**As a** developer  
**I want** to ask Atlas to "add OAuth authentication with Google and GitHub"  
**So that** Atlas orchestrates multiple agents working in parallel on frontend, backend, and tests without me managing the details.

**Acceptance Criteria**:
- Atlas collaborates with planner to create a decomposed plan
- Multiple agents work simultaneously on separate branches
- Branches merge cleanly with tests passing
- Total time < 30 minutes (vs 60+ minutes sequentially)

### Story 2: Simple Task Fast-Path
**As a** developer  
**I want** Atlas to handle simple tasks (e.g., "add a TODO comment") immediately  
**So that** I don't wait for unnecessary planning overhead.

**Acceptance Criteria**:
- Atlas detects task simplicity (< 3 steps, single file)
- Executes directly without invoking planner
- Response time < 30 seconds

### Story 3: Plan Revision on Failure
**As a** developer  
**I want** Atlas to automatically revise plans when steps fail  
**So that** I don't have to manually debug and restart the process.

**Acceptance Criteria**:
- Agent failure triggers feedback to planner
- Planner proposes revised plan with fixes
- System retries automatically (with user approval option)
- Converges within 3 revision cycles

### Story 4: Codebase-Aware Planning
**As a** developer  
**I want** the planner to understand my codebase structure  
**So that** proposed plans align with existing architecture and conventions.

**Acceptance Criteria**:
- Planner receives: tech stack, directory structure, recent commits
- Plan references actual files/modules in codebase
- Suggested agents match tech stack (e.g., use Droid for Python)

---

## Features

### Feature 1: Planner Agent Integration
**Description**: Add reasoning model (DeepSeek-R1, Sonnet 4.5, GPT-5) as a specialist planning agent.

**Capabilities**:
- Decomposes objectives into step-by-step plans
- Identifies parallelization opportunities
- Suggests appropriate coding agents per step
- Provides reasoning trace for transparency

**User Interaction**:
- Automatic invocation for complex tasks
- User can explicitly request planning: "Plan how to add OAuth"
- Round-limited conversation (default: 8 rounds)

**Configuration**:
```yaml
# User can choose planner in ~/.atlas/config.yaml
planner:
  default: planner-deepseek  # or planner-sonnet, planner-gpt5
  max_rounds: 8
  mode: auto  # auto, always, manual
```

### Feature 2: Parallel Orchestration
**Description**: Execute independent steps concurrently using multiple agents.

**Capabilities**:
- Analyzes DAG for parallel execution groups
- Spawns agents in parallel (asyncio-based)
- Monitors progress across agents in real-time
- Aggregates results and handles failures

**User Experience**:
- CLI shows parallel progress: "Agent 1 (frontend): In progress... | Agent 2 (backend): In progress..."
- Speedup metrics shown after completion: "Completed in 15 minutes (3x faster)"

**Constraints**:
- Max parallel agents: 4-6 (configurable)
- Agents must have independent file scopes
- Branch isolation enforced

### Feature 3: Branch-Based Isolation
**Description**: Each agent works on a dedicated branch to prevent conflicts.

**Capabilities**:
- Auto-creates branches: `atlas/{task-id}/{step-description}`
- Agents commit to their branches independently
- Smart merge strategy: auto-merge if no conflicts, else prompt user
- Branch cleanup after successful merge

**User Interaction**:
- User sees: "Created 3 branches for parallel work: frontend-auth, backend-auth, tests"
- After completion: "Merged 3 branches into main. Deleted temporary branches."
- On conflict: "Merge conflict in auth.py. Please resolve manually or ask Atlas to retry."

### Feature 4: Codebase Context Analysis
**Description**: Provide planner with structural understanding of the codebase.

**Capabilities**:
- Directory tree analysis
- Tech stack detection (frameworks, languages)
- Entry point identification
- Test framework detection
- Recent commit history

**User Value**:
- Plans align with existing architecture
- Suggested changes respect conventions
- Agents work on correct files

### Feature 5: Hybrid Planning Mode
**Description**: Atlas intelligently decides when to invoke planner vs. execute directly.

**Heuristics**:
- **Simple Task**: Single file, < 3 steps, no dependencies → Direct execution
- **Complex Task**: Multiple modules, > 3 steps, parallelization opportunities → Planner invoked
- **Uncertain**: User clarification or explicit mode setting

**User Control**:
```yaml
planning_mode: auto  # auto (intelligent), always (always plan), never (direct only)
```

### Feature 6: Feedback Loop & Plan Revision
**Description**: Learn from execution failures and revise plans iteratively.

**Trigger Conditions**:
- Agent fails (compilation error, test failure)
- Merge conflict
- Unexpected dependency
- Timeout

**Revision Process**:
1. Capture error details + context
2. Send feedback to planner
3. Planner proposes revised plan
4. User approval (optional, based on settings)
5. Re-execute

**User Experience**:
- "Step 2 failed: Missing dependency PyJWT. Revising plan..."
- "Revised plan: Added step to install PyJWT. Retry? (Y/n)"

---

## Non-Goals (Out of Scope for v1)

1. **Multi-Planner Collaboration**: Multiple planners debating (future enhancement)
2. **Agent Specialization Learning**: Tracking which agents excel at what (future ML feature)
3. **Cost Optimization**: Model selection based on budget (future enhancement)
4. **Visual Plan Editor**: GUI for editing plans (may add in v2)
5. **Coordinator Agent**: Central integration reviewer (discuss if quality issues arise)

---

## Technical Requirements

### Performance
- Planning phase: < 2 minutes for complex tasks
- Parallel speedup: 2-3x for independent tasks
- Round-trip latency: < 500ms per planner exchange

### Scalability
- Support 4-6 parallel agents initially
- Scale to 10+ agents in future releases
- Handle repositories up to 100k LOC

### Reliability
- Task completion rate: > 90%
- Plan revision convergence: < 3 iterations
- Branch merge success: > 90% auto-merge

### Security
- Agents never access credentials directly (env vars only)
- Branch isolation prevents state leakage
- Audit logs for all agent actions
- Optional human approval gates for production systems

### Compatibility
- Works with existing `delegate_task` and `agent_session` tools
- Backward compatible with SimplePlanner
- No breaking changes to current APIs

---

## User Experience Flow

### Flow 1: Complex Task (Full Planning)
```
User: "Add OAuth authentication with Google and GitHub"
  ↓
Atlas: "This looks complex. Collaborating with planner..."
  ↓
Atlas ↔ Planner: [Planning session, 3-5 rounds]
  ↓
Atlas: "Plan ready. Will spawn 4 agents working in parallel:
       - Agent 1: Backend OAuth routes (codex)
       - Agent 2: Backend JWT handling (codex)
       - Agent 3: Frontend login UI (claude)
       - Agent 4: Integration tests (droid)
       
       Estimated time: 15-20 minutes. Proceed? (Y/n)"
  ↓
User: "Y"
  ↓
[Parallel execution with progress updates]
  ↓
Atlas: "Completed in 17 minutes (3x faster than sequential).
       4 branches merged into main. Tests passing ✓"
```

### Flow 2: Simple Task (Fast Path)
```
User: "Add a TODO comment to the login function"
  ↓
Atlas: [Executes immediately, no planning]
  ↓
Atlas: "Added TODO comment to auth/login.py. Done in 5 seconds."
```

### Flow 3: Plan Revision
```
[During execution]
Agent 2: Failed - ModuleNotFoundError: No module named 'PyJWT'
  ↓
Atlas: "Step 2 failed. Consulting planner for revision..."
  ↓
Planner: "Revised plan: Insert step 1.5 to install PyJWT"
  ↓
Atlas: "Revised plan ready. Retry with fix? (Y/n)"
  ↓
User: "Y"
  ↓
[Re-execution succeeds]
  ↓
Atlas: "Retry successful. All steps completed ✓"
```

---

## Open Questions for User

### 1. Coordinator Agent
Should we add a dedicated "coordinator agent" to review and integrate parallel work, or trust git merge + modern agent quality?

**Vote**:
- [ ] Add coordinator (more safety, slower)
- [ ] No coordinator (trust agents + git, faster)
- [ ] Conditional coordinator (only for large tasks)

### 2. Round Limits
What's the optimal max_rounds for planning sessions?

**Options**:
- [ ] 5 rounds (fast, risk of incomplete plans)
- [ ] 8 rounds (balanced) ← Current recommendation
- [ ] 15 rounds (thorough, slower)
- [ ] Dynamic (scale with task complexity)

### 3. Merge Strategy
How should parallel branches be merged?

**Options**:
- [ ] Sequential (safest, slowest)
- [ ] Octopus merge (fast, complex conflict handling)
- [ ] Smart (auto-merge if clean, else prompt) ← Current recommendation
- [ ] Always prompt user

### 4. Planner Model Priority
Which reasoning model should be default?

**Options**:
- [ ] DeepSeek-R1 (local, fast, no tools)
- [ ] Sonnet 4.5 (API, tools, expensive)
- [ ] GPT-5 (API, most advanced, expensive)
- [ ] Hybrid (try DeepSeek, escalate if needed) ← Current recommendation

### 5. Branch Cleanup
When should we delete agent branches?

**Options**:
- [ ] Immediately after successful merge
- [ ] After 7 days (grace period)
- [ ] User configurable ← Current recommendation
- [ ] Never (let user manage)

---

## Risks & Mitigations

### Risk 1: Infinite Planning Loops
**Mitigation**: Hard limit on max_rounds (8 by default), timeout per round (30s)

### Risk 2: Merge Conflicts
**Mitigation**: Branch isolation, smart merge strategy, coordinator agent (optional)

### Risk 3: Cost (API-based planners)
**Mitigation**: Local DeepSeek-R1 as default, API models opt-in

### Risk 4: Over-Planning Simple Tasks
**Mitigation**: Heuristic-based triage, user can override with `planning_mode: never`

### Risk 5: Parallel Agent Resource Exhaustion
**Mitigation**: Cap at 4-6 agents, queue additional work, resource monitoring

---

## Success Metrics & KPIs

### Phase 1 (Planner Foundation)
- [ ] Planner responds within 30s per round
- [ ] 80% of generated plans are valid (executable)
- [ ] Average rounds used: < 5

### Phase 2 (Parallel Orchestration)
- [ ] Parallel tasks complete 2x faster than sequential
- [ ] Branch merge success rate: > 85%
- [ ] Zero resource exhaustion incidents

### Phase 3 (Hybrid Mode)
- [ ] Task triage accuracy: > 85%
- [ ] Simple tasks execute in < 30s
- [ ] User override rate: < 10%

### Phase 4 (Feedback & Iteration)
- [ ] Plan revision convergence: < 3 iterations
- [ ] Failure recovery rate: > 80%
- [ ] User satisfaction: > 90% (survey)

---

## Timeline & Milestones

| Phase | Deliverable | Duration | Target Date |
|-------|-------------|----------|-------------|
| Phase 1 | Planner agent integration | 1 week | Week of Nov 2 |
| Phase 2 | Parallel orchestration | 1 week | Week of Nov 9 |
| Phase 3 | Hybrid planning mode | 1 week | Week of Nov 16 |
| Phase 4 | Feedback & iteration | 1 week | Week of Nov 23 |
| Testing | Integration & user testing | 1 week | Week of Nov 30 |
| **Launch** | **Production release v1.0** | - | **Dec 7, 2025** |

---

## Dependencies

1. **DeepSeek-R1 via Ollama**: Install and test locally
2. **API Keys**: Anthropic (Sonnet 4.5), OpenAI (GPT-5) for API-based planners
3. **Git branching**: Ensure git version supports required operations
4. **AsyncIO**: Python 3.11+ for concurrent agent execution
5. **Testing Infrastructure**: Expand test coverage for orchestrator

---

## Appendix: Terminology

- **Atlas**: Main conversational agent, orchestrator
- **Planner Agent**: Reasoning model (DeepSeek-R1, Sonnet, GPT-5) for task decomposition
- **Coding Agent**: External coding tools (codex, claude-code, droid)
- **Sub-Agent**: Generic term for planner or coding agents
- **DAG**: Directed Acyclic Graph, represents task dependencies
- **Step**: Single unit of work in a plan
- **Branch Isolation**: Each agent works on a separate git branch
- **Parallel Group**: Set of steps that can execute concurrently
- **Round**: One exchange in a planning conversation
- **Feedback Loop**: Process of revising plans based on execution results

---

**Document Prepared By**: Atlas Development Team  
**Next Review Date**: After user feedback (target: Oct 27, 2025)  
**Approval Required From**: Project Lead, Technical Architect
