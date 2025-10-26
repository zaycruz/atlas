# Multi-Agent Orchestrator Preview

This vertical slice introduces Atlas' multi-agent orchestrator layer. It allows the CLI to delegate a coding task to an external process agent and stream step-level events back to the user.

## Components

- `src/atlas_main/orchestrator/types.py` — data classes for `TaskSpec`, `StepSpec`, `StepResult`, and task events.
- `src/atlas_main/orchestrator/engine.py` — sequential orchestrator that emits events and aggregates results.
- `src/atlas_main/agents/factory.py` — loads agent registrations from `config/agents.yaml` and instantiates adapters.
- `src/atlas_main/agents/process.py` — runs a CLI agent as a subprocess (`stdin` JSON → `stdout` JSON).
- `src/atlas_main/agents/examples/echo_agent.py` — demo agent that echoes the work it would perform.
- `src/atlas_main/config/agents.yaml` — registry file referencing the echo agent (uses `{python}` placeholder for portability).
- `src/atlas_main/cli.py` — new `atlas orchestrate` and `atlas agents list` commands with event streaming.

## Usage

```bash
PYTHONPATH=src python3 -m atlas_main.cli agents list
PYTHONPATH=src python3 -m atlas_main.cli orchestrate "Add tests for the parser"
```

The orchestrator prints step events (started, completed, skipped) and a final summary with artifacts and logs. Extend `agents.yaml` to add more adapters.

### New CLI adapters

Three real coding agents are wired in via `src/atlas_main/config/agents.yaml`:

- `codex` &mdash; GitHub Codex CLI (`codex exec --json`). Requires `codex login` and a working macOS/Linux install. The adapter runs in `--full-auto` mode, captures JSON events, and records the resulting git diff.
- `claude-code` &mdash; Anthropic Claude Code CLI (`claude --print --output-format stream-json`). Requires the Claude CLI with credentials. `HOME` is mapped to the repo to avoid writing outside the workspace and permissions default to `acceptEdits`.
- `droid` &mdash; Factory Droid CLI (`droid exec --output-format json`). Requires `FACTORY_API_KEY`. The adapter sets `--auto medium`, then snapshots git changes.

Each adapter expects a clean git working tree unless `--allow-dirty` is passed. After execution the orchestrator appends a `patch` artifact with `git diff` so downstream tooling can review/apply changes.

### Multi-step orchestration from the CLI

Example: plan with Claude, implement with Codex, then run Droid for verification.

```bash
PYTHONPATH=src python3 -m atlas_main.cli orchestrate \
  --repo /path/to/repo \
  --agent claude-code:"Draft a plan to add pagination to the API" \
  --agent codex:"Implement the agreed pagination changes." \
  --agent droid:"Run tests and summarize failures." \
  "Add pagination support to the list endpoint"
```

Flags of note:

- `--repo` selects the repository root (defaults to current directory).
- `--context-json` injects shared JSON context for all steps.
- `--allow-dirty` skips the clean-tree preflight check.

If any step fails (dirty tree, CLI error, non-zero exit), remaining steps are skipped and the failure is reported.

## Extending the slice

1. **Real coding agents**: add adapters for Codex, Claude, Droide CLIs with richer JSON contracts (diffs, tests, reviews).
2. **Parallel execution**: upgrade `Orchestrator` to schedule DAGs with asyncio tasks and handle merge conflicts.
3. **Policy gates**: integrate tool-based build/test wrappers and `policies.yaml` checks before accepting artifacts.
4. **Memory integration**: tag episodic/semantic writes with `task_id`/`agent_id` to provide shared and personal memory.
5. **UI streaming**: extend the websocket layer and desktop UI to visualise orchestrated tasks and approvals.

Each improvement can build on the existing factory, event pipeline, and CLI surface without rewriting this slice.
