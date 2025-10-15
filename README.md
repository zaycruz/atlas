# Atlas Terminal Chat

Atlas is a terminal-first companion for local Ollama models, pairing a streaming chat loop with layered memory and a growing tool suite. Key features include:
- Layered long-term memory (episodic SQLite, semantic facts, reflections) with automatic harvesting and pruning
- Working memory buffer for recent turns
- Tool registry with iterative calls (file read/write, shell, web search via Crawl4AI)

## Quick start

1. Install Python 3.9+ and ensure a local Ollama daemon is running at `http://localhost:11434`.
2. Install dependencies:
   - With Poetry: `poetry install`
   - Or with pip (editable): `pip install -e .`
3. Run the chat:
   - With Poetry: `poetry run atlas-chat`
   - Or with Python: `python -m atlas_main.cli`

Type into the REPL to chat. Press `Ctrl+D` to exit, or type `/quit`.

### Ollama requirements

Atlas expects a local Ollama daemon on `http://localhost:11434`. To use different models, set environment variables before launching:
- `ATLAS_CHAT_MODEL` (chat model, default `qwen3:latest`)

## Memory model

- Working memory: last few turns are kept in a sliding buffer.
- Long-term memory: episodic log stored in SQLite with semantic/reflection layers backed by JSON files.
- Automatic harvesting and pruning: after every turn the agent extracts durable facts/lessons, applies confidence thresholds, and keeps the stores tidy.

### Abstractive summarization helpers

- Episodic summary helper: `summarize_memories_abstractive(records, client, model=None, ...)`
- Layered snapshot summarizer: `summarize_assembled_context_abstractive(assembled, client, model=None, ...)`

Try it:

```bash
python scripts/visualize_summarizer.py
```

## CLI commands

- `/model <name>` / `/model list` — switch or list models
- `/thinking <on|off>` — show/hide model “thinking” content
- `/log <off|error|warn|info|debug>` — adjust logging
- `/memory stats` — inspect harvest/prune counters for the layered memory stack
- `/memory prune <semantic|reflections|all> [limit] [--review]` — trim long-term stores, optionally consulting the active model before deletions
- `/stats` — display runtime metrics (latency, tool durations, compactions)
- `/quit` — exit the chat

### Tooling

Atlas can request tools while reasoning. The available set is announced in the system prompt and currently includes:

- `web_search`: Uses DuckDuckGo for search results and Crawl4AI for clean content extraction from web pages.
- `memory.search_episodes`: Retrieve prior conversation turns relevant to a query.
- `memory.search_facts`: Lookup semantic facts with knowledge-graph context.
- `memory.explore_graph`: Inspect neighbouring facts in the knowledge graph.

The model can invoke these with directives like `<<tool:web_search|{"query": "topic"}>>` or `<<tool:memory.search_facts|{"query": "tailscale"}>>`. Tool outputs are summarized before being re-ingested so the conversation stays within the model’s context budget.

No additional setup required — Crawl4AI ships as a dependency.

## Security & Permissions

- Tools now declare capabilities (filesystem, process execution, network).
- The policy engine consults `ATLAS_TOOL_POLICY` (`allow`, `deny`, `ask`).
- Decisions are persisted in `~/.atlas/policy.json` and an audit log is written to `ATLAS_AUDIT_LOG` (default `~/.atlas/audit.jsonl`).
- Redaction is enabled by default (`ATLAS_REDACT=1`) to scrub obvious secrets from logs.

## Metrics & Observability

- `/stats` prints recent latency percentiles, snapshot savings, and per-tool success/error counts.
- Metrics can be toggled with `ATLAS_METRICS` (on by default).
- Memory packing is budget-aware; adjust `ATLAS_CONTEXT_WINDOW`, `ATLAS_CONTEXT_SAFETY`, and compaction thresholds (`ATLAS_COMPACT_THRESHOLD`, `ATLAS_COMPACT_TARGET`).
- Large tool outputs are summarized automatically; full payloads are persisted under `ATLAS_TOOL_OUTPUT_DIR` (default `~/.atlas/tool_outputs`).

## Background Watchers (optional)

- File watcher: set `ATLAS_WATCH_DIRS="~/Documents,~/Projects"` to log edits to episodic memory. Configure extensions via `ATLAS_WATCH_EXT` and poll interval via `ATLAS_WATCH_INTERVAL`.
- Clipboard watcher: enable with `ATLAS_CLIPBOARD=1` (requires `pyperclip`). Minimum snippet length and interval can be tuned with `ATLAS_CLIPBOARD_MIN` and `ATLAS_CLIPBOARD_INTERVAL`.
- Watchers run only when layered memory is active and can be disabled by unsetting the variables.

## Programmatic access

- `poetry run atlas-rpc` (or `python -m atlas_main.rpc`) launches a JSON-RPC loop on stdin/stdout.
- Supported methods: `chat`, `tools.list`, `tools.run`, `memory.snapshot`, `stats.get`, `shutdown`.
- Responses follow a simple `{"id": ..., "result": ..., "error": ...}` schema for easy scripting.

## Development notes

- Requires Python 3.9 or newer.
- Install and manage dependencies with [Poetry](https://python-poetry.org/): `poetry install`, then `poetry run atlas-chat`.
- Copy `.env.example` to `.env` and adjust the model names to match your local Ollama catalogue.
- Project context, task status, and roadmap updates live in `docs/PROJECT_TRACKER.md`; keep it current so contributors can resume work quickly.

Atlas is released under the [MIT License](LICENSE).

See [CONTRIBUTING](CONTRIBUTING.md) for guidelines on development workflow.

Legacy design notes that referenced controller/critic, journaling, and broader tool suites remain under `docs/` for reference.
