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
- `/quit` — exit the chat

### Tooling

Atlas can request tools while reasoning. The available set is announced in the system prompt and currently includes:

- `web_search`: Uses DuckDuckGo for search results and Crawl4AI for clean content extraction from web pages.

The model triggers a tool with a directive like `<<tool:web_search|{"query": "topic"}>>`. Tool outputs are fed back into the conversation so the model can continue looping until it reaches an answer.

#### Content Extraction with Crawl4AI

Atlas uses [Crawl4AI](https://github.com/unclecode/crawl4ai) for intelligent web content extraction:

- **Clean Content**: Extracts readable text from web pages, removing ads and navigation
- **LLM-Optimized**: Designed specifically for AI applications
- **Fast & Reliable**: Handles modern web pages with JavaScript
- **Automatic Fallback**: Falls back to simple HTTP requests if Crawl4AI fails

No additional setup required — Crawl4AI ships as a dependency.

## Development notes

- Requires Python 3.9 or newer.
- Install and manage dependencies with [Poetry](https://python-poetry.org/): `poetry install`, then `poetry run atlas-chat`.
- Copy `.env.example` to `.env` and adjust the model names to match your local Ollama catalogue.
- Project context, task status, and roadmap updates live in `docs/PROJECT_TRACKER.md`; keep it current so contributors can resume work quickly.

Atlas is released under the [MIT License](LICENSE).

See [CONTRIBUTING](CONTRIBUTING.md) for guidelines on development workflow.

Legacy design notes that referenced controller/critic, journaling, and broader tool suites remain under `docs/` for reference.
