# Atlas Terminal Chat (Ultra‑Lite)

Atlas Ultra‑Lite is a terminal-first companion for local Ollama models focused on a simple, reliable chat loop with:
- Working memory (recent turns)
- Tool registry with iterative calls (web_search via Crawl4AI)

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

### Abstractive episodic summarization

For a short, high-signal summary of recalled episodes, use the helper in `atlas_main/memory.py`:

- Function: `summarize_memories_abstractive(records, client, model=None, max_items=10, style="bullets"|"paragraph")`
- Default model: `phi3:latest` (override with `ATLAS_SUMMARY_MODEL`).
- Typical flow:
   1. `records = episodic.recall("query", top_k=8)`
   2. `summary = summarize_memories_abstractive(records, client)`

## CLI commands

- `/model <name>` / `/model list` — switch or list models
- `/thinking <on|off>` — show/hide model “thinking” content
- `/log <off|error|warn|info|debug>` — adjust logging
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
- Install with `poetry install` and run with `poetry run atlas-chat`.

Legacy design notes that referenced controller/critic, journaling, and broader tool suites remain under `docs/` for reference.
