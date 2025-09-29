# Atlas Roadmap

## Recently Completed
- Integrated layered memory stack (episodic SQLite, semantic facts, reflections) directly into `AtlasAgent`, including automatic embedding and per-turn context snapshots.
- Reworked web search tool output to deliver narrative summaries drawn from both search snippets and crawled page content, with tests updated accordingly.
- Added concurrent tool execution so multiple model-invoked functions can run in parallel when a response supplies several tool calls at once.
- Expanded the built-in tool suite: `read_file` now supports line limits and regex highlighting, `write_file` gained atomic replace/preserve timestamps/diff preview, `list_dir` can recurse with human-readable sizes, `web_search` accepts domain filters/title-only/meta options, and `shell_command` offers sudo, interactive capture, and retries.
- Layered memory now extracts semantic facts and reflections automatically after each exchange using the local summarizer, keeping long-term knowledge fresh without manual seeding.

## In Flight
- Populate semantic facts and reflection stores so the layered memory snapshot can surface richer background knowledge.
- Broaden automated test coverage to exercise the new layered-memory prompt injection when pytest is fully available in CI.

## Upcoming
- Add user-facing commands or CLI helpers for inspecting and curating semantic and reflection layers.
- Evaluate storage migration paths for existing JSON episodic logs into the SQLite-backed structure.
- Monitor web-search result quality and adjust summarization heuristics (sentence limits, fallbacks) based on real usage feedback.
