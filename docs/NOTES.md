# Atlas Notes

- `LayeredMemoryManager` relies on embeddings via `OllamaClient.embed`; ensure the configured `ATLAS_EMBED_MODEL` exists locally, otherwise the episodic recall will operate without vector scoring and fall back to recency.
- Enable verbose per-layer logging by setting `ATLAS_MEMORY_DEBUG=1`; messages surface through the standard logger, so pair it with `/log debug` in the CLI when you want to watch the flow.
- Semantic and reflection layers ship empty by default; consider seeding `semantic.json` and `reflections.json` with starter content during setup so early interactions benefit from background context.
- Tests currently run through the project virtualenv (`./.venv/bin/pytest`). CI should mirror this path or expose `pytest` globally to avoid command-not-found failures when running test subsets.
- The web search summarizer trims narrative output to 600 characters; monitor whether longer answers are required for certain research tasks and adjust the limit if it becomes restrictive.
- New `shell_command` tool executes commands via `/bin/bash -lc`; it honors an optional `cwd`, enforces a timeout, and truncates very long output to keep responses controllable.
- Multiple tool calls in a single turn are dispatched concurrently through a short-lived thread pool (max 4 workers); be mindful of thread-safety if future tools share mutable state.
- The `/memory` CLI command can display episodic, semantic, reflection content, or storage paths; handy for verifying the SQLite and JSON stores at `ATLAS_MEMORY_DIR` are being updated.
- Tool enhancements worth remembering:
  - `read_file` accepts `max_lines`, `pattern`, and `case_sensitive` to slice output and highlight matches with `<< >>` markers.
  - `write_file` supports `atomic`, `preserve_times`, and `show_diff`—great for safe edits when you want a unified diff.
  - `list_dir` can now walk recursively with `recursive`, `depth`, and `human` flags.
  - `web_search` takes `domain`, `titles_only`, and `include_meta` for scoped, lightweight summaries.
  - `shell_command` adds `sudo`, `interactive`, and `retries`; interactive sessions are captured via a pseudo-tty.
- Semantic memory and reflections are now auto-updated every turn via Phi: if the exchange yields durable facts or assistant lessons, they're appended to `semantic.json` / `reflections.json`. Expect the first few runs to seed these files even if they started empty.
