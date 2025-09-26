# Atlas Terminal Chat

Atlas is a terminal-first Jarvis-style technical copilot for local Ollama models. It layers
three kinds of memory—working, episodic, and semantic—plus lightweight web search to stay personal and
actionable across sessions while keeping prompts compact.

## Quick start

1. Install [Python 3.9+](https://www.python.org/downloads/) if you do not already have it.
2. Clone the repository (replace `YOUR_USERNAME` with your GitHub handle):
   `git clone https://github.com/YOUR_USERNAME/atlas.git && cd atlas`
3. Run the convenience launcher (it installs everything the first time):

   - **macOS / Linux**
     ```bash
     ./atlas.sh
     ```
   - **Windows (PowerShell)**
     ```powershell
   .\atlas.ps1
   ```
    If PowerShell blocks the script, open PowerShell as Administrator once and run:
    `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

   The script will install Ollama (if possible), create a local virtual
   environment, install Python dependencies, pull the default models
   (`qwen2.5:latest` plus the `mxbai-embed-large` embedder), and then
   launch the chat.

4. (Optional) Install a global `atlas` command:

   - **macOS / Linux**
     ```bash
     ./atlas.sh --install-command
     ```
     Then restart your terminal or ensure `~/.local/bin` is on your PATH. After
     that you can simply run `atlas` from anywhere.

   - **Windows (PowerShell)**
     ```powershell
     .\atlas.ps1 -InstallCommand
     ```
     Ensure the folder reported by the script (typically
     `%LOCALAPPDATA%\Microsoft\WindowsApps`) is on your PATH, then open a new
     terminal and run `atlas`.

Type into the REPL to chat. Press `Ctrl+D` (macOS/Linux) or `Ctrl+Z` then `Enter` (Windows) to exit, or type `/quit`.
Responses stream token-by-token so you can watch ideas form in real time.
Atlas can also request internal tools; you'll be prompted for approval when that happens (only for potentially destructive tools).
The CLI renders responses as Markdown with color-coded prompts (green for you,
cyan for Atlas). Use `/model <name>` to switch Ollama models on the fly.

### Ollama requirements

Atlas expects a local [Ollama](https://ollama.com/) daemon running on the
default port (`http://localhost:11434`). The launcher scripts automatically
pull `qwen2.5:latest` for chatting and `mxbai-embed-large` for embeddings.
To use different models, set `ATLAS_CHAT_MODEL` and `ATLAS_EMBED_MODEL`
before running the launcher.

## Memory & autonomy architecture

- **Working memory**: the last few turns kept in a sliding buffer so short-term
  context stays coherent.
- **Episodic memory**: every interaction is embedded with an Ollama embedding
  model (`mxbai-embed-large` by default) and stored in a JSON vector store to be
  recalled when future prompts are similar.
- **Semantic memory**: after each turn the assistant asks the LLM to extract
  durable profile facts, preferences, and goals which are persisted separately
  and summarised at the start of each response.
- **Reflective journal**: the agent can decide to write a short reflection to a SQLite journal at `~/.local/share/atlas/journal.sqlite`. Use `/journal recent` or `/journal search` in the CLI to review entries.
- **Web search tool**: invoke via a tool request `<<tool_request:web_search|{"query":"your topic"}>>` (the model may call it automatically to verify external facts) returning concise DuckDuckGo results.
- **Tool registry**: Atlas can request helper actions (e.g. creating journal entries or reviewing recent turns). Use `/tool list` and `/tool run` to manage these manually.
- **Metacognition tools**: Built-in introspective tools help Atlas observe and improve its own process:
  - `state_snapshot` — write a JSONL snapshot of internal state to `~/.local/share/atlas/state_snapshots.jsonl` (model, recent messages, memory stats, tool usage)
  - `internal_goal_set` / `internal_goal_list` — manage private, agent-only goals in `~/.local/share/atlas/internal_goals.json`
  - `relationship_log` — append collaboration events to `~/.local/share/atlas/relationship_log.jsonl`
  - Lightweight tool usage tracking is stored at `~/.local/share/atlas/tool_usage.json`
- **Periodic snapshots**: The controller captures a `state_snapshot` automatically every N turns (default 12). You can configure this in `src/atlas_main/config/policies.yaml` under `state_snapshot.turns_since_last`.

All memory and internal data is written to `~/.local/share/atlas/` by default. You can change the
paths or models with environment variables: `ATLAS_MEMORY_PATH`,
`ATLAS_SEMANTIC_PATH`, `ATLAS_JOURNAL_PATH`, `ATLAS_CHAT_MODEL`, and
`ATLAS_EMBED_MODEL`.

## CLI commands

- `/tool list` — show all tools (auto vs confirm)
- `/tool run <name> [payload]` — execute a tool
- `/memory status` — show paths and episodic stats
- `/memory recent [N]` — recent long‑term memories
- `/journal recent` — last 5 reflections
- `/journal search <keyword>` — search reflections
- `/snapshot [N]` — write a `state_snapshot` (recent messages N)
- `/igoal set <title> [--notes text] [--status active|paused|done]` — add/update a private goal
- `/igoal list [N] [status]` — list private goals
- `/relationship log <event> [--tags a,b] [--sentiment -1..1]` — log a collaboration event
- `/model <name>` / `/model list` — switch or list models
- `/thinking <on|off>` — show/hide model “thinking” content
- `/loopsteps <N>` — set internal reasoning loop steps

## Development notes

- Requires Python 3.9 or newer.
- Install dependencies with `poetry install` and run tests with
  `poetry run pytest` (or `poetry run python -m unittest discover -s tests`).
- The simple episodic store persists JSON; delete the files under the Atlas
  data directory to reset.
