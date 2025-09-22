# Atlas Terminal Chat

Atlas is a terminal-first personal assistant for local Ollama models. It layers
three kinds of memory—working, episodic, and semantic—to stay personal across
sessions while keeping prompts compact.

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

### Ollama requirements

Atlas expects a local [Ollama](https://ollama.com/) daemon running on the
default port (`http://localhost:11434`). The launcher scripts automatically
pull `qwen2.5:latest` for chatting and `mxbai-embed-large` for embeddings.
To use different models, set `ATLAS_CHAT_MODEL` and `ATLAS_EMBED_MODEL`
before running the launcher.

## Memory architecture

- **Working memory**: the last few turns kept in a sliding buffer so short-term
  context stays coherent.
- **Episodic memory**: every interaction is embedded with an Ollama embedding
  model (`mxbai-embed-large` by default) and stored in a JSON vector store to be
  recalled when future prompts are similar.
- **Semantic memory**: after each turn the assistant asks the LLM to extract
  durable profile facts, preferences, and goals which are persisted separately
  and summarised at the start of each response.

All memory is written to `~/.local/share/atlas/` by default. You can change the
paths or models with environment variables: `ATLAS_MEMORY_PATH`,
`ATLAS_SEMANTIC_PATH`, `ATLAS_CHAT_MODEL`, and `ATLAS_EMBED_MODEL`.

## Development notes

- Requires Python 3.9 or newer.
- Install dependencies with `poetry install` and run tests with
  `poetry run pytest` (or `poetry run python -m unittest discover -s tests`).
- The simple episodic store persists JSON; delete the files under the Atlas
  data directory to reset.
