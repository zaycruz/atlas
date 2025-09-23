# Atlas Migration t### 3. Policies Configuration
- [x] Create `src/atlas_main/config/policies.yaml` with triggers for reflect/disagree (e.g., turns since last reflection, goal changes).
- [x] Load policies in `agent.py` or new controller.

### 4. Reasoning Loop Controller
- [x] Create `src/atlas_main/atlas_core/controller.py` as the orchestrator for the reasoning loop (Perceive → Retrieve → Reason → Tools → Critic → Respond → Reflect).
- [x] Integrate top-K retrieval (BGE/E5) before drafting in `agent.py`.
- [ ] Route all turns through the controller (no bypass). v1 - Todo List

Based on `docs/atlas-architecture-delta.md`. This tracks the migration plan to implement the hybrid approach without adding unnecessary bloat. We'll integrate changes incrementally into the existing codebase, reusing components where possible and adding only essential new files.

## Migration Plan (One Sprint)

### 1. Vector DB Setup
- [x] Install ChromaDB (or LanceDB) via pip.
- [x] Create `src/atlas_main/memory/vector_store/` directory.
- [x] Migrate episodic memory from JSON/files to Chroma collection.
- [x] Implement live retrieval with BGE-small or E5-small embedder (CPU-based).
- [x] Update `enhanced_memory.py` to use vector DB instead of JSON.

### 2. Journal Migration to SQLite
- [x] Create `src/atlas_main/memory/journal.sqlite` schema (table: id, ts, title, body, tags, links).
- [x] Import existing `journal.json` data into SQLite.
- [x] Update `journal.py` to use SQLite instead of JSON.
- [x] Add support for tags and links in journal entries.

### 3. Policies Configuration
- [x] Create `src/atlas_main/config/policies.yaml` with triggers for reflect/disagree (e.g., turns since last reflection, goal changes).
- [ ] Load policies in `agent.py` or new controller.

### 4. Reasoning Loop Controller
- [ ] Create `src/atlas_main/atlas_core/controller.py` as the orchestrator for the reasoning loop (Perceive → Retrieve → Reason → Tools → Critic → Respond → Reflect).
- [ ] Integrate top-K retrieval (BGE/E5) before drafting in `agent.py`.
- [ ] Route all turns through the controller (no bypass).

### 5. First-Class Actions
- [x] Update system prompt in `agent.py` to allow suggest/disagree decisions.
- [x] Implement new tools: Journal (to SQLite), Goal Update (semantic_profile.json), Memory Mark (vector pin), Run Experiment.
- [ ] Allow Qwen3 to emit directives like JOURNAL(...), GOAL_UPDATE(...) during reasoning.
- [ ] Controller validates and executes state-changing tools.

### 6. Nightly Jobs
- [ ] Add T5-small summarizer for journaling and compression.
- [ ] Implement nightly T5 summaries of long threads; store as journal entries with back-links.
- [ ] Optional: MXBAI re-embed/re-rank maintenance jobs.

### 7. QA and Testing
- [ ] Measure latency: Embed (BGE/E5) ≤ 20ms, retrieval ≤ 30ms, total ≤ 120ms.
- [ ] Test autonomy: Journal without user command on triggers.
- [ ] Test disagreement tied to goals.
- [ ] Test memory retrieval for 3 seeded scenarios.
- [ ] Verify journal entries in SQLite with tags/links.
- [ ] Ensure no leakage of internal thoughts; all replies through Reasoning Loop.

## Acceptance Criteria Checklist
- [ ] Latency targets met on Mac Mini.
- [ ] Atlas journals autonomously when triggers hit.
- [ ] Disagreement is goal-tied, not random.
- [ ] Top-K retrieval surfaces correct past threads.
- [ ] Journal entries in SQLite with tags + links to episode IDs.
- [ ] No user-facing leakage of thoughts.
- [ ] All replies flow through Reasoning Loop (logs/flags).

## Plan to Avoid Bloat
- **Incremental**: Implement one section at a time, testing after each.
- **Reuse**: Leverage existing code (e.g., extend `agent.py`, `journal.py`) instead of rewriting.
- **Minimal New Files**: Only add essential files; use placeholders for docs/models.
- **No Over-Engineering**: Start with simple implementations; optimize later if needed.
- **Branching**: Work on dev branch; commit frequently with clear messages.
- **Dependencies**: Add only required packages (ChromaDB, etc.) to `pyproject.toml` or `requirements.txt`.
- **Testing**: Run existing tests after changes; add minimal new tests for new features.

## Current Status
- [x] Docs added (`docs/atlas-architecture-delta.md`).
- [x] Vector DB setup (Step 1).
- [x] Journal to SQLite (Step 2).
- [x] Policies config (Step 3).
- [x] Controller placeholders (Step 4).
- [x] First-class actions (Step 5).
- [ ] Nightly jobs (Step 6) - placeholder needed.
- [ ] QA (Step 7) - run tests, measure latency.</content>
<parameter name="filePath">/Users/master/projects/atlas-main/docs/migration-todo.md
