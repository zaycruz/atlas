# Atlas Project Tracker

_Last updated: 2025-10-02_

This log captures the current state of Atlas memory work so we can resume quickly after interruptions. Update it whenever a task changes state.

## Completed Work

| Date | Task | Notes |
| --- | --- | --- |
| 2025-10-02 | Auto-harvest semantic facts and reflections | Wired `LayeredMemoryManager.process_turn` to persist and dedupe long-term facts/lessons with per-turn LLM harvesting. |
| 2025-10-02 | Semantic/reflection dedupe guards | Added duplicate filtering, persistence, and optional embeddings to semantic facts; ensured reflections skip blank/duplicate entries. |
| 2025-10-02 | Split summary vs. harvesting models | Layered config now captures `summary_model` and `memory_model`, letting us run heavier local models for fact extraction while keeping summaries light. |
| 2025-10-02 | Confidence-aware harvesting filter | Harvest pipeline normalizes model output, enforces minimum confidence thresholds, and tracks accept/drop counts for diagnostics. |
| 2025-10-02 | Automated memory pruning loop | Added deterministic auto-prune, `/memory prune` CLI command, and optional LLM review hook for manual clean-up runs. |
| 2025-10-02 | Harvest/prune instrumentation | Harvest/prune metrics feed `/memory stats`, giving visibility into accepted items, rejections, and pruning activity. |

## Active Tasks

| Task | Status | Details |
| --- | --- | --- |
| Tune prune review prompt | Planned | Iterate on the LLM prompt used to rescue semantic/reflection entries so we only keep genuinely valuable items. |
| Seed baseline semantic facts | Planned | Populate `semantic.json` with stable background knowledge (projects, tools) to reduce early-turn cold starts. |
| Evaluate larger harvesting models | Planned | Benchmark `llama3:8b`, `qwen2:7b`, and `mistral-nemo` for extraction fidelity and latency on the 48 GB RAM laptop. |

Update statuses to `In Progress` or `Done` with dates as work proceeds.

## Roadmap Snapshot

- **Near Term (1-2 sprints):** Complete the new active tasks above, ship a default semantic seed set, and trial the review prompt with a heavier harvesting model.
- **Mid Term:** Add CLI tooling for manual fact curation, experiment with clustered embeddings for semantic search, and evaluate nightly pruning automation.
- **Long Term:** Explore peer-agent background analysis, migrate summaries to disk for history, and benchmark alternative local models for harvesting fidelity.

## Update Checklist

1. When starting a task, change its status to `In Progress` and note the date in the Details column.
2. Once complete, move the row to **Completed Work** with a short description and completion date.
3. Review the roadmap after major milestones to ensure it reflects current priorities.
