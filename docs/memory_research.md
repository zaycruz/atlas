# Memory Layer Research & Recommendations

## 1. Current Atlas Memory Stack
- **Working memory**: maintained in-process via `WorkingMemory` for the last N conversational turns, responsible for immediate context without persistence.
- **Episodic layer**: persisted to SQLite with embeddings for vector recall and recency fallbacks, now augmented with access statistics for more intelligent prioritization.【F:src/atlas_main/memory.py†L47-L141】【F:src/atlas_main/memory_layers.py†L24-L224】
- **Semantic layer**: durable fact store backed by JSON; we now track usage counts, last-access timestamps, quality heuristics, and cosine-similarity deduplication to protect against noisy or redundant insertions.【F:src/atlas_main/memory_layers.py†L124-L319】
- **Reflection layer**: stores lessons learned; the revamp adds heuristic quality scores, priority-ranked retrieval, and smarter pruning to keep reflective insights relevant over time.【F:src/atlas_main/memory_layers.py†L321-L497】
- **Assembler & summarizer**: merges layers into a response context and can summarize via a local LLM to provide a compressed, high-signal synopsis for downstream reasoning.【F:src/atlas_main/memory_layers.py†L499-L752】

## 2. Lessons from the Literature
- **MemGPT / Letta**: highlight hierarchical memory with gating. They score candidate memories for salience and maintain multi-tier storage (context, short-term, long-term) with heuristics such as novelty, recency, and importance to decide promotion into durable stores. Reflection loops double-check stored knowledge for accuracy before recall.
- **Generative Agents (Park et al.)**: use relevance × recency × importance scoring to surface memories that influence planning—showing that simple heuristics can approximate complex mental models.
- **Self-reflection techniques (e.g., Reflexion, LLM+Reflection)**: emphasize structured post-task reviews, logging “lessons” only when they cross a usefulness threshold, and updating importance scores after each usage.
- **Vector databases and knowledge graphs**: best-in-class agent memory stacks blend embedding similarity with metadata filters (e.g., event type, trust) and leverage LLM critics to vet new knowledge before writing to disk.

Across these systems, the recurring ideas are:
1. **Quality & salience gating** before persistence (avoid memorizing fluff).
2. **Usage-aware scoring** so high-impact items become easier to recall.
3. **Redundancy checks** (semantic dedupe) to cap drift.
4. **Scheduled pruning and review**, sometimes with LLM adjudication.
5. **Explicit metadata** (confidence, provenance, timestamps) to support auditability and targeted retrieval.

## 3. Gap Analysis for Atlas
- Prior semantic and reflection layers lacked structured quality assessment, letting short, low-signal snippets persist unchecked.
- Recall order was primarily chronological or pure cosine similarity, ignoring how often an item helped the agent.
- Dedupe logic was textual only—near-duplicate facts with slightly different phrasing could accumulate.
- Pruning strategies did not weigh quality versus recency, risking accidental deletion of high-value knowledge.

## 4. Enhancements Implemented
- **Quality scoring inspired by MemGPT/Letta**: `_quality_features` measures length, structure, verb presence, and actionability; configurable thresholds now gate harvest acceptance for facts and reflections.【F:src/atlas_main/memory_layers.py†L40-L118】【F:src/atlas_main/memory_layers.py†L566-L619】
- **Metadata-enriched semantic facts**: every fact tracks confidence, quality, usage counts, and last access. Retrieval updates these fields and persists them, allowing priority ordering and smarter pruning.【F:src/atlas_main/memory_layers.py†L146-L319】
- **Semantic deduplication**: cosine-based duplicate detection prevents redundant facts, and the existing record is boosted instead of creating drift.【F:src/atlas_main/memory_layers.py†L178-L233】
- **Priority-based fallbacks**: semantic head requests and reflection recalls now surface high-quality, frequently used knowledge rather than naive list slices.【F:src/atlas_main/memory_layers.py†L254-L319】【F:src/atlas_main/memory_layers.py†L381-L450】
- **Reflection governance**: reflections share the same quality gate, track usage/recency, and prune with a combined priority+t-tail safeguard so valuable lessons remain accessible.【F:src/atlas_main/memory_layers.py†L321-L497】
- **Expanded metrics**: harvest stats now capture low-quality rejections to monitor upstream prompt tuning.【F:src/atlas_main/memory_layers.py†L486-L575】

## 5. Next Steps & Research Directions
1. **LLM critic in the loop**: adopt a lightweight reviewer (similar to Letta’s critic) to validate or revise candidate facts before they enter long-term storage.
2. **Task-linked memories**: tag facts/reflections with originating objectives or tool invocations, enabling targeted recall during similar tasks.
3. **Adaptive thresholds**: learn quality cutoffs per user or domain by monitoring retrieval success—mirroring MemGPT’s adaptive gating.
4. **Memory audits**: schedule periodic reflective summaries where the agent revalidates stored knowledge against fresh conversations.
5. **Graph structuring**: experiment with linking facts into a lightweight knowledge graph to support reasoning over dependencies and contradictions.

With these upgrades the Atlas agent retains the layered design while adopting the proven heuristics from contemporary agent-memory literature, improving learning, reflection, and long-term consistency.
