# Model and Tool Routing Spec

## Models
- Brain (always-on): Qwen3 via Ollama
- Live embedder: SentenceTransformer 'all-MiniLM-L6-v2' (BGE-small equivalent)
- Summarizer: T5-small (for journaling/compression)
- Heavy embedder: mxbai-embed-large (offline re-rank)

## Tools
- State-changing: Journal, Goal Update, Memory Mark, Run Experiment
- Guidance: Suggest Direction, Disagree/Challenge
- Utilities: Memory Snapshot, Context Connector, etc.

## Routing Rules
- All turns route through ReasoningController
- Embeddings use live embedder for retrieval
- Tools execute via controller validation
- Critic optional based on policies

