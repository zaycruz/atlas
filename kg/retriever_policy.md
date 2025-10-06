# Atlas Retriever Policy

When a query references a project, topic, artifact, timeframe, or when planning tasks, call `atlas_kg.find` first with ontology filters `{ type, projectId, topicName, layer, timeStart, timeEnd }`. Use at most 10 results. If semantic narrowing is needed, re-rank those candidates with embeddings keyed by id. Cite node IDs in outputs.
