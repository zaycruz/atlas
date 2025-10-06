export type EntityType =
  | 'Project' | 'Topic' | 'File' | 'Memory' | 'Event' | 'Task' | 'Agent' | 'Metric' | 'TimeSpan';

export type Predicate =
  | 'CONTAINS' | 'IN_PROJECT' | 'ABOUT' | 'DERIVED_FROM' | 'HAPPENED_AT'
  | 'REFERENCES' | 'OWNED_BY' | 'MEASURES' | 'USES';

export type MemoryLayer = 'episodic' | 'semantic' | 'reflection';

export interface UpsertEntity { type: EntityType; props: Record<string, any>; }
export interface Link { srcId: string; predicate: Predicate; dstId: string; }
