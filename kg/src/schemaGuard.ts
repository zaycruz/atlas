import { Link, Predicate, MemoryLayer } from './types.js';

const ConceptLabels = new Set(['Topic']);
const RequiresProject = new Set(['Memory', 'File', 'Task', 'Event']);

function hasLabel(node: { labels?: string[] }, label: string) {
  return Array.isArray(node.labels) && node.labels.includes(label);
}

export function validateNode(type: string, props: Record<string, any>) {
  if (!props?.id) throw new Error('Node requires id');

  if (type === 'TimeSpan') {
    if (!props.start || !props.end) throw new Error('TimeSpan requires start & end');
    if (new Date(props.start) > new Date(props.end)) throw new Error('TimeSpan start > end');
  }

  if (type === 'Memory') {
    const layer: MemoryLayer = props.layer;
    if (!['episodic', 'semantic', 'reflection'].includes(layer)) {
      throw new Error('Invalid Memory.layer');
    }
  }
}

export function validateLink(
  srcNode: { labels?: string[]; properties?: Record<string, any> },
  link: Link,
  dstNode: { labels?: string[]; properties?: Record<string, any> }
) {
  switch (link.predicate as Predicate) {
    case 'ABOUT': {
      if (![...ConceptLabels].some((label) => hasLabel(dstNode, label))) {
        throw new Error('ABOUT must target Concept (e.g., Topic)');
      }
      break;
    }
    case 'HAPPENED_AT': {
      if (!hasLabel(dstNode, 'TimeSpan')) {
        throw new Error('HAPPENED_AT must target TimeSpan');
      }
      const props = dstNode.properties ?? {};
      if (!props.start || !props.end) {
        throw new Error('TimeSpan requires start & end');
      }
      if (new Date(props.start) > new Date(props.end)) {
        throw new Error('TimeSpan invalid');
      }
      break;
    }
    default:
      break;
  }
}

export function requireProjectEdgeIfNeeded(type: string, hasInProject: boolean) {
  if (RequiresProject.has(type) && !hasInProject) {
    throw new Error(`${type} requires IN_PROJECT link`);
  }
}

export function ensureEpisodicHasTimespan(layer: MemoryLayer | undefined, hasTimespan: boolean) {
  if (layer === 'episodic' && !hasTimespan) {
    throw new Error('Episodic Memory requires HAPPENED_AT');
  }
}

export { ConceptLabels, RequiresProject };
