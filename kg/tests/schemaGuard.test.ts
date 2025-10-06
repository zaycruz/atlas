import { describe, it, expect } from 'vitest';
import { validateNode, validateLink, requireProjectEdgeIfNeeded, ensureEpisodicHasTimespan } from '../src/schemaGuard.js';
import { Link } from '../src/types.js';

describe('validateNode', () => {
  it('requires id', () => {
    expect(() => validateNode('Project', {} as any)).toThrow('Node requires id');
  });

  it('validates TimeSpan bounds', () => {
    expect(() =>
      validateNode('TimeSpan', { id: 'ts1', start: '2024-01-02T00:00:00Z', end: '2024-01-01T00:00:00Z' })
    ).toThrow('TimeSpan start > end');
  });

  it('rejects invalid memory layer', () => {
    expect(() => validateNode('Memory', { id: 'm1', layer: 'unknown' })).toThrow('Invalid Memory.layer');
  });
});

describe('validateLink', () => {
  it('requires ABOUT target to be concept', () => {
    const link: Link = { srcId: 's', predicate: 'ABOUT', dstId: 'd' };
    expect(() => validateLink({ labels: ['Memory'] }, link, { labels: ['File'] })).toThrow(
      'ABOUT must target Concept (e.g., Topic)'
    );
  });

  it('validates HAPPENED_AT target is timespan with bounds', () => {
    const link: Link = { srcId: 's', predicate: 'HAPPENED_AT', dstId: 'd' };
    expect(() => validateLink({ labels: ['Event'] }, link, { labels: ['TimeSpan'], properties: { start: '2024', end: '2023' } })).toThrow(
      'TimeSpan invalid'
    );
  });
});

describe('requireProjectEdgeIfNeeded', () => {
  it('throws when missing project link', () => {
    expect(() => requireProjectEdgeIfNeeded('Memory', false)).toThrow('Memory requires IN_PROJECT link');
  });
});

describe('ensureEpisodicHasTimespan', () => {
  it('enforces timespan for episodic memory', () => {
    expect(() => ensureEpisodicHasTimespan('episodic', false)).toThrow('Episodic Memory requires HAPPENED_AT');
  });

  it('allows semantic memory without timespan', () => {
    expect(() => ensureEpisodicHasTimespan('semantic', false)).not.toThrow();
  });
});
