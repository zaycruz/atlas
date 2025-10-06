import { Driver, Transaction } from 'neo4j-driver';
import { UpsertEntity, Link, MemoryLayer } from './types.js';
import { validateNode, validateLink, requireProjectEdgeIfNeeded, ensureEpisodicHasTimespan } from './schemaGuard.js';

export class KGService {
  constructor(private driver: Driver) {}

  private async assertNodeGuards(tx: Transaction, id: string, options?: { skipProject?: boolean; skipTimespan?: boolean }) {
    const res = await tx.run(
      `MATCH (n {id:$id})
       RETURN labels(n) AS labels,
              n.layer AS layer,
              exists((n)-[:IN_PROJECT]->(:Project)) AS hasProject,
              exists((n)-[:HAPPENED_AT]->(:TimeSpan)) AS hasTimespan`,
      { id }
    );
    if (!res.records.length) return;
    const record = res.records[0];
    const labels = record.get('labels') as string[];
    const layer = record.get('layer') as MemoryLayer | undefined;
    const hasProject = record.get('hasProject') as boolean;
    const hasTimespan = record.get('hasTimespan') as boolean;

    if (!options?.skipProject) {
      for (const type of ['Memory', 'File', 'Task', 'Event'] as const) {
        if (labels.includes(type)) {
          requireProjectEdgeIfNeeded(type, hasProject);
        }
      }
    }

    if (!options?.skipTimespan && labels.includes('Memory')) {
      ensureEpisodicHasTimespan(layer, hasTimespan);
    }
  }

  async upsertEntity(entity: UpsertEntity): Promise<void> {
    validateNode(entity.type, entity.props);
    const session = this.driver.session();
    try {
      const tx = session.beginTransaction();
      try {
        const result = await tx.run(
          `MERGE (n:${entity.type} {id:$id})
           ON CREATE SET n += $props, n.createdAt = coalesce($props.createdAt, datetime()), n.__isNew = true
           ON MATCH SET n += $props
           WITH n, coalesce(n.__isNew, false) AS created
           REMOVE n.__isNew
           RETURN created AS createdFlag`,
          { id: entity.props.id, props: entity.props }
        );
        const created = result.records[0]?.get('createdFlag') as boolean | undefined;
        await this.assertNodeGuards(tx, entity.props.id, {
          skipProject: Boolean(created),
          skipTimespan: Boolean(created),
        });
        await tx.commit();
      } catch (err) {
        await tx.rollback();
        throw err;
      }
    } finally {
      await session.close();
    }
  }

  async linkEdge(link: Link): Promise<void> {
    const session = this.driver.session();
    try {
      const tx = session.beginTransaction();
      try {
        const lookup = await tx.run(
          `MATCH (s {id:$sid}), (d {id:$did})
           RETURN labels(s) AS sl, s AS sNode, labels(d) AS dl, d AS dNode`,
          { sid: link.srcId, did: link.dstId }
        );
        if (!lookup.records.length) {
          throw new Error('src or dst not found');
        }
        const rec = lookup.records[0];
        const srcNode = rec.get('sNode');
        const dstNode = rec.get('dNode');
        const srcInfo = { labels: rec.get('sl') as string[], properties: srcNode.properties };
        const dstInfo = { labels: rec.get('dl') as string[], properties: dstNode.properties };
        validateLink(srcInfo, link, dstInfo);

        if (link.predicate === 'DERIVED_FROM') {
          const cyc = await tx.run(
            `MATCH (src {id:$src}), (dst {id:$dst})
             MATCH p=(dst)-[:DERIVED_FROM*1..5]->(src)
             RETURN p LIMIT 1`,
            { src: link.srcId, dst: link.dstId }
          );
          if (cyc.records.length) {
            throw new Error('DERIVED_FROM would create a cycle');
          }
        }

        await tx.run(
          `MATCH (s {id:$sid}), (d {id:$did})
           MERGE (s)-[r:${link.predicate}]->(d)`,
          { sid: link.srcId, did: link.dstId }
        );

        await this.assertNodeGuards(tx, link.srcId);
        await this.assertNodeGuards(tx, link.dstId);

        await tx.commit();
      } catch (err) {
        await tx.rollback();
        throw err;
      }
    } finally {
      await session.close();
    }
  }

  async find(opts: {
    type?: string;
    projectId?: string;
    topicName?: string;
    timeStart?: string;
    timeEnd?: string;
    layer?: string;
    limit?: number;
  }) {
    const where: string[] = ['exists(n.id)'];
    let match = `MATCH (n${opts.type ? `:${opts.type}` : ''})`;
    if (opts.projectId) {
      match += `-[:IN_PROJECT]->(:Project {id:$projectId})`;
    }
    if (opts.topicName) {
      match += `-[:ABOUT]->(:Topic {name:$topicName})`;
    }
    if (opts.timeStart || opts.timeEnd) {
      match += `-[:HAPPENED_AT]->(ts:TimeSpan)`;
      if (opts.timeStart) where.push('ts.start >= datetime($timeStart)');
      if (opts.timeEnd) where.push('ts.end <= datetime($timeEnd)');
    }
    if (opts.layer) {
      where.push('n.layer = $layer');
    }
    const limit = Math.min(opts.limit ?? 200, 200);
    const cypher = `${match} ${where.length ? `WHERE ${where.join(' AND ')}` : ''} RETURN n, labels(n) AS labels LIMIT ${limit}`;
    const session = this.driver.session();
    try {
      const res = await session.run(cypher, opts as any);
      return res.records.map((r) => {
        const node = r.get('n');
        return { ...node.properties, labels: r.get('labels') as string[] };
      });
    } finally {
      await session.close();
    }
  }

  async lineage(id: string, direction: 'up' | 'down' = 'up', maxDepth = 3) {
    const boundedDepth = Math.min(Math.max(maxDepth, 1), 6);
    const arrow = direction === 'down'
      ? `-[:DERIVED_FROM*1..${boundedDepth}]->`
      : `<-[:DERIVED_FROM*1..${boundedDepth}]-`;
    const cypher = `MATCH p=(n {id:$id})${arrow}(m) RETURN p`;
    const session = this.driver.session();
    try {
      const res = await session.run(cypher, { id });
      return res.records.map((record) => {
        const path = record.get('p');
        const nodes = path.nodes.map((node: any) => ({
          id: node.properties.id,
          labels: node.labels,
          properties: node.properties,
        }));
        const relationships = path.relationships.map((rel: any) => ({
          type: rel.type,
          start: rel.startNodeElementId,
          end: rel.endNodeElementId,
        }));
        return { nodes, relationships };
      });
    } finally {
      await session.close();
    }
  }
}
