export class KGClient {
  constructor(private base = 'http://localhost:4545') {
    this.base = this.base.replace(/\/$/, '');
  }

  async upsert(type: string, props: any) {
    await fetch(`${this.base}/entity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, props }),
    });
  }

  async link(srcId: string, predicate: string, dstId: string) {
    await fetch(`${this.base}/link`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ srcId, predicate, dstId }),
    });
  }

  async find(query: Record<string, unknown>) {
    const res = await fetch(`${this.base}/find`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(query),
    });
    if (!res.ok) {
      throw new Error(`find failed: ${res.status}`);
    }
    return res.json();
  }

  async lineage(id: string, direction: 'up' | 'down' = 'up', maxDepth = 3) {
    const res = await fetch(`${this.base}/lineage/${encodeURIComponent(id)}?direction=${direction}&maxDepth=${maxDepth}`);
    if (!res.ok) {
      throw new Error(`lineage failed: ${res.status}`);
    }
    return res.json();
  }
}
