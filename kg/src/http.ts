import Fastify from 'fastify';
import { connect } from './neo.js';
import { KGService } from './service.js';
import { runInference } from './inference.js';
import { UpsertEntity, Link } from './types.js';

export async function buildServer() {
  const requiredEnv = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASS'] as const;
  for (const key of requiredEnv) {
    if (!process.env[key]) {
      throw new Error(`Missing environment variable ${key}`);
    }
  }
  const driver = connect(process.env.NEO4J_URI!, process.env.NEO4J_USER!, process.env.NEO4J_PASS!);
  const kg = new KGService(driver);
  const app = Fastify({ logger: true });

  app.post('/entity', async (req, res) => {
    await kg.upsertEntity(req.body as UpsertEntity);
    return { ok: true };
  });

  app.post('/link', async (req, res) => {
    await kg.linkEdge(req.body as Link);
    return { ok: true };
  });

  app.post('/find', async (req, res) => {
    const results = await kg.find(req.body as any);
    return results;
  });

  app.get('/lineage/:id', async (req, res) => {
    const { id } = req.params as { id: string };
    const { direction = 'up', maxDepth } = req.query as { direction?: string; maxDepth?: string };
    const maybeDepth = maxDepth !== undefined ? Number(maxDepth) : undefined;
    const depth = Number.isFinite(maybeDepth) && (maybeDepth as number) > 0 ? (maybeDepth as number) : 3;
    const lineage = await kg.lineage(id, direction === 'down' ? 'down' : 'up', depth);
    return lineage;
  });

  app.post('/admin/inference', async () => {
    await runInference(driver);
    return { ok: true };
  });

  app.addHook('onClose', async () => {
    await driver.close();
  });

  return { app, driver, kg };
}

export async function start() {
  const { app } = await buildServer();
  const port = Number(process.env.PORT ?? 4545);
  await app.listen({ port, host: '0.0.0.0' });
  return app;
}
