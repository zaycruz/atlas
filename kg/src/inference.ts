import { Driver } from 'neo4j-driver';

export async function runInference(driver: Driver): Promise<void> {
  const session = driver.session();
  try {
    await session.run(`
      MATCH (r:Memory {layer:'reflection'})-[:DERIVED_FROM]->(x)-[:ABOUT]->(t:Topic)
      WHERE NOT (r)-[:ABOUT]->(t)
      MERGE (r)-[:ABOUT]->(t)
    `);
    await session.run(`
      MATCH (r:Memory)-[:DERIVED_FROM]->(x)-[:IN_PROJECT]->(p:Project)
      WHERE NOT (r)-[:IN_PROJECT]->(p)
      MERGE (r)-[:IN_PROJECT]->(p)
    `);
  } finally {
    await session.close();
  }
}
