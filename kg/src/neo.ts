import neo4j, { Driver } from 'neo4j-driver';

export function connect(uri: string, user: string, pass: string): Driver {
  return neo4j.driver(uri, neo4j.auth.basic(user, pass), { disableLosslessIntegers: true });
}
