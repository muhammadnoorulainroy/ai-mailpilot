import type { Database } from 'better-sqlite3';

export interface Migration {
  version: number;
  name: string;
  up: (db: Database) => void;
}

export function runMigrations(db: Database, migrations: Migration[]): number[] {
  db.exec(`
    CREATE TABLE IF NOT EXISTS migrations (
      version INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at INTEGER NOT NULL
    )
  `);

  const applied = new Set(
    db.prepare<[], { version: number }>('SELECT version FROM migrations').all().map((r) => r.version),
  );

  const sorted = [...migrations].sort((a, b) => a.version - b.version);
  const ran: number[] = [];

  const insert = db.prepare<[number, string, number]>(
    'INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)',
  );

  for (const m of sorted) {
    if (applied.has(m.version)) continue;

    const tx = db.transaction(() => {
      m.up(db);
      insert.run(m.version, m.name, Date.now());
    });

    tx();
    ran.push(m.version);
  }

  return ran;
}
