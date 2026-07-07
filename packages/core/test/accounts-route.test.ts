/**
 * Account route tests for the per-account discovery consent flag.
 */
import { describe, it, expect } from 'vitest';
import Fastify from 'fastify';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { registerAccountRoutes } from '../src/routes/accounts.js';
import type { AppContext } from '../src/context.js';

async function buildApp() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const app = Fastify();
  await registerAccountRoutes(app, {
    repos: { accounts },
  } as unknown as AppContext);
  await app.ready();
  return { app, db };
}

describe('account discovery consent routes', () => {
  it('creates personal accounts with discovery disabled and lets the user opt in', async () => {
    const { app, db } = await buildApp();
    const created = await app.inject({
      method: 'POST',
      url: '/accounts',
      payload: { address: 'personal@example.com', kind: 'personal' },
    });
    expect(created.statusCode).toBe(200);
    expect(created.json().account).toMatchObject({
      address: 'personal@example.com',
      kind: 'personal',
      discoveryEnabled: false,
    });

    const id = created.json().account.id as string;
    const updated = await app.inject({
      method: 'PATCH',
      url: `/accounts/${id}/discovery`,
      payload: { discoveryEnabled: true },
    });
    expect(updated.statusCode).toBe(200);
    expect(updated.json().account.discoveryEnabled).toBe(true);

    const listed = await app.inject({ method: 'GET', url: '/accounts' });
    expect(listed.json().accounts[0]).toMatchObject({ id, discoveryEnabled: true });

    await app.close();
    db.close();
  });

  it('validates discovery updates and returns 404 for missing accounts', async () => {
    const { app, db } = await buildApp();
    const bad = await app.inject({
      method: 'PATCH',
      url: '/accounts/missing/discovery',
      payload: { discoveryEnabled: 'yes' },
    });
    expect(bad.statusCode).toBe(400);

    const missing = await app.inject({
      method: 'PATCH',
      url: '/accounts/missing/discovery',
      payload: { discoveryEnabled: true },
    });
    expect(missing.statusCode).toBe(404);

    await app.close();
    db.close();
  });
});
