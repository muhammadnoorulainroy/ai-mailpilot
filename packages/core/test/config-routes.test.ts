/**
 * Config route tests for the feature-flag surface (Phase 4). Verifies GET /config exposes `features`,
 * PATCH /config updates a flag and leaves other sections alone, a legacy config without a `features`
 * block still parses and reports the flag false, and an invalid flag value is rejected. saveConfig is
 * stubbed so these tests never write to the real on-disk config.
 */
import { describe, it, expect, vi } from 'vitest';
import Fastify from 'fastify';
import type { Logger } from 'pino';
import { AppConfigSchema } from '../src/config/schema.js';
import { registerConfigRoutes } from '../src/routes/config.js';
import type { AppContext } from '../src/context.js';

vi.mock('../src/config/config.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/config/config.js')>();
  return { ...actual, saveConfig: vi.fn() };
});

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

async function buildApp(rawConfig: Record<string, unknown> = {}) {
  const config = AppConfigSchema.parse(rawConfig);
  const llm = { embed: async () => [] } as unknown as AppContext['llm'];
  const ctx = { config, llm, logger: silentLogger } as unknown as AppContext;
  const app = Fastify();
  await registerConfigRoutes(app, ctx);
  await app.ready();
  return { app, ctx };
}

describe('config routes: feature flags', () => {
  it('GET /config exposes features with multiPrototypeCategories defaulting false', async () => {
    const { app } = await buildApp();
    const res = await app.inject({ method: 'GET', url: '/config' });
    expect(res.statusCode).toBe(200);
    expect(res.json().features).toEqual({ multiPrototypeCategories: false });
    await app.close();
  });

  it('PATCH /config updates features.multiPrototypeCategories and persists it on the context', async () => {
    const { app, ctx } = await buildApp();
    const res = await app.inject({
      method: 'PATCH',
      url: '/config',
      payload: { features: { multiPrototypeCategories: true } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().features.multiPrototypeCategories).toBe(true);
    expect(ctx.config.features.multiPrototypeCategories).toBe(true);
    await app.close();
  });

  it('PATCH /config with features leaves the llm section untouched', async () => {
    const { app, ctx } = await buildApp({
      llm: { baseUrl: 'http://192.168.1.9:1234/v1', embeddingModel: 'bge-m3' },
    });
    const before = ctx.config.llm.baseUrl;
    const res = await app.inject({
      method: 'PATCH',
      url: '/config',
      payload: { features: { multiPrototypeCategories: true } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().llm.baseUrl).toBe(before);
    expect(ctx.config.llm.baseUrl).toBe('http://192.168.1.9:1234/v1');
    await app.close();
  });

  it('an empty features patch leaves the current flag value unchanged', async () => {
    const { app, ctx } = await buildApp({ features: { multiPrototypeCategories: true } });
    const res = await app.inject({
      method: 'PATCH',
      url: '/config',
      payload: { features: {} },
    });
    expect(res.statusCode).toBe(200);
    expect(ctx.config.features.multiPrototypeCategories).toBe(true);
    await app.close();
  });

  it('a legacy config without a features block parses and reports the flag false', async () => {
    const { app } = await buildApp({
      version: 1,
      locale: 'en',
      autoIndex: true,
      indexedFolders: [],
      llm: { baseUrl: 'http://localhost:11434/v1', embeddingModel: 'bge-m3' },
      authToken: 'tok',
    });
    const res = await app.inject({ method: 'GET', url: '/config' });
    expect(res.statusCode).toBe(200);
    expect(res.json().features).toEqual({ multiPrototypeCategories: false });
    expect(res.json().autoIndex).toBe(true);
    await app.close();
  });

  it('ignores an inherited-name feature key while still applying a real flag', async () => {
    const { app, ctx } = await buildApp();
    const res = await app.inject({
      method: 'PATCH',
      url: '/config',
      payload: { features: { toString: 1, multiPrototypeCategories: true } },
    });
    expect(res.statusCode).toBe(200);
    // The real flag is applied; the inherited name is not copied onto the config as an own property.
    expect(ctx.config.features.multiPrototypeCategories).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(ctx.config.features, 'toString')).toBe(false);
  });

  it('rejects an invalid feature flag value with 400', async () => {
    const { app, ctx } = await buildApp();
    const res = await app.inject({
      method: 'PATCH',
      url: '/config',
      payload: { features: { multiPrototypeCategories: 'yes' } },
    });
    expect(res.statusCode).toBe(400);
    expect(ctx.config.features.multiPrototypeCategories).toBe(false);
    await app.close();
  });
});
