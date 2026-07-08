/**
 * Part B: capturing the user's own Thunderbird tags and meaningful folder as user-owned labels
 * (email_user_labels, migration v24). Covers the ingest helpers, the repository replace semantics,
 * the migration + FK cascade, and the /emails/push ingestion path. Emails and embeddings are never
 * touched by this feature.
 */
import { describe, it, expect } from 'vitest';
import Fastify from 'fastify';
import type { Logger } from 'pino';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { EmailUserLabelRepository } from '../src/repositories/email-user-label-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { TriageRepository } from '../src/repositories/triage-repository.js';
import { DashboardService } from '../src/services/dashboard-service.js';
import { UserLabelSuggestionService } from '../src/services/user-label-suggestion-service.js';
import { registerEmailRoutes } from '../src/routes/emails.js';
import { registerDashboardRoutes } from '../src/routes/dashboard.js';
import type { AppContext } from '../src/context.js';
import {
  buildUserLabels,
  folderLabel,
  isGenericFolder,
} from '../src/services/user-label-ingest.js';

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

describe('user-label ingest helpers', () => {
  it('drops MailPilot-managed tags and empty values, keeps user tags', () => {
    const labels = buildUserLabels('/INBOX/Work', [
      { key: 'mailpilot_banking', label: 'Banking Transactions' },
      { key: '$work', label: 'Work' },
      { key: 'empty', label: '' },
    ]);
    expect(labels).toContainEqual({ source: 'thunderbird_tag', key: '$work', label: 'Work' });
    expect(labels.some((l) => l.key === 'mailpilot_banking')).toBe(false);
    expect(labels.some((l) => l.label === '')).toBe(false);
  });

  it('excludes generic system folders but keeps meaningful ones', () => {
    expect(isGenericFolder('/INBOX')).toBe(true);
    expect(isGenericFolder('/Trash')).toBe(true);
    expect(isGenericFolder('/INBOX/University')).toBe(false);
    expect(folderLabel('/INBOX')).toBeNull();
    expect(folderLabel('/INBOX/University')).toEqual({ key: 'inbox_university', label: 'University' });
  });

  it('adds a folder label only for a non-generic folder', () => {
    expect(buildUserLabels('/Inbox', []).length).toBe(0);
    expect(buildUserLabels('/Clients/Acme', [])).toEqual([
      { source: 'folder', key: 'clients_acme', label: 'Acme' },
    ]);
  });
});

describe('email_user_labels migration (v24)', () => {
  it('creates the table with the expected columns and a source CHECK', () => {
    const db = openDatabase(':memory:');
    const cols = (
      db.prepare('PRAGMA table_info(email_user_labels)').all() as Array<{ name: string }>
    ).map((c) => c.name);
    expect(cols).toEqual([
      'account_id',
      'message_id',
      'source',
      'key',
      'label',
      'synced_at',
    ]);
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm1', accountId: acc.id, folder: '/INBOX' }]);
    // The CHECK rejects an unknown source.
    expect(() =>
      db
        .prepare(
          'INSERT INTO email_user_labels (account_id, message_id, source, key, label, synced_at) VALUES (?, ?, ?, ?, ?, ?)',
        )
        .run(acc.id, 'm1', 'bogus', 'k', 'l', 1),
    ).toThrow();
    db.close();
  });
});

describe('EmailUserLabelRepository', () => {
  function seed() {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const labels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    emails.upsertBatch(
      ['m1', 'm2', 'm3'].map((messageId, i) => ({
        messageId,
        accountId: acc.id,
        folder: '/INBOX',
        subject: `subject ${i}`,
        date: 1000 + i,
      })),
    );
    return { db, accounts, emails, labels, accountId: acc.id };
  }

  it('replaces a message label set and reflects a removed tag on the next sync', () => {
    const h = seed();
    h.labels.replaceForEmail(
      h.accountId,
      'm1',
      [
        { source: 'thunderbird_tag', key: '$work', label: 'Work' },
        { source: 'thunderbird_tag', key: '$uni', label: 'Uni' },
      ],
      1,
    );
    expect(h.labels.labelsForEmail(h.accountId, 'm1').map((l) => l.key).sort()).toEqual([
      '$uni',
      '$work',
    ]);
    // Next sync: the user removed $uni.
    h.labels.replaceForEmail(
      h.accountId,
      'm1',
      [{ source: 'thunderbird_tag', key: '$work', label: 'Work' }],
      2,
    );
    expect(h.labels.labelsForEmail(h.accountId, 'm1').map((l) => l.key)).toEqual(['$work']);
  });

  it('aggregates top labels, distinct counts, and representative subjects', () => {
    const h = seed();
    for (const m of ['m1', 'm2', 'm3']) {
      h.labels.replaceForEmail(
        h.accountId,
        m,
        [{ source: 'thunderbird_tag', key: '$work', label: 'Work' }],
        1,
      );
    }
    h.labels.replaceForEmail(
      h.accountId,
      'm1',
      [
        { source: 'thunderbird_tag', key: '$work', label: 'Work' },
        { source: 'thunderbird_tag', key: '$uni', label: 'Uni' },
      ],
      1,
    );
    const top = h.labels.topLabels(h.accountId, 'thunderbird_tag', 10);
    expect(top[0]).toMatchObject({ key: '$work', count: 3 });
    expect(top.find((t) => t.key === '$uni')?.count).toBe(1);
    expect(h.labels.countDistinct(h.accountId, 'thunderbird_tag')).toBe(2);
    expect(h.labels.representativeSubjects(h.accountId, 'thunderbird_tag', '$work', 5).length).toBe(3);
  });

  it('cascades label deletion when its email is deleted', () => {
    const h = seed();
    h.labels.replaceForEmail(
      h.accountId,
      'm1',
      [{ source: 'thunderbird_tag', key: '$work', label: 'Work' }],
      1,
    );
    h.db.prepare('DELETE FROM emails WHERE message_id = ? AND account_id = ?').run('m1', h.accountId);
    expect(h.labels.labelsForEmail(h.accountId, 'm1')).toEqual([]);
  });
});

async function pushHarness() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const emailUserLabels = new EmailUserLabelRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
  const app = Fastify();
  await registerEmailRoutes(app, {
    repos: { accounts, emails, embeddings, emailUserLabels },
    logger: silentLogger,
  } as unknown as AppContext);
  await app.ready();
  const push = (emailsPayload: unknown[]) =>
    app.inject({ method: 'POST', url: '/emails/push', payload: { accountId: acc.id, emails: emailsPayload } });
  return { db, app, emails, embeddings, emailUserLabels, accountId: acc.id, push };
}

describe('/emails/push user-label ingestion', () => {
  it('stores non-MailPilot tags and the folder, ignoring MailPilot tags and generic folders', async () => {
    const h = await pushHarness();
    const res = await h.push([
      {
        messageId: 'm1',
        folder: '/INBOX/University',
        subject: 'hi',
        tags: [
          { key: '$work', label: 'Work' },
          { key: 'mailpilot_banking', label: 'Banking Transactions' },
        ],
      },
      { messageId: 'm2', folder: '/INBOX', subject: 'inbox only', tags: [] },
    ]);
    expect(res.statusCode).toBe(200);
    const m1 = h.emailUserLabels.labelsForEmail(h.accountId, 'm1');
    expect(m1).toContainEqual({ source: 'thunderbird_tag', key: '$work', label: 'Work' });
    expect(m1).toContainEqual({ source: 'folder', key: 'inbox_university', label: 'University' });
    expect(m1.some((l) => l.key === 'mailpilot_banking')).toBe(false);
    // m2 is a generic Inbox folder with no user tags: nothing stored.
    expect(h.emailUserLabels.labelsForEmail(h.accountId, 'm2')).toEqual([]);
    await h.app.close();
  });

  it('updates labels when a tag is removed and when the message moves folders', async () => {
    const h = await pushHarness();
    await h.push([
      { messageId: 'm1', folder: '/Clients/Acme', tags: [{ key: '$hot', label: 'Hot' }] },
    ]);
    expect(h.emailUserLabels.labelsForEmail(h.accountId, 'm1').map((l) => l.key).sort()).toEqual([
      '$hot',
      'clients_acme',
    ]);
    // Re-sync: tag removed, message moved into a different meaningful folder.
    await h.push([{ messageId: 'm1', folder: '/Clients/Globex', tags: [] }]);
    expect(h.emailUserLabels.labelsForEmail(h.accountId, 'm1')).toEqual([
      { source: 'folder', key: 'clients_globex', label: 'Globex' },
    ]);
    await h.app.close();
  });

  it('leaves emails and embeddings untouched', async () => {
    const h = await pushHarness();
    await h.push([{ messageId: 'm1', folder: '/Clients/Acme', subject: 's', tags: [{ key: '$x', label: 'X' }] }]);
    expect(h.emails.count(h.accountId)).toBe(1);
    const embCount = (
      h.db.prepare('SELECT COUNT(*) AS n FROM email_embedding_index').get() as { n: number }
    ).n;
    expect(embCount).toBe(0);
    await h.app.close();
  });
});

describe('dashboard userOrganization summary (Part C)', () => {
  async function seedApp() {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const triage = new TriageRepository(db);
    const categories = new CategoryRepository(db);
    const labels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    emails.upsertBatch(
      ['m1', 'm2', 'm3'].map((messageId) => ({
        messageId,
        accountId: acc.id,
        folder: '/Clients/Acme',
        subject: 's',
      })),
    );
    for (const m of ['m1', 'm2', 'm3']) {
      labels.replaceForEmail(
        acc.id,
        m,
        [
          { source: 'thunderbird_tag', key: '$work', label: 'Work' },
          { source: 'folder', key: 'clients_acme', label: 'Acme' },
        ],
        1,
      );
    }
    labels.replaceForEmail(
      acc.id,
      'm1',
      [
        { source: 'thunderbird_tag', key: '$work', label: 'Work' },
        { source: 'thunderbird_tag', key: '$uni', label: 'Uni' },
        { source: 'folder', key: 'clients_acme', label: 'Acme' },
      ],
      1,
    );
    const dashboard = new DashboardService(emails, triage, categories, labels);
    const app = Fastify();
    await registerDashboardRoutes(app, {
      repos: { accounts },
      services: { dashboard },
    } as unknown as AppContext);
    await app.ready();
    return { app, accountId: acc.id };
  }

  it('includes user-tag and folder counts with top labels, excluding AI categories', async () => {
    const { app, accountId } = await seedApp();
    const res = await app.inject({ method: 'GET', url: `/dashboard?accountId=${accountId}` });
    expect(res.statusCode).toBe(200);
    const org = res.json().userOrganization;
    expect(org.tagCount).toBe(2);
    expect(org.folderCount).toBe(1);
    expect(org.topTags[0]).toMatchObject({ key: '$work', label: 'Work', count: 3 });
    expect(org.topFolders[0]).toMatchObject({ key: 'clients_acme', label: 'Acme', count: 3 });
    await app.close();
  });

  it('reports an empty user organization when no labels exist', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const dashboard = new DashboardService(
      new EmailRepository(db),
      new TriageRepository(db),
      new CategoryRepository(db),
      new EmailUserLabelRepository(db),
    );
    const app = Fastify();
    await registerDashboardRoutes(app, {
      repos: { accounts },
      services: { dashboard },
    } as unknown as AppContext);
    await app.ready();
    const res = await app.inject({ method: 'GET', url: `/dashboard?accountId=${acc.id}` });
    expect(res.json().userOrganization).toEqual({
      tagCount: 0,
      folderCount: 0,
      topTags: [],
      topFolders: [],
    });
    await app.close();
    db.close();
  });
});

describe('UserLabelSuggestionService (Part E, import-prep only)', () => {
  function seed() {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const embeddings = new EmbeddingRepository(db);
    const labels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const ids = ['m1', 'm2', 'm3', 'm4', 'm5'];
    emails.upsertBatch(
      ids.map((messageId, i) => ({
        messageId,
        accountId: acc.id,
        folder: '/Clients/Acme',
        subject: `acme ${i}`,
      })),
    );
    for (const m of ids) {
      labels.replaceForEmail(
        acc.id,
        m,
        [{ source: 'thunderbird_tag', key: '$clients', label: 'clients' }],
        1,
      );
      embeddings.saveEmbedding({ messageId: m, accountId: acc.id, modelId: 'bge-m3' }, axis(0));
    }
    // A rarely-used tag below the MIN_COUNT floor must not be suggested.
    labels.replaceForEmail(
      acc.id,
      'm1',
      [
        { source: 'thunderbird_tag', key: '$clients', label: 'clients' },
        { source: 'thunderbird_tag', key: '$rare', label: 'rare' },
      ],
      1,
    );
    return { acc, emails, embeddings, labels };
  }

  it('suggests only sufficiently-used labels, title-cased, with no coherence when no model given', () => {
    const h = seed();
    const svc = new UserLabelSuggestionService(h.labels, h.embeddings);
    const out = svc.suggest(h.acc.id);
    const clients = out.find((s) => s.key === '$clients');
    expect(clients).toMatchObject({
      source: 'thunderbird_tag',
      count: 5,
      coherence: null,
      suggestedCategoryLabel: 'Clients',
    });
    expect(clients!.representativeSubjects.length).toBeGreaterThan(0);
    expect(out.some((s) => s.key === '$rare')).toBe(false);
  });

  it('scores coherence from embeddings when an embedding model is supplied', () => {
    const h = seed();
    const svc = new UserLabelSuggestionService(h.labels, h.embeddings);
    const clients = svc.suggest(h.acc.id, 'bge-m3').find((s) => s.key === '$clients');
    // All the label's emails share one embedding axis, so they are perfectly coherent.
    expect(clients!.coherence).toBeCloseTo(1, 5);
  });
});
