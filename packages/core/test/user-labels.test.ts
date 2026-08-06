/**
 * Part B: capturing the user's own Thunderbird tags and meaningful folder as user-owned labels
 * (email_user_labels, migration v24). Covers the ingest helpers, the repository replace semantics,
 * the migration + FK cascade, and the /emails/push ingestion path. Emails and embeddings are never
 * touched by this feature.
 */
import { describe, it, expect, vi } from 'vitest';
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
import { TopicDiscoveryService } from '../src/services/topic-discovery-service.js';
import { summarizeUserLabels } from '../src/services/user-label-hints.js';
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

  it('excludes generic system folders (including provider variants) but keeps meaningful ones', () => {
    expect(isGenericFolder('/INBOX')).toBe(true);
    expect(isGenericFolder('/Trash')).toBe(true);
    for (const p of ['/[Gmail]/Sent Mail', '/[Gmail]/All Mail', '/[Gmail]/Starred', '/Bin']) {
      expect(isGenericFolder(p)).toBe(true);
    }
    expect(isGenericFolder('/INBOX/University')).toBe(false);
    expect(folderLabel('/INBOX')).toBeNull();
    expect(folderLabel('/INBOX/University')).toEqual({ key: 'inbox_university', label: 'University' });
  });

  it('matches on the leaf segment: a nested folder whose leaf is a generic word is dropped (by design)', () => {
    // Intentional: a nested folder named like a system folder (e.g. /Clients/Archive) is treated as
    // generic so provider system folders like /[Gmail]/Trash are excluded. Low harm: just no hint.
    expect(isGenericFolder('/Clients/Archive')).toBe(true);
    expect(isGenericFolder('/Clients/Acme')).toBe(false);
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

  it('topLabels shows the most-recently-synced label after a Thunderbird tag rename', () => {
    const h = seed();
    h.labels.replaceForEmail(
      h.accountId,
      'm1',
      [{ source: 'thunderbird_tag', key: '$k', label: 'Old Name' }],
      1,
    );
    h.labels.replaceForEmail(
      h.accountId,
      'm2',
      [{ source: 'thunderbird_tag', key: '$k', label: 'New Name' }],
      2,
    );
    const top = h.labels.topLabels(h.accountId, 'thunderbird_tag', 10).find((t) => t.key === '$k');
    expect(top).toMatchObject({ label: 'New Name', count: 2 });
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

  it('leaves an already-embedded message intact when its labels change on re-sync', async () => {
    const h = await pushHarness();
    await h.push([
      { messageId: 'm1', folder: '/Clients/Acme', subject: 's', body: 'hello', tags: [{ key: '$x', label: 'X' }] },
    ]);
    h.embeddings.saveEmbedding({ messageId: 'm1', accountId: h.accountId, modelId: 'bge-m3' }, axis(0));
    const embCount = () =>
      (h.db.prepare('SELECT COUNT(*) AS n FROM email_embedding_index').get() as { n: number }).n;
    expect(embCount()).toBe(1);
    // Re-sync the same message with a changed tag but the same body: the embedding must survive.
    await h.push([
      { messageId: 'm1', folder: '/Clients/Acme', subject: 's', body: 'hello', tags: [{ key: '$y', label: 'Y' }] },
    ]);
    expect(embCount()).toBe(1);
    expect(h.emails.count(h.accountId)).toBe(1);
    await h.app.close();
  });
});

describe('/emails/user-labels body-less backfill', () => {
  it('captures labels for existing messages without touching emails or embeddings, skipping unknown ids', async () => {
    const h = await pushHarness();
    // m1 is a fully-synced, embedded message; m-missing was never synced.
    await h.push([
      { messageId: 'm1', folder: '/Clients/Acme', subject: 's', body: 'hello', tags: [] },
    ]);
    h.embeddings.saveEmbedding({ messageId: 'm1', accountId: h.accountId, modelId: 'bge-m3' }, axis(0));

    const res = await h.app.inject({
      method: 'POST',
      url: '/emails/user-labels',
      payload: {
        accountId: h.accountId,
        items: [
          { messageId: 'm1', folder: '/Clients/Acme', tags: [{ key: '$work', label: 'Work' }] },
          { messageId: 'm-missing', folder: '/Clients/Acme', tags: [{ key: '$work', label: 'Work' }] },
        ],
      },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().updated).toBe(1);
    expect(h.emailUserLabels.labelsForEmail(h.accountId, 'm1')).toContainEqual({
      source: 'thunderbird_tag',
      key: '$work',
      label: 'Work',
    });
    // The unknown message never gained labels; emails and embeddings are unchanged.
    expect(h.emailUserLabels.labelsForEmail(h.accountId, 'm-missing')).toEqual([]);
    expect(h.emails.count(h.accountId)).toBe(1);
    expect(
      (h.db.prepare('SELECT COUNT(*) AS n FROM email_embedding_index').get() as { n: number }).n,
    ).toBe(1);
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

  it('scores a mid-range coherence for a mixed label and null when too few are embedded', () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const embeddings = new EmbeddingRepository(db);
    const labels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const ids = ['a1', 'a2', 'a3', 'a4'];
    emails.upsertBatch(ids.map((m) => ({ messageId: m, accountId: acc.id, folder: '/X', subject: m })));
    for (const m of ids) {
      labels.replaceForEmail(acc.id, m, [{ source: 'thunderbird_tag', key: '$mix', label: 'mix' }], 1);
    }
    // Half the emails on axis 0, half on axis 1 (orthogonal): coherence is well below 1 but positive.
    embeddings.saveEmbedding({ messageId: 'a1', accountId: acc.id, modelId: 'bge-m3' }, axis(0));
    embeddings.saveEmbedding({ messageId: 'a2', accountId: acc.id, modelId: 'bge-m3' }, axis(0));
    embeddings.saveEmbedding({ messageId: 'a3', accountId: acc.id, modelId: 'bge-m3' }, axis(1));
    embeddings.saveEmbedding({ messageId: 'a4', accountId: acc.id, modelId: 'bge-m3' }, axis(1));
    const mix = new UserLabelSuggestionService(labels, embeddings)
      .suggest(acc.id, 'bge-m3')
      .find((s) => s.key === '$mix');
    expect(mix!.coherence).toBeGreaterThan(0.6);
    expect(mix!.coherence!).toBeLessThan(0.8);

    // A label with only one embedded email cannot be scored: coherence is null.
    const acc2 = accounts.create({ address: 'y@x.com', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'b1', accountId: acc2.id, folder: '/Y', subject: 'b1' },
      { messageId: 'b2', accountId: acc2.id, folder: '/Y', subject: 'b2' },
      { messageId: 'b3', accountId: acc2.id, folder: '/Y', subject: 'b3' },
      { messageId: 'b4', accountId: acc2.id, folder: '/Y', subject: 'b4' },
    ]);
    for (const m of ['b1', 'b2', 'b3', 'b4']) {
      labels.replaceForEmail(acc2.id, m, [{ source: 'thunderbird_tag', key: '$solo', label: 'solo' }], 1);
    }
    embeddings.saveEmbedding({ messageId: 'b1', accountId: acc2.id, modelId: 'bge-m3' }, axis(2));
    const solo = new UserLabelSuggestionService(labels, embeddings)
      .suggest(acc2.id, 'bge-m3')
      .find((s) => s.key === '$solo');
    expect(solo!.coherence).toBeNull();
  });
});

describe('canonicalKey stability across reset (Thunderbird tag identity linchpin)', () => {
  it('re-derives the same canonicalKey for a re-created same-label category after a reset', () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const categories = new CategoryRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const first = categories.create({
      accountId: acc.id,
      label: 'Banking Transactions',
      source: 'auto',
    });
    expect(first.canonicalKey).toBeTruthy();
    // A reset wipes DB categories (ids change) but not Thunderbird tags. The re-discovered category
    // must derive the SAME canonicalKey so its mailpilot_<canonicalKey> tag key is reused, not duplicated.
    db.prepare('DELETE FROM categories WHERE id = ?').run(first.id);
    const second = categories.create({
      accountId: acc.id,
      label: 'Banking Transactions',
      source: 'auto',
    });
    expect(second.id).not.toBe(first.id);
    expect(second.canonicalKey).toBe(first.canonicalKey);
    db.close();
  });
});

describe('summarizeUserLabels hint builder (Part D)', () => {
  function seed() {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const labels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const ids = Array.from({ length: 6 }, (_, i) => `m${i}`);
    emails.upsertBatch(
      ids.map((messageId, i) => ({
        messageId,
        accountId: acc.id,
        folder: '/Clients/Acme',
        subject: `Acme invoice ${i}`,
        date: i,
      })),
    );
    for (const m of ids) {
      labels.replaceForEmail(
        acc.id,
        m,
        [{ source: 'thunderbird_tag', key: '$clients', label: 'Clients' }],
        1,
      );
    }
    labels.replaceForEmail(
      acc.id,
      'm0',
      [
        { source: 'thunderbird_tag', key: '$clients', label: 'Clients' },
        { source: 'thunderbird_tag', key: '$rare', label: 'rare' },
      ],
      1,
    );
    return { acc, labels };
  }

  it('summarizes well-used labels with subjects and drops below-floor ones', () => {
    const h = seed();
    const hint = summarizeUserLabels(h.labels, h.acc.id);
    expect(hint.labelCount).toBe(1);
    expect(hint.text).toContain('Clients (6 emails)');
    expect(hint.text).toContain('weak hints');
    expect(hint.text).not.toContain('rare');
  });

  it('returns an empty hint when there are no meaningful labels', () => {
    const db = openDatabase(':memory:');
    const acc = new AccountRepository(db).create({ address: 'w@x.com', kind: 'work' });
    const hint = summarizeUserLabels(new EmailUserLabelRepository(db), acc.id);
    expect(hint).toEqual({ text: '', labelCount: 0 });
  });
});

describe('discovery hinting from user labels (Part D/F)', () => {
  const summary = (messageId: string, fromAddr: string) => ({
    messageId,
    folder: '/INBOX',
    subject: `Subj ${messageId}`,
    fromAddr,
    date: 1,
    hasAttachments: false,
  });

  function harness(opts: { withRepo?: boolean; allowCloud?: boolean; insufficient?: boolean } = {}) {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emailsRepo = new EmailRepository(db);
    const userLabels = new EmailUserLabelRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const ids = Array.from({ length: 8 }, (_, i) => `u${i}`);
    emailsRepo.upsertBatch(
      ids.map((m) => ({
        messageId: m,
        accountId: acc.id,
        folder: '/Clients/Acme',
        subject: `Acme invoice ${m}`,
      })),
    );
    for (const m of ids) {
      userLabels.replaceForEmail(
        acc.id,
        m,
        [{ source: 'thunderbird_tag', key: '$clients', label: 'Clients' }],
        1,
      );
    }

    let embedN = 0;
    // Vague, banned labels produce zero concrete topics -> the run ends 'insufficient' after the
    // hinted prompt was already sent, exercising the insufficient() audit path.
    const topicsJson = opts.insufficient
      ? JSON.stringify({
          topics: [
            { label: 'Notifications', description: 'x' },
            { label: 'General Updates', description: 'y' },
          ],
        })
      : JSON.stringify({
          topics: [
            { label: 'Receipts & Invoices', description: 'Payment invoices.' },
            { label: 'Banking Transactions', description: 'Bank statements and transfers.' },
            { label: 'Security Alerts', description: 'Login and password alerts.' },
            { label: 'Travel Bookings', description: 'Flights and hotels.' },
            { label: 'Job Opportunities', description: 'Job listings and offers.' },
          ],
        });
    const chat = vi.fn(
      async (o: { provider: string; messages: Array<{ role: string; content: string }> }) => {
        void o;
        return topicsJson;
      },
    );
    const llm = { chat, embed: vi.fn(async () => Array.from(axis(embedN++))) };
    const emailsFake = {
      listSummaries: () => Array.from({ length: 200 }, (_, i) => summary(`m${i}`, `s${i % 5}@x.com`)),
      listSummariesSeeded: () => [],
      listUncategorizedSummariesStable: () => [],
      listSummariesByDomainSeeded: () => [],
      listSenders: () => Array.from({ length: 250 }, (_, i) => ({ fromAddr: `s${i % 5}@x.com` })),
    };
    const embeddings = { listForAccount: () => [], search: () => [] };
    const categories = {
      listForAccount: () => [],
      reconcileAutoCategories: () => ({ live: 5, omitted: [] }),
    };
    const audit = { log: vi.fn() };
    const cfg = () => ({
      allowCloudDiscovery: opts.allowCloud ?? false,
      chatBaseUrl: opts.allowCloud ? 'https://api.openai.com/v1' : undefined,
      chatModel: 'gpt-4o-mini',
    });
    const svc = new TopicDiscoveryService(
      llm as never,
      emailsFake as never,
      embeddings as never,
      categories as never,
      silentLogger,
      undefined,
      audit as never,
      cfg as never,
      opts.withRepo === false ? undefined : (userLabels as never),
    );
    const userPrompt = () =>
      (chat.mock.calls[0]?.[0].messages.find((m) => m.role === 'user')?.content ?? '') as string;
    const auditByStatus = (status: string) =>
      audit.log.mock.calls.map((c) => c[0]).find((a: { status: string }) => a.status === status);
    const okAudit = () => auditByStatus('ok');
    return { svc, acc, chat, userPrompt, okAudit, auditByStatus };
  }

  it('includes user-label hints locally and records it in the audit', async () => {
    const h = harness();
    const res = await h.svc.discover(h.acc.id, 'bge-m3', 'gen');
    expect(res.status).toBe('ok');
    expect(h.chat.mock.calls[0]![0].provider).toBe('main');
    expect(h.userPrompt()).toContain('Clients');
    expect(h.userPrompt()).toContain('weak hints');
    expect(h.okAudit().provider).toBe('local');
    expect(h.okAudit().fieldsRead).toContain('user_label_hints');
  });

  it('sends hints to the cloud only under explicit cloud-discovery opt-in, and audits it', async () => {
    const h = harness({ allowCloud: true });
    const res = await h.svc.discover(h.acc.id, 'bge-m3', 'gen');
    expect(res.status).toBe('ok');
    expect(h.chat.mock.calls[0]![0].provider).toBe('chat');
    expect(h.userPrompt()).toContain('Clients');
    expect(h.okAudit().provider).toBe('cloud');
    expect(h.okAudit().fieldsRead).toContain('user_label_hints');
  });

  it('omits the hint and the audit marker when no user-label repo is wired (old behavior)', async () => {
    const h = harness({ withRepo: false });
    await h.svc.discover(h.acc.id, 'bge-m3', 'gen');
    expect(h.userPrompt()).not.toContain('weak hints');
    expect(h.okAudit().fieldsRead).not.toContain('user_label_hints');
  });

  it('records the hint exposure even when the run ends insufficient after sending the prompt', async () => {
    const h = harness({ insufficient: true });
    const res = await h.svc.discover(h.acc.id, 'bge-m3', 'gen');
    expect(res.status).toBe('insufficient_categories');
    // The hinted prompt was already sent, so the audit must not claim the labels were unexposed.
    expect(h.userPrompt()).toContain('weak hints');
    expect(h.auditByStatus('insufficient').fieldsRead).toContain('user_label_hints');
  });
});
