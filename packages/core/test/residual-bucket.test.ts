/**
 * Residual buckets: cover the uncategorized long tail with signal-matched broad buckets. Verifies the
 * precision matchers (newsletters/promotions/notifications in; security/receipts out), the coverage
 * floor, widening an existing same-purpose category vs creating a new one, and that coverResidual is
 * a no-op unless enabled and only ever files the uncategorized backlog.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { ResidualBucketService } from '../src/services/residual-bucket-service.js';

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

const NEWSLETTERS = [
  ['Your Weekly Digest is here', 'newsletter@substack.com'],
  ['Medium Daily Digest', 'noreply@medium.com'],
  ['This week: top stories in tech', 'digest@news.example.com'],
  ['The Monday Roundup', 'hello@thebulletin.co'],
  ['New posts from the writers you follow', 'noreply@wordpress.com'],
] as const;
const PROMOS = [
  ['50% off everything this weekend', 'deals@shop.com'],
  ['Flash Sale ends tonight', 'offers@brand.com'],
  ['Your exclusive discount code inside', 'promo@store.com'],
  ['Limited time offer: save big', 'marketing@retail.com'],
  ['Price drop on items you viewed', 'noreply@marketplace.com'],
] as const;
const NOTIFS = [
  ['New comment on your post', 'no-reply@platform.com'],
  ['Someone liked your update', 'notifications@social.com'],
  ['You have a new follower', 'noreply@network.com'],
  ['Reminder: your activity summary', 'automated@app.com'],
  ['New reply to your thread', 'donotreply@forum.com'],
] as const;
// Must NEVER be swept into a broad bucket.
const SECURITY = [
  ['Verify your login', 'noreply@bank.com'],
  ['Your password reset code', 'security@site.com'],
] as const;
const RECEIPTS = [
  ['Your order #4491 has shipped', 'noreply@store.com'],
  ['Invoice attached for your payment', 'billing@vendor.com'],
] as const;

/** Seed `copies` uncategorized emails per template subject/sender. */
function seed(copies: number) {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const emails = new EmailRepository(db);
  const categories = new CategoryRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
  const rows: Array<{
    messageId: string;
    accountId: string;
    folder: string;
    subject: string;
    fromAddr: string;
  }> = [];
  let i = 0;
  const add = (templates: ReadonlyArray<readonly [string, string]>, n: number) => {
    for (let c = 0; c < n; c++) {
      for (const [subject, fromAddr] of templates) {
        rows.push({ messageId: `m${i++}`, accountId: acc.id, folder: '/INBOX', subject, fromAddr });
      }
    }
  };
  add(NEWSLETTERS, copies);
  add(PROMOS, copies);
  add(NOTIFS, copies);
  add(SECURITY, copies);
  add(RECEIPTS, copies);
  emails.upsertBatch(rows);
  return { db, accounts, emails, categories, accountId: acc.id };
}

describe('ResidualBucketService.detect', () => {
  it('detects the three broad buckets above the floor, excluding security and receipts', () => {
    const h = seed(8); // 8 * 5 = 40 per bucket, above MIN_BUCKET_COVERAGE
    const svc = new ResidualBucketService(h.categories, h.emails, silentLogger, () => true);
    const plans = svc.detect(h.accountId);
    const byKey = new Map(plans.map((p) => [p.key, p]));
    expect([...byKey.keys()].sort()).toEqual([
      'newsletters_digests',
      'notifications_alerts',
      'promotions_deals',
    ]);
    // 40 matched each; nothing from security/receipts leaked into the counts.
    for (const p of plans) expect(p.matched).toBe(40);
    // No existing categories, so every bucket would be created.
    for (const p of plans) expect(p.targetCategoryId).toBeNull();
  });

  it('drops a bucket below the coverage floor', () => {
    const h = seed(1); // 5 per bucket, below the floor of 25
    const svc = new ResidualBucketService(h.categories, h.emails, silentLogger, () => true);
    expect(svc.detect(h.accountId)).toEqual([]);
  });

  it('targets an existing same-purpose category to widen instead of duplicating', () => {
    const h = seed(8);
    const marketing = h.categories.create({
      accountId: h.accountId,
      label: 'Marketing and Promotions',
      source: 'auto',
    });
    const svc = new ResidualBucketService(h.categories, h.emails, silentLogger, () => true);
    const plans = svc.detect(h.accountId);
    const promo = plans.find((p) => p.key === 'promotions_deals')!;
    expect(promo.targetCategoryId).toBe(marketing.id); // widen existing
    const news = plans.find((p) => p.key === 'newsletters_digests')!;
    expect(news.targetCategoryId).toBeNull(); // no newsletter category -> create
  });
});

describe('ResidualBucketService.coverResidual', () => {
  it('is a no-op when the flag is off', () => {
    const h = seed(8);
    const before = h.categories.countUncategorized(h.accountId);
    const res = new ResidualBucketService(
      h.categories,
      h.emails,
      silentLogger,
      () => false,
    ).coverResidual(h.accountId);
    expect(res).toMatchObject({ buckets: [], totalAssigned: 0 });
    expect(h.categories.countUncategorized(h.accountId)).toBe(before);
  });

  it('creates buckets, widens existing, files matching mail, and never touches security/receipts', () => {
    const h = seed(8);
    const marketing = h.categories.create({
      accountId: h.accountId,
      label: 'Marketing and Promotions',
      source: 'auto',
    });
    const svc = new ResidualBucketService(h.categories, h.emails, silentLogger, () => true);
    const res = svc.coverResidual(h.accountId);

    expect(res.totalAssigned).toBe(120); // 40 newsletters + 40 promos + 40 notifications
    // Promotions widened the existing Marketing category (not a new one).
    const promoBucket = res.buckets.find((b) => b.label === 'Marketing and Promotions')!;
    expect(promoBucket).toMatchObject({ created: false, assigned: 40 });
    // A distinct Newsletters category was created.
    const active = h.categories.listActive(h.accountId).map((c) => c.label);
    expect(active).toContain('Newsletters & Digests');
    expect(active).toContain('Notifications & Alerts');

    // Security (16) and receipts (16) remain uncategorized; nothing was misfiled.
    expect(h.categories.countUncategorized(h.accountId)).toBe(32);
    // The marketing category now carries the 40 promo emails.
    expect(
      h.categories.listActive(h.accountId).find((c) => c.id === marketing.id)?.emailCount,
    ).toBe(40);
  });

  it('only files the uncategorized backlog, leaving already-categorized mail alone', () => {
    const h = seed(8);
    // Pre-assign a promo email to some other category; residual buckets must not touch it.
    const other = h.categories.create({ accountId: h.accountId, label: 'Pinned', source: 'user' });
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'm40', // first promo row (NEWSLETTERS=8*5=40 rows precede)
        accountId: h.accountId,
        categoryId: other.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: 1,
        method: null,
      },
    ]);
    const svc = new ResidualBucketService(h.categories, h.emails, silentLogger, () => true);
    const res = svc.coverResidual(h.accountId);
    // That message stays only in 'Pinned'; residual buckets assigned the other 119 matches.
    const cats = h.categories.getEmailCategories('m40', h.accountId).map((a) => a.categoryId);
    expect(cats).toEqual([other.id]);
    expect(res.totalAssigned).toBe(119);
  });
});

describe('POST /categories/cover-residual route', () => {
  async function app(enabled: boolean) {
    const Fastify = (await import('fastify')).default;
    const { registerCategoryRoutes } = await import('../src/routes/categories.js');
    const h = seed(8);
    h.categories.create({
      accountId: h.accountId,
      label: 'Marketing and Promotions',
      source: 'auto',
    });
    const residualBucket = new ResidualBucketService(
      h.categories,
      h.emails,
      silentLogger,
      () => enabled,
    );
    const ctx = {
      repos: { accounts: h.accounts, categories: h.categories, emails: h.emails },
      services: { residualBucket },
      logger: silentLogger,
    } as never;
    const server = Fastify();
    await registerCategoryRoutes(server, ctx);
    await server.ready();
    return { server, h };
  }

  it('files the tail when enabled and returns the buckets', async () => {
    const { server, h } = await app(true);
    const res = await server.inject({
      method: 'POST',
      url: '/categories/cover-residual',
      payload: { accountId: h.accountId },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().totalAssigned).toBe(120);
    await server.close();
  });

  it('is a no-op when the flag is off', async () => {
    const { server, h } = await app(false);
    const res = await server.inject({
      method: 'POST',
      url: '/categories/cover-residual',
      payload: { accountId: h.accountId },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ totalAssigned: 0, buckets: [] });
    await server.close();
  });
});
