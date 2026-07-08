/**
 * Tests for the collision-safe Thunderbird tag identity (Part A). MailPilot tag keys are derived
 * from a category's stable canonicalKey (not its churny id), leftover same-label MailPilot tags are
 * adopted instead of recreated, and a user tag that happens to share a label is never touched.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { CategoryDto } from '@ai-mailpilot/shared';
import { ensureCategoryTags, tagKeyFor } from '../src/thunderbird/tags.js';

interface FakeTag {
  key: string;
  tag: string;
  color: string;
  ordinal: string;
}

/** In-memory Thunderbird tag store mirroring the create/update/list semantics we rely on. */
function fakeTagStore(initial: Array<Pick<FakeTag, 'key' | 'tag'>> = []) {
  const tags: FakeTag[] = initial.map((t) => ({ color: '#000000', ordinal: '', ...t }));
  const create = vi.fn(async (key: string, tag: string, color: string) => {
    if (tags.some((t) => t.tag.toLowerCase() === tag.toLowerCase())) {
      throw new Error(`Specified tag already exists: ${tag}`);
    }
    tags.push({ key, tag, color, ordinal: '' });
  });
  const update = vi.fn(async (key: string, patch: { tag?: string; color?: string }) => {
    const t = tags.find((x) => x.key === key);
    if (!t) throw new Error('no such tag');
    if (patch.tag !== undefined) t.tag = patch.tag;
    if (patch.color !== undefined) t.color = patch.color;
  });
  const list = vi.fn(async () => tags.map((t) => ({ ...t })));
  vi.stubGlobal('browser', { messages: { tags: { list, create, update } } });
  return { tags, create, update, list };
}

function cat(over: Partial<CategoryDto> & { id: string; label: string; canonicalKey: string }): CategoryDto {
  return {
    accountId: 'a',
    description: null,
    source: 'auto',
    emailCount: 0,
    createdAt: 0,
    updatedAt: 0,
    ...over,
  };
}

describe('ensureCategoryTags: stable key + collision handling', () => {
  beforeEach(() => vi.unstubAllGlobals());

  it('keys the tag by canonicalKey, not the category id', () => {
    expect(tagKeyFor({ id: 'abc-123', canonicalKey: 'banking' })).toBe('mailpilot_banking');
    // Fallback to id only when the canonical key is missing.
    expect(tagKeyFor({ id: 'abc-123' })).toBe('mailpilot_abc-123');
  });

  it('reuses the same tag key after the category id changes but canonicalKey stays', async () => {
    const store = fakeTagStore([{ key: 'mailpilot_banking', tag: 'Banking Transactions' }]);
    const map = await ensureCategoryTags([
      cat({ id: 'new-id-after-reset', label: 'Banking Transactions', canonicalKey: 'banking' }),
    ]);
    expect(map.get('new-id-after-reset')).toBe('mailpilot_banking');
    expect(store.create).not.toHaveBeenCalled();
  });

  it('adopts a leftover MailPilot tag with the same label under an OLD id-based key, no duplicate create', async () => {
    // Simulates the reported reset bug: old id-keyed MailPilot tag survives, new category has a new
    // canonicalKey-based key. The old tag is adopted rather than a second "Banking Transactions" created.
    const store = fakeTagStore([{ key: 'mailpilot_old-uuid', tag: 'Banking Transactions' }]);
    const map = await ensureCategoryTags([
      cat({ id: 'new-id', label: 'Banking Transactions', canonicalKey: 'banking' }),
    ]);
    expect(map.get('new-id')).toBe('mailpilot_old-uuid');
    expect(store.create).not.toHaveBeenCalled();
    expect(store.update).not.toHaveBeenCalled();
  });

  it('never hijacks a user tag with the same label; creates a disambiguated MailPilot tag', async () => {
    const store = fakeTagStore([{ key: '$work', tag: 'Banking Transactions' }]);
    const map = await ensureCategoryTags([
      cat({ id: 'c1', label: 'Banking Transactions', canonicalKey: 'banking' }),
    ]);
    expect(map.get('c1')).toBe('mailpilot_banking');
    expect(store.create).toHaveBeenCalledWith(
      'mailpilot_banking',
      'AI: Banking Transactions',
      expect.any(String),
    );
    // The user tag is left exactly as it was.
    expect(store.tags.find((t) => t.key === '$work')?.tag).toBe('Banking Transactions');
    expect(store.update).not.toHaveBeenCalled();
  });

  it('does not throw when multiple categories reuse existing same-label tags', async () => {
    fakeTagStore([
      { key: 'mailpilot_old_a', tag: 'Invoices' },
      { key: 'mailpilot_old_b', tag: 'Travel' },
    ]);
    await expect(
      ensureCategoryTags([
        cat({ id: 'x', label: 'Invoices', canonicalKey: 'invoices' }),
        cat({ id: 'y', label: 'Travel', canonicalKey: 'travel' }),
      ]),
    ).resolves.toBeInstanceOf(Map);
  });

  it('renaming a category updates only the MailPilot tag label', async () => {
    const store = fakeTagStore([{ key: 'mailpilot_banking', tag: 'Banking Transactions' }]);
    await ensureCategoryTags([
      cat({ id: 'c1', label: 'Banking & Cards', canonicalKey: 'banking' }),
    ]);
    expect(store.update).toHaveBeenCalledWith('mailpilot_banking', { tag: 'Banking & Cards' });
    expect(store.create).not.toHaveBeenCalled();
  });

  it('renaming into a label a user tag already owns falls back to a disambiguated label', async () => {
    const store = fakeTagStore([
      { key: 'mailpilot_banking', tag: 'Banking Transactions' },
      { key: '$mine', tag: 'Banking & Cards' },
    ]);
    await ensureCategoryTags([
      cat({ id: 'c1', label: 'Banking & Cards', canonicalKey: 'banking' }),
    ]);
    expect(store.update).toHaveBeenCalledWith('mailpilot_banking', { tag: 'AI: Banking & Cards' });
    expect(store.tags.find((t) => t.key === '$mine')?.tag).toBe('Banking & Cards');
  });

  it('creates a fresh tag when nothing matches by key or label', async () => {
    const store = fakeTagStore([]);
    const map = await ensureCategoryTags([
      cat({ id: 'c1', label: 'Security Alerts', canonicalKey: 'security' }),
    ]);
    expect(map.get('c1')).toBe('mailpilot_security');
    expect(store.create).toHaveBeenCalledWith(
      'mailpilot_security',
      'Security Alerts',
      expect.any(String),
    );
  });

  it('never collapses two categories: adoption skips a tag that is another category’s own key', async () => {
    // Leftover tag mailpilot_billing labeled "Bills". Category A could adopt it by label, but it is
    // category B's own derived key, so A must not steal it and the two must stay on distinct tags.
    fakeTagStore([{ key: 'mailpilot_billing', tag: 'Bills' }]);
    const map = await ensureCategoryTags([
      cat({ id: 'A', label: 'Bills', canonicalKey: 'bills' }),
      cat({ id: 'B', label: 'Invoices', canonicalKey: 'billing' }),
    ]);
    expect(map.get('B')).toBe('mailpilot_billing');
    expect(map.get('A')).not.toBe('mailpilot_billing');
    expect(new Set(map.values()).size).toBe(2);
  });

  it('handles two categories whose labels differ only in case without throwing', async () => {
    const store = fakeTagStore([]);
    const map = await ensureCategoryTags([
      cat({ id: 'A', label: 'Work', canonicalKey: 'work' }),
      cat({ id: 'B', label: 'work', canonicalKey: 'work_2' }),
    ]);
    expect(map.get('A')).toBe('mailpilot_work');
    expect(map.get('B')).toBe('mailpilot_work_2');
    expect(store.create).toHaveBeenCalledTimes(2);
    const labels = store.tags.map((t) => t.tag.toLowerCase());
    expect(labels).toContain('work');
    expect(labels).toContain('ai: work');
  });

  it('a single failing tag create does not abort tagging for the rest', async () => {
    const store = fakeTagStore([]);
    store.create.mockImplementationOnce(async () => {
      throw new Error('boom');
    });
    const map = await ensureCategoryTags([
      cat({ id: 'A', label: 'Alpha', canonicalKey: 'alpha' }),
      cat({ id: 'B', label: 'Beta', canonicalKey: 'beta' }),
    ]);
    expect(map.get('A')).toBe('mailpilot_alpha');
    expect(map.get('B')).toBe('mailpilot_beta');
    expect(store.create).toHaveBeenCalledTimes(2);
  });
});
