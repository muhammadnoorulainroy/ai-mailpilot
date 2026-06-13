/**
 * Sync preferences for the extension: loading, saving, and the rules that decide
 * which accounts and folders are synced to Core based on account kind and user overrides.
 */
import type { AccountKind } from '@ai-mailpilot/shared';

const STORAGE_KEY = 'sync_prefs_v1';

/** Per-account and per-folder preferences governing which mail is synced to Core. */
export interface SyncPrefs {
  /** Account addresses explicitly enabled for sync to Core. */
  enabledAddresses: string[];
  /** Account addresses explicitly excluded, overriding the default. */
  excludedAddresses: string[];
  /** Folder paths to exclude from sync, account-agnostic. Default excludes common personal and system folders. */
  excludedFolderPaths: string[];
  /** After "Organize inbox", write each category as a Thunderbird tag on its emails. */
  applyTags: boolean;
}

const DEFAULTS: SyncPrefs = {
  enabledAddresses: [],
  excludedAddresses: [],
  excludedFolderPaths: ['Trash', 'Junk', 'Drafts', 'Spam', 'Outbox'],
  applyTags: true,
};

/** Load sync preferences from local storage, falling back to defaults for missing or invalid fields. */
export async function loadSyncPrefs(): Promise<SyncPrefs> {
  const stored = (await browser.storage.local.get(STORAGE_KEY)) as Record<string, unknown>;
  const raw = stored[STORAGE_KEY];
  if (!raw || typeof raw !== 'object') return { ...DEFAULTS };
  const obj = raw as Partial<SyncPrefs>;
  return {
    enabledAddresses: Array.isArray(obj.enabledAddresses)
      ? obj.enabledAddresses
      : DEFAULTS.enabledAddresses,
    excludedAddresses: Array.isArray(obj.excludedAddresses)
      ? obj.excludedAddresses
      : DEFAULTS.excludedAddresses,
    excludedFolderPaths: Array.isArray(obj.excludedFolderPaths)
      ? obj.excludedFolderPaths
      : DEFAULTS.excludedFolderPaths,
    applyTags: typeof obj.applyTags === 'boolean' ? obj.applyTags : DEFAULTS.applyTags,
  };
}

/** Persist sync preferences to local storage. */
export async function saveSyncPrefs(prefs: SyncPrefs): Promise<void> {
  await browser.storage.local.set({ [STORAGE_KEY]: prefs });
}

/**
 * Decide whether to sync an account. Personal accounts are excluded unless
 * explicitly enabled, work and institutional accounts are enabled unless
 * explicitly excluded. enabledAddresses always takes precedence over the kind heuristic.
 */
export function shouldSyncAccount(
  address: string,
  kind: AccountKind,
  prefs: SyncPrefs,
): { sync: boolean; reason: string } {
  if (prefs.excludedAddresses.includes(address)) {
    return { sync: false, reason: 'explicitly excluded by user' };
  }
  if (prefs.enabledAddresses.includes(address)) {
    return { sync: true, reason: 'explicitly enabled by user' };
  }
  if (kind === 'personal') {
    return {
      sync: false,
      reason: 'personal account excluded by default (enable in settings to sync)',
    };
  }
  return { sync: true, reason: `default for ${kind} account` };
}

/**
 * Return true when any case-insensitive path segment equals an excluded name.
 * Segment equality avoids over-matching folders like "Draft Contracts" or
 * "Junk Mail Receipts" that merely contain an excluded word.
 */
export function pathHasExcludedSegment(folderPath: string, excludedNames: string[]): boolean {
  if (excludedNames.length === 0) return false;
  const excluded = new Set(
    excludedNames.map((e) => e.trim().toLowerCase()).filter((e) => e.length > 0),
  );
  if (excluded.size === 0) return false;
  return folderPath
    .toLowerCase()
    .split('/')
    .some((seg) => excluded.has(seg.trim()));
}

/** Return true when the folder path matches any excluded folder name in the preferences. */
export function isFolderExcluded(folderPath: string, prefs: SyncPrefs): boolean {
  return pathHasExcludedSegment(folderPath, prefs.excludedFolderPaths);
}
