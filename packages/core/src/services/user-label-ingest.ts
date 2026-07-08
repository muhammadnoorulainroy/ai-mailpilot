/**
 * Turns a pushed email's folder and Thunderbird tags into user-owned labels. Generic system folders
 * are dropped (they carry no organizing intent) and MailPilot-managed tags are never treated as user
 * tags. Pure helpers so ingestion, stats, and discovery hints all normalize the same way.
 */
import type { UserLabelInput } from '../repositories/email-user-label-repository.js';

/** Prefix marking a Thunderbird tag as MailPilot-managed, never a user-owned signal. */
export const MAILPILOT_TAG_PREFIX = 'mailpilot_';

/**
 * Folder leaf names with no organizing intent, excluded from folder hints and storage. Matched on the
 * folder's last path segment, case-insensitively.
 */
export const GENERIC_FOLDERS = new Set([
  'inbox',
  'archive',
  'archives',
  'all mail',
  'allmail',
  'sent',
  'sent items',
  'trash',
  'deleted',
  'deleted items',
  'junk',
  'spam',
  'bulk mail',
  'drafts',
  'templates',
  'outbox',
]);

/** The last path segment of a folder path (its display name). */
export function folderLeaf(path: string): string {
  const parts = path.split('/').filter((p) => p.length > 0);
  return parts.length > 0 ? parts[parts.length - 1]! : path;
}

/** Whether a folder path is a generic system folder that should not be hinted or stored. */
export function isGenericFolder(path: string): boolean {
  return GENERIC_FOLDERS.has(folderLeaf(path).trim().toLowerCase());
}

/** A stable, safe key for a folder path (full path, normalized), or null for a generic folder. */
export function folderLabel(path: string): { key: string; label: string } | null {
  if (isGenericFolder(path)) return null;
  const key =
    path
      .replace(/^\/+/, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '') || 'folder';
  return { key, label: folderLeaf(path) };
}

/**
 * Build the user-owned labels for one email from its folder and Thunderbird tags. Drops
 * MailPilot-managed tags, empty values, and the generic-folder label.
 */
export function buildUserLabels(
  folder: string,
  tags: Array<{ key: string; label: string }> | undefined,
): UserLabelInput[] {
  const out: UserLabelInput[] = [];
  for (const tag of tags ?? []) {
    if (!tag.key || tag.key.startsWith(MAILPILOT_TAG_PREFIX)) continue;
    if (!tag.label) continue;
    out.push({ source: 'thunderbird_tag', key: tag.key, label: tag.label });
  }
  const folderInfo = folderLabel(folder);
  if (folderInfo) out.push({ source: 'folder', key: folderInfo.key, label: folderInfo.label });
  return out;
}
