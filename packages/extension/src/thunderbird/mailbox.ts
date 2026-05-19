import type { PushEmailItem, AccountKind } from '@ai-mailpilot/shared';

export interface MailboxAccount {
  tbId: string;
  name: string;
  type: string;
  address: string;
  kind: AccountKind;
}

export async function listAccounts(): Promise<MailboxAccount[]> {
  const tbAccounts = await browser.accounts.list(true);
  return tbAccounts
    .filter((a) => a.type === 'imap' || a.type === 'pop3')
    .map((a) => {
      const identity = a.identities?.[0];
      const address = identity?.email ?? a.name;
      return {
        tbId: a.id,
        name: a.name,
        type: a.type,
        address,
        kind: guessKind(address),
      };
    });
}

function guessKind(address: string): AccountKind {
  const domain = address.split('@')[1]?.toLowerCase() ?? '';
  if (/\b(edu|ac\.|mines-|univ-|cnrs|inria)\b/.test(domain)) return 'institutional';
  if (/\b(gmail|yahoo|hotmail|outlook|protonmail|icloud)\b/.test(domain)) return 'personal';
  return 'work';
}

export interface MailboxFolder {
  path: string;
  name: string;
  type?: string;
}

export async function listFolders(accountTbId: string): Promise<MailboxFolder[]> {
  const accounts = await browser.accounts.list(true);
  const account = accounts.find((a) => a.id === accountTbId);
  if (!account?.folders) return [];

  const result: MailboxFolder[] = [];
  function walk(folders: typeof account.folders): void {
    if (!folders) return;
    for (const f of folders) {
      result.push({ path: f.path, name: f.name, type: f.type });
      if (f.subFolders) walk(f.subFolders);
    }
  }
  walk(account.folders);
  return result;
}

export async function fetchEmailsFromFolder(
  accountTbId: string,
  folderPath: string,
  maxMessages = 100,
): Promise<PushEmailItem[]> {
  const accounts = await browser.accounts.list(true);
  const account = accounts.find((a) => a.id === accountTbId);
  if (!account?.folders) return [];

  const folder = findFolder(account.folders, folderPath);
  if (!folder) return [];

  const items: PushEmailItem[] = [];
  let page = await browser.messages.list(folder);

  while (items.length < maxMessages) {
    for (const m of page.messages) {
      if (items.length >= maxMessages) break;
      items.push({
        messageId: m.headerMessageId,
        folder: folderPath,
        subject: m.subject,
        fromAddr: m.author,
        date: m.date.getTime(),
        hasAttachments: false,
      });
    }
    if (!page.id) break;
    page = await browser.messages.continueList(page.id);
    if (page.messages.length === 0) break;
  }

  return items;
}

function findFolder(folders: Array<{ path: string; subFolders?: unknown }>, path: string): unknown {
  for (const f of folders) {
    if (f.path === path) return f;
    if (f.subFolders) {
      const found = findFolder(f.subFolders as typeof folders, path);
      if (found) return found;
    }
  }
  return null;
}
