import { coreClient } from '../api-client/core-client.js';
import { listAccounts, listFolders, fetchEmailsFromFolder } from '../thunderbird/mailbox.js';

const SYNC_LIMIT = 50;
const PROGRESS_POLL_MS = 2000;

async function sync(): Promise<void> {
  console.log('[MailPilot] sync started');

  const tbAccounts = await listAccounts();
  if (tbAccounts.length === 0) {
    console.warn('[MailPilot] no IMAP/POP3 accounts found in Thunderbird');
    return;
  }

  const syncedAccountIds: string[] = [];

  for (const tb of tbAccounts) {
    try {
      const created = await coreClient.createAccount({
        address: tb.address,
        displayName: tb.name,
        kind: tb.kind,
      });
      console.log(`[MailPilot] account ${tb.address} -> ${created.account.id} (${tb.kind})`);

      const folders = await listFolders(tb.tbId);
      console.log(`[MailPilot] ${folders.length} folders for ${tb.address}`);

      const inbox =
        folders.find((f) => f.type === 'inbox') ??
        folders.find((f) => f.name.toLowerCase() === 'inbox');

      if (!inbox) {
        console.warn(`[MailPilot] no inbox found for ${tb.address}`);
        continue;
      }

      console.log(`[MailPilot] fetching up to ${SYNC_LIMIT} messages from "${inbox.path}"...`);
      const emails = await fetchEmailsFromFolder(tb.tbId, inbox.path, SYNC_LIMIT);
      const withBody = emails.filter((e) => e.body && e.body.length > 0).length;
      console.log(`[MailPilot] fetched ${emails.length} messages (${withBody} with body)`);

      if (emails.length === 0) continue;

      const result = await coreClient.pushEmails({
        accountId: created.account.id,
        emails,
      });
      console.log(
        `[MailPilot] pushed ${result.inserted}, total in Core: ${result.total} for ${tb.address}`,
      );

      syncedAccountIds.push(created.account.id);
    } catch (err) {
      console.error(`[MailPilot] sync failed for ${tb.address}:`, err);
    }
  }

  console.log('[MailPilot] sync complete');

  for (const accountId of syncedAccountIds) {
    await embed(accountId);
  }
}

async function embed(accountId: string): Promise<void> {
  try {
    const start = await coreClient.runEmbed({ accountId });
    console.log(
      `[MailPilot] embed: ${start.status}, ${start.pending} pending with model ${start.modelId}`,
    );

    if (start.status === 'started' || start.status === 'already_running') {
      await pollEmbedProgress();
    }
  } catch (err) {
    console.error('[MailPilot] embed failed:', err);
  }
}

async function pollEmbedProgress(): Promise<void> {
  while (true) {
    await new Promise((r) => setTimeout(r, PROGRESS_POLL_MS));
    const progress = await coreClient.embedProgress();
    console.log(
      `[MailPilot] embed progress: ${progress.processed}/${progress.total} (${progress.failed} failed) - ${progress.status}`,
    );
    if (progress.status === 'completed' || progress.status === 'error' || progress.status === 'idle') {
      return;
    }
  }
}

async function setupMenus(): Promise<void> {
  try {
    await browser.menus.create({
      id: 'categorize_email',
      title: 'Categorize with AI MailPilot',
      contexts: ['message_list'],
    });
    await browser.menus.create({
      id: 'find_similar',
      title: 'Find Similar Emails',
      contexts: ['message_list'],
    });
  } catch (err) {
    console.warn('[MailPilot] menu setup failed:', err);
  }
}

browser.menus.onClicked.addListener(async (info) => {
  if (info.menuItemId === 'categorize_email') {
    console.log('[MailPilot] categorize email:', info);
  }
  if (info.menuItemId === 'find_similar') {
    console.log('[MailPilot] find similar:', info);
  }
});

browser.browserAction.onClicked.addListener(async () => {
  try {
    await sync();
  } catch (err) {
    console.error('[MailPilot] sync failed:', err);
  }
});

async function init(): Promise<void> {
  console.log('[MailPilot] starting...');
  await coreClient.loadToken();
  await setupMenus();

  try {
    const health = await coreClient.health();
    console.log('[MailPilot] Core connected:', health);
  } catch (err) {
    console.warn('[MailPilot] Core not reachable. Start the Core Server first.', err);
  }

  const debug = globalThis as unknown as {
    mailpilotSync: typeof sync;
    mailpilotEmbed: typeof embed;
    mailpilotSetToken: (token: string) => Promise<void>;
  };
  debug.mailpilotSync = sync;
  debug.mailpilotEmbed = embed;
  debug.mailpilotSetToken = async (token: string) => {
    await coreClient.setToken(token);
    console.log('[MailPilot] token stored and loaded in-memory');
  };
  console.log(
    '[MailPilot] debug helpers: mailpilotSetToken("<token>"), mailpilotSync(), mailpilotEmbed("<accountId>")',
  );
}

init();
