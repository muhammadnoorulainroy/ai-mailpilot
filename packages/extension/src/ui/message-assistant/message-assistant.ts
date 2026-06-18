/**
 * Popup controller for the message assistant pane. Resolves the displayed
 * email to a synced core account, then drives summary and reply-draft
 * generation while persisting per-message draft state and gating cloud use.
 */
import { coreClient } from '../../api-client/core-client.js';
import type { EmailAssistantSummaryDto } from '@ai-mailpilot/shared';

interface TargetEmail {
  tbMessageId: number;
  messageId: string;
  accountId: string;
  subject: string | null;
  fromAddr: string | null;
}

interface AssistantPopupState {
  draftPrompt?: string;
  draftOutput?: string;
  updatedAt?: number;
}

/** Get an element by id, throwing if it is missing so callers can assume non-null. */
const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

let target: TargetEmail | null = null;
let cloudProvider: string | null = null;
let stateKey: string | null = null;

/** Derive a short human label for a cloud provider from its API base URL. */
function providerLabelFor(baseUrl: string | null | undefined): string | null {
  if (!baseUrl) return null;
  try {
    const host = new URL(baseUrl).hostname.replace(/^api\./, '');
    if (host.includes('openai.com')) return 'OpenAI';
    return host;
  } catch {
    return baseUrl;
  }
}

/** Build the storage key that scopes persisted draft state to account, message, and provider. */
function makeStateKey(t: TargetEmail, provider: string | null): string {
  return `mailpilot:assistant:${t.accountId}:${t.messageId}:${provider ?? 'local'}`;
}

/** Coerce an untrusted stored value into a validated popup state object. */
function asPopupState(value: unknown): AssistantPopupState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const rec = value as Record<string, unknown>;
  return {
    draftPrompt: typeof rec.draftPrompt === 'string' ? rec.draftPrompt : undefined,
    draftOutput: typeof rec.draftOutput === 'string' ? rec.draftOutput : undefined,
    updatedAt: typeof rec.updatedAt === 'number' ? rec.updatedAt : undefined,
  };
}

/** Load the persisted draft state for the current message from local storage. */
async function loadPopupState(): Promise<AssistantPopupState> {
  if (!stateKey) return {};
  const stored = (await browser.storage.local.get(stateKey)) as Record<string, unknown>;
  return asPopupState(stored[stateKey]);
}

/** Merge a patch into the persisted draft state and stamp the update time. */
async function savePopupState(patch: AssistantPopupState): Promise<void> {
  if (!stateKey) return;
  const existing = await loadPopupState();
  await browser.storage.local.set({
    [stateKey]: { ...existing, ...patch, updatedAt: Date.now() },
  });
}

/** Repopulate the prompt and draft fields from persisted state, revealing the draft panel if there is output. */
function restoreDraftState(state: AssistantPopupState): void {
  if (state.draftPrompt !== undefined) {
    $<HTMLTextAreaElement>('draft-prompt').value = state.draftPrompt;
  }
  if (state.draftOutput?.trim()) {
    $('draft-panel').hidden = false;
    $('draft-output-wrap').hidden = false;
    $<HTMLTextAreaElement>('draft-output').value = state.draftOutput;
  }
}

/** Show a status message in the status box, styled by the given kind. */
function setStatus(message: string, kind: 'plain' | 'error' | 'success' = 'plain'): void {
  const box = $('status');
  box.hidden = false;
  box.className = `status-box${kind === 'error' ? ' error' : kind === 'success' ? ' success' : ''}`;
  box.textContent = message;
}

/** Return the header of the email shown in the active tab, throwing if none is open or it lacks a Message-ID. */
async function currentDisplayedMessage(): Promise<TbMessageHeader> {
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  const tabId = tabs[0]?.id;
  if (tabId === undefined) throw new Error('Open an email first.');
  const msg = (await browser.messageDisplay.getDisplayedMessages(tabId))[0] ?? null;
  if (!msg) throw new Error('Open an email first.');
  if (!msg.headerMessageId) throw new Error('This email has no Message-ID, so it is not indexed.');
  return msg;
}

/** Resolve the displayed email and its Thunderbird account to a synced core account, producing the target descriptor. */
async function resolveTarget(): Promise<TargetEmail> {
  const msg = await currentDisplayedMessage();
  const tbAccount = await browser.accounts.get(msg.folder.accountId, false);
  const address = tbAccount?.identities?.[0]?.email ?? tbAccount?.name ?? '';
  if (!address) throw new Error('Could not resolve the Thunderbird account for this email.');

  const coreAccounts = (await coreClient.listAccounts()).accounts;
  const core = coreAccounts.find((a) => a.address.toLowerCase() === address.toLowerCase());
  if (!core) throw new Error('Sync this account with AI MailPilot first.');

  return {
    tbMessageId: msg.id,
    messageId: msg.headerMessageId,
    accountId: core.id,
    subject: msg.subject || null,
    fromAddr: msg.author || null,
  };
}

/** Update the provider badge to reflect whether processing is cloud or local. */
function renderProviderBadge(): void {
  const badge = $('provider-badge');
  if (cloudProvider) {
    badge.textContent = `Cloud: ${cloudProvider}`;
    badge.className = 'provider cloud';
    badge.title = `Email content is sent to ${cloudProvider} when you continue.`;
  } else {
    badge.textContent = 'Local';
    badge.className = 'provider local';
    badge.title = 'Email content stays on this machine.';
  }
}

/** Render a generated summary into the panel, including key points, suggested action, and attachment chips. */
function renderSummary(summary: EmailAssistantSummaryDto): void {
  $('summary-panel').hidden = false;
  $('draft-panel').hidden = false;
  $('summary-text').textContent = summary.summary;
  $('action-state').textContent = summary.actionRequired ? 'Yes' : 'No';
  $('reply-state').textContent = summary.needsReply ? 'Yes' : 'No';
  $('deadline-state').textContent = summary.deadline ?? 'None';

  const pointsWrap = $('key-points-wrap');
  const points = $('key-points');
  points.innerHTML = '';
  if (summary.keyPoints.length > 0) {
    pointsWrap.hidden = false;
    for (const point of summary.keyPoints) {
      const li = document.createElement('li');
      li.textContent = point;
      points.appendChild(li);
    }
  } else {
    pointsWrap.hidden = true;
  }

  const actionWrap = $('suggested-action-wrap');
  if (summary.suggestedAction) {
    actionWrap.hidden = false;
    $('suggested-action').textContent = summary.suggestedAction;
  } else {
    actionWrap.hidden = true;
  }

  const attachmentWrap = $('attachment-wrap');
  const attachmentList = $('attachment-list');
  attachmentList.innerHTML = '';
  if (summary.hasAttachments || summary.attachments.length > 0 || summary.attachmentSummary) {
    attachmentWrap.hidden = false;
    $('attachment-summary').textContent =
      summary.attachmentSummary ??
      (summary.attachments.some((a) => a.included)
        ? 'Extracted attachment text was included.'
        : 'Attachments are present, but no extracted text is available yet.');
    for (const att of summary.attachments) {
      const chip = document.createElement('span');
      chip.className = `attachment-chip${att.included ? ' included' : ''}`;
      chip.textContent = `${att.filename} · ${att.included ? 'included' : att.status}`;
      attachmentList.appendChild(chip);
    }
  } else {
    attachmentWrap.hidden = true;
  }
}

/** Request a summary for the target email and render it, optionally forcing a regenerate past the cache. */