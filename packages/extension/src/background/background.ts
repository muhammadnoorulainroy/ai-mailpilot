import { CORE_SERVER_URL, API_PREFIX } from '@ai-mailpilot/shared';
import type { HealthResponse } from '@ai-mailpilot/shared';

async function checkCoreServer(): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${CORE_SERVER_URL}${API_PREFIX}/health`);
    return response.ok ? (await response.json() as HealthResponse) : null;
  } catch {
    return null;
  }
}

browser.menus.onClicked.addListener(async (info) => {
  if (info.menuItemId === 'categorize_email') {
    // TODO: categorization flow
    console.log('Categorize email:', info);
  }
  if (info.menuItemId === 'find_similar') {
    // TODO: find similar flow
    console.log('Find similar:', info);
  }
});

browser.browserAction.onClicked.addListener(async () => {
  // TODO: toggle sidebar
  console.log('Toolbar button clicked');
});

async function init() {
  console.log('AI MailPilot starting...');
  const health = await checkCoreServer();
  if (health) {
    console.log('Core Server connected:', health);
  } else {
    console.warn('Core Server not reachable. Start AI MailPilot Core first.');
  }
}

init();
