/**
 * Minimal Thunderbird WebExtension API type stubs.
 * Full types at webextension-api.thunderbird.net.
 */

interface TbAccount {
  id: string;
  name: string;
  type: string;
  folders?: TbFolder[];
  identities?: Array<{ email: string; name?: string }>;
}

interface TbFolder {
  accountId: string;
  path: string;
  name: string;
  type?: string;
  subFolders?: TbFolder[];
}

interface TbMessageHeader {
  id: number;
  date: Date;
  author: string;
  recipients: string[];
  subject: string;
  read: boolean;
  flagged: boolean;
  folder: TbFolder;
  headerMessageId: string;
  tags: string[];
}

interface TbMessageList {
  id: string | null;
  messages: TbMessageHeader[];
}

interface TbMessagePart {
  contentType?: string;
  body?: string;
  headers?: Record<string, string[]>;
  partName?: string;
  /** Filename set by Thunderbird for attachment and inline named parts. */
  name?: string;
  parts?: TbMessagePart[];
}

interface TbMenuInfo {
  menuItemId: string | number;
  contexts?: string[];
  menuIds?: Array<string | number>;
  selectedMessages?: TbMessageList;
}

declare namespace browser {
  namespace menus {
    function create(properties: {
      id?: string;
      parentId?: string;
      title?: string;
      contexts?: string[];
      type?: 'normal' | 'separator' | 'checkbox' | 'radio';
      enabled?: boolean;
      visible?: boolean;
    }): Promise<void>;

    function update(
      id: string,
      properties: { title?: string; enabled?: boolean; visible?: boolean },
    ): Promise<void>;
    function remove(menuItemId: string): Promise<void>;
    function removeAll(): Promise<void>;
    function refresh(): Promise<void>;

    const onClicked: {
      addListener(callback: (info: TbMenuInfo, tab?: unknown) => void): void;
    };
    const onShown: {
      addListener(callback: (info: TbMenuInfo, tab?: unknown) => void): void;
    };
  }
  namespace browserAction {
    const onClicked: {
      addListener(callback: () => void): void;
    };
  }
  namespace storage {
    namespace local {
      function get(keys?: string | string[]): Promise<Record<string, unknown>>;
      function set(items: Record<string, unknown>): Promise<void>;
    }
  }
  namespace messages {
    function list(folder: TbFolder): Promise<TbMessageList>;
    function continueList(messageListId: string): Promise<TbMessageList>;
    function get(messageId: number): Promise<TbMessageHeader>;
    function getFull(messageId: number): Promise<TbMessagePart>;
    function update(messageId: number, properties: Record<string, unknown>): Promise<void>;
    function move(messageIds: number[], destination: TbFolder): Promise<void>;
    function query(queryInfo: {
      tags?: { mode: 'all' | 'any'; tags: Record<string, boolean> };
      accountId?: string;
    }): Promise<TbMessageList>;

    /** Fires when new messages are received into a folder. */
    const onNewMailReceived: {
      addListener(callback: (folder: TbFolder, messages: TbMessageList) => void): void;
      removeListener(callback: (folder: TbFolder, messages: TbMessageList) => void): void;
    };

    namespace tags {
      function list(): Promise<TbTag[]>;
      /** Thunderbird tags.create takes three positional args (key, name, hex color), not an object. */
      function create(key: string, tag: string, color: string): Promise<void>;
      function update(key: string, tag: { tag?: string; color?: string }): Promise<void>;
    }
  }

  namespace messageDisplay {
    function getDisplayedMessages(tabId?: number): Promise<TbMessageHeader[]>;
    const onMessagesDisplayed: {
      addListener(callback: (tab: { id?: number }, messages: TbMessageList) => void): void;
      removeListener(callback: (tab: { id?: number }, messages: TbMessageList) => void): void;
    };
  }

  namespace compose {
    type ReplyType = 'replyToSender' | 'replyToList' | 'replyToAll';
    interface ComposeDetails {
      body?: string;
      plainTextBody?: string;
      isPlainText?: boolean;
    }
    function beginReply(
      messageId: number,
      replyType?: ReplyType,
      details?: ComposeDetails,
    ): Promise<unknown>;
  }

  interface TbTag {
    key: string;
    tag: string;
    color: string;
    ordinal: string;
  }
  namespace accounts {
    function list(includeFolders?: boolean): Promise<TbAccount[]>;
    function get(accountId: string, includeFolders?: boolean): Promise<TbAccount | null>;
  }
  namespace folders {
    /** TB 115 parent is a MailFolder or MailAccount, returns the created MailFolder. */
    function create(parent: TbAccount | TbFolder, childName: string): Promise<TbFolder>;
    function getSubFolders(
      parent: TbAccount | TbFolder,
      includeSubFolders?: boolean,
    ): Promise<TbFolder[]>;
    function getFolderInfo(folder: TbFolder): Promise<{
      totalMessageCount: number;
      unreadMessageCount: number;
      favorite: boolean;
    }>;
  }
  namespace tabs {
    function create(properties: { url: string; active?: boolean }): Promise<{ id?: number }>;
    function query(properties: {
      active?: boolean;
      currentWindow?: boolean;
    }): Promise<Array<{ id?: number }>>;
  }
  namespace runtime {
    function sendMessage(message: unknown): Promise<unknown>;
    function openOptionsPage(): Promise<void>;
    function getURL(path: string): string;
    const onMessage: {
      addListener(
        callback: (
          message: unknown,
          sender: unknown,
          sendResponse: (response: unknown) => void,
        ) => boolean | Promise<unknown> | undefined | void,
      ): void;
    };
  }
}
