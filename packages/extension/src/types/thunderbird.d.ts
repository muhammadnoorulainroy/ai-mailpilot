// Minimal Thunderbird WebExtension API type stubs
// Full types: https://webextension-api.thunderbird.net

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
  parts?: TbMessagePart[];
}

declare namespace browser {
  namespace menus {
    function create(properties: {
      id?: string;
      title?: string;
      contexts?: string[];
      type?: 'normal' | 'separator' | 'checkbox' | 'radio';
      enabled?: boolean;
    }): Promise<void>;

    function remove(menuItemId: string): Promise<void>;
    function removeAll(): Promise<void>;

    const onClicked: {
      addListener(callback: (info: { menuItemId: string }) => void): void;
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
    function getFull(messageId: number): Promise<TbMessagePart>;
    function update(messageId: number, properties: Record<string, unknown>): Promise<void>;
    function move(messageIds: number[], destination: TbFolder): Promise<void>;
  }
  namespace accounts {
    function list(includeFolders?: boolean): Promise<TbAccount[]>;
  }
}
