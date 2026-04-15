// Minimal Thunderbird WebExtension API type stubs
// Full types: https://webextension-api.thunderbird.net

declare namespace browser {
  namespace menus {
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
    function getFull(messageId: number): Promise<unknown>;
    function list(folderId: unknown): Promise<{ messages: unknown[] }>;
    function update(messageId: number, properties: Record<string, unknown>): Promise<void>;
    function move(messageIds: number[], folderId: unknown): Promise<void>;
  }
  namespace accounts {
    function list(): Promise<unknown[]>;
  }
}
