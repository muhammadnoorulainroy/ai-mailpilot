export interface ModelPreset {
  id: 'lightweight' | 'recommended' | 'institutional' | 'maximum' | 'custom';
  label: string;
  description: string;
  embeddingModel: string;
  generationModel: string;
  minRamGb: number;
}

export interface AppConfig {
  ollamaUrl: string;
  embeddingModel: string;
  generationModel: string;
  autoIndex: boolean;
  indexedFolders: string[];
  imap: ImapConfig;
  locale: 'en' | 'fr';
}

export interface ImapConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  tls: boolean;
}

export const DEFAULT_CONFIG: AppConfig = {
  ollamaUrl: 'http://localhost:11434',
  embeddingModel: 'bge-m3',
  generationModel: 'qwen3:8b',
  autoIndex: false,
  indexedFolders: ['INBOX'],
  imap: {
    host: '',
    port: 993,
    user: '',
    password: '',
    tls: true,
  },
  locale: 'en',
};
