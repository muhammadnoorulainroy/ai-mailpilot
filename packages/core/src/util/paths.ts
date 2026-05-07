import envPaths from 'env-paths';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const paths = envPaths('ai-mailpilot', { suffix: '' });

export const DATA_DIR = paths.data;
export const CONFIG_DIR = paths.config;
export const LOG_DIR = paths.log;
export const CACHE_DIR = paths.cache;

export const DB_PATH = join(DATA_DIR, 'mailpilot.db');
export const CONFIG_PATH = join(CONFIG_DIR, 'config.json');
export const LOG_PATH = join(LOG_DIR, 'core.log');

export function ensureDirs(): void {
  for (const dir of [DATA_DIR, CONFIG_DIR, LOG_DIR, CACHE_DIR]) {
    mkdirSync(dir, { recursive: true });
  }
}
