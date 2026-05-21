/**
 * Resolves platform-specific data, config, log, and cache directories for the app
 * and exposes the derived file paths used across the core package.
 */
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
export const DB_KEY_PATH = join(CONFIG_DIR, 'db.key');

/**
 * Creates the data, config, log, and cache directories with owner-only access.
 */
export function ensureDirs(): void {
  for (const dir of [DATA_DIR, CONFIG_DIR, LOG_DIR, CACHE_DIR]) {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
  }
}
