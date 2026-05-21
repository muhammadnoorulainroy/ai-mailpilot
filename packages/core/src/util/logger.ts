/**
 * Provides the shared pino logger used across the core package, writing to the
 * log file and, outside production, to stdout.
 */
import pino, { type Logger } from 'pino';
import { LOG_PATH, ensureDirs } from './paths.js';

let _logger: Logger | null = null;

/**
 * Returns the shared pino logger, creating it on first call. Writes to the log
 * file and also to stdout outside production.
 */
export function getLogger(): Logger {
  if (_logger) return _logger;

  ensureDirs();

  const isDev = process.env.NODE_ENV !== 'production';

  _logger = pino(
    {
      level: process.env.LOG_LEVEL ?? (isDev ? 'debug' : 'info'),
      base: undefined,
      timestamp: pino.stdTimeFunctions.isoTime,
    },
    pino.multistream([
      { stream: pino.destination({ dest: LOG_PATH, mkdir: true, sync: false }) },
      ...(isDev ? [{ stream: process.stdout }] : []),
    ]),
  );

  return _logger;
}
