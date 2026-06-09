/**
 * Vitest configuration for the core package, running tests in a single worker thread.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    pool: 'threads',
    poolOptions: {
      threads: { singleThread: true },
    },
  },
});
