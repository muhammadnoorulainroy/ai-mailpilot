/**
 * Vitest configuration for the extension package, limiting test discovery to
 * files matching the test directory pattern.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
  },
});
