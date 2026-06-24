/**
 * tsup build configuration for the core package, bundling the server entry to ESM
 * with type declarations and inlining the shared workspace package.
 */
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/server.ts'],
  format: ['esm'],
  dts: true,
  noExternal: ['@ai-mailpilot/shared'],
  clean: true,
  sourcemap: true,
});
