import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/server.ts'],
  format: ['esm'],
  dts: true,
  noExternal: ['@ai-mailpilot/shared'],
  clean: true,
  sourcemap: true,
});
