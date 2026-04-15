import { defineConfig } from 'vite';
import { resolve } from 'path';
import { copyFileSync, mkdirSync, existsSync, cpSync } from 'fs';

export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        background: resolve(__dirname, 'src/background/background.ts'),
        sidebar: resolve(__dirname, 'src/ui/sidebar/sidebar.ts'),
      },
      output: {
        entryFileNames: '[name]/[name].js',
        chunkFileNames: 'chunks/[name].js',
        assetFileNames: 'assets/[name].[ext]',
      },
    },
    target: 'es2022',
    minify: false,
    sourcemap: true,
  },
  plugins: [
    {
      name: 'copy-extension-files',
      closeBundle() {
        const dist = resolve(__dirname, 'dist');

        // Copy manifest.json
        copyFileSync(resolve(__dirname, 'manifest.json'), resolve(dist, 'manifest.json'));

        // Copy _locales
        const localesDir = resolve(__dirname, '_locales');
        if (existsSync(localesDir)) {
          cpSync(localesDir, resolve(dist, '_locales'), { recursive: true });
        }

        // Copy assets
        const assetsDir = resolve(__dirname, 'assets');
        if (existsSync(assetsDir)) {
          cpSync(assetsDir, resolve(dist, 'assets'), { recursive: true });
        }

        // Copy HTML files next to their JS bundles
        const sidebarDir = resolve(dist, 'sidebar');
        if (!existsSync(sidebarDir)) mkdirSync(sidebarDir, { recursive: true });
        copyFileSync(
          resolve(__dirname, 'src/ui/sidebar/sidebar.html'),
          resolve(sidebarDir, 'sidebar.html'),
        );
        copyFileSync(
          resolve(__dirname, 'src/ui/sidebar/sidebar.css'),
          resolve(sidebarDir, 'sidebar.css'),
        );
      },
    },
  ],
});
