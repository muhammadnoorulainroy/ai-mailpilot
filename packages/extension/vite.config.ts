/**
 * Vite build configuration for the Thunderbird extension. Bundles the
 * background, settings, dashboard, and message-assistant entry points and
 * copies static extension files into the dist output.
 */
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
        settings: resolve(__dirname, 'src/ui/settings/settings.ts'),
        dashboard: resolve(__dirname, 'src/ui/dashboard/dashboard.ts'),
        'message-assistant': resolve(
          __dirname,
          'src/ui/message-assistant/message-assistant.ts',
        ),
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
      /**
       * After the bundle is written, copies the manifest, locales, assets,
       * shared UI sources, and each page's HTML and CSS into dist.
       */
      closeBundle() {
        const dist = resolve(__dirname, 'dist');

        copyFileSync(resolve(__dirname, 'manifest.json'), resolve(dist, 'manifest.json'));

        const localesDir = resolve(__dirname, '_locales');
        if (existsSync(localesDir)) {
          cpSync(localesDir, resolve(dist, '_locales'), { recursive: true });
        }

        const assetsDir = resolve(__dirname, 'assets');
        if (existsSync(assetsDir)) {
          cpSync(assetsDir, resolve(dist, 'assets'), { recursive: true });
        }

        const sharedSrc = resolve(__dirname, 'src/ui/shared');
        const sharedOut = resolve(dist, 'shared');
        if (!existsSync(sharedOut)) mkdirSync(sharedOut, { recursive: true });
        if (existsSync(sharedSrc)) {
          cpSync(sharedSrc, sharedOut, { recursive: true });
        }

        const settingsDir = resolve(dist, 'settings');
        if (!existsSync(settingsDir)) mkdirSync(settingsDir, { recursive: true });
        copyFileSync(
          resolve(__dirname, 'src/ui/settings/settings.html'),
          resolve(settingsDir, 'settings.html'),
        );
        copyFileSync(
          resolve(__dirname, 'src/ui/settings/settings.css'),
          resolve(settingsDir, 'settings.css'),
        );

        const dashboardDir = resolve(dist, 'dashboard');
        if (!existsSync(dashboardDir)) mkdirSync(dashboardDir, { recursive: true });
        copyFileSync(
          resolve(__dirname, 'src/ui/dashboard/dashboard.html'),
          resolve(dashboardDir, 'dashboard.html'),
        );
        copyFileSync(
          resolve(__dirname, 'src/ui/dashboard/dashboard.css'),
          resolve(dashboardDir, 'dashboard.css'),
        );

        const assistantDir = resolve(dist, 'message-assistant');
        if (!existsSync(assistantDir)) mkdirSync(assistantDir, { recursive: true });
        copyFileSync(
          resolve(__dirname, 'src/ui/message-assistant/message-assistant.html'),
          resolve(assistantDir, 'message-assistant.html'),
        );
        copyFileSync(
          resolve(__dirname, 'src/ui/message-assistant/message-assistant.css'),
          resolve(assistantDir, 'message-assistant.css'),
        );
      },
    },
  ],
});
