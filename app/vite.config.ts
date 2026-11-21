import fs from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

const MISSING = '\0pmpp-missing:';
const EXTS = ['', '.tsx', '.ts', '.jsx', '.js', '/index.tsx', '/index.ts'];

/**
 * Dev server only: chapters are written while the app runs, so a chapter's index may
 * name a section file that does not exist yet. Without this, Vite refuses the whole app.
 * With it, that one import becomes a module that throws "still being written", so the
 * section shows a load error and everything else keeps working. When the file appears,
 * the importers are invalidated and the page reloads. Production builds and tsc still
 * report the missing file.
 */
function missingSections(): Plugin {
  const chapters = path.resolve(__dirname, 'src/chapters');
  return {
    name: 'pmpp-missing-sections',
    apply: 'serve',
    enforce: 'pre',
    resolveId(source, importer) {
      if (!importer || !source.startsWith('.')) return null;
      const from = importer.split('?')[0];
      if (!from.startsWith(chapters)) return null;
      const target = path.resolve(path.dirname(from), source);
      if (EXTS.some((e) => fs.existsSync(target + e) && fs.statSync(target + e).isFile())) return null;
      return MISSING + target;
    },
    load(id) {
      if (!id.startsWith(MISSING)) return null;
      const rel = path.relative(chapters, id.slice(MISSING.length));
      return `throw new Error(${JSON.stringify(`${rel} is still being written; reload once it exists`)});`;
    },
    configureServer(server) {
      server.watcher.on('add', (file) => {
        if (!file.startsWith(chapters)) return;
        const stem = file.replace(/\.(tsx|ts|jsx|js)$/, '').replace(/\/index$/, '');
        let hit = false;
        for (const mod of server.moduleGraph.idToModuleMap.values()) {
          if (!mod.id?.startsWith(MISSING) || mod.id.slice(MISSING.length) !== stem) continue;
          hit = true;
          for (const importer of mod.importers) server.moduleGraph.invalidateModule(importer);
          server.moduleGraph.invalidateModule(mod);
        }
        if (hit) server.ws.send({ type: 'full-reload' });
      });
    },
  };
}

export default defineConfig({
  plugins: [missingSections(), react()],
  // Several sim tests run the course emulators (ASan builds, three thread orders). Alone
  // they take seconds; with every test file running at once they can pass vitest's 5 s default.
  test: { environment: 'node', include: ['src/**/*.test.ts'], testTimeout: 60_000 },
});
