// Runs every browser check in this directory against the running app
// (start it first with ./start.sh). Usage: node e2e/run-all.mjs
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const skip = new Set(['browser.mjs', 'shot.mjs', 'run-all.mjs']);
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.mjs') && !skip.has(f)).sort();

let failed = 0;
let total = 0;
for (const f of files) {
  const r = spawnSync(process.execPath, [path.join(dir, f)], { encoding: 'utf8' });
  const lines = (r.stdout || '').split('\n');
  const ok = lines.filter((l) => l.startsWith('ok ')).length;
  const bad = lines.filter((l) => l.startsWith('FAIL'));
  total += ok + bad.length;
  if (r.status !== 0) failed++;
  console.log(`${r.status === 0 ? 'pass' : 'FAIL'}  ${f.padEnd(14)} ${ok} checks passed${bad.length ? `, ${bad.length} failed` : ''}`);
  for (const b of bad) console.log('        ' + b);
  if (r.status !== 0 && !bad.length) console.log('        ' + ((r.stderr || r.stdout || '').trim().split('\n').slice(-4).join('\n        ')));
}
console.log(`\n${files.length - failed} of ${files.length} scripts passed, ${total} checks in all`);
process.exit(failed ? 1 : 0);
