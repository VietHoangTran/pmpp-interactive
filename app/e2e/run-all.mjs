// Runs every browser check in this directory against the running app
// (start it first with ./start.sh). Usage: node e2e/run-all.mjs [prefix ...]
// With prefixes (ch05 ch06-labs app-review) only the scripts whose names start with one of
// them run. A script that runs longer than PMPP_E2E_TIMEOUT seconds (default 1200) is stopped
// and counted as failed, so one hung check cannot stall the whole run.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const skip = new Set(['browser.mjs', 'shot.mjs', 'run-all.mjs']);
const prefixes = process.argv.slice(2);
const files = fs
  .readdirSync(dir)
  .filter((f) => f.endsWith('.mjs') && !skip.has(f) && (!prefixes.length || prefixes.some((p) => f.startsWith(p))))
  .sort();
const timeout = Number(process.env.PMPP_E2E_TIMEOUT || 1200) * 1000;

let failed = 0;
let total = 0;
for (const f of files) {
  const r = spawnSync(process.execPath, [path.join(dir, f)], { encoding: 'utf8', timeout, maxBuffer: 64 << 20 });
  if (r.error) r.stderr = `${r.stderr || ''}\n${r.error.code === 'ETIMEDOUT' ? `stopped after ${timeout / 1000} s` : r.error.message}`;
  const lines = (r.stdout || '').split('\n');
  const ok = lines.filter((l) => l.startsWith('ok ')).length;
  const bad = lines.filter((l) => l.startsWith('FAIL'));
  total += ok + bad.length;
  if (r.status !== 0) failed++;
  console.log(`${r.status === 0 ? 'pass' : 'FAIL'}  ${f.padEnd(24)} ${ok} checks passed${bad.length ? `, ${bad.length} failed` : ''}`);
  for (const b of bad) console.log('        ' + b);
  if (r.status !== 0 && !bad.length) console.log('        ' + ((r.stderr || r.stdout || '').trim().split('\n').slice(-4).join('\n        ')));
}
console.log(`\n${files.length - failed} of ${files.length} scripts passed, ${total} checks in all`);
process.exit(failed ? 1 : 0);
