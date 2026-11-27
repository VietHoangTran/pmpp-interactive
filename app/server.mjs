// One process: serves the app (through Vite) and a small local API.
//   GET/PUT /api/progress        progress/progress.json (yours; sync it between machines)
//   GET     /api/reports         latest lab results per backend, from reports/runs/*.json
//   GET     /api/lab-file        a lab source file, for display
//   GET     /api/race-labs       ids of the labs built for the race emulator (RACE in pmpp_add_lab)
//   GET     /api/frontend-check  per .cu file, 'ok' or 'fail' from the last CUDA front-end check
//                                (reports/runs/checks/cuda-frontend.json), for files unchanged since then
//   GET     /content/...         narration scripts, audio and manifests
import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createServer as createVite } from 'vite';

const APP = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(APP);
const PROGRESS = path.join(REPO, 'progress', 'progress.json');
const PORT = Number(process.env.PORT || 5173);

const MIME = { '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.json': 'application/json', '.txt': 'text/plain; charset=utf-8' };

function send(res, code, body, type = 'application/json') {
  res.writeHead(code, { 'Content-Type': type, 'Cache-Control': 'no-store' });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
}

// Stable key order so the file diffs and merges cleanly in git.
function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === 'object') return Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]));
  return v;
}

async function readBody(req, limit = 4 << 20) {
  const chunks = [];
  let n = 0;
  for await (const c of req) {
    n += c.length;
    if (n > limit) throw new Error('body too large');
    chunks.push(c);
  }
  return Buffer.concat(chunks).toString('utf8');
}

// Resolve `rel` under `root`, refusing anything that escapes it.
function inside(root, rel) {
  const p = path.resolve(root, rel);
  return p === root || p.startsWith(root + path.sep) ? p : null;
}

async function api(req, res, url) {
  if (url.pathname === '/api/progress') {
    if (req.method === 'GET') {
      try {
        return send(res, 200, await fsp.readFile(PROGRESS, 'utf8'));
      } catch {
        return send(res, 200, { version: 1 });
      }
    }
    if (req.method === 'PUT') {
      const data = JSON.parse(await readBody(req));
      await fsp.mkdir(path.dirname(PROGRESS), { recursive: true });
      const tmp = PROGRESS + '.tmp';
      await fsp.writeFile(tmp, JSON.stringify(sortKeys(data), null, 1) + '\n');
      await fsp.rename(tmp, PROGRESS);
      return send(res, 200, { ok: true });
    }
  }
  if (url.pathname === '/api/reports' && req.method === 'GET') {
    const dir = path.join(REPO, 'reports', 'runs');
    const out = {};
    try {
      for (const f of await fsp.readdir(dir)) {
        if (f.endsWith('.json')) out[f.replace(/\.json$/, '')] = JSON.parse(await fsp.readFile(path.join(dir, f), 'utf8'));
      }
    } catch {
      /* no runs yet */
    }
    return send(res, 200, out);
  }
  if (url.pathname === '/api/frontend-check' && req.method === 'GET') {
    // Written by labs/pmpp.py (verify.sh, or `pmpp.py frontend`): { generated, files: { path: { ok, sha256 } } }.
    // A file edited since the check has no status: the check said nothing about its current text.
    const out = {};
    try {
      const rec = JSON.parse(await fsp.readFile(path.join(REPO, 'reports', 'runs', 'checks', 'cuda-frontend.json'), 'utf8'));
      for (const [rel, f] of Object.entries(rec.files ?? {})) {
        const p = inside(path.join(REPO, 'labs'), rel.replace(/^labs\//, ''));
        if (!p || typeof f?.sha256 !== 'string') continue;
        const now = await fsp.readFile(p).catch(() => null);
        if (now && crypto.createHash('sha256').update(now).digest('hex') === f.sha256) out[rel] = f.ok ? 'ok' : 'fail';
      }
    } catch {
      /* no check recorded on this machine */
    }
    return send(res, 200, out);
  }
  if (url.pathname === '/api/race-labs' && req.method === 'GET') {
    // A lab opts in to the race backend in its CMakeLists.txt: pmpp_add_lab(<id> ... RACE).
    const labs = path.join(REPO, 'labs');
    const out = [];
    for (const ch of await fsp.readdir(labs, { withFileTypes: true }).catch(() => [])) {
      if (!ch.isDirectory()) continue;
      for (const lab of await fsp.readdir(path.join(labs, ch.name), { withFileTypes: true }).catch(() => [])) {
        if (!lab.isDirectory()) continue;
        const text = await fsp.readFile(path.join(labs, ch.name, lab.name, 'CMakeLists.txt'), 'utf8').catch(() => '');
        for (const m of text.matchAll(/\bpmpp_add_lab\(\s*([^\s)]+)([^)]*)\)/g)) if (/\bRACE\b/.test(m[2])) out.push(m[1]);
      }
    }
    return send(res, 200, out.sort());
  }
  if (url.pathname === '/api/lab-file' && req.method === 'GET') {
    const rel = url.searchParams.get('path') || '';
    const p = inside(path.join(REPO, 'labs'), rel);
    if (!p || !/\.(cu|metal|mm|cpp|hpp|h|md)$/.test(p)) return send(res, 400, { error: 'bad path' });
    try {
      return send(res, 200, await fsp.readFile(p, 'utf8'), 'text/plain; charset=utf-8');
    } catch {
      return send(res, 404, { error: 'not found' });
    }
  }
  return send(res, 404, { error: 'no such endpoint' });
}

async function content(req, res, url) {
  const p = inside(path.join(REPO, 'content'), decodeURIComponent(url.pathname.slice('/content/'.length)));
  const type = p && MIME[path.extname(p)];
  if (!p || !type) return send(res, 400, { error: 'bad path' });
  try {
    const stat = await fsp.stat(p);
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
    if (range) {   // audio seeking
      const start = range[1] ? Number(range[1]) : 0;
      const end = range[2] ? Number(range[2]) : stat.size - 1;
      res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${start}-${end}/${stat.size}`,
                           'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 });
      return fs.createReadStream(p, { start, end }).pipe(res);
    }
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store' });
    fs.createReadStream(p).pipe(res);
  } catch {
    // A chapter whose narration is not generated yet has no manifest. That is a normal
    // state, not an error: answer null so the browser logs nothing and the player runs
    // silently on estimated timings.
    if (/\/narration\/manifest(?:\.vi)?\.json$/.test(p)) return send(res, 200, null);
    send(res, 404, { error: 'not found' });
  }
}

// One port for everything: the hot-reload websocket is attached to this same
// server, so a second copy of the app (or any other Vite project) cannot clash.
let vite;
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (url.pathname.startsWith('/api/')) return await api(req, res, url);
    if (url.pathname.startsWith('/content/')) return await content(req, res, url);
  } catch (e) {
    return send(res, 500, { error: String(e && e.message ? e.message : e) });
  }
  vite.middlewares(req, res);
});

server.on('error', (e) => {
  if (e.code === 'EADDRINUSE') {
    console.error(`\n  Port ${PORT} is already in use. Is the app already running?\n  To use another port:  PORT=5180 ./start.sh\n`);
    process.exit(1);
  }
  throw e;
});

vite = await createVite({ root: APP, server: { middlewareMode: true, hmr: { server } }, appType: 'spa' });

server.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  PMPP interactive textbook: http://localhost:${PORT}\n  progress file: ${path.relative(REPO, PROGRESS)}\n`);
});
