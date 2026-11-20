// Browser check for section X3.2: the course kernels loaded from the labs, the excerpts and their
// permalinks, the launch-shapes explorer against sim/elemshapes.ts (every style, n, block size,
// N_READS and offset the lesson names), the variant sort against sim/backendmap.ts, the embedded
// op map, and the questions.
// Usage: APP_URL=http://localhost:6721 node e2e/x3-sx32.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;

await goto(page, 'x3/X3.2');
const M = await page.evaluate(async () => {
  const s = await import('/src/sim/x3sources.ts');
  const b = await import('/src/sim/backendmap.ts');
  return { commits: s.PINNED.map((p) => p.commit), items: b.VARIANT_MECHANISMS.map((v) => ({ id: v.id, m: v.mechanism })), mechs: b.MECHANISMS.map((m) => m.id) };
});

// ---------------------------------------------------------------- code
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('vecAdd4Kernel(')), { timeout: 20000 }).catch(() => undefined);
const codes = await page.$$eval('figure.code:not(.x3-excerpt) pre', (ps) => ps.map((p) => p.textContent));
check('lab 2.3’s addResidualKernel is loaded from the lab', codes.some((c) => c.startsWith('__global__ void addResidualKernel') && c.includes('X[i] = X[i] + R[i];')));
check('lab 6.2’s vecAdd4Kernel is loaded from the lab', codes.some((c) => c.startsWith('__global__ void vecAdd4Kernel') && c.includes('float4')));
const ex = await page.$$eval('figure.x3-excerpt', (fs) => fs.map((f) => ({ id: f.dataset.excerpt, href: f.querySelector('a[data-permalink]').href, text: f.textContent })));
check('seven excerpts', ex.length === 7, String(ex.length));
check('every excerpt links at a pinned commit and names its licence', ex.every((e) => M.commits.some((c) => e.href.includes(c)) && /MIT|BSD/.test(e.text)));
check('elided lines are drawn as a gap', ex.find((e) => e.id === 'x32-binary-vv').text.includes('lines left out'));

// ---------------------------------------------------------------- launch shapes
const LS = bench('Launch shapes');
const setRange = async (label, v) => {
  await page.$eval(`${LS} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};
const seg = async (group, text) => {
  await page.evaluate((LS, group, text) => {
    const g = document.querySelector(`${LS} [aria-label="${group}"]`);
    [...g.querySelectorAll('button')].find((b) => b.textContent.trim() === text).click();
  }, LS, group, text);
  await pause(60);
};
const readout = () => page.$eval(`${LS} [data-readout=summary]`, (e) => ({ ...e.dataset, text: e.textContent }));
const model = (cfg) =>
  page.evaluate(async (cfg) => {
    const m = await import('/src/sim/elemshapes.ts');
    const l = m.launch(cfg);
    const c = m.pathCounts(l);
    const t = m.tailSummary(l);
    return { threads: String(l.threads), tailElements: String(t.elements), tailThreads: String(t.threads), divergent: m.divergentWarps(l).join(','), vector: String(c.vector.threads), misaligned: String(c.misaligned.threads), idle: String(c.idle.threads) };
  }, cfg);
const same = async (cfg, label) => {
  const r = await readout();
  const m = await model(cfg);
  const keys = ['threads', 'tailElements', 'tailThreads', 'divergent', 'vector', 'misaligned', 'idle'];
  const ok = keys.every((k) => r[k] === m[k]);
  const cells = await page.$$eval(`${LS} svg[data-part=elements] rect[data-path]`, (rs) => rs.length);
  check(label, ok && cells === cfg.n, JSON.stringify({ r: keys.map((k) => r[k]), m: keys.map((k) => m[k]), cells }));
  return r;
};

let r = await same({ style: 'binary_vv', n: 150, block: 32, nReads: 4, offset: 0 }, 'default: binary_vv, n 150, N_READS 4 matches the model');
check('n 150, N_READS 4: tail of 2 elements in 1 thread, one warp splits', r.tailElements === '2' && r.tailThreads === '1' && r.divergent.split(',').filter(Boolean).length === 1, JSON.stringify(r));
await setRange('pointer offset', 1);
r = await same({ style: 'binary_vv', n: 150, block: 32, nReads: 4, offset: 1 }, 'offset 1 matches the model');
check('offset 1: no vector loads, every main-branch thread misaligned', r.vector === '0' && r.misaligned === '37', JSON.stringify(r));
const strided = await page.$$eval(`${LS} [data-warp-loads="0"] [data-contiguous]`, (s) => s.map((x) => x.dataset.contiguous));
check('offset 1: warp 0’s loads are strided, four of them', strided.length === 4 && strided.every((c) => c === '0'), strided.join(','));
await setRange('pointer offset', 0);
const contiguous = await page.$$eval(`${LS} [data-warp-loads="0"] [data-contiguous]`, (s) => s.map((x) => x.dataset.contiguous));
check('offset 0: warp 0 issues one unbroken vector load', contiguous.length === 1 && contiguous[0] === '1', contiguous.join(','));
await seg('N_READS', '2');
await setRange('pointer offset', 2);
r = await same({ style: 'binary_vv', n: 150, block: 32, nReads: 2, offset: 2 }, 'N_READS 2, offset 2 matches the model');
check('N_READS 2 with a two-float offset is still aligned', r.aligned === '1' && r.misaligned === '0');
await seg('N_READS', '8');
await setRange('pointer offset', 0);
for (const n of [1, 7, 64, 255, 300]) {
  await setRange('n', n);
  await same({ style: 'binary_vv', n, block: 32, nReads: 8, offset: 0 }, `binary_vv N_READS 8, n ${n} matches the model`);
}
if (out) await page.screenshot({ path: out + '-shapes-vv.png' });
await seg('Kernel style', 'one thread per element');
await seg('Block size', '128');
for (const n of [1, 100, 129, 300]) {
  await setRange('n', n);
  await same({ style: 'per-element', n, block: 128, nReads: 8, offset: 0 }, `one thread per element, block 128, n ${n} matches the model`);
}
await seg('Kernel style', 'grid-stride loop');
await seg('Block size', '32');
const sizes = new Set();
for (const n of [10, 150, 300]) {
  await setRange('n', n);
  r = await same({ style: 'grid-stride', n, block: 32, nReads: 8, offset: 0 }, `grid-stride, n ${n} matches the model`);
  sizes.add(r.threads);
}
check('grid-stride: the grid does not change with n', sizes.size === 1, [...sizes].join(','));
if (out) await page.screenshot({ path: out + '-shapes-stride.png' });

// ---------------------------------------------------------------- variant sort
const VS = bench('How each project picks a variant');
const choose = async (id, m) => {
  await page.$eval(`${VS} select[aria-label="mechanism for ${id}"]`, (s, m) => {
    s.value = m;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, m);
};
check('check is disabled until every line has a choice', await page.evaluate((VS) => [...document.querySelectorAll(VS + ' button')].find((b) => b.textContent === 'Check').disabled, VS));
for (const it of M.items) await choose(it.id, it.m === 'compile-time' && it.id === 'llama-template' ? 'pipeline-constant' : it.m);
await page.evaluate((VS) => [...document.querySelectorAll(VS + ' button')].find((b) => b.textContent === 'Check').click(), VS);
await pause(100);
const res = await page.$$eval(`${VS} [data-item]`, (d) => d.map((x) => [x.dataset.item, x.dataset.result]));
check('the sort marks the one wrong line and the rest right', res.every(([id, v]) => (id === 'llama-template' ? v === 'wrong' : v === 'right')), JSON.stringify(res));
check('the score is total minus one', (await page.$eval(`${VS} [data-readout=sort-score]`, (e) => e.textContent)).includes(`${M.items.length - 1} of ${M.items.length}`));
await choose('llama-template', 'compile-time');
await page.evaluate((VS) => [...document.querySelectorAll(VS + ' button')].find((b) => b.textContent === 'Check').click(), VS);
await pause(100);
check('all right after fixing it', (await page.$eval(`${VS} [data-readout=sort-score]`, (e) => e.textContent)).includes(`${M.items.length} of ${M.items.length}`));
check('all six bins are offered', (await page.$$eval(`${VS} [data-item] select`, (s) => s[0].options.length - 1)) === M.mechs.length);

// ---------------------------------------------------------------- the embedded op map
const om = await page.$eval(`${bench('Op map: embedding')} [data-part=cells]`, (e) => [e.dataset.op, e.dataset.project]);
check('the op map is embedded preselected on embedding, llama.cpp', om[0] === 'embedding' && om[1] === 'llama.cpp');

// ---------------------------------------------------------------- questions
const q = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const tail = await q('How many elements go through the scalar tail loop?');
await (await tail.$('input[type=number]')).type('2');
await (await tail.$('button.primary')).click();
await pause(150);
check('q-tail: 2 is accepted', (await tail.evaluate((e) => e.textContent)).includes('Correct.'));
const answer = async (needle, index) => {
  const h = await q(needle);
  await (await h.$$('input[type=radio]'))[index].click();
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
check('q-misaligned: right answer', (await answer('starts one float into its buffer', 0)).includes('Correct.'));
check('q-read-binvv: a wrong answer is explained', (await answer('What does <code>binary_vv</code> add?'.replace(/<\/?code>/g, ''), 1)).includes('Not quite.'));
check('q-pdl: right answer', (await answer('in a HIP build of llama.cpp?', 0)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
