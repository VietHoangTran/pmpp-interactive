// Browser check for section 8.4, "A 3D tile and its halo": the two lab listings, the
// prediction of the basic kernel's FLOP/B, the reader's own threshold, the explorers
// "Stencil or dense filter", "A cube-shaped tile" and "One warp's load", and the
// questions. Everything the page shows is compared with the models it runs on
// (src/sim/stencil3d.ts and src/sim/stencilwarp.ts, imported here directly).
// Usage: node e2e/ch08-s84.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses, tabStatus } from './labstatus.mjs';
import {
  basicAI,
  cubeAI,
  denseFlops,
  denseIdealAI,
  denseTaps,
  haloCount,
  haloFraction,
  maxCubeSide,
  neverRead,
  points,
  role,
  sharedBytesCube,
  stencilFlops,
  stencilIdealAI,
} from '../src/sim/stencil3d.ts';
import { blockView, dims3, warpView } from '../src/sim/stencilwarp.ts';

const out = process.argv[2];
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file and start from an empty one. Reports are
// replaced by made-up ones, clearly named as fixtures, to drive the places that read the
// reader's own devices: a CUDA device, a HIP device with a small block limit, and a Metal
// device with lab 5.1's threshold and lab 8.1's pipeline caps.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const LAB = 'ch08/lab1_stencil3d';
const FIXTURE = {
  cuda: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'cuda', available: true, name: 'e2e fixture device', warpSize: '32', maxThreadsPerBlock: '1024', sharedMemPerBlock: '49152' },
    results: [],
    work_state: {},
  },
  hip: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'hip', available: true, name: 'e2e fixture small device', warpSize: '64', maxThreadsPerBlock: '256', sharedMemPerBlock: '65536' },
    results: [],
    work_state: {},
  },
  metal: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'metal', available: true, name: 'e2e fixture Metal device', maxThreadgroupMemoryLength: '32768' },
    results: [
      {
        lab: 'ch05/lab1_roofline', variant: 'solution', backend: 'metal', tile: 16, built: true, test_rc: 0,
        tests: [{ case: 'fixture', pass: true }],
        bench: [{ name: 'roofline', threshold_flop_per_byte: 7.5, compute_gflops: 1500, bandwidth_gb_per_s: 200 }],
      },
      {
        lab: LAB, variant: 'solution', backend: 'metal', tile: 32, built: true, test_rc: 0,
        tests: [{ case: 'fixture', pass: true }],
        bench: [
          { name: 'limits', tile: 8, max_threads: 1024 },
          { name: 'limits', tile: 32, max_threads: 896 },
        ],
      },
    ],
    work_state: {},
  },
};
await page.setRequestInterception(true);
page.on('request', (r) => {
  const u = r.url();
  if (u.includes('/api/progress'))
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  if (u.includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE) });
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const pressSeg = async (scope, control, label) => {
  const ok = await page.evaluate(
    (scope, control, label) => {
      const btn = [...document.querySelectorAll(`${scope} [data-control="${control}"] button`)].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    control,
    label,
  );
  await pause(80);
  return ok;
};
const setRange = async (sel, value) => {
  await page.$eval(
    sel,
    (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    value,
  );
  await pause(60);
};
const setSelect = async (sel, value) => {
  await page.$eval(
    sel,
    (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
      el.dispatchEvent(new Event('change', { bubbles: true }));
    },
    value,
  );
  await pause(80);
};
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const r2 = (v) => (Math.round(v * 100) / 100).toString();
const pct1 = (v) => (Math.round(v * 1000) / 10).toString() + '%';
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch08/8.4');
await page.waitForSelector(bench('One warp\'s load'));
await page.waitForFunction(() => document.querySelectorAll('figure.code').length >= 2, { timeout: 15000 });

// ---------------------------------------------------------------- the lab listings
const basicFile = fs.readFileSync(path.join(REPO, 'labs', LAB, 'solution/kernel.cu'), 'utf8');
const cubeFile = fs.readFileSync(path.join(REPO, 'labs/ch08/shared/tiled_cube.cu'), 'utf8');
const codes = await page.$$eval('figure.code', (f) => f.map((x) => ({ tabs: [...x.querySelectorAll('.tabs button')].map((b) => b.textContent), code: x.querySelector('pre').textContent, status: x.querySelector('.status')?.textContent ?? '', cap: x.querySelector('figcaption')?.textContent ?? '' })));
check('two listings, each with CUDA and HIP and Metal tabs', codes.length === 2 && codes.every((c) => c.tabs.join('|') === 'CUDA and HIP|Metal'), JSON.stringify(codes.map((c) => c.tabs)));
check('first listing is the basic kernel from the lab file, and only that function', codes[0].code.includes('stencilBasicKernel') && !codes[0].code.includes('stencilRegTiledKernel'), codes[0].code.slice(0, 200));
const sig = basicFile.split('\n').find((l) => l.includes('__global__ void stencilBasicKernel'));
check('the listing is the file\'s own text', !!sig && codes[0].code.includes(sig.trim()));
check('second listing is the cube kernel from labs/ch08/shared', codes[1].code.includes('stencilTiledCubeKernel') && cubeFile.includes('stencilTiledCubeKernel'));
const def = /#define\s+(\w*IN_TILE_DIM)\s+(\d+)/.exec(cubeFile);
check('the cube caption names the file\'s own tile constants', !!def && codes[1].cap.includes(`${def[1]} as ${def[2]}`), codes[1].cap);
// The fixture has no emulator, CUDA or HIP run of the solution; the page still reads this machine's
// CUDA front-end check (GET /api/frontend-check is not intercepted).
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
const wantCu = tabStatus(FIXTURE, LAB, 'kernel.cu', frontend);
check(`CUDA status reads the fixture (no emulator, CUDA or HIP run of the solution) and the front-end check: "${wantCu}"`, codes[0].status === wantCu, codes[0].status);
await page.evaluate(() => [...document.querySelectorAll('figure.code')][0].querySelectorAll('.tabs button')[1].click());
await pause(80);
check('Metal tab shows the Metal kernel, verified on Metal per the fixture', (await text('figure.code pre')).includes('stencil_basic') && (await text('figure.code .status')) === 'verified on Metal');

// ---------------------------------------------------------------- the prediction
{
  const q = await qHandle("basic kernel's ratio of operations to bytes");
  await (await q.$('input[type=number]')).type('0.464');
  await (await q.$('button.primary')).click();
  await pause(150);
  const t = await q.evaluate((e) => e.textContent);
  check(`reveal gives 13 FLOP over 32 bytes, ${basicAI(7)}`, t.includes(`13 FLOP over 32 bytes: ${basicAI(7)} FLOP/B`), t.slice(0, 200));
  check('reveal says the store was left out', t.includes('You left out the store'), t.slice(0, 300));
}

// ---------------------------------------------------------------- your threshold
{
  const t = await text('[data-threshold="metal"]');
  check('your threshold comes from lab 5.1\'s roofline row', t.includes('7.5 FLOP/B') && t.includes(`${Number((7.5 / basicAI(7)).toPrecision(3))} times below it`), t);
}

// ---------------------------------------------------------------- stencil or dense filter
const SH = bench('Stencil or dense filter');
const shapeRows = () =>
  page.$$eval(`${SH} [data-readout="shape"] tbody tr`, (rows) => rows.map((r) => [...r.querySelectorAll('td[data-out]')].map((c) => c.textContent.trim())));
for (const [d, r] of [[3, 1], [2, 2], [3, 3], [1, 2]]) {
  await pressSeg(SH, 'dim', `${d}D`);
  await pressSeg(SH, 'order', String(r));
  const rows = await shapeRows();
  const p = points(d, r);
  const m = denseTaps(d, r);
  const want = [[String(p), String(stencilFlops(p)), String(stencilIdealAI(p))], [String(m), String(denseFlops(m)), String(denseIdealAI(d, r))]];
  check(`${d}D order ${r}: stencil and dense rows match the model`, JSON.stringify(rows) === JSON.stringify(want), JSON.stringify(rows) + ' vs ' + JSON.stringify(want));
  const dots = await page.$$eval(`${SH} svg [data-cell="stencil"]`, (x) => x.length);
  const all = await page.$$eval(`${SH} svg [data-cell]`, (x) => x.length);
  check(`${d}D order ${r}: drawing has ${p} stencil points in ${m} cells`, dots === p && all === m, `${dots} / ${all}`);
}
{
  const t = await page.$$eval(`${SH} [data-table="p190"] tbody tr`, (rows) => rows.map((r) => [...r.querySelectorAll('td')].map((c) => c.textContent.trim())));
  check('the p. 190 table: 1.125, 2.125, 3.125, 4.625 against 2.25, 6.25, 12.25, 85.75',
    JSON.stringify(t.map((r) => [r[2], r[4]])) === JSON.stringify([['1.125', '2.25'], ['2.125', '6.25'], ['3.125', '12.25'], ['4.625', '85.75']]), JSON.stringify(t));
}
if (out) await page.screenshot({ path: out + '-shape.png' });

// ---------------------------------------------------------------- a cube-shaped tile
const CB = bench('A cube-shaped tile');
const cubeOut = async (k) => text(`${CB} [data-out="${k}"]`);
const sliceCounts = () => page.$$eval(`${CB} svg [data-role]`, (x) => x.reduce((a, e) => ({ ...a, [e.dataset.role]: (a[e.dataset.role] ?? 0) + 1 }), {}));
const modelSlice = (t, z) => {
  const c = {};
  for (let y = 0; y < t; y++) for (let x = 0; x < t; x++) { const ro = role(t, 1, x, y, z); c[ro] = (c[ro] ?? 0) + 1; }
  return c;
};
await setRange(`${CB} input[aria-label="tile side"]`, 8);
check('t = 8: 512 of 1,024 threads', (await cubeOut('threads')) === '512 of 1,024', await cubeOut('threads'));
check(`t = 8: ${r2(cubeAI(8))} FLOP/B`, (await cubeOut('ai')).startsWith(r2(cubeAI(8)) + ' of 1.625'), await cubeOut('ai'));
check(`t = 8: halo ${haloCount(8)} (${pct1(haloFraction(8))})`, (await cubeOut('halo')) === `${haloCount(8)} of 512 (${pct1(haloFraction(8))})`, await cubeOut('halo'));
check(`t = 8: ${neverRead(8)} never read`, (await cubeOut('never')).startsWith(`${neverRead(8)} `), await cubeOut('never'));
check('t = 8, book limit: largest legal side 10', (await cubeOut('maxside')) === String(maxCubeSide(1024)));
for (const z of [0, 1, 4]) {
  await setRange(`${CB} input[aria-label="slice"]`, z);
  const got = await sliceCounts();
  check(`slice z = ${z} matches the model's roles`, JSON.stringify(got, Object.keys(got).sort()) === JSON.stringify(modelSlice(8, z), Object.keys(got).sort()), JSON.stringify(got) + ' vs ' + JSON.stringify(modelSlice(8, z)));
}
await setRange(`${CB} input[aria-label="tile side"]`, 10);
check(`t = 10: 1,000 threads fit, ${r2(cubeAI(10))} FLOP/B`, (await cubeOut('threads')) === '1,000 of 1,024' && (await cubeOut('ai')).startsWith(r2(cubeAI(10)) + ' '));
await setRange(`${CB} input[aria-label="tile side"]`, 11);
check('t = 11: over the limit, marked as a failure', (await page.$$(`${CB} [data-out="threads"] .chip.fail`)).length === 1 && (await cubeOut('threads')).includes('1,331 > 1,024'));
check('t = 11: the reach line says it would not launch', (await cubeOut('reach')).includes('would not launch') && (await cubeOut('reach')).includes(`t = 10: ${r2(cubeAI(10))}`));
check('the curve marks the thread limit at t ≤ 10', (await page.$eval(`${CB} [data-mark="limit"]`, (e) => e.textContent)).includes('t ≤ 10'));
await setRange(`${CB} input[aria-label="tile side"]`, 8);
await pressSeg(CB, 'bytes', '8 (double)');
check(`8-byte values: ${r2(cubeAI(8, 1, 8))} FLOP/B and ${n(sharedBytesCube(8, 8))} B`, (await cubeOut('ai')).startsWith(r2(cubeAI(8, 1, 8)) + ' of 0.8125') && (await cubeOut('shared')).startsWith('4,096 B (4 KB)'), (await cubeOut('ai')) + ' / ' + (await cubeOut('shared')));
check('the book\'s t = 8 mark is only drawn for 4-byte values', (await page.$$(`${CB} [data-mark="book"]`)).length === 0);
await pressSeg(CB, 'bytes', '4 (float)');
const limitOptions = await page.$$eval(`${CB} select[data-control="limit"] option`, (o) => o.map((x) => [x.value, x.textContent]));
check('thread limits: the book, and each fixture device', JSON.stringify(limitOptions.map((o) => o[0])) === JSON.stringify(['book', 'cuda', 'hip', 'metal']), JSON.stringify(limitOptions));
await setSelect(`${CB} select[data-control="limit"]`, 'hip');
check('a device with 256 threads per block: largest side 6, t = 8 over', (await cubeOut('maxside')) === '6' && (await cubeOut('threads')).includes('512 > 256'));
await setSelect(`${CB} select[data-control="limit"]`, 'metal');
check('Metal: the lowest pipeline cap lab 8.1 recorded (896) gives side 9', (await cubeOut('maxside')) === String(maxCubeSide(896)) && (await cubeOut('shared')).includes('maxThreadgroupMemoryLength'));
check('Metal: the note says whose pipeline the cap is from', (await text(`${CB} footer`)).includes("register-tiled kernel's pipelines"));
if (out) await page.screenshot({ path: out + '-cube.png' });

// ---------------------------------------------------------------- one warp's load
const WB = bench("One warp's load");
const warpOut = async (k) => text(`${WB} [data-out="${k}"]`);
const cfg = (over = {}) => ({ block: dims3(8, 8, 8), r: 1, blockIdx: dims3(1, 1, 1), warp: 0, width: 32, N: 128, offset: -1, ...over });
const loadCheck = async (name, c) => {
  const v = warpView(c);
  const b = blockView(c);
  const got = [await warpOut('rows'), await warpOut('transactions'), await warpOut('bytes'), await warpOut('eff'), await warpOut('block')];
  const want = [String(v.rows.length), String(v.transactions), `${v.bytesUsed} / ${v.bytesFetched}`, pct1(v.efficiency), `${b.transactions} transactions, ${pct1(b.efficiency)}`];
  check(name, JSON.stringify(got) === JSON.stringify(want), JSON.stringify(got) + ' vs ' + JSON.stringify(want));
  const strips = await page.$$eval(`${WB} [data-row]`, (x) => x.length);
  check(name + ': one memory strip per row', strips === v.rows.length, `${strips}`);
  const used = await page.$$eval(`${WB} [data-mem="used"]`, (x) => x.length);
  check(name + ': dark cells are the loading lanes', used === v.loading, `${used} vs ${v.loading}`);
};
await loadCheck('8 × 8 × 8, interior block, warp 0: 8 transactions, 50%', cfg());
check('the interior cube warp is 8 transactions at 50% (facts A2)', (await warpOut('transactions')) === '8' && (await warpOut('eff')) === '50%');
if (out) await page.screenshot({ path: out + '-warp-cube.png' });
await pressSeg(WB, 'shape', '32 × 8 × 4');
await loadCheck('32 × 8 × 4: one row, 5 transactions, 80%', cfg({ block: dims3(32, 8, 4) }));
await pressSeg(WB, 'offset', '0 (compare)');
await pressSeg(WB, 'bx', '0');
await loadCheck('32 × 8 × 4, offset 0, block 0: aligned, 4 transactions, 100%', cfg({ block: dims3(32, 8, 4), offset: 0, blockIdx: dims3(0, 1, 1) }));
await pressSeg(WB, 'bx', '1');
await loadCheck('32 × 8 × 4, offset 0, block 1: 5 transactions again', cfg({ block: dims3(32, 8, 4), offset: 0 }));
await pressSeg(WB, 'shape', '8 × 8 × 8');
await pressSeg(WB, 'offset', '−1 (the book)');
await pressSeg(WB, 'bx', '0');
await loadCheck('8 × 8 × 8, block 0: the k = −1 lanes load nothing', cfg({ blockIdx: dims3(0, 1, 1) }));
check('block 0: four lanes drawn as skipping the load', (await page.$$eval(`${WB} [data-state="skips"]`, (x) => x.length)) === 4);
await pressSeg(WB, 'bx', '1');
await pressSeg(WB, 'shape', '10 × 10 × 10');
await loadCheck('10 × 10 × 10: rows of ten straddle three segments', cfg({ block: dims3(10, 10, 10) }));
await setRange(`${WB} input[aria-label="warp"]`, 31);
check('10 × 10 × 10, last warp: 8 threads and 24 padded lanes', (await warpOut('padded')).includes('8 threads; 24 lanes are padding'), await warpOut('padded'));
await pressSeg(WB, 'shape', '8 × 8 × 8');
await setRange(`${WB} input[aria-label="warp"]`, 0);
await pressSeg(WB, 'width', '64 (AMD wave64)');
await loadCheck('wave64: one wavefront is 8 rows, a whole plane', cfg({ width: 64 }));
check('the Metal width is offered but disabled', await page.$eval(`${WB} [data-control="width"]`, (e) => [...e.querySelectorAll('button')].some((b) => b.textContent === 'Metal' && b.disabled && b.title.includes('not documented'))));
check('pressing Metal does nothing', !(await pressSeg(WB, 'width', 'Metal')));
await pressSeg(WB, 'width', '32');
check('the explorer has no compute phase (trimmed: chapter 8 is condensed)', (await page.$$(`${WB} [data-control="phase"]`)).length === 0);
check('the trimmed questions are gone', (await page.$$eval('.q', (qs) => qs.filter((q) => /64-wide wavefronts, how many rows|how many of its 16 warps have both/.test(q.textContent)).length)) === 0);

// ---------------------------------------------------------------- questions
const answer = async (needle, value) => {
  const q = await qHandle(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
let t = await answer('What is the largest cube side', 10);
check('q-cube-max: 10 is correct', t.includes('Correct.'), t.slice(0, 200));
t = await answer('Under NVIDIA\'s 32-byte transaction model, how many transactions', 4);
check('q-transactions: 4 is wrong and the answer 8 is given', t.includes('Not quite. The answer is 8 transactions.'), t.slice(0, 200));
t = await answer('how many of the 512 loaded elements does no thread ever read', neverRead(8));
check('q-never-read: 80 is correct', t.includes('Correct.'));
{
  const q = await qHandle('as the book suggests. Is each warp');
  await (await q.$$('input[type=radio]'))[0].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  const qt = await q.evaluate((e) => e.textContent);
  check('q-x32: five segments and 0.85 is correct', qt.includes('Correct.'));
  check('q-x32: the answer says the flat tile is larger (360 against 216) but a smaller share of the loads',
    qt.includes('360 outputs for 1,024 loads') && qt.includes("the cube's 216 of 512") && qt.includes("30 × 6 × 2 = 360 points, is larger than the cube's 216") && !/output tile shrinks/.test(qt), qt.slice(0, 400));
}
{
  const q = await qHandle('Switch the t = 8 tile from 4-byte to 8-byte values');
  await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('becomes compute-bound')).querySelector('input').click());
  await (await q.$('button.primary')).click();
  await pause(120);
  const qt = await q.evaluate((e) => e.textContent);
  check('q-double: "the ratio halves, so the kernel becomes compute-bound" is wrong, with the right link', qt.includes('Not quite.') && qt.includes('towards the memory-bound side'), qt.slice(0, 400));
}
check('the basic kernel\'s caption does not claim the HIP path compiled', (await text('body')).includes('that path has not been compiled on AMD hardware yet') && !(await text('body')).includes('The HIP build compiles the CUDA file unchanged'));
check('the AI curve keeps a readable width and scrolls in its own frame', await page.$eval('[data-scroll="ai-curve"]', (e) => getComputedStyle(e).overflowX === 'auto' && parseFloat(getComputedStyle(e.querySelector('svg')).minWidth) >= 500));

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch08/8.4', [
  { label: 'CUDA and HIP', lab: 'ch08/lab1_stencil3d', file: 'kernel.cu' },
  { label: 'Metal', lab: 'ch08/lab1_stencil3d', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
