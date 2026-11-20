// Browser check for section 22.5: the "A decode step on the roofline" explorer and the "Batch
// slots on a server" explorer, both against their models (src/sim/batching.ts and
// src/sim/serving.ts, imported directly), the two predictions and the questions.
// The reports the page reads are the real ones plus one stubbed device ("stubgpu", lab 5.1 and
// lab 22.2 rows), so the overlay of measured points and their knees is exercised on any machine.
// Usage: node e2e/ch22-s225.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  BOOK_DEVICE, PROJ_4096, bStar, kneeOf, m8Shape, precisionFor, stepAt, stepBStar,
} from '../src/sim/batching.ts';
import { MIXES, makeRequests, serve } from '../src/sim/serving.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// ---- reports: the real ones plus a stubbed GPU
const STUB = { peak: 2000, bw: 250 }; // threshold 8 FLOP/B
const STEP_ROWS = [];
const SHAPE = { d: 2048, layers: 2, vocab: 32000 }; // lab 22.2's driver halves d on a small device; the rows say so
for (const batch of [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]) {
  // M2-like: flat, then in proportion past 24 rows. M3-like: flat to 128 (its tile), then doubling.
  // Lab 22.1's gather (gemv) runs batches up to 8 only, flat.
  STEP_ROWS.push({ name: 'step', gemm: 'm2', batch, seconds: 0.004 * Math.max(1, batch / 24), ...SHAPE });
  STEP_ROWS.push({ name: 'step', gemm: 'm3', batch, seconds: 0.006 * Math.ceil(batch / 128), ...SHAPE });
  if (batch <= 8) STEP_ROWS.push({ name: 'step', gemm: 'gemv', batch, seconds: 0.003, ...SHAPE });
}
const stubRun = {
  generated: '2026-11-20T12:00:00',
  device: { kind: 'device', backend: 'stubgpu', available: true, name: 'Stub GPU' },
  results: [
    { lab: 'ch05/lab1_roofline', variant: 'solution', backend: 'stubgpu', tile: 0, built: true, test_rc: 0, tests: [],
      bench: [{ name: 'roofline', compute_gflops: STUB.peak, bandwidth_gb_per_s: STUB.bw }] },
    { lab: 'ch22/lab2_batching', variant: 'solution', backend: 'stubgpu', tile: 0, built: true, test_rc: 0, tests: [], bench: STEP_ROWS },
  ],
};
const real = await (await fetch(BASE + '/api/reports')).json();
const REPORTS = { ...real, stubgpu: stubRun };

await page.setRequestInterception(true);
page.on('request', (r) => {
  // Leave no trace in the real progress file.
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  if (r.method() === 'GET' && r.url().includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(REPORTS) });
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const rel = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)) + 1e-15;
const E = 'A decode step on the roofline';
const V = 'Batch slots on a server';
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(120);
  return done;
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(200);
};
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const values = async (plot) => JSON.parse(await attr(`${bench(E)} svg[data-plot="${plot}"] polyline[data-series="model"]`, 'data-values'));
const f32 = precisionFor(4);
const f16 = precisionFor(2);

await goto(page, 'ch22/22.5');
await page.waitForSelector(bench(E));
await page.waitForSelector(bench(V));

// ---- explorer 1: the book's example device, one 4096 -> 4096 projection
const options = await page.$$eval(`${bench(E)} select[aria-label="device"] option`, (os) => os.map((o) => [o.value, o.textContent]));
check("the device list offers the book's example, the stubbed device and custom numbers",
  options[0][0] === 'book' && options[0][1].includes("book's example") && options.some(([v, t]) => v === 'stubgpu' && t.includes('Stub GPU')) && options.at(-1)[0] === 'custom',
  JSON.stringify(options));
await page.select(`${bench(E)} select[aria-label="device"]`, 'book');
await clickText(E, '4096 → 4096');
await clickText(E, '4 (float32)');
await clickText(E, 'none');
let bs = await attr(`${bench(E)} [data-bstar]`, 'data-bstar');
const want32 = bStar(4096, 4096, 20, f32);
check("book's device, 4096 → 4096 float32: B* = 40.8 at the threshold of 20", bs === '40.8' && bs === want32.toFixed(1) && (await attr(`${bench(E)} [data-threshold]`, 'data-threshold')) === '20', bs);
check('the plots mark B* with a dashed line', (await page.$$(`${bench(E)} [data-bstar-line="40.8"]`)).length === 2);
let v = await values('time');
check('time per step is the model\'s at every plotted batch (no padding)', v.every(([b, s]) => rel(s, stepAt(PROJ_4096, b, f32, 1, BOOK_DEVICE).seconds)), JSON.stringify(v.slice(0, 3)));
if (out) await page.screenshot({ path: out + '-explorer-book.png', clip: await page.$eval(bench(E), (e) => { e.scrollIntoView({ block: 'start' }); const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });

await clickText(E, '2 (16-bit)');
bs = await attr(`${bench(E)} [data-bstar]`, 'data-bstar');
const want16 = bStar(4096, 4096, 20, f16);
check('16-bit roughly halves B*: 20.2', bs === want16.toFixed(1) && Number(bs) / 40.8 > 0.45 && Number(bs) / 40.8 < 0.55, bs);
await clickText(E, '4 (float32)');

await clickText(E, '128 rows (M3)');
v = await values('time');
const flat = v.filter(([b]) => b <= 128).map(([, s]) => s);
const at256 = v.find(([b]) => b === 256)[1];
check('with M3\'s 128-row padding the line is flat to 128, then doubles', flat.every((s) => s === flat[0]) && rel(at256, 2 * flat[0])
  && rel(flat[0], stepAt(PROJ_4096, 1, f32, 128, BOOK_DEVICE).seconds), JSON.stringify(v));
check('and at batch 1 the projection is limited by arithmetic, on padding rows', (await text(`${bench(E)} tr[data-piece="proj"]`)).includes('arithmetic'));
await clickText(E, 'none');

// ---- the whole step: pieces and the batch slider
await clickText(E, 'lab 22.2 step');
await setRange(`${bench(E)} input[aria-label="batch"]`, 6);
const shape = m8Shape(REPORTS);
const r64 = stepAt(shape, 64, f32, 1, BOOK_DEVICE);
const pieces = await page.$$eval(`${bench(E)} tr[data-piece]`, (rs) => rs.map((r) => r.dataset.piece));
check('lab 22.2\'s step at batch 64: six pieces, and time per step is the model\'s',
  JSON.stringify(pieces) === JSON.stringify(['proj', 'head', 'rowwise', 'softmax', 'argmax', 'embed']) && rel(Number(await attr(`${bench(E)} [data-step-seconds]`, 'data-step-seconds')), r64.seconds),
  JSON.stringify(pieces));
check('the step\'s B* is the model\'s (all GEMMs together)', (await attr(`${bench(E)} [data-bstar]`, 'data-bstar')) === stepBStar(shape, f32, BOOK_DEVICE).toFixed(1));
check('the current batch is marked on both plots', (await page.$$(`${bench(E)} [data-current="64"]`)).length === 2);
check('no measured points on the book\'s device', (await page.$$(`${bench(E)} [data-measured]`)).length === 0);

// ---- the stubbed device: overlay and knees
await page.select(`${bench(E)} select[aria-label="device"]`, 'stubgpu');
await pause(200);
check('your device: its threshold is lab 5.1\'s peak over bandwidth', (await attr(`${bench(E)} [data-threshold]`, 'data-threshold')) === String(STUB.peak / STUB.bw));
const series = await page.$$eval(`${bench(E)} svg[data-plot="time"] [data-measured]`, (gs) => gs.map((g) => [g.dataset.measured, Number(g.dataset.points)]));
const knees = await page.$$eval(`${bench(E)} svg[data-plot="time"] [data-measured-knee]`, (gs) => gs.map((g) => Number(g.dataset.measuredKnee)));
const kneeM2 = kneeOf(STEP_ROWS.filter((r) => r.gemm === 'm2'));
const kneeM3 = kneeOf(STEP_ROWS.filter((r) => r.gemm === 'm3'));
check('lab 22.2\'s rows appear as three measured series (M2, M3, and the gather up to batch 8), on both plots',
  JSON.stringify(series) === JSON.stringify([['stubgpu:m8:m2', 10], ['stubgpu:m8:m3', 10], ['stubgpu:m8:gemv', 4]]) && (await page.$$(`${bench(E)} svg[data-plot="tokens"] [data-measured]`)).length === 3,
  JSON.stringify(series));
check(`their knees are marked: M2 at ${kneeM2}, M3 at ${kneeM3} (flat to its tile)`, JSON.stringify(knees) === JSON.stringify([kneeM2, kneeM3]) && kneeM3 === 256, JSON.stringify(knees));
const note = await text(`${bench(E)} [data-overlay-note]`);
check('the panel names the series and their knees, and says the gather has none up to 8', note.includes(`lab 22.2, M2 (lab 6.4's tiled kernel)`) && note.includes(`knee at ${kneeM2}`) && note.includes(`knee at ${kneeM3}`)
  && note.includes("lab 22.1's gather") && note.includes('no knee up to 8'), note);
check('the shape comes from the rows: d = 2048, 2 layers', (await text(bench(E))).includes('d = 2048, 2 layers, vocabulary 32,000 (from your rows)'));
if (out) await page.screenshot({ path: out + '-explorer-overlay.png', clip: await page.$eval(bench(E), (e) => { e.scrollIntoView({ block: 'start' }); const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await clickText(E, '2 (16-bit)');
check('measurements are float32: at 16-bit they are hidden and the page says why',
  (await page.$$(`${bench(E)} [data-measured]`)).length === 0 && (await text(bench(E))).includes('Your measurements are float32'));
await clickText(E, '4 (float32)');

// ---- custom numbers
await page.select(`${bench(E)} select[aria-label="device"]`, 'custom');
await pause(150);
await page.$eval(`${bench(E)} input[aria-label="custom peak"]`, (el) => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '5000'); el.dispatchEvent(new Event('input', { bubbles: true })); });
await page.$eval(`${bench(E)} input[aria-label="custom bandwidth"]`, (el) => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '500'); el.dispatchEvent(new Event('input', { bubbles: true })); });
await pause(200);
check('custom numbers: threshold 10, and B* follows', (await attr(`${bench(E)} [data-threshold]`, 'data-threshold')) === '10'
  && (await attr(`${bench(E)} [data-bstar]`, 'data-bstar')) === stepBStar(shape, f32, { peak: 5000, bw: 500 }).toFixed(1));

// Back to the book's device for the server explorer.
await page.select(`${bench(E)} select[aria-label="device"]`, 'book');
await setRange(`${bench(E)} input[aria-label="batch"]`, 0);
await pause(200);

// ---- explorer 2: batch slots on a server
const stepTime = (() => { const t = []; return (a) => (t[a] ??= stepAt(shape, a, f32, 1, BOOK_DEVICE).seconds); })();
const run = (load, mix, slots, seed = 1) => {
  const requests = makeRequests({ count: 1000, rate: load / (stepTime(1) * MIXES[mix].mean), mix, seed });
  return { static: serve({ requests, slots, policy: 'static', stepTime }), continuous: serve({ requests, slots, policy: 'continuous', stepTime }) };
};
const table = () => page.$$eval(`${bench(V)} tr[data-policy]`, (rs) => Object.fromEntries(rs.map((r) => [r.dataset.policy, Object.fromEntries([...r.querySelectorAll('td[data-k]')].map((td) => [td.dataset.k, Number(td.dataset.v)]))])));
let t = await table();
let m = run(64, 'long', 16);
check('heavy load, long-tailed, 16 slots: the table is serving.ts\'s, and continuous beats static on tokens per second at equal slots',
  rel(t.static.tps, m.static.tokensPerSecond) && rel(t.continuous.tps, m.continuous.tokensPerSecond) && rel(t.continuous.short, m.continuous.meanFinishShort)
  && t.continuous.tps > 2 * t.static.tps && t.continuous.short < t.static.short && t.continuous.util > t.static.util,
  JSON.stringify(t));
check('the verdict states the ratio', (await text(`${bench(V)} [data-verdict]`)).includes('continuous batching makes'));
check('the chart shows the continuous run, with no finished request holding a slot', (await attr(`${bench(V)} [data-slot-chart]`, 'data-slot-chart')) === 'continuous' && (await page.$$(`${bench(V)} [data-held]`)).length === 0);
if (out) await page.screenshot({ path: out + '-server-continuous.png', clip: await page.$eval(bench(V), (e) => { e.scrollIntoView({ block: 'start' }); const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await clickText(V, 'static');
check('switching the chart to static shows grey slots held by finished requests', (await attr(`${bench(V)} [data-slot-chart]`, 'data-slot-chart')) === 'static' && (await page.$$(`${bench(V)} [data-held]`)).length > 10);
if (out) await page.screenshot({ path: out + '-server-static.png', clip: await page.$eval(bench(V), (e) => { e.scrollIntoView({ block: 'start' }); const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });

await setRange(`${bench(V)} input[aria-label="slot limit"]`, 8); // 256 slots
const t256 = await table();
await setRange(`${bench(V)} input[aria-label="slot limit"]`, 5); // 32 slots
const t32 = await table();
check('past the knee, raising the limit from 32 to 256 buys little throughput and lengthens the worst time per token',
  t256.continuous.tps / t32.continuous.tps < 1.3 && t256.continuous.worst > 2 * t32.continuous.worst && rel(t256.continuous.tps, run(64, 'long', 256).continuous.tokensPerSecond),
  JSON.stringify({ t32: t32.continuous, t256: t256.continuous }));
await setRange(`${bench(V)} input[aria-label="arrival rate"]`, 0); // a quarter of one slot's capacity
t = await table();
check('under light load the policies make the same tokens per second and the slots are mostly empty',
  Math.abs(t.continuous.tps / t.static.tps - 1) < 0.01 && t.continuous.batch < 4 && (await text(`${bench(V)} [data-verdict]`)).includes('mostly empty'),
  JSON.stringify(t));
await clickText(V, 'similar');
await clickText(V, 'New arrivals');
t = await table();
m = run(0.25, 'short', 32, 2);
check('the length mix and a new draw of arrivals feed the model', rel(t.static.tps, m.static.tokensPerSecond) && rel(t.continuous.tps, m.continuous.tokensPerSecond), JSON.stringify(t));

// ---- the predictions
const predict = (id) => page.evaluateHandle((id) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(id)), id);
let q = await predict('By what factor does throughput change');
await (await q.$('input[type=number]')).type('2.67');
await (await q.$('button.primary')).click();
await pause(150);
let qt = await q.evaluate((e) => e.textContent);
check('p-batch8 reveals 8/3 = 2.67× and the 3× latency per query', qt.includes('You predicted 2.67') && qt.includes('2.67×') && qt.includes('That is what you had.'), qt);
q = await predict('smallest batch at which a 4096');
await (await q.$('input[type=number]')).type('41');
await (await q.$('button.primary')).click();
await pause(150);
qt = await q.evaluate((e) => e.textContent);
check('p-book-knee reveals 41 (B* = 40.8) with the intensity curve', qt.includes('You predicted 41') && qt.includes('B* = 40.8') && qt.includes('Exactly.')
  && (await q.evaluate((e) => e.querySelector('[data-intensity-curve]')?.dataset.intensityCurve)) === '40.8', qt);

// ---- questions (choices clicked by written index; browser.mjs asks for the written order)
const answer = async (needle, index) => {
  const h = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(n)), needle);
  const radios = await h.$$('input[type=radio]');
  await radios[index].click();
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
check('q-who-pays: batch 8 is right, 3 units against 8', (await answer('Eight users are waiting', 0)).startsWith('Correct.'));
check('q-attention: "attention most of all" is wrong, and the explanation cites chapter 20', (await answer('Which parts of a decode step', 2)).includes('Not quite.'));
check('q-plateau: tile padding, not proof of bandwidth', (await answer('flat from batch 1 to 128', 0)).startsWith('Correct.'));
check('q-half-bytes: about half', (await answer('from float32 to 16-bit', 0)).startsWith('Correct.'));
check('q-batched-gemm: no, one GEMM against one shared weight', (await answer('runs many GEMMs in one call', 0)).startsWith('Correct.'));
check('q-continuous, q-cpu-gpu and q-scaling answer correctly', (await answer('continuous (iteration-level) batching matter', 0)).startsWith('Correct.')
  && (await answer('same trade-off as CPU against GPU', 0)).startsWith('Correct.') && (await answer('weak scaling', 0)).startsWith('Correct.'));
check('all eight questions are on the page', (await page.$$eval('.q .prompt', (ps) => ps.length)) === 10);

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await pause(200);
if (out) await page.screenshot({ path: out + '-explorer-light.png', clip: await page.$eval(bench(E), (e) => { e.scrollIntoView({ block: 'start' }); const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width nothing makes the page scroll sideways', narrow);
if (out) {
  await page.$eval(bench(E), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
  await page.$eval(bench(V), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-server.png' });
}

const expected404 = (e) => /Failed to load resource: the server responded with a status of 404/.test(e) || /\/content\/ch22\/narration\/(manifest\.json|[a-z-]+-\d\.mp3)/.test(e);
const bad = errors.filter((e) => !expected404(e));
check('no console or page errors (a missing narration manifest is allowed)', bad.length === 0, bad.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
