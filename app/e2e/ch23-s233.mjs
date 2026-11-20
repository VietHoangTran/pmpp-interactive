// Browser check for section 23.3: the Fig. 23.13 explorer, both predictions, every control
// and preset of the stepper "Copy, compute, copy back", its read-outs for the E1 and E2
// cases, its Metal view, the "your device" preset (with this machine's reports, and with
// none), and two questions. Expected values come from the model itself
// (src/sim/chunkpipe.ts, imported directly).
// Usage: node e2e/ch23-s233.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  DEFAULTS, HALO_BOOK, LANE_NAME, ONE_ENGINE, PRESETS, START, bestChunks, haloLanes, parseLabRows, simulate,
} from '../src/sim/chunkpipe.ts';

const out = process.argv[2];
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const r2 = (v) => Number(Number(v).toFixed(2));

async function session({ noReports = false, width = 1440, dark = false } = {}) {
  const { browser, page, errors } = await open({ width, height: 1000, dark });
  await page.setRequestInterception(true);
  page.on('request', (r) => {
    // Leave no trace in the real progress file.
    if (r.method() === 'PUT' && r.url().includes('/api/progress'))
      return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    if (noReports && r.url().includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: '{}' });
    return r.continue();
  });
  await goto(page, 'ch23/23.3');
  await page.waitForSelector('[data-stepper]');
  await page.waitForSelector('[data-stepper] .code pre');
  return { browser, page, errors };
}

const S = 'section.bench[aria-label="Copy, compute, copy back"]';
const { browser, page, errors } = await session();

const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const readout = (k) => text(`${S} [data-out="${k}"]`);
const num = async (k) => Number((await readout(k)).replace(/[^0-9.\-]/g, ''));
const clickIn = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const b = [...root.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const seg = async (name, v) => {
  await page.$eval(`${S} [data-seg="${name}"] button[data-v="${v}"]`, (b) => b.click());
  await pause(80);
};
const segDisabled = (name) => page.$eval(`${S} [data-seg="${name}"] button`, (b) => b.disabled);
const setRange = async (scope, aria, v) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setCfg = async (o) => {
  const map = { n: 'Number of chunks', streams: 'Number of streams', a: 'Total copy-in time', b: 'Total kernel time', c: 'Total copy-out time', overhead: 'Per-operation cost' };
  for (const [k, v] of Object.entries(o)) {
    if (map[k]) await setRange(S, k === 'streams' && (await page.$(`${S} input[aria-label="Buffers in flight"]`)) ? 'Buffers in flight' : map[k], v);
    else if (k === 'queues') await seg('queues', v === Infinity ? 'inf' : String(v));
    else await seg(k, String(v));
  }
};
const narration = () => text(`${S} [data-narrate]`);
const clock = () => text(`${S} [data-out="clock"]`);
const totalIs = async (cfg, name) => {
  const want = simulate(cfg).total;
  const got = await num('total');
  check(name, Math.abs(got - r2(want)) < 0.006, `got ${got}, model ${want}`);
};

// ---- the listing: the checked snippets, three tabs
const tabs = await page.$$eval('.prose > figure.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('the listing has CUDA, Metal host and Metal kernel tabs', ['CUDA', 'Metal host', 'Metal kernel'].every((t) => tabs.includes(t)), tabs.join('|'));
const listing = await text('.prose > figure.code pre');
check('the CUDA tab is labs/ch23/snippets/staged_loop.cu', listing.includes('void stagedBreadthFirst(const Job& job)') && listing.includes('cudaMemcpyHostToDevice'));
check('its status uses the fixed vocabulary', (await text('.prose > figure.code .tabs .status')).startsWith('emulator-checked and front-end-checked'));
const tab = async (label) => {
  await page.evaluate((label) => [...document.querySelectorAll('.prose > figure.code .tabs button')].find((b) => b.textContent.trim() === label).click(), label);
  await pause(80);
  return [await text('.prose > figure.code pre'), await text('.prose > figure.code .tabs .status')];
};
let [code, status] = await tab('Metal host');
check('the Metal host tab is staged_loop_metal.mm, verified on Metal', code.includes('[last[k] waitUntilCompleted];') && status === 'verified on Metal', status);
[code, status] = await tab('Metal kernel');
check('the Metal kernel tab is staged_loop.metal', code.includes('kernel void chunkKernel') && status === 'verified on Metal');
await tab('CUDA');

// ---- Fig. 23.13 explorer
const H = 'section.bench[aria-label="Two modes, then one"]';
const halo = async () => page.$$eval(`${H} [data-halo]`, (g) => g.map((e) => [e.dataset.halo, Number(e.dataset.total)]));
let h = Object.fromEntries(await halo());
check("Fig. 23.13 in the book's proportions: 18 without overlap, 12 with", h.plain === haloLanes(HALO_BOOK, false).total && h.overlap === haloLanes(HALO_BOOK, true).total, JSON.stringify(h));
check('and the exchange is hidden', (await text(`${H} [data-out="halo-verdict"]`)).includes('the communication is hidden'));
await setRange(H, 'Time of each halo exchange', 4);
h = Object.fromEntries(await halo());
const h4 = haloLanes({ ...HALO_BOOK, exchange: 4 }, true);
check('a longer exchange shows through by the model\'s amount', h.overlap === h4.total && (await text(`${H} [data-out="halo-verdict"]`)).includes(`run ${h4.exposed} past the interior`), JSON.stringify(h));
check('the norm reduction is drawn last in both diagrams', (await page.$$eval(`${H} [data-bar="reduce L2 norm"]`, (b) => b.length)) === 2);

// ---- the stepper's starting case: not the prediction's
await totalIs(START, `the stepper starts on its own case (${simulate(START).total}), not the prediction's 18`);
check('serial read-out', (await num('serial')) === simulate(START).serial);
check('nothing has run yet', (await narration()).startsWith('Nothing has run yet. 12 operations'));
await clickIn(S, 'Step');
let n = await narration();
check('step 1: the host queues everything at 0 and the first copy starts', n.includes('t = 0.') && n.includes('The host thread queues all 12 operations at time 0') && n.includes("Chunk 0's copy-in starts on the copy-in engine"), n);
const hl = () => page.$$eval(`${S} .code .ln.hl`, (l) => l.map((x) => Number(x.dataset.n)));
check('the snippet lines that issued it are marked (47 and 35)', JSON.stringify(await hl()) === JSON.stringify([35, 47]), JSON.stringify(await hl()));
await clickIn(S, 'Step');
const s0 = simulate(START);
check('step 2 moves to the next event time', (await clock()).startsWith(`t = ${r2(s0.steps[1].t)} `), await clock());
check('the operation drawn as current is the one the step started', (await page.$$eval(`${S} [data-op]`, (g) => g.length)) >= 2);
await clickIn(S, 'Back');
check('Back returns to the previous event', (await clock()).startsWith('t = 0 '), await clock());
await clickIn(S, 'Run to the end');
n = await narration();
check('run to the end: all done at the total', n.includes(`All done at ${r2(s0.total)}`), n);
const drawn = await page.$$eval(`${S} [data-op]`, (g) => g.map((e) => e.dataset.state));
check('every operation is drawn, and finished', drawn.length === s0.ops.length && drawn.every((x) => x === 'done'), JSON.stringify(drawn));
check('the busiest lane is named', (await readout('bottleneck')).startsWith(LANE_NAME[s0.bottleneck]));
check('the bubble read-out is the model\'s', (await num('bubble')) === r2(s0.bubble));
await clickIn(S, 'Restart');
check('Restart clears the timeline', (await narration()).startsWith('Nothing has run yet') && (await page.$$(`${S} [data-op]`)).length === 0);
await clickIn(S, 'Play');
await pause(1500);
const played = await clock();
await clickIn(S, 'Pause');
check('Play advances by itself', !played.startsWith('t = · '), played);
await setRange(S, 'Time', 500);
check('the time scrubber moves the clock', (await clock()).startsWith(`t = ${r2(Math.max(s0.total, s0.serial) / 2)} `), await clock());
if (out) await page.screenshot({ path: out + '-stepper-start.png' });

// ---- p-pipeline
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How long does the pipelined job take?')));
await (await pq.$('input[type=number]')).type('18');
await (await pq.$('button.primary')).click();
await pause(150);
check('p-pipeline: the reveal gives the model\'s 18', (await pq.evaluate((e) => e.textContent)).includes('The model gives 18'));
await page.$eval('[data-load="p-pipeline"]', (b) => b.click());
await pause(150);
const pPipe = PRESETS.find((p) => p.id === 'p-pipeline').cfg;
await totalIs(pPipe, 'loading it into the stepper shows 18');
check('E1 holds for it and gives 18', (await page.$eval(`${S} [data-form="e1"]`, (e) => [e.dataset.holds, e.dataset.value].join())) === 'yes,18');
check('speed-up 32 / 18', (await readout('speedup')) === `${r2(32 / 18)}×`);

// ---- presets
for (const p of PRESETS) {
  await page.$eval(`${S} [data-preset="${p.id}"]`, (b) => b.click());
  await pause(100);
  await totalIs(p.cfg, `preset "${p.label}" gives the model's total`);
}
await page.$eval(`${S} [data-preset="best-practices"]`, (b) => b.click());
await pause(100);
check("Best Practices example: NVIDIA's estimate holds and equals the total", (await page.$eval(`${S} [data-form="nv"]`, (e) => [e.dataset.holds, e.dataset.value].join())) === 'yes,18');

// ---- streams: 12 chunks of 1, 1, 1
await page.$eval(`${S} [data-preset="equal"]`, (b) => b.click());
await pause(100);
await setCfg({ n: 12, a: 12, b: 12, c: 12, streams: 2 });
const two = { ...DEFAULTS, n: 12, a: 12, b: 12, c: 12, streams: 2 };
await totalIs(two, '12 chunks of 1, 1, 1 with 2 streams: 19');
check('with 2 streams E1 does not apply', (await page.$eval(`${S} [data-form="e1"]`, (e) => e.dataset.holds)) === 'no');
for (let i = 0; i < 12; i++) {
  if ((await clock()).startsWith('t = 2 ')) break;
  await clickIn(S, 'Step');
}
n = await narration();
check("at t = 2 the narration says why chunk 2's copy-in waits", n.includes("Chunk 2's copy-in waits: stream 0 is still copying out chunk 0."), n);
await setCfg({ streams: 3 });
await totalIs({ ...two, streams: 3 }, 'and 14 with 3 streams');
check('with 3 streams E1 holds', (await page.$eval(`${S} [data-form="e1"]`, (e) => e.dataset.holds)) === 'yes');
await setCfg({ streams: 1 });
await totalIs({ ...two, streams: 1 }, 'one stream: the one-piece time, 36');
if (out) await page.screenshot({ path: out + '-stepper-streams.png' });

// ---- per-operation cost: an optimum chunk count
await setCfg({ streams: 16, overhead: 0.25 });
const costly = { ...two, streams: 16, overhead: 0.25 };
const best = bestChunks(costly);
check(`with a cost of 0.25 the fastest chunk count is the model's (n = ${best.n})`, (await readout('best-n')).startsWith(`n = ${best.n} `), await readout('best-n'));
await setCfg({ overhead: 0 });

// ---- queues: breadth first, 8 streams on 4 queues
await setCfg({ n: 8, a: 8, b: 8, c: 8, streams: 8, order: 'breadth' });
const q = { ...DEFAULTS, n: 8, a: 8, b: 8, c: 8, streams: 8, order: 'breadth' };
await totalIs(q, 'breadth first, 8 streams, unlimited queues: 10');
await setCfg({ queues: 4 });
await totalIs({ ...q, queues: 4 }, "HIP's 4 queues serialise unrelated streams: 13");
await setCfg({ queues: 8 });
await totalIs({ ...q, queues: 8 }, "CUDA's 8 queues do not bind with 8 streams: 10");
await setCfg({ queues: Infinity, order: 'depth' });

// ---- p-oneengine
const oq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Before you switch the stepper to one copy engine')));
await (await oq.$('input[type=number]')).type('10');
await (await oq.$('button.primary')).click();
await pause(150);
const oText = await oq.evaluate((e) => e.textContent);
check('p-oneengine: the reveal gives 16, and 10 breadth first', oText.includes('16: no gain at all') && oText.includes('takes 10'), oText);
await page.$eval('[data-load="one-depth"]', (b) => b.click());
await pause(150);
await totalIs(ONE_ENGINE, 'one engine, chunk by chunk: 16');
check('the lanes show one shared copy engine', (await page.$$eval(`${S} [data-timeline] [data-lane="copy"] text`, (t) => t.map((x) => x.textContent))).includes(LANE_NAME.copy));
await clickIn(S, 'Run to the end');
await setRange(S, 'Time', 1);
for (let i = 0; i < 3; i++) await clickIn(S, 'Back');
await clickIn(S, 'Restart');
await clickIn(S, 'Step');
n = await narration();
check("the narration explains chunk 1's copy-in waiting behind chunk 0's copy-out", n.includes("queued ahead of it, has not started (it waits for chunk 0's kernel)"), n);
await page.$eval('[data-load="one-breadth"]', (b) => b.click());
await pause(150);
await totalIs({ ...ONE_ENGINE, order: 'breadth' }, 'one engine, all copy-ins first: 10');
await seg('engines', '2');
await totalIs({ ...ONE_ENGINE, order: 'breadth', engines: 2 }, 'two engines: 10 either way');
await seg('engines', '1');

// ---- pageable
await page.$eval(`${S} [data-preset="equal"]`, (b) => b.click());
await pause(100);
await seg('host', 'pageable');
const eq = { ...PRESETS.find((p) => p.id === 'equal').cfg, host: 'pageable' };
await totalIs(eq, 'pageable: the one-piece time');
check('pageable: the host thread is the busiest lane', (await readout('bottleneck')).startsWith('Host thread'));
await clickIn(S, 'Run to the end');
check('pageable: the host lane shows blocked and staging spans', (await page.$$eval(`${S} [data-host]`, (g) => g.map((e) => e.dataset.host))).filter((k) => k === 'staging').length === simulate(eq).ops.filter((o) => o.kind !== 'kernel').length);
if (out) await page.screenshot({ path: out + '-stepper-pageable.png' });
await seg('host', 'pinned');

// ---- Metal view
await seg('view', 'metal');
const mcfg = { ...PRESETS.find((p) => p.id === 'equal').cfg, view: 'metal' };
await totalIs(mcfg, 'Metal view: the model\'s total');
check('Metal view: lanes are the CPU and the GPU', JSON.stringify(await page.$$eval(`${S} [data-timeline] g[data-lane] > text`, (t) => t.map((x) => x.textContent))) === JSON.stringify([LANE_NAME.cpu, LANE_NAME.gpu]));
check('Metal view: S becomes K, buffers in flight', !!(await page.$(`${S} input[aria-label="Buffers in flight"]`)));
check('Metal view: engines, order, host memory and queues are off', (await segDisabled('engines')) && (await segDisabled('order')) && (await segDisabled('host')) && (await segDisabled('queues')));
check('Metal view: a lower bound instead of a closed form', (await readout('lower-bound')).includes(String(r2(simulate(mcfg).lowerBound))));
check('Metal view: the code is the Metal host file', (await text(`${S} .code .tabs`)).includes('Metal host') && (await text(`${S} .code pre`)).includes('waitUntilCompleted'));
await setCfg({ streams: 1 });
await totalIs({ ...mcfg, streams: 1 }, 'Metal, K = 1: nothing overlaps (the one-piece time)');
// GPU-bound, so the CPU has to wait for a buffer to come free
await setCfg({ a: 4, b: 16, c: 4, n: 8, streams: 2 });
await totalIs({ ...mcfg, a: 4, b: 16, c: 4, n: 8, streams: 2 }, 'Metal, K = 2, GPU-bound: the model\'s total');
await clickIn(S, 'Restart');
for (let i = 0; i < 40; i++) {
  if ((await narration()).includes('waitUntilCompleted')) break;
  await clickIn(S, 'Step');
}
check('Metal, K = 2: the CPU waits in waitUntilCompleted for a busy buffer', (await narration()).includes('The CPU waits in waitUntilCompleted: buffer'), await narration());
check('and the marked lines are the Metal file\'s', (await hl()).every((x) => x >= 21 && x <= 46) && (await hl()).length > 0, JSON.stringify(await hl()));
if (out) await page.screenshot({ path: out + '-stepper-metal.png' });
await seg('view', 'cuda');

// ---- the reader's device: this machine's reports
const reports = await (await fetch(BASE + '/api/reports')).json();
const devs = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) => {
  const rows = reports[b].results.filter((r) => r.lab === 'ch23/lab1_overlap' && r.variant === 'solution').flatMap((r) => r.bench);
  const view = b === 'metal' ? 'metal' : 'cuda';
  const got = parseLabRows(rows, view, b);
  return got.parts ? [{ b, view, ...got }] : [];
});
const devButtons = await page.$$eval(`${S} [data-preset^="device-"]`, (b) => b.map((x) => x.dataset.preset));
if (devs.length) {
  check('the "your device" preset is offered for each backend with lab 23.1 rows', devs.every((d) => devButtons.includes('device-' + d.b)), JSON.stringify(devButtons));
  const d = devs[0];
  await page.$eval(`${S} [data-preset="device-${d.b}"]`, (b) => b.click());
  await pause(200);
  const dcfg = { ...DEFAULTS, a: d.parts.a, b: d.parts.b, c: d.parts.c, view: d.view, engines: d.parts.engines ?? 2, n: 16, streams: 4 };
  await totalIs(dcfg, `your device (${d.b}): the model on the measured parts`);
  const pts = await page.$$eval(`${S} [data-device] circle[data-measured]`, (c) => c.map((e) => [Number(e.dataset.measured), Number(e.dataset.units)]));
  const want = d.staged.filter((r) => r.streams === 4 && (d.view === 'metal' || (r.order === 'depth' && r.host === 'pinned')));
  check('the lab\'s measurements for the same settings are plotted', pts.length === want.length && want.every((w) => pts.some(([c, u]) => c === w.chunks && Math.abs(u - w.units) < 1e-6)), JSON.stringify({ pts, want: want.length }));
  if (out) await page.screenshot({ path: out + '-stepper-device.png' });
} else check('with no lab 23.1 rows there is no device preset', devButtons.length === 0, JSON.stringify(devButtons));

// ---- questions
const ask = async (needle, act) => {
  const qh = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await act(qh);
  await (await qh.$('button.primary')).click();
  await pause(120);
  return qh.evaluate((e) => e.textContent);
};
let v = await ask('How long does it take, in ms?', async (qh) => (await qh.$('input[type=number]')).type('15'));
check('q-bottleneck accepts 15 ms', v.includes('Correct.'), v.slice(0, 200));
v = await ask('queue all of them in one stream', async (qh) => (await qh.$$('input[type=radio]'))[0].click());
check('q-onestream: "none" is right', v.includes('Correct.'));
v = await ask('what plays the role of the copy-in', async (qh) => (await qh.$$('input[type=radio]'))[1].click());
check('q-metal: a blit pass on a copy engine is wrong, with a reason', v.includes('Not quite.') && v.includes('not as separate copy hardware'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();

// ---- with no reports at all, no device preset and no comparison
{
  const { browser: b2, page: p2, errors: e2 } = await session({ noReports: true });
  check('with no reports: no device preset', (await p2.$$(`${S} [data-preset^="device-"]`)).length === 0);
  check('with no reports: no device comparison', (await p2.$$(`${S} [data-device]`)).length === 0);
  check('no errors without reports', e2.length === 0, e2.join(' | '));
  await b2.close();
}

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
