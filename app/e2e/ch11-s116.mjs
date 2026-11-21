// Browser check for section 11.6: the "Three phases" explorer, "One warp, two ways to load",
// the staging prediction and its measured reveal, the narrated scene "Three phases of a
// coarsened scan", the listings and two questions. Expected values come from the models,
// src/sim/coarsescan.ts and src/sim/warpscan.ts, imported inside the page through the app's
// own module server (coarsescan.ts imports its neighbours without file extensions, which
// Node cannot resolve on its own).
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-s116.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
let labMisses = 0;
page.on('response', (r) => { if (r.status() === 404 && r.url().includes('/api/lab-file')) labMisses++; });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setRange = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const el = document.querySelector(`${scope} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};
/** Run a function of the models inside the page. */
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = { ...(await import('/src/sim/coarsescan.ts')), ...(await import('/src/sim/warpscan.ts')) };
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);

await goto(page, 'ch11/11.6');
await page.waitForSelector(bench('Three phases'));

// ---------------------------------------------------------------- Three phases
const T = bench('Three phases');
const ph = await model((m) => m.threePhases(m.demoValues(16), 4));
const runs = await page.$$eval(`${T} [data-run]`, (els) => els.map((e) => e.dataset.run));
check('4 threads of 4: phase 1 is Fig. 11.6\'s scanned segments', JSON.stringify(runs) === JSON.stringify(ph.runs.map((r) => r.join(' '))), JSON.stringify(runs));
const adds = await page.$$eval(`${T} [data-added]`, (els) => els.map((e) => Number(e.dataset.added)));
check('phase 3 adds 0, 7, 14, 20', JSON.stringify(adds) === JSON.stringify(ph.added));
const finals = (await page.$$eval(`${T} [data-final]`, (els) => els.map((e) => e.dataset.final))).join(' ');
check('the result is the scan', finals === ph.final.join(' ') && (await text(`${T} [data-phase2]`)).includes('the result is the scan'));
const k = (key) => page.$eval(`${T} [data-k="${key}"]`, (e) => Number(e.textContent.replace(/,/g, '')));
check('counts at N = 16, P = 4: 32 additions, 9 steps; Kogge-Stone 49', (await k('work')) === 32 && (await k('steps')) === 9 && (await k('ks-work')) === 49);
await clickText(T, 'factor 1 (P = N)');
const pn = await model((m) => [m.coarsenedWork(1024, 1024).total, m.coarsenedSteps(1024, 1024).total, m.koggeStoneWork(1024)]);
check('factor 1: P = N = 1024, the model\'s counts', (await k('work')) === pn[0] && (await k('steps')) === pn[1] && (await k('ks-work')) === pn[2], `${await k('work')} ${await k('steps')}`);
check('too large to draw: no phases table', (await page.$(`${T} [data-phases]`)) === null);
await setRange(T, 'BLOCK_DIM', 3); // 32 threads
await setRange(T, 'COARSE_FACTOR', 16);
const m32 = await model((m) => [m.coarsenedWork(512, 32).total, m.coarsenedSteps(512, 32).total]);
check('P = 32, F = 16: N = 512 and the model\'s counts', (await text(`${T} [data-n]`)) === '512' && (await k('work')) === m32[0] && (await k('steps')) === m32[1]);
await clickText(T, 'Fig. 11.6\'s sixteen');
if (out) await page.screenshot({ path: out + '-three.png' });

// ---------------------------------------------------------------- One warp, two ways to load
const L = bench('One warp, two ways to load');
const tx = (mode) => page.$eval(`${L} [data-readout="${mode}"]`, (e) => Number(e.dataset.transactions));
const onCount = (mode) => page.$$eval(`${L} [data-mode="${mode}"] rect[data-on="1"]`, (els) => els.map((e) => Number(e.dataset.el)));
check('factor 4, c = 0: 4 sectors staged, 16 direct', (await tx('staged')) === 4 && (await tx('direct')) === 16, `${await tx('staged')} ${await tx('direct')}`);
await clickText(L, 'c + 1');
const st1 = await onCount('staged');
const want1 = await model((m) => [m.warpLoad('staged', 1, 4, 32, 32, 32).elements, m.warpLoad('direct', 1, 4, 32, 32, 32).elements]);
check('c = 1: staged reads elements 32 to 63', JSON.stringify(st1) === JSON.stringify(want1[0]));
check('c = 1: direct reads 1, 5, 9, …', JSON.stringify(await onCount('direct')) === JSON.stringify([...want1[1]].sort((a, b) => a - b)));
await page.click(`${L} [data-mode="staged"] rect[data-el="70"]`);
await pause(100);
const who = await model((m) => [m.loaderOf('staged', 70, 32, 4), m.loaderOf('direct', 70, 32, 4), m.ownerOf(70, 4)]);
const pick = await text(`${L} [data-pick]`);
check('clicking element 70 names its loaders and its owner', pick.includes(`thread ${who[0].thread} loads it in iteration ${who[0].iter}`) && pick.includes(`thread ${who[1].thread} would, in iteration ${who[1].iter}`) && pick.includes(`thread ${who[2]} scans it`), pick);
await clickText(L, 'F = 8');
check('factor 8: direct touches 32 sectors, one per lane', (await tx('direct')) === 32 && (await tx('staged')) === 4);
await page.evaluate((L) => {
  const s = document.querySelector(`${L} select`);
  s.value = 'amd128';
  s.dispatchEvent(new Event('change', { bubbles: true }));
}, L);
await pause(120);
check('AMD 128-byte lines: a chunk is one line', (await tx('staged')) === 1);
if (out) await page.screenshot({ path: out + '-load.png' });

// ---------------------------------------------------------------- prediction with a measured reveal
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const pred = await q('how many times faster will the staged one be');
await (await pred.$('input[type=number]')).type('2');
await (await pred.$('button.primary')).click();
await pause(400);
const reveal = await pred.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
const reports = await (await fetch(BASE + '/api/reports')).json();
const rows = Object.entries(reports).filter(([b]) => b !== 'emu').flatMap(([, r]) => r.results.filter((x) => x.lab === 'ch11/lab2_coarsened' && x.variant === 'solution').flatMap((x) => x.bench));
const rt = rows.find((r) => r.kernel === 'register_tiled' && r.block === 256 && r.coarse === 4);
const tm = rows.find((r) => r.kernel === 'thread_major' && r.block === 256 && r.coarse === 4);
if (rt && tm) {
  const measured = await pred.$eval('[data-measured]', (e) => Number(e.dataset.measured));
  check('reveal: the measured ratio is the reports\' thread_major / register_tiled', Math.abs(measured - tm.seconds / rt.seconds) < 1e-9, String(measured));
} else check('reveal: no measurement yet, says how to get one', reveal.includes('No measurement on this machine yet'), reveal);
check('reveal repeats the prediction', reveal.includes('You predicted 2'));

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="Three phases of a coarsened scan"]';
await page.waitForSelector(S);
const labels = await page.$$eval(`${S} .seg button`, (b) => b.map((x) => x.textContent.trim()));
check('scene: six parts', JSON.stringify(labels) === JSON.stringify(['1. The block', '2. Load', '3. Phase 1', '4. Phase 2', '5. Phase 3', '6. Store']), labels.join('|'));
const manifest = await (await fetch(BASE + '/content/ch11/narration/manifest.json')).json();
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const goBeat = async (n, p) => {
  await page.evaluate((S, n) => [...document.querySelectorAll(`${S} .seg button`)][n - 1].click(), S, n);
  await pause(80);
  await page.evaluate((S, v) => {
    const el = document.querySelector(`${S} input[type=range]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, S, Math.round(p * 1000));
  await pause(120);
};
const stage = () => page.$eval(`${S} .stage`, (e) => e.textContent.replace(/\s+/g, ' '));
for (let n = 1; n <= 6; n++) {
  await goBeat(n, 0.5);
  const script = strip(await (await fetch(`${BASE}/content/ch11/narration/coarse-phases-${n}.txt`)).text());
  const hasAudio = !!manifest?.clips?.[`coarse-phases-${n}`];
  if (!hasAudio) check(`part ${n}: no audio, so its script is shown`, (await text(`${S} .caption`)) === script);
}
const d = await model((m) => m.sceneData());
await goBeat(2, 0.3);
check('part 2: load instruction 2 reads one contiguous piece of 4', (await stage()).includes('load instruction 2 of 4') && (await stage()).includes('1 contiguous piece'));
await goBeat(2, 0.95);
check('part 2: the strided alternative splits it into 4 pieces', (await stage()).includes(`${d.direct[0].runs} separate pieces`));
await goBeat(4, 0.98);
check('part 4: the scanned totals 7, 14, 20, 31', (await stage()).includes(d.phases.scannedTotals.join(', ')));
await goBeat(6, 0.98);
const kw = await model((m) => m.koggeStoneWork(16));
check('part 6: 4 accesses per element, 32 additions, Kogge-Stone 49', (await stage()).includes(`${d.perElement} shared accesses per element; ${d.work} additions, Kogge-Stone ${kw}`), await stage());
// captions are an option and are remembered
const capBox = async () => page.evaluateHandle((S) => [...document.querySelectorAll(`${S} label.check`)].find((l) => l.textContent.includes('captions')).querySelector('input'), S);
await (await capBox()).click();
await pause(100);
check('captions switch is remembered in the browser', (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');
await (await capBox()).click();
// parts advance by themselves with the sound off
await page.evaluate((S) => [...document.querySelectorAll(`${S} label.check`)].find((l) => l.textContent.includes('sound')).querySelector('input').click(), S);
await goBeat(1, 0.995);
await clickText(S, 'Resume') || (await clickText(S, 'Play'));
await pause(1500);
const pressed = await page.$$eval(`${S} .seg button`, (b) => b.findIndex((x) => x.getAttribute('aria-pressed') === 'true'));
check('part 1 runs into part 2 by itself', pressed >= 1, String(pressed));
await clickText(S, 'Pause');
if (out) await page.screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- listings
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('Fig. 11.12 as printed', tabs.includes('Fig. 11.12 as printed'), tabs.join('|'));
check('listing is the printed kernel with its 35 lines', (await page.$$('.code pre .ln')).length === 35);
check('lab 11.2 tabs once the lab exists', tabs.includes('lab 11.2 (CUDA and HIP)') && tabs.includes('lab 11.2 (Metal)'), tabs.join('|'));

// ---------------------------------------------------------------- questions
const owner = await q('Which thread later scans that element?');
await (await owner.$('input[type=number]')).type('129');
await (await owner.$('button.primary')).click();
await pause(120);
check('q-owner accepts 129', (await owner.evaluate((e) => e.textContent)).includes('Correct.'));
const sectors = await q('How many 32-byte sectors does it touch?');
await (await sectors.$('input[type=number]')).type('16');
await (await sectors.$('button.primary')).click();
await pause(120);
check('q-sectors accepts 16', (await sectors.evaluate((e) => e.textContent)).includes('Correct.'));

const forgiven = errors.filter((e) => !(e.includes('404') && labMisses-- > 0));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch11/11.6', [
  { label: 'lab 11.2 (CUDA and HIP)', lab: 'ch11/lab2_coarsened', file: 'kernel.cu' },
  { label: 'lab 11.2 (Metal)', lab: 'ch11/lab2_coarsened', file: 'kernel.metal' },
]);

check('no console or page errors', forgiven.length === 0, forgiven.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
