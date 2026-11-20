// Browser check for section 15.2: the prediction that opens the intensity explorer, the
// explorer itself (8, 32, k-invariance, m = 1, bytes per element, thresholds), the grid
// explorer, the narrated scene "One wide block instead of four narrow ones" (parts,
// captions switch, drawings against the model) and the questions. Values come from
// src/sim/gemmreuse.ts, imported here directly.
// It passes with or without content/ch15/narration/manifest.json.
// Usage: node e2e/ch15-s152.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { gridFetches, intensity, regionCounts, tilePair } from '../src/sim/gemmreuse.ts';

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
const close = (a, b, rel = 1e-6) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));
const flat = (s) => s.replace(/\s+/g, ' ').trim();
const IB = 'section.bench[aria-label="Intensity of an output tile"]';
const GB = 'section.bench[aria-label="Fetches across the grid"]';
const SC = 'section.scene[aria-label="One wide block instead of four narrow ones"]';

const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};
const setNumber = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};
const clickIn = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label || x.textContent.trim().startsWith(label));
    if (!b) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(100);
  return ok;
};
const outVal = (scope, key) => page.$eval(`${scope} [data-out="${key}"]`, (e) => e.dataset.value ?? e.textContent.trim());
const intensityShown = async () => Number(await page.$eval(`${IB} [data-out="intensity"]`, (e) => e.dataset.value));
const regimeShown = () => page.$eval(`${IB} [data-regime]`, (e) => e.dataset.regime);
const done = (id) => page.$eval(`${IB} [data-check="${id}"]`, (e) => e.dataset.done === '1');
const setTile = async (m, n, k) => {
  if (m != null) await setRange(`${IB} input[aria-label="m"]`, m);
  if (n != null) await setRange(`${IB} input[aria-label="n"]`, n);
  if (k != null) await setRange(`${IB} input[aria-label="k"]`, k);
};

await goto(page, 'ch15/15.2');

// ---------------------------------------------------------------- prediction opens the explorer
check('the explorer is closed until a prediction is committed', !!(await page.$(`${IB} [data-gated]`)));
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('128 × 128 output tile with 4-byte')));
await (await pq.$('input[type=number]')).type('30');
await (await pq.$('button.primary')).click();
await pause(200);
const reveal = await pq.evaluate((e) => e.textContent);
check('the reveal gives 32 FLOP/B and remembers the prediction', reveal.includes('You predicted 30') && reveal.includes('32 FLOP/B'), flat(reveal));
await page.waitForSelector(`${IB} [data-out="intensity"]`);
check('the explorer opens after the commit', !(await page.$(`${IB} [data-gated]`)));

// ---------------------------------------------------------------- the book's two numbers
check('the explorer starts at 64 x 64: 16 FLOP/B, no checkpoint reached yet', close(await intensityShown(), intensity(64, 64)) && !(await done('eight')) && !(await done('thirtytwo')));
await setTile(32, 32, null);
check('32 x 32 in float32 shows 8 FLOP/B', close(await intensityShown(), intensity(32, 32)) && intensity(32, 32) === 8);
check('8 is memory-bound against the book\'s example threshold', (await regimeShown()) === 'memory-bound');
check('checkpoint: the book\'s 8 is reached', await done('eight'));
await setTile(128, 128, null);
check('128 x 128 shows 32 FLOP/B', close(await intensityShown(), 32));
check('32 is compute-bound against 20', (await regimeShown()) === 'compute-bound');
check('checkpoint: the book\'s 32 is reached', await done('thirtytwo'));
const t8 = tilePair(128, 128, 8);
check('operations and bytes per pair are the model\'s', Number((await outVal(IB, 'ops')).replace(/,/g, '')) === t8.ops && Number((await outVal(IB, 'bytes')).replace(/,/g, '')) === t8.traffic);
check('shared memory for the pair at k = 8 is 8 KB', (await outVal(IB, 'smem')) === '8 KB', await outVal(IB, 'smem'));
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}.png` });
};
await shot('explorer-32', IB);

// ---------------------------------------------------------------- k cancels
let same = true;
for (const k of [16, 32, 64, 1]) {
  await setTile(null, null, k);
  same &&= close(await intensityShown(), 32);
}
check('dragging k to 16, 32, 64 and 1 leaves 32 FLOP/B', same);
await setTile(null, null, 16);
check('at k = 16 the pair needs 16 KB and K = 4096 takes 256 pairs', (await outVal(IB, 'smem')) === '16 KB' && (await outVal(IB, 'pairs')) === '256');
check('checkpoint: k tried three times without effect', await done('kfree'));

// ---------------------------------------------------------------- other shapes
await setTile(128, 64, null);
check('128 x 64 shows 21.33', close(await intensityShown(), intensity(128, 64)) && Math.abs(intensity(128, 64) - 21.333) < 0.001);
await setTile(59, 128, null);
check('59 x 128 reaches the threshold', (await regimeShown()) === 'compute-bound' && close(await intensityShown(), intensity(59, 128)));
await setTile(58, 128, null);
check('58 x 128 does not', (await regimeShown()) === 'memory-bound');
await setTile(1, 128, null);
check('m = 1 stays below 0.5', (await intensityShown()) < 0.5 && close(await intensityShown(), intensity(1, 128)));
check('checkpoint: m = 1 reached', await done('one'));
await setTile(1, 256, null);
check('m = 1 with the widest tile is still below 0.5', (await intensityShown()) < 0.5);
await setTile(128, 128, null);
await clickIn(IB, '1 (8-bit)');
check('1-byte elements give 128 FLOP/B for 128 x 128', close(await intensityShown(), intensity(128, 128, 1)) && intensity(128, 128, 1) === 128);
await clickIn(IB, '0.5 (4-bit)');
check('4-bit elements give 256', close(await intensityShown(), 256));
await clickIn(IB, '4 (float32)');

// ---------------------------------------------------------------- thresholds
const reports = await (await fetch(BASE + '/api/reports')).json();
const measured = Object.entries(reports).filter(([b]) => b !== 'emu' && b !== 'race').flatMap(([b, run]) => {
  const r = run.results.filter((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution').flatMap((x) => x.bench).find((x) => x.name === 'roofline');
  return r && r.compute_gflops && r.bandwidth_gb_per_s ? [{ b, th: r.compute_gflops / r.bandwidth_gb_per_s }] : [];
});
const options = await page.$$eval(`${IB} select[aria-label="threshold"] option`, (os) => os.map((o) => o.value));
check('threshold menu offers the book\'s example, each measured device and a custom number',
  options[0] === 'book' && options.at(-1) === 'custom' && measured.every((m) => options.includes(m.b)), JSON.stringify(options));
if (measured.length) {
  const { b, th } = measured[0];
  await page.select(`${IB} select[aria-label="threshold"]`, b);
  await pause(80);
  // Find a tile on each side of the reader's own threshold with the model, and check the chip.
  const below = [8, 16, 32, 64, 128].find((s) => intensity(s, s) < th);
  const above = [256, 128, 64, 32].find((s) => intensity(s, s) >= th);
  if (below) { await setTile(below, below, null); check(`your ${b} threshold: ${below} x ${below} is memory-bound`, (await regimeShown()) === 'memory-bound'); }
  if (above) { await setTile(above, above, null); check(`your ${b} threshold: ${above} x ${above} is compute-bound`, (await regimeShown()) === 'compute-bound'); }
  const shown = await page.$eval(`${IB} [data-shape]`, (e) => e.textContent);
  check('the threshold shown is lab 5.1\'s compute over bandwidth', shown.includes(String(Number(th.toPrecision(3)))), shown);
} else console.log('note no lab 5.1 roofline rows on this machine; the measured-threshold checks are skipped');
await page.select(`${IB} select[aria-label="threshold"]`, 'custom');
await pause(60);
await setNumber(`${IB} input[aria-label="custom threshold"]`, 40);
await setTile(128, 128, null);
check('a custom threshold of 40 makes 128 x 128 memory-bound', (await regimeShown()) === 'memory-bound');
await page.select(`${IB} select[aria-label="threshold"]`, 'book');
await shot('explorer-custom', IB);

// ---------------------------------------------------------------- the grid
let g = gridFetches(4096, 4096, 4096, 128, 128);
check('grid: 4096 cubed with 128 x 128 tiles fetches each A element 32 times', (await outVal(GB, 'afetch')) === String(g.aFetches) && g.aFetches === 32);
check('grid: and each B element 32 times, intensity 32', (await outVal(GB, 'bfetch')) === '32' && close(Number(await outVal(GB, 'grid-intensity')), 32));
await page.select(`${GB} select[aria-label="bN"]`, '32');
await page.select(`${GB} select[aria-label="bM"]`, '32');
await pause(80);
check('grid: with 32-wide tiles each A element is fetched 128 times', (await outVal(GB, 'afetch')) === '128' && close(Number(await outVal(GB, 'grid-intensity')), 8));
await page.select(`${GB} select[aria-label="bN"]`, '128');
await page.select(`${GB} select[aria-label="bM"]`, '128');
await clickIn(GB, '4096 → 4096 layer, batch 8');
g = gridFetches(8, 4096, 4096, 128, 128);
check('grid preset batch 8: 3.76 FLOP/B, each weight fetched once', close(Number(await outVal(GB, 'grid-intensity')), g.intensity) && (await outVal(GB, 'bfetch')) === '1' && Math.abs(g.intensity - 3.765) < 0.001);
await clickIn(GB, '1024 → 4096 layer, batch 1');
g = gridFetches(1, 4096, 1024, 128, 128);
check('grid preset batch 1: below 0.5 FLOP/B', close(Number(await outVal(GB, 'grid-intensity')), g.intensity) && g.intensity < 0.5);
await setNumber(`${GB} input[aria-label="grid M"]`, 300);
g = gridFetches(300, 4096, 1024, 128, 128);
check('grid: M = 300 needs 3 block rows and the model\'s intensity', (await outVal(GB, 'blocks')) === `3 × 32` && close(Number(await outVal(GB, 'grid-intensity')), g.intensity));
await shot('grid', GB);

// ---------------------------------------------------------------- the scene
const manifestRes = await fetch(BASE + '/content/ch15/narration/manifest.json');
const manifest = manifestRes.ok ? await manifestRes.json() : null;
const scripts = [];
for (let n = 1; n <= 5; n++) scripts.push(flat((await (await fetch(`${BASE}/content/ch15/narration/reuse-${n}.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '')));
check('five scripts of 35 to 60 words', scripts.every((s) => { const w = s.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(' ').length).join(','));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const parts = await page.$$eval(`${SC} .seg button`, (bs) => bs.map((b) => b.textContent.trim()));
check('the scene offers five parts', parts.length === 5 && parts[0].startsWith('1. Along a row') && parts[4].startsWith('5. One row'), JSON.stringify(parts));
const capBox = () => page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('there is a captions switch, off by default', (await (await capBox()).evaluate((e) => e.checked)) === false);
const hasClip1 = !!manifest?.clips?.['reuse-1'];
const caption = async () => flat(await page.$eval(`${SC} .caption`, (e) => e.textContent));
if (!hasClip1) check('with no audio, part 1 shows its script as a caption anyway', (await caption()) === scripts[0], await caption());
await (await capBox()).click();
await pause(100);
check('switching captions on shows part 1\'s script', (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(SC);
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
check('the captions choice is remembered after a reload', (await (await capBox()).evaluate((e) => e.checked)) === true);

const seek = async (v) => setRange(`${SC} input[type=range]`, v);
const part = async (n) => { await clickIn(SC, `${n}.`); };
const counter = (key) => page.$eval(`${SC} [data-counter="${key}"] [data-value]`, (e) => Number(e.textContent));
const svgAttr = (sel, attr) => page.$eval(`${SC} ${sel}`, (e, a) => Number(e.getAttribute(a)), attr);

await part(1);
await seek(1000);
let c = regionCounts(128, 32, 8, 4);
check('part 1 at its end: band 0 of A fetched 4 times, each B band once', (await counter('A0')) === c.aBand[0] && (await counter('B0')) === 1 && (await counter('B3')) === 1 && (await counter('A1')) === 0);
check('part 1 drawing marks A band 0 as the hot band', (await page.$eval(`${SC} [data-band="A0"]`, (e) => e.dataset.hot)) === '1');
await part(2);
await seek(1000);
c = regionCounts(128, 32, 8);
check('part 2 at its end: every band fetched 4 times', (await Promise.all(['A0', 'A3', 'B0', 'B3'].map(counter))).every((v) => v === 4));
check('part 2: requested bytes are 4 times what the data needs', (await svgAttr('[data-requested]', 'data-requested')) === c.requested && (await svgAttr('[data-needed]', 'data-needed')) === c.needed && c.requested === 4 * c.needed);
if (out) await page.screenshot({ path: out + '-scene-2.png' });
await part(3);
await seek(1000);
c = regionCounts(128, 128, 8);
check('part 3: one block, each band once, requested = needed', (await counter('A0')) === 1 && (await counter('B0')) === 1 && (await svgAttr('[data-requested]', 'data-requested')) === c.requested && c.requested === c.needed);
check('part 3 draws the merged block', !!(await page.$(`${SC} [data-merged]`)));
if (out) await page.screenshot({ path: out + '-scene-3.png' });
await part(4);
await seek(0);
const i0 = await svgAttr('[data-intensity]', 'data-intensity');
const k0 = await svgAttr('[data-k]', 'data-k');
await seek(1000);
const i1 = await svgAttr('[data-intensity]', 'data-intensity');
const k1 = await svgAttr('[data-k]', 'data-k');
check('part 4: k slides from 8 to 32 and the ratio stays 32', k0 === 8 && k1 === 32 && close(i0, 32) && close(i1, 32), JSON.stringify({ k0, k1, i0, i1 }));
check('part 4: operations and bytes are the model\'s at k = 32', (await svgAttr('[data-ops]', 'data-ops')) === tilePair(128, 128, 32).ops && (await svgAttr('[data-bytes]', 'data-bytes')) === tilePair(128, 128, 32).traffic);
await part(5);
await seek(1000);
const m5 = await svgAttr('[data-m]', 'data-m');
const n5 = await svgAttr('[data-n]', 'data-n');
const i5 = await svgAttr('[data-intensity]', 'data-intensity');
check('part 5 at its end: one row, a wide tile, below 0.5 FLOP/B', m5 === 1 && n5 === 4096 && close(i5, intensity(1, 4096)) && i5 < 0.5, JSON.stringify({ m5, n5, i5 }));
check('part 5 caption is its own script', (await caption()) === scripts[4]);
if (out) await page.screenshot({ path: out + '-scene-5.png' });

// Play runs the animation (on estimated timings when there is no audio).
await part(1);
await page.evaluate((sel) => { const a = document.querySelector(sel + ' audio'); if (a) a.muted = true; }, SC);
await clickIn(SC, 'Play');
await pause(1500);
const pos = await page.$eval(`${SC} input[type=range]`, (e) => Number(e.value));
check('play moves the animation forward', pos > 0, String(pos));
await clickIn(SC, 'Pause');

// ---------------------------------------------------------------- questions
const qOf = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const q = await qOf(needle);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const choice = async (needle, idx) => {
  const q = await qOf(needle);
  await (await q.$$('.opts label'))[idx].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
check('q-rect: 21.3 is accepted', (await numeric('128 × 64 output tile', 21.3)).includes('Correct.'));
check('q-batch59: 58 is not enough', (await numeric('smallest b that reaches', 58)).includes('Not quite. The answer is 59'));
check('q-fetches: 32 is accepted', (await numeric('each element of A fetched', 32)).includes('Correct.'));
check('q-bk: intensity unchanged, shared memory doubles', (await choice('tile depth bK doubles', 0)).includes('Correct.'));
check('q-decode: a wider tile does not help', (await choice('has m = 1', 1)).includes('Not quite.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
