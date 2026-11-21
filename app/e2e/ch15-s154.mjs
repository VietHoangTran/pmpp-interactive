// Browser check for section 15.4: the prediction, the "Two orders of the same loops"
// explorer and stepper, the narrated scene "Filling a register tile with outer products"
// (captions, parts, playing muted, and each part's drawing against the model), the two mm
// listings from labs 15.1 and 15.2, and the questions. Numbers come from
// src/sim/gemmreads.ts, imported directly. Passes with or without
// content/ch15/narration/manifest.json.
// Usage: APP_URL=http://localhost:5173 node e2e/ch15-s154.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { BOOK_SHAPE, fmas, readsPerFma, registersFor, run, sharedReadsV1, sharedReadsV2, smallInputs, stateAt } from '../src/sim/gemmreads.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const T = 'section.bench[aria-label="Two orders of the same loops"]';
const S = 'section.scene[aria-label="Filling a register tile with outer products"]';

async function click(scope, label) {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(120);
  return ok;
}
async function slide(scope, label, value) {
  await page.evaluate((scope, label, value) => {
    const el = document.querySelector(scope).querySelector(`input[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
}
const counts = () => page.$$eval(T + ' [data-counts] [data-v]', (els) => Object.fromEntries(els.map((e) => [e.dataset.v, e.textContent.trim()])));
const panelReads = () => page.$$eval(T + ' [data-order] [data-reads-total]', (els) => els.map((e) => Number(e.textContent)));
const panelC = (order) => page.$$eval(T + ` [data-order="${order}"] [data-c]`, (els) => els.map((e) => Number(e.dataset.v)));

await goto(page, 'ch15/15.4');
await page.waitForSelector(T);
await pause(300);

// ---- prediction
{
  const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many times fewer shared-memory reads')));
  await (await q.$('input[type=number]')).type('8');
  await (await q.$('button.primary')).click();
  await pause(150);
  const r = await q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
  const r1 = sharedReadsV1(BOOK_SHAPE), r2 = sharedReadsV2(BOOK_SHAPE);
  check('the prediction reveals 8 times, 1024 against 128', r.includes(`${r1 / r2} times.`) && r.includes(`= ${r1} reads`) && r.includes(`= ${r2} values`) && r.includes('You had it.'), r);
}

// ---- explorer counts
await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
let c = await counts();
check('the book\'s shape: 1024 against 128 reads, 512 multiply-adds, 0.25 per multiply-add, 80 registers',
  c.r1 === '1024' && c.r2 === '128' && c.fmas === '512' && c.per === '0.25' && c.regs === String(registersFor(BOOK_SHAPE, 'v2')), JSON.stringify(c));
check('the ratio line says 8× fewer', (await text(T + ' [data-ratio]')).includes('8× fewer'));
for (const [label, s] of [['16 × 4', { tM: 16, tN: 4, bK: 8 }], ['32 × 2', { tM: 32, tN: 2, bK: 8 }]]) {
  await click(T, label);
  c = await counts();
  check(`preset ${label}: ${sharedReadsV2(s)} reads, ${readsPerFma(s, 'v2')} per multiply-add`, c.r2 === String(sharedReadsV2(s)) && c.per === String(readsPerFma(s, 'v2')) && c.r1 === String(sharedReadsV1(s)), JSON.stringify(c));
}
await slide(T, 'tM', 5);
await slide(T, 'tN', 7);
await slide(T, 'bK', 3);
const odd = { tM: 5, tN: 7, bK: 3 };
c = await counts();
check('sliders 5 × 7, bK 3: counts follow the closed forms', c.r1 === String(sharedReadsV1(odd)) && c.r2 === String(sharedReadsV2(odd)) && c.fmas === String(fmas(odd)), JSON.stringify(c));

// ---- stepper on the small preset
await click(T, 'small: 3 × 3, bK 4');
const small = { tM: 3, tN: 3, bK: 4 };
const x = smallInputs(small);
const t1 = run(small, x, 'v1'), t2 = run(small, x, 'v2');
check('nothing has run yet', (await panelReads()).join() === '0,0' && (await text(T + ' [data-narrate]')).startsWith('Nothing has run'));
await click(T, 'Step');
let pr = await panelReads();
check('one step: the first order has read 2 values, the second has loaded both strips (6 reads)', pr[0] === stateAt(t1, 1).sharedReads && pr[1] === stateAt(t2, 1).sharedReads && pr[1] === 6, pr.join());
const nar = await text(T + ' [data-narrate]');
check('the narration spells out both multiply-adds', nar.includes('c[0][0] += a[0][0] · b[0][0]') && nar.includes('first 6 strip values copied to registers') && nar.includes('c[0][0] += a_r[0] · b_r[0]'), nar);
const hot = await page.$$eval(T + ' [data-order="v1"] [data-a] rect', (rs) => rs.filter((r) => r.getAttribute('fill') === 'var(--shared)').length);
check('the first order lights one slot of a', hot === 1, String(hot));
await click(T, 'Next rank-1 update');
pr = await panelReads();
check('after one rank-1 update (9 multiply-adds): 18 reads against 6', pr[0] === stateAt(t1, 9).sharedReads && pr[1] === stateAt(t2, 9).sharedReads && pr[0] === 18, pr.join());
check('the second order\'s patch after one rank-1 update is the outer product of the strips', JSON.stringify(await panelC('v2')) === JSON.stringify(stateAt(t2, 9).C));
if (out) await page.screenshot({ path: out + '-stepper.png' });
await click(T, 'Run to end');
pr = await panelReads();
const c1 = await panelC('v1'), c2 = await panelC('v2');
check('at the end: 24 reads against 12, and both patches equal the model\'s', pr[0] === t1.sharedReads && pr[1] === t2.sharedReads && JSON.stringify(c1) === JSON.stringify(t1.C) && JSON.stringify(c2) === JSON.stringify(t1.C), JSON.stringify({ pr, c1, c2 }));
check('the stepper says both orders hold the same values', (await page.$(T + ' [data-same]')) !== null);
await click(T, 'the book: 8 × 8, bK 8');
await click(T, 'Run to end');
check('the book\'s shape, run to the end: 1024 against 128 reads', (await panelReads()).join() === '1024,128');

// ---- the scene
const res = await fetch(BASE + '/content/ch15/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch15/narration/outer-product-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('six scripts, each 35 to 60 words, no symbols read aloud', words.every((w) => w >= 35 && w <= 60) && scripts.every((s) => !/[<>=×%/_*+]/.test(s)), words.join(' '));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const part = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const svg = () => page.evaluate((S) => {
  const s = document.querySelector(S + ' .stage svg');
  return {
    v1: Number(s.querySelector('[data-count="v1"]')?.textContent ?? NaN),
    v2: s.querySelector('[data-count="v2"]') ? Number(s.querySelector('[data-count="v2"]').textContent) : null,
    same: !!s.querySelector('[data-same]'),
    zoom: s.hasAttribute('data-zoom'),
    filled: [...s.querySelectorAll('[data-c] text')].filter((t) => t.textContent !== '').length,
    maxReads: Math.max(0, ...[...s.querySelectorAll('[data-a]')].map((g) => Number(g.dataset.reads))),
    text: s.textContent,
  };
}, S);
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts with their names', labels.join('|') === ['One output', 'Read again', 'Swap the loops', 'Outer product', 'Steps 1 to 7', 'Two levels'].map((l, i) => `${i + 1}. ${l}`).join('|'), labels.join('|'));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await text(S + ' footer')).includes('No narration manifest found'));
check('captions are off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['outer-product-1'];
if (!audible) check('a part without audio shows its script anyway', (await caption()) === scripts[0], String(await caption()));
await setBox('captions', true);
check('captions on: part 1\'s script', (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered', await (await box('captions')).evaluate((e) => e.checked));

// play muted
await setBox('sound', false);
const before = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
await page.evaluate((sel) => document.querySelector(sel + ' button.primary').click(), S);
await pause(2500);
const moved = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const silent = await page.$eval(S + ' audio', (a) => a.paused);
check('with sound off it plays on the clock and stays silent', moved > before + 20 && silent, JSON.stringify({ before, moved, silent }));
await page.evaluate((sel) => document.querySelector(sel + ' button.primary').click(), S);
await pause(150);

// each part's drawing against the model
const v1 = run(BOOK_SHAPE, smallInputs(BOOK_SHAPE), 'v1');
await part('1. One output');
await seek(1000);
let d = await svg();
check('part 1 at its end: output (0, 0) done with 16 reads', d.v1 === stateAt(v1, 8).sharedReads && d.v1 === 16 && d.text.includes('output (0, 0)'), JSON.stringify(d));
await part('2. Read again');
await seek(300);
d = await svg();
check('part 2 early: the values of A are being read a second time', d.maxReads === 2, JSON.stringify({ maxReads: d.maxReads, v1: d.v1 }));
await seek(1000);
d = await svg();
check('part 2 at its end: 1024 reads, every value of A read 8 times', d.v1 === 1024 && d.maxReads === 8 && d.text.includes(`1024 shared reads for ${fmas(BOOK_SHAPE)} multiply-adds`), JSON.stringify({ v1: d.v1, maxReads: d.maxReads }));
await part('3. Swap the loops');
await seek(1000);
d = await svg();
check('part 3 at its end: both strips loaded, 16 reads, nothing computed yet', d.v2 === 16 && d.filled === 0 && d.text.includes('a_r') && d.text.includes('b_r'), JSON.stringify({ v2: d.v2, filled: d.filled }));
await part('4. Outer product');
await seek(1000);
d = await svg();
check('part 4 at its end: all 64 cells hold a value after 16 reads', d.v2 === 16 && d.filled === 64 && stateAt(run(BOOK_SHAPE, smallInputs(BOOK_SHAPE), 'v2'), 64).updates.every((n) => n === 1), JSON.stringify({ v2: d.v2, filled: d.filled }));
await part('5. Steps 1 to 7');
await seek(1000);
d = await svg();
check('part 5 at its end: 128 against 1024 and the same tile as the first order', d.v2 === 128 && d.v1 === 1024 && d.same && d.text.includes('8 times fewer'), JSON.stringify({ v1: d.v1, v2: d.v2, same: d.same }));
if (out) await page.screenshot({ path: out + '-scene5.png' });
await part('6. Two levels');
await seek(1000);
d = await svg();
check('part 6 zooms out to the block: shared tiles, register strips, accumulated outer product', d.zoom && d.text.includes('A_s (shared)') && d.text.includes('accumulated outer product'), d.text.slice(0, 120));
check('part 6 shows its own script as the caption', (await caption()) === scripts[5]);

// ---- the two mm listings
const figs = await page.$$eval('[data-two-mm] figure.code', (fs) => fs.map((f) => ({ tab: f.querySelector('.tabs button').textContent.trim(), code: f.querySelector('pre').textContent })));
check('two mm listings, from labs 15.1 and 15.2', figs.length === 2 && figs[0].tab.startsWith('mm, lab 15.1') && figs[1].tab.startsWith('mm, lab 15.2'), JSON.stringify(figs.map((f) => f.tab)));
check('the first reads a and b in the innermost loop; the second fills a_r and b_r', figs[0]?.code.includes('c[row][col] += a[row * lda + i] * b[i * ldb + col];') && figs[1]?.code.includes('c[row][col] += a_r[row] * b_r[col];'));

// ---- questions
async function numeric(prompt, value) {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), prompt);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
}
async function choice(prompt, optionText) {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), prompt);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), optionText);
  await pause(80);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
}
check('question: a value in a_r feeds 8 multiply-adds', await numeric('how many multiply-adds of that step use', 8));
check('question: the square tile needs the fewest reads', await choice('which shape needs the fewest', '8 × 8'));
check('question: intensity comes from the output tile; register tiling cuts shared reads', await choice('Which statement says precisely', 'came from the 128 × 128 output tile'));
check('question: double-buffered strips overlap the next reads with this step', await choice('What does the second set make possible', 'Reading step i + 1'));
check('a wrong answer is marked wrong', !(await choice('which shape needs the fewest', '32 × 2')));

const expected404 = (u) => /\/content\/ch15\/narration\/outer-product-\d\.mp3$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch15/15.4', [
  { label: 'mm, lab 15.1 (Fig. 15.6)', lab: 'ch15/lab1_coarse', file: 'kernel.cu' },
  { label: 'mm, lab 15.2 (Fig. 15.9)', lab: 'ch15/lab2_regtile', file: 'kernel.cu' },
]);

check('no console or page errors', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
