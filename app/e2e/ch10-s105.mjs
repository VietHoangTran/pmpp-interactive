// Browser check for section 10.5: the request prediction, the segment explorer (three
// kernels, sizes, widths, steps, warps) and the questions. Expected values come from
// src/sim/reduction.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s105.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { requests } from '../src/sim/reduction.ts';

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
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const EX = 'section.bench[aria-label="Which pieces does a warp touch?"]';
const press = async (label, group) => {
  const done = await page.evaluate((root, label, group) => {
    const scope = group ? document.querySelector(`${root} [aria-label="${group}"]`) : document.querySelector(root);
    const b = [...scope.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, EX, label, group);
  await pause(60);
  return done;
};
const total = () => text(`${EX} [data-readout="total"]`).then(Number);
const warpReq = () => page.$eval(`${EX} [data-readout="warp"]`, (e) => Number(e.dataset.requests));
const hits = () => page.$$eval(`${EX} svg[data-part="segments"] g[data-hit="1"]`, (g) => g.length);

await goto(page, 'ch10/10.5');
await page.waitForSelector(EX);

// Predict first.
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many global-memory requests does it make for 256 values')));
await (await pred.$('input[aria-label="your prediction"]')).type('40');
await pred.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
await pause(120);
check('the prediction reveals 36 from 12 warp-iterations', (await pred.evaluate((e) => e.textContent)).includes('The model counts 36: 12 warp-iterations'));

// Simple kernel at 256.
check('simple, 256: 141 requests', (await total()) === 141);
check('the book\'s formula line says 141', (await text(`${EX} [data-readout="book"]`)).includes('141'));
check('iteration 1, warp 0: 6 requests over 2 pieces', (await warpReq()) === 6 && (await hits()) === 2);
const marks = await page.$$eval(`${EX} svg[data-part="segments"] rect[data-mark]`, (r) => r.filter((x) => x.dataset.mark).length);
check('iteration 1, warp 0: 64 elements marked (32 own, 32 partners)', marks === 64, String(marks));
for (let i = 0; i < 4; i++) await press('Step');
const n5 = await text(`${EX} [data-readout="warp"]`);
check('iteration 5 (stride 16), warp 0: 2 elements in 2 pieces per access, 6 requests', n5.includes('iteration 5, stride 16') && n5.includes('2 elements in 2 pieces') && (await warpReq()) === 6, n5);
await press('Step');
check('iteration 6, warp 0: one lane, 3 requests', (await warpReq()) === 3);
if (out) await page.screenshot({ path: out + '-s105-simple.png' });

// Convergent.
await press('convergent', 'Kernel');
check('convergent, 256: 36 requests', (await total()) === 36);
check('iteration 1, warp 0: 3 requests, one piece each', (await warpReq()) === 3);
await press('Step');
await page.select(`${EX} select[aria-label="Warp"]`, '3');
await pause(60);
check('iteration 2, warp 3: idle, no requests', (await text(`${EX} [data-readout="warp"]`)).includes('Warp 3 has no active lane'));

// Shared memory.
await press('shared memory', 'Kernel');
check('shared, 256: 9 requests', (await total()) === 9);
check('the load: warp 0 makes 2 requests', (await warpReq()) === 2);
await press('Step');
check('the tree: shared memory only', (await text(`${EX} [data-readout="warp"]`)).includes('shared memory only'));

// 2048 values, all three kernels.
await press('2048', 'N');
check('shared, 2048: 65', (await total()) === 65);
await press('convergent', 'Kernel');
check('convergent, 2048: 204', (await total()) === 204);
await press('simple', 'Kernel');
check('simple, 2048: 1149', (await total()) === 1149);
check('more than 8 pieces: only the touched ones are drawn', (await page.$eval(`${EX} svg[data-part="segments"]`, (s) => Number(s.dataset.rows))) === 2);

// 64-wide warps.
await press('256', 'N');
await page.select(`${EX} select[aria-label="Warp width"]`, 'w64');
await pause(80);
check('simple, 256, 64-wide: the model\'s count', (await total()) === requests('simple', 256, 64).total);
check('64-wide warp 0 spans four pieces in iteration 1', (await hits()) === 4);

// Questions.
const q = async (snippet) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), snippet);
const q1 = await q('2048 values with 32-wide warps');
await (await q1.$('input[type=number]')).type('1149');
await (await q1.$('button.primary')).click();
await pause(100);
check('q-req-2048 accepts 1149', (await q1.evaluate((e) => e.textContent)).includes('Correct.'));
const q2 = await q('32,000 logits');
await q2.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
await (await q2.$('button.primary')).click();
await pause(100);
check('q-softmax-loop: A coalesces', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));
const q3 = await q('first iteration, what fraction');
await q3.evaluate((e) => e.querySelectorAll('input[type=radio]')[1].click());
await (await q3.$('button.primary')).click();
await pause(100);
check('q-half-used: one half', (await q3.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
