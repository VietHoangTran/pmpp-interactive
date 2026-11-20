// Browser check for the shell, the "before you start" page and section 5.1:
// navigation, predictions, the roofline explorer and the linear-layer explorer.
// Usage: node e2e/s51.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const setRange = (sel, value) =>
  page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);

// ---- home ----
await goto(page, '');
check('home lists the whole course', (await page.$$('table.data tbody tr')).length >= 27);
check('home links to chapter 5', !!(await page.$('a[href="#/ch05/start"]')));
const reports = await (await fetch(BASE + '/api/reports')).json();
const real = Object.keys(reports).filter((b) => b !== 'emu');
if (real.length) check('home shows this machine\'s device as it reports itself', (await text('.page')).includes(String(reports[real[0]].device.name)));

// ---- before you start ----
await goto(page, 'ch05/start');
check('prerequisite page shows the naive kernel with tier colours', (await page.$$('.code .v-global')).length >= 5 && (await page.$$('.code .v-reg')).length >= 3);
let q = await question('By how many array elements');
await (await q.$('input[type=number]')).type('1');
await (await q.$('button.primary')).click();
await pause(150);
check('a wrong numeric answer is marked wrong and shows the answer', (await q.evaluate((e) => e.textContent)).includes('Not quite. The answer is 6'));
q = await question('Which threads of the block');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(600);
check('a right choice is marked correct', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
check('answers are written to the progress file with their ids', puts.some((p) => p.answers?.['ch05/start/q-mapping']?.correct === true && p.answers?.['ch05/start/q-rowmajor']?.correct === false));
check('first answers enter the review schedule', puts.some((p) => p.review?.['ch05/start/q-rowmajor']?.box === 0 && p.review?.['ch05/start/q-mapping']?.box === 0));

// ---- 5.1 ----
await goto(page, 'ch05/5.1');
q = await question('per byte of global memory it touches');
check('prediction hides its answer until committed', !(await q.evaluate((e) => e.textContent)).includes('0.083'));
await (await q.$('input[type=number]')).type('0.5');
await (await q.$('button.primary')).click();
await pause(150);
const pred = await q.evaluate((e) => e.textContent);
check('committing reveals the worked answer and reacts to the guess', pred.includes('You predicted 0.5') && pred.includes('0.083') && pred.includes('counted the store'), pred);

const R = 'Roofline';
let panel = await text(bench(R) + ' .panel');
check('book hardware: threshold 20 FLOP/B', panel.includes('threshold 20'), panel);
check('book hardware: naive matmul ceiling about 838 GFLOP/s, memory-bound', /naivematmul0\.25memory-bound838/.test(panel.replace(/\s+/g, '')), panel);
await setRange(bench(R) + ' label.field input[type=range]', 32);
await pause(100);
panel = (await text(bench(R) + ' .panel')).replace(/\s+/g, '');
check('tile 32 on the book hardware: 8 FLOP/B, 26,800 GFLOP/s, still memory-bound', panel.includes('tiled,T=328memory-bound26,800'), panel);
await setRange(bench(R) + ' label.field input[type=range]', 96);
await pause(100);
panel = (await text(bench(R) + ' .panel')).replace(/\s+/g, '');
check('a tile wide enough crosses into compute-bound', panel.includes('tiled,T=9624compute-bound66,900'), panel);
if (out) await page.screenshot({ path: out + '-roofline-book.png' });

// "Your device" appears only when lab 5.1 has been benchmarked on it (not after ./verify.sh --quick).
const measuredOn = real.find((b) =>
  reports[b].results.some((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution' && x.bench.some((r) => r.name === 'roofline')),
);
const offered = await page.$$eval(bench(R) + ' select option', (o) => o.map((x) => x.value));
check('hardware list offers a device only when lab 5.1 measured it', measuredOn ? offered.includes(measuredOn) : offered.join() === 'book,custom', offered.join());
if (measuredOn) {
  const b = measuredOn;
  await page.select(bench(R) + ' select', b);
  await pause(150);
  const row = reports[b].results.find((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution').bench.find((x) => x.name === 'roofline');
  panel = await text(bench(R));
  check('selecting my device uses the bandwidth and throughput lab 5.1 measured', panel.includes('Measured by lab 5.1') && panel.includes(`threshold ${Number(row.threshold_flop_per_byte.toPrecision(3))}`), panel.slice(-300));
  const marks = await page.$$eval(bench(R) + ' svg text', (t) => t.map((x) => x.textContent));
  check('my measured naive point is plotted', marks.some((m) => m.startsWith('naive, measured')), marks.join('|'));
  if (row.naive_gflops > row.naive_model_ceiling_gflops * 1.02)
    check('a measurement above the model ceiling is called out', panel.includes('above the roof'));
  if (out) await page.screenshot({ path: out + '-roofline-mine.png' });
}

const L = 'A linear layer, by batch size';
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.includes('16-bit')).click(), bench(L));
await pause(100);
check('batch 1 at 16 bits gives the book\'s 1 FLOP/B', (await text(bench(L) + ' .panel')).includes('0.999 FLOP/B'), await text(bench(L) + ' .panel'));
await setRange(bench(L) + ' input[type=range]', 9);
await pause(100);
check('batch 512 at 16 bits gives the book\'s 315 FLOP/B', (await text(bench(L) + ' .panel')).includes('315 FLOP/B'), await text(bench(L) + ' .panel'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
