// Browser check for section 9.3: the "Atomic throughput" explorer reproduces the book's
// numbers and the model (src/sim/atomicrate.ts, imported directly), the skew prediction
// reveals lab 9.1's measurements or says what to run, and exercises 1 to 4 answer.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-s93.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { histogramBound, perLocationRate } from '../src/sim/atomicrate.ts';
import { histogram, hottest, preset } from '../src/sim/histogram.ts';

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
const close = (a, b) => Math.abs(a - b) <= Math.abs(b) * 1e-9;
const E = 'section.bench[aria-label="Atomic throughput"]';
const press = async (group, label) => {
  const ok = await page.evaluate((E, group, label) => {
    const root = group ? document.querySelector(`${E} [role=group][aria-label="${group}"]`) : document.querySelector(E);
    const b = [...root.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, E, group, label);
  await pause(80);
  return ok;
};
const setRange = async (label, v) => {
  await page.$eval(`${E} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const read = () =>
  page.evaluate((E) => {
    const one = document.querySelector(E + ' [data-one]');
    const bound = document.querySelector(E + ' [data-bound]');
    return { one: Number(one.dataset.one), oneText: one.textContent, bound: Number(bound.dataset.bound), boundText: bound.textContent, panel: document.querySelector(E).textContent };
  }, E);

await goto(page, 'ch09/9.3');
await page.waitForSelector(E);
await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));

// ---- the book's example
let r = await read();
check('book preset: one location 2.5 M/s', close(r.one, 2.5e6) && r.oneText === '2.5 M/s', JSON.stringify(r).slice(0, 200));
check('book preset: 256 uniform bins 640 M/s', close(r.bound, 640e6) && r.boundText === '640 M/s', r.boundText);
check('book preset: 128 GB/s and 50 times below the memory\'s elements', r.panel.includes('128 GB/s') && r.panel.includes('50 times fewer'));

// ---- the busiest bin sets the pace
await press('Data', 'set the busiest bin');
await setRange("Busiest bin's share", 0.5);
r = await read();
check('a 50% busiest bin gives 5 M/s', close(r.bound, 5e6) && r.boundText === '5 M/s', r.boundText);
await press('Data', 'constant');
r = await read();
check('constant data runs at the one-location rate', close(r.bound, 2.5e6));
await press('Data', 'a bright image');
r = await read();
const bright = hottest(histogram(preset('sky').values, 1)).share;
check('a bright image uses the sky preset\'s busiest-bin share (31 of 64)', close(r.bound, 2.5e6 / bright) && r.panel.includes('48.4%'), String(r.bound));
await page.select(E + ' select[aria-label="Bins"]', '4');
await pause(80);
r = await read();
check('the bright image in 4 bins: share 35 of 64', close(r.bound, 2.5e6 / (35 / 64)), String(r.bound));
await press('Data', 'uniform');
r = await read();
check('uniform over 4 bins: 10 M/s', close(r.bound, 10e6) && r.boundText === '10 M/s', r.boundText);

// ---- shorter latency, faster clock, the memory cap
await press('Where the atomic is performed', 'last-level cache');
r = await read();
check('last-level cache at its illustrative 30 cycles: 1 GHz / 60', close(r.one, perLocationRate(1e9, 30)), String(r.one));
await setRange('Latency per access', 20);
r = await read();
check('20-cycle cache accesses: 25 M/s on one location', close(r.one, 25e6) && r.oneText === '25 M/s');
await press('Where the atomic is performed', 'shared memory');
await setRange('Latency per access', 1);
await page.select(E + ' select[aria-label="Bins"]', '256');
await pause(80);
r = await read();
check('1-cycle shared-memory atomics over 256 bins hit the memory cap of 32 G/s', close(r.bound, 32e9) && r.panel.includes('capped by memory'), r.boundText);
await press(null, "Book's example");
await press('Clock', 'custom');
await page.$eval(E + ' input[type=number]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '2');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(80);
r = await read();
check('a 2 GHz clock doubles both: 5 M/s on one location, 1.28 G/s uniform', close(r.one, 5e6) && close(r.bound, histogramBound(5e6, 1 / 256).rate), JSON.stringify([r.one, r.bound]));
await press(null, "Book's example");
r = await read();
check('Book\'s example restores 2.5 M/s and 640 M/s', close(r.one, 2.5e6) && close(r.bound, 640e6));
// The reader's own measured slowdown is the answer to the prediction below, so nothing on the
// page shows it before a number is committed.
const reportsNow = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const measured = Object.entries(reportsNow).some(([b, run]) => b !== 'emu' && b !== 'race' && (run?.results ?? []).some((x) => x.lab === 'ch09/lab1_racehunt' && x.variant === 'solution' && (x.bench ?? []).some((r) => r.name === 'basic')));
console.log(measured ? 'note lab 9.1 measurements exist on this machine' : 'note no lab 9.1 measurements on this machine');
check('the explorer shows no measured panel', (await page.$(E + ' [data-part="measured"]')) === null);
check('no measured slowdown is on the page before the prediction is committed', (await page.$('[data-part="skew-measured"]')) === null);
if (out) await page.screenshot({ path: out + '-explorer.png' });

// ---- text
const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the book\'s worked numbers are in the text', body.includes('128 GB/s') && body.includes('32 G four-byte elements') && body.includes('2.5 M/s') && body.includes('640 M/s'));
const notes = await page.$$eval('.note', (n) => n.map((x) => x.className + ': ' + x.textContent));
check('Today notes cite A4 (L2 atomics) and A9 (compiler aggregation)', notes.some((t) => t.includes('today') && t.includes('A4') && t.includes('lts__t_sectors_srcunit_tex_op_atom')) && notes.some((t) => t.includes('today') && t.includes('A9') && t.includes('B5')));
check('red versus atom is labelled not in the book', notes.some((t) => t.includes('beyond') && t.includes('gpuAtomicAddNoReturn')));
check('the explorer and the scene state what they simplify', notes.filter((t) => t.includes('caveat')).length >= 2);
check('chapter 10\'s one-atomic-per-block commit is the bridge', body.includes('Fig. 10.18'));
check('remedy 1 says which rates move and that the memory caps them', body.includes('the per-location and whole-histogram rates move by the same factor, until the histogram reaches what the memory can deliver') && !body.includes('every rate moves'));

// ---- predict the skew slowdown
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('constant image than on uniform random bytes')));
await (await pq.$('input[type=number]')).type('256');
await (await pq.$('button.primary')).click();
await pause(400);
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the prediction is committed and revealed', reveal.includes('You predicted 256'));
check('the reveal shows the measurement or says what to run', measured ? reveal.includes('slower on the constant image') : reveal.includes('python3 labs/pmpp.py run 9.1 --solution --bench'), reveal.slice(0, 300));

// ---- exercises and questions
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
async function numeric(text, value) {
  const q = await question(text);
  await (await q.$('input[type=number]')).evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
}
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
}
let t = await numeric('100 ns in total', 10);
check('exercise 1: 10 M/s', t.includes('Correct.') && t.includes('exercise 1'), t.slice(0, 200));
t = await numeric('90% of the atomics hit', 52);
check('exercise 2: 52 (the plain average of 4 and 100) is wrong, 73.5 revealed', t.includes('Not quite. The answer is 73.5'));
t = await numeric('90% of the atomics hit', 73.5);
check('exercise 2: 73.5 accepted, 13.6 ns shown', t.includes('Correct.') && t.includes('13.6 ns'));
t = await numeric('5 floating-point operations for every atomic', 50);
check('exercise 3: 50 M FLOP/s', t.includes('Correct.'));
t = await numeric('privatised into shared memory', 4.5);
check('exercise 4: 4.5 is within tolerance, and the 2 ns reading is explained', t.includes('Correct.') && t.includes('2.3 GFLOP/s') && t.includes('Unclear in the PDF'), t.slice(0, 400));
t = await numeric('half of the pixels fall in one bin', 5);
check('the half-skewed histogram: 5 M/s', t.includes('Correct.') && t.includes('128 times slower'));
t = await choose('all-white image', 'more threads hide more latency');
check('more threads: "hide more latency" is wrong', t.includes('Not quite.') && t.includes('cannot'));
t = await choose('all-white image', 'longer');
check('more threads: accepted', t.includes('Correct.'));

// ---- narrow, dark
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways scrolling at 400 px', wide <= 1, String(wide));
if (out) {
  await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-dark.png' });
}

const real = errors.filter((e) => !/status of 404/.test(e) && !/content\/ch09\/narration\/(manifest\.json|atomic-wait-\d\.mp3)/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0, real.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
