// Browser check for section 12.2: the lab listing, the prediction, the "One counter, many
// threads" explorer (against src/sim/filter.ts, imported directly) and two questions.
// Usage: node e2e/ch12-s122.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses, tabStatus } from './labstatus.mjs';
import { FRACTIONS, atomicCounts, expectedAtomics, patternSource } from '../src/sim/filter.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const X = 'One counter, many threads';
const setRange = async (label, v) => {
  await page.$eval(`${bench(X)} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const setSelect = async (label, v) => {
  await page.$eval(`${bench(X)} select[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(150);
};
const setWarp = async (w) => {
  await page.evaluate((sel, w) => [...document.querySelectorAll(sel + ' [aria-label="warp width"] button')].find((b) => b.textContent.trim() === String(w)).click(), bench(X), w);
  await pause(150);
};
const table = () => page.$$eval(`${bench(X)} tr[data-count]`, (rows) => Object.fromEntries(rows.map((r) => [r.dataset.count, { value: Number(r.dataset.value), ratio: r.querySelectorAll('td')[2].textContent.trim() }])));
const near = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
const fi = (f) => FRACTIONS.indexOf(f);

await goto(page, 'ch12/12.2');
await page.waitForSelector(bench(X));

// ---- the listing: lab 12.1's Fig. 12.2
const hasListing = await page.waitForSelector('.code .tabs button', { timeout: 5000 }).then(() => true).catch(() => false);
if (hasListing) {
  const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
  check('listing shows the CUDA and Metal tabs', tabs.includes('CUDA') && tabs.includes('Metal'), tabs.join('|'));
  const code = await page.$eval('.code pre', (e) => e.textContent);
  check('the CUDA tab is lab 12.1\'s filterAtomicKernel', code.includes('filterAtomicKernel') && code.includes('fetch_add(1, cuda::memory_order_relaxed)'));
  const labReports = await (await fetch(BASE + '/api/reports')).json();
  const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
  check('the status is the labs page\'s, from this machine\'s runs', (await page.$eval('.code .tabs .status', (e) => e.textContent)) === tabStatus(labReports, 'ch12/lab1_unstable', 'kernel.cu', frontend));
} else check('listing shows lab 12.1 (labs/ch12/lab1_unstable/solution/ must exist)', false);

// ---- predict
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('times fewer')));
await (await pred.$('input[type=number]')).type('16');
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.textContent);
check('the prediction reveals 524,288 against 32,768: 16 times fewer', reveal.includes('524,288') && reveal.includes('32,768') && reveal.includes('16 times fewer') && reveal.includes('That is your number'), reveal);

// ---- explorer: independent keys, 2^20 keys, blocks of 256, warps of 32
const n = 2 ** 20;
let t = await table();
let e = expectedAtomics(n, 0.5, 32, 256);
check('at 50 percent: the expected counts of the model', near(t.perKey.value, e.perKey) && near(t.perWarp.value, e.perWarp) && t.perBlock.value === 4096, JSON.stringify(t));
check('per warp is 16.0 times fewer, per block 128 times', t.perWarp.ratio === '16.0×' && t.perBlock.ratio === '128.0×', JSON.stringify(t));
check('the plot draws three curves and marks the current fraction', (await page.$$(`${bench(X)} [data-curve]`)).length === 3 && !!(await page.$(`${bench(X)} [data-marker]`)));
if (out) await page.$eval(bench(X), (el) => el.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-explorer.png' });

await setRange('fraction kept', fi(0.01));
t = await table();
e = expectedAtomics(n, 0.01, 32, 256);
check('at 1 percent: barely better, 1.16 times', near(t.perWarp.value, e.perWarp) && t.perWarp.ratio === '1.16×', JSON.stringify(t));
check('per block is unchanged: 4096', t.perBlock.value === 4096);
if (out) await page.screenshot({ path: out + '-one-percent.png' });

await setRange('fraction kept', fi(1));
t = await table();
check('at 100 percent: 32 times', t.perWarp.ratio === '32.0×', JSON.stringify(t));

await setRange('fraction kept', fi(0));
t = await table();
check('at 0 percent nothing is kept, and every block still adds its zero', t.perKey.value === 0 && t.perWarp.value === 0 && t.perBlock.value === 4096
  && (await page.$eval(`${bench(X)} [data-summary]`, (el) => el.textContent)).includes('every block still adds its zero'));
check('0 percent is not marked on the log axis, and the page says so', !(await page.$(`${bench(X)} [data-marker]`)) && (await page.$eval(bench(X), (el) => el.textContent)).includes('no place on a log axis'));

// ---- layouts at 1 percent: runs and sorted keys share warps
await setRange('fraction kept', fi(0.01));
for (const p of ['runs', 'sorted', 'spread']) {
  await setSelect('layout', p);
  t = await table();
  const c = atomicCounts(patternSource(p, n, 0.01), 32, 256);
  check(`layout "${p}" at 1 percent: the model's exact counts`, t.perKey.value === c.perKey && t.perWarp.value === c.perWarp && t.perBlock.value === c.perBlock, JSON.stringify([t, c]));
  if (p === 'runs') check('runs of kept keys: the warp scheme recovers (more than 10 times fewer)', c.perKey / c.perWarp > 10 && parseFloat(t.perWarp.ratio) > 10, t.perWarp.ratio);
}
if (out) await page.screenshot({ path: out + '-spread.png' });

// ---- warp width, block size, number of keys
await setSelect('layout', 'random');
await setRange('fraction kept', fi(0.5));
await setWarp(64);
t = await table();
check('warps of 64 at 50 percent: 32 times fewer', t.perWarp.ratio === '32.0×', JSON.stringify(t));
await setSelect('block size', 1024);
t = await table();
check('blocks of 1024: 2^20 / 1024 = 1024 atomics per block scheme', t.perBlock.value === 1024, JSON.stringify(t));
await setRange('number of keys', 24);
t = await table();
check('2^24 keys at 50 percent: 8,388,608 per key', t.perKey.value === 8388608 && (await page.$eval(bench(X), (el) => el.textContent)).includes('Keys: 224 = 16,777,216'), JSON.stringify(t));

// ---- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((el) => el.textContent.includes('still holds 5 from the previous call')));
await (await q.$('input[type=number]')).type('10');
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question: the highest index written is 10', (await q.evaluate((el) => el.textContent)).includes('Correct.'));
const q2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((el) => el.textContent.includes('1 percent of them pass')));
await (await q2.$('input[type=number]')).type('1.16');
await (await q2.$('button.primary')).click();
await pause(120);
check('numeric question: 1.16 times at 1 percent', (await q2.evaluate((el) => el.textContent)).includes('Correct.'));
const q3 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((el) => el.textContent.includes('runs twice on the same input')));
await q3.evaluate((el) => [...el.querySelectorAll('label')].find((l) => l.textContent.startsWith('Only the count')).querySelector('input').click());
await (await q3.$('button.primary')).click();
await pause(120);
check('choice question: a wrong answer is marked and explained', (await q3.evaluate((el) => el.textContent)).includes('Not quite.'));

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await setRange('fraction kept', fi(0.05));
if (out) await page.$eval(bench(X), (el) => el.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-light.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
check('at a narrow width the curves\' legend is in view without scrolling sideways', await page.evaluate((sel) => {
  const items = [...document.querySelector(sel).querySelectorAll('[data-legend-item]')];
  return items.length === 3 && items.every((e) => e.getBoundingClientRect().right <= document.documentElement.clientWidth);
}, bench(X)));
if (out) await page.$eval(bench(X), (el) => el.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch12/12.2', [
  { label: 'CUDA', lab: 'ch12/lab1_unstable', file: 'kernel.cu' },
  { label: 'Metal', lab: 'ch12/lab1_unstable', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
