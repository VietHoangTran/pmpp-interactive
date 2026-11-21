// Browser check for section 11.7: "What registers save" (the access counter over Figs. 11.12
// and 11.13), the "Bank map", the listings and three questions. Expected values come from
// src/sim/coarsescan.ts, imported inside the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-s117.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
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
  await pause(150);
};
const select = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(150);
};
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/coarsescan.ts');
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);

await goto(page, 'ch11/11.7');
await page.waitForSelector(bench('What registers save'));

// ---------------------------------------------------------------- What registers save
const A = bench('What registers save');
await page.waitForSelector(`${A} [data-annotated]`);
const cellsOf = (phase) => page.$$eval(`${A} tr[data-phase="${phase}"] td[data-k]`, (els) => els.map((e) => Number(e.textContent)));
const expect = (F, who) => model((m, F, who) => [m.sharedAccesses('fig11.12', F, who), m.sharedAccesses('fig11.13', F, who)].map((a) => ({ total: a.bufferS.total, per: a.perElement })), F, who);
let e = await expect(4, 'other');
check('factor 4: 26 and 16 accesses per thread', JSON.stringify(await cellsOf('per-thread')) === JSON.stringify([e[0].total, e[1].total]), JSON.stringify(await cellsOf('per-thread')));
check('factor 4: 6.5 and 4 per element', JSON.stringify(await cellsOf('per-element')) === JSON.stringify([6.5, 4]));
const scanRow = await cellsOf('scan');
check('the scan phase: 9 shared accesses become 4', JSON.stringify(scanRow) === JSON.stringify([9, 4]), JSON.stringify(scanRow));
const notes13 = await page.$$eval(`${A} [data-note]`, (els) => els.map((x) => Number(x.dataset.note)));
const lines13 = await model((m) => m.ACCESS_LINES['fig11.13'].map((l) => l.line));
check('Fig. 11.13 listing annotates exactly the model\'s lines', JSON.stringify(notes13) === JSON.stringify(lines13), JSON.stringify(notes13));
check('line 16 runs F − 1 = 3 times', (await text(`${A} [data-note="16"]`)).includes('×3: 3 shared reads'));
await clickText(A, 'Fig. 11.12 (run in shared memory)');
await page.waitForSelector(`${A} [data-annotated="fig11.12"]`);
const notes12 = await page.$$eval(`${A} [data-note]`, (els) => els.map((x) => Number(x.dataset.note)));
const lines12 = await model((m) => m.ACCESS_LINES['fig11.12'].map((l) => l.line));
check('Fig. 11.12 listing annotates exactly the model\'s lines', JSON.stringify(notes12) === JSON.stringify(lines12), JSON.stringify(notes12));
check('Fig. 11.12 line 13: 2 reads and a write, 3 times', (await text(`${A} [data-note="13"]`)).includes('×3: 6 shared reads, 3 shared writes'));
await setRange(A, 'COARSE_FACTOR', 8);
e = await expect(8, 'other');
check('factor 8: per element 6.75 and still 4', JSON.stringify(await cellsOf('per-element')) === JSON.stringify([e[0].per, e[1].per]) && e[1].per === 4, JSON.stringify(await cellsOf('per-element')));
await select(A, 'thread', 'first');
e = await expect(8, 'first');
check('thread 0 skips the add in Fig. 11.12 only', JSON.stringify(await cellsOf('per-thread')) === JSON.stringify([e[0].total, e[1].total]) && e[0].total === 38, JSON.stringify(await cellsOf('per-thread')));
check('thread 0 skips line 26 in Fig. 11.12', (await text(`${A} [data-note="26"]`)).includes('thread 0 skips it'));
if (out) await page.screenshot({ path: out + '-access.png' });

// ---------------------------------------------------------------- Bank map
const B = bench('Bank map');
const degree = () => page.$eval(`${B} svg[data-degree]`, (x) => Number(x.dataset.degree));
const bankCount = (b) => page.$eval(`${B} [data-bank="${b}"]`, (x) => Number(x.dataset.count));
const want = (F, banks, pad, c = 0, side = 'scan') => model((m, F, banks, pad, c, side) => m.bankMap(F, banks, pad, { c, side, blockDim: 256 }).degree, F, banks, pad, c, side);
check('factor 4, 32 banks: the model\'s 4-way conflict', (await degree()) === (await want(4, 32, false)) && (await degree()) === 4);
await setRange(B, 'COARSE_FACTOR', 8);
check('factor 8: 8-way, bank 0 serves 8 words', (await degree()) === 8 && (await bankCount(0)) === 8);
check('the narration names lanes 0, 4, …, 28 and gcd(8, 32)', (await text(`${B} [data-bank-says]`)).includes('lanes 0, 4, 8, 12, 16, 20, 24, 28') && (await text(`${B} [data-bank-says]`)).includes('gcd(8, 32) = 8'));
const padBox = () => page.evaluate((B) => [...document.querySelectorAll(`${B} label.check`)].find((l) => l.textContent.includes('pad')).querySelector('input').click(), B);
await padBox();
await pause(120);
check('padding: no conflict at factor 8', (await degree()) === 1 && (await want(8, 32, true)) === 1);
await setRange(B, 'COARSE_FACTOR', 3);
const worst = await page.$eval(`${B} [data-bank-says]`, (x) => Number(x.dataset.worst));
check('padding at factor 3 leaves a 2-way conflict in some iteration', worst === 2 && worst === (await model((m) => m.worstDegree(3, 32, true))));
await padBox();
await pause(120);
check('factor 3 unpadded: conflict-free', (await degree()) === 1 && (await page.$eval(`${B} [data-bank-says]`, (x) => Number(x.dataset.worst))) === 1);
await setRange(B, 'COARSE_FACTOR', 8);
await select(B, 'banks', '64');
check('64 banks, factor 8: the model\'s 4-way', (await degree()) === (await want(8, 64, false)) && (await degree()) === 4);
check('64 banks are drawn', (await page.$$(`${B} [data-bank]`)).length === 64);
await clickText(B, 'load side: buffer_s[c·BLOCK_DIM + t]');
check('load side: conflict-free', (await degree()) === 1);
if (out) await page.screenshot({ path: out + '-banks.png' });

// ---------------------------------------------------------------- listings
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('Fig. 11.13 as printed, 37 lines, 10 marked', tabs.includes('Fig. 11.13 as printed') && (await page.$$('.code pre .ln')).length === 37 && (await page.$$('.code pre .ln.hl')).length === 10, tabs.join('|'));
check('lab 11.2 tabs once the lab exists', tabs.includes('lab 11.2 (CUDA and HIP)') && tabs.includes('lab 11.2 (Metal)'));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'lab 11.2 (CUDA and HIP)').click());
await pause(100);
check('the lab tab is RegisterTiledScanKernel', (await text('.code pre')).includes('RegisterTiledScanKernel') && (await text('.code pre')).includes('#pragma unroll'));

// ---------------------------------------------------------------- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((x) => x.textContent.includes(n)), needle);
const shared = await q('how many accesses to buffer_s does Fig. 11.13 make');
await (await shared.$('input[type=number]')).type('4');
await (await shared.$('button.primary')).click();
await pause(120);
check('q-shared accepts 4', (await shared.evaluate((x) => x.textContent)).includes('Correct.'));
const bank = await q('How many lanes land in the same bank?');
await (await bank.$('input[type=number]')).type('8');
await (await bank.$('button.primary')).click();
await pause(120);
check('q-bank accepts 8', (await bank.evaluate((x) => x.textContent)).includes('Correct.'));
const odd = await q('which coarsening factors make the scan-side access');
await (await odd.$$('input[type=radio]'))[0].click();
await (await odd.$('button.primary')).click();
await pause(120);
check('q-odd: the odd factors', (await odd.evaluate((x) => x.textContent)).includes('Correct.'));

const forgiven = errors.filter((x) => !(x.includes('404') && labMisses-- > 0));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch11/11.7', [
  { label: 'lab 11.2 (CUDA and HIP)', lab: 'ch11/lab2_coarsened', file: 'kernel.cu' },
  { label: 'lab 11.2 (Metal)', lab: 'ch11/lab2_coarsened', file: 'kernel.metal' },
]);

check('no console or page errors', forgiven.length === 0, forgiven.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
