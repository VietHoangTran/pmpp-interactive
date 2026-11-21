// Browser check for section 14.5: the "One pass, thread by thread" stepper (every state compared
// with src/sim/radixsort.ts parallelPass), its predict mode and its four broken kernels, the
// listings, and the five questions.
// Usage: APP_URL=http://localhost:5173 node e2e/ch14-s145.mjs [--dark] [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { BOOK_KEYS, bin, parallelPass, randomKeys } from '../src/sim/radixsort.ts';

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const B = 'section.bench[aria-label="One pass, thread by thread"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (label) => {
  const ok = await page.evaluate((B, label) => {
    const b = [...document.querySelectorAll(B + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, B, label);
  await pause(80);
  return ok;
};
const select = async (label, value) => {
  await page.evaluate((B, label, value) => {
    const l = [...document.querySelectorAll(B + ' label.field')].find((x) => x.textContent.trim().startsWith(label));
    const s = l.querySelector('select');
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, B, label, value);
  await pause(100);
};
const steps = async (n) => { for (let k = 0; k < n; k++) await press('Step'); };
const narr = () => text(B + ' .narrate');
const hl = () => page.$$eval(B + ' .code .ln.hl', (ls) => ls.map((l) => Number(l.dataset.n)));
const regs = (col) => page.$$eval(`${B} td[data-col="${col}"]`, (tds) => tds.map((td) => td.textContent.trim()));
const outSlots = () => page.$$eval(B + ' g[data-out-slot]', (gs) => gs.map((g) => [g.querySelector('text').textContent, g.dataset.outState]));
const outcome = () => text(B + ' [data-part="outcome"]');

await goto(page, 'ch14/14.5');
await page.waitForSelector(B);
const pp = parallelPass(BOOK_KEYS, 0, 4);

// ---------------------------------------------------------------- following thread 6 (block 1, thread 2)
check('nothing has run at first', (await narr()).startsWith('Nothing has run. 16 keys'));
await steps(1);
check('step 1: thread 2 of block 1 computes i = 6', (await narr()).includes('= 1·4 + 2 = 6'), await narr());
check('step 1: the launch line and the index line are highlighted', JSON.stringify(await hl()) === '[3,32]', JSON.stringify(await hl()));
await steps(3);
check("step 4: bits[6] = 1, and the bits row is Fig. 14.6's", (await narr()).includes('stores bits[6] = 1') &&
  JSON.stringify(await page.$$eval(B + ' g[data-bits-slot] text', (ts) => ts.slice(0, 16).map((t) => Number(t.textContent)))) === JSON.stringify(pp.bits));
await steps(1);
check('step 5: the scan is one wall across every block', !!(await page.$(B + ' [data-wall]')) && JSON.stringify(await hl()) === '[33]');
const scanned = await page.$$eval(B + ' g[data-bits-slot] text', (ts) => ts.map((t) => Number(t.textContent)));
check("step 5: the scanned array is Fig. 14.6's ones before, plus entry 16 = 8", JSON.stringify(scanned) === JSON.stringify(pp.scan), JSON.stringify(scanned));
if (out) await page.screenshot({ path: out + '-scan.png' });
await steps(1);
check('step 6: kernel 2 reloads the key and recomputes the bit', (await narr()).includes('recomputes its bit, 1') && JSON.stringify(await hl()) === '[14,16,17,34,35]', JSON.stringify(await hl()));
await steps(1);
check('step 7: ones before 3, total 8', (await narr()).includes('numOnesBefore = bits[6] = 3; numOnesTotal = bits[16] = 8'), await narr());
await steps(1);
check('step 8: a one, so 16 − 8 + 3 = 11', (await narr()).includes('16 − 8 + 3 = 11'), await narr());
check('step 8: the register table holds every ones-before and destination', JSON.stringify(await regs('before')) === JSON.stringify(pp.threads.map((t) => String(t.onesBefore))) && JSON.stringify(await regs('dst')) === JSON.stringify(pp.threads.map((t) => String(t.dst))));
await steps(1);
check('step 9: thread 6 writes output[11] = 0101', (await narr()).includes('writes output[11] = 0101'));
check('step 9: the output row is pass 1 of Fig. 14.4, every slot written once', JSON.stringify((await outSlots()).map((x) => x[0])) === JSON.stringify(pp.out.map((k) => bin(k, 4))) && (await outSlots()).every((x) => x[1] === 'ok'));
check('step 9: 16 arrows, to the model\'s destinations', JSON.stringify(await page.$$eval(B + ' line[data-arrow]', (ls) => ls.map((l) => Number(l.dataset.to)))) === JSON.stringify(pp.threads.map((t) => t.dst)));
check('the outcome: the stable partition', (await outcome()).includes('every slot written once'));
// follow another thread by clicking its column
await page.$eval(`${B} td[data-thread="13"][data-col="key"]`, (td) => td.click());
await pause(100);
check('clicking column 13 follows thread 1 of block 3: output[6] = 1010', (await narr()).includes('Thread 1 of block 3 writes output[6] = 1010'), await narr());
if (out) await page.screenshot({ path: out + '-end.png' });

// ---------------------------------------------------------------- predict first
await press('Restart');
await page.evaluate((B) => [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes('predict first')).querySelector('input').click(), B);
await pause(100);
const inputs = await page.$$eval(B + ' td input[type=number]', (is) => is.map((i) => i.getAttribute('aria-label')));
check('predict mode: eight boxes, for threads 2, 6, 11 and 13', inputs.length === 8 && [2, 6, 11, 13].every((i) => inputs.includes(`thread ${i} ones before`) && inputs.includes(`thread ${i} destination`)), JSON.stringify(inputs));
await press('Run to the end');
check('predict mode: their values stay hidden after the scan', (await page.$$eval(B + ' g[data-bits-slot] text', (ts) => ts.slice(0, 16).map((t) => t.textContent))).filter((v) => v === '?').length === 4);
const fill = async (i, what, v) => {
  const sel = `${B} input[aria-label="thread ${i} ${what}"]`;
  await page.$eval(sel, (el) => { el.focus(); });
  await page.type(sel, String(v));
};
for (const i of [2, 6, 11, 13]) {
  await fill(i, 'ones before', pp.threads[i].onesBefore);
  await fill(i, 'destination', i === 13 ? 99 : pp.threads[i].dst);
}
await press('Check my predictions');
check('one wrong destination: the check says some are off', (await text(B + ' [data-part="predict"]')).includes('some numbers are off'));
check('...and marks exactly that one wrong', (await page.$$(B + ' [data-grade="wrong"]')).length === 1 && (await page.$$(B + ' [data-grade="right"]')).length === 7);
check('after the check the model\'s numbers are shown', (await regs('dst'))[13].startsWith(String(pp.threads[13].dst)));
// predict again, all right
await page.evaluate((B) => [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes('predict first')).querySelector('input').click(), B);
await page.evaluate((B) => [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes('predict first')).querySelector('input').click(), B);
await pause(100);
await page.$eval(`${B} input[aria-label="thread 13 destination"]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await fill(13, 'destination', pp.threads[13].dst);
await press('Check my predictions');
check('all eight right: the check says so', (await text(B + ' [data-part="predict"]')).includes('all eight numbers right'));
if (out) await page.screenshot({ path: out + '-predict.png' });
await page.evaluate((B) => [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes('predict first')).querySelector('input').click(), B);
await pause(100);

// ---------------------------------------------------------------- the broken kernels
const runBroken = async (defect) => {
  await select('Kernel', defect);
  await press('Run to the end');
  return parallelPass(BOOK_KEYS, 0, 4, defect);
};
let m = await runBroken('zero-formula');
let o = await outcome();
check('ones use the zero formula: collisions and holes, as the model', o.includes(`${m.collisions.length} slots written twice (${m.collisions.join(', ')})`) && o.includes(`${m.holes.length} never written (${m.holes.join(', ')})`), o);
check('...and the collided slots are drawn', (await outSlots()).filter((x) => x[1] === 'collision').length === m.collisions.length);
if (out) await page.screenshot({ path: out + '-collisions.png' });
m = await runBroken('scan-n');
o = await outcome();
check('scan of n entries: 16 out-of-range reads of bits[16]', o.includes('16 reads of bits[16] out of range'), o);
check('...the total is "??" for every thread, the ones\' destinations "?"', (await regs('total')).every((v) => v === '??') && (await regs('dst')).filter((v) => v === '?').length === m.unknown.length);
m = await runBroken('inclusive');
o = await outcome();
check('inclusive scan: one write out of range and slot 8 never written', o.includes('1 never written (8)') && o.includes('1 write out of range'), o);
m = await runBroken('last-as-total');
o = await outcome();
check('total from the last of n entries: right on the book keys, and the page says why', o.includes('every slot written once') && o.includes("last key's bit is 0"), o);
// random keys: find a shuffle whose last key has bit 0 = 1 (the stepper starts at seed 3 and adds 1 per shuffle)
await select('Keys', 'random');
let seed = 3;
while ((randomKeys(16, 8, seed)[15] & 1) === 0) {
  await press('Shuffle');
  seed++;
}
await press('Run to the end');
m = parallelPass(randomKeys(16, 8, seed), 0, 4, 'last-as-total');
o = await outcome();
check(`with random keys whose last bit is 1 (seed ${seed}) it breaks: a write out of range and a hole`, !m.correct && o.includes('1 write out of range') && o.includes(`never written (${m.holes.join(', ')})`), o);
check('the keys shown are the model\'s random keys', JSON.stringify(await regs('key')) === JSON.stringify(randomKeys(16, 8, seed).map((k) => bin(k, 8))));
await select('Kernel', 'none');
await select('Bit', 5);
await press('Run to the end');
const r5 = parallelPass(randomKeys(16, 8, seed), 5, 4);
check('random keys, bit 5, corrected kernel: the model\'s destinations and a correct pass', JSON.stringify(await regs('dst')) === JSON.stringify(r5.threads.map((t) => String(t.dst))) && (await outcome()).includes('every slot written once'));

// ---------------------------------------------------------------- listings
const codes = await page.$$eval('.code', (cs) => cs.map((c) => c.textContent));
check('the scan listing is shown with its status', codes.some((c) => c.includes('exclusive_scan_kernel(unsigned int* bits, unsigned int N)') && c.includes('emulator-checked')));
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const lab1 = fs.existsSync(path.join(repo, 'labs/ch14/lab1_twosorts/solution/kernel.cu'));
const tabs = await page.$$eval('.code .tabs button', (bs) => bs.map((b) => b.textContent));
check(lab1 ? "lab 14.1's reference is shown in tabs" : 'lab 14.1 is not written yet, so no lab listing', lab1 ? tabs.includes('CUDA (and HIP)') : !tabs.includes('CUDA (and HIP)'), JSON.stringify(tabs));

// ---------------------------------------------------------------- questions
const question = (t) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), t);
const numeric = async (t, v) => {
  const q = await question(t);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const choose = async (t, start) => {
  const q = await question(t);
  await q.evaluate((e, s) => [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().startsWith(s)).querySelector('input').click(), start);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
check('q-dst-one: 17', (await numeric('has bit 1 and 6 ones before it', 17)).includes('Correct.'));
check('q-dst-zero: 7', (await numeric('now the key at index 13 has bit 0', 7)).includes('Correct.'));
check('q-total: N + 1 entries', (await choose('Why is a scan of the N bits not enough', 'Its last entry counts the ones before the last key')).includes('Correct.'));
check('q-runs: 2 runs', (await numeric('Into how many contiguous runs', 2)).includes('Correct.'));
check('q-split: blocks are not resident together or ordered', (await choose("Why is lab 14.1's pass three launches", 'Every block must write its bits')).includes('Correct.'));

// ---------------------------------------------------------------- phone width
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

// A lab file that is not written yet answers 404; that is the lab's, not this page's.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
