// Browser check for section 10.9: the lab listing, the segment calculator, the commit-order
// explorer, the prediction and its reveal (with this machine's reports, then with injected
// ones), and the questions. Expected values come from src/sim/multiblock.ts, imported here.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s109.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { blockSums, blockView, commit, f32hex, f32short, history, logits, runOf, segments, threadLoads } from '../src/sim/multiblock.ts';

const out = process.argv[2];
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtInt = (n) => n.toLocaleString('en-US');
const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|commit-order-\d\.mp3)$/.test(u);

/** A page with progress writes swallowed and, optionally, /api/reports replaced. */
async function page0(reports) {
  const { browser, page, errors } = await open({ width: 1440, height: 1000 });
  const missing = [];
  page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });
  await page.setRequestInterception(true);
  page.on('request', (r) => {
    if (r.method() === 'PUT' && r.url().includes('/api/progress'))
      return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    if (reports && r.method() === 'GET' && r.url().includes('/api/reports'))
      return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(reports) });
    // A clean slate for predictions and answers, so the check does not depend on the reader's progress.
    if (r.method() === 'GET' && r.url().endsWith('/api/progress'))
      return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify({ version: 1, sections: {}, answers: {}, review: {} }) });
    return r.continue();
  });
  return { browser, page, errors, missing };
}
const realErrors = (errors, missing) => [
  ...errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e)),
  ...missing.filter((u) => !expected404(u)),
];

const { browser, page, errors, missing } = await page0(null);
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const click = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
/** Set a React-controlled input or select by its aria-label. */
const setField = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const el = document.querySelector(`${scope} [aria-label="${label}"]`);
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
};
const setCheck = async (scope, word, on) => {
  await page.evaluate((scope, word, on) => {
    const box = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, word, on);
  await pause(100);
};

await goto(page, 'ch10/10.9');
await page.waitForSelector(bench('Segment calculator'));

// ---------------------------------------------------------------- the listing
await page.waitForFunction(() => document.querySelectorAll('figure.code .tabs button').length >= 1, { timeout: 8000 }).catch(() => undefined);
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('listing offers the CUDA kernel, its host function and the Metal kernel', ['CUDA kernel', 'CUDA host', 'Metal kernel'].every((t) => tabs.includes(t)), tabs.join('|'));
const hl = await page.$$eval('figure.code .ln.hl', (l) => l.map((x) => x.textContent.trim()));
check('the CUDA tab marks exactly the commit lines: the HIP call, then the book\'s two', hl.length === 3 && hl[0].startsWith('atomicAdd(output') && hl[1].startsWith('cuda::atomic_ref') && hl[2].startsWith('output_ref.fetch_add'), JSON.stringify(hl));
check('the CUDA tab\'s status is the chapter\'s fixed wording', (await text('figure.code .status')) === 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware', await text('figure.code .status'));
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')].find((b) => b.textContent.trim() === 'CUDA host').click());
await pause(80);
const hlHost = await page.$$eval('figure.code .ln.hl', (l) => l.map((x) => x.textContent.trim()));
check('the host tab marks the zeroing of the total', hlHost.length === 1 && hlHost[0].startsWith('cudaMemset(out, 0'), JSON.stringify(hlHost));
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')].find((b) => b.textContent.trim() === 'Metal kernel').click());
await pause(80);
const hlMetal = await page.$$eval('figure.code .ln.hl', (l) => l.map((x) => x.textContent.trim()));
const metalSrc = await text('figure.code pre');
check('the Metal tab is the kernel with atomic_fetch_add_explicit, marked', metalSrc.includes('kernel void coarsened_sum(') && hlMetal.length === 1 && hlMetal[0].startsWith('atomic_fetch_add_explicit'), JSON.stringify(hlMetal));
check('the Metal tab\'s status is "verified on Metal"', (await text('figure.code .status')) === 'verified on Metal', await text('figure.code .status'));

// ---------------------------------------------------------------- segment calculator
const SC = bench('Segment calculator');
const seg = segments(1_000_000, 1024);
check('default: 1,000,000 elements need 489 blocks', (await text(SC + ' [data-testid=blocks]')) === fmtInt(seg.blocks), await text(SC + ' [data-testid=blocks]'));
check('default selection is the last block, 488', (await text(SC + ' [data-testid=block]')) === '488');
check('the last block holds 576 real elements (q-lastblock)', (await text(SC + ' [data-testid=last-real]')) === '576');
check('the selected block holds 576 real elements and 448 idle threads', (await text(SC + ' [data-testid=facts]')).includes('448 of its threads have nothing to load'));
const loads0 = threadLoads(1_000_000, 1024, 1, 488, 0);
await click(SC, 'Step');
let n = await text(SC + ' .narrate');
check('step 1 shows load 0 with its index arithmetic', n.includes(`= 999,424 + 0 + 0·1024 = ${fmtInt(loads0[0].index)}`) && n.includes(`adds input[${fmtInt(loads0[0].index)}]`), n);
await click(SC, 'Step');
n = await text(SC + ' .narrate');
check('step 2: load 1 is past N and the guard adds 0', n.includes(fmtInt(loads0[1].index)) && n.includes('the guard makes the thread add 0'), n);
const rows = await page.$$eval(SC + ' tr[data-load]', (r) => r.map((x) => [Number(x.dataset.index), x.dataset.in === '1']));
check('the load table matches the model', JSON.stringify(rows) === JSON.stringify(loads0.map((l) => [l.index, l.inRange])), JSON.stringify(rows));
check('guarded: the verdict says idle threads add 0', (await text(SC + ' [data-testid=verdict]')).includes('idle threads still take part and add 0'));
if (out) await page.screenshot({ path: out + '-calc-guarded.png' });

await setCheck(SC, 'guard loads', false);
check('without the guard the last block reads 1,472 elements past the end', (await text(SC + ' [data-testid=verdict]')).includes(`reads ${fmtInt(seg.pastEnd)} elements past the end`));
check('and the table says out of bounds', (await text(SC + ' table')).includes('out of bounds'));
await click(SC, 'Step');
await click(SC, 'Step');
check('the narration marks the read as out of bounds', (await page.$eval(SC + ' .narrate', (e) => e.className)).includes('bad') && (await text(SC + ' .narrate')).includes('out of bounds'));
if (out) await page.screenshot({ path: out + '-calc-noguard.png' });
await setCheck(SC, 'guard loads', true);

await click(SC, 'First idle thread');
const idle = threadLoads(1_000_000, 1024, 1, 488, 576);
check('"First idle thread" picks thread 576, which loads nothing real', (await text(SC + ' [data-testid=thread]')) === '576' && idle.every((l) => !l.inRange));

await setField(SC, 'coarsening factor', 2);
const seg2 = segments(1_000_000, 1024, 2);
check('C = 2: ceil(1,000,000 / 4096) blocks, and the last block as the model says',
  (await text(SC + ' [data-testid=blocks]')) === fmtInt(seg2.blocks) && (await text(SC + ' [data-testid=last-real]')) === fmtInt(seg2.lastReal),
  `${await text(SC + ' [data-testid=blocks]')} / ${await text(SC + ' [data-testid=last-real]')}`);
check('C = 2: each thread has 4 loads', (await page.$$(SC + ' tr[data-load]')).length === 4);
await setField(SC, 'coarsening factor', 1);
await setField(SC, 'N', 4096);
check('N = 4096 (a multiple of 2048): 2 blocks, nothing past the end', (await text(SC + ' [data-testid=blocks]')) === '2' && (await text(SC + ' [data-testid=facts]')).includes('N is a multiple of the segment'));
await setField(SC, 'N', 1000);
await setField(SC, 'threads per block', 32);
const v = blockView(1000, 32, 1, segments(1000, 32).blocks - 1);
check('N = 1000, 32 threads: the last block as the model says', (await text(SC + ' [data-testid=real]')) === fmtInt(v.real) && (await text(SC + ' [data-testid=blocks]')) === fmtInt(segments(1000, 32).blocks));

// ---------------------------------------------------------------- commit order
const CE = bench('Blocks commit in any order');
const values = logits(65536, 2026);
const sums = blockSums(values, 256);
for (let i = 0; i < 19; i++) await click(CE, 'Run again');
const runs = await page.$eval(CE + ' [data-testid=history]', (e) => [Number(e.dataset.runs), Number(e.dataset.distinct)]);
const h = history(sums, 'atomic', 20);
check('20 atomic runs: the distinct count matches the model, and is more than 1', runs[0] === 20 && runs[1] === h.distinct.length && h.distinct.length > 1, JSON.stringify(runs));
const listed = await page.$$eval(CE + ' [data-testid=history] li', (l) => l.map((x) => x.textContent.trim().slice(0, 10)));
check('the history lists the model\'s bit patterns', JSON.stringify(listed) === JSON.stringify(h.distinct), JSON.stringify(listed));
const r19 = runOf(sums, 'atomic', 19);
check('run 20\'s total in hex is the model\'s', (await text(CE + ' [data-testid=total-hex]')) === f32hex(r19.total));
const firstRow = await page.$$eval(CE + ' tr[data-commit="0"] td', (t) => t.map((x) => x.textContent.trim()));
check('commit 1 of run 20: the block and its float32 sum', firstRow[1] === String(r19.order[0]) && firstRow[2] === f32short(sums[r19.order[0]]), JSON.stringify(firstRow));
check('the verdict says the same data gave different totals', (await text(CE + ' [data-testid=verdict]')).includes(`gave ${h.distinct.length} different totals`));
if (out) await page.screenshot({ path: out + '-commit-atomic.png' });

await click(CE, 'partials, then a fixed second pass');
check('switching design starts a new history', (await page.$eval(CE + ' [data-testid=history]', (e) => e.dataset.runs)) === '1');
for (let i = 0; i < 5; i++) await click(CE, 'Run again');
check('two-pass: 6 runs, 1 distinct total', (await page.$eval(CE + ' [data-testid=history]', (e) => e.dataset.distinct)) === '1');
check('two-pass total is the fixed tree\'s', (await text(CE + ' [data-testid=total-hex]')) === f32hex(commit(sums, 'fixed').total));
check('two-pass verdict: the same bits every run', (await text(CE + ' [data-testid=verdict]')).includes('the same bits every run'));

await click(CE, 'atomic commit (Fig. 10.18)');
await setField(CE, 'values', 4096);
await setField(CE, 'threads per block', 1024);
for (let i = 0; i < 5; i++) await click(CE, 'Run again');
const two = blockSums(logits(4096, 2026), 1024);
check('4,096 values in 1024-thread blocks: 2 blocks, 1 distinct total', two.length === 2 && (await page.$eval(CE + ' [data-testid=history]', (e) => e.dataset.distinct)) === '1');
check('and the explanation: float addition is commutative', (await text(CE + ' [data-testid=verdict]')).includes('Float addition is commutative'));

// ---------------------------------------------------------------- questions
const ask = async (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
let q = await ask('How many real elements does the last block');
await (await q.$('input[type=number]')).type('576');
await (await q.$('button.primary')).click();
await pause(120);
check('q-lastblock accepts 576', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await ask('launches the kernel of Fig. 10.18 twice');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith('Twice the sum')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('q-twice accepts "twice the sum"', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await ask('regression test compares the sum');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('memory_order_seq_cst')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('q-reproducible rejects seq_cst and says why', (await q.evaluate((e) => e.textContent)).includes('does not choose which block adds first'));

// ---------------------------------------------------------------- prediction, this machine's reports
let p = await ask('How many distinct results');
await (await p.$('input[type=number]')).type('1');
await p.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(300);
const reveal = await p.evaluate((e) => e.textContent);
check('the reveal shows this machine\'s count, or says how to measure it', reveal.includes('You predicted 1') && (/distinct results? (in|on)/.test(reveal) || reveal.includes('No measurement on this machine yet')), reveal.slice(0, 200));

check('no console or page errors (a missing narration manifest is allowed)', realErrors(errors, missing).length === 0, realErrors(errors, missing).join(' | '));
await browser.close();

// ---------------------------------------------------------------- prediction, injected reports
const row = (backend, distinct) => ({
  lab: 'ch10/lab1_sum', variant: 'solution', backend, tile: 0, built: true, test_rc: 0, tests: [{ case: 'x', pass: true }],
  bench: [{ name: 'determinism', n: 16777216, runs: 20, distinct }],
});
const fake = {
  metal: { generated: 'test', device: { name: 'Test device A (injected by this check)' }, results: [row('metal', 7)] },
  emu: { generated: 'test', device: { name: 'emulator (injected)' }, results: [row('emu', 1)] },
};
const second = await page0(fake);
await goto(second.page, 'ch10/10.9');
await second.page.waitForSelector('section.bench');
p = await second.page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many distinct results')));
await (await p.$('input[type=number]')).type('1');
await p.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(300);
const lines = await p.evaluate((e) => [...e.querySelectorAll('p[data-backend]')].map((x) => [x.dataset.backend, x.dataset.distinct, x.textContent.replace(/\s+/g, ' ')]));
const metalLine = lines.find((l) => l[0] === 'metal');
const emuLine = lines.find((l) => l[0] === 'emu');
check('injected: the device line shows 7 distinct results in 20 runs on 16,777,216 floats', !!metalLine && metalLine[1] === '7' && metalLine[2].includes('7 distinct results in 20 runs on 16,777,216 floats'), JSON.stringify(metalLine));
check('injected: a prediction of 1 is told the atomic kernel is not repeatable', !!metalLine && metalLine[2].includes('You expected the atomic kernel to be repeatable'));
check('injected: the emulator\'s 1 is explained by its fixed order', !!emuLine && emuLine[2].includes('it can only ever show 1'), JSON.stringify(emuLine));
if (out) await second.page.screenshot({ path: out + '-reveal.png' });
check('no console or page errors with injected reports', realErrors(second.errors, second.missing).length === 0, realErrors(second.errors, second.missing).join(' | '));
await second.browser.close();

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
