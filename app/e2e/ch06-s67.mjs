// Browser check for section 6.7: the exchange listing, the double-buffer stepper and
// bug hunt, the outcome table, the tiled-kernel listings and the questions.
// Expected values are computed by the model the page runs on (src/sim/doublebuf.ts),
// imported directly. The narrated scene on this page has its own check,
// ch06-scene-double-buffer.mjs.
// Usage: node e2e/ch06-s67.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { KIND_LABEL, SCHEDULES, VARIANTS, firstFailure, run } from '../src/sim/doublebuf.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

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
  await pause(60);
  return done;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};

await goto(page, 'ch06/6.7');
const S = bench('Step through the exchange');
await page.waitForSelector(S);
await page.waitForSelector('figure.code .tabs button');

// ---- the listing
const first = await text('figure.code pre');
check('the exchange listing is the snippet, its one-buffer loop highlighted',
  first.includes('swapPointers(inBuffer, outBuffer);') && (await page.$$eval('figure.code', (f) => f[0].querySelectorAll('.ln.hl').length)) === 6);

// ---- stepper
const narr = () => text(S + ' .narrate');
const hl = () => text(S + ' .code .ln.hl');
const result = () => text(S + ' [data-k="result"]');
const toEnd = async () => {
  for (let i = 0; i < 60; i++) if (!(await clickText(S, 'Run to next barrier'))) break;
  for (let i = 0; i < 80; i++) if (!(await clickText(S, 'Step'))) break;
};
const cfg = (variant) => ({ threads: 4, iterations: 3, variant });
let tr = run(cfg('single-both'), 'in-order');
await clickText(S, 'Step');
let n = await narr();
const s0 = tr.steps[0];
check("step 1 is thread 0 reading its neighbour's slot, with what the model found there",
  s0.op === 'read' && n.startsWith(`Thread 0, iteration 0: reads its neighbour's slot ${s0.read.slot} and finds ${s0.read.value} (the initial value).`), n);
check('the highlighted line is the read', (await hl()).includes('float v = buffer[other];'));
await clickText(S, 'Step');
check('step 2: thread 0 waits at the barrier after the read', (await narr()).includes('reaches the barrier after the read') && (await hl()).includes('// after the read'));
await clickText(S, 'Run to next barrier');
check('run to next barrier ends with the barrier opening', (await narr()).startsWith('The barrier after the read opens'), await narr());
await toEnd();
check('both barriers, in order: every thread hands back the right value', (await result()).includes('Every thread hands back the right value.') && (await result()).includes(`Handed back: ${tr.output.join(', ')}`), await result());
check('there is no failure to go to', !(await page.evaluate((S) => [...document.querySelector(S).querySelectorAll('button')].some((b) => b.textContent === 'Go to the first failure'), S)));

// No barrier after the read: write-after-read.
await setSelect(S + ' select[aria-label="kernel"]', 'single-no-read-barrier');
tr = run(cfg('single-no-read-barrier'), 'in-order');
check('removing the barrier after the read offers a failure to go to', await clickText(S, 'Go to the first failure'));
n = await narr();
check('the first failure is the model\'s: a write-after-read', tr.violations[0].kind === 'write-after-read' && n.includes(`Failure (${KIND_LABEL['write-after-read']}): ${tr.violations[0].message}.`), n);
check('the stepper stopped just after it', (await text(S + ' [aria-live="polite"]')) === `step ${firstFailure(tr)} of ${tr.steps.length}`);
check('the slot read too late is marked', (await page.$$(S + ' .mx.shared span.bad')).length === 1);
if (out) await page.screenshot({ path: out + '-war.png' });
await toEnd();
check(`and the result is wrong for ${tr.wrong.length} of 4 threads`, (await result()).includes(`${tr.wrong.length} of 4 threads hand back a wrong value.`), await result());

// Lockstep hides it.
await setSelect(S + ' select[aria-label="thread order"]', 'lockstep');
await toEnd();
check('in lockstep the same kernel gives the right answer', (await result()).includes('Every thread hands back the right value.'), await result());

// No barrier after the write: read-after-write.
await setSelect(S + ' select[aria-label="thread order"]', 'in-order');
await setSelect(S + ' select[aria-label="kernel"]', 'single-no-write-barrier');
tr = run(cfg('single-no-write-barrier'), 'in-order');
await clickText(S, 'Go to the first failure');
n = await narr();
check('without the barrier after the write the first failure is a read-after-write', tr.violations[0].kind === 'read-after-write' && n.includes(tr.violations[0].message), n);

// Two buffers.
await setSelect(S + ' select[aria-label="kernel"]', 'double');
tr = run(cfg('double'), 'in-order');
check('two buffers: buffers A and B are drawn', (await page.$$(S + ' .mx.shared')).length === 2);
const releaseAt = tr.steps.findIndex((s) => s.op === 'release');
await clickText(S, 'Run to next barrier');
await clickText(S, 'Step');
n = await narr();
check('after the barrier, thread 0 swaps its own pointers: it now reads B and writes A', tr.steps[releaseAt + 1].op === 'swap' && n.includes('swaps its own two pointers. From now on it reads buffer B and writes buffer A.'), n);
const ptr = await page.$$eval(S + ' table.threads tbody tr', (r) => r.map((x) => x.children[2].textContent.trim()));
check('the thread table shows only thread 0 swapped so far', ptr[0] === 'B, A' && ptr.slice(1).every((p) => p === 'A, B'), ptr.join(' | '));
await toEnd();
check('two buffers, one barrier: correct, with no failure to go to', (await result()).includes('Every thread hands back the right value.') && !(await page.evaluate((S) => [...document.querySelector(S).querySelectorAll('button')].some((b) => b.textContent === 'Go to the first failure'), S)));
if (out) await page.screenshot({ path: out + '-double.png' });

// Forgotten swap: wrong even in lockstep, and not a race.
await setSelect(S + ' select[aria-label="kernel"]', 'double-no-swap');
await setSelect(S + ' select[aria-label="thread order"]', 'lockstep');
tr = run(cfg('double-no-swap'), 'lockstep');
await clickText(S, 'Go to the first failure');
check('a forgotten swap fails in lockstep too, as a wrong buffer', (await narr()).includes('wrong buffer (no race)') && tr.violations[0].kind === 'wrong-buffer', await narr());
await toEnd();
check('every thread hands back a wrong value', (await result()).includes('4 of 4 threads hand back a wrong value.'));

// ---- outcome table
const O = bench('Every thread order at once');
check('the table is hidden until a prediction is made', !(await page.$(O + ' tbody')));
await page.evaluate((O) => {
  for (const l of document.querySelectorAll(O + ' label.check'))
    if (['one buffer, both barriers', 'two buffers, one barrier'].includes(l.textContent.trim())) l.querySelector('input').click();
}, O);
await pause(80);
await clickText(O, 'Reveal the table');
check('predicting the two correct kernels is confirmed', (await text(O + ' [data-k="prediction"]')).startsWith('as you predicted'), await text(O + ' [data-k="prediction"]'));
const table = await page.$$eval(O + ' tbody tr', (rows) => rows.map((r) => ({ v: r.dataset.variant, cells: [...r.querySelectorAll('td[data-outcome]')].map((c) => ({ o: c.dataset.outcome, k: c.dataset.kinds, t: c.textContent })) })));
const model = VARIANTS.map((v) => ({ v, cells: SCHEDULES.map((s) => run(cfg(v), s)) }));
check('every cell matches the model: outcome, number wrong and failure kinds',
  table.length === 6 && model.every((m, i) => table[i].v === m.v && m.cells.every((r, k) => {
    const c = table[i].cells[k];
    const kinds = [...new Set(r.violations.map((x) => x.kind))].join(' ');
    return c.o === (r.correct ? 'correct' : 'wrong') && c.k === kinds && (r.correct || c.t.startsWith(`${r.wrong.length} of 4 wrong`));
  })), JSON.stringify(table.map((r) => r.cells.map((c) => c.o))));
const row = (v) => table.find((r) => r.v === v).cells;
check('the double-buffered row is correct under every order', row('double').every((c) => c.o === 'correct'));
check("'no barrier after the read' fails in order, with a write-after-read", row('single-no-read-barrier')[1].o === 'wrong' && row('single-no-read-barrier')[1].k === 'write-after-read');
check('lockstep hides every missing barrier, but not the forgotten swap',
  ['single-no-read-barrier', 'single-no-write-barrier', 'double-no-barrier'].every((v) => row(v)[0].o === 'correct') && row('double-no-swap')[0].o === 'wrong');
await setSelect(O + ' select[aria-label="threads in the table"]', 8);
const eight = await page.$$eval(O + ' tbody tr[data-variant="double"] td[data-outcome]', (c) => c.map((x) => x.dataset.outcome));
check('with 8 threads the double-buffered row is still correct everywhere', eight.length === 4 && eight.every((o) => o === 'correct'));
if (out) await page.screenshot({ path: out + '-table.png' });

// ---- the tiled listings
const tabs = await page.$$eval('figure.code .tabs', (t) => t.map((x) => [...x.querySelectorAll('button')].map((b) => b.textContent)));
const tiled = tabs.findIndex((t) => t.includes('Double-buffered (CUDA)'));
check('the tiled listing offers chapter 5, the double-buffered CUDA and Metal kernels', tiled >= 0 && ['Chapter 5: two barriers', 'Double-buffered (CUDA)', 'Double-buffered (Metal)'].every((l) => tabs[tiled].includes(l)), JSON.stringify(tabs));
const fig = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Double-buffered (CUDA)')));
const figText = () => fig.evaluate((f) => f.querySelector('pre').textContent);
check('the first tab is lab 5.2\'s kernel with two barriers', ((await figText()).match(/__syncthreads\(\)/g) ?? []).length === 2);
await fig.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Double-buffered (CUDA)').click());
await pause(80);
const cu = await figText();
check("the CUDA tab is lab 6.5's reference solution, with one barrier in the loop", cu.includes('doubleBufferedMatmulKernel') && (cu.match(/__syncthreads\(\);/g) ?? []).length === 1, cu.slice(0, 120));
await fig.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Double-buffered (Metal)').click());
await pause(80);
const status = await fig.evaluate((f) => f.querySelector('.tabs .status')?.textContent ?? '');
check('the Metal tab is the Metal kernel, its status in the fixed vocabulary', (await figText()).includes('threadgroup_barrier') && /^(verified on Metal|unverified: no run on this machine|unverified: failed on this machine)$/.test(status), status);

// ---- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qs = await q('double-buffered version need');
await (await qs.$('input[type=number]')).type('16384');
await (await qs.$('button.primary')).click();
await pause(120);
check('q-smem accepts 16384 bytes', (await qs.evaluate((e) => e.textContent)).includes('Correct.'));
// The limit it is compared with is the reader's, from this machine's reports, never a figure in the text.
const rep = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const want = Object.entries(rep).filter(([b]) => b !== 'emu' && b !== 'race').flatMap(([b, run]) => {
  const v = Number(run.device.maxThreadgroupMemoryLength ?? run.device.sharedMemPerBlock);
  return isFinite(v) && v > 0 ? [[b, String(v)]] : [];
});
const shown = await qs.evaluate((e) => [...e.querySelectorAll('[data-smem-limit]')].map((x) => [x.dataset.backend ?? '', x.dataset.smemLimit, x.textContent]));
const smemText = await qs.evaluate((e) => e.textContent);
check(`q-smem: the per-block limit comes from the reports (${want.map((w) => w.join(' ')).join(', ') || 'none reported'}), and no size is printed in the text`,
  (want.length
    ? JSON.stringify(shown.map(([b, v]) => [b, v])) === JSON.stringify(want) && shown.every(([, v, t]) => t.includes(Number(v) >= 16384 ? 'fit, with' : 'do not fit'))
    : shown.length === 1 && shown[0][1] === 'none' && shown[0][2].includes('pmpp.py doctor'))
  && !smemText.includes('32 KB') && smemText.includes('maxThreadgroupMemoryLength') && smemText.includes('sharedMemPerBlock'), JSON.stringify(shown));
const choose = async (needle, label) => {
  const h = await q(needle);
  await h.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const which = await choose('which barrier do two buffers make unnecessary', 'The one after the read');
check('q-which: the barrier after the read, and the erratum is explained', which.includes('Correct.') && which.includes('says the "second"'));
check('q-swap: "a race" is wrong, with the reason', (await choose('forgets the swap', 'A race that only some')).includes('wrong there too'));
check('q-async: "only the reads that cross warps" is wrong: the rule is per thread',
  (await choose('Which reads can see stale data', 'Only the two reads that cross')).includes('per thread, not per warp'));
const qa = await choose('Which reads can see stale data', 'Every read of tile[(t + 1) % 64]');
check('q-async: every neighbour read is the answer, with NVIDIA\'s rule quoted', qa.includes('Correct.') && qa.includes('each thread only waits for its own LDGSTS copies'), qa);
check('answers went to the (intercepted) progress API', puts >= 1, String(puts));

// ---- the scene is on the page
check('the narrated scene "Two buffers, one barrier" is on the page', !!(await page.$('section.scene[aria-label="Two buffers, one barrier"]')));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|double-buffer-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch06/6.7', [
  { label: 'Chapter 5: two barriers', lab: 'ch05/lab2_tiled', file: 'kernel.cu' },
  { label: 'Double-buffered (CUDA)', lab: 'ch06/lab5_doublebuf', file: 'kernel.cu' },
  { label: 'Double-buffered (Metal)', lab: 'ch06/lab5_doublebuf', file: 'kernel.metal' },
]);

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
