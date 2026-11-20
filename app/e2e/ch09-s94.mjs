// Browser check for section 9.4 (privatisation): the explorer "Copies and contenders",
// the two lab listings, the bug hunt "Seven broken privatised kernels" with its outcome
// table, the fit table, the questions and the prediction. Expected numbers are the ones
// src/sim/histpriv.test.ts pins (that model imports another module without a file
// extension, so it is not imported here directly).
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-s94.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const E = bench('Copies and contenders');
const H = bench('Seven broken privatised kernels');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const select = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${label}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(150);
};
const typeInto = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(150);
};
const clickIn = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(150);
  return done;
};
const readout = () => page.$eval(E + ' [data-readout]', (e) => ({ ...e.dataset, text: e.textContent.replace(/\s+/g, ' ') }));

await goto(page, 'ch09/9.4');
await page.waitForSelector(E);

// ---------------------------------------------------------------- explorer
await select(E, 'input', 'constant');
await select(E, 'elements', 4096);
await select(E, 'threads per block', 256);
await select(E, 'bins', '256');
await select(E, 'budget', 'custom');
check('explorer: the custom budget says where its starting 48 comes from', (await text(E + ' [data-part="custom-default"]')).includes("CUDA's documented per-block default without opt-in (ch09-facts A8)"));
await typeInto(E + ' input[aria-label="custom budget"]', 32);
const want = {
  'one public copy': { scheme: 'public', copies: '1', contention: '4096', commit: '0', cc: '0' },
  'two copies': { scheme: 'two', copies: '2', contention: '2048', commit: '2', cc: '2' },
  'four copies': { scheme: 'four', copies: '4', contention: '1024', commit: '4', cc: '4' },
  'one per block': { scheme: 'block', copies: '16', contention: '256', commit: '16', cc: '16' },
};
for (const [label, w] of Object.entries(want)) {
  check(`explorer: the "${label}" button exists`, await clickIn(E, label));
  const r = await readout();
  check(`explorer, constant image, ${label}: ${w.copies} copies, longest queue ${w.contention}, ${w.commit} commit atomics, ${w.cc} per public bin`,
    r.scheme === w.scheme && r.copies === w.copies && r.contention === w.contention && r.commit === w.commit && r.commitContention === w.cc,
    JSON.stringify(r));
}
let r = await readout();
check('explorer: one per block says 16x shorter and fits in shared memory', r.text.includes('16× shorter') && r.fits === 'yes' && r.text.includes('Fig. 9.10'), r.text);
const rows = await page.$$eval(E + ' tbody tr[data-row]', (trs) => trs.map((t) => [...t.querySelectorAll('td')].map((d) => d.textContent.trim())));
check('explorer: the comparison table lists all four schemes with the same queues', JSON.stringify(rows.map((x) => x[2])) === JSON.stringify(['4,096', '2,048', '1,024', '256']), JSON.stringify(rows));
if (out) await page.screenshot({ path: out + '-explorer.png' });

await select(E, 'bins', 'qwen3');
r = await readout();
check('explorer: Qwen3-8B against a 32 KiB budget does not fit, so the copies stay in global memory',
  r.fits === 'no' && r.text.includes('does not fit') && r.text.includes('593.5 KiB') && r.text.includes('Fig. 9.9'), r.text);
check('explorer: with token ids the input labels switch to tokens', (await text(E)).includes('one token repeated'));
await select(E, 'bins', '256');
await select(E, 'input', 'uniform');
r = await readout();
check('explorer: uniform noise still fits, and the commit is at most copies x bins', r.fits === 'yes' && r.text.includes('at most copies × bins = 4,096'), r.text);

// ---------------------------------------------------------------- listings
const codes = await page.$$eval('figure.code', (fs) => fs.map((f) => ({ tabs: [...f.querySelectorAll('.tabs button')].map((b) => b.textContent), code: f.querySelector('pre').textContent })));
const pool = codes.find((c) => c.code.includes('histogramPool'));
check('Version 1 shows lab 9.1\'s pool kernel with CUDA and Metal tabs', !!pool && pool.tabs.includes('CUDA and HIP') && pool.tabs.includes('Metal') && pool.code.includes('binsPool[blockIdx.x*NUM_BINS]'), JSON.stringify(codes.map((c) => c.tabs)));
await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('histogramPool'));
  [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal').click();
});
await pause(150);
check('its Metal tab uses a device-memory barrier and atomic loads', (await page.$$eval('figure.code pre', (ps) => ps.map((p) => p.textContent).join('\n'))).includes('threadgroup_barrier(mem_flags::mem_device)'));
const shared = codes.find((c) => c.code.includes('histogramShared'));
const sharedMissing = await page.$$eval('[data-listing="missing"]', (e) => e.length);
check('Version 2 shows lab 9.2\'s shared-memory kernel, or says where it will come from', (!!shared && shared.code.includes('__shared__ unsigned int bins_s[NUM_BINS]')) || sharedMissing === 1);

// ---------------------------------------------------------------- bug hunt
await page.waitForSelector(H + ' [data-variant]');
const variants = await page.$$eval(H + ' [data-variant]', (v) => v.map((x) => x.dataset.variant));
check('the hunt has seven broken kernels', variants.length === 7, JSON.stringify(variants));
check('kernel B removes the first barrier, kernel C the second', (await text(H + ' [data-variant="no-barrier-after-zeroing"]')).includes('removes line 20: __syncthreads();') && (await text(H + ' [data-variant="no-barrier-before-commit"]')).includes('removes line 26: __syncthreads();'));
const tick = async (v, label) => {
  await page.evaluate((H, v, label) => {
    const card = document.querySelector(`${H} [data-variant="${v}"]`);
    [...card.querySelectorAll('label.check')].find((l) => l.textContent.trim() === label).querySelector('input').click();
  }, H, v, label);
  await pause(80);
};
const reveal = async (v) => {
  await page.evaluate((H, v) => [...document.querySelector(`${H} [data-variant="${v}"]`).querySelectorAll('button')].find((b) => b.textContent === 'Reveal').click(), H, v);
  await pause(150);
};
// A: emu yes, race no. D: emu wrongly ticked.
await tick('no-zeroing', 'emu');
await tick('no-zeroing', 'a GPU run');
await reveal('no-zeroing');
check('kernel A, ticked emu: graded right', (await page.$eval(H + ' [data-variant="no-zeroing"] [data-graded]', (e) => e.dataset.graded)) === 'right');
check('kernel A: all 8 bins wrong in every order, race clean', (await text(H + ' [data-variant="no-zeroing"]')).match(/8 of 8 bins wrong/g)?.length === 3 && (await text(H + ' [data-variant="no-zeroing"]')).includes('clean'));
await tick('plain-shared-update', 'emu');
await reveal('plain-shared-update');
check('kernel D, ticked emu only: graded not quite, pointing at emu and the race backend', (await text(H + ' [data-variant="plain-shared-update"] [data-graded]')).includes('emu and the race backend'));
check('kernel D: the race names the update against itself in block 0, bin 1', (await text(H + ' [data-variant="plain-shared-update"]')).includes('bins_s[1] of block 0, threads 0 and 1'));
check('kernel D: the reveal says the real race backend misses it on this launch and catches it at lab 9.2\'s size',
  (await text(H + ' [data-variant="plain-shared-update"] [data-why]')).includes('misses it in most runs') && (await text(H + ' [data-variant="plain-shared-update"] [data-why]')).includes('the size lab 9.2 launches'));
check('kernel D: the checkboxes lock once revealed', await page.$eval(H + ' [data-variant="plain-shared-update"] input[type=checkbox]', (e) => e.disabled));
if (out) await page.screenshot({ path: out + '-hunt-two.png' });

check('reveal all', await clickIn(H, 'Reveal all'));
const matrix = await page.$$eval(H + ' tr[data-summary]', (trs) => Object.fromEntries(trs.map((t) => [t.dataset.summary, [...t.querySelectorAll('td[data-outcome]')].map((d) => d.dataset.outcome)])));
const MATRIX = {
  'no-zeroing': ['wrong-8', 'wrong-8', 'wrong-8', 'clean'],
  'no-barrier-after-zeroing': ['wrong-1', 'wrong-1', 'wrong-1', 'race'],
  'no-barrier-before-commit': ['wrong-1', 'wrong-1', 'wrong-2', 'race'],
  'plain-shared-update': ['correct', 'correct', 'correct', 'race'],
  'plain-commit': ['correct', 'correct', 'correct', 'race'],
  'block-scope-commit': ['correct', 'correct', 'correct', 'clean'],
  'commit-bound-blockDim': ['wrong-2', 'wrong-2', 'wrong-2', 'clean'],
};
check('the outcome table is the pinned matrix', JSON.stringify(matrix) === JSON.stringify(MATRIX), JSON.stringify(matrix));
const b = await text(H + ' [data-variant="no-barrier-after-zeroing"]');
check('kernel B: the race is the zeroing store (line 57) and the update (line 61), as ThreadSanitizer prints them', b.includes('the zeroing store (line 57)') && b.includes('the update of the private copy (line 61)'), b);
check('kernel B, in order: bin 1 holds 1, not 3', b.includes('bin 1 holds 1, not 3'), b);
check('kernel F: right everywhere here, undefined behaviour under CUDA', (await text(H + ' [data-variant="block-scope-commit"]')).includes('undefined behaviour'));
check('kernel G: bins 5 and 6 lost', (await text(H + ' [data-variant="commit-bound-blockDim"] [data-why]')).includes('Bins 5 and 6 had counts: 2 bins wrong'));
check('every card shows its kernel name once revealed', (await page.$$eval(H + ' [data-variant][data-revealed="yes"]', (e) => e.length)) === 7);
if (out) await page.screenshot({ path: out + '-hunt-all.png', fullPage: false });

// ---------------------------------------------------------------- fit table, before and after its question
const fitRows = () => page.$$eval('table[data-part="fit"] tbody tr[data-bins]', (trs) => trs.map((t) => ({ bins: t.dataset.bins, cells: [...t.querySelectorAll('td')].map((d) => d.textContent.trim()), fits: [...t.querySelectorAll('td[data-fits]')].map((d) => d.dataset.fits) })));
let fit = await fitRows();
check('fit table: GPT-2\'s size waits until its question is answered; the others show', fit.length === 4 && fit[1].cells[2] === 'answer the question above first' && !JSON.stringify(fit).includes('196.3') && fit[2].cells[2] === '593.5 KiB' && fit[3].cells[2] === '785.5 KiB', JSON.stringify(fit));
check('fit table: on every reported device 256 bins fit and no vocabulary does', fit[0].fits.every((x) => x === 'yes') && fit.slice(1).every((row) => row.fits.every((x) => x === 'no')), JSON.stringify(fit.map((f) => f.fits)));

// ---------------------------------------------------------------- questions
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  if (typeof value === 'number') await (await q.$('input[type=number]')).type(String(value));
  else await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), value.option);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
check('q-ex5a: 524,288 is accepted', (await answer('Exercise 5a', 524288)).includes('Correct.'));
check('q-ex5b: 65,536 is accepted', (await answer('Exercise 5b', 65536)).includes('Correct.'));
check('q-vocab-kib: 196.3 KiB is accepted', (await answer('GPT-2\'s vocabulary has 50,257 tokens', 196.3)).includes('Correct.'));
fit = await fitRows();
check('fit table: once answered, GPT-2\'s row shows 196.3 KiB', fit[1].cells[2] === '196.3 KiB', JSON.stringify(fit[1]));
check('q-scope-commit: the data-race answer is right', (await answer('committing thread per bin', { option: 1 })).includes('Correct.'));
check('q-why-block: a wrong option is explained', (await answer('rather than each thread', { option: 1 })).includes('just unnecessary'));
// Questions whose answer the page explains come before the explanation (predict, then read).
const order = await page.evaluate(() => {
  const kids = [...document.querySelector('.prose').children];
  const at = (f) => kids.findIndex(f);
  return {
    scopeQ: at((k) => k.classList.contains('q') && k.textContent.includes('committing thread per bin')),
    scopeP: at((k) => k.tagName === 'P' && k.textContent.includes('Get the second one wrong')),
    blockQ: at((k) => k.classList.contains('q') && k.textContent.includes('rather than each thread')),
    blockP: at((k) => k.tagName === 'P' && k.textContent.includes('A block is the natural owner')),
  };
});
check('q-scope-commit and q-why-block come before the paragraphs that answer them', order.scopeQ >= 0 && order.scopeQ < order.scopeP && order.blockQ >= 0 && order.blockQ < order.blockP, JSON.stringify(order));
const body94 = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the vocabulary sentence after the explorer names GPT-2\'s opt-in exception', body94.includes('only GPT-2\'s fits, and only behind CUDA\'s opt-in on some compute capabilities') && !body94.includes('fits none of the documented ones'));
check('the MPS choice is placed in HistogramKernel.mm', body94.includes('operations/HistogramKernel.mm') && !body94.includes('HistogramKernel.metal makes the same choice'));
const notes94 = await page.$$eval('.note', (n) => n.map((x) => x.className + ': ' + x.textContent));
check('the profiler-guide note no longer claims the book\'s order', notes94.some((t) => t.includes('which is this section\'s kernel')) && !notes94.some((t) => t.includes('in the book\'s order')));
check('the animation states what it simplifies', notes94.some((t) => t.includes('caveat') && t.includes('Counts, not time') && t.includes('real blocks are far larger')));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---------------------------------------------------------------- prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('coarsening factor of 16')));
await (await pq.$('input[type=number]')).type('20');
await (await pq.$('button.primary')).click();
await pause(200);
const reveal2 = await pq.evaluate((e) => e.textContent);
check('the prediction is committed and compared with lab 9.2\'s rows, or says how to get them', reveal2.includes('You predicted 20') && (reveal2.includes('No measurement on this machine yet') || reveal2.includes('shared-memory kernel (coarsening factor 16) ran at')), reveal2.slice(0, 200));

// ---------------------------------------------------------------- the scene is on the page
check('the narrated animation is on the page with five parts', (await page.$$('section.scene[aria-label="Privatise, then commit"] .seg button')).length === 5);

// ---------------------------------------------------------------- a phone-sized screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const wide = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, win: window.innerWidth }));
check('at 390 px wide the page itself does not scroll sideways', wide.doc <= wide.win, JSON.stringify(wide));
if (out) {
  await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
}

const expected404 = (u) => /\/content\/ch09\/narration\/(manifest\.json|privatise-\d\.mp3)$/.test(u) || /\/api\/lab-file\?path=ch09%2Flab2_privatised/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest or lab 9.2 file is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
