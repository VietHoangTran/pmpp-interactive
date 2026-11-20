// Browser check for section 13.3 (group 1's parts): the co-rank explorer (search, predict
// mode, swapped tie tests, presets, your own arrays, forced splits), the worked-example
// table, the warp explorer, the listings and the page's own questions. The swapped tie rule
// is shown by the explorer and by part 6 of the scene; there is no separate ties bench.
// Every number asserted is recomputed here from the models (src/sim/corank.ts and
// src/sim/mergeaccess.ts), not typed in. The staging part (group 2) and the narrated
// scene have their own checks.
// Usage: node e2e/ch13-s133.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto } from './browser.mjs';

// mergeaccess.ts imports its siblings without a file extension (as the app's bundler
// allows); let Node resolve those to the .ts files, then load the models.
registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const { coRankTrace, splitAt } = await import('../src/sim/corank.ts');
const { basicLayout, makeData, searchCost, threadsFor, warpStep } = await import('../src/sim/mergeaccess.ts');

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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const X = bench('Co-rank explorer');
const Wb = bench('Split the work, watch a warp');

const RA = [1, 7, 8, 9, 10];
const RB = [7, 10, 10, 12];

const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setRange = async (scope, aria, v) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setText = async (scope, aria, v) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(50);
};
const setSelect = async (scope, labelStart, v) => {
  await page.evaluate((scope, labelStart, v) => {
    const s = [...document.querySelectorAll(scope + ' label.field')].find((l) => l.textContent.trim().startsWith(labelStart)).querySelector('select');
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(v));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, labelStart, v);
  await pause(100);
};
const setBox = async (scope, word, on) => {
  await page.evaluate((scope, word, on) => {
    const i = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (i.checked !== on) i.click();
  }, scope, word, on);
  await pause(80);
};
const ds = (scope, sel) => page.$eval(`${scope} [data-explorer${sel ? '="' + sel + '"' : ''}]`, (e) => ({ ...e.dataset }));
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());

await goto(page, 'ch13/13.3');
await page.waitForSelector(X);

// ---- the listings
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the co_rank listing and the kernel listing with three tabs are shown',
  ['co_rank', 'merge_basic_kernel', 'merge_sequential'].every((t) => tabs.includes(t)), tabs.join('|'));
// Only this group's listings (co_rank, the basic kernel); the staging part shows lab files with their own labels.
const status = await page.$$eval('.code', (cs) => cs
  .filter((c) => [...c.querySelectorAll('.tabs button')].some((b) => ['co_rank', 'merge_basic_kernel'].includes(b.textContent)))
  .map((c) => c.querySelector('.tabs .status')?.textContent ?? ''));
check('listings carry the verification status in the fixed vocabulary',
  status.length === 3 && status.every((s) => s === 'emulator-checked and front-end-checked; unverified on CUDA hardware'), status.join('|'));
check('the printed line 03 is quoted', (await text('.page')).includes('int elementsPerThread = ceil((m+n)/(blockDim.x*gridDim.x));'));

// ---- co-rank explorer: the book's trace for k = 3 (Figs. 13.6 to 13.8)
const t3 = coRankTrace(3, RA, RB);
let d = await ds(X, 'corank');
check('starts at the running example, k = 3, iteration 0 not yet tested, (i, j, i_low, j_low) = (3, 0, 0, 0)',
  d.k === '3' && d.shown === '0' && [d.i, d.j, d.ilow, d.jlow].join() === [t3.init.i, t3.init.j, t3.init.iLow, t3.init.jLow].join(), JSON.stringify(d));
check('the starting values are explained', (await text(X + ' .narrate')).includes('i = min(k, m) = min(3, 5) = 3'));
const hl = () => page.$$eval(X + ' .code .ln.hl', (ls) => ls.map((l) => Number(l.dataset.n)));
check('the initial lines 2 to 5 are highlighted', JSON.stringify(await hl()) === '[2,3,4,5]', JSON.stringify(await hl()));
await clickText(X, 'Step');
d = await ds(X, 'corank');
check('step 1 runs the tests: i is too high, A[2] = 8 > B[0] = 7', d.verdict === t3.steps[0].test && (await text(X + ' tr[data-test="1"]')).includes('A[2] = 8 > B[0] = 7'), JSON.stringify(d));
check('the second test is skipped (else-if)', (await text(X + ' tr[data-test="2"]')).includes('not evaluated'));
check('the move is spelled out: delta 2, j = 2, i = 1', (await text(X + ' .narrate')).includes('= 2; j_low = j = 0; j = 0 + 2 = 2; i = 3 − 2 = 1'), await text(X + ' .narrate'));
check('code line 9 is highlighted', (await hl())[0] === 9);
check('the window of candidates shrinks to 0..2', (await page.$eval(X + ' [data-window]', (e) => e.dataset.window)) === '0-2');
await clickText(X, 'Step');
await clickText(X, 'Step');
d = await ds(X, 'corank');
check('iteration 1 at (1, 2, 0, 0): j is too high, B[1] = 10 ≥ A[1] = 7',
  [d.i, d.j, d.ilow, d.jlow].join() === '1,2,0,0' && d.verdict === 'j-too-high' && (await text(X + ' tr[data-test="2"]')).includes('B[1] = 10 ≥ A[1] = 7'), JSON.stringify(d));
await clickText(X, 'Run to end');
d = await ds(X, 'corank');
check('ends at (2, 1, 1, 0): done, co_rank(3) = 2, after 2 moves, the bound',
  [d.i, d.j, d.ilow, d.jlow].join() === '2,1,1,0' && d.verdict === 'done' && d.result === String(t3.result) && d.moves === '2' && d.bound === String(t3.bound), JSON.stringify(d));
check('the result matches the stable merge\'s first 3 outputs', (await page.$eval(X + ' [data-prefix-ok]', (e) => e.dataset.prefixOk)) === '1');
check('i + j = k badge', (await text(X + ' [data-sum]')) === 'i + j = 2 + 1 = 3 = k');
if (out) await page.screenshot({ path: out + '-explorer-done.png' });

// ---- predict mode
await clickText(X, 'Restart');
await setBox(X, 'predict mode', true);
check('predict mode offers the three verdicts instead of Step', await page.$eval(X, (e) => [...e.querySelectorAll('button')].some((b) => b.textContent.trim() === 'i too high')));
await clickText(X, 'done');
let n = await text(X + ' .narrate');
check('a wrong prediction is corrected inline with the reason', n.startsWith('Not quite: you said neither: done.') && n.includes('A[2] = 8 > B[0] = 7'), n);
await clickText(X, 'Step');
await clickText(X, 'j too high');
n = await text(X + ' .narrate');
check('a right prediction is confirmed', n.startsWith('Right.'), n);
await setBox(X, 'predict mode', false);

// ---- swapped tie tests at k = 6
await setRange(X, 'k', 6);
await setBox(X, 'swap the tie tests', true);
await clickText(X, 'Run to end');
d = await ds(X, 'corank');
const s6 = coRankTrace(6, RA, RB, 'swapped');
check('with the tests swapped, co_rank(6) is 4 (ties sent to B), not 5', d.result === String(s6.result) && s6.result === 4 && d.rule === 'swapped', JSON.stringify(d));
check('and the explorer says it disagrees with the stable merge', (await page.$eval(X + ' [data-prefix-ok]', (e) => e.dataset.prefixOk)) === '0');
const swappedLine9 = await page.$$eval(X + ' .code .ln', (ls) => ls[8].textContent);
check('the listing shows the swapped comparisons', swappedLine9.includes('A[i - 1] >= B[j]') && (await text(X + ' .code .tabs')).includes('tie tests swapped'), swappedLine9);
await setBox(X, 'swap the tie tests', false);

// ---- presets
await clickText(X, 'All keys equal');
for (const k of [0, 3, 7, 9]) {
  await setRange(X, 'k', k);
  await clickText(X, 'Run to end');
  d = await ds(X, 'corank');
  check(`all keys equal, k = ${k}: co_rank is min(k, m) = ${Math.min(k, 5)}`, d.result === String(Math.min(k, 5)), JSON.stringify(d));
}
{
  const kLabel = await page.$eval(`${X} input[aria-label="k"]`, (e) => e.closest('label').textContent.replace(/\s+/g, ' ').trim());
  check('at k = m + n the slider says it splits C[0..9) between A and B, not that it finds C[9]',
    kLabel === 'k = 9 (split C[0..9) between A and B)' && !kLabel.includes('comes from'), kLabel);
}
await clickText(X, 'All of B below A');
await setRange(X, 'k', 4);
await clickText(X, 'Run to end');
d = await ds(X, 'corank');
const bb = coRankTrace(4, [6, 7, 8, 9, 10], [1, 2, 3, 4]);
check('all of B below A, k = 4: the moves reach the bound (3)', d.moves === d.bound && d.bound === String(bb.bound) && bb.bound === 3 && d.result === '0', JSON.stringify(d));
await clickText(X, 'A empty');
await setRange(X, 'k', 3);
await clickText(X, 'Run to end');
d = await ds(X, 'corank');
check('A empty: co_rank is 0 with no moves and no reads', d.result === '0' && d.moves === '0', JSON.stringify(d));
const before = (await ds(X, 'corank')).arrays;
await clickText(X, 'Random');
const r1 = (await ds(X, 'corank')).arrays;
await clickText(X, 'Random');
const r2 = (await ds(X, 'corank')).arrays;
check('Random gives a new pair of sorted arrays each time', r1 !== before && r2 !== r1 && r1.split('|').every((s) => s === '' || s.split(',').map(Number).every((v, q, a) => q === 0 || a[q - 1] <= v)), `${r1} / ${r2}`);

// ---- your own arrays
await setText(X, 'your A', '3, 1');
await clickText(X, 'Use these');
check('unsorted input is rejected with a reason', (await text(X + ' [role="alert"]')).includes('not sorted'));
await setText(X, 'your A', '2, 4, 4, 6');
await setText(X, 'your B', '4, 5');
await clickText(X, 'Use these');
d = await ds(X, 'corank');
check('sorted input is accepted', d.arrays === '2,4,4,6|4,5', d.arrays);
await setRange(X, 'k', 3);
await clickText(X, 'Run to end');
d = await ds(X, 'corank');
check('your arrays, k = 3: the explorer agrees with the model', d.result === String(coRankTrace(3, [2, 4, 4, 6], [4, 5]).result), JSON.stringify(d));

// ---- forced splits (Fig. 13.4)
await clickText(X, 'The running example');
await setRange(X, 'k', 4);
await clickText(X, 'Try any split');
for (const i of [4, 2, 3]) {
  await setRange(X, 'force i', i);
  d = await ds(X, 'corank');
  const sp = splitAt(4, i, RA, RB);
  check(`k = 4, i = ${i}: ${sp.valid ? 'valid' : sp.failing + ', C unsorted'}`,
    d.forceI === String(i) && d.valid === (sp.valid ? '1' : '0') && d.failing === (sp.failing ?? '') && d.sorted === (sp.sortedKeys ? '1' : '0'), JSON.stringify(d));
  if (i === 4) {
    const nn = await text(X + ' .narrate');
    check('i = 4 names the failing test and the direction', nn.includes('A[3] = 9 > B[0] = 7') && nn.includes('Move i down'), nn);
    const bad = await page.$$eval(X + ' [data-cell^="C"][data-mark="fail"]', (c) => c.length);
    check('the out-of-order elements at the cut are marked', bad >= 2, String(bad));
    if (out) await page.screenshot({ path: out + '-split-wrong.png' });
  }
}
await setRange(X, 'k', 6);
await setRange(X, 'force i', 4);
d = await ds(X, 'corank');
check('Fig. 13.3(b)\'s labels (k = 6, i = 4): keys sorted but not stable', d.sorted === '1' && d.stable === '0' && d.failing === 'j-too-high', JSON.stringify(d));
check('and the narration says the tie went the wrong way', (await text(X + ' .narrate')).includes('tie went the wrong way'));

// ---- ties: no separate bench; the text points at part 6 of the scene and the explorer's box
check('there is no "Ties at the cuts" bench (the swap is shown by the explorer and part 6 of the scene)', !(await page.$('section.bench[aria-label="Ties at the cuts"]')));
check('the tie paragraph points at part 6 of the animation and at the explorer\'s swap box',
  (await text('.prose')).includes('Part 6 of the animation above shows it at the running example\'s cuts') && (await text('.prose')).includes('"swap the tie tests" box'));

// ---- the worked example of p. 314
const row1 = await page.$$eval('table[data-worked] tbody tr', (rs) => rs.map((r) => r.textContent));
check('worked example: thread 1 calls (&A[2], 3, &B[1], 0, &C[3])', row1[1]?.includes('(&A[2], 3, &B[1], 0, &C[3])'), row1.join(' | '));

// ---- warp explorer
const model = (data, total, per, formula, warp, s, ww = 32, seg = 32) => {
  const dd = makeData(data, total, 1);
  const L = basicLayout(dd.A, dd.B, threadsFor(total, per), formula);
  return { L, ws: warpStep(L, warp, ww, s, 4, seg), cost: searchCost(L) };
};
let w = await ds(Wb, 'warp');
let mm = model('distinct', 256, 8, 'corrected', 0, 0);
check('default: 256 outputs, 8 per thread, 32 threads; the store touches 32 segments, 12.5 % used',
  w.threads === '32' && w.perThread === '8' && w.writes === String(mm.ws.writes.transactions) && w.writes === '32' && Number(w.writesEff) === 0.125, JSON.stringify(w));
check('the lanes of the warp take more than one path', Number(w.paths.split(',').length) > 1 && w.paths === mm.ws.paths.join(','), w.paths);
{
  const chip = await page.$eval(`${Wb} [data-divergence]`, (e) => ({ text: e.textContent.trim(), cls: e.className, loops: e.dataset.loops }));
  const loops = new Set(mm.ws.paths.map((p) => (p === 'A' || p === 'B' ? 'main' : p))).size;
  check(`the paths chip says "${mm.ws.paths.length} paths at this step", claims nothing about serialising, and is red only when lanes are in different loops (${loops} here)`,
    chip.text === `${mm.ws.paths.length} paths at this step` && chip.loops === String(loops) && chip.cls.includes('fail') === loops > 1 && !chip.text.includes('one after another'), JSON.stringify(chip));
  const order = await page.evaluate((Wb) => {
    const q = [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many segments does one 32-lane store touch'));
    const b = document.querySelector(Wb);
    return q && b ? q.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING : 0;
  }, Wb);
  check('q-coalesce comes before the warp explorer, so the prediction is made first', order !== 0);
  const prose = await text('.prose');
  check('the divergence paragraph is hedged (a select is possible) and labelled ours',
    prose.includes('Lanes in different loops take different paths for certain') && prose.includes('may turn into a select') &&
    (await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes('the book lists two memory costs for this kernel (p. 314)')))));
}
check('search reads match the model', w.searchReads === String(mm.cost.searchReads), `${w.searchReads} vs ${mm.cost.searchReads}`);
const per8 = Number(w.searchPerOutput);
if (out) await page.screenshot({ path: out + '-warp-8.png' });
await setRange(Wb, 'step', 5);
w = await ds(Wb, 'warp');
mm = model('distinct', 256, 8, 'corrected', 0, 5);
check('step 5 still touches 32 segments, and the reads match the model', w.step === '5' && w.writes === '32' && w.readsA === String(mm.ws.readsA.transactions) && w.readsB === String(mm.ws.readsB.transactions), JSON.stringify(w));
await setRange(Wb, 'outputs per thread', 1);
w = await ds(Wb, 'warp');
check('one output per thread: the store coalesces into 4 segments', w.writes === '4' && Number(w.writesEff) === 1, JSON.stringify(w));
const per1 = Number(w.searchPerOutput);
await setRange(Wb, 'outputs per thread', 16);
const per16 = Number((await ds(Wb, 'warp')).searchPerOutput);
check('search reads per output fall as outputs per thread grow (1, 8, 16)', per1 > per8 && per8 > per16 && per1 / per16 > 8, `${per1} ${per8} ${per16}`);
await setRange(Wb, 'outputs per thread', 1);
await setSelect(Wb, 'Warp width', 64);
await setSelect(Wb, 'Segment model', 'amd128');
w = await ds(Wb, 'warp');
check('64 lanes and 128-byte lines: one output per thread stores into 2 lines', w.writes === '2', JSON.stringify(w));
await setSelect(Wb, 'Segment model', 'metal');
check('Metal: the segment size is offered as a choice, not a fact', (await text(Wb)).includes('Your choice (not a fact)'));
await setSelect(Wb, 'Segment model', 'nvidia');
await setSelect(Wb, 'Warp width', 32);
await setRange(Wb, 'total', 1000);
await setRange(Wb, 'outputs per thread', 3);
await clickText(Wb, 'as printed');
w = await ds(Wb, 'warp');
mm = model('distinct', 1000, 3, 'printed', 0, 0);
check('as printed, 1,000 outputs on 334 threads: 2 per thread, 332 outputs never written',
  w.threads === '334' && w.perThread === '2' && w.unwritten === String(mm.L.unwritten.length) && mm.L.unwritten.length === 332, JSON.stringify(w));
check('the unwritten tail is drawn and named', !!(await page.$(Wb + ' [data-unwritten-band]')) && (await text(Wb + ' [data-unwritten-text]')).includes('332 outputs'));
if (out) await page.screenshot({ path: out + '-warp-printed.png' });
await clickText(Wb, 'corrected');
w = await ds(Wb, 'warp');
check('corrected: 3 per thread, nothing left unwritten', w.perThread === '3' && w.unwritten === '0', JSON.stringify(w));
await setSelect(Wb, 'Data', 'a-before-b');
await setRange(Wb, 'outputs per thread', 8);
w = await ds(Wb, 'warp');
check('all of A before B: warp 0 only copies the rest of A (one path)', w.paths === 'tailA', w.paths);

// ---- questions
const answerNum = async (needle, v) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-iterations accepts 21', await answerNum('At most how many times can co_rank', 21));
check('q-ceil-printed accepts 232', await answerNum('How many outputs are never', 232));
check('q-coalesce accepts 32', await answerNum('how many segments does one 32-lane store touch', 32));
const choose = async (needle, optionStart) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await q.evaluate((e, optionStart) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(optionStart)).querySelector('input').click();
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-wrong-split: the first test fires', await choose('someone sets', '7 lands after 9'));
check('q-tie-test: unstable at the cuts', await choose('A colleague writes the tests', 'At the cuts, ties now go to B'));
check('q-one-per-thread: two whole searches per output', await choose('Why not always use one', 'Each output would pay for two whole binary searches'));
check('q-one-per-thread: the explanation counts log2 N passes of up to four reads', (await text('.prose')).includes('each about log2 N passes of up to four scattered reads'));
// Where the right option sits: the chapter spreads them over all four positions.
const correctAt = await page.evaluate((needles) => needles.map((nd) => {
  const q = [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(nd));
  return [...q.querySelectorAll('.opts label')].findIndex((l) => l.classList.contains('right'));
}), ['someone sets', 'A colleague writes the tests', 'Why not always use one']);
check('the three answered choice questions have their right option in positions 1, 4 and 3', JSON.stringify(correctAt) === '[0,3,2]', JSON.stringify(correctAt));

// ---- themes and a narrow screen
if (out) {
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
  await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-explorer-light.png' });
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
  await page.screenshot({ path: out + '-explorer-dark.png' });
  await page.$eval(Wb, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-warp-dark.png' });
}
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
{
  const fit = await page.$eval(`${Wb} [data-access="c"]`, (tr) => {
    const td = tr.querySelector('[data-eff]');
    const chip = td.querySelector('.chip');
    const wrap = tr.closest('table').parentElement;
    const c = chip.getBoundingClientRect();
    return { td: td.getBoundingClientRect().width, chip: c.width, right: c.right, box: wrap.getBoundingClientRect().right, scroll: wrap.scrollLeft, scrolls: getComputedStyle(wrap).overflowX, text: chip.textContent };
  });
  check('at 400 px the store row\'s efficiency chip is whole and in view without scrolling (the byte columns scroll inside the box)',
    fit.td >= fit.chip && fit.chip > 40 && fit.scroll === 0 && fit.right <= fit.box + 1 && fit.scrolls === 'auto' && fit.text.includes('% used'), JSON.stringify(fit));
}
if (out) {
  await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
  await page.$eval(Wb, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-warp.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
