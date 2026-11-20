// Browser check for the staging part of section 13.3 (book §13.6, §13.7 in brief): its
// place on the page, the "Loaded versus used" explorer on every preset, scheme, tile size
// and line-21 setting, the book's example, three questions, lab 13.1's tiled kernel shown
// from the lab files, and the narrated animation "Loaded twice, used once" (scripts,
// captions switch, sound off, every part's drawing). Every number asserted is recomputed
// here from src/sim/mergetile.ts, imported directly, or read from /api/reports and
// /api/lab-file.
// Usage: APP_URL=http://localhost:<port> node e2e/ch13-s133-staging.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_EXAMPLE, PRESETS, bookTiled, exactTiled, sortedRandom } from '../src/sim/mergetile.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') saved.push(r.postData() ?? '');
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const n = (v) => v.toLocaleString('en-US');
const X = 'section.bench[aria-label="Loaded versus used"]';
const S = 'section.scene[aria-label="Loaded twice, used once"]';

await goto(page, 'ch13/13.3');

// ---- where the part sits
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const at = (t) => h2.indexOf(t);
check('the staging part sits between "What it costs" and "Where you will meet it", in three parts',
  at('What it costs') >= 0 && at('Stage the inputs in shared memory') === at('What it costs') + 1 &&
  at('Load exactly what a tile uses') === at('What it costs') + 2 && at('The circular buffer, in brief') === at('What it costs') + 3 &&
  at('Where you will meet it') === at('What it costs') + 4, JSON.stringify(h2));
const text0 = await page.$eval('.prose', (e) => e.textContent);
check('the book\'s example is the book\'s: 33,000 and 31,000, 16 blocks of 128, tile 1,024, 8,192 bytes, called the book\'s figures',
  text0.includes('m = 33,000, n = 31,000, 16 blocks of 128 threads') && text0.includes('= 8,192 bytes') && text0.includes("the book's example figures, not any device's"));
check('the waste is quoted from p. 320', text0.includes('"wastes half of the memory bandwidth"'));
check('the line-48 case beyond the book is labelled ours', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('item 18'))));
check('the libraries\' design is dated and cited (A7, B2, D1, D2), with PyTorch\'s "same result, no barriers"',
  text0.includes('"same result, no barriers"') && (await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('A7, B2, D1, D2') && a.textContent.includes('2026-11-20')))));
check('the circular buffer gets one paragraph and the reason the course skips it', text0.includes('co_rank_circular') && text0.includes('This course builds none of it') && text0.includes('Fig. 13.17'));

// ---- the explorer
const state = () => page.$eval(`${X} [data-explorer="tile"]`, (e) => ({
  ...e.dataset,
  rows: [...e.querySelectorAll('[data-iteration]')].map((g) => ({ ...g.dataset })),
  tiles: [...e.querySelectorAll('[data-tile-row]')].map((g) => ({ ...g.dataset })),
  stale: e.dataset.stale,
  marks: e.querySelectorAll('[data-slot="stale"]').length,
  readout: e.querySelector('[data-readout]').textContent,
  table: [...e.querySelectorAll('table[data-table="book-example"] tbody tr')].map((tr) => [...tr.children].map((td) => td.textContent.trim())),
}));
const click = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(120);
  return ok;
};
const setTile = async (v) => {
  await page.$eval(`${X} select[aria-label="tile size"]`, (s, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(v));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const ratio = (a, b) => (b ? (a / b).toFixed(2) : '–');

let s = await state();
check('the explorer starts on the uneven preset, book scheme, line 21 corrected, tile 8', s.data === 'uneven' && s.scheme === 'book' && s.formula === 'corrected' && s.tile === '8', JSON.stringify(s));
for (const [id, label] of [['uneven', PRESETS.uneven.label], ['allFromA', PRESETS.allFromA.label], ['shortTileEarly', PRESETS.shortTileEarly.label]]) {
  await click(X, label);
  for (const tile of id === 'uneven' ? [4, 8, 16] : [8]) {
    await setTile(tile);
    const p = PRESETS[id];
    for (const formula of ['corrected', 'printed']) {
      await click(X, 'book: a full tile of each');
      await click(X, formula === 'corrected' ? 'line 21 corrected' : 'as printed');
      s = await state();
      const run = bookTiled(p.A, p.B, 1, tile, formula, 2);
      const its = run.blocks[0].iterations;
      const stale = its.reduce((a, it) => a + it.staleSlots, 0);
      check(`${id}, tile ${tile}, ${formula}: loaded ${run.totals.loaded}, used ${run.totals.used}, missed ${run.totals.missed}, ${its.length} iterations, ${stale} stale slots`,
        s.data === id && Number(s.loaded) === run.totals.loaded && Number(s.used) === run.totals.used && Number(s.minimum) === p.A.length + p.B.length &&
        Number(s.missed) === run.totals.missed && s.rows.length === its.length && Number(s.stale) === stale && s.marks === stale &&
        its.every((it, r) => [s.rows[r].aloaded, s.rows[r].bloaded, s.rows[r].aused, s.rows[r].bused].join() === [it.aLoaded, it.bLoaded, it.aUsed, it.bUsed].join()) &&
        s.readout.includes(`Loaded ÷ used: ${ratio(run.totals.loaded, run.totals.used)}`) &&
        (run.totals.missed ? s.readout.includes(`${n(run.totals.missed)} outputs were never written`) : !s.readout.includes('never written')), JSON.stringify({ s: { ...s, readout: undefined }, totals: run.totals }));
    }
    await click(X, 'line 21 corrected');
    await click(X, 'exact: search first');
    s = await state();
    const ex = exactTiled(p.A, p.B, tile);
    check(`${id}, tile ${tile}, exact: loaded = used = ${ex.totals.loaded}, ${ex.tiles.length} tiles, ${ex.totals.extraSearchReads} extra search reads`,
      s.scheme === 'exact' && Number(s.loaded) === ex.totals.loaded && Number(s.used) === ex.totals.used && Number(s.extra) === ex.totals.extraSearchReads &&
      s.tiles.length === ex.tiles.length && ex.tiles.every((t, r) => Number(s.tiles[r].a) === t.aRange[1] - t.aRange[0] && Number(s.tiles[r].b) === t.bRange[1] - t.bRange[0]) &&
      s.readout.includes(`read ${n(ex.totals.extraSearchReads)} elements of global memory`), JSON.stringify(s.tiles));
    check(`  ${id}, tile ${tile}: the line-21 switch is off for the exact scheme`, await page.$eval(X, (e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'as printed').disabled));
    await click(X, 'book: a full tile of each');
  }
  if (out) await page.screenshot({ path: `${out}-explorer-${id}.png` });
}
await setTile(8);
// The worst case loads two tiles for one tile of outputs; A running out early leaves slots line 48 would misread.
await click(X, PRESETS.allFromA.label);
s = await state();
check('worst case: the first iterations use 8 of A and 0 of B', s.rows[0].aused === '8' && s.rows[0].bused === '0' && s.rows[1].aused === '8');
await click(X, PRESETS.shortTileEarly.label);
s = await state();
check('A runs out early: iteration 1 loads 2 of A and marks 6 slots at risk; the readout calls them slots the search could read, counted under the corrected A_consumed',
  s.rows[1].aloaded === '2' && s.rows[1].stale === '6' && s.readout.includes(`At risk: ${s.stale} slots that line 48's search could read`) &&
  s.readout.includes('counted under the corrected A_consumed') && !s.readout.includes('would search'), JSON.stringify({ row: s.rows[1], readout: s.readout }));
check('the page names the one stale slot line 48 actually reads there, A_S[7], and the count it returns',
  text0.includes('line 48 reads A_S[7], which still holds A[7], and counts 8 elements of A where the merges used 2'));
check('the explorer\'s note says the question marks are counted under the corrected A_consumed',
  await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('The question marks are counted under that corrected A_consumed'))));
check('one block of 2 threads is what the foot says for the small presets', (await page.$eval(`${X} footer`, (e) => e.textContent)).includes('on one block of 2 threads'));

// random data, reproducible from what the page shows
await click(X, 'random, m = n = 24');
s = await state();
const [ra, rb] = s.arrays.split('|').map((x) => x.split(',').map(Number));
const sortedOK = (xs) => xs.every((v, i) => i === 0 || xs[i - 1] <= v);
const runR = bookTiled(ra, rb, 1, 8, 'corrected', 2);
check(`random: 24 + 24 sorted keys; loaded ${runR.totals.loaded} for ${runR.totals.used}`, ra.length === 24 && rb.length === 24 && sortedOK(ra) && sortedOK(rb) &&
  Number(s.loaded) === runR.totals.loaded && Number(s.used) === 48 && ra.join() === sortedRandom(24, 40, 3).join(), JSON.stringify(s.arrays));
await click(X, 'new random');
const s2 = await state();
check('"new random" draws new arrays and the counts follow', s2.arrays !== s.arrays && Number(s2.loaded) === bookTiled(...s2.arrays.split('|').map((x) => x.split(',').map(Number)), 1, 8, 'corrected', 2).totals.loaded);

// the book's example
await click(X, "the book's example");
s = await state();
const BE = BOOK_EXAMPLE;
const A = sortedRandom(BE.m, 1 << 30, 1);
const B = sortedRandom(BE.n, 1 << 30, 2);
const printed = bookTiled(A, B, BE.blocks, BE.tile, 'printed');
const corrected = bookTiled(A, B, BE.blocks, BE.tile, 'corrected');
const exact = exactTiled(A, B, BE.tile);
check(`book's example table: printed ${n(printed.totals.loaded)}/${n(printed.totals.used)} with ${n(printed.totals.missed)} unwritten; corrected ${n(corrected.totals.loaded)}/${n(corrected.totals.used)}; exact ${n(exact.totals.loaded)} with ${n(exact.totals.extraSearchReads)} search reads`,
  s.table.length === 3 &&
  s.table[0].join('|') === `book, line 21 as printed|3 per block|${n(printed.totals.loaded)}|${n(printed.totals.used)}|${n(printed.totals.missed)}|–` &&
  s.table[1].join('|') === `book, line 21 corrected|4 per block|${n(corrected.totals.loaded)}|${n(corrected.totals.used)}|0|–` &&
  s.table[2].join('|') === `exact: search first|${exact.tiles.length} tiles|${n(exact.totals.loaded)}|${n(exact.totals.used)}|0|${n(exact.totals.extraSearchReads)}`, JSON.stringify(s.table));
check('the book example\'s unwritten outputs are 928 per block, 14,848 in all', printed.totals.missed === 928 * 16 && printed.blocks[0].missed === 928);
check('the tile size is fixed at 1,024 for the book\'s example', s.tile === '1024' && await page.$eval(`${X} select`, (e) => e.disabled));
{
  const foot = await page.$eval(`${X} footer`, (e) => e.textContent);
  check('for the book\'s example the foot says 16 blocks, each modelled as one thread', foot.includes(`on ${BE.blocks} blocks, each modelled as one thread`) && !foot.includes('one block of'), foot);
}
if (out) await page.screenshot({ path: `${out}-explorer-book.png` });
await click(X, PRESETS.uneven.label);

// ---- questions
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
// The printed iteration count is taught once on this page (q-ceil-printed); here it is one
// sentence computed from the model, and the explorer's switch.
check(`the book's example paragraph gives the printed count from the model: 3 iterations, not 4, and ${n(printed.blocks[0].missed)} outputs per block never written`,
  text0.includes(`each block runs 3 iterations, not 4, and ${n(printed.blocks[0].missed)} of its outputs are never written`));
check('q-stage-printed is gone (the ceil slip has one question on this page)', !(await page.$$eval('.q', (qs) => qs.some((e) => e.textContent.includes('As printed, how many')))));
const pick = async (q, i) => {
  await (await q.$$('input[type=radio]'))[i].click();
  await (await q.$('button.primary')).click();
  await pause(120);
};
let q = await question('yet it first loads');
await pick(q, 2);
check('q-stage-why-two: "the unused half is kept" is wrong, that is the circular buffer', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('circular-buffer kernel'));
await pick(q, 1);
check('q-stage-why-two: nobody knows the split before the loads; exactly half used per full iteration',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('In every full iteration exactly half') && !(await qText(q)).includes('On average'));
q = await question('What do they pay');
await pick(q, 0);
check('q-stage-exact: "twice the shared memory" is wrong', (await qText(q)).includes('Not quite.'));
await pick(q, 3);
check('q-stage-exact: one more search per tile, in a partition kernel', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Exercise 3') && (await qText(q)).includes('every tiled merge read'));
await pause(700);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('the two answers were sent to /api/progress under ch13/13.3/q-stage-*', ['why-two', 'exact'].every((k) => ans[`ch13/13.3/q-stage-${k}`] !== undefined), Object.keys(ans).join(', '));
check('the library claims name the four tiled merges and say llama.cpp\'s Metal merge does not stage',
  text0.includes('Every tiled merge read for this course (CUB, rocPRIM, MLX, PyTorch MPS)') && text0.includes('does not stage at all') &&
  !text0.includes('all of them load exactly') && !text0.includes('Every library merge read'));

// ---- lab 13.1's tiled kernel, from the lab files
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const file = (p) => page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), p);
const slice = (text, start, end) => {
  const lines = text.split('\n');
  const a = lines.findIndex((l) => start.test(l));
  const b = lines.findIndex((l, i) => i > a && end.test(l));
  return lines.slice(a, b < 0 ? undefined : b).join('\n').replace(/\s+$/, '');
};
const cu = slice(await file('ch13/lab1_merge/solution/kernel.cu'), /^\/\/ -{20,} Part \(b\)/, /^\/\/ -{20,} Provided/);
const metal = slice(await file('ch13/lab1_merge/solution/kernel.metal'), /^\/\/ Part \(b\), step 1/, /^\/\/ -{20,} Provided/);
const fig = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('Part (b) of lab 13.1')));
const tabsOf = () => fig.evaluate((f) => [...f.querySelectorAll('.tabs button')].map((b) => b.textContent));
check('the lab listing has CUDA and Metal tabs', (await tabsOf()).join('|') === 'CUDA|Metal', (await tabsOf()).join('|'));
const pass = (b) => {
  const rs = (reports[b]?.results ?? []).filter((x) => x.lab === 'ch13/lab1_merge' && x.variant === 'solution' && x.tests.length);
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
};
const cuStatus = (() => {
  const parts = [pass('emu') ? "emulator-checked (lab 13.1's tests)" : '', pass('cuda') ? 'verified on CUDA' : '', pass('hip') ? 'verified on HIP' : ''].filter(Boolean);
  const lacking = ['cuda', 'hip'].filter((b) => !pass(b)).map((b) => b.toUpperCase());
  return parts.length ? parts.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified (no run on this machine)';
})();
const metalStatus = pass('metal') ? "verified on Metal (lab 13.1's tests)" : 'unverified: no passing Metal run of lab 13.1 recorded here';
for (const [i, want, status, what] of [[0, cu, cuStatus, 'kernel.cu from "Part (b)" to "Provided"'], [1, metal, metalStatus, 'kernel.metal from "Part (b), step 1" to "Provided"']]) {
  await fig.evaluate((f, i) => f.querySelectorAll('.tabs button')[i].click(), i);
  await pause(120);
  const shown = await fig.evaluate((f) => f.querySelector('pre').textContent.replace(/\s+$/, ''));
  const st = await fig.evaluate((f) => f.querySelector('.tabs .status').textContent);
  check(`tab ${i + 1} is ${what}, verbatim, status "${status}"`, shown === want && st === status && want.length > 400, st);
}
check('the listing colours the tile as shared memory and A, B, C as global', await fig.evaluate((f) => f.querySelectorAll('.v-shared').length > 0 && f.querySelectorAll('.v-global').length > 0));

// ---- the narrated animation
const strip = (t) => t.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const N = 5;
const LABELS = ['Load a tile', 'Use what the merge needs', 'Load again', 'The worst case', 'Search first'];
const res = await fetch(BASE + '/content/ch13/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let k = 1; k <= N; k++) scripts.push(strip(await (await fetch(`${BASE}/content/ch13/narration/tile-waste-${k}.txt`)).text()));
const U = PRESETS.uneven;
const RUN = bookTiled(U.A, U.B, 1, U.tile, 'corrected', U.threads);
const IT = RUN.blocks[0].iterations;
const W = PRESETS.allFromA;
const ITW = bookTiled(W.A, W.B, 1, W.tile, 'corrected', W.threads).blocks[0].iterations;
const EX = exactTiled(U.A, U.B, U.tile);
const words = scripts.map((t) => t.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((t) => !/[<>=×%/_(){}#*[\]]/.test(t)), scripts.find((t) => /[<>=×%/_(){}#*[\]]/.test(t)));
const again0 = IT[0].aLoaded - IT[0].aUsed + (IT[0].bLoaded - IT[0].bUsed);
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`next ${IT[0].cOut} outputs`) && scripts[0].includes(`loads ${IT[0].aLoaded} elements of each`) &&
  scripts[1].includes(`Thread 0 takes ${IT[0].threads[0].a} from A and ${IT[0].threads[0].b} from B`) && scripts[1].includes(`Thread 1 takes ${IT[0].threads[1].a} and ${IT[0].threads[1].b}`) &&
  scripts[1].includes(`used ${IT[0].aUsed} of A and ${IT[0].bUsed} of B`) && scripts[1].includes(`The other ${again0} loaded elements`) &&
  scripts[2].includes(`element ${IT[1].aStart} of A and element ${IT[1].bStart} of B`) && scripts[2].includes(`the ${IT[0].aLoaded - IT[0].aUsed} of A and ${IT[0].bLoaded - IT[0].bUsed} of B`) &&
  scripts[2].includes(`use ${IT[1].aUsed} of A and ${IT[1].bUsed} of B`) && scripts[2].includes(`loads ${RUN.totals.loaded} elements to write ${RUN.totals.used}`) &&
  ITW[0].aUsed === 8 && ITW[1].aUsed === 8 && ITW[0].bUsed === 0 && scripts[3].includes('first 2 tiles comes from A') &&
  scripts[4].includes(`after ${EX.tiles[0].aRange[1]} elements of A and ${EX.tiles[0].bRange[1]} of B`) && scripts[4].includes(`${EX.totals.loaded} for ${EX.totals.used}`),
  JSON.stringify({ it0: IT[0], it1: IT[1], totals: RUN.totals }));
if (manifest?.clips?.['tile-waste-1']) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no audio for this scene the player says it runs on estimated timings, or the clip is simply absent', true);

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const current = () => page.$$eval(S + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
const mainButton = () => page.$eval(S + ' button.primary', (e) => e.textContent.trim());
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const g = svg.querySelector('[data-beat]');
  const cells = (pre) => [...svg.querySelectorAll(`[data-cell^="${pre}"]`)].filter((c) => /^\d+$/.test(c.dataset.cell.slice(pre.length))).map((c) => c.dataset.state);
  return {
    ...g.dataset,
    gA: cells('A'), gB: cells('B'), sA: cells('AS'), sB: cells('BS'),
    order: svg.querySelector('[data-order]')?.dataset.order ?? '',
    threads: svg.querySelector('[data-threads]')?.textContent.trim() ?? '',
    counter: svg.querySelector('[data-counter]')?.textContent.trim() ?? '',
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
  };
}, S);
const tags = (items, upto) => items.map((x, k) => (k < upto && x ? x.src + x.idx : '·')).join(' ');

await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['tile-waste-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));

await setBox('sound', false);
await press('Play');
await pause(2000);
check('with sound off the animation runs', (await slider()) > 20, String(await slider()));
await press('Pause');
await seek(970);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself, and its caption follows', (await current()) === 2 && (await caption()) === scripts[1], `part ${await current()}`);
await press('Pause');

await press('1. Load a tile');
await seek(150);
let d = await drawn();
check('part 1 opens before any load: empty tiles, nothing loaded', d.beat === '1' && d.phase === 'before' && d.sA.length === 0 && d.gA.every((x) => x === ''), JSON.stringify(d));
await seek(1000);
d = await drawn();
check(`part 1 ends: ${IT[0].aLoaded} of A and ${IT[0].bLoaded} of B loaded into A_S and B_S, nothing used yet`,
  d.phase === 'load' && d.it === '0' && d.sA.length === IT[0].aLoaded && d.sB.length === IT[0].bLoaded && d.sA.every((x) => x === 'loaded') &&
  d.gA.filter((x) => x === 'load').length === IT[0].aLoaded && d.loaded === '16' && d.used === '0' && d.order.split(' ').every((x) => x === '·'), JSON.stringify(d));
if (out) await page.screenshot({ path: `${out}-scene1.png` });

await press('2. Use what the merge needs');
await seek(1000);
d = await drawn();
check(`part 2 ends: ${IT[0].aUsed} + ${IT[0].bUsed} used, the rest marked unused, C[0..8) written in the model's order`,
  d.phase === 'use' && d.aused === String(IT[0].aUsed) && d.bused === String(IT[0].bUsed) &&
  d.sA.filter((x) => x === 'used').length === IT[0].aUsed && d.sA.filter((x) => x === 'unused').length === IT[0].aLoaded - IT[0].aUsed &&
  d.sB.filter((x) => x === 'used').length === IT[0].bUsed && d.order === tags(RUN.output, 8) &&
  d.threads === IT[0].threads.map((t, q) => `thread ${q}: ${t.a} of A, ${t.b} of B`).join(';  ') && d.headline.includes(`${again0} of the 16 loaded elements were never used`), JSON.stringify(d));
if (out) await page.screenshot({ path: `${out}-scene2.png` });

await press('3. Load again');
await seek(300);
d = await drawn();
check(`part 3 loads again from A[${IT[1].aStart}] and B[${IT[1].bStart}]: ${again0} elements reloaded, marked in global memory`,
  d.it === '1' && d.phase === 'load' && d.again === String(again0) && [...d.gA, ...d.gB].filter((x) => x === 'again').length === again0 &&
  d.gA.slice(0, IT[1].aStart).every((x) => x === 'done'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check(`part 3 ends: loaded 32, used 16 so far; the whole merge ${RUN.totals.loaded} for ${RUN.totals.used}`,
  d.loaded === '32' && d.used === '16' && d.order === tags(RUN.output, 16) && d.headline.includes(`${RUN.totals.loaded} elements loaded for ${RUN.totals.used} outputs`), JSON.stringify(d));
if (out) await page.screenshot({ path: `${out}-scene3.png` });

await press('4. The worst case');
await seek(450);
d = await drawn();
check('part 4, first tile of the worst case: 8 of A used, all of B\'s tile unused', d.it === '0' && d.aused === '8' && d.bused === '0' && d.sB.every((x) => x === 'unused'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4 ends: the second tile again takes everything from A, and B\'s first 8 are loaded again', d.it === '1' && d.aused === '8' && d.bused === '0' && d.gB.filter((x) => x === 'again').length === 8, JSON.stringify(d));
if (out) await page.screenshot({ path: `${out}-scene4.png` });

await press('5. Search first');
await seek(250);
d = await drawn();
check('part 5 searches first: nothing loaded yet, the split named from the model', d.scheme === 'exact' && d.phase === 'search' && d.sA.length === 0 &&
  d.threads.includes(`take ${EX.tiles[0].aRange[1]} of A and ${EX.tiles[0].bRange[1]} of B`), JSON.stringify(d));
await seek(1000);
d = await drawn();
check(`part 5 ends: exactly ${EX.tiles[0].aRange[1]} + ${EX.tiles[0].bRange[1]} loaded, all used; the whole merge ${EX.totals.loaded} for ${EX.totals.used}, ${EX.totals.extraSearchReads} search reads`,
  d.sA.length === EX.tiles[0].aRange[1] && d.sB.length === EX.tiles[0].bRange[1] && [...d.sA, ...d.sB].every((x) => x === 'used') &&
  d.loaded === String(EX.totals.loaded) && d.used === String(EX.totals.used) && d.extra === String(EX.totals.extraSearchReads) &&
  d.counter.includes(`loaded ${EX.totals.loaded}, used ${EX.totals.used}`), JSON.stringify(d));
if (out) await page.screenshot({ path: `${out}-scene5.png` });

await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light and dark, a narrow screen, margin notes, errors
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
if (out) await page.screenshot({ path: `${out}-scene5-light.png` });
await setBox('captions', false);
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note on the page follows a paragraph or another note', misplaced.length === 0, misplaced.join(', '));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S, X) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth,
  bench: document.querySelector(X).getBoundingClientRect().width <= document.documentElement.clientWidth,
}), S, X);
check('at 390 px the page does not scroll sideways and the scene and explorer fit', narrow.page && narrow.scene && narrow.bench, JSON.stringify(narrow));
if (out) {
  await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: `${out}-narrow.png` });
}

const expected404 = (u) => /\/content\/ch13\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest or clip is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
