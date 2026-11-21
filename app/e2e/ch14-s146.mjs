// Browser check for section 14.6: the "Where a warp's writes land" explorer (every key set,
// block size, warp width, digit width, pass, memory unit and design), the bucket-table
// explorer (presets, threads, edited counts, the wrong layout), two predictions, the lab
// listing and the questions. Every number asserted is recomputed here from the model
// (src/sim/radixlocal.ts), not typed in. The narrated scene has its own check
// (ch14-scene-local-sort.mjs).
// Usage: node e2e/ch14-s146.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

// The models import their siblings without a file extension (as the app's bundler allows).
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
const M = await import('../src/sim/radixlocal.ts');
const RC = await import('../src/sim/radixcost.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const dec = (v) => (Math.round(v * 10) / 10).toFixed(1);
const WW = 'Where a warp\'s writes land';
const BT = 'The bucket table';
const bench = (t) => `section.bench[aria-label="${t}"]`;
const shot = async (sel, name) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

/** Press the button with this text inside the seg group with this label, in a bench. */
const seg = async (title, group, text) => {
  const done = await page.evaluate((sel, group, text) => {
    const g = document.querySelector(sel + ` .seg[aria-label="${group}"]`);
    const b = g && [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === text);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, bench(title), group, text);
  await pause(80);
  return done;
};
const range = async (title, label, v) => {
  await page.$eval(`${bench(title)} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const select = async (title, label, v) => {
  await page.$eval(`${bench(title)} select[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
};
const txt = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const ww = (s) => `${bench(WW)} ${s}`;
const avg = async () => Object.fromEntries(await page.$$eval(ww('[data-avg]'), (es) => es.map((e) => [e.dataset.avg, e.textContent.trim()])));

// The model's view of the explorer for given settings.
const KEYS = { uniform: M.uniformKeys(4096, 1), softmax: M.softmaxKeys(4096, 1), book: M.BOOK_KEYS };
function model({ ks = 'uniform', B = 256, W = 32, r = 1, pass = 0, unit = 32, design = 'direct' }) {
  const input = M.passInput(KEYS[ks], r, pass);
  const shift = pass * r;
  const d = M.warpStores(M.directPass(input, shift, r, B), W, 4, unit);
  const l = M.warpStores(M.localPass(input, shift, r, B), W, 4, unit);
  return { stores: design === 'direct' ? d : l, d: M.summarise(d), l: M.summarise(l) };
}
const storeMatches = async (w, label) => {
  const units = await txt(ww('[data-readout="units"]'));
  const s = await txt(ww('[data-readout="store"]'));
  const digits = await txt(ww('[data-readout="digits"]'));
  check(label, units.startsWith(`${w.units.length} `) && s.includes(`${w.runs} run`) && s.endsWith(`need ${w.minUnits}.`) && digits === `Lane digits: ${w.digits.join(' ')}`,
    JSON.stringify({ units, s, digits, model: { units: w.units.length, runs: w.runs, min: w.minUnits } }));
};
const avgMatches = async (m, label) => {
  const a = await avg();
  check(label, a['direct-runs'] === dec(m.d.meanRuns) && a['local-runs'] === dec(m.l.meanRuns) && a['direct-units'] === dec(m.d.meanUnits) && a['local-units'] === dec(m.l.meanUnits),
    JSON.stringify({ page: a, model: [m.d.meanRuns, m.l.meanRuns, m.d.meanUnits, m.l.meanUnits].map(dec) }));
};

await goto(page, 'ch14/14.6');
await page.waitForSelector(bench(WW));
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });

// ---- explorer 1: defaults (random keys, 256 threads, 32 lanes, r = 1, pass 0, NVIDIA sectors, direct)
let m = model({});
await storeMatches(m.stores[0], 'default store 1 matches the model: runs, sectors, minimum, lane digits');
await avgMatches(m, 'default averages match the model for both designs');
check('one bit: a direct store makes at most 2 runs and 4 to 7 sectors', m.stores.every((w) => w.runs <= 2 && w.units.length >= 4 && w.units.length <= 7));
check('the strip marks one triangle per run', Number(await page.$eval(ww('[data-strip="wide"]'), (e) => e.dataset.runs)) === m.stores[0].runs);
await shot(bench(WW), 'warp-default');

await range(WW, 'Which warp store', 37);
await storeMatches(m.stores[37], 'store 38 matches the model');

// r = 8
await seg(WW, 'Digit width r', '8');
m = model({ r: 8 });
await storeMatches(m.stores[0], 'r = 8: store 1 matches the model');
await avgMatches(m, 'r = 8: averages match, the direct scatter nears a sector per key');
check('r = 8: direct averages more than 28 sectors per store', m.d.meanUnits > 28 && (await avg())['direct-units'] === dec(m.d.meanUnits));
check('the runs bound is stated as min(lanes, 2^r)', (await txt(bench(WW))).includes('at most 32 runs per store here'));
await seg(WW, 'Design', 'local sort, then write');
await storeMatches(model({ r: 8, design: 'local' }).stores[0], 'local design: store 1 matches the model');

// r = 4: the local sort's big win
await seg(WW, 'Digit width r', '4');
m = model({ r: 4, design: 'local' });
await avgMatches(m, 'r = 4: averages match');
check('r = 4: the local sort cuts sectors per store by more than half', m.l.meanUnits < m.d.meanUnits / 2);
await shot(bench(WW), 'warp-r4-local');

// a block no wider than a warp: nothing changes
await seg(WW, 'Threads per block', '4');
const a4 = await avg();
check('4 threads per block: both designs touch the same sectors', a4['direct-units'] === a4['local-units'] && a4['direct-units'] === dec(model({ r: 4, B: 4 }).d.meanUnits), JSON.stringify(a4));
await seg(WW, 'Threads per block', '128');
await avgMatches(model({ r: 4, B: 128 }), '128 threads per block: averages match');

// a 64-wide wavefront
await seg(WW, 'Warp width', '64 (AMD gfx9)');
m = model({ r: 4, B: 128, W: 64, design: 'local' });
await avgMatches(m, '64-lane warps: averages match');
await storeMatches(m.stores[0], '64-lane warps: store 1 matches');
check('64-lane warps: the coalesced minimum is 8 sectors', m.stores[0].minUnits === 8);
await seg(WW, 'Warp width', '32');
await seg(WW, 'Threads per block', '256');

// passes: slider for 32-bit keys
await range(WW, 'Pass', 5);
m = model({ r: 4, pass: 5, design: 'local' });
await avgMatches(m, 'r = 4, pass 5: averages match');
check('the pass label names its bits', (await txt(bench(WW))).includes('Pass 5 of passes 0 to 7 (bits 20 to 23)'));

// memory units
await select(WW, 'Memory unit', 'nv-line');
m = model({ r: 4, pass: 5, unit: 128, design: 'local' });
await avgMatches(m, '128-byte lines: averages match');
await select(WW, 'Memory unit', 'amd-64');
await avgMatches(model({ r: 4, pass: 5, unit: 64, design: 'local' }), 'AMD 64-byte lines: averages match');
await select(WW, 'Memory unit', 'metal');
check('Metal: the page says no unit is documented', (await page.$(ww('[data-metal-note]'))) !== null && (await txt(ww('[data-metal-note]'))).includes('Apple documents no coalescing rule'));
await seg(WW, 'Your segment size (bytes)', '64');
await avgMatches(model({ r: 4, pass: 5, unit: 64, design: 'local' }), 'Metal with a 64-byte pick: averages match');
await select(WW, 'Memory unit', 'nv-sector');

// the softmax row: the last pass is already coalesced
await seg(WW, 'Keys', 'a 4,096-token softmax row');
await seg(WW, 'Digit width r', '4');
await range(WW, 'Pass', 7);
m = model({ ks: 'softmax', r: 4, pass: 7 });
await avgMatches(m, 'softmax row, r = 4, last pass: averages match');
check('softmax row, last pass: one run per store, direct', (await avg())['direct-runs'] === '1.0');
await range(WW, 'Pass', 0);
await avgMatches(model({ ks: 'softmax', r: 4, pass: 0 }), 'softmax row, first pass: averages match');

// the book's keys
await seg(WW, 'Keys', "the book's sixteen");
check('the book: only 4 threads per block can be chosen', await page.$eval(ww('.seg[aria-label^="Threads per block"] button[aria-pressed="true"]'), (b) => b.textContent.trim()) === '4'
  && !(await seg(WW, 'Threads per block (the figure has 4)', '256')));
await seg(WW, 'Digit width r', '1');
await seg(WW, 'Design', 'direct scatter');
const hits = await page.$$eval(ww('[data-out][data-hit="1"]'), (es) => es.map((e) => Number(e.dataset.out)).sort((a, b) => a - b));
check('the book, direct: block 0 writes positions 0, 1, 8 and 9 (Fig. 14.5)', hits.join(',') === '0,1,8,9', hits.join(','));
const outs = await page.$$eval(ww('[data-out] text'), (es) => es.map((e) => e.textContent).join(' '));
check('the book: the output row is iteration 1 of Fig. 14.4', outs === M.passInput(M.BOOK_KEYS, 1, 1).map((k) => M.bin(k, 4)).join(' '), outs);
await shot(bench(WW), 'warp-book-direct');
await range(WW, 'Which warp store', 2);
await seg(WW, 'Design', 'local sort, then write');
const hits2 = await page.$$eval(ww('[data-out][data-hit="1"]'), (es) => es.map((e) => Number(e.dataset.out)).sort((a, b) => a - b));
check('the book, local: block 2 writes 4, 12, 13 and 14 (p. 342)', hits2.join(',') === '4,12,13,14', hits2.join(','));
const src = await page.$$eval(ww('[data-src] text'), (es) => es.map((e) => e.textContent).join(' '));
check('the book, local: the source row is each block sorted in shared memory (Fig. 14.8)',
  src === M.localSortBlocks(M.BOOK_KEYS, 0, 1, 4).flatMap((b) => b.sorted).map((k) => M.bin(k, 4)).join(' '), src);
await storeMatches(model({ ks: 'book', B: 4, design: 'local' }).stores[2], 'the book, local: block 2 store matches the model');
await shot(bench(WW), 'warp-book-local');

// ---- prediction before the explorer
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Before you open the explorer')));
await (await pred.$('input[type=number]')).type('20');
await (await pred.$('button.primary')).click();
await pause(150);
const p8 = M.summarise(M.warpStores(M.directPass(KEYS.uniform, 0, 8, 256), 32, 4, 32));
const predText = await pred.evaluate((e) => e.textContent);
check('the r = 8 prediction reveals the model\'s figure', predText.includes(`gives ${dec(p8.meanUnits)} sectors per store`) && predText.includes('More than you said'), predText);

// ---- explorer 2: the bucket table
const bt = (s) => `${bench(BT)} ${s}`;
const starts = () => page.$$eval(bt('[data-start]'), (es) => Object.fromEntries(es.map((e) => [e.dataset.start, Number(e.textContent)])));
const readout = () => txt(bt('[data-readout="thread"]'));
const clickThread = async (b, t) => {
  await page.$eval(bt(`button[aria-label="block ${b} thread ${t}"]`), (e) => e.click());
  await pause(80);
};
let st = await starts();
check('Fig. 14.9: zero-bucket starts 0, 2, 4, 5 and one-bucket starts 8, 10, 12, 15',
  [0, 1, 2, 3].map((b) => st[`${b}-0`]).join(',') === '0,2,4,5' && [0, 1, 2, 3].map((b) => st[`${b}-1`]).join(',') === '8,10,12,15', JSON.stringify(st));
const tbl = await page.$$eval(bt('[data-table]'), (es) => es.map((e) => Number(e.textContent)));
const scn = await page.$$eval(bt('[data-scan]'), (es) => es.map((e) => Number(e.textContent)));
const bookCounts = M.localSortBlocks(M.BOOK_KEYS, 0, 1, 4).map((b) => b.counts);
check('the table is digit-major and its scan is the model\'s', tbl.join(',') === M.bucketTable(bookCounts, 'digit-major').join(',') && scn.join(',') === M.exclusiveScan(tbl).join(','), `${tbl} | ${scn}`);
check('the default thread (block 2, thread 0) writes its zero to 4', (await readout()).includes('= 4.'), await readout());
await seg(BT, 'Presets', 'p. 342: block 2');
check('preset block 2: thread 1 writes its first one to 12 + 0 = 12', (await readout()).includes('12 + 0 = 12'), await readout());
await clickThread(2, 3);
check('block 2, thread 3 writes 14', (await readout()).includes('12 + 2 = 14'), await readout());
await seg(BT, 'Presets', 'p. 342: block 3');
check('preset block 3: thread 3 writes the block\'s one to 15', (await readout()).includes('15 + 0 = 15'), await readout());
await clickThread(3, 2);
check('block 3, thread 2 writes a zero to 5 + 2 = 7', (await readout()).includes('5 + 2 = 7'), await readout());
check('the book\'s keys are shown in the threads while the counts are the book\'s', (await txt(bt('button[aria-label="block 3 thread 3"]'))).includes('0111'));
await shot(bench(BT), 'table-book');

// edit the counts to the question's [3, 1, 2, 2]
await page.$eval(bt('button[aria-label="more zeros in block 0"]'), (e) => e.click());
await pause(60);
await page.$eval(bt('button[aria-label="fewer zeros in block 1"]'), (e) => e.click());
await pause(60);
await page.$eval(bt('button[aria-label="more zeros in block 2"]'), (e) => e.click());
await pause(60);
await page.$eval(bt('button[aria-label="fewer zeros in block 3"]'), (e) => e.click());
await pause(100);
const counts = await page.$$eval(bt('[data-counts]'), (es) => es.map((e) => e.dataset.counts));
const qc = [[3, 1], [1, 3], [2, 2], [2, 2]];
check('the counts can be edited and stay four per block', counts.join('|') === qc.map((c) => c.join(' ')).join('|'), counts.join('|'));
st = await starts();
const ms = M.starts(qc, 'digit-major');
check('edited counts: every start follows the model', [0, 1, 2, 3].every((b) => st[`${b}-0`] === ms[b][0] && st[`${b}-1`] === ms[b][1]), JSON.stringify(st));
await clickThread(1, 2);
check('edited counts: block 1, thread 2 writes 9 + 1 = 10 (the q-thread example)', (await readout()).includes('9 + 1 = 10'), await readout());
check('edited counts: the output is partitioned', (await page.$eval(bt('[data-verdict]'), (e) => e.dataset.verdict)) === 'partitioned');

// the wrong layout
await seg(BT, 'Table layout before the scan', 'block by block (wrong)');
const wrongDigits = M.placeFromCounts(qc, 'block-major').sort((a, b) => a.addr - b.addr).map((x) => x.digit).join('');
check('block-major: the output digits follow the model and are not partitioned',
  (await page.$eval(bt('[data-output]'), (e) => e.dataset.output)) === wrongDigits && (await page.$eval(bt('[data-verdict]'), (e) => e.dataset.verdict)) === 'not-partitioned', wrongDigits);
check('block-major: the misplaced cells are marked as failures', (await page.$$(bt('[data-bad="1"]'))).length > 0);
check('block-major: the table header lists entries block by block', (await page.$$eval(bt('thead th'), (es) => es.map((e) => e.textContent.trim()))).slice(1, 3).join('|') === 'zeros b0|ones b0');
await shot(bench(BT), 'table-wrong');
await seg(BT, 'Table layout before the scan', 'all zero counts, then all one counts');
check('back to digit-major: partitioned again', (await page.$eval(bt('[data-verdict]'), (e) => e.dataset.verdict)) === 'partitioned');

// ---- questions
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-start accepts 12', (await answer("block 2's one bucket start", M.starts(qc, 'digit-major')[2][1])).includes('Correct.'));
check('q-thread accepts 10', (await answer("block 1's thread 2", 10)).includes('Correct.'));
check('q-sectors accepts 5 and rejects 4', (await answer('somewhere else', 4)).includes('The answer is 5'));
const layoutQ = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Why must the table list')));
await (await layoutQ.$$('input[type=radio]'))[1].click();
await (await layoutQ.$('button.primary')).click();
await pause(120);
check('q-layout: the second option is right', (await layoutQ.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- the lab listing and the measured prediction
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the lab 14.2 listing has CUDA-and-HIP and Metal tabs', tabs.includes('CUDA and HIP') && tabs.includes('Metal'), tabs.join('|'));
const status = await page.$eval('.code .tabs .status', (e) => e.textContent);
const reports = await (await fetch(BASE + '/api/reports')).json();
{
  // The labs page's wording (LabKit's solutionTabStatus), from the shared helper, with this
  // machine's CUDA front-end check.
  const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
  const want = tabStatus(reports, RC.LAB2, 'kernel.cu', frontend);
  check(`the CUDA tab's status is the labs page's, from the reports: "${want}"`, status === want, status);
}
const devs = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').filter((b) => {
  const rows = reports[b].results.filter((x) => x.lab === RC.LAB2 && x.variant === 'solution').flatMap((x) => x.bench);
  return RC.measuredLocal(rows, 1, 1) != null && RC.measuredOneBit(rows) != null;
});
const gain = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many times faster')));
await (await gain.$('input[type=number]')).type('2');
await (await gain.$('button.primary')).click();
await pause(150);
const measured = await gain.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((x) => x.dataset.measured));
check(devs.length ? 'the gain prediction reveals one line per measured device' : 'with no lab 14.2 rows the gain prediction says how to measure',
  devs.length ? devs.every((d) => measured.includes(d)) : measured.join() === 'none' && (await gain.evaluate((e) => e.textContent)).includes('pmpp.py run 14.2'), measured.join());

// ---- themes and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await shot(bench(WW), 'warp-dark');
await shot(bench(BT), 'table-dark');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
await shot(bench(WW), 'warp-narrow');
await shot(bench(BT), 'table-narrow');

const expected404 = (u) => /\/content\/ch14\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u) || /\/api\/lab-file\?path=ch14%2Flab2_multibit/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest or lab 14.2 file is allowed)', realErrors.length === 0 && missing.every(expected404),
  [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
