// Browser check for section 7.6: the lab listing, the tap explorer (Fig. 7.15), the toy
// cache explorer, the narrated scene "Where the halo comes from", a prediction and the
// questions. Every number it expects comes from the models, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/ch07-s76.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { blockHistogram, classifyTaps, makeProblem, runCached, warpSplits } from '../src/sim/convcached.ts';
import { sceneFacts, summarise } from '../src/sim/halocache.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const T1 = 'Where each tap comes from';
const T2 = 'Will the halo be in the cache?';
const SC = 'section.scene[aria-label="Where the halo comes from"]';
const bench = (t) => `section.bench[aria-label="${t}"]`;
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a).catch(() => null);
const setSelect = async (scope, label, value) => {
  const ok = await page.evaluate((sel, label, value) => {
    const l = [...document.querySelectorAll(sel + ' label.field')].find((x) => x.textContent.trim().startsWith(label));
    const s = l?.querySelector('select');
    if (!s) return false;
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, bench(scope), label, value);
  await pause(150);
  return ok;
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(200);
};
const press = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const b = [...document.querySelectorAll(sel + ' button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(150);
  return done;
};

await goto(page, 'ch07/7.6');
await page.waitForSelector(bench(T1));

// ---------------------------------------------------------------- the listing
const labCu = path.join(repo, 'labs/ch07/lab2_tiledconv/solution/kernel.cu');
if (fs.existsSync(labCu)) {
  // The reference listing is part B's answer: it waits behind a button unless the reader's lab 7.2 passes.
  const allReports = await (await fetch(BASE + '/api/reports')).json();
  const repaired = Object.values(allReports).some((r) => {
    if (r.work_state?.['ch07/lab2_tiledconv'] === 'untouched') return false;
    const rs = r.results.filter((x) => x.lab === 'ch07/lab2_tiledconv' && x.variant === 'work' && x.tests.length);
    return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
  });
  if (!repaired) {
    check('the repaired kernel is gated until asked for', !!(await page.$('[data-listing="gated"]')) && !(await page.$('.code .tabs button')));
    await press('[data-listing="gated"]', 'Show the repaired kernel now');
  } else check('with lab 7.2 passing here, the listing shows without asking', !(await page.$('[data-listing="gated"]')));
  await page.waitForSelector('.code .tabs button');
  const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
  check('listing shows CUDA and Metal kernel tabs', tabs.includes('CUDA') && tabs.includes('Metal kernel'), tabs.join('|'));
  const code = await page.$eval('.code pre', (e) => e.textContent);
  check('CUDA tab is the cached part of lab 7.2\'s solution, and only that part',
    code.includes('convCachedKernel') && !code.includes('convTiledHaloKernel') && !code.includes('launchConvTiledCached'), code.slice(0, 200));
  const status = await page.$eval('.code .tabs .status', (e) => e.textContent);
  check('the CUDA tab status uses the fixed vocabulary', /emulator-checked|emulator: /.test(status) && /(verified|unverified) on CUDA/.test(status) && !/^verified/.test(status), status);
} else {
  check('with lab 7.2 missing, the page says so instead of showing an error', !!(await page.$('[data-listing="missing"]')));
}

// ---------------------------------------------------------------- explorer 1: taps
const spec = { tile: 32, r: 2, width: 101, height: 99, bx: 1, by: 1 };
const h = blockHistogram(spec);
check('default interior block: 23716 shared, 1884 global, 0 ghost of 25600 taps (the model agrees)',
  (await attr(bench(T1) + ' [data-hist]', 'data-hist')) === '23716,1884,0,25600' && h.shared === 23716 && h.global === 1884,
  await attr(bench(T1) + ' [data-hist]', 'data-hist'));
check('reads of N per output if no cache helps: 2.84', (await attr(bench(T1) + ' [data-reads-all]', 'data-reads-all')) === '2.84');
check('thread (0, 0): 9 shared taps and 16 global',
  (await attr(bench(T1) + ' [data-mine-shared]', 'data-mine-shared')) === '9' && (await attr(bench(T1) + ' [data-mine-global]', 'data-mine-global')) === '16');
check('the guard table shows tx - R + fCol = 4294967294 for tx 0, R 2, fCol 0',
  (await attr(bench(T1) + ' tr[data-fcol="0"] [data-u32]', 'data-u32')) === '4294967294');
const rowTxt = await page.$eval(bench(T1) + ' tr[data-fcol="0"]', (e) => [...e.querySelectorAll('td')].map((t) => t.textContent));
check('... with ">= 0" true and "< TILE_DIM" false', rowTxt[3] === 'true' && rowTxt[4] === 'false', rowTxt.join('|'));
check('32 of 32 warps of 32 diverge at the in-tile test', (await attr(bench(T1) + ' [data-warps]', 'data-warps')) === `${warpSplits(32, 2, 32).diverging},32`);
await setSelect(T1, 'Warp width', 64);
check('with warps of 64: 16 of 16', (await attr(bench(T1) + ' [data-warps]', 'data-warps')) === '16,16');
await setSelect(T1, 'Warp width', 32);
check('the corrected kernel matches the reference on the whole image', (await attr(bench(T1) + ' [data-run]', 'data-run')).startsWith('0,'));
const shot = async (sel, name) => { if (out) await (await page.$(sel)).screenshot({ path: `${out}-${name}.png` }); };
await shot(bench(T1), 'taps');

// Click thread (2, 2) in the footprint: its 25 taps all stay in shared memory.
const S = 32 + 4;
await page.evaluate((sel, k) => {
  const r = document.querySelector(sel + ' svg[aria-label^="The block\'s footprint"]').querySelectorAll('rect')[k];
  r.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}, bench(T1), (2 + 2) * S + (2 + 2));
await pause(150);
check('clicking a cell selects thread (2, 2), whose taps all read shared memory',
  (await attr(bench(T1) + ' [data-thread]', 'data-thread')) === '2,2' && (await attr(bench(T1) + ' [data-mine-global]', 'data-mine-global')) === '0');

// Line 21 as printed.
await page.evaluate((sel) => {
  [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('line 21 as printed')).querySelector('input').click();
}, bench(T1));
await pause(400);
check('printed mode, the block: wrong, past the tile and outside N_s counts are the model\'s',
  (await attr(bench(T1) + ' [data-block-printed]', 'data-block-printed')) === `${h.printedWrong},${h.printedPastTile},${h.printedOutside}`,
  await attr(bench(T1) + ' [data-block-printed]', 'data-block-printed'));
await setSelect(T1, 'threadIdx.x', 31);
await setSelect(T1, 'threadIdx.y', 31);
const t31 = classifyTaps({ ...spec, tx: 31, ty: 31 }).taps;
const want31 = [t31.filter((p) => p.wrongElement).length, t31.filter((p) => p.pastTile).length, t31.filter((p) => p.outsideArray).length].join(',');
check('printed mode, thread (31, 31): every shared tap is wrong and some leave N_s', (await attr(bench(T1) + ' [data-thread-printed]', 'data-thread-printed')) === want31 && Number(want31.split(',')[2]) > 0,
  `${await attr(bench(T1) + ' [data-thread-printed]', 'data-thread-printed')} vs ${want31}`);
check('the printed reads past the tile are drawn', (await page.$$(bench(T1) + ' [data-printed="outside"]')).length === Number(want31.split(',')[2]));
const printedRun = runCached(makeProblem(101, 99, 2), { tile: 32, asPrinted: true });
check('running the whole kernel as printed: the model\'s wrong-output and outside counts',
  (await attr(bench(T1) + ' [data-run]', 'data-run')) === `${printedRun.wrong.length},${101 * 99},${printedRun.pastTileReads},${printedRun.outsideReads}`,
  await attr(bench(T1) + ' [data-run]', 'data-run'));
await shot(bench(T1), 'printed');

// Corner block: ghost taps.
await press(bench(T1), 'corner');
const hc = blockHistogram({ ...spec, bx: 0, by: 0 });
check('the corner block has ghost taps, as the model counts', (await attr(bench(T1) + ' [data-hist]', 'data-hist')) === `${hc.shared},${hc.global},${hc.ghost},${hc.taps}` && hc.ghost > 0);

// ---------------------------------------------------------------- explorer 2: the toy cache
const base = { tilesX: 8, tilesY: 8, tile: 32, r: 2, lineFloats: 32, capacity: 1024, resident: 8, slices: 1, placement: 'in-order', seed: 1 };
const misses = () => attr(bench(T2) + ' [data-misses]', 'data-misses').then(Number);
const lines = (id) => attr(bench(T2) + ` [data-bar="${id}"]`, 'data-lines').then(Number);
const s0 = summarise(base);
check('default: one shared L2 of 1024 lines reaches the ideal (2048 line fetches)', (await misses()) === s0.lines.run && s0.lines.run === s0.lines.ideal, String(await misses()));
check('... and the bars carry the model\'s bytes per output', (await attr(bench(T2) + ' [data-bar="allMiss"]', 'data-bytes')) === s0.bytes.allMiss.toFixed(4));
const slider = bench(T2) + ' input[type=range]';
await setRange(slider, 0);
check('capacity 0 gives the all-miss count', (await misses()) === (await lines('allMiss')) && (await misses()) === s0.lines.allMiss, `${await misses()} vs ${s0.lines.allMiss}`);
await setRange(slider, 12);
check('an unbounded cache gives the ideal', (await attr(bench(T2) + ' [data-capacity]', 'data-capacity')) === 'unbounded' && (await misses()) === s0.lines.ideal);
await setRange(slider, 9); // 4096 lines
const one = await misses();
await setSelect(T2, 'L2 slices', 8);
await setSelect(T2, 'Placement on slices', 'random');
const eight = await misses();
const model8 = summarise({ ...base, capacity: 4096, slices: 8, placement: 'random' }).lines.run;
check('eight slices with random placement give more misses than one L2 (the model\'s count)', eight > one && eight === model8, `${one} -> ${eight} (model ${model8})`);
check('... and more DRAM traffic than Fig. 7.12 with no reuse', Number(await attr(bench(T2) + ' [data-bar="run"]', 'data-bytes')) > Number(await attr(bench(T2) + ' [data-bar="fig712"]', 'data-bytes')));
check('the heat map has one cell per block', Number(await attr(bench(T2) + ' [data-heat]', 'data-heat')) === 64);
await shot(bench(T2), 'cache');

// ---------------------------------------------------------------- the scene
const F = sceneFacts();
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 5; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch07/narration/halo-from-cache-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('five scripts, each 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)));
check('script 2 says what the model finds: 2 columns of A, every line still in L2', scripts[1].includes('last 2 columns') && F.bLeftHits === F.bLeft && F.bLeft === 8);
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const labels = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts with their names', labels.join('|') === '1. Block A|2. Next door|3. A row later|4. Split L2|5. Which bar', labels.join('|'));
const capBox = () => page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('there is a captions switch, off by default', !(await (await capBox()).evaluate((e) => e.checked)));
const res = await fetch(BASE + '/content/ch07/narration/manifest.json');
const manifest = res.ok ? await res.json() : null; // the server answers null while no audio exists
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
if (!manifest?.clips?.['halo-from-cache-1'])
  check('with no audio, part 1 shows its script even with captions off',
    (await page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => '')) === scripts[0]);
else check('with audio and captions off, no caption is shown', !(await page.$(SC + ' .caption')));
const seekEnd = () => setRange(SC + ' input[type=range]', 1000);
const counter = (name) => page.$eval(`${SC} [data-counter="${name}"]`, (g) => [Number(g.dataset.hits), Number(g.dataset.of)]).catch(() => null);
await press(SC, '1. Block A');
await seekEnd();
check('part 1 ends: A\'s 8 lines all came from DRAM', JSON.stringify(await counter('a-load')) === '[0,8]', JSON.stringify(await counter('a-load')));
await press(SC, '2. Next door');
await seekEnd();
check('part 2 ends: B finds all 8 of A\'s lines in L2, and its own 8 too',
  JSON.stringify(await counter('b-left')) === JSON.stringify([F.bLeftHits, F.bLeft]) && JSON.stringify(await counter('b-load')) === '[8,8]');
const hl = await page.$eval(SC + ' [data-headline]', (e) => e.textContent);
check('... and the headline says so', hl.includes(`${F.bLeftHits} of ${F.bLeft} lines found in L2`), hl);
await shot(SC, 'scene2');
await press(SC, '3. A row later');
await seekEnd();
check('part 3 ends: C finds 0 of A\'s 2 bottom lines in L2', JSON.stringify(await counter('c-on-a')) === JSON.stringify([F.cOnABottomHits, F.cOnABottom]) && F.cOnABottom === 2 && F.cOnABottomHits === 0);
await press(SC, '4. Split L2');
await seekEnd();
const dies = await attr(SC + ' [data-dies]', 'data-dies');
check('part 4 ends: A and B on different dies, B finds 0 of 8 of A\'s lines',
  dies === `${F.sliceA},${F.sliceB}` && F.sliceA !== F.sliceB && JSON.stringify(await counter('split-b-left')) === '[0,8]', dies);
await shot(SC, 'scene4');
await press(SC, '5. Which bar');
await seekEnd();
const bars = await page.$$eval(SC + ' [data-bar]', (gs) => Object.fromEntries(gs.map((g) => [g.dataset.bar, Number(g.dataset.bytes)])));
check('part 5: bars are the model\'s 8, 9.5, 16.3 and 19.1 bytes per output',
  bars.ideal === F.bytes.ideal && bars.one === F.bytes.one && bars.split === F.bytes.split && bars.none === F.bytes.noReuse, JSON.stringify(bars));
const soundBox = await page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'), SC);
await soundBox.click();
await press(SC, '1. Block A');
await press(SC, 'Play');
await pause(1500);
const moved = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('with sound off, the animation runs', moved > 20, String(moved));
await press(SC, 'Pause');

// ---------------------------------------------------------------- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('leave shared memory and read')));
await (await pred.$('input[type=number]')).type('10');
await (await pred.$('button.primary')).click();
await pause(200);
check('the prediction reveals the model\'s 7.4%', (await pred.evaluate((e) => e.querySelector('[data-reveal]')?.dataset.reveal)) === '7.4');
const choose = async (needle, optionText) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, t) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(t)).querySelector('input').click(), optionText);
  await pause(80);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
check('q-unsigned accepts "true: wraps to 4294967294"', (await choose('In CUDA C++, what is', 'wraps to 4294967294')).includes('Correct.'));
const qw = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('do lanes disagree at the in-tile test')));
await (await qw.$('input[type=number]')).type('32');
await (await qw.$('button.primary')).click();
await pause(150);
check('q-warps accepts 32', (await qw.evaluate((e) => e.textContent)).includes('Correct.'));
const qm = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('counting the tile loads')));
await (await qm.$('input[type=number]')).type('2.84');
await (await qm.$('button.primary')).click();
await pause(150);
check('q-miss accepts 2.84', (await qm.evaluate((e) => e.textContent)).includes('Correct.'));

// ---------------------------------------------------------------- the measured box
const reports = await (await fetch(BASE + '/api/reports')).json();
const runs = Object.entries(reports).filter(([b, r]) => !['emu', 'race'].includes(b) && !/emulator/i.test(String(r.device?.name ?? '')));
const rows = runs.flatMap(([, r]) => r.results.filter((x) => x.lab === 'ch07/lab2_tiledconv' && x.variant === 'solution').flatMap((x) => x.bench));
const cached = rows.filter((x) => x.name === 'tiled_cached');
if (cached.length) {
  const c = cached[0];
  const hh = rows.find((x) => x.name === 'tiled_halo' && x.tile === c.tile && x.radius === c.radius);
  const cell = await attr(`[data-tile="${c.tile}"][data-radius="${c.radius}"]`, 'data-ratio');
  check('measured ratio cell is this machine\'s tiled_cached over tiled_halo', hh && cell === (c.gflops / hh.gflops).toFixed(3), `${cell}`);
} else check('with no lab 7.2 measurement, the page says what to run, then reload', !!(await page.$('[data-measured="none"]')) &&
  (await page.$eval('[data-measured="none"]', (e) => e.textContent)).includes('reload this page'));
const vendor = await page.$$eval('table[data-table="vendor-caches"] tbody tr', (trs) => trs.map((t) => t.children[0].textContent.trim()));
check('the per-vendor cache facts are a table in the text, not a tall margin note', JSON.stringify(vendor) === JSON.stringify(['NVIDIA', 'AMD', 'Apple']), JSON.stringify(vendor));
check('every margin note follows a paragraph or list, never a bench, table or code block',
  await page.$$eval('.prose > aside.note', (as) => as.every((a) => { const p = a.previousElementSibling; return p && !p.matches('section.bench, figure.code, .wide, table'); })));

// ---------------------------------------------------------------- themes and a narrow screen
if (out) {
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
  await shot(bench(T2), 'cache-dark');
  await shot(SC, 'scene4-dark');
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
}
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 });
await pause(400);
const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at phone width the page does not scroll sideways', sideways <= 0, String(sideways));
if (out) {
  await shot(bench(T1), 'taps-narrow');
  await shot(SC, 'scene-narrow');
}

const expected404 = (u) => /\/content\/ch07\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404),
  [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch07/7.6', [
  { label: 'CUDA', lab: 'ch07/lab2_tiledconv', file: 'kernel.cu' },
  { label: 'Metal kernel', lab: 'ch07/lab2_tiledconv', file: 'kernel.metal' },
], { reveal: ['Show the repaired kernel now'] });

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
