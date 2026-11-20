// Browser check for section 15.5: the prediction, the store-pattern visualiser (classify,
// then counts for every pattern, width, N and segment size), the lane picker (selects,
// hover, click, both mappings and widths), the narrated scene "From a block tile to one
// lane's four sub-tiles" (every beat's drawing against the model, captions option, script
// shown with no audio) and two questions. Expected values come from src/sim/gemmremap.ts,
// imported inside the page through the app's own module server.
// Usage: node e2e/ch15-s155.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
/** Run fn(model, ...args) inside the page against src/sim/gemmremap.ts. */
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/gemmremap.ts');
    return (0, eval)(src)(m, ...args);
  }, fn.toString(), args);
const pick = async (scope, label, value) => {
  const ok = await page.evaluate((scope, label, value) => {
    const root = document.querySelector(scope);
    // The label's own text (before its control) must match exactly: "Warp" is not "Warp width".
    const f = [...root.querySelectorAll('label.field')].find((l) => l.childNodes[0]?.textContent.trim() === label);
    const s = f?.querySelector('select');
    if (!s) return false;
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, scope, label, value);
  await pause(80);
  return ok;
};
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};

await goto(page, 'ch15/15.5');
const V = bench('One warp, one store instruction');
const L = bench('Lane picker');
const SC = 'section.scene[aria-label="From a block tile to one lane\'s four sub-tiles"]';
await page.waitForSelector(V);

// ---- prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('scalar store instruction')));
await (await pq.$('input[type=number]')).type('32');
await (await pq.$('button.primary')).click();
await pause(150);
const scalarSectors = await model((m) => m.storeInstruction({ ...m.DEFAULT_STORE, pattern: 'scalar8x8' }).transactions);
const pText = await pq.evaluate((e) => e.textContent);
check('prediction reveal states the model’s sector count', scalarSectors === 32 && pText.includes(`It touches ${scalarSectors}, as you said`), pText);

// ---- the store visualiser
const svgData = () => page.$eval(V + ' svg', (s) => ({ t: Number(s.dataset.transactions), e: Number(s.dataset.efficiency), rows: Number(s.dataset.rows), vector: s.dataset.vector }));
const expectStore = (spec) => model((m, spec) => {
  const r = m.storeInstruction(spec);
  return { t: r.transactions, e: r.efficiency, rows: r.rows.length, vector: r.vector ? 'yes' : 'no' };
}, spec);
const base = { W: 32, N: 1024, warp: 0, row: 0, col: 0, segmentBytes: 32 };
const SHORT = { scalar8x8: '8 × 8, scalar (§15.3)', 'float4-8x8': '8 × 8, float4 (Exercise 2)', remap4x4: 'four 4 × 4, float4 (§15.5)' };
const truth = await model((m) => Object.fromEntries(['scalar8x8', 'float4-8x8', 'remap4x4'].map((p) => [p, m.coalesced(p)])));
check('model classifies the three patterns as not, not, coalesced', JSON.stringify(truth) === JSON.stringify({ scalar8x8: false, 'float4-8x8': false, remap4x4: true }), JSON.stringify(truth));

check('counts are hidden until the pattern is classified', !(await page.$(V + ' [data-readout]')) && !!(await page.$(V + ' [data-classify="scalar8x8"]')));
const pctText = (e) => `${Math.round(e * 1000) / 10} %`;
for (const p of ['scalar8x8', 'float4-8x8', 'remap4x4']) {
  await press(V, SHORT[p]);
  const want = await expectStore({ ...base, pattern: p });
  const got = await svgData();
  check(`${p}: picture matches the model (${want.t} sectors, ${pctText(want.e)}, ${want.rows} rows)`, JSON.stringify(got) === JSON.stringify(want), JSON.stringify([got, want]));
  // Answer the opposite of the truth for the second pattern, to see the correction.
  const wrongOnPurpose = p === 'float4-8x8';
  const say = truth[p] !== wrongOnPurpose ? 'Coalesced' : 'Not coalesced';
  await press(V, say);
  const t = await text(V + ' [data-counts]');
  check(`${p}: classification verdict is ${wrongOnPurpose ? 'corrected' : 'confirmed'}`, t.includes(wrongOnPurpose ? 'Not quite.' : 'Right.'), t);
  check(`${p}: readout shows ${want.t} sectors at ${pctText(want.e)}`, t.includes(`${want.t} sectors touched`) && t.includes(pctText(want.e)), t);
}
check('the three default counts are 32 @ 12.5 %, 32 @ 50 %, 16 @ 100 %',
  JSON.stringify(await Promise.all(['scalar8x8', 'float4-8x8', 'remap4x4'].map((p) => expectStore({ ...base, pattern: p }).then((r) => [r.t, r.e]))))
    === JSON.stringify([[32, 0.125], [32, 0.5], [16, 1]]));
const tot = await model((m) => m.warpTileTotal('remap4x4', 32, 1024, 0, 32));
check('whole-tile line for the remap matches the model', (await text(V + ' [data-total]')).includes(`${tot.transactions} sectors in all`) && tot.transactions === tot.minimum);
await page.$eval(V, (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-visualiser-remap.png' });

// Width 64.
await pick(V, 'Warp width', 64);
let want = await expectStore({ ...base, W: 64, pattern: 'remap4x4' });
let got = await svgData();
check('width 64: the remap is 32 sectors at 100 % on 8 rows', JSON.stringify(got) === JSON.stringify(want) && want.t === 32 && want.e === 1 && want.rows === 8, JSON.stringify([got, want]));
const stripLen = await page.$$eval(V + ' g[data-strip]', (gs) => gs.map((g) => g.querySelectorAll('rect[stroke]').length));
check('width 64: every strip is 4 sectors (128 contiguous bytes)', stripLen.length === 8 && stripLen.every((n) => n === 4), JSON.stringify(stripLen));
await pick(V, 'Warp width', 32);

// N = 1020, an odd tile row: 64-byte runs straddle sectors.
await pick(V, 'N, the width of C', 1020);
await pick(V, 'Instruction: tile row', 1);
want = await expectStore({ ...base, N: 1020, row: 1, pattern: 'remap4x4' });
got = await svgData();
check('N = 1020, tile row 1: 24 sectors at 66.7 %', JSON.stringify(got) === JSON.stringify(want) && want.t === 24, JSON.stringify([got, want]));
// N = 1001: scalar fallback.
await pick(V, 'N, the width of C', 1001);
got = await svgData();
want = await expectStore({ ...base, N: 1001, row: 1, pattern: 'remap4x4' });
check('N = 1001: float4 ruled out, scalar stores', got.vector === 'no' && JSON.stringify(got) === JSON.stringify(want), JSON.stringify([got, want]));
check('N = 1001: the fallback is explained', (await text(V)).includes('The kernel takes its scalar path'));
const colOpts = await page.$$eval(V + ' label.field', (ls) => ls.find((l) => l.textContent.trim().startsWith('tile columns')).querySelectorAll('option').length);
check('N = 1001: eight scalar column choices', colOpts === 8, String(colOpts));
await pick(V, 'N, the width of C', 1024);
await pick(V, 'Instruction: tile row', 0);
// 128-byte lines.
await pick(V, 'Segment', 128);
want = await expectStore({ ...base, segmentBytes: 128, pattern: 'remap4x4' });
got = await svgData();
check('128-byte lines: the 32-wide remap fills half of each line', got.e === 0.5 && JSON.stringify(got) === JSON.stringify(want), JSON.stringify([got, want]));
check('readout names lines, not sectors, for AMD segments', (await text(V + ' [data-readout]')).includes('lines touched'));
await pick(V, 'Segment', 32);

// ---- the lane picker
await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' }));
await pause(150);
const subtiles = () => page.$$eval(L + ' svg rect[data-subtile]', (rs) => rs.map((r) => r.dataset.subtile).sort());
const expSub = (t, W) => model((m, t, W) => m.remapThread(t, W).subTiles.map((s) => `${s.row},${s.col}`).sort(), t, W);
check('default: warp 6, lane 5 shows the model’s four sub-tiles', JSON.stringify(await subtiles()) === JSON.stringify(await expSub(197, 32)), JSON.stringify(await subtiles()));
check('table lists warp 6’s tile from row 64, column 64', (await text(L + ' table')).includes('rows 64–127, columns 64–95'));
await pick(L, 'Lane', 0);
check('warp 6, lane 0 squares at (64,64) (64,80) (96,64) (96,80)', JSON.stringify(await subtiles()) === JSON.stringify(['64,64', '64,80', '96,64', '96,80']), JSON.stringify(await subtiles()));
if (out) await page.screenshot({ path: out + '-lanes.png' });
// Hover an element: the status names its owner.
const hoverAt = async (row, col) => {
  const box = await page.$eval(L + ' svg', (s) => { const b = s.getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, vb: s.viewBox.baseVal.width }; });
  const k = box.w / box.vb;
  await page.mouse.move(box.x + (24 + col * 4 + 2) * k, box.y + (24 + row * 4 + 2) * k);
  await pause(120);
};
await hoverAt(70, 81);
const own = await model((m) => m.ownerOf('remap', 32, 70, 81));
let status = await text(L + ' [role=status]');
check(`hover (70, 81): thread ${own.t}, warp ${own.warp}, lane ${own.lane}`, status.includes(`belongs to thread ${own.t}: warp ${own.warp}, lane ${own.lane}`) && status.includes(`element (${own.r}, ${own.c})`), status);
await page.mouse.down();
await page.mouse.up();
await pause(120);
const sel = await page.$eval(L, (b) => [...b.querySelectorAll('label.field select')].map((s) => s.value));
check('clicking the element picks its owner', sel[1] === String(own.warp) && sel[2] === String(own.lane), JSON.stringify(sel));
check('…and draws the owner’s sub-tiles', JSON.stringify(await subtiles()) === JSON.stringify(await expSub(own.t, 32)));
// Width 64.
await pick(L, 'Warp width', 64);
const warpOpts = await page.$$eval(L + ' label.field', (ls) => ls.find((l) => l.textContent.trim().startsWith('Warp') && !l.textContent.includes('width')).querySelectorAll('option').length);
check('width 64: four wavefronts to choose from', warpOpts === 4, String(warpOpts));
await pick(L, 'Warp', 3);
await pick(L, 'Lane', 9);
check('width 64, wavefront 3, lane 9: model sub-tiles', JSON.stringify(await subtiles()) === JSON.stringify(await expSub(3 * 64 + 9, 64)), JSON.stringify(await subtiles()));
check('width 64: a lane row stores 128 contiguous bytes', (await text(L + ' table')).includes('8 lanes × 16 B = 128 contiguous bytes'));
// The §15.3 mapping.
await press(L, '§15.3: one 8 × 8');
const ct = await model((m) => m.coarseThread(3 * 64 + 9));
check('§15.3 view: the thread’s 8 × 8 patch', JSON.stringify(await subtiles()) === JSON.stringify([`${ct.tRow},${ct.tCol}`]), JSON.stringify(await subtiles()));
await hoverAt(70, 81);
const own2 = await model((m) => m.ownerOf('coarse', 64, 70, 81));
status = await text(L + ' [role=status]');
check(`§15.3 hover (70, 81): thread ${own2.t}`, status.includes(`belongs to thread ${own2.t}`) && own2.t === 8 * 16 + 10, status);
await press(L, '§15.5: four 4 × 4');
await pick(L, 'Warp width', 32);

// ---- the scene
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const manifest = await (await fetch(BASE + '/content/ch15/narration/manifest.json')).json().catch(() => null);
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push((await (await fetch(`${BASE}/content/ch15/narration/remap-${n}.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim());
check('six scripts of 35 to 60 words', scripts.every((s) => { const w = s.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(' ').length).join(','));
check('captions option exists, off by default', (await text(SC)).includes('captions') && !(await page.$eval(SC + ' label.check:last-of-type input', (i) => i.checked).catch(() => true)));
const seek = async (v) => {
  await page.$eval(SC + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const beatBtn = async (k) => {
  await page.evaluate((sel, k) => [...document.querySelectorAll(sel + ' .seg button')][k].click(), SC, k);
  await pause(150);
};
const stage = () => page.evaluate((sel) => {
  const svg = document.querySelector(sel + ' .stage svg');
  const mem = svg.querySelector('g[data-sectors]');
  return {
    beat: Number(svg.dataset.beat),
    headline: svg.querySelector('[data-headline]')?.textContent ?? '',
    sectors: mem ? Number(mem.dataset.sectors) : null,
    useful: mem ? Number(mem.dataset.useful) : null,
    rows: mem ? Number(mem.dataset.rows) : null,
    counter: svg.querySelector('[data-counter]')?.textContent ?? '',
    strips: [...svg.querySelectorAll('g[data-strip-row]')].map((g) => Number(g.dataset.stripSectors)),
    warpTiles: svg.querySelectorAll('[data-warp-tile]').length,
    waveTiles: svg.querySelectorAll('[data-wave-tile]').length,
    squares: [...svg.querySelectorAll('[data-lane-square]')].map((g) => g.dataset.laneSquare),
    written: svg.querySelector('g[data-written]') ? Number(svg.querySelector('g[data-written]').dataset.written) : null,
  };
}, SC);
const sceneSpecs = [
  { pattern: 'scalar8x8', W: 32, warp: 6 },
  { pattern: 'float4-8x8', W: 32, warp: 6 },
  null,
  null,
  { pattern: 'remap4x4', W: 32, warp: 6 },
  { pattern: 'remap4x4', W: 64, warp: 3 },
];
for (let k = 0; k < 6; k++) {
  await beatBtn(k);
  const cap = await page.$(SC + ' .caption');
  const capText = cap ? (await cap.evaluate((e) => e.textContent)).replace(/\s+/g, ' ').trim() : '';
  if (!manifest) check(`part ${k + 1}: with no audio its script is shown`, capText === scripts[k], capText);
  await seek(1000);
  const s = await stage();
  check(`part ${k + 1}: the drawing is beat ${k + 1}`, s.beat === k + 1, JSON.stringify(s.beat));
  const spec = sceneSpecs[k];
  if (spec) {
    const m = await model((m, spec) => {
      const r = m.storeInstruction({ ...spec, N: 1024, row: 0, col: 0, segmentBytes: 32 });
      return { t: r.transactions, e: r.efficiency, rows: r.rows.length, strips: m.rowStrips(r, 32).map((x) => x.seg1 - x.seg0), bytes: [r.bytesNeeded, r.bytesFetched] };
    }, spec);
    check(`part ${k + 1}: memory panel is the model's instruction (${m.t} sectors, ${pctText(m.e)})`,
      s.sectors === m.t && s.useful === m.e && s.rows === m.rows && JSON.stringify(s.strips) === JSON.stringify(m.strips), JSON.stringify([s, m]));
    check(`part ${k + 1}: counter reads ${m.t} sectors`, s.counter.startsWith(`${m.t} sectors`) && s.counter.includes(pctText(m.e)), s.counter);
    check(`part ${k + 1}: every lane's store is drawn`, s.written === spec.W, String(s.written));
  }
  if (k === 2) check('part 3: eight warp tiles, zoomed to warp 6', s.warpTiles === 8 && s.headline.includes('2 × 4 warp tiles of 64 × 32'), JSON.stringify(s));
  if (k === 3) {
    const sq = await model((m) => Array.from({ length: 32 }, (_, l) => m.remapThread(192 + l, 32).subTiles.map((x) => `${l}:${x.row},${x.col}`)).flat().sort());
    check('part 4: all 32 lanes’ squares in all four quadrants, from the model', JSON.stringify([...s.squares].sort()) === JSON.stringify(sq), `${s.squares.length} squares`);
    check('part 4: headline gives quadrants 32 × 16 and lanes 8 × 4', s.headline.includes('32 × 16') && s.headline.includes('8 × 4'), s.headline);
  }
  if (k === 5) check('part 6: four wavefront tiles; lane rows of 8 write 128 bytes', s.waveTiles === 4 && s.headline.includes('lane rows of 8 write 128 bytes'), JSON.stringify(s.headline));
  if (out && (k === 0 || k === 3 || k === 4 || k === 5)) {
    await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: `${out}-scene-${k + 1}.png` });
  }
}

// ---- questions
const askQ = async (needle, act) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await act(q);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
let r = await askQ('After the remap, how many 32-byte sectors', async (q) => (await q.$('input[type=number]')).type('16'));
check('numeric question: 16 sectors is correct', r.includes('Correct.'), r.slice(0, 120));
r = await askQ('You do Exercise 2', async (q) => (await q.$$('input[type=radio]'))[2].click());
check('choice question: 32 sectors, half written, is correct', r.includes('Correct.'), r.slice(0, 120));
r = await askQ('compiled for a width of 32', async (q) => (await q.$$('input[type=radio]'))[0].click());
check('choice question: a wrong answer is marked and explained', r.includes('Not quite.') && r.includes('exactly one owner'), r.slice(0, 160));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
