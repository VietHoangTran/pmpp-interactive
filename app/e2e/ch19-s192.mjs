// Browser check for section 19.2: the loop classification, the prediction and the
// "From blockIdx to pixel" explorer (forward and reverse, the three host codes, the planted
// swap, the grid limits), the corrected kernel from lab 19.1, the narrated animation "What
// the direct kernel reads", the warp-reads explorer for Exercise 2, and the questions.
// Expected values come from src/sim/cnnlayer.ts and src/sim/convgrid.ts. It passes with or
// without content/ch19/narration/manifest.json: with no audio the player runs on estimated
// timings and shows each script regardless.
// Usage: node e2e/ch19-s192.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { BOOK_F, BOOK_LAYER, BOOK_X, directTrace, readsOfThread } from '../src/sim/cnnlayer.ts';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const bench = (t) => `section.bench[aria-label="${t}"]`;
const LOOPS = bench('Which loops can be split across threads?');
const GM = bench('From blockIdx to pixel');
const WR = bench("One warp's input reads");
const SC = 'section.scene[aria-label="What the direct kernel reads"]';

/** Run fn(model, ...args) inside the page against src/sim/convgrid.ts. */
const grid = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/convgrid.ts');
    return (0, eval)(src)(m, ...args);
  }, fn.toString(), args);

async function press(scope, label, exact = true) {
  const ok = await page.evaluate((scope, label, exact) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => (exact ? x.textContent.trim() === label : x.textContent.trim().includes(label)));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label, exact);
  await pause(100);
  return ok;
}
async function setValue(sel, v) {
  await page.$eval(sel, (el, v) => {
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  }, v);
  await pause(120);
}
const data = (sel) => page.$eval(sel, (e) => ({ ...e.dataset, text: e.textContent.replace(/\s+/g, ' ').trim() }));
const preset = (k) => setValue(GM + ' select[aria-label="Preset"]', k);
const host = (h) => setValue(GM + ' select[aria-label="Host code"]', h);
const swapBox = (on) => page.evaluate((gm, on) => {
  const i = [...document.querySelectorAll(gm + ' label.check')].find((l) => l.textContent.includes('planted bug')).querySelector('input');
  if (i.checked !== on) i.click();
}, GM, on).then(() => pause(150));
const clickRect = async (sel) => {
  await page.$eval(sel, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(120);
};
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.replace(/\s+/g, ' ').includes(t)), text);
async function answerChoice(text, optionStart) {
  const q = await question(text);
  const found = await q.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.replace(/\s+/g, ' ').trim().startsWith(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return { found, text: norm(await q.evaluate((e) => e.textContent)) };
}
async function answerNumber(text, value) {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return norm(await q.evaluate((e) => e.textContent));
}

await goto(page, 'ch19/19.2');
await page.waitForSelector(GM);

// ---- which loops can be split
check('seven loop cards, outermost first', (await page.$$eval(LOOPS + ' [data-loop]', (c) => c.map((x) => x.dataset.loop).join(''))) === 'nmhwcpq');
check('Check waits until every loop is marked', !(await press(LOOPS, 'Check')));
const mark = (v, role) => page.evaluate((s, v, role) => [...document.querySelector(`${s} [data-loop="${v}"]`).querySelectorAll('button')].find((b) => b.textContent.startsWith(role)).click(), LOOPS, v, role);
for (const v of 'nmhw') await mark(v, 'independent');
await mark('c', 'independent');
for (const v of 'pq') await mark(v, 'accumulates');
await pause(80);
await press(LOOPS, 'Check');
let d = await data(LOOPS + ' [data-out="loops"]');
check('marking c independent: 1 of 7 wrong, and c is flagged', d.text === '1 of 7 wrong: c' && (await page.$eval(LOOPS + ' [data-loop="c"]', (e) => e.dataset.result)) === 'wrong', d.text);
await mark('c', 'accumulates');
await pause(80);
await press(LOOPS, 'Check');
check('all seven right after fixing c', (await data(LOOPS + ' [data-out="loops"]')).text === 'all 7 right');
check('the feedback names the read-modify-write on Y[n, m, h, w]', (await data(LOOPS + ' [data-out="loops-why"]')).text.includes('Y[n, m, h, w]'));
let r = await answerChoice('can be split across threads with no atomics', 'n, m, h and w');
check('q-loops: n, m, h and w is correct', r.found && r.text.includes('Correct.'));

// ---- the prediction, then the explorer
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many of the 784 pixels')));
await (await pq.$('input[type=number]')).type('256');
await (await pq.$('button.primary')).click();
await pause(120);
const pv = norm(await pq.evaluate((e) => e.textContent));
check('p-cover: committing 256 reveals 256 and the 528-pixel band', pv.includes('256.') && pv.includes('528') && pv.includes('Exactly.'), pv.slice(0, 200));

// default: Fig. 19.6
d = await data(GM + ' [data-out="geometry"]');
check('Fig. 19.6 by default: gridDim(4, 4, 1)', d.gx === '4' && d.gy === '4' && d.gz === '1', JSON.stringify(d));
d = await data(GM + ' [data-out="coverage"]');
check('every pixel of the four 32 × 32 maps computed once', d.once === '4096' && d.never === '0', JSON.stringify(d));
check('the grid draws 4 × 4 blocks', (await page.$$(GM + ' [data-grid="blocks"] rect')).length === 16);
// LeNet C1 as printed
await preset(1);
const c1 = await grid((m) => m.coverage({ N: 1, M: 6, Hout: 28, Wout: 28, tile: 16, division: 'floor' }, { swap: false, guard: false }));
d = await data(GM + ' [data-out="coverage"]');
check(`LeNet C1 as printed: ${c1.once} computed, ${c1.never} never (256 and 528 per map)`, Number(d.once) === c1.once && Number(d.never) === c1.never && c1.once === 6 * 256, JSON.stringify(d));
check('as printed W_grid = 1', (await data(GM + ' [data-out="geometry"]')).gy === '1');
if (out) await page.$eval(GM, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-floor.png' });
await host('ceil-raw');
const raw = await grid((m) => m.coverage({ N: 1, M: 6, Hout: 28, Wout: 28, tile: 16, division: 'ceil' }, { swap: false, guard: false }));
d = await data(GM + ' [data-out="coverage"]');
check(`ceiling, no check: ${raw.stray} strays, ${raw.multi} pixels stored twice, ${raw.pastEnd} stores past the end`, Number(d.stray) === raw.stray && Number(d.multi) === raw.multi && Number(d.pastend) === raw.pastEnd && raw.multi > 0 && raw.pastEnd > 0, JSON.stringify(d));
await host('ceil-guard');
d = await data(GM + ' [data-out="coverage"]');
check('ceiling with a bounds check: all 784 per map once, 240 idle per map', d.once === String(6 * 784) && d.never === '0' && d.idle === String(6 * 240), JSON.stringify(d));
check('blocks on the right and bottom edges are marked as having threads outside the map', (await page.$$eval(GM + ' [data-grid="blocks"] rect', (rs) => rs.filter((x) => x.getAttribute('fill') === 'var(--rule)').length)) === 6 * 3);

// the question's thread
await preset(3);
await setValue(GM + ' input[aria-label="sample (blockIdx.z)"]', 1);
await clickRect(GM + ' [data-block="3,2"]');
await clickRect(GM + ' [data-thread="5,9"]');
d = await data(GM + ' [data-out="map"]');
const qy = await grid((m) => { const p = { N: 2, M: 4, Hout: 32, Wout: 32, tile: 16, division: 'ceil' }; return m.mapThread(p, m.geometry(p), { bx: 3, by: 2, bz: 1, tx: 5, ty: 9 }).yIndex; });
check('block (3, 2, 1), thread (5, 9) computes (1, 3, 25, 5)', d.n === '1' && d.m === '3' && d.h === '25' && d.w === '5' && Number(d.y) === qy, JSON.stringify(d));
check('the readout shows the division and remainder', d.text.includes('tile row = 2 / 2 = 1') && d.text.includes('tile column = 2 % 2 = 0'), d.text);
check('the block\'s tile is outlined in map 3', (await page.$$(GM + ' svg[data-map="3"] rect[stroke-width="2.5"]')).length === 1);
if (out) await page.$eval(GM, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-thread.png' });
await swapBox(true);
await clickRect(GM + ' [data-block="3,2"]');
await clickRect(GM + ' [data-thread="5,9"]');
d = await data(GM + ' [data-out="map"]');
check('with / and % swapped the same thread computes (1, 3, 9, 21)', d.h === '9' && d.w === '21', JSON.stringify(d));
d = await data(GM + ' [data-out="coverage"]');
check('on the square 2 × 2 tile grid the swap still computes every pixel once', d.once === String(4 * 1024) && d.never === '0' && d.multi === '0', JSON.stringify(d));
// rectangular
await preset(2);
d = await data(GM + ' [data-out="coverage"]');
const rect = await grid((m) => m.coverage({ N: 1, M: 2, Hout: 32, Wout: 64, tile: 16, division: 'ceil' }, { swap: true, guard: true }));
check(`32 × 64 with the swap: ${rect.never} pixels never computed, ${rect.idle} threads idle`, Number(d.never) === rect.never && Number(d.idle) === rect.idle && rect.never === 2 * 32 * 32, JSON.stringify(d));
await clickRect(GM + ' [data-block="0,2"]');
check('block (0, 2) is sent to row 32, outside the map', (await data(GM + ' [data-out="tile-outside"]')).text.includes('row 32'));
if (out) await page.$eval(GM, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-swap.png' });
await swapBox(false);

// reverse mode
await press(GM, 'reverse: pick a pixel');
const box = await page.$eval(GM + ' svg[data-map="0"]', (e) => { const r = e.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
await page.mouse.click(box.x + (40.5 / 64) * box.w, box.y + (20.5 / 32) * box.h);
await pause(120);
check('clicking a pixel asks for its thread', (await data(GM + ' [data-out="reverse"]')).text.includes('Y[0, 0, 20, 40]'));
const own = await grid((m) => { const p = { N: 1, M: 2, Hout: 32, Wout: 64, tile: 16, division: 'ceil' }; return m.ownerOf(p, m.geometry(p), 0, 0, 20, 40); });
const guess = async (g) => {
  for (const [k, v] of Object.entries(g)) await setValue(`${GM} input[aria-label="${k}"]`, v);
  await press(GM, 'Reveal');
  return data(GM + ' [data-out="reveal"]');
};
d = await guess({ bx: own.bx, by: own.by + 1, bz: own.bz, tx: own.tx, ty: own.ty });
check('a wrong prediction is marked and the right thread named', d.right === 'no' && d.writers === '1' && d.text.includes(`blockIdx (${own.bx}, ${own.by}, ${own.bz}), threadIdx (${own.tx}, ${own.ty})`), d.text);
d = await guess(own);
check(`the model's owner, blockIdx (0, 6, 0), threadIdx (8, 4), is right`, d.right === 'yes' && own.by === 6 && own.tx === 8 && own.ty === 4, JSON.stringify(d));
await press(GM, 'pick a thread');

// limits
await preset(4);
d = await data(GM + ' [data-out="limits"]');
check('one 4096 × 4096 map needs 65,536 tiles: over CUDA\'s and HIP\'s y limit', d.cuda === 'over' && d.hip === 'over' && (await data(GM + ' [data-out="geometry"]')).gy === '65536', JSON.stringify(d));
check('too big to draw or simulate, and it says so', !!(await page.$(GM + ' [data-out="grid-too-big"]')) && !!(await page.$(GM + ' [data-out="coverage-too-big"]')));
await setValue(GM + ' input[aria-label="H_out"]', 256);
await setValue(GM + ' input[aria-label="W_out"]', 256);
await setValue(GM + ' input[aria-label="M (output maps)"]', 256);
await setValue(GM + ' input[aria-label="N (samples)"]', 256);
d = await data(GM + ' [data-out="limits"]');
const thr = await grid((m) => m.geometry({ N: 256, M: 256, Hout: 256, Wout: 256, tile: 16, division: 'ceil' }).threads);
check(`N = M = 256 at 256 × 256: ${thr} threads, within CUDA, over HIP's 2^32 − 1`, d.cuda === 'ok' && d.hip === 'over' && thr === 2 ** 32, JSON.stringify(d));

// ---- the corrected kernel
await page.waitForFunction(() => [...document.querySelectorAll('.code .tabs button')].some((b) => b.textContent === 'Metal host'), { timeout: 8000 }).catch(() => undefined);
const fig = () => page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => [...x.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host'));
  return f ? { code: f.querySelector('pre').textContent, status: f.querySelector('.tabs .status')?.textContent ?? '', tabs: [...f.querySelectorAll('.tabs button')].map((b) => b.textContent) } : null;
});
let k = await fig();
check('the listing has CUDA, Metal kernel and Metal host tabs', !!k && JSON.stringify(k.tabs) === JSON.stringify(['CUDA', 'Metal kernel', 'Metal host']), JSON.stringify(k?.tabs));
check('the CUDA tab is the direct region only: kernel and launch, no unroll, no markers',
  !!k && k.code.includes('convDirectKernel') && k.code.includes('launchConvDirect') && k.code.includes('if (h < H_out && w < W_out)') && !k.code.includes('[lesson:') && !/unroll/i.test(k.code.replace(/\/\/.*$/gm, '')), k?.code.slice(0, 200));
check('its status uses the fixed vocabulary', !!k && /emulator-checked|unverified/.test(k.status) && !/verified on CUDA/.test(k.status.replace('unverified on CUDA', '')), k?.status);
await page.evaluate(() => [...[...document.querySelectorAll('figure.code')].find((x) => [...x.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host')).querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal kernel').click());
await pause(100);
k = await fig();
check('the Metal kernel tab takes uint3 positions', k.code.includes('kernel void conv_direct') && k.code.includes('[[threadgroup_position_in_grid]]') && !k.code.includes('[lesson:'));
await page.evaluate(() => [...[...document.querySelectorAll('figure.code')].find((x) => [...x.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host')).querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal host').click());
await pause(100);
k = await fig();
check('the Metal host tab dispatches whole threadgroups', k.code.includes('dispatchThreadgroups') && k.code.includes('encodeConvDirect'));
r = await answerChoice('computes which (n, m, h, w)', '(1, 3, 25, 5)');
check('q-mapping: (1, 3, 25, 5) is correct', r.found && r.text.includes('Correct.'));
r = await answerChoice('tests on 64 × 64 outputs all pass', 'With a square grid of tiles');
check('q-swap: the square tile grid is the reason', r.found && r.text.includes('Correct.') && r.text.includes('Tile columns 2 and 3'));
let n = await answerNumber("C3 (C = 6, K = 5)", 0.25);
check('q-ratio: 0.25 FLOP/B is correct', n.includes('Correct.'));

// ---- the narrated animation
const tr = directTrace(BOOK_LAYER, BOOK_X, BOOK_F, 2);
const scripts = [1, 2, 3, 4, 5, 6].map((i) => fs.readFileSync(path.join(repo, `content/ch19/narration/direct-reads-${i}.txt`), 'utf8').trim());
const manifest = await fetch(BASE + '/content/ch19/narration/manifest.json').then((x) => (x.ok ? x.json() : null)).catch(() => null);
check('six scripts of 35 to 60 words each', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }));
check('scripts are written for the ear: no symbols read aloud', scripts.every((x) => !/[<>=×%/_(){}[\]*+]/.test(x)));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const labels = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered', JSON.stringify(labels) === JSON.stringify(['1. Two blocks', '2. One thread', '3. Four threads', '4. One weight', '5. Block 1', '6. Totals']), labels.join(' | '));
const capBox = () => page.evaluateHandle((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('there is a captions switch, off by default', !(await (await capBox()).evaluate((e) => e.checked)));
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
if (!manifest?.clips?.['direct-reads-1']) {
  check('a part with no audio shows its script even with captions off', (await caption()) === scripts[0], await caption());
} else {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
  check('with audio and captions off, no caption is shown', (await caption()) === null);
}
await (await capBox()).click();
await pause(120);
check('switching captions on shows part 1\'s script', (await caption()) === scripts[0]);
const svgData = () => page.$eval(SC + ' .stage svg', (e) => ({ ...e.dataset, text: [...e.querySelectorAll('text')].map((t) => t.textContent).join(' | ') }));
const go = async (i, at) => {
  await page.evaluate((SC, i) => [...document.querySelectorAll(SC + ' .seg button')][i - 1].click(), SC, i);
  await pause(120);
  await setValue(SC + ' input[type=range]', at);
};
await go(1, 1000);
let sv = await svgData();
check('part 1 draws the grid of 2 blocks, gridDim (2, 1, 1)', sv.blocks === String(tr.blocks.length) && sv.text.includes('gridDim (2, 1, 1)'), JSON.stringify(sv).slice(0, 200));
await go(2, 500);
sv = await svgData();
const t0 = readsOfThread(tr.blocks[0], 0);
check('halfway through part 2 the register holds the model\'s running sum', Number(sv.step) > 0 && Number(sv.step) < 12 && Number(sv.sum) === t0[Number(sv.step) - 1].sum, JSON.stringify(sv).slice(0, 200));
await go(2, 1000);
sv = await svgData();
check('part 2 ends after 12 steps with acc = 14', sv.step === '12' && sv.sum === '14' && sv.xreads === '12' && sv.text.includes('acc 14'), JSON.stringify(sv).slice(0, 200));
check('the caption follows the part', (await caption()) === scripts[1]);
await go(3, 1000);
sv = await svgData();
const heat = await page.$$eval(SC + ' .stage svg g[data-x]', (gs) => gs.slice(0, 9).map((g) => Number(g.dataset.count)));
check(`part 3 ends at ${tr.blocks[0].xReads} reads of ${tr.blocks[0].distinctX} values, centre 4, edges 2, corners 1`, sv.xreads === String(tr.blocks[0].xReads) && sv.distinct === '27' && JSON.stringify(heat) === JSON.stringify(tr.blocks[0].xHeat.slice(0, 9)), JSON.stringify({ sv: sv.xreads, heat }));
await go(4, 400);
sv = await svgData();
check('part 4: four threads, one filter address per step', sv.faddresses === '1' && sv.fthreads === '4', JSON.stringify(sv).slice(0, 200));
await go(5, 1000);
sv = await svgData();
check(`part 5 ends at ${tr.xReads} input reads`, sv.xreads === String(tr.xReads) && sv.distinct === '27');
const heat2 = await page.$$eval(SC + ' .stage svg g[data-x]', (gs) => gs.slice(0, 9).map((g) => Number(g.dataset.count)));
check('the centre of each map has now been fetched 8 times', JSON.stringify(heat2) === JSON.stringify(tr.xHeat.slice(0, 9)), JSON.stringify(heat2));
await go(6, 1000);
sv = await svgData();
check('part 6: 192 FLOP for 768 bytes, 0.25 FLOP per byte', sv.ratio === '0.25' && sv.text.includes('192 FLOP / 768 bytes = 0.25 FLOP per byte'), sv.text.slice(-200));
if (out) await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-scene6.png' });
await go(3, 500);
if (out) await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-scene3.png' });
await go(1, 0);
await page.evaluate((SC) => { const s = [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'); if (s.checked) s.click(); }, SC);
await press(SC, 'Play', false);
await pause(1500);
const pos = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('Play runs the animation on its clock', pos > 30, String(pos));
await press(SC, 'Pause');
r = await answerChoice('keep one input tile in shared memory', 'Each block computes one output map');
check('q-reuse: one map per block, shared memory per block', r.found && r.text.includes('Correct.'));

// ---- one warp's reads
const wr = (o) => page.evaluate(async (o) => {
  const m = await import('/src/sim/convgrid.ts');
  const c = await import('/src/sim/coalesce.ts');
  const r = m.warpReads({ layout: 'NCHW', dims: { N: 1, C: 4, H: 18, W: 18 }, K: 3, tile: 16, warpWidth: 32, warp: 0, c: 0, p: 0, q: 0, lanes: 'w', segmentBytes: c.SEGMENT_MODELS.nvidia.segmentBytes, ...o, dims: { N: 1, C: 4, H: 18, W: 18, ...(o.dims ?? {}) } });
  return { segments: r.access.transactions, lanes: r.lanes.length, stride: r.stride };
}, o);
const warp = () => data(WR + ' [data-out="warp"]');
let w = await warp();
let e = await wr({});
check(`NCHW, C = 4: ${e.segments} sectors, lanes 4 bytes apart`, Number(w.segments) === e.segments && e.segments === 5 && w.stride === '4', JSON.stringify(w));
check('the strip draws one box per sector', (await page.$$(WR + ' [data-out="strip"] [data-seg]')).length === e.segments);
await press(WR, 'NHWC');
w = await warp();
e = await wr({ layout: 'NHWC' });
check(`NHWC, C = 4: ${e.segments} sectors, lanes 16 bytes apart`, Number(w.segments) === e.segments && w.stride === '16', JSON.stringify(w));
await setValue(WR + ' input[aria-label="C"]', 8);
w = await warp();
check('NHWC, C = 8: one sector per lane, 32', w.segments === '32' && Number(w.efficiency) === 0.125, JSON.stringify(w));
if (out) await page.$eval(WR, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-nhwc.png' });
await press(WR, 'CHWN');
await setValue(WR + ' input[aria-label="N"]', 8);
w = await warp();
e = await wr({ layout: 'CHWN', dims: { N: 8, C: 8 } });
check(`CHWN, N = 8, lanes on w: ${e.segments} sectors`, Number(w.segments) === e.segments && e.segments === 32, JSON.stringify(w));
await press(WR, 'different samples', false);
await setValue(WR + ' input[aria-label="N"]', 32);
w = await warp();
e = await wr({ layout: 'CHWN', dims: { N: 32, C: 8 }, lanes: 'n' });
check('CHWN with lanes on samples (N = 32): 4 sectors, fully used', Number(w.segments) === e.segments && e.segments === 4 && Number(w.efficiency) === 1, JSON.stringify(w));
await press(WR, 'NCHW');
w = await warp();
check('NCHW with lanes on samples: 32 sectors', w.segments === '32');
await press(WR, 'neighbouring w', false);
await setValue(WR + ' select[aria-label="Warp width"]', 64);
w = await warp();
check('a 64-wide wavefront has 64 lanes in 4 runs', w.lanes === '64' && w.runs === '4', JSON.stringify(w));
check('F is one address for the whole warp', (await data(WR + ' [data-out="f"]')).text.includes('One address for the whole warp'));
r = await answerChoice('which layout of X lets one warp', 'NCHW');
check('q-layout: NCHW is correct', r.found && r.text.includes('Correct.'));

// ---- the captions choice survives a reload; dark theme; narrow width
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(SC);
await pause(500);
check('the captions choice is remembered after a reload', await (await capBox()).evaluate((e) => e.checked));
await (await capBox()).click();
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await pause(200);
if (out) await page.$eval(GM, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-dark.png' });
if (out) await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-scene-dark.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', over <= 1, `overflow ${over}px`);
if (out) await page.$eval(LOOPS, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow-loops.png' });
if (out) await page.$eval(GM, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow-grid.png' });
if (out) await page.$eval(WR, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow-warp.png' });

const expected404 = (u) => /\/content\/ch19\/narration\/(manifest\.json|direct-reads-\d\.mp3)$/.test(u);
const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
