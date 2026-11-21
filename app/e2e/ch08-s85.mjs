// Browser check for section 8.5 "Marching in z, then registers": the narrated scene,
// the z-march stepper, the outcome table, the slab explorer, the erratum hunt, the two
// lab listings and the questions. Expected values come from the models
// (src/sim/zmarch.ts, src/sim/stencilslab.ts), imported here directly.
// It passes with or without content/ch08/narration/manifest.json.
// Usage: node e2e/ch08-s85.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { blocksSlab, cubeAI, planesLoaded, sharedBytes, slabAI, slabAILimit, threadsCube, threadsSlab } from '../src/sim/stencilslab.ts';
import { CORRECT, interiorBlock, localZ, makeProblem, planeTimeline, runAll, runBlock } from '../src/sim/zmarch.ts';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(40);
  return done;
};
const select = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const s = [...document.querySelectorAll(scope + ' label.field')].find((l) => l.textContent.startsWith(label)).querySelector('select');
    s.value = value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(150);
};
const tick = async (scope, label, on) => {
  await page.evaluate((scope, label, on) => {
    const box = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, label, on);
  await pause(150);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();

await goto(page, 'ch08/8.5');
await page.waitForSelector(bench('Step one block through z'));

// ================================================================ the scene
const SC = 'section.scene[aria-label="Marching through z"]';
const res = await fetch(BASE + '/content/ch08/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch08/narration/z-march-${n}.txt`)).text()));
const PROB10 = makeProblem(10);
const TL = planeTimeline({ N: 10, inTile: 6, kernel: 'planes', ...CORRECT }, PROB10);
const TLR = planeTimeline({ N: 10, inTile: 6, kernel: 'registers', ...CORRECT }, PROB10);
const L = (z) => localZ(TL.iStart, z);
const lastF = TL.frames[TL.frames.length - 1];

await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const parts = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('scene: six parts', parts.length === 6, parts.join(' | '));
check('scene: six scripts, 35 to 60 words each', scripts.every((s) => { const w = s.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(' ').length).join(' '));
const capBox = () => page.evaluateHandle((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('scene: a captions switch, off by default', !(await (await capBox()).evaluate((e) => e.checked)));
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
const audible = !!manifest?.clips?.['z-march-1'];
if (audible) check('scene: with audio and captions off, no caption', (await caption()) === null);
else check('scene: a part with no audio shows its script regardless', (await caption()) === scripts[0], String(await caption()));
await (await capBox()).click();
await pause(150);
check('scene: captions on shows part 1\'s script', (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(SC);
await pause(500);
check('scene: the captions choice is remembered after a reload', await (await capBox()).evaluate((e) => e.checked));

const goPart = async (k, pos) => {
  await page.evaluate((S, k) => [...document.querySelectorAll(S + ' .seg button')][k - 1].click(), SC, k);
  await pause(120);
  await setRange(SC + ' input[type=range]', Math.round(pos * 1000));
};
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const vis = (el) => { for (let e = el; e && e !== svg; e = e.parentElement) if (Number(e.getAttribute('opacity') ?? 1) < 0.3) return false; return true; };
  return {
    text: [...svg.querySelectorAll('text')].filter(vis).map((t) => t.textContent.trim()).join(' | '),
    counters: Object.fromEntries([...svg.querySelectorAll('g[data-counter]')].map((g) => [g.dataset.counter, g.dataset.value])),
    boxes: Object.fromEntries([...svg.querySelectorAll('g[data-box]')].filter(vis).map((g) => [g.dataset.box, { at: Number(g.dataset.at), tone: g.dataset.tone, stale: g.dataset.stale === '1' }])),
    outputs: Object.fromEntries([...svg.querySelectorAll('g[data-output]')].map((g) => [g.dataset.output, Number(g.dataset.done)])),
    active: [...svg.querySelectorAll('g[data-thread]')].filter((g) => g.dataset.active === '1').length,
  };
}, SC);

await goPart(1, 0.95);
let d = await drawn();
check('scene part 1: 216 threads here, 32,768 for a 32-wide cube, the 1,024 limit', d.text.includes(`${threadsCube(6)} threads`) && d.text.includes(threadsCube(32).toLocaleString('en-US')) && d.text.includes('1,024'), d.text);
check('scene part 1: caption follows the part', (await caption()) === scripts[0]);
if (out) await page.screenshot({ path: out + '-scene1.png' });

await goPart(2, 0.97);
d = await drawn();
check('scene part 2: 36 threads, 16 compute, 2 planes loaded into prev and curr',
  d.counters['threads'] === String(threadsSlab(6)) && d.counters['of which compute'] === '16' && d.counters['planes loaded'] === String(TL.frames[0].planesLoaded) &&
  d.boxes.inPrev_s?.at === L(TL.frames[0].hold.prev) && d.boxes.inCurr_s?.at === L(TL.frames[0].hold.curr), JSON.stringify(d.counters) + JSON.stringify(d.boxes));
check('scene part 2: caption follows the part', (await caption()) === scripts[1]);

await goPart(3, 0.99);
d = await drawn();
check('scene part 3: plane 2 in next, output plane 1 complete (16 outputs)', d.boxes.inNext_s?.at === L(TL.frames[1].hold.next) && d.outputs['1'] === 16 && d.counters['outputs stored'] === String(TL.frames[2].stores), JSON.stringify(d));
await goPart(3, 0.3);
d = await drawn();
check('scene part 3: barrier 1 is drawn while the block waits', d.text.includes('barrier 1'), d.text);

await goPart(4, 0.2);
d = await drawn();
check('scene part 4: barrier 2 before the shift; the window has not moved yet', d.text.includes('barrier 2') && d.boxes.inPrev_s.at === 0 && d.boxes.inCurr_s.at === 1 && !d.boxes.inNext_s.stale, JSON.stringify(d.boxes));
await goPart(4, 0.98);
d = await drawn();
check('scene part 4: after the shift, prev holds plane 1, curr plane 2, next is free', d.boxes.inPrev_s.at === L(TL.frames[3].hold.prev) && d.boxes.inCurr_s.at === L(TL.frames[3].hold.curr) && d.boxes.inNext_s.stale, JSON.stringify(d.boxes));
check('scene part 4: names the true and the false dependence', d.text.includes('true dependence') && d.text.includes('false dependence'), d.text);

await goPart(5, 0.98);
d = await drawn();
check('scene part 5: 6 planes loaded for 4 output planes, all 64 outputs', d.counters['planes loaded'] === String(lastF.planesLoaded) && lastF.planesLoaded === 6 && Object.values(d.outputs).every((v) => v === 16) && d.counters['outputs stored'] === '64', JSON.stringify(d.counters));
check('scene part 5: width 32 reaches 1.47 FLOP/B with 1,024 threads, the 8-wide cube 0.96',
  d.text.includes(`1,024 threads, ${slabAI(32, 30).toFixed(2)} FLOP/B`) && d.text.includes(`512 threads, ${cubeAI(8).toFixed(2)} FLOP/B`), d.text);

await goPart(6, 0.99);
d = await drawn();
const r6 = TLR.frames[5];
check('scene part 6: registers kernel loads exactly what the three-plane kernel loads', d.counters['loaded, registers'] === String(r6.loads) && d.counters['loaded, 3 planes'] === String(TL.frames[5].loads) && r6.loads === TL.frames[5].loads, JSON.stringify(d.counters));
check('scene part 6: shared memory keeps one plane, 4,096 B at t = 32', d.counters['shared planes'] === '1' && d.counters['bytes at t = 32'] === sharedBytes('registers', 32).toLocaleString('en-US') && !d.boxes.inPrev_s && !d.boxes.inNext_s && d.boxes.inCurr_s, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene6.png' });

// Playing: the clock runs and a part hands over to the next by itself.
await goPart(2, 0.97);
await clickText(SC, 'Resume') || await clickText(SC, 'Play');
await pause(2500);
const nowPart = await page.$$eval(SC + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
check('scene: at the end of a part the next one starts by itself', nowPart === 3, String(nowPart));
check('scene: and its caption follows', (await caption()) === scripts[2]);
await clickText(SC, 'Pause');

// ================================================================ the stepper
const S = bench('Step one block through z');
const narrate = () => text(S + ' .narrate');
const hl = () => page.$$eval(S + ' .code .ln.hl', (ls) => ls.map((l) => l.textContent));
const counters = () => page.$eval(S + ' [data-out="counters"]', (e) => ({ ...e.dataset, perOutput: e.querySelector('[data-out="per-output"]').textContent, perIter: e.querySelector('[data-out="per-thread-iter"]').textContent }));
const toEnd = async () => { for (let i = 0; i < 60; i++) if (!(await clickText(S, 'Run to next barrier'))) break; };
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

const cfg6 = { N: 6, inTile: 4, ...CORRECT };
const prob6 = makeProblem(6);
const ib = interiorBlock({ ...cfg6, kernel: 'planes' });
const trP = runBlock({ ...cfg6, kernel: 'planes' }, prob6, ib.bx, ib.by, ib.bz, 'in-order', 5);
check('stepper: starts on the block that touches no edge', (await narrate()).includes(`Block (${ib.bx}, ${ib.by}, ${ib.bz})`), await narrate());
check('stepper: the code defines the tile width the default problem runs (4)', (await text(S + ' .code')).includes('#define IN_TILE_DIM 4') && !(await text(S + ' .code')).includes('#define IN_TILE_DIM 6'));
await clickText(S, 'Step');
let n = await narrate();
const w0 = trP.steps[0].writes[0];
check('stepper step 1: thread (0,0) loads the plane before into inPrev_s, value from the model', n.includes(`Thread (tx 0, ty 0) loads in at (z ${trP.steps[0].global.z}, y ${trP.steps[0].global.j}, x ${trP.steps[0].global.k}) = ${w0.value} into inPrev_s[0][0]`), n);
check('stepper step 1: the guard and the load line are highlighted', (await hl()).some((l) => l.includes('inPrev_s[ty][tx] = in[')), (await hl()).join(' / '));
await clickText(S, 'Run to next barrier');
n = await narrate();
check('stepper: run to next barrier stops on barrier 1 opening', n.startsWith('Barrier 1 opens'), n);
await clickText(S, 'Step');
await clickText(S, 'Back');
check('stepper: Back undoes a step', (await text(S)).includes(`step ${trP.steps.findIndex((s) => s.op === 'release1') + 1} of ${trP.steps.length}`));
// Step to the first computing thread.
const firstCompute = trP.steps.findIndex((s) => s.op === 'compute' && s.active);
for (let i = 0; i < 40; i++) {
  const t = await page.$eval(S, (e) => Number(/step (\d+) of/.exec(e.textContent)[1]));
  if (t >= firstCompute + 1) break;
  await clickText(S, 'Step');
}
n = await narrate();
const fc = trP.steps[firstCompute];
check('stepper: the first compute shows its sum and the value the model computes', n.includes(`Thread (tx ${fc.tx}, ty ${fc.ty}) computes out`) && n.includes(`= ${fc.value}.`) && n.includes('Seven shared-memory reads'), n);
check('stepper: the compute highlights the stencil statement', (await hl()).some((l) => l.includes('out[i*N*N')), (await hl()).join(' / '));
await toEnd();
check('stepper, three planes: the block matches the CPU sweep', (await text(S)).includes('outputs match the CPU sweep'));
let c = await counters();
check('stepper, three planes: 7 shared reads per output', c.perOutput === '7 per output', c.perOutput);
check('stepper, three planes: 2 reads + 2 writes per thread per iteration for the shift', c.perIter.startsWith('2 reads + 2 writes'), c.perIter);
check('stepper, three planes: global loads as the model counts them', Number(c.loads) === trP.counters.globalLoads, c.loads);
const loadsPlanes = Number(c.loads);
if (out) await page.screenshot({ path: out + '-stepper-planes.png' });

await select(S, 'Kernel', 'registers');
await toEnd();
c = await counters();
check('stepper, registers: correct, 4 shared reads per output, 1 write per thread per iteration', (await text(S)).includes('outputs match the CPU sweep') && c.perOutput === '4 per output' && c.perIter.startsWith('0 reads + 1 writes'), JSON.stringify(c));
check('stepper, registers: the global-load counter is the same as with three planes', Number(c.loads) === loadsPlanes, `${c.loads} vs ${loadsPlanes}`);
check('stepper, registers: each thread\'s registers are listed', (await text(S + ' table.threads thead')).includes('inPrev') && (await text(S + ' table.threads tbody')).includes('z'));

await select(S, 'Kernel', 'planes');
await tick(S, 'barrier 2', false);
const trNo2 = runBlock({ ...cfg6, kernel: 'planes', barrier2: false }, prob6, ib.bx, ib.by, ib.bz, 'in-order', 5);
check('stepper: removing barrier 2 offers a failure to jump to', await clickText(S, 'Go to the first failure'));
n = await narrate();
check('stepper: the failure is the model\'s first overwrite-before-read', n.includes(trNo2.violations[0].message), n);
check('stepper: the failing slot is marked', (await page.$$(S + ' .mx.shared span.bad')).length >= 1);
if (out) await page.screenshot({ path: out + '-stepper-fail.png' });
await toEnd();
check('stepper: without barrier 2 the block is wrong', (await text(S)).includes('outputs are wrong'));
await select(S, 'Thread order', 'lockstep');
await toEnd();
check('stepper: in lockstep the same broken kernel gives the right answer', (await text(S)).includes('outputs match the CPU sweep'));
await tick(S, 'barrier 2', true);
await select(S, 'Coefficients', 'as-printed-8.10');
await toEnd();
check('stepper: the pairing as printed makes every output wrong', /(\d+) of this block's \1 outputs are wrong/.test(await text(S)), await text(S + ' [data-out="verdict"]'));
check('stepper: as printed, the code shows c3 with y + 1', (await text(S + ' .code')).includes('c3*inCurr_s[ty + 1][tx]'));
await select(S, 'Coefficients', 'book-8.6');
await select(S, 'Problem', '1');
check('stepper: the book\'s picture is tile 6 on 10^3, block (1, 1, 1) by default', (await narrate()).includes('Block (1, 1, 1) is a plane of 6 × 6 threads'), await narrate());
check('stepper: with tile 6 the code defines IN_TILE_DIM as 6', (await text(S + ' .code')).includes('#define IN_TILE_DIM 6'));
check('stepper: the simplification note says what lockstep means and how unwritten slots are drawn',
  await page.evaluate(() => [...document.querySelectorAll('aside.note')].some((a) => a.textContent.includes('no real block promises') && a.textContent.includes('drawn as ·') && !a.textContent.includes('single warp'))));

// ================================================================ the outcome table
const O = bench('Every thread order at once');
const hidden = await page.$$eval(O + ' td[data-outcome]', (tds) => tds.map((x) => x.dataset.outcome));
check('outcomes: every cell is hidden until the reader asks', hidden.length === 24 && hidden.every((x) => x === 'hidden'), JSON.stringify(hidden));
check('outcomes: a button reveals them', await clickText(O, 'Show the results'));
const cells = await page.$$eval(O + ' tbody tr', (rows) => rows.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((x) => x.dataset.outcome)));
const rowNames = await page.$$eval(O + ' tbody tr', (rows) => rows.map((r) => r.dataset.row));
check('outcomes: six rows (two kernels × three barrier settings), four orders', cells.length === 6 && cells.every((r) => r.length === 4), JSON.stringify(cells));
const both = rowNames.map((r, i) => (r.endsWith('both barriers') ? i : -1)).filter((i) => i >= 0);
check('outcomes: with both barriers every order is correct', both.length === 2 && both.every((i) => cells[i].every((x) => x === 'correct')));
check('outcomes: lockstep is correct in every row', cells.every((r) => r[0] === 'correct'));
check('outcomes: a missing barrier fails in the two run-ahead orders', rowNames.every((r, i) => r.endsWith('both barriers') || (cells[i][1] === 'wrong' && cells[i][2] === 'wrong')));
const modelCell = runAll({ ...cfg6, kernel: 'registers', barrier2: false }, prob6, 'reversed', 5);
const shown = await page.$$eval(O + ' tbody tr[data-row="registers/without barrier 2"] td[data-outcome]', (tds) => tds[2].textContent);
check('outcomes: a cell shows the model\'s count and kind', shown.includes(`${modelCell.wrong.length} of ${prob6.interior.length} wrong`) && shown.includes('overwritten before read'), shown);
check('lockstep is explained after the barrier question: chapter 4\'s any-order rule and NVIDIA\'s warps', await page.evaluate(() =>
  [...document.querySelectorAll('.prose > p')].some((p) => p.textContent.includes('lockstep column hides both bugs') && p.textContent.includes('any order') && p.textContent.includes('within a warp'))));

// ================================================================ the explorer
const X = bench('Slab depth and the register budget');
await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
const slab = () => page.$eval(X + ' [data-out="slab"]', (e) => ({ ...e.dataset, regs: e.querySelector('[data-out="regs"]').textContent }));
let s = await slab();
check('explorer: t = 32, D = 30 as coded gives 1.47, limit 1.52, 12,288 B, 729 blocks on 256^3',
  s.ai === slabAI(32, 30).toFixed(2) && s.limit === slabAILimit(32).toFixed(2) && Number(s.shared) === 12288 && Number(s.blocks) === blocksSlab(256, 32, 30), JSON.stringify(s));
check('explorer: the plot marks the book\'s printed 1.52 as the deep-slab limit, the cube at 0.96 and the as-coded point',
  (await text(X + ' [data-mark="limit"]')).includes("the book's printed 1.52") && (await text(X + ' [data-mark="cube8"]')).includes('0.96') && (await text(X + ' [data-mark="as-coded"]')).includes('D = 30: 1.47'));
await clickText(X, 'registers');
s = await slab();
check('explorer: registers divide shared memory by three and leave FLOP/B alone', Number(s.shared) === 4096 && s.ai === '1.47' && s.regs === '3', JSON.stringify(s));
await setRange(X + ' input[aria-label="slab depth"]', 9);
s = await slab();
check('explorer: a 512-deep slab almost reaches the limit, with far fewer blocks', s.ai === slabAI(32, 512).toFixed(2) && Number(s.blocks) === blocksSlab(256, 32, 512) && blocksSlab(256, 32, 512) < blocksSlab(256, 32, 30), JSON.stringify(s));
await clickText(X, 'as coded: D = t − 2r');
await clickText(X, '2');
s = await slab();
check('explorer: order 2 needs a window of 5 values and loads D + 4 planes', s.regs === '5' && Number(s.planes) === planesLoaded(32 - 4, 2), JSON.stringify(s));
const devText = await text(X + ' [data-out="devices"]');
check('explorer: device figures come from reports, or the box says how to get them', devText.includes('No device reports') || /reports|maxThreadgroupMemoryLength|sharedMemPerBlock|maxThreadsPerBlock/.test(devText), devText);
if (out) await page.screenshot({ path: out + '-explorer.png' });

// ================================================================ listings and the erratum hunt
const tabsOf = (sel) => page.$$eval(sel + ' .tabs button', (b) => b.map((x) => x.textContent));
const planesFig = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('stencilPlanesKernel') || f.textContent.includes('labs/ch08/shared/planes.cu')));
const planesTabs = await planesFig.evaluate((f) => [...f.querySelectorAll('.tabs button')].map((b) => b.textContent));
check('listing: Fig. 8.10 from lab 8.1 with CUDA and Metal tabs', planesTabs.includes('CUDA') && planesTabs.includes('Metal'), planesTabs.join('|'));
const labPlanes = path.join(REPO, 'labs/ch08/shared/planes.cu');
if (fs.existsSync(labPlanes)) check('listing: the CUDA tab is the lab file', (await planesFig.evaluate((f) => f.textContent)).includes('stencilPlanesKernel'));
const statusOk = (st) => /verified on Metal|emulator-checked|front-end-checked|unverified/.test(st) && !/\bverified on CUDA\b/.test(st.replace('unverified on CUDA', ''));
const st1 = await planesFig.evaluate((f) => f.querySelector('.status')?.textContent ?? '');
check('listing: the status uses the project\'s vocabulary', statusOk(st1), st1);
check('listing: Fig. 8.12 corrected, with CUDA and Metal tabs', (await tabsOf('#s85-fig812')).join('|') === 'CUDA|Metal');
const labReg = path.join(REPO, 'labs/ch08/lab1_stencil3d/solution/kernel.cu');
if (fs.existsSync(labReg)) check('listing: Fig. 8.12 shows the lab\'s stencilRegTiledKernel', (await text('#s85-fig812')).includes('stencilRegTiledKernel'));

const H = bench('Fig. 8.12 as printed');
await page.$eval(H, (e) => e.scrollIntoView({ block: 'start' }));
const pick = async (q) => { await page.$eval(`${H} [data-line="${q}"] button`, (b) => b.click()); await pause(40); };
for (const q of [0, 1, 5, 6, 7]) await pick(q);
await clickText(H, 'Check');
let hunt = await page.$eval(H + ' [data-out="hunt"]', (e) => ({ ...e.dataset, t: e.textContent }));
check('erratum hunt: the five flagged lines found, no false alarm', hunt.found === '5' && hunt.false === '0', JSON.stringify(hunt));
check('erratum hunt: the reveal names inNext_s and the prose error on p. 196', (await text(H)).includes('no inNext_s') && hunt.t.includes('Fig. 8.10'));
const issues = await page.$$eval(H + ' [data-line]', (ls) => ls.map((l) => [l.dataset.issue, l.querySelector('b')?.textContent ?? '']));
check('erratum hunt: the c3/c4 lines are "inconsistent", not "wrong"; inNext_s and the numbering are errors',
  issues[0][0] === 'inconsistent' && issues[1][0] === 'inconsistent' && issues[0][1].startsWith('Inconsistent') && issues[5][0] === 'error' && issues[5][1].startsWith('Wrong') &&
  (await text(H)).includes('Inconsistent with Figs. 8.6 and 8.8: the results differ unless c3 = c4'), JSON.stringify(issues));
await clickText(H, 'Try again');
await pick(3);
await clickText(H, 'Check');
hunt = await page.$eval(H + ' [data-out="hunt"]', (e) => ({ ...e.dataset }));
check('erratum hunt: a correct line flagged counts as a false alarm', hunt.found === '0' && hunt.false === '1', JSON.stringify(hunt));
check('erratum hunt: links to the corrected listing', await clickText(H, 'Show the corrected listing'));
if (out) await page.screenshot({ path: out + '-hunt.png' });

// ================================================================ questions
const q = async (id) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), id);
const pred = await q('marches through 30 output planes');
await (await pred.$('input[type=number]')).type('1.47');
await (await pred.$('button.primary')).click();
await pause(150);
const predText = await pred.evaluate((e) => e.textContent);
check('prediction: committing 1.47 reveals the working and the book\'s 1.52', predText.includes('You predicted 1.47') && predText.includes('That is what you had') && predText.includes('1.52'), predText.slice(0, 200));
const ex2a = await q('coarsening factor of 16');
await (await ex2a.$('input[type=number]')).type('18432');
await (await ex2a.$('button.primary')).click();
await pause(120);
check('exercise 2a accepts 18,432', (await ex2a.evaluate((e) => e.textContent)).includes('Correct.'));
const ex2b = await q('How many output elements does it compute over its lifetime');
await (await ex2b.$('input[type=number]')).type('14400');
await (await ex2b.$('button.primary')).click();
await pause(120);
check('exercise 2b is its own question and accepts 14,400', (await ex2b.evaluate((e) => e.textContent)).includes('Correct.'));
check('exercise 2c (not yet answered) no longer gives 2b\'s answer', !(await (await q('arithmetic intensity in FLOP/B, counting every load')).evaluate((e) => e.textContent)).includes('14,400'));
check('the slab plot keeps a readable width and scrolls in its own frame', await page.$eval('[data-scroll="slab-plot"]', (e) => getComputedStyle(e).overflowX === 'auto' && parseFloat(getComputedStyle(e.querySelector('svg')).minWidth) >= 580));
check('"On Metal" gives its three details in the text column', await page.evaluate(() => {
  const h = [...document.querySelectorAll('.prose > h2')].find((x) => x.textContent === 'On Metal');
  const ul = h?.nextElementSibling?.nextElementSibling;
  return ul?.tagName === 'UL' && ul.children.length === 3 && ul.textContent.includes('uint3') && ul.textContent.includes('constexpr');
}));
check('the indexed window is not called correct without its test, nor llama.cpp\'s a proven hazard', await page.evaluate(() => {
  const t = document.querySelector('.prose').textContent;
  return t.includes('a unit test runs a model of its loop') && t.includes('The same shape appears in real code') && !t.includes('not a made-up hazard');
}));
const who = await q('read by more than one thread');
await who.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith('Only inCurr_s')).querySelector('input').click());
await (await who.$('button.primary')).click();
await pause(120);
check('who-reads question accepts "only inCurr_s"', (await who.evaluate((e) => e.textContent)).includes('Correct.'));
const div = await q('idle lanes alongside active ones');
await (await div.$('input[type=number]')).type('32');
await (await div.$('button.primary')).click();
await pause(120);
check('diverge question rejects 32 and gives 30', (await div.evaluate((e) => e.textContent)).includes('The answer is 30'));

// ================================================================ errors
const expected404 = (u) =>
  /\/content\/ch08\/narration\/(manifest\.json|z-march-\d\.mp3)$/.test(u) ||
  (/\/api\/lab-file\?path=/.test(u) && !fs.existsSync(path.join(REPO, 'labs', decodeURIComponent(u.split('path=')[1]))));
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch08/8.5', [
  { label: 'CUDA', lab: 'ch08/lab1_stencil3d', file: 'kernel.cu', where: 'Fig. 8.10', prefix: 'tested with lab 8.1: ' },
  { label: 'Metal', lab: 'ch08/lab1_stencil3d', file: 'kernel.metal', where: 'Fig. 8.10', prefix: 'tested with lab 8.1: ' },
  { label: 'CUDA', lab: 'ch08/lab1_stencil3d', file: 'kernel.cu', where: 'Fig. 8.12' },
  { label: 'Metal', lab: 'ch08/lab1_stencil3d', file: 'kernel.metal', where: 'Fig. 8.12' },
]);

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
