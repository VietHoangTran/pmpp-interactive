// Browser check for section 16.6: the lab listing, the fronts explorer, both tile steppers,
// the bank explorer, the launch prediction and every question. Expected values come from
// the tiling model (src/sim/wavetile.ts, imported here) and, for banks, from
// notes/today/ch16-facts.md E1 and the closed forms the page states.
// Usage: node e2e/ch16-s166.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { fill, launchPlan, loadsOf, cellAt, makeSequences, runTile, summary, busyPerStep } from '../src/sim/wavetile.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const B = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelector(sel).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B(scope), label);
  await pause(70);
  return done;
};
const setRange = async (scope, label, v) => {
  await page.$eval(`${B(scope)} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const setSelect = async (scope, label, v) => {
  await page.$eval(`${B(scope)} select[aria-label="${label}"]`, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(150);
};
const setCheck = async (scope, label, on) => {
  await page.evaluate((sel, label, on) => {
    const box = [...document.querySelector(sel).querySelectorAll('label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== on) box.click();
  }, B(scope), label, on);
  await pause(150);
};
const readout = (id) => text(`${B(F)} [data-readout="${id}"]`);
const states = () => page.$$eval(`${B(F)} rect[data-cell]`, (rs) => rs.reduce((m, r) => ({ ...m, [r.dataset.state]: (m[r.dataset.state] ?? 0) + 1 }), {}));

await goto(page, 'ch16/16.6');
const F = 'Fronts of tiles';
const S1 = 'Inside one tile';
const S2 = 'Inside one sheared tile';
const K = 'Which bank does each thread hit?';
for (const t of [F, S1, S2, K]) await page.waitForSelector(B(t));
await page.waitForSelector('section.scene[aria-label="Shearing a tile"]');

// ---------------------------------------------------------------- the lab listing
await page.waitForSelector('.code .tabs button', { timeout: 10000 });
const tabs = await page.$$eval('figure.code', (fs) => fs.map((f) => [...f.querySelectorAll('.tabs button')].map((b) => b.textContent)).find((t) => t.includes('Metal host')) ?? []);
check('the listing has CUDA, Metal kernel and Metal host tabs', ['CUDA', 'Metal kernel', 'Metal host'].every((t) => tabs.includes(t)), tabs.join('|'));
const listing = await page.$$eval('figure.code', (fs) => fs.map((f) => f.textContent).find((t) => t.includes('swHyperKernel')) ?? '');
check('the CUDA tab is lab 16.1\'s solution, both kernels', listing.includes('swSquareKernel') && listing.includes('swHyperKernel') && listing.includes('launchSwHyper'));
const status = await page.$$eval('figure.code .tabs .status', (s) => s.map((x) => x.textContent));
check('the listing\'s status uses the verification vocabulary', status.some((s) => /emulator-checked|emulator: no run|unverified/.test(s)) && !status.some((s) => /verified on (CUDA|HIP)/.test(s) && !/unverified/.test(s)), status.join(' | '));

// ---------------------------------------------------------------- fronts explorer
let g = { Lseq: 24, W: 4, scheme: 'square' };
let sum = summary(g);
check('default (L_seq 24, W 4, square): readouts are the model\'s',
  (await readout('T')) === String(sum.T) && (await readout('launches')) === String(sum.launches) && (await readout('most')) === String(sum.mostTiles)
  && (await readout('blocks')) === String(sum.blocks) && (await readout('steps')) === String(sum.stepsPerTile) && (await readout('critical')) === String(sum.criticalSteps)
  && (await readout('busy')) === String(Number(sum.busyMean.toFixed(2))),
  JSON.stringify([await readout('T'), await readout('launches'), await readout('most'), await readout('busy')]));
check('tiles-per-launch bars are 1, 2, ..., 6, ..., 1', (await page.$eval(`${B(F)} svg[data-bars]`, (e) => e.dataset.bars)) === sum.perLaunch.join(','));
check('busy threads per step in a square tile are 1, 2, 3, 4, 3, 2, 1', (await page.$eval(`${B(F)} svg[data-busy]`, (e) => e.dataset.busy)) === busyPerStep(4, 'square').join(','));
let st = await states();
check('launch 0 lights the 16 cells of tile (0, 0)', st.now === 16 && st.done === undefined, JSON.stringify(st));
for (let i = 0; i < 5; i++) await clickText(F, 'Next launch');
let plan = launchPlan(g, 5);
let lt = await text(`${B(F)} [data-launch-text]`);
check('launch 5: the narration lists the model\'s tiles from firstRow', lt.includes(`Launch d = 5: ${plan.tiles.length} blocks, firstRow = ${plan.firstRow}`) && lt.includes(plan.tiles.map((t) => `(${t.row}, ${t.col})`).join(', ')), lt);
st = await states();
check('launch 5: 6 tiles of 16 cells now, 15 tiles done', st.now === 96 && st.done === 15 * 16, JSON.stringify(st));

// W = 1 is the baseline
await setRange(F, 'L_seq', 48);
await setRange(F, 'tile width W', 1);
check('W = 1, L_seq = 48: 95 launches, the baseline\'s 2·L_seq − 1', (await readout('launches')) === '95' && (await readout('launches').then(() => text(`${B(F)} table.data`))).includes('2·L_seq − 1'));
check('W = 1: one step per tile, one thread busy', (await readout('steps')) === '1' && (await readout('busy')) === '1');
// find the W with the fewest launches at L_seq = 48
const launchesByW = [];
for (let W = 1; W <= 12; W++) {
  await setRange(F, 'tile width W', W);
  launchesByW.push(Number(await readout('launches')));
}
check('launches fall as W grows, to 7 at W = 12 (T = 4)', launchesByW.every((n, i) => n === summary({ Lseq: 48, W: i + 1, scheme: 'square' }).launches) && launchesByW[11] === 7, launchesByW.join(','));
if (out) await page.$eval(B(F), (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-fronts-square.png' });

// sheared
await setRange(F, 'tile width W', 3);
await setRange(F, 'L_seq', 9);
await clickText(F, 'sheared tiles');
g = { Lseq: 9, W: 3, scheme: 'sheared' };
sum = summary(g);
check('9 x 9 in 3 x 3 sheared tiles: 8 launches, at most 2 tiles, 12 tile computations, 3 steps',
  (await readout('launches')) === '8' && (await readout('most')) === '2' && (await readout('blocks')) === '12' && (await readout('steps')) === '3' && (await readout('busy')) === '3',
  JSON.stringify([await readout('launches'), await readout('most'), await readout('blocks')]));
check('sheared bars are the notes\' schedule: 1, 1, 2, 2, 2, 2, 1, 1', (await page.$eval(`${B(F)} svg[data-bars]`, (e) => e.dataset.bars)) === '1,1,2,2,2,2,1,1');
await setRange(F, 'launch d', 2);
lt = await text(`${B(F)} [data-launch-text]`);
check('sheared launch 2 runs tiles (0, 2) and (1, 0)', lt.includes('(0, 2), (1, 0)') && lt.includes('tile_col = d − 2·tile_row'), lt);
st = await states();
const f9 = fill(g, makeSequences(9).a, makeSequences(9).b);
const nowCells = f9.launchOf.filter((x) => x === 2).length;
check('the cells lit at launch 2 are the model\'s, part tiles included', st.now === nowCells && nowCells < 18, JSON.stringify({ st, nowCells }));
check('every tile is outlined: 12 parallelograms', (await page.$$(`${B(F)} polygon[data-tile]`)).length === 12);
// the '<' defect
await setCheck(F, 'tile_col', true);
const fl = fill({ ...g, lastColLess: true }, makeSequences(9).a, makeSequences(9).b);
st = await states();
check('"<" leaves the model\'s right-edge cells never computed, marked in the failure colour',
  (await readout('never')) === String(fl.neverComputed) && st.never === fl.neverComputed && fl.neverComputed === 9 && (await page.$$(`${B(F)} polygon[data-skipped]`)).length === 3,
  JSON.stringify({ never: await readout('never'), st }));
check('and the cells computed from them are counted as missing', Number(await readout('missing')) === fl.missing && fl.missing > fl.neverComputed);
if (out) await page.screenshot({ path: out + '-fronts-less.png' });
await setCheck(F, 'tile_col', false);
check('switching back to "<=" computes every cell', (await states()).never === undefined);

// ---------------------------------------------------------------- stepper, square
const narr = (s) => text(`${B(s)} [data-narrate]`);
const arith = (s) => page.$$eval(`${B(s)} [data-arith] tr`, (rs) => rs.map((r) => [...r.children].map((c) => c.textContent.trim()).join(' : ')));
const inputs = (s) => page.$$eval(`${B(s)} tr[data-load]`, (rs) => rs.map((r) => [r.dataset.load, r.dataset.from, r.children[1].textContent, r.children[2].textContent].join(' ')));
await clickText(S1, 'Step');
let n = await narr(S1);
check('square step 0: only thread 0 computes; threads 1 to 3 skip at the bounds test', n.startsWith('Step 0: thread 0 each compute one cell') && n.includes('Threads 1, 2 and 3 skip it (q_tile = -1 is outside 0 to 3') && n.includes('the warp splits'), n);
for (let i = 0; i < 3; i++) await clickText(S1, 'Step');
n = await narr(S1);
check('square step 3: all four threads busy, then the barrier', n.includes('Step 3: threads 0, 1, 2 and 3 each compute one cell. Every thread is busy.') && n.includes('Barrier: every thread has finished step 3.'), n);
const sg = { Lseq: 12, W: 4, scheme: 'square' };
const c13 = cellAt(sg, { row: 1, col: 1 }, 3, 1);
let ar = await arith(S1);
check('thread 1 at step 3: q_tile = 3 − 1 = 2, r = 6, q = 7', ar.includes('q_tile = d_tile − r_tile : 3 − 1 = 2') && ar.includes(`r = W·tile_row + r_tile + 1 : 4·1 + 1 + 1 = ${c13.r}`) && ar.includes(`q = W·tile_col + q_tile + 1 : 4·1 + 2 + 1 = ${c13.q}`) && ar.some((x) => x.includes('computes (6, 7)')), ar.join(' | '));
let ins = await inputs(S1);
const want = loadsOf(sg, c13).map((l) => `${l.name} ${l.from} (${l.r}, ${l.q}) ${l.from === 'tile' ? `swTile slot (${l.slot[0]}, ${l.slot[1]})` : 'H, global memory'}`);
check('its three inputs and where they come from are the model\'s', JSON.stringify(ins) === JSON.stringify(want), JSON.stringify(ins));
const hl = await page.$$eval(`${B(S1)} .code .ln.hl`, (ls) => ls.map((l) => l.textContent));
check('the highlighted lines are the tile branches of sqLoadN, sqLoadW and sqLoadNW', hl.length === 3 && hl.every((l) => l.includes(': swTile[')), hl.join(' | '));
await setSelect(S1, 'selected thread', 0);
ins = await inputs(S1);
check('thread 0 at step 3 reads north and north-west from global memory (top row of the tile)', ins[0].startsWith('n global') && ins[2].startsWith('nw global') && ins[1].startsWith('w tile'), ins.join(' | '));
for (let i = 0; i < 20; i++) await clickText(S1, 'Run to the end of the step');
check('lockstep with the barrier: the tile matches the CPU table', (await page.$eval(`${B(S1)} [data-verdict]`, (e) => e.dataset.verdict)) === 'correct');
if (out) await page.$eval(B(S1), (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-stepper-square.png' });

// break the barrier
await setSelect(S1, 'thread order', 'last');
await setCheck(S1, 'barrier between steps', false);
const bad = runTile(sg, { row: 1, col: 1 }, { order: 'last', barrier: false });
check('without the barrier, last thread first: a bad read to jump to', await clickText(S1, 'Go to the first bad read'));
n = await narr(S1);
const badRead = bad.ticks[bad.firstBad].parts[0].reads.find((r) => r.problem);
check('the failure is the model\'s: thread 3 reads a slot of thread 2 before it is written',
  n.includes(`Failure: thread 3 read slot (${badRead.slot[0]}, ${badRead.slot[1]}) before thread ${badRead.slot[0]} wrote it (read before write)`), n);
check('the slot is marked in the failure colour', (await page.$$(`${B(S1)} [data-slots] span.bad`)).length >= 1);
if (out) await page.screenshot({ path: out + '-stepper-bad.png' });
for (let i = 0; i < 60; i++) await clickText(S1, 'Step');
check('and the tile ends wrong', (await page.$eval(`${B(S1)} [data-verdict]`, (e) => e.dataset.verdict)) === 'wrong');
await setSelect(S1, 'thread order', 'lockstep');
for (let i = 0; i < 20; i++) await clickText(S1, 'Run to the end of the step');
check('the same broken kernel in lockstep gives the right answer', (await page.$eval(`${B(S1)} [data-verdict]`, (e) => e.dataset.verdict)) === 'correct');
await setSelect(S1, 'thread order', 'first');
check('thread 0 first without the barrier fails at its store', await clickText(S1, 'Go to the first bad read'));
n = await narr(S1);
check('the failure is a store before the other threads wrote', n.includes('Failure: thread 0 stored slot (1, 0) to the table before thread 1 had written it'), n);

// ---------------------------------------------------------------- stepper, sheared left edge
const hg = { Lseq: 12, W: 4, scheme: 'sheared' };
await setSelect(S2, 'which tile', 'edge');
await clickText(S2, 'Step');
await clickText(S2, 'Step');
n = await narr(S2);
const idle1 = [0, 1, 2, 3].filter((t) => !cellAt(hg, { row: 1, col: 0 }, 1, t).active);
check('left-edge sheared tile, step 1: the model\'s threads skip, left of the table', n.includes(`Thread${idle1.length === 1 ? '' : 's'} ${idle1.length === 1 ? idle1[0] : idle1.slice(0, -1).join(', ') + ' and ' + idle1.at(-1)} skip`) && n.includes('is left of the table'), n);
await setSelect(S2, 'selected thread', 1);
ins = await inputs(S2);
const c11 = cellAt(hg, { row: 1, col: 0 }, 1, 1);
const lw = loadsOf(hg, c11);
check('thread 1 at step 1 computes (6, 1) and reads its west input (6, 0) from a zeroed tile slot', c11.r === 6 && c11.q === 1 && lw[1].from === 'tile' && lw[1].q === 0 && ins[1] === `w tile (6, 0) swTile slot (${lw[1].slot.join(', ')})`, ins.join(' | '));
const v = await page.$eval(`${B(S2)} tr[data-load="w"] td:last-child`, (e) => e.textContent);
check('and that read is 0, the boundary value', v === '0', v);
for (let i = 0; i < 10; i++) await clickText(S2, 'Run to the end of the step');
check('zeroed: the left-edge tile is correct', (await page.$eval(`${B(S2)} [data-verdict]`, (e) => e.dataset.verdict)) === 'correct');
await setCheck(S2, 'zero the tile first', false);
check('not zeroed: a bad read to jump to', await clickText(S2, 'Go to the first bad read'));
n = await narr(S2);
check('it is a read of a slot nothing wrote or zeroed', n.includes('which nothing wrote or zeroed: garbage'), n);
for (let i = 0; i < 10; i++) await clickText(S2, 'Run to the end of the step');
check('and the tile ends wrong', (await page.$eval(`${B(S2)} [data-verdict]`, (e) => e.dataset.verdict)) === 'wrong');
const hl2 = await page.$$eval(`${B(S2)} .code .ln`, (ls) => ls.map((l) => l.textContent).join('\n'));
check('the sheared stepper shows hyLoadN, hyLoadW and hyLoadNW', hl2.includes('hyLoadN(') && hl2.includes('hyLoadNW(') && hl2.includes('(q_tile - 2)'));
check('and captions them as the lab\'s solution', (await text(`${B(S2)} figure.code`)).includes('Lab 16.1\'s solution for these three helpers'));
const s1code = await page.$$eval(`${B(S1)} .code .ln`, (ls) => ls.map((l) => l.textContent).join('\n'));
check('the square stepper has no scheme switch and never shows the sheared loads', !s1code.includes('hyLoad') && !(await page.$(`${B(S1)} .seg[aria-label="Tiles"]`)), s1code.slice(0, 80));
if (out) await page.$eval(B(S2), (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-stepper-sheared.png' });

// ---------------------------------------------------------------- banks
const deg = () => page.$eval(`${B(K)} [data-degree]`, (e) => Number(e.dataset.degree));
const words = () => page.$$eval(`${B(K)} svg[data-banks] g[data-thread]`, (gs) => gs.map((g) => [Number(g.dataset.thread), g.dataset.word]).sort((a, b) => a[0] - b[0]).map((x) => x[1]));
check('default: sheared W = 16 unpadded, step 5 write is 8-way', (await deg()) === 8);
check('lane t writes word 16t + 5', (await words()).join(',') === Array.from({ length: 16 }, (_, t) => 16 * t + 5).join(','));
await setSelect(K, 'layout', 'rowgap');
check('one gap per row: no conflict, words 17t + 5', (await deg()) === 1 && (await words()).join(',') === Array.from({ length: 16 }, (_, t) => 17 * t + 5).join(','));
const summaryRow = (k, l) => page.$$eval(`${B(K)} td[data-cell^="${k}/${l}/"]`, (cs) => cs.map((c) => Number(c.dataset.value)).join(','));
const E1 = { 'square/unpadded': '1,1,1,1', 'square/rowgap': '2,8,32,32', 'square/pad': '1,1,32,2', 'sheared/unpadded': '2,8,32,32', 'sheared/rowgap': '1,1,1,1', 'sheared/pad': '1,1,1,2' };
const got = {};
for (const key of Object.keys(E1)) got[key] = await summaryRow(...key.split('/'));
check('the summary table is ch16-facts E1 on NVIDIA', JSON.stringify(got) === JSON.stringify(E1), JSON.stringify(got));
check('the W = 32 cells show the 31 to 32 range', (await text(`${B(K)} td[data-cell="sheared/unpadded/32"]`)) === '31–32');
await clickText(K, 'square tiles');
await clickText(K, 'W = 32');
await setRange(K, 'step', 31);
check('square tiles padded at W = 32: step 31 write is 32-way', (await deg()) === 32);
await setSelect(K, 'layout', 'unpadded');
check('square tiles unpadded: none', (await deg()) === 1);
const strides = (await words()).map(Number);
check('the square kernel strides W − 1 = 31 words', strides.slice(1).every((w, i) => w - strides[i] === 31), strides.slice(0, 5).join(','));
await setSelect(K, 'banks', 'amd64');
check('64 banks, 64 lanes: sheared unpadded is 1, 4, 16, 64', (await summaryRow('sheared', 'unpadded')) === '1,4,16,64' && (await summaryRow('sheared', 'pad')) === '1,1,1,1');
await setSelect(K, 'banks', 'nvidia');
await clickText(K, 'W = 64');
await clickText(K, 'sheared tiles');
await setSelect(K, 'layout', 'pad');
check('W = 64 has two warps to choose from', (await page.$$(`${B(K)} select[aria-label="warp"] option`)).length === 2);
const worstPad = await page.$$eval(`${B(K)} tr[data-access]`, (rs) => rs.map((r) => r.dataset.worstDegree).join(','));
check('the book\'s pad() at W = 64 leaves 2-way conflicts', worstPad === '2,2,2,2', worstPad);
if (out) await page.$eval(B(K), (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-banks.png' });

// ---------------------------------------------------------------- order: questions before the tools that answer them
const order = await page.evaluate((sels) => {
  const all = [...document.querySelectorAll('.prose > *')];
  const at = (pred) => all.findIndex(pred);
  const q = (needle) => at((e) => e.classList.contains('q') && e.textContent.includes(needle));
  const bench = (t) => at((e) => e.matches(t));
  const simp = (i) => all[i - 1]?.matches('aside.note') && all[i - 1].textContent.startsWith('What this model leaves out');
  const scene = at((e) => e.querySelector?.('section.scene[aria-label="Shearing a tile"]') || e.matches('section.scene[aria-label="Shearing a tile"]'));
  return { nw: q('In which stored column of row r_tile − 1'), s2: bench(sels.S2), banks16: q('How many passes does the bank hardware need'), pad: q('same one-gap-per-row padding'), k: bench(sels.K), kSimp: simp(bench(sels.K)), scene, sceneSimp: simp(scene) };
}, { S2: B(S2), K: B(K) });
check('q-nw-offset comes before the sheared stepper', order.nw >= 0 && order.nw < order.s2, JSON.stringify(order));
check('q-banks16 and q-pad-square come before the bank explorer', order.banks16 >= 0 && order.pad >= 0 && order.banks16 < order.k && order.pad < order.k, JSON.stringify(order));
check('a Simplification note sits just before the bank explorer and the shear scene', order.kSimp && order.sceneSimp, JSON.stringify(order));

// ---------------------------------------------------------------- the prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Sheared, how many launches?')));
await (await pq.$('input[type=number]')).type('6');
await (await pq.$('button.primary')).click();
await pause(150);
const rv = await pq.evaluate((e) => e.textContent);
check('the launch prediction reveals 8 = 3T − 1 against 5', rv.includes('You predicted 6') && rv.includes('8 launches, 3T − 1 with T = 3, against 2T − 1 = 5'), rv);

// ---------------------------------------------------------------- every question
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(needle)), needle);
  if (typeof value === 'number') await (await q.$('input[type=number]')).type(String(value));
  else await (await q.$$('.opts input[type=radio]'))[value.pick].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const qs = [
  ['L_seq = 1000', 63],
  ['front d = 6', 3],
  ['A full 16 × 16 square tile', 8.26],
  ['writes column t of every tile row', { pick: 0 }],
  ['In which stored column of row r_tile − 1', { pick: 0 }],
  ['zeroes its tile before computing', { pick: 0 }],
  ['When is it likely to lose to square tiles', { pick: 0 }],
  ['How many passes does the bank hardware need', 8],
  ['same one-gap-per-row padding', { pick: 0 }],
  ['__vimax3_s32_relu(nw + s, w + g, n + g)', { pick: 0 }],
  ['Mamba-2 prefill', { pick: 0 }],
];
for (const [needle, value] of qs) {
  const verdict = await answer(needle, value);
  check(`question "${needle}" accepts the right answer`, verdict.startsWith('Correct.'), verdict.slice(0, 120));
}
const wrong = await (async () => {
  const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes('Mamba-2 prefill')));
  await (await q.$$('.opts input[type=radio]'))[2].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong')?.textContent ?? ''));
})();
check('a wrong choice says why', wrong.startsWith('Not quite.') && wrong.includes('host loop'), wrong.slice(0, 160));
check('eleven questions on the page, one prediction', (await page.$$eval('.q', (qs) => qs.length)) === 12);

// ---------------------------------------------------------------- narrow screen and themes
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px the page does not scroll sideways', narrow);
const fits = await page.evaluate((sels) => sels.map((s) => document.querySelector(s).getBoundingClientRect().right <= window.innerWidth + 1), [F, S1, S2, K].map(B));
check('every interactive fits the screen at 390 px', fits.every(Boolean), JSON.stringify(fits));
const chart = await page.$eval(`${B(K)} svg[data-banks]`, (s) => ({ w: s.getBoundingClientRect().width, scroll: s.parentElement.scrollWidth > s.parentElement.clientWidth }));
check('at 390 px the bank chart keeps a readable width and scrolls inside its box', chart.w >= 659 && chart.scroll, JSON.stringify(chart));
if (out) {
  for (const t of [F, S1, K]) {
    await page.$eval(B(t), (e) => e.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: `${out}-narrow-${t.split(' ')[0].toLowerCase()}.png` });
  }
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
  for (const t of [F, S2, K]) {
    await page.$eval(B(t), (e) => e.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: `${out}-light-${t.split(' ')[0].toLowerCase()}.png` });
  }
}

const expected404 = (u) => /\/content\/ch16\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e) && !(e.startsWith('requestfailed') && expected404(e.slice(15))));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch16/16.6', [
  { label: 'CUDA', lab: 'ch16/lab1_wavefront', file: 'kernel.cu' },
  { label: 'Metal kernel', lab: 'ch16/lab1_wavefront', file: 'kernel.metal' },
  { label: 'Metal host', lab: 'ch16/lab1_wavefront', file: 'launch.mm' },
]);

check('no console or page errors', realErrors.length === 0, realErrors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
