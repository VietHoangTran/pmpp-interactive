// Browser check for section 16.3: the pattern explorer, the Floyd-Warshall stepper and the
// "who waits for whom" explorer, each compared with the model it runs on (imported here
// directly), both predictions, every question, the corrected listing, and the page in
// both themes and at phone width. The scene has its own check, ch16-scene-neighbours.mjs.
// Usage: node e2e/ch16-s163.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { inputsOf, presetFronts, stencil, wavefronts } from '../src/sim/wavefront.ts';
import { FIG_16_5, INF, VERTICES, runSteps, stateAfter, stepTrace } from '../src/sim/floydwarshall.ts';
import { flagsAfter, frontMakespan, schedule } from '../src/sim/wavesched.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const P = bench('Which cells can run together');
const F = bench('One Floyd-Warshall step');
const W = bench('Who waits for whom');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
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
const select = async (scope, aria, value) => {
  await page.$eval(`${scope} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(100);
};
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  const input = await q.$('input[type=number]');
  await input.evaluate((el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
};
const answerChoice = async (needle, index) => {
  const q = await qHandle(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
};
const commit = async (needle, value) => {
  const q = await qHandle(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await qHandle(needle)).evaluate((e) => e.textContent.replace(/\s+/g, ' '));
};

await goto(page, 'ch16/16.3');
await page.waitForSelector(P);

// ================================================================ the pattern explorer
const cellSteps = () => page.$$eval(P + ' g[data-cell]', (g) => g.map((x) => Number(x.dataset.step)));
const readout = (k) => text(`${P} td[data-k="${k}"]`);
let f = presetFronts('smith-waterman', 8, 8);
check('explorer starts on Smith-Waterman 8 x 8, every cell at the model\'s step', JSON.stringify(await cellSteps()) === JSON.stringify(f.step));
check('readouts: 15 steps, widest 8, mean 4.27, farthest input 2 back, rows 1 step apart',
  (await readout('steps')) === '15' && (await readout('widest')) === '8 cells' && (await readout('mean')) === f.mean.toFixed(2) && (await readout('reach')) === '2' && (await readout('lag')) === '1 step',
  JSON.stringify([await readout('steps'), await readout('widest'), await readout('mean'), await readout('reach'), await readout('lag')]));
const bars = await page.$$eval(P + ' g[data-bar]', (g) => g.map((x) => Number(x.dataset.size)));
check('the bar chart is the model\'s cells per step', JSON.stringify(bars) === JSON.stringify(f.sizes), JSON.stringify(bars));
// Select a cell: its inputs are drawn and named.
await page.$eval(P + ' g[data-cell="2,5"]', (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(100);
let arrows = await page.$$eval(P + ' line[data-input]', (l) => l.map((x) => x.dataset.input).sort());
check('clicking cell (2, 5) draws arrows from its three inputs', JSON.stringify(arrows) === JSON.stringify(inputsOf(f, 2, 5).map(([r, c]) => `${r},${c}`).sort()), JSON.stringify(arrows));
check('and says when it runs and how many share its step', (await text(P + ' [data-selected]')).startsWith('Cell (2, 5) runs at step 7.') && (await text(P + ' [data-selected]')).includes(`${f.sizes[7]} cells share its step`), await text(P + ' [data-selected]'));
if (out) await page.screenshot({ path: out + '-explorer-sw.png', clip: await page.$eval(P, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });

await press(P, 'H.264');
f = presetFronts('h264', 8, 8);
check('H.264: c + 2r, 22 steps, each row two steps after the row above', JSON.stringify(await cellSteps()) === JSON.stringify(f.step) && (await readout('steps')) === '22' && (await readout('lag')) === '2 steps', await readout('steps'));
await press(P, 'Checkerboard');
f = presetFronts('checkerboard', 8, 8);
check('checkerboard: one row per step, constant fronts of 8', JSON.stringify(await cellSteps()) === JSON.stringify(f.step) && (await readout('steps')) === '8' && (await readout('widest')) === '8 cells', await readout('steps'));
await press(P, 'Fibonacci');
check('Fibonacci: one row, 12 steps of one cell', (await page.$eval(P + ' svg[data-grid]', (s) => s.dataset.grid)) === '1x12' && (await readout('steps')) === '12' && (await readout('widest')) === '1 cells' && (await readout('lag')) === 'one row only');
// A stencil of your own: Smith-Waterman without above-left.
await press(P, 'Smith-Waterman');
await page.$eval(P + ' input[data-neighbour="NW"]', (e) => e.click());
await pause(100);
f = wavefronts(8, 8, stencil(['N', 'W']));
const pressed = await page.$$eval(P + ' .seg button[aria-pressed="true"]', (b) => b.length);
check('unticking above-left leaves no preset selected; the same 15 steps, but nothing reads two fronts back',
  pressed === 0 && (await readout('steps')) === String(f.steps) && (await readout('reach')) === '1', JSON.stringify([pressed, await readout('steps'), await readout('reach')]));
await page.$eval(P + ' input[data-neighbour="NE"]', (e) => e.click());
await select(P, 'Rows', 5);
await select(P, 'Columns', 7);
f = wavefronts(5, 7, stencil(['N', 'W', 'NE']));
check('a custom stencil on a 5 x 7 table: the model\'s steps for every cell', JSON.stringify(await cellSteps()) === JSON.stringify(f.step) && (await readout('steps')) === String(f.steps), JSON.stringify([await readout('steps'), f.steps]));

// ---------------------------------------------------------------- prediction and questions on fronts
let t = await commit('cell to its left and the cell above-left (the', 12);
check('p-sw8: a committed prediction of 12 gets the answer 15 and R + C - 1', t.includes('You predicted 12') && t.includes('It takes 15') && t.includes('R + C − 1'), t.slice(0, 400));
t = await answerNumeric('the H.264 pattern of Fig. 16.2(d)', 15);
check('q-h264: 15 is wrong, the answer is 22', t.includes('Not quite. The answer is 22 steps.'), t.slice(0, 200));
t = await answerNumeric('the H.264 pattern of Fig. 16.2(d)', 22);
check('q-h264: 22 is right, with c + 2r', t.includes('Correct.') && t.includes('c + 2r'), t.slice(0, 300));
t = await answerNumeric('how many earlier fronts must stay in memory', 2);
check('q-live: 2 fronts', t.includes('Correct.') && t.includes('traceback'), t.slice(0, 300));
t = await answerChoice('memoised recursive solver', 1);
check('q-topdown: "GPU code cannot call functions" is wrong', t.includes('Not quite.') && t.includes('Metal 2.4'), t.slice(0, 300));
t = await answerChoice('memoised recursive solver', 0);
check('q-topdown: uncoalesced lookups and a run-time schedule', t.includes('Correct.'), t.slice(0, 200));

// ================================================================ the Floyd-Warshall stepper
const tableVals = () => page.$$eval(F + ' [data-cell]', (c) => c.map((x) => x.dataset.v));
const modelVals = (d) => d.flat().map((v) => (v === INF ? '' : String(v)));
const progress = () => text(F + ' [data-progress]');
const chips = () => page.$$eval(F + ' [data-check]', (c) => Object.fromEntries(c.map((x) => [x.dataset.check, x.className.includes('ok') ? 'ok' : 'fail'])));
const highlighted = () => page.$$eval(F + ' .code .ln.hl', (l) => l.map((x) => x.textContent));
const v = (n) => VERTICES.indexOf(n);

check('the stepper starts on Fig. 16.5\'s table as printed', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(FIG_16_5)));
let tr = stepTrace(FIG_16_5, 0);
await press(F, 'Next block');
check('k = a, block a: the narration names the block and its shared value', (await text(F + ' [data-narrate]')).startsWith('Block for row a (blockIdx.y = 0): thread 0 reads dist[a][a]'), await text(F + ' [data-narrate]'));
await press(F, 'Next block');
check('k = a, block b retires: FW_INF, and the retire line is highlighted', (await text(F + ' [data-narrate]')).includes('It is FW_INF') && (await highlighted()).some((l) => l.includes('if (dist_k_col == FW_INF) return;')), JSON.stringify(await highlighted()));
await press(F, 'Run the step');
check(`k = a: ${tr.retired.length} of 8 blocks retire and nothing changes`, (await progress()).includes(`blocks run: 8 of 8; retired: ${tr.retired.length}; cells changed: 0`), await progress());
let ch = await chips();
check('k = a: both checks pass', ch['row-col'] === 'ok' && ch.same === 'ok', JSON.stringify(ch));
const retiredRows = await page.$$eval(F + ' [data-row][data-retired="1"]', (r) => r.map((x) => x.dataset.row));
check('the seven retired rows are marked', JSON.stringify(retiredRows) === JSON.stringify(tr.retired.map((i) => VERTICES[i])), JSON.stringify(retiredRows));

await press(F, 'c');
tr = stepTrace(runSteps(FIG_16_5, 2), 2);
check('k = c starts from the table after steps a and b', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(tr.before)));
for (let i = 0; i < 3; i++) await press(F, 'Next block');
check('after three blocks the table is the model\'s', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(stateAfter(tr, 3))));
await press(F, 'Run the step');
const changed = await page.$$eval(F + ' [data-changed="1"]', (c) => c.map((x) => `${x.dataset.cell}=${x.dataset.v}`).sort());
const want = tr.changed.map((c) => `${c.i},${c.j}=${c.to}`).sort();
check('k = c changes a to e 23, b to e 19, g to e 23 and h to e 13', JSON.stringify(changed) === JSON.stringify(want) && want.join(' ') === '0,4=23 1,4=19 6,4=23 7,4=13', JSON.stringify(changed));
check('k = c: rows d and f retire', (await progress()).includes('retired: 2; cells changed: 4'), await progress());
if (out) await page.screenshot({ path: out + '-stepper-c.png', clip: await page.$eval(F, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });

// Break dist[k][k] = 0: row c and column c change while others read them.
await page.$eval(F + ' input[data-neg]', (e) => e.click());
await pause(120);
const neg = stepTrace(runSteps(FIG_16_5, 2), 2, { diag: -1 });
const negTwo = stepTrace(runSteps(FIG_16_5, 2), 2, { diag: -1, mode: 'two-buffers' });
for (let i = 0; i < 5; i++) await press(F, 'Next block'); // a, b, c, d (retires), e: e reads the new row c
const nar = await page.$eval(F + ' [data-narrate]', (e) => ({ bad: e.classList.contains('bad'), t: e.textContent }));
check('with dist[c][c] = -1, block e reads a row c that block c already changed, and the narration says so', nar.bad && nar.t.includes('the result depends on who ran first'), JSON.stringify(nar));
await press(F, 'Run the step');
ch = await chips();
check('both checks fail: row/column c changed, in place differs from two buffers', ch['row-col'] === 'fail' && ch.same === 'fail' && JSON.stringify(neg.after) !== JSON.stringify(negTwo.after), JSON.stringify(ch));
check('the table is the model\'s in-place result', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(neg.after)));
await select(F, 'Block order', 'bottom-first');
await press(F, 'Run the step');
const negB = stepTrace(runSteps(FIG_16_5, 2), 2, { diag: -1, order: 'bottom-first' });
check('bottom row first gives a different table again: a race', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(negB.after)) && JSON.stringify(negB.after) !== JSON.stringify(neg.after));
await select(F, 'Table', 'two-buffers');
await press(F, 'Run the step');
check('two buffers gives the model\'s two-buffer table', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(negTwo.after)));
if (out) await page.screenshot({ path: out + '-stepper-race.png', clip: await page.$eval(F, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await page.$eval(F + ' input[data-neg]', (e) => e.click());
await pause(100);
await press(F, 'Run the step');
ch = await chips();
check('dist[c][c] back to 0: with two buffers, both checks pass again', ch['row-col'] === 'ok' && ch.same === 'ok', JSON.stringify(ch));
await press(F, 'Next k');
check('Next k moves to d with the table after c', JSON.stringify(await tableVals()) === JSON.stringify(modelVals(runSteps(FIG_16_5, 3))) && (await text(F + ' [data-narrate]')).startsWith('Step k = d'));

// The listing.
const tabs = await page.$$eval('.prose > figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the corrected listing has a kernel tab and a host-loop tab', tabs.includes('kernel') && tabs.includes('host loop'), tabs.join('|'));
const status = await page.$eval('.prose > figure.code .tabs .status', (s) => s.textContent);
check('its status uses the fixed vocabulary', status === 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', status);
const listing = await page.$eval('.prose > figure.code pre', (p) => p.textContent);
check('the listing is the corrected one: integer sentinel, inRange flag', listing.includes('#define FW_INF (INT_MAX / 2)') && listing.includes('bool inRange') && !listing.includes('INFINITY'));

// Floyd-Warshall questions.
t = await answerNumeric('How many cells does step k = c change', 2);
check('q-fw-changes: 2 is wrong (that is step c on the printed table)', t.includes('Not quite. The answer is 4 cells.'), t.slice(0, 200));
t = await answerNumeric('How many cells does step k = c change', 4);
check('q-fw-changes: 4, with the four cells named', t.includes('Correct.') && t.includes('a to e = 23, b to e = 19, g to e = 23 and h to e = 13') && t.includes('Rows d and f'), t.slice(0, 600));
t = await answerChoice('update dist in place while other threads', 1);
check('q-fw-inplace: lockstep is wrong', t.includes('Not quite.') && t.includes('nothing orders blocks'), t.slice(0, 300));
t = await answerChoice('update dist in place while other threads', 0);
check('q-fw-inplace: dist[k][k] = 0 keeps row k and column k fixed', t.includes('Correct.') && t.includes('double buffering'), t.slice(0, 400));
t = await answerChoice('comes from <cmath>', 1);
check('q-infinity: "INFINITY converts to the largest int" is wrong', t.includes('Not quite.') && t.includes('the int becomes a float'), t.slice(0, 300));
t = await answerChoice('comes from <cmath>', 0);
check('q-infinity: the exits never fire; INT_MAX / 2 as the sentinel', t.includes('Correct.') && t.includes('FW_INF = INT_MAX / 2'), t.slice(0, 500));
t = await answerNumeric('16 × 16 blocks', 0.125);
check('q-ex1: 0.125 reads per cell', t.includes('Correct.') && t.includes('about 1.004 per cell'), t.slice(0, 400));

// ================================================================ who waits for whom
t = await commit('starts as soon as the tile above it and the tile to its left are done', 9);
check('p-flags-2sm: a prediction of 9 is right', t.includes('You predicted 9') && t.includes('Right:'), t.slice(0, 300));
const slots = () => page.$$eval(W + ' g[data-slot]', (g) => g.map((x) => `${x.dataset.slot}:${x.dataset.state}:${x.dataset.tile}`));
const modelSlots = (sch, n) => {
  const outS = [];
  for (let s = 0; s < sch.S; s++) for (let u = 0; u < n; u++) {
    const c = sch.timeline[Math.min(u, sch.timeline.length - 1)][s];
    outS.push(`${u},${s}:${c.state}:${c.tile ?? ''}`);
  }
  return outS;
};
const wout = (k) => text(`${W} td[data-k="${k}"]`);
let sch = schedule(4, 2, 'front');
check('the explorer starts on the book\'s example: relaunch 10, flags 9', (await wout('front')) === '10 steps' && (await wout('counter')) === '9 steps' && (await wout('makespan')) === '10 steps');
await press(W, 'Step');
await press(W, 'Step');
const flags = await page.$$eval(W + ' g[data-tile][data-done="1"]', (g) => g.map((x) => Number(x.dataset.tile)).sort((a, b) => a - b));
check('after two steps the flags of tiles 0, 1 and 2 are set', JSON.stringify(flags) === JSON.stringify([...flagsAfter(sch, 2)].sort((a, b) => a - b)) && flags.join() === '0,1,2', flags.join());
await press(W, 'Run to the end');
check('relaunch per front: the chart is the model\'s timeline, Fig. 16.21\'s left schedule', JSON.stringify(await slots()) === JSON.stringify(modelSlots(sch, 10)));
check('4 SM-steps without work, and a barrier after every front but the last', (await wout('idle')) === '4 of 4' && (await page.$$(W + ' line[data-barrier]')).length === sch.barriers.length - 1, await wout('idle'));
if (out) await page.screenshot({ path: out + '-wait-front.png', clip: await page.$eval(W, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await press(W, 'flags, tiles from a counter');
await press(W, 'Run to the end');
sch = schedule(4, 2, 'counter');
check('flags: the chart is the model\'s, 9 steps, the right schedule of Fig. 16.21', JSON.stringify(await slots()) === JSON.stringify(modelSlots(sch, 9)) && (await wout('makespan')) === '9 steps' && (await wout('idle')) === '2 of 2');
check('the slot that claims tile 1 waits in step 1', (await slots())[9] === '0,1:spin:1', (await slots()).slice(9, 11).join(' '));
await press(W, 'one persistent block per tile row');
await select(W, 'Blocks start', 'reversed');
await press(W, 'Run to the end');
sch = schedule(4, 2, 'rows', 'reversed');
check('one block per row, started in reverse on 2 SMs: stuck, and the page says so',
  sch.stuck && (await wout('makespan')) === 'never finishes' && !!(await page.$(W + ' [data-stuck]')) && (await slots()).every((x) => x.includes(':spin:')), JSON.stringify(await slots()));
if (out) await page.screenshot({ path: out + '-wait-stuck.png', clip: await page.$eval(W, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y + window.scrollY, width: r.width, height: r.height }; }) });
await select(W, 'Blocks start', 'index');
await press(W, 'Run to the end');
check('started in index order it completes, in 9 steps', (await wout('makespan')) === `${schedule(4, 2, 'rows', 'index').makespan} steps` && !(await page.$(W + ' [data-stuck]')));
await select(W, 'SMs', 4);
check('4 SMs: relaunch and flags both 7 steps', (await wout('front')) === '7 steps' && (await wout('counter')) === '7 steps' && frontMakespan(4, 4) === 7);
await select(W, 'SMs', 1);
await select(W, 'Tiles per side', 6);
check('1 SM, 6 x 6 tiles: both 36 steps', (await wout('front')) === '36 steps' && (await wout('counter')) === '36 steps');
await select(W, 'SMs', 3);
await select(W, 'Tiles per side', 8);
await press(W, 'flags, tiles from a counter');
check(`3 SMs, 8 x 8: flags ${schedule(8, 3, 'counter').makespan} steps against ${frontMakespan(8, 3)}`,
  (await wout('front')) === `${frontMakespan(8, 3)} steps` && (await wout('counter')) === `${schedule(8, 3, 'counter').makespan} steps`);

t = await answerNumeric('The same 4 × 4 tiles on 4 SMs', 0);
check('q-enough-sms: nothing saved', t.includes('Correct.') && t.includes('2T − 1'), t.slice(0, 300));
t = await answerChoice('one row of tiles by blockIdx.x', 2);
check('q-row-owner: "blocks start in blockIdx order" is wrong', t.includes('Not quite.') && t.includes('any order'), t.slice(0, 300));
t = await answerChoice('one row of tiles by blockIdx.x', 0);
check('q-row-owner: nothing promises the row above has started; HIP unverified, Metal relaunch', t.includes('Correct.') && t.includes('unverified') && t.includes('the relaunch stays'), t.slice(0, 600));
t = await answerChoice('CTC loss gives each block whole sequences', 0);
check('q-ctc: the dependency stays inside one block', t.includes('Correct.') && t.includes('dependency order'), t.slice(0, 400));

// ================================================================ notes, rungs table, saving
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
for (const [pp, what] of [['pp. 374–376', '§16.2'], ['pp. 376–377', '§16.3'], ['pp. 379–380', 'the in-place argument'], ['pp. 397–399', '§16.8']])
  check(`a Source note cites ${what} (${pp})`, notes.some((n) => n.startsWith('Book') && n.includes(pp)), notes.filter((n) => n.startsWith('Book')).join(' | '));
check('three Today notes, each dated', notes.filter((n) => n.startsWith('Today') && n.includes('2026-11-20')).length === 3);
check('the recursion note names cudaLimitStackSize, hipLimitStackSize and maxCallStackDepth', notes.some((n) => n.includes('cudaLimitStackSize') && n.includes('hipLimitStackSize') && n.includes('maxCallStackDepth')));
check('the CTC bridge is marked as not in the book', notes.some((n) => n.startsWith('Not in the book') && n.includes('D2')));
check('three Simplification notes, one per interactive', notes.filter((n) => n.startsWith('What this model leaves out')).length === 3);
const rungs = await page.$$eval('.prose > table.data tbody tr', (r) => r.map((x) => [...x.querySelectorAll('td')].map((c) => c.textContent)));
check('the backend table has five rungs, Metal has no grid barrier and no lane mask', rungs.length === 5 && rungs[1][3] === 'none' && rungs[4][3].includes('no mask') && rungs[4][2].includes('64-bit'), JSON.stringify(rungs));
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('ten questions and two predictions on the page', labels.filter((l) => l === 'Predict first').length === 2 && labels.length === 12, labels.join('|'));
await pause(700);
const last = saved.at(-1) ?? '';
const ids = ['q-topdown', 'q-h264', 'q-live', 'q-fw-changes', 'q-fw-inplace', 'q-infinity', 'q-ex1', 'q-enough-sms', 'q-row-owner', 'q-ctc', 'p-sw8', 'p-flags-2sm'];
check('every answer and prediction was saved through PUT /api/progress (intercepted)', ids.every((id) => last.includes('ch16/16.3/' + id)), ids.filter((id) => !last.includes('ch16/16.3/' + id)).join(' '));

// ================================================================ themes and phone width
for (const theme of ['light', 'dark']) {
  await page.evaluate((th) => { document.documentElement.dataset.theme = th; }, theme);
  await pause(150);
  if (out) for (const [sel, name] of [[P, 'explorer'], [F, 'stepper'], [W, 'wait']]) {
    await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
    await pause(100);
    await page.screenshot({ path: `${out}-${name}-${theme}.png` });
  }
}
await page.setViewport({ width: 390, height: 860, deviceScaleFactor: 1 });
await pause(400);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('no horizontal page scroll at 390 px', overflow <= 0, String(overflow));
const fits = await page.evaluate((sels) => sels.map((s) => document.querySelector(s).getBoundingClientRect().right <= window.innerWidth + 1), [P, F, W]);
check('all three interactives fit the screen at 390 px', fits.every(Boolean), JSON.stringify(fits));
if (out) for (const [sel, name] of [[P, 'explorer'], [F, 'stepper'], [W, 'wait']]) {
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}-narrow.png` });
}

const expected404 = (u) => /\/content\/ch16\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
