// Browser check for section 4.4, "Warps and SIMD hardware": the prediction about thread
// (3, 5) of an 8 x 8 block, the warp painter (presets, widths, hover, caps, the Metal
// probe), the stepper "From index to lane", the stepper "One fetch, many lanes", and the
// questions. Everything the page shows is compared with the models it runs on
// (src/sim/warps.ts and src/sim/divergence.ts, imported here directly).
// Usage: node e2e/ch04-s44.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { STRAIGHT_LINE, inputA, inputB } from '../src/sim/divergence.ts';
import { coordOf, cut, dims, laneOf, linearWorking, place, threadsIn, warpOf } from '../src/sim/warps.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so the
// prediction is uncommitted whatever the reader has already done. Reports are replaced
// too: first by none at all, then by made-up ones (clearly named as fixtures) to drive
// the places that read the reader's own devices and lab 4.1's Metal probe.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const probeRow = (tx, ty, groups) => ({ name: 'probe', threadgroup_x: tx, threadgroup_y: ty, width_in_kernel: 32, threadExecutionWidth: 32, simdgroups_per_threadgroup: groups, consecutive_runs: true });
const FIXTURE = {
  cuda: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'cuda', available: true, name: 'e2e fixture device', warpSize: '32', maxThreadsPerBlock: '1024' },
    results: [],
    work_state: {},
  },
  hip: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'hip', available: true, name: 'e2e fixture wide device', warpSize: '64', maxThreadsPerBlock: '256' },
    results: [],
    work_state: {},
  },
  metal: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'metal', available: true, name: 'e2e fixture Metal device' },
    results: [{
      lab: 'ch04/lab1_launchconfig', variant: 'solution', backend: 'metal', tile: 16, built: true, test_rc: 0,
      tests: [{ case: 'fixture', pass: true }],
      bench: [
        { name: 'queried', device: 'e2e fixture Metal device', threadExecutionWidth: 32, maxTotalThreadsPerThreadgroup: 512, probe_threadExecutionWidth: 32, probe_maxTotalThreadsPerThreadgroup: 512 },
        probeRow(16, 16, 8),
        probeRow(8, 8, 2),
      ],
    }],
    work_state: {},
  },
};
let fixture = false;
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  const u = r.url();
  if (u.includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  if (u.includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture ? FIXTURE : {}) });
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (label) => `section.bench[aria-label="${label}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
/** Press the button with exactly this text inside `scope` (optionally inside the group with this aria-label). */
const press = async (scope, label, group) => {
  const done = await page.evaluate((scope, label, group) => {
    const root = document.querySelector(group ? `${scope} [role="group"][aria-label="${group}"]` : scope);
    const btn = root && [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label, group ?? null);
  await pause(80);
  return done;
};
const disabled = (scope, label) => page.evaluate((scope, label) => [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label)?.disabled ?? null, scope, label);
const setInput = async (scope, aria, value) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(90);
};
const setSelect = async (scope, aria, value) => {
  await page.$eval(`${scope} select[aria-label="${aria}"]`, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(120);
};
const clickSvg = async (sel) => {
  await page.$eval(sel, (el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(80);
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const theme = async (t) => {
  await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, t);
  await pause(120);
};
const shot = async (sel, name) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

const P = bench('Warp painter');
const S = bench('From index to lane');
const F = bench('One fetch, many lanes');

await goto(page, 'ch04/4.4');
await page.waitForSelector(P);
await page.waitForSelector(F);

// ---------------------------------------------------------------- the page
check('every simulation says what it leaves out', (await page.$$('.note.caveat')).length >= 2, String((await page.$$('.note.caveat')).length));
check('the page marks what is not in the book', (await page.$$('.note.beyond')).length >= 3, String((await page.$$('.note.beyond')).length));
check('vendor facts are dated', (await page.$$('.note.today')).length >= 4, String((await page.$$('.note.today')).length));
const bridge = await page.$$eval('table.data tbody tr', (trs) => trs.map((tr) => tr.textContent));
check('the table of code you will read names the four places the width appears', bridge.length === 4 && ['C10_WARP_SIZE', 'ggml_cuda_get_physical_warp_size', 'N_SIMDWIDTH 32', 'SIMD_SIZE = 32'].every((s) => bridge.some((r) => r.includes(s))), JSON.stringify(bridge));

// ---------------------------------------------------------------- the prediction
const PID = 'ch04/4.4/p-warp-of-thread';
const predictQ = () => question('Commit before you open the painter');
let q = await predictQ();
check('the prediction shows no answer before it is committed', !(await q.evaluate((e, id) => !!e.querySelector(`[data-reveal="${id}"]`), PID)));
await (await q.$('input[type=number]')).type('1');
await (await q.$('button.primary')).click();
await pause(250);
q = await predictQ();
const at = place(3, 5, 0, dims(8, 8), 32);
const reveal = await q.evaluate((e, id) => e.querySelector(`[data-reveal="${id}"]`)?.textContent.replace(/\s+/g, ' ').trim() ?? '', PID);
check('the reveal confirms warp 1 with the arithmetic: index 43, lane 11', at.warp === 1 && at.index === 43 && at.lane === 11 && reveal.startsWith('Yes:') && reveal.includes(`the linear index is ${linearWorking(3, 5, 0, dims(8, 8))}`) && reveal.includes('lane 43 mod 32 = 11'), reveal);
const marked = await q.evaluate((e) => [...e.querySelectorAll('svg rect[data-thread]')].filter((r) => r.getAttribute('stroke-width') === '2.5').map((r) => [r.dataset.thread, r.dataset.warp]));
check('and marks that thread in a drawing of the block', eq(marked, [['3,5,0', '1']]), JSON.stringify(marked));
await pause(600); // the progress file is written after a short delay
const stored = saved.map((s) => { try { return JSON.parse(s).answers?.[PID]?.value; } catch { return undefined; } }).filter((v) => v !== undefined).pop();
check('the prediction is saved under ch04/4.4/p-warp-of-thread', stored === 1, JSON.stringify(stored));

// ---------------------------------------------------------------- the warp painter
const readout = async () => {
  const o = {};
  for (const n of ['threads', 'warps', 'padded', 'rows']) o[n] = await text(`${P} [data-readout="cut"] [data-out="${n}"]`);
  return o;
};
/** Every thread cell of the block drawing: [x, y, z] and the warp and lane it shows. */
const blockCells = (scope) => page.$$eval(`${scope} [data-view="block"] rect[data-thread]`, (rs) => rs.map((r) => ({ t: r.dataset.thread.split(',').map(Number), warp: Number(r.dataset.warp), lane: Number(r.dataset.lane), index: Number(r.dataset.index), painted: r.hasAttribute('data-painted') })));
const warpRows = (scope) => page.$$eval(`${scope} svg[data-view="warps"]`, (svgs) => {
  const s = svgs[0];
  return { real: s.querySelectorAll('rect[data-index]').length, pad: s.querySelectorAll('rect[data-pad]').length, rows: s.querySelectorAll('text').length - 2 };
});
const agreesWithModel = (cells, d, width) => cells.length === threadsIn(d) && cells.every((c) => {
  const p = place(c.t[0], c.t[1], c.t[2], d, width);
  return p.warp === c.warp && p.lane === c.lane && p.index === c.index;
});
const paints = async (d, width, why) => {
  const k = cut(d, width);
  const o = await readout();
  const cells = await blockCells(P);
  const wr = await warpRows(P);
  check(why, o.threads === String(k.threads) && o.warps === String(k.warps) && o.padded === String(k.padded) && agreesWithModel(cells, d, width) && wr.real === k.threads && wr.pad === k.padded && wr.rows === k.warps,
    JSON.stringify({ o, cells: cells.length, wr, model: k }));
  return o;
};
const preset = (label) => press(P, label, 'Presets');

let o = await paints(dims(16, 16), 32, 'the painter opens on a 16 × 16 block, not the prediction\'s 8 × 8, at the book\'s 32: 8 warps, every cell where the model puts it');
check('so the drawing under the prediction does not show its answer: no 8 × 8 block is drawn', (await blockCells(P)).length === 256 && !(await page.$(`${P} [data-view="block"] rect[data-thread="3,5,0"][data-warp="1"]`)));
await preset('8 × 8');
o = await paints(dims(8, 8), 32, 'the 8 × 8 preset at the book\'s 32: 2 warps, every cell where the model puts it');
check('a warp of it covers 4 rows', o.rows === 'one warp covers 4 rows of the block', o.rows);
const w0 = (await blockCells(P)).filter((c) => c.warp === 0).map((c) => c.t[1]);
check('warp 0 is rows 0 to 3 and warp 1 rows 4 to 7, as on p. 76', eq([...new Set(w0)], [0, 1, 2, 3]));
check('nothing is described until a thread is pointed at', (await text(`${P} [data-readout="thread"]`)) === 'Point at a thread.');
await clickSvg(`${P} [data-view="block"] rect[data-thread="3,5,0"]`);
const said = await text(`${P} [data-readout="thread"]`);
check('pointing at thread (3, 5) gives its arithmetic', said === 'Thread (x 3, y 5): linear index 3 + 5·8 = 43. Warp 43 / 32 rounded down = 1; lane 43 mod 32 = 11.', said);
await clickSvg(`${P} svg[data-view="warps"] rect[data-index="33"]`);
check('pointing at a lane in the warp rows names the same thread', (await text(`${P} [data-readout="thread"]`)).startsWith('Thread (x 1, y 4): linear index 1 + 4·8 = 33.'), await text(`${P} [data-readout="thread"]`));
await shot(P, 'painter-8x8');

await preset('4 × 4');
o = await paints(dims(4, 4), 32, '4 × 4 is half a warp: one warp with 16 idle lanes (Fig. 4.7)');
const strip = (await blockCells(P)).sort((a, b) => a.index - b.index).slice(0, 5).map((c) => `${c.t[1]},${c.t[0]}`);
check('its linear order is the strip of Fig. 4.7: T(0,0) T(0,1) T(0,2) T(0,3) T(1,0)', eq(strip, ['0,0', '0,1', '0,2', '0,3', '1,0']), JSON.stringify(strip));
await shot(P, 'painter-4x4');
await preset('16 × 16');
o = await paints(dims(16, 16), 32, '16 × 16: 8 warps');
check('two rows per warp at 32', o.rows === 'one warp covers 2 rows of the block', o.rows);
await preset('4 × 8 × 2');
o = await paints(dims(4, 8, 2), 32, 'x 4, y 8, z 2 (the book\'s 2 × 8 × 4): 2 warps');
const byZ = (await blockCells(P)).reduce((m, c) => ({ ...m, [c.t[2]]: [...new Set([...(m[c.t[2]] ?? []), c.warp])] }), {});
check('warp 0 is all of z = 0 and warp 1 all of z = 1, drawn as two panels', eq(byZ, { 0: [0], 1: [1] }) && (await page.$$(`${P} [data-view="block"] svg`)).length === 2, JSON.stringify(byZ));
await clickSvg(`${P} [data-view="block"] rect[data-thread="1,2,1"]`);
check('thread (1, 2, 1) is index 41, warp 1, lane 9', (await text(`${P} [data-readout="thread"]`)) === 'Thread (x 1, y 2, z 1): linear index 1 + 2·4 + 1·4·8 = 41. Warp 41 / 32 rounded down = 1; lane 41 mod 32 = 9.', await text(`${P} [data-readout="thread"]`));
await shot(P, 'painter-3d');
await preset('48 × 1');
o = await paints(dims(48), 32, '48 threads: 2 warps, the second padded with 16 idle lanes (p. 75)');
check('a single row says so', o.rows === 'one row: a warp is a run of consecutive x', o.rows);

await setSelect(P, 'Warp width', 'w64');
o = await paints(dims(48), 64, 'the same 48 threads at 64 wide: one warp, 16 idle lanes');
await preset('16 × 16');
o = await paints(dims(16, 16), 64, '16 × 16 at 64 wide: 4 warps');
check('four rows per warp at 64', o.rows === 'one warp covers 4 rows of the block', o.rows);
await preset('8 × 8');
await paints(dims(8, 8), 64, '8 × 8 at 64 wide: a single warp');
check('thread (0, 4) now shares warp 0 with thread (0, 0)', (await blockCells(P)).find((c) => c.t[0] === 0 && c.t[1] === 4).warp === 0);
await shot(P, 'painter-8x8-64');
await setSelect(P, 'Warp width', 'book');

await setInput(P, 'blockDim.x', 12);
await setInput(P, 'blockDim.y', 8);
o = await paints(dims(12, 8), 32, 'a 12 × 8 block typed in: 3 warps, none padded');
check('its warps cut through rows', o.rows === 'warp boundaries fall inside rows', o.rows);
await setInput(P, 'blockDim.x', 32);
await setInput(P, 'blockDim.y', 2);
o = await readout();
check('a 32 × 2 block: one warp per row', o.rows === 'one warp covers 1 row of the block' && o.warps === '2', JSON.stringify(o));
await setInput(P, 'blockDim.x', 64);
await setInput(P, 'blockDim.y', 2);
o = await readout();
check('a 64 × 2 block: a row is two warps', o.rows === 'one row of the block is 2 warps' && o.warps === '4', JSON.stringify(o));
await setInput(P, 'blockDim.y', 32);
check('a block of 2048 threads is over the book\'s cap of 1024 and nothing is drawn', (await text(`${P} [data-over]`)).includes('2048 threads in a block is more than the 1024') && !(await page.$(`${P} [data-view="block"]`)));
await setInput(P, 'blockDim.x', 0);
check('a zero dimension is refused', (await text(P)).includes('Each dimension of a block is a whole number, 1 or more.'));
await setInput(P, 'blockDim.x', 200);
await setInput(P, 'blockDim.y', 1);
check('a row too long to draw is still cut into warps', (await readout()).warps === '7' && (await text(P)).includes('A row of 200 threads is too long to draw here') && !!(await page.$(`${P} svg[data-view="warps"]`)));

// The Metal choice with no report.
await preset('8 × 8');
await setSelect(P, 'Warp width', 'metal');
check('Metal with no probe result draws nothing and says what to run', (await page.$eval(`${P} [data-metal]`, (e) => e.dataset.metal)) === 'none' && (await text(`${P} [data-metal]`)).includes('python3 labs/pmpp.py run 4.1 --bench') && !(await page.$(`${P} [data-readout="cut"]`)), await text(`${P} [data-metal]`));
await setSelect(P, 'Warp width', 'book');

// ---------------------------------------------------------------- from index to lane
const stepSaid = () => text(`${S} [data-readout="step"]`);
const painted = async () => (await blockCells(S)).filter((c) => c.painted).map((c) => c.index).sort((a, b) => a - b);
check('the stepper uses the painter\'s block and starts empty', (await stepSaid()).startsWith('Nothing placed yet.') && (await painted()).length === 0, await stepSaid());
await press(S, 'Step');
check('step 1 places index 0 in lane 0 of warp 0', (await stepSaid()).startsWith('Index 0: thread (x 0, y 0, z 0), because 0 + 0·8 = 0. It goes to warp 0, lane 0.') && eq(await painted(), [0]), await stepSaid());
for (let i = 0; i < 8; i++) await press(S, 'Step');
const s8 = await stepSaid();
check('index 8 is where x runs out: y goes up by one, worth blockDim.x = 8 positions', s8.startsWith('Index 8: thread (x 0, y 1, z 0), because 0 + 1·8 = 8. It goes to warp 0, lane 8.') && s8.includes('one step of y is worth 8 positions (blockDim.x)'), s8);
await press(S, 'Back');
check('Back takes one index back', (await stepSaid()).startsWith('Index 7: thread (x 7, y 0, z 0)') && (await painted()).length === 8);
await press(S, 'Fill this warp');
const filled = await stepSaid();
check('Fill this warp ends on index 31, the last lane, with rows 0 to 3 painted', filled.startsWith('Index 31: thread (x 7, y 3, z 0)') && filled.includes('That fills warp 0.') && eq(await painted(), [...Array(32).keys()]), filled);
const painted8 = await page.$$eval(`${S} svg[data-view="warps"] rect[data-index]`, (rs) => rs.filter((r) => r.getAttribute('fill') !== 'var(--cell)').length);
check('and the warp rows show 32 lanes filled', painted8 === 32, String(painted8));
await shot(S, 'stepper-8x8');
await press(S, 'Fill this warp');
check('filling the next warp uses up the block', (await stepSaid()).startsWith('Index 63: thread (x 7, y 7, z 0)') && (await stepSaid()).includes('That fills warp 1.') && (await disabled(S, 'Step')) === true);
await press(S, 'Restart');
await press(S, 'Play');
await pause(1200);
const played = (await painted()).length;
await press(S, 'Pause');
check('Play places threads by itself', played >= 3 && played <= 12, String(played));

await preset('4 × 8 × 2');
check('changing the block in the painter restarts the stepper', (await stepSaid()).startsWith('Nothing placed yet.'));
await press(S, 'Fill this warp');
await press(S, 'Step');
const z1 = await stepSaid();
check('in the 3D block, index 32 is where z goes up: worth blockDim.x · blockDim.y = 32 positions', z1.startsWith('Index 32: thread (x 0, y 0, z 1), because 0 + 0·4 + 1·4·8 = 32. It goes to warp 1, lane 0.') && z1.includes('one step of z is worth 32 positions'), z1);
await preset('48 × 1');
await press(S, 'Fill this warp');
await press(S, 'Fill this warp');
const end48 = await stepSaid();
check('48 threads: the walk ends with 16 lanes of the last warp empty', end48.startsWith('Index 47: thread (x 47, y 0, z 0), because 47.') && end48.includes('The block is used up and the last warp has 16 lanes with no thread.'), end48);
await preset('8 × 8');

// ---------------------------------------------------------------- one fetch, many lanes
const panel = async (w) => page.$eval(`${F} .panel[data-warp="${w}"]`, (p) => ({
  fetches: Number(p.querySelector('[data-out="fetches"]').textContent),
  laneOps: Number(p.querySelector('[data-out="laneops"]').textContent),
  next: [...p.querySelectorAll('li')].findIndex((l) => l.hasAttribute('data-pc')),
  regs: Object.fromEntries([...p.querySelectorAll('tr[data-reg]')].map((tr) => [tr.dataset.reg, [...tr.querySelectorAll('td')].slice(1).map((td) => td.textContent.trim())])),
}));
const fetchSaid = () => text(`${F} [data-readout="fetch"]`);
const lanes = (n, f) => Array.from({ length: n }, (_, i) => String(f(i)));
const dots = (n) => Array(n).fill('·');
let a = await panel(0);
check('the fetch stepper starts with two warps of 8 lanes, nothing fetched', a.fetches === 0 && a.next === 0 && eq(a.regs.r2, dots(8)) && (await fetchSaid()).startsWith('Nothing has been fetched.'), JSON.stringify(a));
await press(F, 'Step warp 0');
a = await panel(0);
let b = await panel(1);
check('one step of warp 0: 1 fetch, 8 lane-operations, r2 = a[i] in every lane', a.fetches === 1 && a.laneOps === 8 && eq(a.regs.r2, lanes(8, inputA)) && eq(a.regs.r3, dots(8)) && a.next === 1, JSON.stringify(a));
check('warp 1 has not moved: its own program counter', b.fetches === 0 && b.next === 0 && eq(b.regs.r2, dots(8)));
check('the readout names the one fetch and two lanes\' own values', (await fetchSaid()) === `Processing block 0 fetched "${STRAIGHT_LINE[0].text}" once, and all 8 lanes of warp 0 executed it. Lane 0: r2 = ${inputA(0)}, for its own i = 0. Lane 7: r2 = ${inputA(7)}, for its own i = 7.`, await fetchSaid());
await press(F, 'Step both');
await press(F, 'Step warp 0');
a = await panel(0);
b = await panel(1);
const sum0 = lanes(8, (i) => inputA(i) + inputB(i));
check('after three fetches warp 0 holds r1 = r2 + r3 in every lane, 24 lane-operations', a.fetches === 3 && a.laneOps === 24 && eq(a.regs.r1, sum0) && eq(a.regs.r3, lanes(8, inputB)), JSON.stringify(a));
check('the readout shows the add in lane 0 and lane 7', (await fetchSaid()).includes(`Lane 0: ${inputA(0)} + ${inputB(0)} = ${inputA(0) + inputB(0)}. Lane 7: ${inputA(7)} + ${inputB(7)} = ${inputA(7) + inputB(7)}.`), await fetchSaid());
check('warp 1 is one instruction behind, on its own threads 8 to 15', b.fetches === 1 && eq(b.regs.r2, lanes(8, (i) => inputA(8 + i))), JSON.stringify(b));
check('a finished warp cannot step again', (await disabled(F, 'Step warp 0')) === true && (await disabled(F, 'Step warp 1')) === false);
await shot(F, 'fetch-8');
await setSelect(F, 'Width of a warp', '32');
await press(F, 'Step both');
await press(F, 'Step both');
await press(F, 'Step both');
a = await panel(0);
b = await panel(1);
check('at 32 lanes: still 3 fetches per warp, now 96 lane-operations', a.fetches === 3 && a.laneOps === 96 && b.fetches === 3 && b.laneOps === 96 && eq(b.regs.r1, lanes(32, (i) => inputA(32 + i) + inputB(32 + i))), JSON.stringify({ a: [a.fetches, a.laneOps], b: [b.fetches, b.laneOps] }));
await shot(F, 'fetch-32');
await press(F, 'Restart');
check('Restart clears both warps', (await panel(0)).fetches === 0 && (await panel(1)).fetches === 0);

// ---------------------------------------------------------------- questions
const answerNumber = async (needle, value) => {
  const qq = await question(needle);
  await (await qq.$('input[type=number]')).type(String(value));
  await (await qq.$('button.primary')).click();
  await pause(150);
  return qq.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const qq = await question(needle);
  await (await qq.$$('input[type=radio]'))[index].click();
  await (await qq.$('button.primary')).click();
  await pause(150);
  return qq.evaluate((e) => e.textContent);
};
check('the 3D question accepts warp 1, the model\'s answer', place(1, 2, 1, dims(4, 8, 2), 32).warp === 1 && (await answerNumber('blockDim x 4, y 8, z 2', 1)).includes('Correct.'));
check('exercise 1a and 1b: 32 warps; 16 is rejected', (await answerNumber('exercise 1, parts a and b', 16)).includes('Not quite. The answer is 32 warps.'));
check('48-thread blocks: 2 and 1 warps, 16 idle lanes each', cut(dims(48), 32).padded === 16 && cut(dims(48), 64).padded === 16 && (await answerChoice('48-thread blocks', 2)).includes('Correct.'));
check('rows per warp: a wrong option explains itself', (await answerChoice('How many rows of the block does one warp cover', 3)).includes('Those are the numbers of warps in the block'));
check('the hard-coded 32: correct answer accepted', (await answerChoice('threadIdx.x % 32', 3)).includes('Correct.'));
check('all five questions of the section are on the page', (await page.$$eval('.q', (qs) => qs.filter((x) => x.querySelector('.kind')?.textContent.match(/Check your reasoning|Work it out/)).length)) === 5);

// ---------------------------------------------------------------- with reports from devices
fixture = true;
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(P);
await pause(500);
const widths = await page.$$eval(`${P} select[aria-label="Warp width"] option`, (os) => os.map((x) => [x.value, x.textContent]));
check('the painter offers the book, 64, each reported CUDA or HIP width, and Metal', eq(widths.map((w) => w[0]), ['book', 'w64', 'cuda', 'hip', 'metal']) && widths[3][1] === 'yours: e2e fixture wide device (hip), 64', JSON.stringify(widths));
await setSelect(P, 'Warp width', 'hip');
await preset('16 × 16');
await paints(dims(16, 16), 64, 'the reported 64-wide device cuts 16 × 16 into 4 warps');
await setInput(P, 'blockDim.x', 32);
check('a 32 × 16 block is over that device\'s cap of 256', (await text(`${P} [data-over]`)).includes('512 threads in a block is more than the 256'), await text(P));
await theme('light');
await preset('8 × 8');
await setSelect(P, 'Warp width', 'metal');
const probed = await page.$eval(`${P} [data-metal]`, (e) => ({ state: e.dataset.metal, rows: e.querySelectorAll('tr[data-probe-row]').length, text: e.textContent.replace(/\s+/g, ' ') }));
check('Metal with a probe result lists what the probe recorded', probed.state === 'probed' && probed.rows === 2 && probed.text.includes('e2e fixture Metal device') && probed.text.includes("the reference solution's probe kernel"), JSON.stringify(probed));
check('and draws consecutive runs of the recorded width, for a shape the probe ran', probed.text.includes('applies consecutive runs of 32 to your block, which is one of the shapes the probe ran') && agreesWithModel(await blockCells(P), dims(8, 8), 32));
check('the cap on Metal is the probe pipeline\'s limit as recorded, not a fixed number', (await text(`${P} footer`)).includes("capped at 512 here: the maxTotalThreadsPerThreadgroup of the probe's pipeline"), await text(`${P} footer`));
await shot(P, 'painter-metal-light');
await setInput(P, 'blockDim.x', 32);
await setInput(P, 'blockDim.y', 32);
check('a 32 × 32 threadgroup is over that recorded limit', (await text(`${P} [data-over]`)).includes('1024 threads in a block is more than the 512'));
await preset('16 × 16');
await shot(S, 'stepper-light');
await press(F, 'Step both');
await shot(F, 'fetch-light');

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate((scopes) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  benches: scopes.every((s) => document.querySelector(s).getBoundingClientRect().width <= document.documentElement.clientWidth),
}), [P, S, F]);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.benches, JSON.stringify(narrow));
await shot(P, 'narrow-painter');
await shot(F, 'narrow-fetch');
await theme('dark');
await shot(P, 'narrow-painter-dark');

// ---------------------------------------------------------------- errors
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
