// Browser check for section 21.2 (scatter, gather, coarsening, coalescing): the five
// listings, the three predictions, the explorer in its scatter-and-gather and coarsened
// forms (with the erratum gate), the operation table, the write-statement stepper, and the
// questions. Expected values come from src/sim/dcs.ts, imported inside the page through the
// app's module server. Progress is neither read from nor written to the real file.
// Usage: node e2e/ch21-s212.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1,"sections":{},"answers":{},"review":{}}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const E1 = bench('Who writes where: scatter and gather');
const E2 = bench('Who writes where: coarsened kernels');
const WS = bench('Write statements, one warp at a time');
const data = (sel) => page.$eval(sel, (e) => ({ ...e.dataset }));
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(120);
  return ok;
};
const setRange = async (scope, aria, v) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const setSelect = async (scope, label, v) => {
  await page.evaluate((scope, label, v) => {
    const s = [...document.querySelectorAll(scope + ' label.field')].find((l) => l.textContent.trim().startsWith(label)).querySelector('select');
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(v));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, v);
  await pause(150);
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerChoice = async (needle, index) => {
  const q = await question(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
const predict = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await question(needle)).evaluate((e) => e.textContent);
};
const model = (fn, arg) => page.evaluate(async (src, arg) => {
  const m = await import('/src/sim/dcs.ts');
  return new Function('m', 'arg', 'return (' + src + ')(m, arg)')(m, arg);
}, fn.toString(), arg);

await goto(page, 'ch21/21.2');
await page.waitForSelector(E1);

// ---- listings
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('five listings: scatter, gather, Fig. 21.8 as printed, corrected, Fig. 21.10',
  ['scatter (Fig. 21.5)', 'gather (Fig. 21.6)', 'Fig. 21.8 as printed: do not use', 'Fig. 21.8, corrected', 'blockDim.x stride (Fig. 21.10)'].every((t) => tabs.includes(t)), tabs.join(' | '));
const listings = await page.$$eval('.code pre', (p) => p.map((x) => x.textContent));
check('the scatter listing keeps the atomicAdd and adds the bound', listings.some((l) => l.includes('atomicAdd(&energygrid[row + i]') && l.includes('if (n >= numatoms * 4) return;')));
check('the printed listing has 34 numbered lines with line 05 and 30 to 33 marked',
  await page.evaluate(() => {
    const pre = [...document.querySelectorAll('.code')].find((c) => c.textContent.includes('as printed: do not use'));
    const lines = [...pre.querySelectorAll('.ln')];
    const hl = lines.filter((l) => l.classList.contains('hl')).map((l) => Number(l.dataset.n));
    return lines.length === 34 && JSON.stringify(hl) === JSON.stringify([5, 30, 31, 32, 33]) && lines[4].textContent.includes('blockDim.x*COARSEN_FACTOR + threadIdx.x');
  }));
check('every corrected listing carries the verification status',
  (await page.$$eval('.code .tabs .status', (s) => s.map((x) => x.textContent))).filter((s) => s.startsWith('front-end-checked; emulator-checked (python3 labs/ch21/snippets/check.py)')).length === 4);

// ---- predict: lockstep
let t = await predict('how many distinct grid addresses', 1);
check('p-lockstep reveal: one address, and it was what you had', t.includes('You predicted 1') && t.includes('One.') && t.includes('That is what you had.'), t.slice(0, 200));

// ---- explorer 1: scatter, then gather
const M1 = E1 + ' [data-view="mapping"]';
const exp = await model((m) => {
  const atoms = m.seededAtoms(5, 16, 4, 1);
  const run = (kernel, over = {}) => m.mapKernel({ kernel, W: 16, H: 4, atoms, blockX: 4, blockY: 2, F: 4, ...over });
  const sc = run('scatter');
  const race = run('scatter', { atomic: false });
  const ga = run('gather');
  const wide = m.mapKernel({ kernel: 'gather', W: 24, H: 4, atoms: m.seededAtoms(5, 24, 4, 1), blockX: 4, blockY: 2, F: 4 });
  return {
    sc: { verdict: sc.verdict, max: Math.max(...sc.writers.map((w) => w.length)), const: sc.constAddrsPerWarp, result: m.f32text(sc.potentials[3]), direct: sc.direct[3].toPrecision(9) },
    race: { verdict: race.verdict, wrong: race.wrong },
    ga: { verdict: ga.verdict, const: ga.constAddrsPerWarp, result: m.f32text(ga.potentials[3]), threads: ga.launch.threads },
    wide: { result: m.f32text(wide.potentials[21]) },
  };
});
let d = await data(M1);
check('scatter: atomicAdd needed, every point has as many writers as atoms (5), 4 constant addresses (one per lane)',
  d.kernel === 'scatter' && d.verdict === exp.sc.verdict && d.verdict === 'atomic' && Number(d.writersMax) === 5 && Number(d.writersMin) === 5 && Number(d.const) === exp.sc.const && exp.sc.const === 4, JSON.stringify(d));
check('scatter: point (3, 0) shows the model\'s float32 result and float64 sum',
  (await text(M1 + ' [data-readout="result"]')) === exp.sc.result && (await text(M1 + ' [data-readout="direct"]')) === exp.sc.direct);
check('badges: atomicAdd needed, result equals the direct sum', (await text(M1 + ' [data-badges]')).includes('atomicAdd needed: every point has 5 writers') && (await text(M1 + ' [data-badges]')).includes('result equals the direct sum'));
await page.evaluate((sel) => document.querySelector(sel + ' [data-atom="2"]').dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 1 })), M1);
await page.evaluate((sel) => document.querySelector(sel + ' [data-atom="2"]').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 })), M1);
await pause(150);
t = await text(M1 + ' [data-thread-text]');
check('clicking atom 2 selects its thread and shows n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = 8', t.includes('n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (0·4 + 2)·4 = 8: atom 2'), t);
await page.$eval(M1 + ' input[type=checkbox]', (e) => e.click());
await pause(150);
d = await data(M1);
check('plain += instead of atomicAdd: a data race and wrong results', d.verdict === 'race' && d.correct === 'no' && exp.race.wrong === 64 && (await text(M1 + ' [data-badges]')).includes('result wrong at 64 points'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scatter-race.png' });
await press(M1, 'gather (21.6)');
d = await data(M1);
check('gather: one writer per point, one constant address per warp instruction, correct',
  d.verdict === 'one-writer' && Number(d.writersMax) === 1 && Number(d.const) === exp.ga.const && d.correct === 'yes', JSON.stringify(d));
check('gather: the same point\'s float32 result from its own order', (await text(M1 + ' [data-readout="result"]')) === exp.ga.result);
await page.$eval(M1 + ' [data-point="21"]', (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(120);
await press(M1, '(1,0|1,1)');
t = await text(M1 + ' [data-thread-text]');
check('clicking point (5, 1) lists its one writer; its index arithmetic is i = 1·4 + 1 = 5, j = 0·2 + 1 = 1',
  t.includes('block (1, 0), thread (1, 1)') && t.includes('i = blockIdx.x·blockDim.x + threadIdx.x = 1·4 + 1 = 5') && t.includes('= 1'), t);
await setRange(M1, 'slice width', 24);
check('a wider slice recomputes the mapping: 96 points, and the selected point (now (21, 0)) shows the model\'s sum', (await page.$$(M1 + ' [data-point]')).length === 96 && (await text(M1 + ' [data-readout="result"]')) === exp.wide.result);
// drag atom 0 to the right: the potentials change, the gather result stays the direct sum
const before = await text(M1 + ' [data-readout="result"]');
await page.$eval(M1 + ' svg', (e) => e.scrollIntoView({ block: 'center' }));
await pause(150);
const c0 = await page.$eval(M1 + ' [data-atom="0"] circle', (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
await page.mouse.move(c0.x, c0.y);
await page.mouse.down();
for (let k = 1; k <= 6; k++) await page.mouse.move(c0.x + k * 14, c0.y + k * 4);
await page.mouse.up();
await pause(200);
const c1 = await page.$eval(M1 + ' [data-atom="0"] circle', (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
check('dragging an atom moves it and changes the sums; gather still equals the direct sum',
  c1.x - c0.x > 40 && (await text(M1 + ' [data-readout="result"]')) !== before && (await data(M1)).correct === 'yes', JSON.stringify({ c0, c1 }));
await press(M1, 'New random atoms');
check('new random atoms: still one writer per point', (await data(M1)).verdict === 'one-writer');
if (out) await page.screenshot({ path: out + '-gather.png' });

// ---- questions around gather
t = await answerChoice('Why does it need no atomic operation', 0);
check('q-why-no-atomic: the right option is accepted', t.includes('Correct.'));
t = await answerChoice('Why accept that?', 1);
check('q-dilemma: a wrong option is marked and explained', t.includes('Not quite.') && t.includes('trade-off'), t.slice(-300));
t = await answerChoice('raises an error on an MPS tensor', 0);
check('q-deterministic: the right option is accepted', t.includes('Correct.'));

// ---- coarsening: prediction, the erratum gate, the coarsened explorer
t = await predict('how many constant-memory loads does one thread make per atom', 4);
check('p-loads reveal: four loads', t.includes('Four:') && t.includes('That is what you had.'));
check('before the erratum question is answered, the coarsened explorer is held back', !!(await page.$('[data-gate="q-printed"]')) && !(await page.$(E2)));
t = await answerChoice('which of its 128 points does block 0 write', 0);
check('q-printed: points 0 to 34 is right', t.includes('Correct.'));
await page.waitForSelector(E2);
const M2 = E2 + ' [data-view="mapping"]';
const exp2 = await model((m) => {
  const atoms = m.seededAtoms(5, 16, 4, 1);
  const run = (kernel, over = {}) => m.mapKernel({ kernel, W: 16, H: 4, atoms, blockX: 4, blockY: 2, F: 4, ...over });
  const pr = run('printed');
  const pr32 = m.mapKernel({ kernel: 'printed', W: 32, H: 4, atoms: m.seededAtoms(5, 32, 4, 1), blockX: 8, blockY: 2, F: 4 });
  return {
    printed: { verdict: pr.verdict, unwritten: pr.unwritten, conflicts: pr.conflicts, max: Math.max(...pr.writers.map((w) => w.length)) },
    pr32: { unwritten: pr32.unwritten, conflicts: pr32.conflicts },
    adjacent: run('adjacent').verdict, strided: run('strided').verdict,
    g4: m.opsGather(4), c4: m.opsCoarsened(4), a4: [m.arithmetic(m.opsGather(4)), m.arithmetic(m.opsCoarsened(4), true), m.arithmetic(m.opsCoarsened(4))],
    g8: m.opsGather(8), c8: m.opsCoarsened(8),
  };
});
d = await data(M2);
check('the printed line 05: a data race, points never written, up to 4 writers per point',
  d.kernel === 'printed' && d.verdict === 'race' && Number(d.unwritten) === exp2.printed.unwritten && exp2.printed.unwritten > 0 && Number(d.writersMax) === 4 && Number(d.conflicts) === exp2.printed.conflicts, JSON.stringify(d));
check('hatched points are drawn for the unwritten ones', (await page.$$(M2 + ' [data-writers="0"]')).length === exp2.printed.unwritten);
await setRange(M2, 'slice width', 32);
await setSelect(M2, 'blockDim.x', 8);
d = await data(M2);
check('printed, 32 points a row, blocks of 8: block 0 covers 0 to 10 of its 32', Number(d.unwritten) === exp2.pr32.unwritten && exp2.pr32.unwritten === 4 * 21, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-printed.png' });
await press(M2, 'coarsened, adjacent (21.8 fixed)');
d = await data(M2);
check('corrected adjacent ownership: one writer per point and the direct sum', d.verdict === 'one-writer' && d.correct === 'yes' && Number(d.unwritten) === 0);
await press(M2, 'coarsened, stride (21.10)');
d = await data(M2);
check('blockDim.x stride: one writer per point and the direct sum', d.verdict === 'one-writer' && d.correct === 'yes');
const opRow = (op) => page.$eval(`${M2} [data-op="${op}"]`, (r) => [r.querySelector('[data-g]').textContent.trim(), r.querySelector('[data-c]').textContent.trim()]);
let lo = await opRow('loads');
let tot = await opRow('total');
check('operation table at F = 4: 16 loads against 4; 40 operations against 25 (27 as written)',
  lo[0] === String(exp2.g4.loads) && lo[1] === String(exp2.c4.loads) && tot[0] === String(exp2.a4[0]) && tot[1] === `${exp2.a4[1]} (${exp2.a4[2]} as written)` && exp2.a4.join() === '40,25,27', JSON.stringify({ lo, tot }));
check('the book\'s printed figures sit beside them: 48 and 24', (await text(M2 + ' [data-op="total"]')).includes('48') && (await text(M2 + ' [data-op="total"]')).includes('24'));
await setRange(M2, 'points per coarsened thread', 8);
lo = await opRow('loads');
const mul = await opRow('mul');
const dv = await opRow('div');
const sq = await opRow('sqrt');
check('Exercise 2 at F = 8: 32 loads against 4, multiplications 10 (16 as written), 8 divisions and 8 roots on both sides',
  lo.join() === '32,4' && mul[1] === `${exp2.c8.mul} (${exp2.c8.mulUnhoisted} as written)` && exp2.c8.mul === 10 && dv.join() === '8,8' && sq.join() === '8,8', JSON.stringify({ lo, mul, dv, sq }));
if (out) await page.screenshot({ path: out + '-ops.png' });
t = await answerChoice('Which per-point work stays exactly as it was?', 0);
check('q-ex2: one division and one square root per point', t.includes('Correct.'));
t = await answerChoice('You raise the coarsening factor from 4 to 16', 2);
check('q-ex3: a wrong option is explained', t.includes('Not quite.') && t.includes('blockDim.x'));

// ---- coalescing: prediction and the write-statement stepper
t = await predict('How many 32-byte sectors does that one statement touch?', 16);
check('p-adjacent-sectors reveal: 16, and 64 for the four statements', t.includes('16.') && t.includes('64 sectors') && t.includes('That is what you had.'));
const W1 = WS + ' [data-view="writes"]';
const ws = await model((m) => {
  const run = (o, over = {}) => m.writeStatements({ ownership: o, F: 4, warpWidth: 32, blockX: 128, blockY: 1, rowPitch: 1024, segmentBytes: 32, ...over });
  return {
    adj: run('adjacent').statements.map((s) => s.access.transactions), adjTot: run('adjacent').transactions,
    str: run('strided').statements.map((s) => s.access.transactions), strTot: run('strided').transactions,
    f4: run('float4').transactions,
    wave: run('strided', { warpWidth: 64, blockX: 32, blockY: 2, segmentBytes: 128 }).statements.map((s) => s.runs),
    wave128: run('strided', { warpWidth: 64, blockX: 128, segmentBytes: 128 }).statements.map((s) => s.runs),
  };
});
d = await data(W1);
check('adjacent ownership: statement 1 touches 16 sectors; all four 64',
  Number(d.statement) === 0 && Number(d.transactions) === ws.adj[0] && ws.adj[0] === 16 && Number(d.total) === ws.adjTot && ws.adjTot === 64, JSON.stringify(d));
check('the readout says 16 sectors and 25 % used', (await text(W1 + ' [data-readout="statement"]')).includes('16 sectors') && (await text(W1 + ' [data-readout="statement"]')).includes('(25 %)'));
check('the strip outlines exactly the 16 sectors this statement touches', (await page.$$(W1 + ' [data-seg="on"]')).length === 16);
await press(W1, 'Next statement');
d = await data(W1);
check('next statement: statement 2, 32 sectors so far', Number(d.statement) === 1 && Number(d.sofar) === 32);
await press(W1, 'Play');
await pause(2400);
d = await data(W1);
check('play runs to the last statement: 64 sectors so far', Number(d.statement) === 3 && Number(d.sofar) === 64, JSON.stringify(d));
await press(W1, 'blockDim.x stride (Fig. 21.10)');
d = await data(W1);
check('blockDim.x stride: 4 sectors per statement, 16 in all', Number(d.transactions) === ws.str[0] && ws.str[0] === 4 && Number(d.total) === ws.strTot && ws.strTot === 16, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-stride.png' });
await press(W1, 'one float4 store (Exercise 4)');
d = await data(W1);
check('one float4 store: 16 sectors in one request', Number(d.statements) === 1 && Number(d.total) === ws.f4 && ws.f4 === 16, JSON.stringify(d));
await press(W1, 'blockDim.x stride (Fig. 21.10)');
await setSelect(W1, 'Group width', '64');
await setSelect(W1, 'blockDim.x', 32);
await setSelect(W1, 'blockDim.y', 2);
await setSelect(W1, 'Memory unit', 'amd128');
d = await data(W1);
check('64-lane wavefront, blockDim.x = 32: each statement is two runs, two rows of the block',
  Number(d.runs) === ws.wave[0] && ws.wave[0] === 2 && Number(d.width) === 64 && Number(d.segment) === 128 && (await page.$$(W1 + ' [data-strip]')).length === 2, JSON.stringify(d));
await setSelect(W1, 'blockDim.x', 128);
check('with blockDim.x = 128 it is one run again', Number((await data(W1)).runs) === ws.wave128[0] && ws.wave128[0] === 1);
const appleOpt = await page.evaluate((sel) => [...document.querySelectorAll(sel + ' option')].find((o) => o.value === 'apple').textContent, W1);
check('the Apple option names a width recorded on this Mac, or says none is recorded', /Apple: \d+ lanes, from lab [\d.]+ on this Mac/.test(appleOpt) || appleOpt.includes('not recorded here yet'), appleOpt);
await setSelect(W1, 'Memory unit', 'metal');
check('Apple\'s memory unit is a labelled choice', (await text(W1)).includes('Segment (your choice)'));

// ---- the remaining questions
t = await answerChoice('Is Fig. 21.10\'s stride still right there?', 0);
check('q-wavefront: right option accepted', t.includes('Correct.'));
t = await answerChoice('[[threads_per_threadgroup]]', 1);
check('q-metal-edge: dispatchThreadgroups distractor explained', t.includes('Not quite.') && t.includes('dispatchThreadgroups'));
t = await answerChoice('What must be true of the slice?', 0);
check('q-ex4: 16-byte row alignment accepted', t.includes('Correct.'));
check('the page links to lab 21.1 and to section 15.5', !!(await page.$('a[href="#/ch21/labs"]')) && !!(await page.$('a[href="#/ch15/15.5"]')));

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
if (out) {
  await page.$eval(E2, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-light.png' });
}
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 400 px the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(E1, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (e) => /content\/ch21\/narration\/(manifest\.json|[\w-]+\.mp3)/.test(e) || /status of 404/.test(e);
const real = errors.filter((e) => !expected404(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0, real.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
