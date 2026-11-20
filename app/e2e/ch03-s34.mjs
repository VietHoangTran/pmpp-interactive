// Browser check for section 3.4, "Matrix multiplication": the hand-trace stepper, the
// narrated scene, the "who feeds this element" explorer, the thread-granularity
// explorer, the predictions and the questions. Everything the page shows is compared
// with the model it runs on (src/sim/naivemm.ts, imported here directly).
// Usage: node e2e/ch03-s34.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  CUDA_DOCUMENTED, blockInputs, counts, granularity, gridFor, makeProblem, makeRect, maxWidth, maxWidthInt32, owners,
  threadFor, trace, traceRect,
} from '../src/sim/naivemm.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so the
// predictions are uncommitted whatever the reader has already done. The second
// half of the check swaps in a made-up report (clearly named as a fixture) to
// drive the two places that read the reader's own device.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const FIXTURE = {
  cuda: {
    generated: '2026-11-20T00:00:00',
    device: {
      kind: 'device', backend: 'cuda', available: true, name: 'e2e fixture device',
      'maxGridSize.x': '2147483647', 'maxGridSize.y': '1000', 'maxGridSize.z': '65535',
      maxThreadsPerBlock: '1024', 'maxThreadsDim.x': '1024', 'maxThreadsDim.y': '1024', 'maxThreadsDim.z': '64',
    },
    results: [{
      lab: 'ch03/lab4_granularity', variant: 'solution', backend: 'cuda', tile: 16, built: true, test_rc: 0,
      tests: [{ case: 'fixture', pass: true }],
      bench: [
        { name: 'granularity', mode: 'element', width: 512, seconds: 0.002 },
        { name: 'granularity', mode: 'row', width: 512, seconds: 0.05 },
        { name: 'granularity', mode: 'col', width: 512, seconds: 0.08 },
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
  if (fixture && u.includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE) });
  return r.continue();
});
// The narration manifest and audio for this chapter may not exist yet; those 404s are expected.
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const big = (v) => v.toLocaleString('en-US');
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const bench = (label) => `section.bench[aria-label="${label}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.trim());
const texts = (sel) => page.$$eval(sel, (es) => es.map((e) => e.textContent.trim()));
/** Press the button with exactly this text inside `scope` (optionally inside the group with this aria-label). */
const press = async (scope, label, group) => {
  const done = await page.evaluate((scope, label, group) => {
    const root = document.querySelector(group ? `${scope} [role="group"][aria-label="${group}"]` : scope);
    const btn = root && [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label, group ?? null);
  await pause(70);
  return done;
};
const pressLabel = async (scope, aria) => {
  await page.$eval(`${scope} button[aria-label="${aria}"]`, (b) => b.click());
  await pause(70);
};
const setRange = async (scope, label, value) => {
  await page.$eval(`${scope} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(90);
};
const lit = async (scope, id) => (await page.$eval(`${scope} svg[data-strip="${id}"]`, (e) => e.dataset.lit)).split(',').filter(Boolean).map(Number);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);

await goto(page, 'ch03/3.4');
const T = bench('Trace one thread by hand');
const F = bench('Who feeds this element');
const G = bench('How much does one thread do?');
const S = 'section.scene';
await page.waitForSelector(T);
await page.waitForSelector(G);

// ---------------------------------------------------------------- listings
const kernelText = await text('figure.code pre');
check('the kernel listing is the book\'s twelve lines', kernelText.includes('MatrixMulKernel') && kernelText.includes('Pvalue += M[row*Width+k]*N[k*Width+col];') && (await page.$$eval('figure.code', (f) => f[0].querySelectorAll('.ln').length)) === 12);
check('the listing says how it was checked', (await text('figure.code .status')).includes('front-end-checked'));
const launchFirst = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('launchMatrixMul'));
  return f ? { n: f.querySelector('.ln').dataset.n, text: f.textContent } : null;
});
check('the launch listing continues the same file from line 14 and rounds the grid up', !!launchFirst && launchFirst.n === '14' && launchFirst.text.includes('(Width + BLOCK_WIDTH - 1) / BLOCK_WIDTH') && launchFirst.text.includes('MatrixMulKernel<<<dimGrid, dimBlock>>>'), JSON.stringify(launchFirst)?.slice(0, 120));

// ---------------------------------------------------------------- the hand trace
const prob = makeProblem(4);
const narrate = () => text(T + ' .narrate');
const step = () => text(T + ' [data-readout="step"]');
const hot = (view, array) => page.$$eval(`${T} [data-view="${view}"][data-array="${array}"] span.hot`, (es) => es.map((e) => Number(e.dataset.i)));
const rows = () => page.$$eval(T + ' table[data-table="trace"] tr[data-k]', (trs) => trs.map((tr) => ({
  k: Number(tr.dataset.k), m: tr.querySelector('[data-col="m-index"]').textContent.trim(), n: tr.querySelector('[data-col="n-index"]').textContent.trim(), sum: Number(tr.querySelector('[data-col="sum"]').textContent),
})));

check('the stepper opens on the book\'s thread with nothing run', (await narrate()).startsWith('Nothing has run yet') && (await step()) === 'step 0 of 9' && (await text(T + ' [data-readout="owner"]')) === 'Thread (0,0) of Block (0,0)', await step());
await press(T, 'Step');
check('step 1 is line 03 with the row arithmetic', (await narrate()).includes('row = blockIdx.y·blockDim.y + threadIdx.y = 0·2 + 0 = 0') && (await text(T + ' .code .ln.hl')).includes('int row'), await narrate());
await press(T, 'Step');
check('step 2 is line 04 with the column arithmetic', (await narrate()).includes('col = blockIdx.x·blockDim.x + threadIdx.x = 0·2 + 0 = 0') && (await text(T + ' .code .ln.hl')).includes('int col'), await narrate());
await press(T, 'Step');
check('step 3 is the guard, and it names the element the thread owns', (await narrate()).includes('this thread owns P[0][0]') && (await text(T + ' .code .ln.hl')).includes('if ((row < Width)'), await narrate());
await press(T, 'Step');
await press(T, 'Step');
const t00 = trace(prob, 0, 0);
let s0 = t00.steps[0];
let n = await narrate();
check('k = 0 evaluates both flat indices and the running sum', n.includes(`M[0·4 + 0] = M[${s0.mIndex}] = ${s0.mValue}`) && n.includes(`N[0·4 + 0] = N[${s0.nIndex}] = ${s0.nValue}`) && n.includes(`= ${s0.sum}.`), n);
check('the loop line is highlighted', (await text(T + ' .code .ln.hl')).includes('Pvalue += M[row*Width+k]*N[k*Width+col];'));
check('the cells read at k = 0 are lit in the 2D view and in the flat strips', eq(await hot('2d', 'M'), [s0.mIndex]) && eq(await hot('2d', 'N'), [s0.nIndex]) && eq(await hot('flat', 'M'), [s0.mIndex]) && eq(await hot('flat', 'N'), [s0.nIndex]));
await press(T, 'Step');
await press(T, 'Step');
const s2 = t00.steps[2];
check('at k = 2 the M marker is at 2 and the N marker at 8', eq(await hot('flat', 'M'), [s2.mIndex]) && eq(await hot('flat', 'N'), [s2.nIndex]) && s2.mIndex === 2 && s2.nIndex === 8, JSON.stringify([await hot('flat', 'M'), await hot('flat', 'N')]));
check('the registers panel shows the running sum', (await text(T + ' [data-readout="registers"]')).includes(`Pvalue = ${s2.sum}`), await text(T + ' [data-readout="registers"]'));
await press(T, 'Back');
check('Back undoes one step', (await step()) === 'step 6 of 9' && eq(await hot('flat', 'N'), [t00.steps[1].nIndex]), await step());
await press(T, 'Run to the end');
let r = await rows();
check('the table is the book\'s: M indices 0, 1, 2, 3 and N indices 0, 4, 8, 12 (p. 63)', eq(r.map((x) => x.m), ['0·4+0 = 0', '0·4+1 = 1', '0·4+2 = 2', '0·4+3 = 3']) && eq(r.map((x) => x.n), ['0·4+0 = 0', '1·4+0 = 4', '2·4+0 = 8', '3·4+0 = 12']), JSON.stringify(r));
check('the running sums in the table are the model\'s', eq(r.map((x) => x.sum), t00.steps.map((x) => x.sum)), JSON.stringify(r.map((x) => x.sum)));
n = await narrate();
check('the last step is the one store, to P[0]', n.includes(`P[0·4 + 0] = P[0] = ${t00.store.value}`) && (await text(T + ' .code .ln.hl')).includes('P[row*Width+col] = Pvalue;'), n);
check('the stored value matches M × N and is drawn in both views of P', (await text(T + ' [data-readout="result"]')) === `P[0] = ${prob.ref[0]} matches M × N` && eq(await page.$$eval(T + ' [data-array="P"] span.good', (es) => es.map((e) => Number(e.dataset.i))), [0, 0]));
if (out) { await (await page.$(T)).screenshot({ path: out + '-trace-book.png' }); }

// Thread (1,0) of Block (0,0): the first prediction's thread.
await press(T, '(1,0)', 'Thread');
check('picking another thread restarts the trace', (await step()) === 'step 0 of 9' && (await text(T + ' [data-readout="owner"]')) === 'Thread (1,0) of Block (0,0)');
await press(T, 'Run to the end');
const rc10 = threadFor(2, { y: 0, x: 0 }, { y: 1, x: 0 });
const t10 = trace(prob, rc10.row, rc10.col);
r = await rows();
check('Thread (1,0) of Block (0,0) reads N[8] and M[6] at k = 2', r[2].n.endsWith('= 8') && r[2].m.endsWith('= 6') && t10.steps[2].nIndex === 8 && t10.steps[2].mIndex === 6, JSON.stringify(r[2]));
check('and it stores to P[4]', (await narrate()).includes(`P[1·4 + 0] = P[4] = ${t10.store.value}`) && t10.store.index === 4, await narrate());
// Thread (0,0) of Block (1,0): the second prediction's thread.
await press(T, '(0,0)', 'Thread');
await press(T, '(1,0)', 'Block');
await press(T, 'Run to the end');
const t20 = trace(prob, 2, 0);
check('Thread (0,0) of Block (1,0) stores to P[8]', (await narrate()).includes(`P[2·4 + 0] = P[8] = ${t20.store.value}`) && eq(await page.$$eval(T + ' [data-view="flat"][data-array="P"] span.good', (es) => es.map((e) => Number(e.dataset.i))), [8]), await narrate());
check('its table walks row 2 of M: indices 8 to 11', eq((await rows()).map((x) => x.m.split('= ')[1]), ['8', '9', '10', '11']));

// ---------------------------------------------------------------- the two numeric predictions
let q = await question('which flat index of');
await (await q.$('input[type=number]')).type('8');
await (await q.$('button.primary')).click();
await pause(150);
let qt = await q.evaluate((e) => e.textContent);
check('the N-index prediction reveals N[8] and M[6]', qt.includes('You predicted 8.') && qt.includes('Yes.') && qt.includes('N[8]') && qt.includes('M[6]'), qt.slice(-300));
q = await question('which flat index of P does it write');
await (await q.$('input[type=number]')).type('7');
await (await q.$('button.primary')).click();
await pause(150);
qt = await q.evaluate((e) => e.textContent);
check('a wrong store prediction is told the index is 8', qt.includes('You predicted 7.') && qt.includes('It is 8.') && qt.includes('P[2][0]'), qt.slice(-300));

// ---------------------------------------------------------------- the narrated scene
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const script = async (k) => strip(await (await fetch(`${BASE}/content/ch03/narration/sixteen-threads-${k}.txt`)).text());
const part = async (k, pos) => {
  await page.evaluate((sel, k) => document.querySelectorAll(sel + ' .seg button')[k - 1].click(), S, k);
  await pause(150);
  if (pos != null) {
    await page.$eval(S + ' input[type=range]', (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }, pos);
    await pause(150);
  }
  return page.$eval(S + ' svg', (e) => [...e.querySelectorAll('text')].map((t) => t.textContent));
};
check('the scene has six parts and a captions switch', (await page.$$(S + ' .seg button')).length === 6 && (await text(S)).includes('captions'));
// With no audio a part shows its script regardless; with audio, captions are an option.
if (!(await page.$(S + ' .caption'))) {
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), S);
  await pause(120);
}
const scripts = await Promise.all([1, 2, 3, 4, 5, 6].map(script));
check('every script is 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(','));
let svg = await part(1, 1000);
check('part 1 shows its script and places all sixteen threads', (await text(S + ' .caption')) === scripts[0] && svg.includes('16 of 16 threads placed, one per element of P') && (await page.$$eval(S + ' svg circle', (c) => c.length)) === 16, svg.slice(-3).join('|'));
svg = await part(1, 500);
const placed = Number(svg.find((x) => x.includes('threads placed')).split(' ')[0]);
const o = owners(2, 0, 0);
check('half-way through part 1 the newest thread\'s row and column sums are written out', placed > 0 && placed < 16 && svg.some((x) => /^row = \d·2 \+ \d = \d$/.test(x)) && svg.some((x) => /^it owns P\[\d\]\[\d\]$/.test(x)) && o.block.y === 0, svg.slice(-6).join('|'));
svg = await part(2, 1000);
check('part 2 ends with the first thread\'s sum', svg.includes(`after 4 steps, Pvalue = ${t00.store.value}`), svg.slice(-3).join('|'));
svg = await part(3, 620);
const kNow = Number(svg.find((x) => x.startsWith('k = ')).slice(4));
check('part 3 draws the flat arrays and the two indices for the current k', svg.includes('M as stored: 16 floats, row after row') && svg.includes(`M[row·4 + k] = M[0·4 + ${kNow}] = M[${t00.steps[kNow].mIndex}]`) && svg.includes(`N[k·4 + col] = N[${kNow}·4 + 0] = N[${t00.steps[kNow].nIndex}]`), svg.slice(-5).join('|'));
svg = await part(3, 1000);
check('part 3 ends on the two index sequences', svg.includes('M: 0, 1, 2, 3.  N: 0, 4, 8, 12.'), svg.slice(-2).join('|'));
svg = await part(4, 1000);
check('part 4 ends with Block (0,0)\'s four elements stored', [0, 1, 4, 5].every((i) => svg.includes(String(prob.ref[i]))) && svg.includes('4 elements of P written, one per thread'), svg.slice(-3).join('|'));
svg = await part(5, 1000);
const shown = svg.filter((x) => /^\d+$/.test(x)).map(Number);
check('part 5 ends with all of P stored and says nobody waited', prob.ref.every((v) => shown.filter((x) => x === v).length >= prob.ref.filter((x) => x === v).length) && svg.includes('no thread waited for another') && svg.some((x) => x.startsWith('Block order here: ')), svg.slice(-3).join('|'));
svg = await part(6, 900);
check('part 6 shows the kernel\'s loop line and points to chapter 5', svg.includes('Pvalue += M[row*Width+k]*N[k*Width+col];') && svg.includes('Chapter 5 asks: how often was each input fetched?') && (await text(S + ' .caption')) === scripts[5], svg.slice(-3).join('|'));
// It plays: with sound off the clock alone drives it.
await page.evaluate((sel) => document.querySelectorAll(sel + ' .seg button')[1].click(), S);
await page.evaluate((sel) => { const box = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'); if (box.checked) box.click(); }, S);
await pause(100);
// The one primary button reads Play, Resume or Pause depending on where the player is.
await page.$eval(S + ' button.primary', (b) => b.click());
await pause(1600);
const moving = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const label = await text(S + ' button.primary');
await page.$eval(S + ' button.primary', (b) => b.click());
await pause(300);
const heldAt = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
await pause(500);
check('with sound off, playing moves the picture and pausing holds it', moving > 20 && label === 'Pause' && Math.abs((await page.$eval(S + ' input[type=range]', (e) => Number(e.value))) - heldAt) <= 2, `${moving} ${label} ${heldAt}`);
if (out) await (await page.$(S)).screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- who feeds this element
await page.$eval(F, (e) => e.scrollIntoView({ block: 'start' }));
const total = (name) => text(`${F} [data-total="${name}"]`);
const role = async (name) => (await texts(`${F} tr[data-role="${name}"] td`)).slice(1);
const who = () => text(F + ' [data-readout="who"]');
let rect = makeRect(8, 8, 8);
let own = owners(2, 1, 2);
let tr = traceRect(rect, 1, 2, 'book');
check('it opens on P[1][2] of an 8 × 8 product and names the owning thread and block', (await who()).includes(`P[1][2] = ${rect.C[10]} belongs to Thread (${own.thread.y},${own.thread.x}) of Block (${own.block.y},${own.block.x})`) && (await who()).includes('row 1 of M and column 2 of N'), await who());
check('counters: threads, multiply-adds and reads per thread', (await total('threads')) === big(counts(8, 8, 8).threads) && (await total('madds')) === '8' && (await total('reads')) === '16' && (await total('launched')) === big(gridFor(8, 8, 2).threads));
check('square: Width does all three jobs', eq(await role('loop'), ['Width', '8']) && eq(await role('gridx'), ['ceil(Width / 2)', '4']) && eq(await role('gridy'), ['ceil(Width / 2)', '4']), JSON.stringify([await role('loop'), await role('gridx'), await role('gridy')]));
check('the strips light the model\'s slots: M 1 apart, N 8 apart', eq(await lit(F, 'first'), tr.steps.map((s) => s.aIndex)) && eq(await lit(F, 'second'), tr.steps.map((s) => s.bIndex)) && (await page.$eval(F + ' [data-stride]', (e) => e.dataset.stride)) === '8');
await pressLabel(F, 'P[5][3]');
own = owners(2, 5, 3);
tr = traceRect(rect, 5, 3, 'book');
check('clicking P[5][3] moves the selection, its block and its reads', (await who()).includes(`P[5][3] = ${rect.C[43]} belongs to Thread (${own.thread.y},${own.thread.x}) of Block (${own.block.y},${own.block.x})`) && eq(await lit(F, 'first'), tr.steps.map((s) => s.aIndex)) && eq(await lit(F, 'second'), tr.steps.map((s) => s.bIndex)), await who());
check('the 2D views light row 5 of M and column 3 of N', eq(await page.$$eval(F + ' [data-cell^="first-"][data-on="1"]', (es) => es.map((e) => Number(e.dataset.cell.split('-')[1]))), tr.steps.map((s) => s.aIndex)) && eq(await page.$$eval(F + ' [data-cell^="second-"][data-on="1"]', (es) => es.map((e) => Number(e.dataset.cell.split('-')[1]))), tr.steps.map((s) => s.bIndex)));
await press(F, "PyTorch's: read along a row");
const tt = traceRect(rect, 5, 3, 'torch');
check('PyTorch\'s layout: the same thread now reads eight neighbouring slots, col*k + i', eq(await lit(F, 'second'), tt.steps.map((s) => s.bIndex)) && eq(tt.steps.map((s) => s.bIndex), [24, 25, 26, 27, 28, 29, 30, 31]) && (await page.$eval(F + ' [data-stride]', (e) => e.dataset.stride)) === '1' && (await text(F + ' [data-formula="second"]')) === 'W[col*Width + i]' && (await who()).includes('row 3 of W'), JSON.stringify(await lit(F, 'second')));
check('the counts do not change with the layout', (await total('madds')) === '8' && (await total('reads')) === '16' && eq(await lit(F, 'first'), tr.steps.map((s) => s.aIndex)));
await press(F, "the book's: read down a column");
await press(F, 'its whole block');
const bi = blockInputs(8, 8, 2, own.block);
const union = [...new Set(bi.rows.flatMap((rr) => bi.cols.flatMap((cc) => traceRect(rect, rr, cc, 'book').steps.map((s) => s.bIndex))))].sort((a, b) => a - b);
check('block view: the block\'s rows of M and columns of N', (await who()).startsWith(`Block (${own.block.y},${own.block.x}) covers rows ${bi.rows[0]} to ${bi.rows[1]} and columns ${bi.cols[0]} to ${bi.cols[1]} of P: 4 of its 4 threads pass the guard`) && eq(await lit(F, 'second'), union) && (await lit(F, 'first')).length === 16, await who());
if (out) await (await page.$(F)).screenshot({ path: out + '-feeds-square.png' });

// Rectangular.
await press(F, 'one thread');
await page.$eval(F + ' label.check input', (e) => e.click());
await pause(100);
check('switching on "rectangular" renames the matrices and offers j, k and l', (await page.$$(F + ' input[type=range]')).length === 3 && (await text(F + ' [data-matrix="A"] figcaption')).startsWith('A j × k = 3 × 8') && (await text(F + ' [data-matrix="C"] figcaption')).startsWith('C j × l = 3 × 6'));
await setRange(F, 'j', 5);
await setRange(F, 'k', 7);
await setRange(F, 'l', 10);
await press(F, '4', 'BLOCK_WIDTH');
rect = makeRect(5, 7, 10);
const grid = gridFor(5, 10, 4);
check('j = 5, k = 7, l = 10: the three matrices have the right shapes', (await page.$$(F + ' [data-cell^="first-"]')).length === 35 && (await page.$$(F + ' [data-cell^="second-"]')).length === 70 && (await page.$$(F + ' [data-matrix="C"] button')).length === 50);
check('k bounds the loop, l shapes grid x, j shapes grid y', eq(await role('loop'), ['k', '7']) && eq(await role('gridx'), ['ceil(l / 4)', String(grid.x)]) && eq(await role('gridy'), ['ceil(j / 4)', String(grid.y)]) && grid.x === 3 && grid.y === 2, JSON.stringify([await role('loop'), await role('gridx'), await role('gridy')]));
check('thread counts for the rectangle: 50 own an element, 96 are launched', (await total('threads')) === '50' && (await total('launched')) === '96' && (await total('madds')) === '7' && (await total('reads')) === '14');
await pressLabel(F, 'C[4][9]');
tr = traceRect(rect, 4, 9, 'book');
check('C[4][9] in the book\'s layout reads B at i*l + col: 9, 19, … 69', eq(await lit(F, 'second'), tr.steps.map((s) => s.bIndex)) && eq(tr.steps.map((s) => s.bIndex), [9, 19, 29, 39, 49, 59, 69]) && eq(await lit(F, 'first'), [28, 29, 30, 31, 32, 33, 34]) && (await text(F + ' [data-formula="second"]')) === 'B[i*l + col]', JSON.stringify(await lit(F, 'second')));
await press(F, "PyTorch's: read along a row");
check('and in PyTorch\'s layout W at col*k + i: 63 to 69', eq(await lit(F, 'second'), [63, 64, 65, 66, 67, 68, 69]) && (await text(F + ' [data-formula="second"]')) === 'W[col*k + i]' && (await text(F + ' [data-matrix="W"] figcaption')).includes('l × k = 10 × 7'), JSON.stringify(await lit(F, 'second')));
await press(F, 'its whole block');
const edge = blockInputs(5, 10, 4, owners(4, 4, 9).block);
check('an edge block: only some of its threads pass the guard', (await who()).includes(`${edge.active} of its 16 threads pass the guard`) && edge.active === 2, await who());
if (out) await (await page.$(F)).screenshot({ path: out + '-feeds-rect.png' });
// Back to square, largest size.
await page.$eval(F + ' label.check input', (e) => e.click());
await pause(100);
await setRange(F, 'Width', 16);
await press(F, '8', 'BLOCK_WIDTH');
check('Width 16 with BLOCK_WIDTH 8: a 2 × 2 grid of 256 threads', eq(await role('gridx'), ['ceil(Width / 8)', '2']) && (await total('threads')) === '256' && (await total('launched')) === '256' && (await page.$$(F + ' [data-matrix="P"] button')).length === 256);

// ---------------------------------------------------------------- thread granularity
await page.$eval(G, (e) => e.scrollIntoView({ block: 'start' }));
const cell = (row, mode) => text(`${G} tr[data-row="${row}"] td[data-mode="${mode}"]`);
const MODES = ['element', 'row', 'column'];
let gs = Object.fromEntries(MODES.map((m) => [m, granularity(m, 8, 16)]));
check('Width 8: threads launched, threads with work and multiply-adds per thread, for all three kernels', (await Promise.all(MODES.map(async (m) => (await cell('launched', m)) === big(gs[m].threads) && (await cell('working', m)) === big(gs[m].working) && (await cell('madds', m)) === big(gs[m].multiplyAddsPerThread)))).every(Boolean));
check('the total arithmetic is the same in every column', eq(await Promise.all(MODES.map((m) => cell('total', m))), ['512', '512', '512']));
await setRange(G, 'Width', 19);
gs = Object.fromEntries(MODES.map((m) => [m, granularity(m, 2048, 16)]));
check('Width 2048: about 4.2 million threads per element against 2048 per row', (await cell('working', 'element')) === '4,194,304' && (await cell('working', 'row')) === '2,048' && (await cell('madds', 'row')) === '4,194,304' && (await cell('madds', 'element')) === '2,048' && gs.row.multiplyAddsPerThread === 4194304);
check('grid shapes: 2D for per element, 1D for the others', (await cell('grid', 'element')) === '2D: 128 × 128 blocks of 16 × 16' && (await cell('grid', 'row')) === '1D: 8 blocks of 256' && (await cell('grid', 'column')) === '1D: 8 blocks of 256', await cell('grid', 'element'));
check('store strides: one store, 1 apart, Width apart', (await cell('stride', 'element')) === 'one store' && (await cell('stride', 'row')) === '1' && (await cell('stride', 'column')) === '2,048');
check('largest Width under the documented limits, and under a 32-bit index', (await cell('most', 'element')) === big(maxWidth('element', CUDA_DOCUMENTED, 16)) && (await cell('most', 'element')) === '1,048,560' && (await cell('most', 'row')) === big(maxWidth('row', CUDA_DOCUMENTED, 16)) && (await cell('int32', 'column')) === big(maxWidthInt32()) && (await cell('int32', 'element')) === '46,340', await cell('most', 'row'));
check('beyond 16 only a corner of P is drawn', (await page.$$(G + ' [data-matrix="P"] button')).length === 256 && (await text(G + ' [data-matrix="P"] figcaption')).includes('top-left 16 × 16 of 2,048 × 2,048'));
await setRange(G, 'Width', 4);
await press(G, 'one thread per column');
await pressLabel(G, 'P[3][5]');
const pressed = () => page.$$eval(G + ' [data-matrix="P"] button[aria-pressed="true"]', (bs) => bs.map((b) => b.getAttribute('aria-label')));
check('per column: clicking P[3][5] selects the thread that writes all of column 5', eq(await pressed(), [0, 1, 2, 3, 4, 5, 6, 7].map((rr) => `P[${rr}][5]`)) && (await text(G + ' [data-readout="thread"]')) === 'Thread 5 writes column 5 of P, top to bottom: 8 stores, 8 apart in memory.', await text(G + ' [data-readout="thread"]'));
await press(G, 'one thread per row');
await pressLabel(G, 'P[3][5]');
check('per row: the same click selects the thread that writes row 3', eq(await pressed(), [0, 1, 2, 3, 4, 5, 6, 7].map((cc) => `P[3][${cc}]`)) && (await text(G + ' [data-readout="thread"]')) === 'Thread 3 writes row 3 of P, left to right: 8 stores, 1 apart in memory.', await text(G + ' [data-readout="thread"]'));
await press(G, 'one thread per element');
await pressLabel(G, 'P[3][5]');
check('per element: one thread, one element', eq(await pressed(), ['P[3][5]']) && (await text(G + ' [data-readout="thread"]')) === 'Thread 29 writes P[3][5]: 1 store.', await text(G + ' [data-readout="thread"]'));
if (out) await (await page.$(G)).screenshot({ path: out + '-granularity.png' });

// ---------------------------------------------------------------- the committed prediction for lab 3.4
const P = '[data-predict="ch03/3.4/p-granularity-rank"]';
check('the ranking prediction cannot be committed before a choice is made', await page.$eval(P + ' button.primary', (b) => b.disabled));
await page.$$eval(P + ' input[type=radio]', (rs) => rs[0].click());
await pause(80);
await page.$eval(P + ' button.primary', (b) => b.click());
await pause(200);
qt = await text(P);
check('committing it shows the choice and either a measurement or what to run', qt.includes('You predicted: One thread per element, and both others take more than 10 times as long.') && !!(await page.$(P + ' [data-measured]')) && qt.includes('not expected values'), qt.slice(0, 200));

// ---------------------------------------------------------------- questions
const answer = async (needle, pick) => {
  const h = await question(needle);
  if (typeof pick === 'number') await (await h.$$('input[type=radio]'))[pick].click();
  else await (await h.$('input[type=number]')).type(pick);
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
qt = await answer('M[k*Width+row] * N[col*Width+k]', 3);
check('q-swapk: the transpose of N M is accepted and explained', qt.includes('Correct.') && qt.includes('element (col, row) of N · M'), qt.slice(-200));
qt = await answer('Which size sets the grid', 0);
check('q-rect: a wrong choice gets its own reason', qt.includes('Not quite.') && qt.includes('C has l columns'), qt.slice(-300));
qt = await answer('still fits in a signed', String(maxWidthInt32()));
check('q-limit: 46340 is accepted', qt.includes('Correct.') && qt.includes('1,048,560'), qt.slice(-300));
qt = await answer('What do you pass as N', 1);
check('q-linear: transposing once on the host is accepted', qt.includes('Correct.'), qt.slice(-200));
qt = await answer('Which kernel body is right', 1);
check('q-ex2: the guarded row walk is accepted', qt.includes('Correct.') && qt.includes('level 2'), qt.slice(-200));
qt = await answer('what does a one-thread-per-row kernel change', 0);
check('q-ex1c: fewer, longer threads is accepted', qt.includes('Correct.') && qt.includes('4,194,304'), qt.slice(-200));

// Lab 3.3's kernel is shown on request, from the lab's own files.
await page.evaluate(() => document.querySelector('[data-solution="3.3"] button').click());
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('naiveGemmKernel')), { timeout: 8000 }).catch(() => undefined);
const lab3 = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('naiveGemmKernel'));
  return f ? { tabs: [...f.querySelectorAll('.tabs button')].map((b) => b.textContent), text: f.textContent } : null;
});
check('the rectangular kernel is lab 3.3\'s solution, with CUDA and Metal tabs', !!lab3 && eq(lab3.tabs, ['CUDA', 'Metal kernel', 'Metal host']) && lab3.text.includes('Cvalue += A[row*k + i]*B[i*l + col];') && lab3.text.includes('naiveGemmKernel<<<dimGrid, dimBlock>>>'), JSON.stringify(lab3?.tabs));

// ---------------------------------------------------------------- progress
await pause(900);
const last = saved[saved.length - 1] ?? '';
const ids = ['ch03/3.4/p-n-index', 'ch03/3.4/p-store', 'ch03/3.4/p-granularity-rank', 'ch03/3.4/q-swapk', 'ch03/3.4/q-rect', 'ch03/3.4/q-limit', 'ch03/3.4/q-linear', 'ch03/3.4/q-ex2', 'ch03/3.4/q-ex1c'];
check('answers and predictions are saved under their ids, and only to the intercepted request', ids.every((id) => last.includes(`"${id}"`)), ids.filter((id) => !last.includes(`"${id}"`)).join(' '));

// ---------------------------------------------------------------- with a reader's CUDA report (a fixture)
fixture = true;
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(G);
await pause(500);
const options = await page.$$eval(G + ' select[aria-label="Launch limits"] option', (os) => os.map((x) => x.textContent));
check('a reported device with grid limits is offered beside the documented ones', eq(options, ["CUDA's documented limits", 'Your e2e fixture device (cuda)']), options.join('|'));
await page.select(G + ' select[aria-label="Launch limits"]', 'cuda');
await pause(120);
const fixLimits = { gridX: 2147483647, gridY: 1000, blockX: 1024, blockY: 1024, threadsPerBlock: 1024 };
check('its limits replace the documented ones: 1000 blocks in y allow 16,000 rows per element', (await cell('most', 'element')) === big(maxWidth('element', fixLimits, 16)) && (await cell('most', 'element')) === '16,000' && (await cell('most', 'row')) === big(maxWidth('row', fixLimits, 16)) && (await text(G + ' [data-readout="limits"]')).includes('your e2e fixture device (cuda) reported'), await text(G + ' [data-readout="limits"]'));
await page.$$eval(P + ' input[type=radio]', (rs) => rs[2].click());
await pause(80);
await page.$eval(P + ' button.primary', (b) => b.click());
await pause(200);
const measured = await page.$eval(P + ' [data-measured="cuda"]', (e) => ({ text: e.textContent, fastest: e.dataset.fastest })).catch(() => null);
check('with lab 3.4 results the prediction is compared with the measurement, and nothing is invented', !!measured && measured.fastest === 'element' && measured.text.includes('per element 2 ms, per row 50 ms, per column 80 ms') && measured.text.includes('per row took 25 times as long, per column took 40 times as long') && measured.text.includes('That is not the option you chose.'), measured?.text);

// ---------------------------------------------------------------- errors
const expected404 = (u) => /\/content\/ch03\/narration\/(manifest\.json|sixteen-threads-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
