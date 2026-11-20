// Browser check for section X1.4: the "Radix select" stepper (every preset kind, k, the warp
// widths, the walk, the gather and its ballots), the "Passes and bytes against k and V"
// explorer, the speed-up prediction and questions. Every number asserted is recomputed here
// from the model (src/sim/topkselect.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx14.mjs [--dark] [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

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
const M = await import('../src/sim/topkselect.ts');

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const S = bench('Radix select');
const C = bench('Passes and bytes against k and V');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const seg = async (scope, group, label) => {
  await page.evaluate((scope, group, label) => {
    const g = document.querySelector(`${scope} [role=group][aria-label="${group}"]`);
    [...g.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click();
  }, scope, group, label);
  await pause(120);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const narrate = () => text(S + ' .narrate');
const states = () => page.$$eval(S + ' td[data-col]', (tds) => Object.fromEntries(tds.map((t) => [Number(t.dataset.col), t.dataset.state])));
const hist = () => page.$$eval(S + ' [data-part=hist] tr[data-bin]', (rs) => rs.map((r) => [Number(r.dataset.bin), Number(r.dataset.count), r.dataset.mark]));
const slots = () => page.$$eval(S + ' [data-slot]', (ss) => ss.map((s) => s.dataset.col));
const passesShown = () => page.$eval(S + ' [data-passes]', (e) => Number(e.dataset.passes)).catch(() => null);
const nz = (h) => h.map((c, b) => [b, c]).filter(([, c]) => c > 0).reverse();

await goto(page, 'x1/X1.4');
await page.waitForSelector(S);

// ---------------------------------------------------------------- the stepper, the scene's row
const scene = M.STEP_PRESETS.find((p) => p.id === 'scene');
let tr = M.selectTrace(scene.logits, scene.k, 16, 32);
check('stepper opens on the animation\'s row with k = 12', (await page.$eval(S + ' input[aria-label=k]', (e) => e.value)) === '12' && (await narrate()).startsWith('Nothing has run yet'));
check('64 cells, thread t reads t, t + 16, t + 32, t + 48', Object.keys(await states()).length === 64);
await press(S, 'Step');
let st = await states();
check('pass 1, round 1: all 16 lanes of column t do the atomic', [...Array(16).keys()].every((c) => st[c] === 'active') && (await narrate()).includes("every lane's key matches"), await narrate());
check('pass 1, round 1: the histogram so far is the model\'s', JSON.stringify((await hist()).map(([b, c]) => [b, c])) === JSON.stringify(nz(tr.passes[0].rounds[0].histAfter)));
await press(S, 'To the next walk');
const p0 = tr.passes[0];
let h = await hist();
check('walk 1: the full histogram of byte 1, highest bin first', JSON.stringify(h.map(([b, c]) => [b, c])) === JSON.stringify(nz(p0.hist)));
check('walk 1: bin C0 is the crossing bin, bins above in, below out', h.find(([b]) => b === 0xc0)?.[2] === 'crossing' && h.filter(([, , m]) => m === 'in').map(([b]) => b).join() === '193' && h.filter(([b]) => b < 0xc0).every(([, , m]) => m === 'out'));
check('walk 1: 6 in, 45 out, 13 undecided; 6 still needed', (await page.$eval(S + ' [data-counts]', (e) => e.dataset.counts)) === '6,45,13' && (await text(S + ' [data-need]')) === '6');
check('walk 1: the prefix is 0xC0 and three undecided bytes', (await text(S + ' [data-prefix]')) === M.prefixHex(p0.prefixOut, p0.maskOut));
check('walk 1: the narration names the walk and the crossing bin', (await narrate()).startsWith('Walk from bin FF down') && (await narrate()).includes('in bin C0'));
await press(S, 'Step');
st = await states();
const r0 = tr.passes[1].rounds[0];
check('pass 2, round 1: only lanes matching the prefix are active', [...Array(16).keys()].every((t) => (st[t] === 'active') === r0.lanes[t].match), JSON.stringify(st));
check('pass 2, round 1: the warp splits at the prefix test', (await page.$eval(S + ' tr[data-warp="0"]', (e) => e.dataset.diverged)) === (r0.warps[0].diverged ? '1' : '0') && r0.warps[0].diverged);
if (out) await page.screenshot({ path: out + '-stepper-pass2.png' });

// Warp width 8: two warps per round, each with its own split.
await seg(S, 'Warp width', '8');
check('warp width 8: two warp groups', (await page.$$(S + ' tr[data-warp]')).length === 2);
const tr8 = M.selectTrace(scene.logits, scene.k, 16, 8);
const div8 = await page.$$eval(S + ' tr[data-warp]', (rs) => rs.map((r) => r.dataset.diverged));
check('warp width 8: each warp splits as the model says', JSON.stringify(div8) === JSON.stringify(tr8.passes[1].rounds[0].warps.map((w) => (w.diverged ? '1' : '0'))), JSON.stringify(div8));
await seg(S, 'Warp width', '64');
check('warp width 64: one group, threads in lanes 0 to 15 only', (await page.$$(S + ' tr[data-warp]')).length === 1 && (await text(S + ' tr[data-warp="0"]')).includes('threads in lanes 0 to 15 only'));
await seg(S, 'Warp width', '32');

// The gather and the output.
const views = tr.passes.length * 5;
await press(S, 'Restart');
for (let i = 0; i < views + 2; i++) await press(S, 'Step');
check('gather round 2 shows the ballots for its warp', (await page.$$(S + ' [data-ballot-warp]')).length === 1 && (await narrate()).startsWith('Gather, round 2'));
const g1 = tr.gather[1];
const tieHex = await page.$eval(S + ' [data-ties]', (e) => e.dataset.ties);
const want = (lanes) => {
  let s = '';
  for (let d = lanes.length / 4 - 1; d >= 0; d--) {
    let v = 0;
    for (let j = 0; j < 4; j++) if (lanes[d * 4 + j]) v |= 1 << j;
    s += v.toString(16).toUpperCase();
  }
  return '0x' + s;
};
check('gather round 2: the tie ballot is the model\'s (two ties, columns 22 and 23)', tieHex === want(g1.warps[0].tieBallot) && g1.warps[0].tieCount === 2, tieHex);
await press(S, 'Run to the end');
check('end: 12 slots hold the model\'s columns, in column order', JSON.stringify((await slots()).map(Number)) === JSON.stringify(tr.out.map((o) => o.col)));
check('end: four passes and one gather', (await passesShown()) === 4);
st = await states();
check('end: the tie at column 23 is cut, 0 and 22 kept', st[23] === 'dropped' && st[0] === 'kept' && st[22] === 'kept');
if (out) await page.screenshot({ path: out + '-stepper-end.png' });

// k = 5 on the same row.
await page.$eval(S + ' input[aria-label=k]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '5');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(150);
await press(S, 'Run to the end');
check('k = 5: five slots, the reference top-5 in column order', JSON.stringify((await slots()).map(Number)) === JSON.stringify(M.referenceTopK(scene.logits, 5)));

// Presets.
await seg(S, 'Row', 'peaked');
await press(S, 'Run to the end');
check('peaked, k = 3: the select stops after one pass', (await passesShown()) === 1 && (await narrate()).startsWith('Done'));
await press(S, 'Restart');
await press(S, 'To the next walk');
check('peaked: the walk says the whole bin is needed', (await narrate()).includes('the select stops here'));
await seg(S, 'Row', 'top 24 bits shared');
await press(S, 'To the next walk');
h = await hist();
check('top 24 bits shared: pass 1 finds a single bin', h.length === 1 && h[0][2] === 'crossing');
await press(S, 'Run to the end');
check('top 24 bits shared: four passes', (await passesShown()) === 4);
await seg(S, 'Row', 'ties straddling the k-th');
const ties = M.STEP_PRESETS.find((p) => p.id === 'ties');
const trT = M.selectTrace(ties.logits, ties.k, 16, 32);
await press(S, 'Run to the end');
check('ties: 2 of the 5 tied logits kept, the lowest columns', trT.need === 2 && trT.equal === 5 && JSON.stringify((await slots()).map(Number)) === JSON.stringify(trT.out.map((o) => o.col)));
await seg(S, 'Row', 'negatives only');
const neg = M.STEP_PRESETS.find((p) => p.id === 'negative');
await press(S, 'Run to the end');
check('negatives only: the model\'s output', JSON.stringify((await slots()).map(Number)) === JSON.stringify(M.referenceTopK(neg.logits, neg.k)));

// ---------------------------------------------------------------- the cost explorer
const rowsOf = () => page.$$eval(C + ' tr[data-design]', (rs) => rs.map((r) => [r.dataset.design, Number(r.dataset.perRow), Number(r.dataset.passes)]));
let kNow = await page.$eval(C + ' input[aria-label=k]', (e) => Number(e.value));
const kOf = (V, pos) => Math.max(1, Math.min(V, Math.round(Math.exp((pos / 1000) * Math.log(V)))));
let model = M.costs(151936, kOf(151936, kNow), 64);
check('cost: three designs with the model\'s bytes per row and passes', JSON.stringify(await rowsOf()) === JSON.stringify(model.map((c) => [c.id, c.perRow, c.passes])), JSON.stringify(await rowsOf()));
check('cost: k opens at 40', kOf(151936, kNow) === 40);
const ratio = await page.$eval(C + ' [data-readout=ratio]', (e) => Number(e.dataset.ratio));
check('cost: the sort moves about 4 times the select\'s bytes at k = 40', Math.abs(ratio - M.trafficRatio(151936, 40)) < 1e-9 && ratio > 3.99);
await setInput(C + ' input[aria-label=k]', 1000);
check('cost: at k = V the ratio is 80 / 28', Math.abs((await page.$eval(C + ' [data-readout=ratio]', (e) => Number(e.dataset.ratio))) - 80 / 28) < 1e-9);
await setInput(C + ' input[aria-label=k]', 700);
kNow = 700;
await seg(C, 'Vocabulary', '50,257');
model = M.costs(50257, kOf(50257, kNow), 64);
check('cost: 50,257 tokens, the model\'s bytes', JSON.stringify(await rowsOf()) === JSON.stringify(model.map((c) => [c.id, c.perRow, c.passes])));
await seg(C, 'Batch', '1');
check('cost: batch 1 says one block does all the work', (await text(C + ' [data-readout=blocks]')).startsWith('Batch 1: one block'));
check('cost: bisection reads the row 33 times', (await rowsOf())[2][2] === 33);
if (out) await page.screenshot({ path: out + '-cost.png' });

// Measured column or the instruction to produce it.
const reports = await (await fetch(BASE + '/api/reports')).json();
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const anyLab = real.some((b) => (reports[b].results ?? []).some((r) => r.lab === M.LAB_X11 && r.variant === 'solution' && M.measuredPairs(r.bench).length));
check(anyLab ? 'cost: lab X1.1 measurements are shown' : 'cost: with no lab X1.1 run, the command to produce one is shown',
  anyLab ? !!(await page.$(C + ' [data-part=measured]')) : (await text(C + ' [data-part=measured-none]')).includes('pmpp.py run x1/lab1'));

// ---------------------------------------------------------------- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Lab X1.1 runs both')));
await (await pred.$('input[type=number]')).type('4');
await (await pred.$('button.primary')).click();
await pause(200);
const reveal = await pred.evaluate((e) => e.textContent);
check('prediction is committed and the reveal reads this machine\'s reports', reveal.includes('You predicted 4') && (anyLab ? reveal.includes('×') : reveal.includes('No measurement on this machine yet')), reveal.slice(0, 200));

const numeric = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
const choice = async (needle, index) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-passes accepts 4', await numeric('at most how many histogram passes', M.MAX_PASSES));
check('q-bisect accepts 32', await numeric('fixes one bit per counting pass', M.BISECT_PASSES));
check('q-bytes-ratio accepts 80 / 20', await numeric('Sort bytes divided by select bytes', M.SORT_PER_KEY / M.SELECT_PER_KEY));
check('q-hf-ties: transformers 3, M5 2', await choice('TopKLogitsWarper', 0));
check('q-k1: top-k with k = 1 is argmax', await choice('is top-k the same as greedy argmax', 0));
check('q-integer: exact under every math mode', await choice('compare integer keys instead of floats', 0));
check('q-batch1: one block per row', await choice('At batch 1, what limits', 0));

// Phone width: nothing makes the page scroll sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

// Lab X1.1's files are loaded by their planned path; until the lab exists they answer 404.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
