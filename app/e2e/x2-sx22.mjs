// Browser check for section X2.2: the "One block, six rules" stepper, the error explorer,
// the "spend a bit" prediction and three questions. Every number asserted is recomputed in
// the page from app/src/sim/quant.ts through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx22.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
// Leave no trace in the real progress file, and start from no answers so the prediction is fresh.
page.on('request', (r) =>
  r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : '{"version":1,"sections":{},"answers":{},"review":{}}' })
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
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const b = [...root.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const setSelect = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const setRange = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

await goto(page, 'x2/X2.2');
const S = bench('One block, six rules');
await page.waitForSelector(S);
await page.waitForSelector('.code pre');

const M = await page.evaluate(async () => {
  const q = await import('/src/sim/quant.ts');
  const v = q.demoBlock();
  const t = q.quantTrace('q4_0', v);
  const tf = q.quantTrace('q4_0', v, false);
  const sym = q.quantTrace('sym4', v);
  const code = q.quantTrace('mlx-code', v);
  const big = q.quantTrace('q4_0', q.demoBlock('normal', 2.5, '-'));
  const sw = (dist, scheme) => q.errorSweep({ dist, scheme }).map((p) => ({ k: `${p.gran}:${p.bits}`, rel: p.rel }));
  const tAff = q.errorSweep({ dist: 't3', scheme: 'affine' });
  const best45 = q.bestUnder(tAff, 4.5);
  const pr = q.errorSweep({ dist: 't3', scheme: 'affine', bits: [4, 5], grans: [32, 64] });
  const g = (b, gr) => pr.find((p) => p.bits === b && p.gran === gr).rel;
  return {
    deq21: t.dequant[21], deq21f: tf.dequant[21], rel: t.rel, ext: t.extreme.value, symClips: sym.clipped.filter(Boolean).length,
    zeroCode: code.zeroCode, bigRel: big.rel, nSE: sw('normal', 'signed-extreme'), tAff: tAff.map((p) => ({ k: `${p.gran}:${p.bits}`, rel: p.rel })),
    best45: { bits: best45.bits, gran: String(best45.gran) }, lower: g(4, 32) < g(5, 64) ? 0 : 1,
  };
});

// The listing is the host C++ reference.
check('listing shows q4_0_ref.h', (await text('.code pre')).includes('quantize_q4_0_block'));

// Stepper, default block, Q4_0.
let n = await text(S + ' .narrate');
check('step 0 describes the block', n.includes('32 float32 weights'), n);
await press(S, 'Step');
n = await text(S + ' .narrate');
check('step 1 finds the signed extreme, weight 5 at -0.800', n.includes('weight 5, -0.800'), n);
await press(S, 'Step');
n = await text(S + ' .narrate');
check('step 2 derives d = 0.1000', n.includes('d = -0.800 / -8 = 0.1000'), n);
await press(S, 'To the end');
const cells = await page.$$eval(S + ' [data-w]', (els) => els.map((e) => ({ w: +e.dataset.w, clipped: e.dataset.clipped, deq: e.querySelector('[data-deq]') ? +e.querySelector('[data-deq]').dataset.deq : null })));
check('only weight 21 is clipped', cells.filter((c) => c.clipped === 'yes').map((c) => c.w).join(',') === '21', JSON.stringify(cells.filter((c) => c.clipped === 'yes')));
check('weight 21 dequantises to the model value 0.6998…', close(cells[21].deq, M.deq21) && close(M.deq21, 0.6998291015625), String(cells[21].deq));
const rel = await page.$eval(S + ' [data-rel]', (e) => +e.dataset.rel);
check('block error matches the model', close(rel, M.rel), `${rel} vs ${M.rel}`);
check('summary says 1 clipped, 16 levels', (await text(S + ' [data-rel]')).includes('1 clipped') && (await text(S + ' [data-rel]')).includes('16 levels'));
if (out) await page.screenshot({ path: out + '-stepper.png' });

// FP16 storage off: weight 21 comes back as 7 times the float d.
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('FP16')).querySelector('input').click(), S);
await pause(120);
const deqF = await page.$eval(S + ' [data-w="21"] [data-deq]', (e) => +e.dataset.deq);
check('without FP16 storage weight 21 is the model’s 0.70000001', close(deqF, M.deq21f) && deqF !== M.deq21, String(deqF));
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('FP16')).querySelector('input').click(), S);
await pause(80);

// Symmetric ±7: nothing clips.
await setSelect(S, 'rule', 'sym4');
await press(S, 'To the end');
check('symmetric ±7 clips nothing', (await text(S + ' [data-rel]')).includes(`${M.symClips} clipped`) && M.symClips === 0);
check('symmetric ±7 has 15 levels', (await text(S + ' [data-rel]')).includes('15 levels'));
// MLX as coded: a code lands on zero.
await setSelect(S, 'rule', 'mlx-code');
await press(S, 'To the end');
check('MLX as coded names the code on zero', (await text(S + ' [data-rel]')).includes(`code ${M.zeroCode} on zero`), await text(S + ' [data-rel]'));
// A big negative outlier, Q4_0.
await setSelect(S, 'rule', 'q4_0');
await press(S, 'negative');
await setRange(S, 'outlier magnitude', 2.5);
await press(S, 'To the end');
const relBig = await page.$eval(S + ' [data-rel]', (e) => +e.dataset.rel);
check('outlier 2.5: error matches the model and is larger than with 0.8', close(relBig, M.bigRel) && relBig > M.rel, `${relBig} vs ${M.bigRel}`);

// Error explorer.
const E = bench('Error against block size and bit width');
const table = () => page.$$eval(E + ' [data-cell]', (els) => els.map((e) => ({ k: e.dataset.cell, rel: +e.dataset.rel })));
let t1 = await table();
check('explorer table: 49 cells', t1.length === 49, String(t1.length));
check('explorer, normal, Q4_0-style: every cell matches errorSweep', t1.every((c) => close(c.rel, M.nSE.find((m) => m.k === c.k).rel)));
await press(E, 'Student t (3)');
await setSelect(E, 'scheme', 'affine');
t1 = await table();
check('explorer, t3, affine: every cell matches errorSweep', t1.every((c) => close(c.rel, M.tAff.find((m) => m.k === c.k).rel)));
check('t3 at 4 bits: one scale per tensor is far worse than block 32', t1.find((c) => c.k === 'tensor:4').rel > 2 * t1.find((c) => c.k === '32:4').rel);
await press(E, '4.5');
const best = await page.$eval(E + ' [data-part="best"]', (e) => ({ bits: e.dataset.bits, gran: e.dataset.gran }));
check('budget 4.5 picks the model’s best point', best.bits === String(M.best45.bits) && best.gran === M.best45.gran, JSON.stringify(best));
check('the plot draws one curve per granularity', (await page.$$(E + ' [data-curve]')).length === 7);
if (out) await page.screenshot({ path: out + '-sweep.png' });

// The spend-a-bit prediction.
const P = '[data-part="spend"]';
await page.$eval(P, (e) => e.scrollIntoView());
await page.evaluate((P) => [...document.querySelectorAll(P + ' button')][1].click(), P);
await pause(150);
const lower = await page.$eval(P + ' [data-lower]', (e) => +e.dataset.lower).catch(() => null);
check('prediction reveals which option lowers the error, from the model', lower === M.lower && lower === 1, String(lower));
check('reveal echoes the choice', (await text(P)).includes('You chose: keep group 64'));

// Questions.
const ask = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  if (typeof value === 'number' && (await q.$('input[type=number]'))) await (await q.$('input[type=number]')).type(String(value));
  else await (await q.$$('input[type=radio]'))[value].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-q40-clip accepts 0.7', await ask('Another weight in the same block is', 0.7));
check('q-bits accepts 4.5', await ask('How many bits per weight is that', 4.5));
check('q-q40-zero: the guarded inverse', await ask('A block of 32 zeros', 0));
check('q-q4k accepts 4.5', await ask('256 weights in 144 bytes', 4.5));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
