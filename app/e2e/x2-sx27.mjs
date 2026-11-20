// Browser check for section X2.7: "Where the step's bytes go", the step speed-up prediction,
// "A toy model, quantised" and four questions. Model values are recomputed in the page from
// app/src/sim/qstep.ts through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx27.mjs [screenshot-prefix]
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
const pressIn = async (scope, group, label) => {
  const ok = await page.evaluate((scope, group, label) => {
    const g = document.querySelector(`${scope} [aria-label="${group}"]`);
    const b = [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, scope, group, label);
  await pause(100);
  return ok;
};
const setValue = async (scope, sel, value, ev) => {
  await page.evaluate((scope, sel, value, ev) => {
    const s = document.querySelector(`${scope} ${sel}`);
    const proto = s.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event(ev, { bubbles: true }));
  }, scope, sel, value, ev);
  await pause(120);
};
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

await goto(page, 'x2/X2.7');
const B = bench("Where the step's bytes go");
await page.waitForSelector(B);

const M = await page.evaluate(async () => {
  const q = await import('/src/sim/qstep.ts');
  const sh = q.M8_DEFAULT;
  const dev = q.BOOK_DEVICE;
  const custom = { peak: 2000, bw: 100 };
  const toy = q.toyModel();
  const run = (f) => {
    const r = q.greedyRun(toy, { W: [f, f], E: f });
    return { flips: r.steps.filter((s) => s.flipped).length, div: r.firstDivergence };
  };
  return {
    layers1: q.predictedSpeedup(sh, { layers: 'q4_0', embed: 'f32' }, 1, dev),
    all1: q.predictedSpeedup(sh, { layers: 'q4_0', embed: 'q4_0' }, 1, dev),
    all8: q.predictedSpeedup(sh, { layers: 'q4_0', embed: 'q4_0' }, 8, dev),
    allCustom8: q.predictedSpeedup(sh, { layers: 'q4_0', embed: 'q4_0' }, 8, custom),
    q8custom8: q.predictedSpeedup(sh, { layers: 'q8_0', embed: 'q8_0' }, 8, custom),
    shape: q.predictedSpeedup({ kind: 'step', d: 2048, layers: 8, vocab: 32000 }, { layers: 'q8_0', embed: 'q8_0' }, 8, custom),
    toyQ4: run('q4_0'), toyQ8: run('q8_0'), toyA3: run('affine3'),
  };
});
const stepData = () => page.$eval(B + ' [data-part="step"]', (e) => ({ s: +e.dataset.speedup, w: +e.dataset.wratio, t: e.textContent }));

// ---------------- the bytes explorer
let d = await stepData();
check('default: layers Q4_0, head F32, batch 1: the model’s 1.21 times', close(d.s, M.layers1) && d.s.toFixed(2) === '1.21', String(d.s));
check('default: weight bytes shrink 1.21 times', d.w.toFixed(2) === '1.21' && d.t.includes('shrink 1.21 times'), d.t);
await pressIn(B, 'embedding format', 'Q4_0');
d = await stepData();
check('everything Q4_0: the model’s step speed-up, a little under 7.11', close(d.s, M.all1) && d.s < 7.11 && d.s > 7, String(d.s));
check('everything Q4_0: weight bytes shrink 7.11 times', d.w.toFixed(2) === '7.11');
check('the head carries most bytes in F32', await page.$eval(B + ' svg[data-bar="F32"] [data-piece="head"] rect', (r) => +r.getAttribute('width') > 400));
await setValue(B, 'input[aria-label="batch"]', 8, 'input');
d = await stepData();
check('batch 8: the model’s smaller speed-up', close(d.s, M.all8) && d.s < M.all1, String(d.s));
const headBound = await page.$eval(B + ' tr[data-piece="head"]', (r) => r.textContent);
check('batch 8 on the book’s example: the head is arithmetic-bound in the model', headBound.includes('arithmetic'), headBound);
if (out) await page.screenshot({ path: out + '-bytes.png' });
await setValue(B, 'select[aria-label="device"]', 'custom', 'change');
await setValue(B, 'input[aria-label="peak GFLOP/s"]', 2000, 'input');
await setValue(B, 'input[aria-label="bandwidth GB/s"]', 100, 'input');
d = await stepData();
check('custom device: matches the model', close(d.s, M.allCustom8), `${d.s} vs ${M.allCustom8}`);
await pressIn(B, 'layer format', 'Q8_0');
await pressIn(B, 'embedding format', 'Q8_0');
d = await stepData();
check('Q8_0 throughout: matches the model', close(d.s, M.q8custom8), `${d.s} vs ${M.q8custom8}`);
await pressIn(B, 'shape', 'custom');
await setValue(B, 'input[aria-label="d"]', 2048, 'input');
await setValue(B, 'input[aria-label="layers"]', 8, 'input');
d = await stepData();
check('custom shape (d 2048, 8 layers): matches the model', close(d.s, M.shape), `${d.s} vs ${M.shape}`);

// ---------------- the prediction
const P = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('all-Q4_0 step')));
await (await P.$('input[type=number]')).type('5');
await (await P.$('button.primary')).click();
await pause(200);
const ceiling = await page.$eval('[data-part="reveal"]', (e) => +e.dataset.ceiling).catch(() => null);
check('prediction reveals the model’s ceiling for the book’s example', ceiling != null && close(ceiling, M.all1), String(ceiling));
check('reveal echoes the prediction', (await P.evaluate((e) => e.textContent)).includes('You predicted 5'));

// ---------------- the toy model
const T = bench('A toy model, quantised');
await page.$eval(T, (e) => e.scrollIntoView());
const toy = () => page.$eval(T + ' [data-part="toy"]', (e) => ({ flips: +e.dataset.flips, div: e.dataset.div, red: e.closest('section').querySelectorAll('[data-flipped="yes"]').length }));
let t = await toy();
check('Q4_0 everywhere: flips and divergence as the model', t.flips === M.toyQ4.flips && t.div === String(M.toyQ4.div ?? '') && t.red === t.flips, JSON.stringify(t));
for (const a of ['W0 (layer 0)', 'W1 (layer 1)', 'E (embedding and head)']) await setValue(T, `select[aria-label="${a}"]`, 'q8_0', 'change');
t = await toy();
check('Q8_0 everywhere: no flips, no divergence', t.flips === M.toyQ8.flips && t.flips === 0 && t.div === '', JSON.stringify(t));
for (const a of ['W0 (layer 0)', 'W1 (layer 1)', 'E (embedding and head)']) await setValue(T, `select[aria-label="${a}"]`, 'affine3', 'change');
t = await toy();
check('affine 3-bit everywhere: more flips than Q4_0, as the model', t.flips === M.toyA3.flips && t.flips > M.toyQ4.flips, JSON.stringify(t));
const toks = await page.$$eval(T + ' [data-part="tokens"] tr', (rs) => rs.map((r) => r.querySelectorAll('td').length));
check('token rows show the start token and 32 steps', toks.every((n) => n === 33), toks.join(','));
if (out) await page.screenshot({ path: out + '-toy.png' });

// ---------------- questions
const ask = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  if (typeof value === 'number') await (await q.$('input[type=number]')).type(String(value));
  else await (await q.$$('input[type=radio]'))[value.idx].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-step-ratio accepts 1.21', await ask('Quantise only the two layers to Q4_0', 1.21));
check('q-embed: the tied head', await ask('Once the embedding table E is quantised', { idx: 0 }));
check('q-margin: small margins', await ask('Which greedy steps are at risk', { idx: 0 }));
check('q-batch8: it shrinks', await ask('At batch 8 every weight read serves 8 rows', { idx: 0 }));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
