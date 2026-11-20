// Browser check for section 20.1: the "which kernel is this block?" classification and the
// FLOP explorer, both against app/src/sim/llmflops.ts, plus questions.
// Usage: node e2e/ch20-s201.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BLOCKS, BOOK_SHAPE, crossover, decodeBytes, judge, layerFlops } from '../src/sim/llmflops.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const CL = 'section.bench[aria-label="Which kernel is this block?"]';
const EX = `section.bench[aria-label="Where a layer's arithmetic goes"]`;
const clickText = async (scope, label) => {
  const ok = await page.evaluate(
    (scope, label) => {
      const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
      if (!b || b.disabled) return false;
      b.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return ok;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(40);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};

await goto(page, 'ch20/20.1');
await page.waitForSelector(CL);

// ---- classification
const rows = await page.$$eval(CL + ' tr[data-block]', (rs) => rs.map((r) => r.dataset.block));
check('the classification lists every block of the model, in order', JSON.stringify(rows) === JSON.stringify(BLOCKS.map((b) => b.id)), rows.join(','));
check('Check is disabled until something is answered', !(await clickText(CL, 'Check')));

// A mixed set of answers: odd rows right, even rows given the first wrong option.
const PRIMS = ['host', 'gather', 'elementwise', 'gemm', 'rowreduce', 'attention'];
const MILES = ['M0', 'M1-M3', 'M4', 'M5', 'M6', 'none'];
const chosen = BLOCKS.map((b, i) =>
  i % 2 ? { p: b.primitive[0], m: b.milestone[0] } : { p: PRIMS.find((p) => !b.primitive.includes(p)), m: b.milestone[0] },
);
for (const [i, b] of BLOCKS.entries()) {
  await setSelect(`${CL} tr[data-block="${b.id}"] select[aria-label^="What runs"]`, chosen[i].p);
  await setSelect(`${CL} tr[data-block="${b.id}"] select[aria-label^="Milestone"]`, chosen[i].m);
}
await clickText(CL, 'Check');
const verdicts = await page.$$eval(CL + ' tr[data-block]', (rs) => rs.map((r) => r.dataset.verdict));
const want = BLOCKS.map((b, i) => {
  const v = judge(b, chosen[i].p, chosen[i].m);
  return v.primitive && v.milestone ? 'right' : 'wrong';
});
check('each row\'s verdict matches the model', JSON.stringify(verdicts) === JSON.stringify(want), JSON.stringify(verdicts));
const score = await page.$eval(CL + ' [data-score]', (e) => e.dataset.score);
check('the score counts the right rows', Number(score) === want.filter((v) => v === 'right').length, score);
const whys = await page.$$eval(CL + ' tr[data-why]', (rs) => rs.map((r) => r.textContent));
check('every row shows its reason after checking', whys.length === BLOCKS.length && whys.every((w, i) => w === BLOCKS[i].why), String(whys.length));
const wrongChip = await page.$eval(`${CL} tr[data-block="tokenizer"]`, (r) => r.textContent);
check('a wrong "what runs it" is named as such', wrongChip.includes('what runs it'), wrongChip);
check('no discovery note while rows are wrong', (await page.$(CL + ' [data-discovery]')) === null);

await clickText(CL, 'Show the answers');
const all = await page.$$eval(CL + ' tr[data-block]', (rs) => rs.map((r) => r.dataset.verdict));
check('"Show the answers" fills every row right', all.every((v) => v === 'right') && (await page.$eval(CL + ' [data-score]', (e) => e.dataset.score)) === String(BLOCKS.length));
const disc = await page.$eval(CL + ' [data-discovery]', (e) => e.textContent).catch(() => '');
check('with all rows right, the discovery: only three rows say M6', disc.includes('Only three rows say M6'), disc);
const m6 = await page.$$eval(CL + ' tr[data-block]', (rs) => rs.filter((r) => r.querySelector('select[aria-label^="Milestone"]').value === 'M6').map((r) => r.dataset.block));
check('the three M6 rows are Q K^T, mask and softmax, P V', JSON.stringify(m6) === JSON.stringify(['qkt', 'softmax', 'pv']), m6.join(','));
// Accepted alternatives: Q K^T as a GEMM is also right.
await setSelect(`${CL} tr[data-block="qkt"] select[aria-label^="What runs"]`, 'gemm');
await clickText(CL, 'Check');
check('Q K^T run as a GEMM is accepted too', (await page.$eval(`${CL} tr[data-block="qkt"]`, (r) => r.dataset.verdict)) === 'right');
if (out) await page.screenshot({ path: out + '-classify.png' });

// ---- explorer
const readout = async () => ({
  ratio: Number(await page.$eval(EX + ' [data-ratio]', (e) => e.dataset.ratio)),
  cross: Number(await page.$eval(EX + ' [data-cross]', (e) => e.dataset.cross)),
  shares: await page.$$eval(EX + ' [data-part]', (ps) => Object.fromEntries(ps.map((p) => [p.dataset.part, Number(p.dataset.share)]))),
  n: await page.$eval(EX + ' [data-n]', (e) => Number(e.textContent.replace(/,/g, ''))),
});
const near = (a, b, tol = 6e-5) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const agrees = (r, n, shape, phase) => {
  const f = layerFlops(n, shape, phase);
  return (
    near(r.ratio, Number(f.attnOverLinear.toFixed(4))) &&
    r.cross === Math.round(crossover(shape, phase)) &&
    ['proj', 'qk', 'pv', 'ffn'].every((k) => near(r.shares[k], Number((f[k] / f.total).toFixed(4))))
  );
};
let r = await readout();
const book4096 = { ...BOOK_SHAPE, dModel: 4096 };
check('default: N = 4096, d_model = 4096, products / linear = 1/6, level at 24,576', r.n === 4096 && agrees(r, 4096, book4096, 'prefill') && r.cross === 24576, JSON.stringify(r));
await setRange(EX + ' input[aria-label="context length (log scale)"]', 16);
r = await readout();
check('N = 65,536: the products overtake the linear layers', r.n === 65536 && r.ratio > 1 && agrees(r, 65536, book4096, 'prefill'), JSON.stringify(r));
const marker = await page.$(EX + ' [data-crossmark]');
check('the plot marks the crossover', marker !== null);
await clickText(EX, 'SwiGLU, 3 matrices');
await setRange(EX + ' input[aria-label="d_ff over d_model"]', 3.5);
r = await readout();
const swi = { dModel: 4096, ffRatio: 3.5, ffn: 'swiglu', causalSkip: false };
check('SwiGLU with d_ff = 3.5 d_model moves the crossover to 29,696', agrees(r, 65536, swi, 'prefill') && r.cross === 29696, JSON.stringify(r));
await page.$eval(EX + ' label.check input', (el) => el.click());
await pause(80);
r = await readout();
check('skipping the masked half roughly doubles it (59,391)', agrees(r, 65536, { ...swi, causalSkip: true }, 'prefill') && r.cross === 59391, JSON.stringify(r));
await clickText(EX, 'one decode token');
r = await readout();
const skipDisabled = await page.$eval(EX + ' label.check input', (el) => el.disabled);
check('one decode token: same crossover as prefill without skipping, skipping disabled', agrees(r, 65536, { ...swi, causalSkip: true }, 'decode') && r.cross === 29696 && skipDisabled, JSON.stringify(r));
{
  const b = await page.$eval(EX + ' [data-bytes]', (e) => ({ w: Number(e.querySelector('[data-wbytes]').dataset.wbytes), c: Number(e.querySelector('[data-cbytes]').dataset.cbytes), x: Number(e.querySelector('[data-bcross]').dataset.bcross) })).catch(() => null);
  const want = decodeBytes(65536, { ...swi, causalSkip: true }, 1, 2);
  check('decode shows weight and cache bytes from the model, crossing at b × N = the FLOP crossover', b && b.w === want.weights && b.c === want.cache && b.x === Math.round(want.crossBatchN) && b.x === 29696, JSON.stringify(b));
}
await clickText(EX, 'prefill of N tokens');
check('prefill hides the decode bytes readout', (await page.$(EX + ' [data-bytes]')) === null);
await clickText(EX, 'one decode token');
await setSelect(EX + ' select[aria-label="d_model"]', 512);
await setRange(EX + ' input[aria-label="context length (log scale)"]', 0);
r = await readout();
check('d_model 512, N = 1: the products are a sliver', r.n === 1 && agrees(r, 1, { ...swi, dModel: 512, causalSkip: true }, 'decode') && r.ratio < 0.001, JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-explorer.png' });

// ---- questions
const q = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const typeIn = async (part, v) => {
  const h = await q(part);
  await (await h.$('input[type=number]')).type(String(v));
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
const pick = async (part, label) => {
  const h = await q(part);
  await h.evaluate((e, label) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
let t = await typeIn('divided by the FLOPs of the linear layers', 0.1667);
check('q-flops-share: 1/6 is right', t.includes('Correct.'));
t = await typeIn('DeepSeek-V2 has 128', 16384);
check('q-hidden: 16384 is right', t.includes('Correct.'));
t = await pick('LayerNorm (Fig. 20.1', "M4's row softmax");
check('q-layernorm: M4 is right', t.includes('Correct.'));
t = await pick('cache K after the rotation', 'Rotate every cached key');
check('q-rope: re-rotating the cache is wrong, with the reason', t.includes('Not quite.') && t.includes('rotated once'), t.slice(0, 120));
await pause(600);
check('answers write progress (intercepted, never to disk)', puts >= 1, String(puts));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
