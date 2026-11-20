// Browser check for appendix A section A.3: the bit-pattern classifier (Fig. A.8), the
// "What comes out?" table and the softmax overflow explorer, each against its model
// (src/sim/fpadd.ts, src/sim/softmaxnum.ts, src/sim/minifloat.ts, imported in the page).
// Usage: node e2e/appA-sa3.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const C = bench('Classify a bit pattern');
const W = bench('What comes out?');
const S = bench('Softmax overflow');
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
const pressPreset = async (scope, id) => {
  const ok = await page.evaluate((scope, id) => {
    const b = document.querySelector(`${scope} button[data-preset="${id}"]`);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, id);
  await pause(80);
  return ok;
};
const panel = () => page.$eval(C + ' .panel', (e) => ({ cls: e.dataset.cls, value: e.dataset.value, text: e.textContent }));
const setText = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};

await goto(page, 'appA/A.3');
await page.waitForSelector(C);

// ---- the models, as the page's own code sees them
const model = await page.evaluate(async () => {
  const fp = await import('/src/sim/fpadd.ts');
  const mf = await import('/src/sim/minifloat.ts');
  const presets = {};
  for (const f of fp.CLASSIFIER_FORMATS)
    presets[f.id] = fp.classifierPresets(f).map((p) => ({ id: p.id, bits: p.bits, cls: p.bits === null ? null : mf.decode(f, p.bits).cls, value: p.bits === null ? null : String(mf.decode(f, p.bits).value) }));
  return {
    presets,
    table: fp.specialTable().map((r) => ({ id: r.id, outcome: r.outcome, value: typeof r.value === 'boolean' ? String(r.value) : mf.show(r.value) })),
    nanFp32: (() => { const c = fp.patternCounts(mf.FP32); return c.qnan + c.snan; })(),
  };
});

// ---- classifier
let p = await panel();
check('starts on FP32 1.0, a normal number', p.cls === 'normal' && p.value === '1', JSON.stringify(p));
let allPresets = true;
for (const pr of model.presets.fp32) {
  await pressPreset(C, pr.id);
  p = await panel();
  if (p.cls !== pr.cls || p.value !== pr.value) { allPresets = false; check(`FP32 preset ${pr.id}`, false, JSON.stringify([p.cls, p.value, pr])); }
}
check('every FP32 preset shows the class and value the model gives (zero, denormals, normal, infinity, quiet and signalling NaN)', allPresets);
await pressPreset(C, 'snan');
check('signalling NaN: the top mantissa bit is clear', (await panel()).text.includes('Top mantissa bit clear'));
check('the Fig. A.8 row for NaN is highlighted', await page.$eval(C + ' tr[data-fig="nan"]', (e) => e.getAttribute('style')?.includes('background')));
check('the FP32 NaN count is 2^24 − 2', (await page.$eval(C + ' [data-nan-count]', (e) => Number(e.dataset.nanCount))) === 16777214 && model.nanFp32 === 16777214);
// Toggle a bit: 1.0 with the top mantissa bit set is 1.5.
await pressPreset(C, 'one');
await page.$eval(C + ' button[data-bit="22"]', (e) => e.click());
await pause(80);
p = await panel();
check('flipping the top mantissa bit of 1.0 gives 1.5, and the hex field follows', p.value === '1.5' && (await page.$eval(C + ' input[aria-label="bits in hex"]', (e) => e.value)) === '0x3fc00000', JSON.stringify(p));
await setText(C + ' input[aria-label="bits in hex"]', '0x00000001');
p = await panel();
check('typing 0x00000001 gives the smallest denormal', p.cls === 'subnormal' && Number(p.value) === 2 ** -149, JSON.stringify(p));
// E4M3: no infinity; 448 breaks Fig. A.8.
await press(C, 'E4M3');
check('E4M3: the +∞ preset is disabled, with the reason given', (await page.$eval(C + ' button[data-preset="inf"]', (e) => e.disabled)) && (await page.$eval(C + ' [data-missing]', (e) => e.textContent)).includes('has no infinity'));
await pressPreset(C, 'max');
p = await panel();
check('E4M3 largest finite is 448, with exponent bits all ones', p.value === '448' && p.text.includes('all ones'), JSON.stringify(p));
check('and the panel says Fig. A.8 would call it NaN', (await page.$(C + ' [data-disagree]')) !== null);
let e4ok = true;
for (const pr of model.presets.e4m3) {
  const disabled = await page.$eval(`${C} button[data-preset="${pr.id}"]`, (e) => e.disabled);
  if (disabled !== (pr.bits === null)) e4ok = false;
}
check('E4M3 presets are enabled exactly where the model has a pattern', e4ok);
if (out) await (await page.$(C)).screenshot({ path: out + '-classifier.png' });
await press(C, 'E2M1');
check('E2M1: no NaN and no infinity', (await page.$eval(C + ' button[data-preset="nan"]', (e) => e.disabled)) && (await page.$eval(C + ' button[data-preset="inf"]', (e) => e.disabled)));
await press(C, 'E5M2');
await pressPreset(C, 'nan');
check('E5M2 NaN: quiet and signalling are not defined', (await panel()).text.includes('does not divide its NaN patterns'));

// ---- what comes out?
const rows = await page.$$eval(W + ' tbody tr[data-row]', (rs) => rs.map((r) => ({ id: r.dataset.row, outcome: r.dataset.outcome })));
check('twelve rows, with the outcomes the model computes', rows.length === 12 && rows.every((r, i) => r.id === model.table[i].id && r.outcome === model.table[i].outcome), JSON.stringify(rows));
check('reveal is disabled before any answer', await page.$eval(W + ' button.primary', (e) => e.disabled));
// Answer every row right except the last, which gets a wrong answer.
for (let i = 0; i < rows.length; i++) {
  const want = i === rows.length - 1 ? (rows[i].outcome === 'nan' ? 'zero' : 'nan') : rows[i].outcome;
  await page.$eval(`${W} tr[data-row="${rows[i].id}"] select`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, want);
}
await pause(80);
await press(W, 'Reveal');
check('the score counts 11 of 12', (await page.$eval(W + ' [data-score]', (e) => Number(e.dataset.score))) === 11);
const marks = await page.$$eval(W + ' [data-mark]', (ms) => ms.map((m) => m.dataset.mark));
check('each row is marked right or wrong', marks.filter((m) => m === 'right').length === 11 && marks[marks.length - 1] === 'wrong', marks.join(','));
check('each row\'s rule is shown under it after the reveal', (await page.$$(W + ' tr[data-rule]')).length === 12);
const shownVals = await page.$$eval(W + ' td[data-result]', (tds) => tds.map((t) => t.textContent.trim()));
check('the results shown are the model\'s: NaN for 0/0, +inf for FLT_MAX × 2, 1 for fmaxf(NaN, 1)', shownVals.every((v, i) => v === model.table[i].value) && shownVals[0] === 'NaN' && shownVals[11] === '1', shownVals.join(' | '));
if (out) await (await page.$(W)).screenshot({ path: out + '-table.png' });

// ---- softmax overflow
const setScale = async (lg) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, lg);
  await pause(100);
};
const setSubtract = async (on) => {
  await page.evaluate((S, on) => {
    const box = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('subtract')).querySelector('input');
    if (box.checked !== on) box.click();
  }, S, on);
  await pause(100);
};
const sm = () => page.$eval(S, (b) => {
  const t = b.querySelector('table');
  return {
    inf: t?.dataset.firstInf ?? null,
    nan: t?.dataset.firstNan ?? null,
    ok: b.querySelector('[data-ok]')?.dataset.ok,
    status: b.querySelector('[data-ok]')?.textContent ?? '',
    cells: [...b.querySelectorAll('td[data-q]')].map((c) => [c.dataset.q, c.dataset.v, c.dataset.mark]),
  };
});
const modelSm = (logits, lg, fmt, sub) => page.evaluate(async (logits, lg, fmt, sub) => {
  const m = await import('/src/sim/softmaxnum.ts');
  const f = m.SOFTMAX_FORMATS.find((x) => x.id === fmt);
  const t = m.softmaxTrace(logits, 2 ** lg, f, sub);
  return {
    inf: t.firstInf ? `${t.firstInf.index}:${t.firstInf.quantity}` : '',
    nan: t.firstNan ? `${t.firstNan.index}:${t.firstNan.quantity}` : '',
    ok: t.ok ? '1' : '0',
    t: t.rows.map((r) => String(r.t)),
    p: t.rows.map((r) => String(r.p)),
  };
}, logits, lg, fmt, sub);
const presetLogits = await page.evaluate(async () => (await import('/src/sim/softmaxnum.ts')).SOFTMAX_PRESETS);
const typical = presetLogits.find((x) => x.id === 'typical').logits;
const cellsOf = (s, q) => s.cells.filter((c) => c[0] === q).map((c) => c[1]);

let s = await sm();
check('FP32, typical row, scale 1: fine', s.ok === '1' && s.inf === '' && s.nan === '', JSON.stringify(s).slice(0, 200));
await press(S, 'FP16');
await setScale(1);
s = await sm();
let m = await modelSm(typical, 1, 'fp16', false);
check('FP16 at scale 2: the first infinity is element 4\'s term and the first NaN its probability, as the model says', s.inf === m.inf && s.nan === m.nan && m.inf === '3:t' && m.nan === '3:p', JSON.stringify([s.inf, s.nan, m.inf, m.nan]));
check('the terms and probabilities shown are the model\'s', JSON.stringify(cellsOf(s, 't')) === JSON.stringify(m.t) && JSON.stringify(cellsOf(s, 'p')) === JSON.stringify(m.p));
check('the infinite term and the NaN are marked in the fail colour, with a cause', s.cells.some((c) => c[0] === 't' && c[2] === 'inf') && s.cells.some((c) => c[0] === 'p' && c[2] === 'nan') && s.status.includes('overflows FP16') && s.status.includes('∞ / ∞'), s.status);
if (out) await (await page.$(S)).screenshot({ path: out + '-softmax-fail.png' });
await setSubtract(true);
s = await sm();
m = await modelSm(typical, 1, 'fp16', true);
check('subtracting the maximum removes the overflow', s.ok === '1' && s.inf === '' && m.ok === '1' && JSON.stringify(cellsOf(s, 'p')) === JSON.stringify(m.p));
await press(S, 'BF16');
await setScale(4);
await setSubtract(false);
s = await sm();
m = await modelSm(typical, 4, 'bf16', false);
check('BF16 at scale 16 overflows too, where the model says', s.inf === m.inf && s.inf === '3:t', JSON.stringify([s.inf, m.inf]));
await press(S, 'FP16');
await setScale(0);
await pressPreset(S, 'flat');
s = await sm();
const flat = presetLogits.find((x) => x.id === 'flat').logits;
m = await modelSm(flat, 0, 'fp16', false);
check('eight logits of 10 in FP16: the sum overflows at element 3, no term does, no NaN, all probabilities 0', s.inf === '2:s' && m.inf === '2:s' && s.nan === '' && cellsOf(s, 'p').every((v) => v === '0') && s.status.includes('No NaN, and still wrong'), JSON.stringify(s.status));
// Edit a logit: 60000 at scale 2 overflows the scaled logit; subtracting the max then gives ∞ − ∞.
await setText(S + ' input[aria-label="logit 1"]', '60000');
await setScale(1);
await setSubtract(true);
s = await sm();
check('a logit of 60000 at scale 2 in FP16: the scaled logit overflows and z − max is ∞ − ∞', s.inf === '0:z' && s.nan === '0:d' && s.status.includes('∞ − ∞'), JSON.stringify([s.inf, s.nan]));
await pressPreset(S, 'wide');
s = await sm();
check('very negative logits underflow to 0, harmlessly', s.ok === '1' && (await page.$eval(S, (b) => b.textContent)).includes('(underflow)'));

// ---- questions
const q = (needle) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), needle);
const qn = await q('How many of the 2');
await (await qn.$('input[type=number]')).type('16777214');
await (await qn.$('button.primary')).click();
await pause(120);
check('numeric question: 16,777,214 NaN patterns is accepted', (await qn.evaluate((e) => e.textContent)).includes('Correct.'));
const qs = await q('smallest integer logit');
await (await qs.$('input[type=number]')).type('11');
await (await qs.$('button.primary')).click();
await pause(120);
check('numeric question: 11 is rejected, the answer 12 is given', (await qs.evaluate((e) => e.textContent)).includes('The answer is 12'));
const qc = await q('output contains NaN');
await (await qc.$$('input[type=radio]'))[0].click();
await (await qc.$('button.primary')).click();
await pause(120);
check('choice question: an overflowed exp is the cause', (await qc.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
