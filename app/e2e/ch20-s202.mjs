// Browser check for section 20.2: the "one head, every matrix" viewer against
// app/src/sim/attnref.ts (every cell of P and O, the narration, the mask, the scale, the max
// subtraction and the +90 overflow), plus questions.
// Usage: node e2e/ch20-s202.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { attention, differing, makeAttnProblem, maxAbsDiff } from '../src/sim/attnref.ts';

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
const V = 'section.bench[aria-label="One head, every matrix"]';
// The viewer's own number format.
const num = (x, digits = 2) => {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '∞';
  if (x === -Infinity) return '−∞';
  return Number(x.toFixed(digits)).toString().replace('-', '−');
};
const grid = (name) => page.$$eval(`${V} [data-grid="${name}"] span`, (s) => s.map((x) => ({ r: +x.dataset.r, c: +x.dataset.c, t: x.textContent, cls: x.className })));
const sameAs = (cells, M, digits = 2) => cells.every((x) => x.t === num(M[x.r][x.c], digits));
const narr = () => page.$eval(V + ' [data-narrate]', (e) => e.textContent);
const toggle = async (label) => {
  await page.evaluate((V, label) => [...document.querySelectorAll(V + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input').click(), V, label);
  await pause(80);
};
const clickText = async (label) => {
  await page.evaluate((V, label) => [...document.querySelectorAll(V + ' button')].find((b) => b.textContent.trim().startsWith(label)).click(), V, label);
  await pause(80);
};
const setRange = async (label, v) => {
  await page.$eval(`${V} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const pickCell = async (name, r, c) => {
  await page.$eval(`${V} [data-grid="${name}"] span[data-r="${r}"][data-c="${c}"]`, (e) => e.click());
  await pause(60);
};

await goto(page, 'ch20/20.2');
await page.waitForSelector(V);

// ---- default: N 6, d 4, seed 1, everything on
let prob = makeAttnProblem(6, 4, 1);
let res = attention(prob, { precision: 'f32' });
let P = await grid('P');
check('P has 6 × 6 cells and every one matches the model (float32)', P.length === 36 && sameAs(P, res.P), JSON.stringify(P.slice(0, 6).map((x) => x.t)));
check('O matches the model', sameAs(await grid('O'), res.O));
check('Q and K^T show the problem\'s values', sameAs(await grid('Q'), prob.Q, 1) && (await grid('KT')).every((x) => x.t === num(prob.K[x.c][x.r], 1)));
const rowsLive = [0, 1, 2, 3, 4, 5].map((i) => P.filter((x) => x.r === i && !x.cls.includes('void')).length);
check('row i of P has i + 1 non-zero cells; the rest are dashed zeros', JSON.stringify(rowsLive) === '[1,2,3,4,5,6]', JSON.stringify(rowsLive));
let n = await narr();
check('the narration writes out P[3][1] with its dot product and value', n.includes('P[3][1]') && n.includes('= ' + num(prob.Q[3].reduce((s, q, k) => s + q * prob.K[1][k], 0), 2)) && n.includes('p = ' + num(res.P[3][1], 4)), n);
await pickCell('P', 1, 4);
n = await narr();
check('a masked cell explains why it is exactly 0', n.includes('P[1][4] is masked') && n.includes('exactly 0') && n.includes('Row 1 has 2 non-zero entries'), n);
await pickCell('S', 4, 2);
n = await narr();
check('picking a cell of S explains the matching cell of P', n.includes('P[4][2]') && n.includes('p = ' + num(res.P[4][2], 4)), n);
if (out) await page.screenshot({ path: out + '-viewer.png' });

// ---- the max: off changes nothing at ordinary logits...
await toggle('subtract the row max');
res = attention(prob, { precision: 'f32', subtractMax: false });
P = await grid('P');
check('without the max subtraction, at ordinary logits P is the same', sameAs(P, res.P) && P.every((x) => x.t !== 'NaN'));
// ...and overflows at +90
await clickText('Offset every logit by +90');
res = attention(prob, { precision: 'f32', subtractMax: false, offset: 90 });
P = await grid('P');
const nanModel = res.P.flat().filter(Number.isNaN).length;
const nanShown = Number(await page.$eval(V + ' [data-nan]', (e) => e.dataset.nan).catch(() => '0'));
check('+90 without the max: NaN cells exactly where the model overflows', nanModel > 0 && nanShown === nanModel && sameAs(P, res.P) && P.filter((x) => x.t === 'NaN').every((x) => x.cls.includes('bad')), `${nanShown} vs ${nanModel}`);
n = await narr();
check('the narration names the float32 limit', n.includes('beyond float32 above 88.72') || !n.includes('∞'), n);
if (out) await page.screenshot({ path: out + '-overflow.png' });
// ...and the max fixes it
await toggle('subtract the row max');
res = attention(prob, { precision: 'f32', offset: 90 });
P = await grid('P');
const base = attention(prob, { precision: 'f32' });
check('+90 with the max: no NaN, P as the model has it, and within 1e-4 of no offset', sameAs(P, res.P) && !(await page.$(V + ' [data-nan]')) && maxAbsDiff(res.P, base.P) < 1e-4);

// ---- the mask
await toggle('causal mask');
const off = attention(prob, { precision: 'f32', offset: 90, mask: false });
const leakModel = [...new Set(differing(off.O, res.O, 1e-6, true).map(([r]) => r))].join(',');
const leakShown = await page.$eval(V + ' [data-leak]', (e) => e.dataset.leak).catch(() => '');
check('without the mask, the leaking rows of O are the model\'s, never the last row', leakShown === leakModel && leakShown.includes('0') && !leakShown.split(',').includes('5'), `${leakShown} vs ${leakModel}`);
check('without the mask, P has no zeros left', (await grid('P')).every((x) => x.t !== '0'));
check('M shows no −∞ without the mask', (await grid('M')).every((x) => x.t === '0'));
await toggle('causal mask');
await clickText('Logits offset by +90');

// ---- the scale
await toggle('scale by 1/√d');
res = attention(prob, { precision: 'f32', scale: false });
check('without 1/√d, P is the model\'s unscaled softmax', sameAs(await grid('P'), res.P));
const peak = (M) => M.reduce((s, r) => s + Math.max(...r), 0);
check('and its rows are sharper', peak(res.P) > peak(base.P));
await toggle('scale by 1/√d');

// ---- size and new numbers
await setRange('N', 8);
await pickCell('P', 5, 0);
const info = await page.$eval(V + ' [data-rowinfo]', (e) => e.textContent);
check('N = 8: row 5 of P has 6 non-zero entries (q-row-nonzeros)', info.includes('Row 5 of P has 6 non-zero entries'), info);
await clickText('New numbers');
prob = makeAttnProblem(8, 4, 2);
check('"New numbers" gives the model\'s next problem', sameAs(await grid('P'), attention(prob, { precision: 'f32' }).P) && (await grid('P')).length === 64);
await setRange('d', 6);
prob = makeAttnProblem(8, 6, 2);
check('d = 6: Q is 8 × 6 and P still matches', (await grid('Q')).length === 48 && sameAs(await grid('P'), attention(prob, { precision: 'f32' }).P));

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
let t = await typeIn('How many MiB does S take', 64);
check('q-s-bytes: 64 MiB is right', t.includes('Correct.'));
t = await typeIn('How many non-zero entries does row 5', 6);
check('q-row-nonzeros: 6 is right', t.includes('Correct.'));
t = await pick('subtracting each row', "A row's shift scales");
check('q-shift: the per-row factor cancels', t.includes('Correct.'));
t = await pick('You forget the 1/√d', 'Nothing: softmax');
check('q-no-scale: "scale-invariant" is wrong, with the reason', t.includes('Not quite.') && t.includes('not a factor'), t.slice(0, 120));
t = await typeIn('emits “Processors”', 9);
check('q-rows-x: 9 rows', t.includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
