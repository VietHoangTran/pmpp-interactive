// Browser check for section X1.2, "Temperature, softmax and greedy": the Temperature
// explorer (every row, both views, float32, the T slider, clicking bars, both tie rules and
// the greedy button), lab 10.3's softmax listing, the ratio Predict and the questions.
// Expected values come from src/sim/chain.ts, run inside the page through the app's
// module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx12.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const B = 'section.bench[aria-label="Temperature"]';
const clickIn = async (group, label) => {
  const ok = await page.evaluate((B, group, label) => {
    const b = [...document.querySelectorAll(`${B} [aria-label="${group}"] button`)].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, B, group, label);
  await pause(80);
  return ok;
};
const T_MIN = 0.05, T_MAX = 4;
const toSlider = (T) => Math.round((Math.log(T / T_MIN) / Math.log(T_MAX / T_MIN)) * 1000);
const tFrom = (v) => Number((T_MIN * Math.pow(T_MAX / T_MIN, v / 1000)).toPrecision(2));
const setT = async (T) => {
  await page.$eval(`${B} input[aria-label="T"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, toSlider(T));
  await pause(80);
  return tFrom(toSlider(T));
};
const setFloat = async (on) => {
  await page.evaluate((B, on) => {
    const i = [...document.querySelectorAll(`${B} label.check`)].find((l) => l.textContent.includes('float32')).querySelector('input');
    if (i.checked !== on) i.click();
  }, B, on);
  await pause(80);
};
const bars = () => page.$$eval(`${B} g[data-token]`, (gs) => gs.map((g) => Number(g.dataset.prob)));
const readout = (k) => page.$eval(`${B} [data-readout="${k}"]`, (e) => ({ ...e.dataset, text: e.textContent.replace(/\s+/g, ' ') }));
const clickBar = async (i) => {
  await page.$eval(`${B} g[data-token="${i}"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(80);
};
const model = (rowId, T, float32) =>
  page.evaluate(async (rowId, T, float32) => {
    const m = await import('/src/sim/chain.ts');
    const l = m.TEMP_ROWS.find((r) => r.id === rowId).logits;
    const p = m.softmax(l, 1 / T, float32);
    return { logits: l, probs: p, top: Math.max(...p), H: m.entropyBits(p), argmax: m.argmax(l, 'lowest'), argmaxHigh: m.argmax(l, 'highest'), mismatch: float32 ? m.reciprocalMismatch(l, T).length : 0, pair: m.OPENING_PAIR };
  }, rowId, T, float32);
const close = (a, b, rel = 1e-5) => Math.abs(a - b) <= rel * Math.max(Math.abs(b), 1e-12);

await goto(page, 'x1/X1.2');
await page.waitForSelector(B);

// ---------------------------------------------------------------- opening state: T = 0.5, the pair 1 apart
let M = await model('peaked', 0.5, false);
let r = await readout('ratio');
check('the explorer opens at T = 0.5 on the pair whose logits differ by exactly 1', (await page.$eval(B, (e) => e.textContent)).includes('T = 0.50') && r.text.includes(`p(token ${M.pair[0]}) / p(token ${M.pair[1]})`) && M.logits[M.pair[0]] - M.logits[M.pair[1]] === 1, r.text);
check('the ratio reads e² = 7.389', close(Number(r.ratio), Math.exp(2), 1e-5) && r.text.includes('7.389'), r.ratio);
let got = await bars();
check('every bar is the model\'s probability (double)', got.every((p, i) => close(p, M.probs[i])), JSON.stringify(got.slice(0, 4)));
check('top-1 and entropy readouts match the model', (await readout('top1')).text.includes(M.top.toFixed(4)) && (await readout('entropy')).text.includes(M.H.toFixed(3)));
if (out) await page.screenshot({ path: out + '-open.png' });

// ---------------------------------------------------------------- T moves; the argmax does not
const tallest = (ps) => ps.indexOf(Math.max(...ps));
const seen = [];
for (const T of [0.05, 0.2, 1, 2.5, 4]) {
  const t = await setT(T);
  M = await model('peaked', t, false);
  got = await bars();
  seen.push(tallest(got));
  check(`peaked row at T = ${t}: bars match the model`, got.every((p, i) => close(p, M.probs[i])));
}
check('the tallest bar never moves with T', seen.every((i) => i === M.argmax), seen.join(' '));
await setT(0.05);
check('near T = 0 the peaked row is one-hot', Number((await readout('top1')).text.split(' ').pop()) > 0.999);

// ---------------------------------------------------------------- the tied row
check('tied row button', await clickIn('Row', 'two tied maxima'));
M = await model('tied', 0.05, false);
got = await bars();
check('near T = 0 the two tied maxima split the mass evenly', close(got[1], 0.5, 1e-3) && got[1] === got[4], `${got[1]} ${got[4]}`);
await page.evaluate((B) => [...document.querySelectorAll(`${B} button`)].find((b) => b.textContent.trim() === 'greedy').click(), B);
await pause(80);
check('greedy, lowest index: token 1', (await readout('greedy')).greedy === String(M.argmax) && M.argmax === 1);
check('tie rule button (highest)', await clickIn('Tie rule', 'highest index (llama.cpp Metal)'));
await page.evaluate((B) => [...document.querySelectorAll(`${B} button`)].find((b) => b.textContent.trim() === 'greedy').click(), B);
await pause(80);
r = await readout('greedy');
check('greedy, highest index: token 4, and the readout says the rules disagree here', r.greedy === String(M.argmaxHigh) && M.argmaxHigh === 4 && r.text.includes('disagree'), r.text);
if (out) await page.screenshot({ path: out + '-tied.png' });

// ---------------------------------------------------------------- the flat row, clicking bars, the logits view
check('flat row button', await clickIn('Row', 'flat'));
let t = await setT(1.5);
await clickBar(2);
await clickBar(9);
M = await model('flat', t, false);
r = await readout('ratio');
check('clicking two bars compares them, as exp(difference / T)', r.text.includes('p(token 2) / p(token 9)') && close(Number(r.ratio), Math.exp((M.logits[2] - M.logits[9]) / t), 1e-5), r.text);
await page.evaluate((B) => [...document.querySelectorAll(`${B} button`)].find((b) => b.textContent.trim() === 'greedy').click(), B);
await pause(80);
check('on the flat row both tie rules agree', (await readout('greedy')).text.includes('agree'));
check('logits view button', await clickIn('Show', 'logits / T'));
const labels = await page.$$eval(`${B} g[data-token] text:last-child`, (e) => e.map((x) => Number(x.textContent)));
check('the logits view shows l / T', labels.every((v, i) => Math.abs(v - M.logits[i] / t) < 0.051), JSON.stringify(labels.slice(0, 4)));
await clickIn('Show', 'probabilities');

// ---------------------------------------------------------------- float32, and divide vs multiply by 1/T
await clickIn('Row', 'peaked');
await setFloat(true);
for (const T of [3, 2]) {
  t = await setT(T);
  M = await model('peaked', t, true);
  got = await bars();
  check(`float32 at T = ${t}: bars are the model's float32 softmax`, got.every((p, i) => close(p, M.probs[i], 1e-6)));
  r = await readout('reciprocal');
  check(`float32 at T = ${t}: ${M.mismatch} logits differ between dividing and multiplying by 1/T`, Number(r.mismatch) === M.mismatch, r.text);
}
check('at T = 2 the reciprocal is exact: no difference', Number((await readout('reciprocal')).mismatch) === 0);
await setT(3);
check('at T = 3 at least one logit differs in the last bit', Number((await readout('reciprocal')).mismatch) > 0);
if (out) await page.screenshot({ path: out + '-float32.png' });
await setFloat(false);
check('float32 off removes the reciprocal readout', (await page.$(`${B} [data-readout="reciprocal"]`)) === null);

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has CUDA and Metal tabs', tabs.includes('CUDA') && tabs.includes('Metal kernel'), tabs.join('|'));
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check("the CUDA tab is lab 10.3's softmaxRowsKernel and its launcher", code.includes('__global__ void softmaxRowsKernel') && code.includes('void launchSoftmaxRows'));
check('the scale lines are marked', (await page.$$eval('figure.code .ln.hl', (e) => e.filter((x) => x.textContent.includes('scale')).length)) >= 3);
check('the CUDA tab states its status in the course vocabulary', (await page.$eval('figure.code .status', (e) => e.textContent)).includes('front-end-checked'));

// ---------------------------------------------------------------- the Predict and the questions
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
const pq = await qHandle('ratio of their probabilities');
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  await (await pq.$('input[type=number]')).type('7.389');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
check('the ratio reveal gives e² = 7.389', (await page.$eval('[data-reveal="ratio"]', (e) => e.textContent)).includes('7.389'));
async function answer(text, how, numeric = false) {
  const q = await qHandle(text);
  if (!numeric) await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), how);
  else {
    const input = await q.$('input[type=number]');
    await input.click({ clickCount: 3 });
    await input.type(String(how));
  }
  await pause(50);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
check('q-tie-limit: [0.5, 0.5, 0]', (await answer('two exact maxima', 0)).startsWith('Correct.'));
check('q-tie-limit: [1, 0, 0] is wrong', (await answer('two exact maxima', 1)).startsWith('Not quite.'));
check('q-fold-bytes: 8', (await answer('its own elementwise kernel', 8, true)).startsWith('Correct.'));
check('q-reciprocal: yes, in the last bit', (await answer('precomputed 1/T', 0)).startsWith('Correct.'));
check('q-greedy-ties: 3, 3 and 7', (await answer('equal maxima at columns 3 and 7', 0)).startsWith('Correct.'));
check('q-softmax-share: none of it', (await answer('which softmax work dominates', 0)).startsWith('Correct.'));

const todays = await page.$$eval('aside.note.today', (a) => a.map((x) => x.textContent));
check('every Today note was checked in the facts pass', todays.length >= 3 && todays.every((x) => x.includes('Checked 2026-11-20 in the facts pass')));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
