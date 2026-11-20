// Browser check for section X1.7: the "Philox round by round" stepper (the three known-answer
// presets, typed inputs, the stream + 1 column, the PyTorch MPS reading), the "Bits to u"
// explorer (jumps, typed x, the binade strip and its counts), the avalanche prediction, the
// questions and the code tabs. Every number asserted is recomputed here from the model
// (src/sim/philox.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx17.mjs [--dark] [screenshot-prefix]
import { open, goto } from './browser.mjs';

const P = await import('../src/sim/philox.ts');

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
await page.setRequestInterception(true);
const puts = [];
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts.push(r.postData());
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const S = bench('Philox round by round');
const X = bench('Bits to u');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(100);
  return done;
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    const proto = el.tagName === 'INPUT' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const rowWords = (round, col) => page.$$eval(`${S} tr[data-round="${round}"] [data-w^="${col}-"]`, (e) => e.map((x) => x.dataset.value));
const rowDiffs = (round) => page.$$eval(`${S} tr[data-round="${round}"] [data-w^="b-"]`, (e) => e.map((x) => x.dataset.diff === '1'));

await goto(page, 'x1/X1.7');
await page.waitForSelector(S);

// ---------------------------------------------------------------- the stepper
const zA = P.m5Input(0n, 0n, 0n);
const zT = P.philoxTrace(zA.ctr, zA.key);
const zAv = P.avalanche(zA.ctr, P.m5Input(0n, 1n, 0n).ctr, zA.key);
check('stepper opens on seed, stream and step 0, nothing run', (await text(S + ' .narrate')).startsWith('Nothing has run yet') && (await page.$eval(S + ' [data-step]', (e) => e.dataset.step)) === '0');
await press(S, 'Step');
check("step 1: round 1's words, right column differs in word 1 only",
  JSON.stringify(await rowWords(1, 'a')) === JSON.stringify(zT[0].out.map(P.hex)) && JSON.stringify(await rowDiffs(1)) === JSON.stringify(zAv.masks[0]));
check('step 1: the detail panel shows both products', (await text(S + ' [data-part=detail]')).includes(P.hex(P.M0)) && (await text(S + ' [data-part=detail]')).includes(P.hex(P.M1)));
for (let r = 2; r <= 5; r++) await press(S, 'Step');
const diffs = await page.$$eval(S + ' tr[data-differing]', (e) => e.map((x) => Number(x.dataset.differing)));
check('rounds 1 to 5: 1, 1, 2, 3, 4 words differ', JSON.stringify(diffs) === '[1,1,2,3,4]', JSON.stringify(diffs));
await press(S, 'Run to the end');
const res = await page.$eval(S + ' [data-part=result]', (e) => ({ kat: e.dataset.kat, ok: e.dataset.katOk, all: e.dataset.allBy }));
check("all zero, run to the end: Random123's zero vector, check mark", res.kat === 'zero' && res.ok === '1' && JSON.stringify(await rowWords(10, 'a')) === JSON.stringify(P.KATS[0].expected.map(P.hex)), JSON.stringify(res));
check('all four words differ from round 5 on', res.all === '5');
check("M5's u for the row is the model's", Number(await page.$eval(S + ' [data-u]', (e) => e.dataset.u)) === P.m5U(0n, 0n, 0n));
if (out) await page.screenshot({ path: out + '-stepper.png' });

for (const k of P.KATS.slice(1)) {
  await press(S, k.label);
  await press(S, 'Run to the end');
  const r = await page.$eval(S + ' [data-part=result]', (e) => ({ kat: e.dataset.kat, ok: e.dataset.katOk }));
  check(`preset "${k.label}": ten rounds give the published vector`, r.kat === k.id && r.ok === '1' && JSON.stringify(await rowWords(10, 'a')) === JSON.stringify(k.expected.map(P.hex)), JSON.stringify(r));
}

// typed inputs: a large stream, any step
await setInput(S + ' input[aria-label=seed]', '42');
await setInput(S + ' input[aria-label=stream]', '1000000000000000000');
await setInput(S + ' input[aria-label=step]', '0x10');
await press(S, 'Run to the end');
const w = P.m5Words(42n, 10n ** 18n, 16n);
check('typed seed 42, stream 10^18, step 0x10: the words are the model\'s, no published vector',
  JSON.stringify(await rowWords(10, 'a')) === JSON.stringify(w.map(P.hex)) && (await page.$eval(S + ' [data-part=result]', (e) => e.dataset.kat)) === '');
check('the stream + 1 column is the model\'s', JSON.stringify(await rowWords(10, 'b')) === JSON.stringify(P.m5Words(42n, 10n ** 18n + 1n, 16n).map(P.hex)));
check('the row\'s u is the model\'s', Number(await page.$eval(S + ' [data-u]', (e) => e.dataset.u)) === P.m5U(42n, 10n ** 18n, 16n));
await setInput(S + ' input[aria-label=stream]', '-3');
check('a negative stream is refused with a message', !!(await page.$(S + ' [data-invalid]')));
await setInput(S + ' input[aria-label=stream]', '0');

// the PyTorch MPS reading
await page.$eval(S + ' input[aria-label="PyTorch MPS as we read its source"]', (e) => e.click());
await pause(150);
await press(S, 'Run to the end');
check('MPS reading: 9 rounds', (await page.$eval(S + ' [data-step]', (e) => e.dataset.step)) === '9');
const mres = await page.$eval(S + ' [data-part=result]', (e) => e.dataset.mpsOk);
check("MPS reading, seed 42 and stream 0: E3's transcribed words, check mark", mres === '1' && JSON.stringify(await rowWords(9, 'a')) === JSON.stringify(P.E3[1].mps.map(P.hex)));
check('the result says the ten-round words differ', (await text(S + ' [data-part=result]')).includes(P.E3[1].ten.map(P.hex).join(' ')));
await page.$eval(S + ' input[aria-label="PyTorch MPS as we read its source"]', (e) => e.click());
await pause(150);

// ---------------------------------------------------------------- bits to u
const conv = async () => page.$$eval(X + ' tr[data-conv]', (e) => Object.fromEntries(e.map((x) => [x.dataset.conv, [Number(x.dataset.u), x.dataset.bits]])));
const expectConv = (x) => ({ curand: P.curandUniform(x), hip: P.hipUniformAMD(x), m5: P.m5Uniform(x) });
for (const j of P.JUMPS) {
  await press(X, j.label);
  const c = await conv();
  const e = expectConv(j.x);
  const agree = await page.$eval(X + ' [data-agree]', (e) => e.dataset.agree);
  check(`x = ${j.label}: three floats and bit patterns as the model, vendors ${e.curand === e.hip ? 'agree' : 'differ'}`,
    c.curand[0] === e.curand && c.hip[0] === e.hip && c.m5[0] === e.m5 && c.curand[1] === P.hex(P.f32bits(e.curand)) && agree === (e.curand === e.hip ? '1' : '0'), JSON.stringify(c));
}
check('0xFFFFFF80: both vendors give exactly 1.0, M5 does not', (await text(X + ' [data-part=conversions]')).split('exactly 1.0').length === 3);
await setInput(X + ' input[aria-label=x]', '0x1000002');
const c2 = await conv();
check('typed x = 0x1000002 (2^24 + 2): the vendors differ, a tie rounded to even', c2.curand[0] !== c2.hip[0] && c2.hip[0] === P.hipUniformAMD(0x1000002));
const strip = await page.$eval(X + ' [data-differ]', (e) => [Number(e.dataset.differ), Number(e.dataset.ones), Number(e.dataset.onesHip)]);
check('the strip reports 2^24 differing inputs and 128 ones for each vendor', strip[0] === 2 ** 24 && strip[1] === 128 && strip[2] === 128, JSON.stringify(strip));
const shares = await page.$$eval(X + ' [data-binade]', (e) => e.map((x) => [Number(x.dataset.binade), Number(x.dataset.share)]));
check('binade shares: 1 below 2^23, 1/2 in [2^23, 2^24), 1/4 in [2^24, 2^25), 0 above',
  shares.length === 33 && shares.every(([k, s]) => (k <= 22 ? s === 1 : k === 23 ? s === 0.5 : k === 24 ? s === 0.25 : s === 0)), JSON.stringify(shares.slice(20, 27)));
const sample = Number(await page.$eval(X + ' [data-sample]', (e) => e.dataset.sample));
check("the seeded sample's count is the model's", sample === P.sampleConversions(200000, 1).differ);
if (out) await page.screenshot({ path: out + '-bits.png' });

// ---------------------------------------------------------------- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((q) => q.textContent.includes('After how many rounds')));
await (await pred.$('input')).type('4');
await (await pred.$('button.primary')).click();
await pause(300);
const reveal = await page.$eval('[data-reveal]', (e) => e.dataset.reveal).catch(() => null);
check('the avalanche prediction reveals 5', reveal === '5', String(reveal));

const answer = async (fragment, how) => {
  const q = await page.evaluateHandle((f) => [...document.querySelectorAll('.q')].find((x) => x.querySelector('.prompt')?.textContent.includes(f)), fragment);
  if (typeof how === 'number') (await q.$$('input[type=radio]'))[how].click();
  else await (await q.$('input[type=number]')).type(how);
  await pause(80);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((x) => x.querySelector('.verdict')?.className ?? '');
};
check('q-one: 128 is right', (await answer('return exactly 1.0', '128')).includes('right'));
check('q-differ: 256 is right', (await answer('1 in how many inputs', '256')).includes('right'));
check('q-stateless: the first option is right', (await answer('counter-based generator suit', 0)).includes('right'));
check('q-metal: a wrong option is marked wrong', (await answer('Metal Shading Language give', 1)).includes('wrong'));
check('answers were saved through PUT /api/progress (intercepted)', puts.length > 0);

// ---------------------------------------------------------------- code
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('philoxRound')), { timeout: 10000 }).catch(() => null);
const codeTabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('the code shows CUDA and Metal tabs of the snippet', codeTabs.includes('CUDA and HIP') && codeTabs.includes('Metal'), JSON.stringify(codeTabs));
const status = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
check('the CUDA tab uses the course vocabulary', status.includes('emulator-checked') && status.includes('unverified'), status);
const notice = await page.$eval('figure.code pre', (e) => e.textContent.includes('D. E. Shaw Research'));
check("Random123's notice is shown with the code", notice);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
