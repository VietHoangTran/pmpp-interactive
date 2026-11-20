// Browser check for section X3.9: the "Compare two runs" explorer at four perturbation sizes,
// the narrated scene "A test that fails late" (five parts, captions, scripts, every part's
// drawing against the model), the entry-point table and three questions. Expected values
// come from src/sim/tokencompare.ts, run here in Node.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx39.mjs [--dark] [screenshot-prefix]
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
const M = await import('../src/sim/tokencompare.ts');

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
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const C = 'section.bench[aria-label="Compare two runs"]';
const S = 'section.scene[aria-label="A test that fails late"]';
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};

await goto(page, 'x3/X3.9');
await page.waitForSelector(C);
await page.waitForSelector(S);

// ---------------------------------------------------------------- the explorer
const readout = () => page.$eval(C + ' [data-part="readout"]', (e) => ({ ...e.dataset, text: e.textContent.replace(/\s+/g, ' ') }));
const badCells = (row) => page.$$eval(`${C} [data-row="${row}"] td[data-bad="yes"]`, (els) => els.length);
const flaggedBars = () => page.$$eval(`${C} [data-part="margins"] rect[data-flagged="yes"]`, (els) => els.map((e) => Number(e.dataset.step)));

let r = await readout();
check('starts with all tokens matching and nothing flagged', r.first === '' && r.fr === '0' && r.flagged === '', JSON.stringify(r));
for (const e of [-5, -3.2, -3, -2]) {
  await setRange(C + ' input[aria-label="perturbation size"]', e);
  const m = M.compare(10 ** e);
  r = await readout();
  const label = `s = 1e${e}`;
  check(`${label}: first divergence ${m.firstDivergence}, ${m.frDiffers.length} free-running tokens differ`,
    r.first === String(m.firstDivergence ?? '') && Number(r.fr) === m.frDiffers.length, JSON.stringify(r));
  check(`${label}: teacher forced differs at ${m.tfDiffers.join(',') || 'no step'}, flagged ${m.flagged.join(',')}`,
    r.tf === m.tfDiffers.join(',') && r.flagged === m.flagged.join(','), JSON.stringify(r));
  check(`${label}: the streams mark ${m.frDiffers.length} and ${m.tfDiffers.length} cells`,
    (await badCells('fr')) === m.frDiffers.length && (await badCells('tf')) === m.tfDiffers.length);
  check(`${label}: the margin chart flags the same steps`, (await flaggedBars()).join(',') === m.flagged.join(','));
  check(`${label}: the largest NMSE shown is the model's`, Math.abs(Number(r.maxNmse) / m.maxTfNmse - 1) < 1e-9, r.maxNmse);
  if (e === -5) {
    check('at 1e-5 a token flips while NMSE is far under 1e-7', m.maxTfNmse < M.NMSE_LLAMACPP / 1000 && r.text.includes('under llama.cpp'), r.text);
    if (out) await page.screenshot({ path: out + '-compare.png' });
  }
}
await setRange(C + ' input[aria-label="perturbation size"]', -3.2);
check('a step can be flagged without flipping (13 at 10^-3.2)', (await readout()).flagged === '13,19' && (await readout()).tf === '19');

// ---------------------------------------------------------------- the scene
const N = 5;
const LABELS = ['Two runs', 'Wide margins', 'One close step', 'Free running', 'Teacher forced'];
const res = await fetch(BASE + '/content/x3/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x3/narration/late-divergence-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
const labels = await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()));
check('five parts with their labels', JSON.stringify(labels) === JSON.stringify(LABELS), labels.join('|'));
const capBox = () => page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
check('captions are off by default', (await capBox()) === false);
const go = async (k) => {
  await page.evaluate((S, k) => document.querySelectorAll(S + ' .seg button')[k].click(), S, k);
  await pause(150);
};
const svg = () => page.$eval(S + ' svg', (e) => ({ ...e.dataset }));
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
const seek = (v) => setRange(S + ' input[type=range]', v);
const sceneBad = (row) => page.$$eval(`${S} rect[data-row="${row}"][data-bad="yes"]`, (els) => els.map((e) => Number(e.dataset.t)));
const SC = M.compare(M.SCENE_S);
const FLIP = SC.tfDiffers[0];
check('the scene model has exactly one flip, at step 19', SC.tfDiffers.join(',') === '19' && FLIP === 19);

for (let k = 0; k < N; k++) {
  await go(k);
  const hasAudio = !!manifest?.clips?.[`late-divergence-${k + 1}`];
  if (!hasAudio) check(`part ${k + 1}: without audio its script is shown`, (await caption()) === scripts[k], await caption());
  await seek(1000);
  const d = await svg();
  const row = SC.steps[Number(d.step)];
  const gpuTok = d.gpu === 'fr' ? row.frTok : row.tfTok;
  check(`part ${k + 1}: tokens drawn are the model's at step ${d.step}`, Number(d.refTok) === row.refTok && Number(d.gpuTok) === gpuTok, JSON.stringify(d));
  if (k === 0 || k === 1) check(`part ${k + 1}: no token differs yet`, (await sceneBad(1)).length === 0 && Number(d.step) < FLIP);
  if (k === 2) {
    check('part 3: the close step, the GPU takes the runner-up', Number(d.step) === FLIP && Number(d.gpuTok) !== Number(d.refTok));
    if (out) await page.screenshot({ path: out + '-scene-flip.png' });
  }
  if (k === 3) check('part 4: free running, every token from 19 on differs', (await sceneBad(1)).join(',') === SC.frDiffers.join(','), (await sceneBad(1)).join(','));
  if (k === 4) {
    check('part 5: teacher forced, only step 19 differs', d.gpu === 'tf' && (await sceneBad(2)).join(',') === SC.tfDiffers.join(','), (await sceneBad(2)).join(','));
    check('part 5: the flag is drawn', (await page.$eval(S + ' svg', (e) => e.textContent)).includes('flagged'));
  }
}
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), S);
await pause(100);
check('the captions switch turns on and is remembered', (await capBox()) === true && (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');
await page.evaluate(() => localStorage.setItem('captions', 'off'));
// It plays by itself (silently: no audio, or sound off).
await go(0);
await page.evaluate((S) => {
  const b = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input');
  if (b.checked) b.click();
}, S);
await page.$eval(S + ' button.primary', (b) => b.click());
await pause(1500);
check('it plays: the position moves', Number(await page.$eval(S + ' input[type=range]', (e) => e.value)) > 0);
await page.$eval(S + ' button.primary', (b) => b.click());

// ---------------------------------------------------------------- the table and questions
const entry = await page.$eval('[data-part="entry"]', (e) => e.textContent);
check('the entry-point table names all four entry points', ['mx.fast.metal_kernel', 'mx.fast.cuda_kernel', 'torch.mps.compile_shader', 'load_inline'].every((s) => entry.includes(s)));
const answerNumeric = async (needle, v) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
const answerChoice = async (needle, idx) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), idx);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-nmse accepts 0.0001', await answerNumeric('What NMSE does test-backend-ops report', 0.0001));
check('q-cuda-grid: the grid counts threads', await answerChoice('as you would launch a CUDA kernel', 0));
check('q-ai-policy: an AI-written PR description is not allowed', !(await answerChoice('which of these is allowed', 1)));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
