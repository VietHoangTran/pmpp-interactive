// Browser check for the narrated animation "One decode step as the batch grows" in section
// 22.5: six parts, captions are an option (off by default, remembered), parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the model
// (src/sim/batching.ts, imported directly: lab 22.2's step, float32, the book's example
// threshold of 20 FLOP/B, in passes over the weights). The numbers the scripts say are the
// model's too. It passes with or without content/ch22/narration/manifest.json.
// Usage: node e2e/ch22-scene-decode-step.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { M8_DEFAULT, precisionFor, relativeStep, stepBStar } from '../src/sim/batching.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const near = (a, b, tol = 0.0015) => Math.abs(a - b) <= tol;
const S = 'section.scene[aria-label="One decode step as the batch grows"]';
const N = 6;
const LABELS = ['Batch 1', 'Batch 8', 'Per-row work', 'The knee', 'Two curves', 'Your device'];

// ---- the model
const T = 20;
const f32 = precisionFor(4);
const rel = (B, pad = 1) => relativeStep(M8_DEFAULT, B, f32, pad, T);
const BSTAR = stepBStar(M8_DEFAULT, f32, { peak: T, bw: 1 });
const r3 = (v) => Math.round(v * 1000) / 1000;

const res = await fetch(BASE + '/content/ch22/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch22/narration/decode-step-${n}.txt`)).text()));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const current = () => page.$$eval(S + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
const mainButton = () => page.$eval(S + ' button.primary', (e) => e.textContent.trim());
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused, t: a.currentTime }));
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const q = (sel) => svg.querySelector(sel);
  const g = q('[data-step]');
  const num = (el, k) => (el && el.dataset[k] !== undefined && el.dataset[k] !== '' ? Number(el.dataset[k]) : null);
  return {
    batch: svg.dataset.batch === '' ? null : Number(svg.dataset.batch),
    headline: q('[data-headline]')?.textContent.trim() ?? '',
    mem: num(g, 'mem'), comp: num(g, 'comp'), row: num(g, 'row'), step: num(g, 'step'), gemm: num(g, 'gemm'),
    tokens: q('[data-tokens]') ? Number(q('[data-tokens]').dataset.tokens) : null,
    busy: num(q('[data-busy]'), 'busy'),
    knees: [...svg.querySelectorAll('[data-knee]')].map((e) => Number(e.dataset.knee)),
    curves: Object.fromEntries([...svg.querySelectorAll('[data-curve]')].map((e) => [e.dataset.curve, { upto: Number(e.dataset.upto), first: Number(e.dataset.first), v128: Number(e.dataset.v128), last: Number(e.dataset.last) }])),
    text: svg.textContent,
  };
}, S);

await goto(page, 'ch22/22.5');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the model\'s: batch 1 and 8, the knee at about ' + Math.round(BSTAR) + ' at 20 operations per byte, tiles of 128 and 16',
  scripts[0].includes('batch 1') && scripts[1].includes('8 requests') && scripts[1].includes('8 times busier')
  && scripts[3].includes(`about ${Math.round(BSTAR)} rows`) && scripts[3].includes('threshold of 20 operations per byte')
  && scripts[5].includes('128 rows') && scripts[5].includes('steps of 16')
  && near(rel(8).compute / rel(1).compute, 8, 1e-9) && rel(8).step < 1.05,
  JSON.stringify({ bstar: BSTAR }));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['decode-step-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));

// ---- runs with the sound off, and parts advance by themselves
await setBox('sound', false);
let d = await drawn();
check('before playing, part 1 shows batch 1 with no token out yet', d.batch === 1 && d.tokens === 0, JSON.stringify(d));
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(985);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part against the model
await press('1. Batch 1');
await seek(1000);
d = await drawn();
const r1 = rel(1);
check('part 1 ends: one pass over the weights, arithmetic busy 2.5% of the GEMM time, one token out',
  d.batch === 1 && near(d.mem, r3(r1.memory)) && near(d.comp, r3(r1.compute)) && near(d.comp, 2 / (T * 4)) && near(d.step, r3(r1.step)) && d.tokens === 1 && near(d.busy, r3(r1.compute / r1.gemm)),
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Batch 8');
await seek(500);
d = await drawn();
check('part 2 midway: the weights are still streaming and no tokens are out yet', d.batch === 8 && d.tokens === 0, JSON.stringify(d));
await seek(1000);
d = await drawn();
const r8 = rel(8);
check('part 2 ends: the same weight stream (within 1%), 8 times the arithmetic, 8 tokens out in almost the same step',
  d.batch === 8 && near(d.mem, r3(r8.memory)) && d.mem / r3(r1.memory) < 1.011 && near(d.comp, 0.2) && d.tokens === 8 && d.step < 1.05,
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Per-row work');
await seek(0);
const first = await drawn();
await seek(1000);
d = await drawn();
check('part 3 doubles the batch from 1 to 32 and the per-row band grows with it',
  first.batch === 1 && d.batch === 32 && near(d.row, r3(rel(32).row)) && d.row > 20 * first.row && d.step < 1.25,
  JSON.stringify({ first: [first.batch, first.row], last: [d.batch, d.row, d.step] }));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The knee');
await seek(0);
const k0 = await drawn();
await seek(1000);
d = await drawn();
const r256 = rel(256);
check('part 4 goes from 32 (under the knee) to 256, where arithmetic outlasts the weight stream and the step grows with B',
  k0.batch === 32 && k0.comp < k0.mem && d.batch === 256 && near(d.comp, r3(r256.compute)) && d.comp > d.mem && near(d.step, r3(r256.step)) && d.busy === 1
  && d.headline.includes(`B* = ${BSTAR.toFixed(1)}`),
  JSON.stringify({ k0: [k0.batch, k0.comp, k0.mem], d }));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Two curves');
await seek(1000);
d = await drawn();
check('part 5 ends: both curves drawn to batch 1024 from the model, the knee marked at B*',
  d.curves.tokens?.upto === 1024 && near(d.curves.tokens.last, r3(rel(1024).tokensPerUnit), 0.002) && near(d.curves.perToken.last, r3(rel(1024).step), 0.002)
  && d.knees.length === 2 && d.knees.every((k) => near(k, Number(BSTAR.toFixed(1)), 1e-9)),
  JSON.stringify(d.curves) + ' ' + d.knees);
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Your device');
await seek(1000);
d = await drawn();
const c = d.curves;
check('part 6 ends: with a 128-row tile the step is nearly flat to 128 (the per-row band only), without it it has more than tripled',
  near(c.pad128.first, r3(rel(1, 128).step)) && c.pad128.v128 / c.pad128.first < 1.2 && c.pad1.v128 / c.pad1.first > 3 && near(c.pad16.first, c.pad1.first),
  JSON.stringify(c));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('4. The knee');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part4-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch22\/narration\/(manifest\.json|decode-step-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
