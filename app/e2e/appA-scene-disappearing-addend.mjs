// Browser check for the narrated animation "The disappearing addend" in section A.4: six
// parts, captions are an option (off by default, remembered), parts advance by themselves,
// it runs with the sound off, and every part's drawing agrees with the model
// (src/sim/fpadd.ts: alignTrace and runningSum in FP16, imported in the page). The numbers
// the scripts say are the model's too. It passes with or without
// content/appA/narration/manifest.json.
// Usage: node e2e/appA-scene-disappearing-addend.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

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
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const S = 'section.scene[aria-label="The disappearing addend"]';
const N = 6;
const LABELS = ['Two numbers', 'Align', 'Round', 'Keep adding', 'Stuck', 'Ways out'];

const res = await fetch(BASE + '/content/appA/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appA/narration/disappearing-addend-${n}.txt`)).text()));

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
const svgData = () => page.$eval(S + ' .stage svg', (e) => ({ ...e.dataset, headline: e.querySelector('[data-headline]')?.textContent ?? '', text: e.textContent }));
const grs = () => page.$eval(S + ' .stage svg', (e) => { const g = e.querySelector('[data-g]'); return g ? `${g.dataset.g}${g.dataset.r}${g.dataset.s}` : null; });

await goto(page, 'appA/A.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- the model
const model = await page.evaluate(async () => {
  const fp = await import('/src/sim/fpadd.ts');
  const mf = await import('/src/sim/minifloat.ts');
  const t = fp.alignTrace(mf.FP16, 1000, 0.3, 'rne');
  const c = t.steps[0];
  const shifts = t.steps.filter((s) => s.kind === 'shift');
  const r = t.steps.find((s) => s.kind === 'round');
  const run = fp.runningSum(mf.FP16, 1000, 0.3, 400);
  return {
    big: c.big.exp, small: c.small.exp, diff: c.diff, p: t.p, last: shifts.at(-1), result: r.result, below: r.below, above: r.above, err: r.errorUlps,
    acc: run.acc, exact: run.exact, stall: run.stallAt, fp32Window: mf.FP32.m + 1,
  };
});

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*^+]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*^+]/.test(s)));
check('the numbers the scripts say are the model\'s: exponents 9 and minus 2, 11 places, 1000.5, about 4 tenths of a ULP, stuck at 1024, a 24-bit FP32 window',
  model.big === 9 && model.small === -2 && model.diff === 11 && model.p === 11
  && scripts[0].includes('exponent 9') && scripts[0].includes('exponent minus 2') && scripts[0].includes('11 significant bits')
  && scripts[1].includes('That is 11 places') && scripts[2].includes('1000.5') && Math.round(model.err * 10) === 4 && scripts[2].includes('4 tenths')
  && model.acc[model.stall] === 1024 && scripts[4].includes('stuck at 1024') && scripts[5].includes(`${model.fp32Window} bits`),
  JSON.stringify({ diff: model.diff, err: model.err, stall: model.stall }));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['disappearing-addend-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', await (await box('captions')).evaluate((e) => e.checked));

// ---- each part's drawing, against the model
await seek(1000);
let d = await svgData();
check('part 1: 1000 and 0.3 with exponents 11 apart', d.beat === '1' && Number(d.diff) === model.diff && d.text.includes('1000') && d.text.includes('0.3'), JSON.stringify(d).slice(0, 200));
if (out) await (await page.$(S)).screenshot({ path: out + '-part1.png' });

await press('2. Align');
check('part 2\'s caption is its script', (await caption()) === scripts[1]);
await seek(400);
d = await svgData();
const k = Number(d.k);
check('part 2, part way: some shifts done, the exponent counts up with them', d.beat === '2' && k > 0 && k < model.diff && Number(d.exp) === model.small + k, JSON.stringify([d.k, d.exp]));
await seek(1000);
d = await svgData();
check('part 2 at the end: 11 shifts, window empty, tail is 0.3\'s 11 bits, guard 1 round 0 sticky 1 (the model\'s last shift)',
  Number(d.k) === model.diff && d.window === model.last.window && d.tail === model.last.tail && (await grs()) === `${model.last.g}${model.last.r}${model.last.s}`,
  JSON.stringify([d.window, d.tail, await grs(), model.last]));
if (out) await (await page.$(S)).screenshot({ path: out + '-part2.png' });

await press('3. Round');
await seek(1000);
d = await svgData();
check('part 3: neighbours 1000 and 1000.5, nearest even picks 1000.5, about 0.4 ULP', Number(d.result) === model.result && Number(d.below) === model.below && Number(d.above) === model.above && d.result === '1000.5' && Math.abs(Number(d.err) - model.err) < 1e-4, JSON.stringify(d).slice(0, 220));
if (out) await (await page.$(S)).screenshot({ path: out + '-part3.png' });

await press('4. Keep adding');
await seek(300);
d = await svgData();
check('part 4, early: the drawn running sum is the model\'s at that step, below 1024', Number(d.acc) === model.acc[Number(d.k)] && Number(d.acc) < 1024 && Number(d.spacing) === 0.5, JSON.stringify([d.k, d.acc, d.spacing]));
await seek(1000);
d = await svgData();
check('part 4 at the end: past 1024, the spacing is 1', Number(d.acc) === model.acc[Number(d.k)] && Number(d.acc) === 1024 && Number(d.spacing) === 1 && d.headline.includes('doubles'), JSON.stringify([d.k, d.acc, d.spacing, d.headline]));
if (out) await (await page.$(S)).screenshot({ path: out + '-part4.png' });

await press('5. Stuck');
await seek(1000);
d = await svgData();
check('part 5: after 400 additions FP16 still holds 1024 while the exact sum is past 1120', Number(d.k) === 400 && Number(d.acc) === 1024 && Number(d.exact) === model.exact[400] && Number(d.exact) > 1120 && Number(d.stall) === model.stall, JSON.stringify([d.k, d.acc, d.exact, d.stall]));
check('part 5: the stuck curve is drawn in the fail colour', await page.$eval(S + ' [data-curve="fp16"]', (e) => e.getAttribute('stroke') === 'var(--fail)'));
if (out) await (await page.$(S)).screenshot({ path: out + '-part5.png' });

await press('6. Ways out');
await seek(1000);
d = await svgData();
const ways = await page.$$eval(S + ' [data-way]', (es) => es.map((e) => e.textContent));
check('part 6 names three ways out: a tree, Kahan (A.6), an FP32 accumulator, and draws the FP32 sum', ways.length === 3 && ways[0].includes('tree') && ways[1].includes('Kahan') && ways[2].includes('FP32') && (await page.$(S + ' [data-curve="fp32"]')) !== null, ways.join(' | '));
if (out) await (await page.$(S)).screenshot({ path: out + '-part6.png' });

// ---- playing: parts advance by themselves; muted, it runs on the clock
await setBox('sound', false);
await press('5. Stuck');
await seek(985);
await press('Resume');
await pause(1600);
check('at the end of a part the next one starts by itself', (await current()) === 6, String(await current()));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2);
await press('1. Two numbers');
await page.$eval(S + ' button.primary', (b) => b.click()); // "Play" or "Resume"
await pause(100);
await pause(1500);
const a = await page.$eval(S + ' audio', (e) => ({ paused: e.paused }));
check('with sound off the audio stays silent and the animation still runs', a.paused && (await slider()) > 30, JSON.stringify({ a, slider: await slider() }));
await press('Pause');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
