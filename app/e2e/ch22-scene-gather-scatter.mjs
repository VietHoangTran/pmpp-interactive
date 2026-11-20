// Browser check for the narrated animation "Gather and scatter, in time" in section 22.2: six
// parts, the scripts (35 to 60 words, nothing read aloud that is a symbol, numbers that match
// the model), captions as an option (off by default, remembered; shown regardless for a part
// with no audio), parts that advance by themselves, sound off, and every part's drawing against
// src/sim/decompose.ts. Passes with or without content/ch22/narration/manifest.json.
// Usage: node e2e/ch22-scene-gather-scatter.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { SCENE, sceneModel } from '../src/sim/decompose.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const S = 'section.scene[aria-label="Gather and scatter, in time"]';
const N = 6;
const LABELS = ['Two ways to cut', 'Gather', 'Scatter', 'The queue', 'Gather is not free', 'Four questions'];

// The model's numbers.
const M = sceneModel();
const TICKS = M.scatterTicks.ticks.length;

const res = await fetch(BASE + '/content/ch22/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch22/narration/gather-scatter-${n}.txt`)).text()));

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
  const ds = (sel) => ({ ...(svg.querySelector(sel)?.dataset ?? {}) });
  return {
    beat: ds('[data-beat]').beat ?? null,
    out: ds('[data-row="out"]'),
    hist: ds('[data-row="hist"]'),
    edges: ds('[data-edges]'),
    reads: ds('[data-reads]'),
    scatterTick: ds('[data-scatter-tick]').scatterTick ?? null,
    gatherDone: ds('[data-gather-done]').gatherDone ?? null,
    questions: ds('[data-questions]').questions ?? null,
    criteria: ds('[data-criteria]').criteria ?? null,
    regs: [...svg.querySelectorAll('text')].filter((t) => t.style.fill === 'var(--reg)').map((t) => t.textContent),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
  };
}, S);

await goto(page, 'ch22/22.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*[\]+]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*[\]+]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`${M.p.nIn} inputs, ${M.p.nOut} outputs`) && scripts[0].includes(`sum of ${SCENE.w} neighbouring inputs`) && scripts[0].includes(`up to ${Math.max(...M.fanOut)} outputs`)
  && scripts[1].includes(`its ${SCENE.w} inputs`) && scripts[1].includes(`All ${M.gather.metrics.plainStores} writes`) && M.gather.metrics.atomicUpdates === 0
  && scripts[2].includes(`Every output now has ${M.scatter.metrics.worstWriters} threads`) && M.scatter.writers.every((w) => w === M.scatter.metrics.worstWriters)
  && scripts[3].includes(`Scatter needs ${TICKS} ticks`) && scripts[3].includes(`all ${M.gatherTicks.ticks[0].served.length} of its writes on the first tick`) && M.gatherTicks.ticks.length === 1
  && scripts[4].includes(`read ${Math.max(...M.reads)} times`),
  JSON.stringify({ fan: M.fanOut, ticks: TICKS, reads: M.reads }));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['gather-scatter-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));
await setBox('captions', false);
check('switching captions off: the caption shows only for a part without audio', audible ? !(await hasCaption()) : (await hasCaption()) && (await caption()) === scripts[0]);
await setBox('captions', true);

// ---- runs with the sound off, and parts advance by themselves
await setBox('sound', false);
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

// ---- every part's drawing against the model
const at = async (label, v, name) => {
  await press(label);
  await seek(v);
  if (out && name) await page.screenshot({ path: `${out}-${name}.png` });
  return drawn();
};
let d = await at('1. Two ways to cut', 1000, 'part1');
check('part 1: all 18 contributions drawn; fan-out 1 2 3 3 3 3 2 1',
  d.beat === '1' && d.edges.edges === String(M.p.edges.length) && d.edges.fanout === M.fanOut.join(' ') && d.headline.includes('every input feeds up to 3 outputs'), JSON.stringify(d));
d = await at('2. Gather', 300);
check('part 2 midway: registers hold partial sums, nothing written yet', d.beat === '2' && d.out.written === '0' && d.out.writers === '1 1 1 1 1 1', JSON.stringify(d));
d = await at('2. Gather', 1000, 'part2');
check('part 2 ends: 6 one-writer outputs written on tick 1 with the model\'s sums',
  d.out.written === String(M.p.nOut) && d.regs.filter((t) => /^\d+$/.test(t)).join(' ') === M.sums.join(' ') && d.headline === 'tick 1: 6 writes, 6 different cells, no atomics', JSON.stringify(d));
d = await at('3. Scatter', 200);
check('part 3 early: arrows still growing, no conflict marked yet', d.beat === '3' && d.out.atomic === 'no', JSON.stringify(d));
d = await at('3. Scatter', 1000, 'part3');
check('part 3 ends: 3 writers on every output, marked atomic, 18 updates',
  d.out.atomic === 'yes' && d.out.writers === M.scatter.writers.join(' ') && d.headline === `6 outputs, 3 writers each: ${M.scatter.metrics.atomicUpdates} atomic updates`, JSON.stringify(d));
d = await at('4. The queue', 50);
check('part 4 starts: every queue holds the 3 issued updates', d.beat === '4' && d.out.tick === '0' && d.out.waiting === M.scatterTicks.issued.join(' '), JSON.stringify(d));
d = await at('4. The queue', 450, 'part4');
const t4 = Number(d.out.tick);
check('part 4 midway: the queues drain one update per tick, as the model serialises them',
  t4 >= 1 && t4 < TICKS && d.out.waiting === M.scatterTicks.ticks[t4 - 1].waiting.join(' ') && d.scatterTick === String(t4) && d.gatherDone === '6', JSON.stringify(d));
d = await at('4. The queue', 1000);
check('part 4 ends: 3 ticks, every queue empty', d.out.tick === String(TICKS) && d.out.waiting === '0 0 0 0 0 0' && d.headline.startsWith(`${TICKS} ticks to drain 18 updates`), JSON.stringify(d));
d = await at('5. Gather is not free', 400, 'part5a');
check('part 5 first half: 18 reads of 8 inputs, the middle four read 3 times', d.beat === '5' && d.reads.reads === M.reads.join(' ') && d.reads.readsTotal === '18', JSON.stringify(d));
d = await at('5. Gather is not free', 1000, 'part5b');
const bin1 = M.histGather.workers[1];
check('part 5 ends: a bin thread has inspected all 8 inputs and found its own',
  d.hist.inspected === String(M.hp.nIn) && d.hist.inspects === String(bin1.inspects) && d.hist.found === String(bin1.useful.length) && d.hist.values === M.hp.values.join(' '), JSON.stringify(d));
d = await at('6. Four questions', 1000, 'part6');
check('part 6 ends: four questions and the book\'s five criteria', d.beat === '6' && d.questions === '4' && d.criteria === '5', JSON.stringify(d));

// ---- narrow screen: the stage scrolls rather than squashing
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.$eval(S + ' .stage', (e) => ({ sw: e.scrollWidth, cw: e.clientWidth, page: document.documentElement.scrollWidth }));
check('at 390 px the page does not scroll sideways', narrow.page <= 392, JSON.stringify(narrow));
if (out) await page.screenshot({ path: `${out}-narrow.png` });

const expected404 = (u) => /\/content\/ch22\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
