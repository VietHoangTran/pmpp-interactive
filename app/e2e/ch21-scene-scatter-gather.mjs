// Browser check for the narrated animation "Scatter and gather, step by step" in section
// 21.2: six parts, captions are an option (off by default, remembered), it runs with the
// sound off, parts advance by themselves, the scripts' numbers are the model's, and every
// part's drawing agrees with src/sim/dcs.ts (imported inside the page).
// It passes with or without content/ch21/narration/manifest.json.
// Usage: node e2e/ch21-scene-scatter-gather.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

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
const S = 'section.scene[aria-label="Scatter and gather, step by step"]';
const N = 6;
const LABELS = ['The pairs', 'Scatter: one point', 'Scatter: every point', 'Gather: one atom', 'Gather: one write', 'Tally'];

const res = await fetch(BASE + '/content/ch21/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch21/narration/scatter-gather-${n}.txt`)).text()));

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
  return {
    ...svg.dataset,
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    queue: svg.querySelector('[data-queue]')?.dataset.queue ?? null,
    runs: svg.querySelector('[data-runs]')?.dataset.runs ?? null,
    broadcast: svg.querySelector('[data-broadcast]')?.dataset.broadcast ?? null,
    lines: svg.querySelectorAll('line').length,
    text: svg.textContent.replace(/\s+/g, ' '),
  };
}, S);

await goto(page, 'ch21/21.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// The model's numbers, computed in the page from the same module the scene draws from.
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/dcs.ts');
  const { W, atoms, runA, runB } = m.SCENE;
  const val = (v) => (v >= 0 ? ' ' : '') + v.toFixed(4);
  const sc = m.scatterTimeline(atoms, W);
  const ga = m.gatherTimeline(atoms, W);
  return {
    W, nAtoms: atoms.length, runA, runB,
    diff: m.orderSensitive(atoms, W, runA, runB).map((x) => x.point),
    totA: m.f32text(m.scatterTotal(atoms, 3, runA)), totB: m.f32text(m.scatterTotal(atoms, 3, runB)),
    cells0: Array.from({ length: W }, (_, i) => (i === 0 ? val(m.scatterTotal(atoms, 0, runA)) : val(0))),
    cellsEnd: Array.from({ length: W }, (_, i) => val(m.scatterTotal(atoms, i, runA))),
    reg0: ga[0].registers.map(m.f32text), regEnd: ga.at(-1).registers.map(m.f32text), regEndCells: ga.at(-1).registers.map(val),
    constPerStep: new Set(sc[0].constAddrs).size,
    tally: m.sceneTally(atoms, W),
  };
});

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
const T = M.tally;
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`row of ${M.W} grid points with ${M.nAtoms} atoms`) && scripts[0].includes(`${M.W * M.nAtoms} pairs`)
  && scripts[1].includes(`Each of ${M.nAtoms} threads`) && scripts[1].includes(`the ${M.nAtoms} atomic updates`)
  && scripts[2].includes('point 3 ends with totals that differ') && M.diff.includes(3)
  && scripts[3].includes(`There are ${M.W} threads`)
  && scripts[4].includes(`After ${M.nAtoms} steps`)
  && scripts[5].includes(`${T.scatterUpdates} atomic updates to ${T.scatterAddresses} addresses, ${T.updatesPerAddress} on each`)
  && scripts[5].includes(`read ${T.scatterConstPerStep} different constant addresses`) && scripts[5].includes(`${T.gatherAdds} additions in registers and ${T.gatherWrites} plain writes`)
  && scripts[5].includes(`read ${T.gatherConstPerStep} address per step`),
  JSON.stringify(M.tally));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['scatter-gather-1'];
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
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(970);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part's drawing against the model
await press('1. The pairs');
await seek(1000);
let d = await drawn();
check('part 1 ends: all 24 atom-point pairs drawn, split by atom or by point', d.mode === 'setup' && d.lines === 24 && d.headline.includes('24 pairs: split them by atom'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Scatter: one point');
await seek(500);
d = await drawn();
check('part 2 midway: four threads on point 0, three updates still queued', d.mode === 'scatter' && d.hot === '0' && d.queue === '3' && d.headline.includes('3 updates waiting'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 2 ends: the queue is applied and point 0 holds the model\'s float32 total', d.step === '0' && d.queue === '0' && d.cells === JSON.stringify(M.cells0), JSON.stringify({ d: d.cells, m: M.cells0 }));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Scatter: every point');
await seek(300);
d = await drawn();
check('part 3 early: another four-way collision further along the row', d.mode === 'scatter' && Number(d.hot) >= 1 && Number(d.hot) <= 5 && d.headline.includes('four-way collision'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 3 ends: point 3 under two orders, the model\'s two different float32 totals',
  d.runs === `${M.totA}|${M.totB}` && M.totA !== M.totB && d.cells === JSON.stringify(M.cellsEnd) && d.text.includes(`thread order ${M.runB.join(', ')}`), JSON.stringify({ runs: d.runs, M: [M.totA, M.totB] }));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Gather: one atom');
await seek(1000);
d = await drawn();
check('part 4 ends: every lane reads atom 0 (one address, broadcast) and the registers hold the model\'s step-0 values',
  d.mode === 'gather' && d.broadcast === '0' && d.registers === JSON.stringify(M.reg0), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Gather: one write');
await seek(300);
d = await drawn();
check('part 5 midway: a later atom is being read', d.mode === 'gather' && Number(d.step) >= 1 && Number(d.step) <= 3 && d.broadcast === d.step, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 5 ends: every point written once with the final registers, the same bits every run',
  d.registers === JSON.stringify(M.regEnd) && d.cells === JSON.stringify(M.regEndCells) && d.headline.includes('same bits') && d.text.includes('6 plain writes'), JSON.stringify(d));

await press('6. Tally');
await seek(1000);
d = await drawn();
check('part 6: the tally from the model',
  d.text.includes(`${T.scatterUpdates} atomic updates to ${T.scatterAddresses} addresses (${T.updatesPerAddress} each)`) && d.text.includes(`${T.scatterConstPerStep} constant addresses per step`)
  && d.text.includes(`${T.gatherAdds} register additions, ${T.gatherWrites} plain writes`) && d.text.includes(`${T.gatherConstPerStep} constant address per step`), d.text);
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Scatter: every point');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part3-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch21\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
