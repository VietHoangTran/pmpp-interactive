// Browser check for the narrated animation "Lanes exchanging values" in section 10.7: six
// parts, captions are an option, parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with the model (src/sim/shuffle.ts, imported directly).
// It passes with or without content/ch10/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch10-scene-shuffle.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { ones, smallValues, warpReduceTrace } from '../src/sim/shuffle.ts';

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const S = 'section.scene[aria-label="Lanes exchanging values"]';
const N = 6;
const LABELS = ['Values in registers', 'Halve, add, repeat', 'Everyone computes', 'Butterfly', 'A lane that left early', 'Sixty-four lanes'];

// The scene's cases, from the model.
const V = smallValues(32);
const DOWN = warpReduceTrace(V, { pattern: 'down' });
const XOR = warpReduceTrace(V, { pattern: 'xor' });
const GONE = warpReduceTrace(V, { pattern: 'down', tail: { from: 20, mode: 'return' } });
const ZERO = warpReduceTrace(V, { pattern: 'down', tail: { from: 20, mode: 'zero' } });
const W64 = warpReduceTrace(ones(64), { pattern: 'down', loopWidth: 32 });
const FIX64 = warpReduceTrace(ones(64), { pattern: 'down' });
const rowsOf = (t, upto) => [t.initial.map((v, i) => (t.live[i] ? (v == null ? '?' : String(v)) : '-')).join(' '),
  ...t.steps.slice(0, upto).map((s) => s.lanes.map((l) => (l.live ? (l.after == null ? '?' : String(l.after)) : '-')).join(' '))];

const res = await fetch(BASE + '/content/ch10/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch10/narration/shuffle-${n}.txt`)).text()));

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
    rows: [...svg.querySelectorAll('g[data-row]')].map((g) => g.dataset.values),
    cone: svg.querySelectorAll('g[data-cone="1"]').length,
    arrows: svg.querySelector('g[data-arrows]') ? Number(svg.querySelector('g[data-arrows]').dataset.count) : null,
    counters: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((g) => [g.dataset.counter, g.querySelector('[data-value]').textContent.trim()])),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    ringed: [...svg.querySelectorAll('g[data-lane] rect')].filter((r) => r.getAttribute('stroke-width') === '2.2').length,
  };
}, S);

await goto(page, 'ch10/10.7');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
const firstUndefined = GONE.steps.findIndex((s) => s.lanes[0].after == null);
check('the numbers the scripts say are the model\'s: 16 above, 162, 160 / 31 / 129, lane 20, the second step, 32 and 64',
  scripts[0].includes('16 lanes above it') && scripts[0].includes('The top 16 lanes')
  && scripts[1].includes(`Here that is ${DOWN.lane0}`) && DOWN.lane0 === 162
  && scripts[2].includes(`${DOWN.laneOps} operations`) && scripts[2].includes(`Only ${DOWN.useful} of them`) && scripts[2].includes(`other ${DOWN.wasted}`)
  && scripts[3].includes(`the total, ${XOR.total}`) && XOR.holders.length === 32
  && scripts[4].includes('ends at lane 20') && scripts[4].includes('after the second step') && firstUndefined === 1
  && scripts[5].includes(`each holds ${W64.final[0]}`) && W64.final[32] === 32 && scripts[5].includes(`no lane holds ${W64.total}`),
  JSON.stringify([DOWN.lane0, DOWN.laneOps, DOWN.useful, DOWN.wasted, firstUndefined, W64.final[0]]));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['shuffle-1'];
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
check('before playing, part 1 shows 32 lanes and their starting values', same(d.rows, rowsOf(DOWN, 0)), JSON.stringify(d.rows));
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
await press('1. Values in registers');
await seek(450);
d = await drawn();
check('part 1 midway: the arrows of the first shuffle, 16 lanes receiving', d.arrows === 16 && d.rows.length === 1 && d.headline === 'every lane asks for the value 16 lanes above it', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 1 ends: the row after stride 16 is the model\'s', same(d.rows, rowsOf(DOWN, 1)) && d.headline === 'lanes 0 to 15 receive; lanes 16 to 31 keep their own', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Halve, add, repeat');
await seek(1000);
d = await drawn();
check('part 2 ends: five rows as the model computes them, lane 0 holds 162', same(d.rows, rowsOf(DOWN, 5)) && d.headline === 'after 5 steps lane 0 holds 162, the sum of all 32' && d.ringed === 1, JSON.stringify(d.headline));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Everyone computes');
await seek(1000);
d = await drawn();
check('part 3 ends: the cone is 32 starting values and 31 lane-steps; counters 160, 31, 129',
  d.cone === 32 + DOWN.useful && d.counters['lane-operations'] === '160' && d.counters['reach lane 0'] === '31' && d.counters.wasted === '129', JSON.stringify(d.counters) + ' cone ' + d.cone);
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Butterfly');
await seek(1000);
d = await drawn();
check('part 4 ends: the butterfly\'s rows, every lane holds 162, all 32 ringed', same(d.rows, rowsOf(XOR, 5)) && d.rows[5].split(' ').every((v) => v === '162') && d.ringed === 32 && d.headline === 'every lane holds 162', JSON.stringify(d.headline));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. A lane that left early');
await seek(650);
d = await drawn();
check('part 5 before the fix: the model\'s rows, lane 0 undefined', same(d.rows, rowsOf(GONE, 5)) && d.rows[5].split(' ')[0] === '?' && d.headline.includes('reached lane 0'), JSON.stringify(d.rows.slice(-1)));
if (out) await page.screenshot({ path: out + '-part5.png' });
await seek(1000);
d = await drawn();
check('part 5 ends with the fix: lanes add 0, lane 0 holds 96', same(d.rows, rowsOf(ZERO, 5)) && d.rows[5].split(' ')[0] === String(ZERO.lane0) && ZERO.lane0 === 96, d.headline);

await press('6. Sixty-four lanes');
await seek(650);
d = await drawn();
check('part 6 before the fix: 64 lanes, 5 steps, lane 0 and lane 32 hold 32', same(d.rows, rowsOf(W64, 5)) && d.rows[5].split(' ')[0] === '32' && d.rows[5].split(' ')[32] === '32' && d.ringed === 2, d.headline);
if (out) await page.screenshot({ path: out + '-part6.png' });
await seek(1000);
d = await drawn();
check('part 6 ends with the fix: 6 steps, lane 0 holds 64', same(d.rows, rowsOf(FIX64, 6)) && d.rows[6].split(' ')[0] === '64' && d.headline === 'loop from warpSize / 2: 6 steps, lane 0 holds 64', d.headline);

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Everyone computes');
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

const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|shuffle-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
