// Browser check for the narrated animation "Static and continuous batching" in section
// 22.5: five parts, captions are an option (off by default, remembered), parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the models
// (src/sim/serving.ts for the slots, src/sim/batching.ts for step times, both imported
// directly). The numbers the scripts say are the models' too. It passes with or without
// content/ch22/narration/manifest.json.
// Usage: node e2e/ch22-scene-continuous-batching.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { M8_DEFAULT, precisionFor, relativeStep, stepBStar } from '../src/sim/batching.ts';
import { SCENE_NAMES, SCENE_REQUESTS, SCENE_SHORT_MAX, SCENE_SLOTS, makeRequests, serve, slotsAt, waitingAt } from '../src/sim/serving.ts';

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
const S = 'section.scene[aria-label="Static and continuous batching"]';
const N = 5;
const LABELS = ['Six requests', 'Static, 4 slots', 'Continuous', 'The tally', 'The knob'];

// ---- the models, as the scene uses them
const f32 = precisionFor(4);
const base = relativeStep(M8_DEFAULT, 1, f32, 1, 20).step;
const stepTime = (a) => relativeStep(M8_DEFAULT, a, f32, 1, 20).step / base;
const run = (policy) => serve({ requests: SCENE_REQUESTS, slots: SCENE_SLOTS, policy, stepTime, shortMax: SCENE_SHORT_MAX });
const ST = run('static');
const CO = run('continuous');
const f2 = (v) => (Math.round(v * 100) / 100).toString();
const name = (ids) => ids.map((i) => SCENE_NAMES[i]).join(' ');
const short = SCENE_REQUESTS.filter((r) => r.length <= SCENE_SHORT_MAX).map((r) => r.id);
const HEAVY = makeRequests({ count: 600, rate: 1e6, mix: 'short', seed: 1 });
const knob = [1, 2, 4, 8, 16, 32, 64, 128, 256].map((slots) => serve({ requests: HEAVY, slots, policy: 'continuous', stepTime }));
const BSTAR = stepBStar(M8_DEFAULT, f32, { peak: 20, bw: 1 });

const res = await fetch(BASE + '/content/ch22/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch22/narration/continuous-batching-${n}.txt`)).text()));

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
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused }));
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const q = (sel) => svg.querySelector(sel);
  const charts = [...svg.querySelectorAll('[data-chart]')].map((c) => ({
    policy: c.dataset.chart,
    shown: Number(c.dataset.shown),
    cells: [...c.querySelectorAll('[data-cell]')].map((g) => [g.dataset.cell, g.dataset.id, g.dataset.state]),
    returned: [...c.querySelectorAll('[data-returned]')].map((t) => t.dataset.returned),
  }));
  const tally = q('[data-tally]');
  return {
    policy: svg.dataset.policy,
    headline: q('[data-headline]')?.textContent.trim() ?? '',
    requests: [...svg.querySelectorAll('[data-request]')].map((g) => [g.dataset.request, Number(g.dataset.length)]),
    charts,
    queue: q('[data-queue]')?.dataset.queue ?? null,
    tokensStep: q('[data-tokens-step]') ? Number(q('[data-tokens-step]').dataset.tokensStep) : null,
    idle: q('[data-idle]') ? Number(q('[data-idle]').dataset.idle) : null,
    total: q('[data-steps-total]')?.dataset.stepsTotal ?? '',
    tally: tally ? { ...tally.dataset } : null,
    knob: [...svg.querySelectorAll('[data-knob]')].map((g) => [g.dataset.knob, Number(g.dataset.n)]),
  };
}, S);
// The cells a chart should show for steps 0..upTo-1, from the model.
const expectCells = (r, upTo) => {
  const out = [];
  for (let k = 0; k < Math.min(upTo, r.steps.length); k++)
    slotsAt(r, k).forEach((c, s) => { if (c) out.push([`${s},${k}`, SCENE_NAMES[c.id], c.state]); });
  return JSON.stringify(out.sort());
};
const cellsOf = (chart) => JSON.stringify([...chart.cells].sort());

await goto(page, 'ch22/22.5');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the models\': 6 requests, 4 slots, 2 short ones held, 12 static steps against 8 for 26 tokens, three names',
  scripts[0].startsWith(`${SCENE_REQUESTS.length} requests`) && scripts[1].includes(`with ${SCENE_SLOTS} slots`) && scripts[1].includes('The 2 short ones')
  && ST.served.filter((r) => r.admitStep === 0 && r.lastStep + 1 < r.returnStep).length >= 2
  && scripts[3].includes(`took ${ST.steps.length} steps to make these ${ST.tokens} tokens`) && scripts[3].includes(`took ${CO.steps.length}.`)
  && scripts[2].includes('3 names') && ['iteration-level scheduling', 'in-flight batching', 'continuous batching'].every((n) => scripts[2].includes(n)),
  JSON.stringify({ st: ST.steps.length, co: CO.steps.length, tokens: ST.tokens }));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['continuous-batching-1'];
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
check('before playing, part 1 lists the six requests with their lengths and an empty chart',
  JSON.stringify(d.requests) === JSON.stringify(SCENE_REQUESTS.map((r) => [SCENE_NAMES[r.id], r.length])) && d.charts[0].cells.length === 0, JSON.stringify(d.requests));
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved }));
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

// ---- every part against the models
await press('1. Six requests');
await seek(1000);
d = await drawn();
check('part 1 ends: the first step fills one column, A B C D each get a token', cellsOf(d.charts[0]) === expectCells(CO, 1) && d.charts[0].cells.length === 4, cellsOf(d.charts[0]));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Static, 4 slots');
await seek(500);
d = await drawn();
const kMid = d.charts[0].shown - 1;
check(`part 2 midway (step ${kMid}): the model's cells, A and C finished but holding their slots, E and F waiting`,
  d.policy === 'static' && cellsOf(d.charts[0]) === expectCells(ST, kMid + 1) && d.queue === name(waitingAt(ST, kMid))
  && d.charts[0].cells.some(([, id, st]) => id === 'A' && st === 'done') && d.charts[0].cells.some(([, id, st]) => id === 'C' && st === 'done')
  && d.queue === 'E F' && d.tokensStep === ST.steps[kMid].active && d.idle === SCENE_SLOTS - ST.steps[kMid].active,
  JSON.stringify({ kMid, queue: d.queue, tokens: d.tokensStep, idle: d.idle }));
if (out) await page.screenshot({ path: out + '-part2-mid.png' });
await seek(1000);
d = await drawn();
check(`part 2 ends: E and F start only after B, all done after ${ST.steps.length} steps`,
  cellsOf(d.charts[0]) === expectCells(ST, ST.steps.length) && d.total === String(ST.steps.length)
  && ST.served.find((r) => r.id === 4).admitStep === ST.served.find((r) => r.id === 1).lastStep + 1, d.total);
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Continuous');
await seek(1000);
d = await drawn();
const returned = CO.served.filter((r) => r.length <= SCENE_SHORT_MAX).map((r) => SCENE_NAMES[r.id]).sort().join(' ');
check(`part 3 ends: E takes C's slot, F takes A's, short requests come back early, all done after ${CO.steps.length} steps`,
  d.policy === 'continuous' && cellsOf(d.charts[0]) === expectCells(CO, CO.steps.length) && d.total === String(CO.steps.length)
  && [...d.charts[0].returned].sort().join(' ') === returned && returned === name(short).split(' ').sort().join(' ')
  && d.charts[0].cells.every(([, , st]) => st === 'gen'), JSON.stringify(d.charts[0].returned));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The tally');
await seek(1000);
d = await drawn();
const t = d.tally;
check('part 4: the tally is serving.ts\'s: steps, tokens per unit and short requests\' finish times for both policies',
  t && t.staticSteps === String(ST.steps.length) && t.continuousSteps === String(CO.steps.length)
  && t.staticTps === f2(ST.tokensPerSecond) && t.continuousTps === f2(CO.tokensPerSecond)
  && t.staticShort === f2(ST.meanFinishShort) && t.continuousShort === f2(CO.meanFinishShort)
  && CO.tokensPerSecond > ST.tokensPerSecond && CO.meanFinishShort < ST.meanFinishShort, JSON.stringify(t));
check('part 4 shows both charts in full', d.charts.length === 2 && cellsOf(d.charts[0]) === expectCells(ST, 99) && cellsOf(d.charts[1]) === expectCells(CO, 99));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The knob');
await seek(1000);
d = await drawn();
const tps = knob.map((r) => r.tokensPerSecond);
const tpt = knob.map((r) => r.meanTimePerToken);
check('part 5: both curves drawn over slot limits 1 to 256; throughput rises then flattens past the knee while time per token climbs',
  d.knob.length === 2 && d.knob.every(([, n]) => n === 9) && tps[5] / tps[0] > 20 && tps[8] / tps[6] < 1.3 && tpt[8] / tpt[5] > 3 && BSTAR > 32 && BSTAR < 64,
  JSON.stringify({ knob: d.knob, tps: tps.map((v) => v.toFixed(2)), tpt: tpt.map((v) => v.toFixed(2)) }));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('2. Static, 4 slots');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part2-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch22\/narration\/(manifest\.json|continuous-batching-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
