// Browser check for the narrated animation "A warp at a branch" in section 4.5: seven
// parts, captions are an option, the parts advance by themselves, it runs with the sound
// off, and every part's drawing agrees with the divergence model (src/sim/divergence.ts,
// imported here directly).
// It passes with or without content/ch04/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch04-scene-warp-branch.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FIG49, STRAIGHT_LINE, fig410Trips, ifElse, loop, maskBelow, run, straightLine } from '../src/sim/divergence.ts';

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
const S = 'section.scene[aria-label="A warp at a branch"]';
const N = 7;
const LABELS = ['In step', 'The split', 'Pass A', 'Pass B', 'The count', 'A loop', 'Order'];

// The scene's cases, from the model.
const W = FIG49.width;
const T = FIG49.threshold;
const COST = 3;
const STRAIGHT = run(straightLine(W));
const BRANCH = run(ifElse(W, maskBelow(W, T), COST, COST));
const EDGE = run(ifElse(W, maskBelow(W, W), COST, COST));
const MIXED = run(ifElse(W, maskBelow(W, T), COST, COST), 'interleaved');
const TRIPS = fig410Trips(W);
const LOOP = run(loop(W, TRIPS));
const BODY = LOOP.issues.filter((x) => x.part === 'body');
const WANTS = maskBelow(W, T).map((a) => (a ? 'A' : 'B')).join(' ');

const res = await fetch(BASE + '/content/ch04/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch04/narration/warp-branch-${n}.txt`)).text()));

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
// What the stage draws: each row's active lanes and part, the label of each row, the
// lanes' wishes, the threshold mark, the barrier, the counters and the headline.
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const rows = [...svg.querySelectorAll('g[data-row]')];
  return {
    counts: rows.map((g) => Number(g.dataset.count)),
    parts: rows.map((g) => g.dataset.part),
    labels: rows.map((g) => g.querySelector('text').textContent.replace('▸ ', '').trim()),
    // Lanes drawn solid in each row, read from the drawing itself.
    solid: rows.map((g) => [...g.querySelectorAll('rect')].filter((r) => /var\(--ink(-2)?\)/.test(r.getAttribute('fill') ?? '')).length),
    wants: svg.querySelector('[data-wants]')?.dataset.wants ?? null,
    threshold: svg.querySelector('[data-threshold]') ? Number(svg.querySelector('[data-threshold]').dataset.threshold) : null,
    barrier: svg.querySelector('[data-barrier]') ? Number(svg.querySelector('[data-barrier]').dataset.barrier) : null,
    counters: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((g) => [g.dataset.counter, g.querySelector('[data-value]').textContent.trim()])),
    side: [...svg.querySelectorAll('[data-side]')].map((t) => t.textContent.trim()),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
  };
}, S);
const counts = (issues) => issues.map((x) => x.count);

await goto(page, 'ch04/4.5');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('seven scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the model\'s: 24 and 8, 3 instructions, 8 steps against 5, trip counts 4 to 8',
  scripts[1].includes('below 24') && scripts[1].includes('The first 24 lanes') && scripts[1].includes('The last 8') && scripts[2].includes('its 3 instructions')
  && scripts[4].includes(`This warp spent ${BRANCH.spent} where a uniform warp needs ${BRANCH.uniform}`) && scripts[4].includes('from 24 up to 32')
  && scripts[5].includes('from 4 to 8') && scripts[5].includes(`For ${LOOP.stmts[1].uniformIterations} rounds every lane works`)
  && Math.min(...TRIPS) === 4 && Math.max(...TRIPS) === 8 && BRANCH.spent === 8 && BRANCH.uniform === 5,
  JSON.stringify([BRANCH.spent, BRANCH.uniform]));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('seven parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
// The captions choice is remembered per browser; this check starts from a clean profile.
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['warp-branch-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script, without audio tags', (await hasCaption()) && (await caption()) === scripts[0] && !(await caption()).includes('['));
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));
await setBox('captions', false);
check('switching captions off works', !(await (await box('captions')).evaluate((e) => e.checked)) && (audible ? !(await hasCaption()) : await hasCaption()));
await setBox('captions', true);

// ---- runs with the sound off, and parts advance by themselves
await setBox('sound', false);
let d = await drawn();
check('before playing, part 1 shows one instruction executed by all 32 lanes', same(d.counts, [W]) && same(d.solid, [W]) && d.labels[0] === STRAIGHT_LINE[0].text, JSON.stringify(d));
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
await press('1. In step');
await seek(1000);
d = await drawn();
check('part 1 ends: the three straight-line instructions, each fetched once and executed by all 32 lanes',
  same(d.counts, counts(STRAIGHT.issues)) && same(d.solid, [W, W, W]) && same(d.labels, STRAIGHT_LINE.map((x) => x.text))
  && d.counters['instructions fetched'] === '3' && d.counters['lane-operations done'] === String(STRAIGHT.useful) && d.headline === `3 fetches, each executed by all ${W} lanes`,
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. The split');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(100);
d = await drawn();
check('part 2 first: every lane evaluates the test, nothing split yet', same(d.counts, [W]) && d.wants === null && d.threshold === null && d.headline === 'every lane evaluates the condition on its own index', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 2 ends: 24 lanes want A, 8 want B, one stream, the split marked at lane 24',
  d.wants === WANTS && d.threshold === T && d.labels[0] === 'test: threadIdx.x < 24' && d.headline === `${T} lanes want A, ${W - T} want B, and there is one instruction stream`, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Pass A');
await seek(1000);
d = await drawn();
const passA = BRANCH.issues.slice(0, 1 + COST);
check('part 3 ends: the test, then pass A three times with 24 lanes running and 8 masked',
  same(d.counts, counts(passA)) && same(d.counts, [32, 24, 24, 24]) && same(d.solid, d.counts) && same(d.parts, ['op', 'A', 'A', 'A'])
  && d.counters['warp-steps spent'] === '4' && d.counters['lanes left waiting'] === `${(W - T) * COST} lane-steps` && d.headline === 'pass A: 24 lanes execute, 8 are masked and wait',
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Pass B');
await seek(300);
d = await drawn();
check('part 4 early: pass B runs 8 lanes while 24 wait', d.parts.includes('B') && d.counts[d.counts.length - 1] === 8 && d.headline === 'pass B: 8 lanes execute, 24 wait', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4 ends with the whole branch as the model issues it, then C with all 32',
  same(d.counts, counts(BRANCH.issues)) && same(d.counts, [32, 24, 24, 24, 8, 8, 8, 32]) && same(d.solid, d.counts) && same(d.parts, BRANCH.issues.map((x) => x.part))
  && d.counters['warp-steps spent'] === String(BRANCH.spent) && d.counters['lanes left waiting'] === `${BRANCH.idle} lane-steps` && d.headline === `then C, with all ${W} lanes again`,
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The count');
await seek(100);
d = await drawn();
check('part 5 first: 8 warp-steps spent where a uniform warp needs 5, 96 lane-steps waiting',
  d.threshold === T && d.counters['warp-steps spent'] === String(BRANCH.spent) && d.counters['a uniform warp needs'] === String(BRANCH.uniform) && d.counters['lanes left waiting'] === `${BRANCH.idle} lane-steps` && BRANCH.idle === 96
  && d.headline === 'threshold 24: two paths inside one warp, two passes', JSON.stringify(d));
await seek(500);
d = await drawn();
const mid = d.threshold;
const midTrace = mid == null ? EDGE : run(ifElse(W, maskBelow(W, mid), COST, COST));
check('part 5 midway: the threshold slides and the counters are the model\'s for it', (mid == null || (mid > T && mid < W)) && d.counters['warp-steps spent'] === String(midTrace.spent) && d.counters['lanes left waiting'] === `${midTrace.idle} lane-steps` && same(d.counts, counts(midTrace.issues)), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 5 ends at 32: the split is on the edge of the warp, one pass, 5 steps, nothing waits',
  d.threshold === null && same(d.counts, counts(EDGE.issues)) && same(d.counts, [32, 32, 32, 32, 32]) && d.counters['warp-steps spent'] === String(EDGE.spent) && EDGE.spent === 5
  && d.counters['lanes left waiting'] === '0 lane-steps' && d.labels[0] === 'test: threadIdx.x < 32' && d.headline === 'the split is on the edge of the warp: one path, no second pass', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. A loop');
await seek(450);
d = await drawn();
check('part 6 midway: the first iterations keep every lane busy', d.counts.length >= 1 && d.counts.length < BODY.length && same(d.counts, counts(BODY.slice(0, d.counts.length))) && d.wants === TRIPS.join(' '), JSON.stringify(d));
await seek(1000);
d = await drawn();
const loopIdle = BODY.reduce((a, r) => a + W - r.count, 0);
check('part 6 ends: 8 iterations, lanes dropping out as Fig. 4.10\'s counts say: 32, 32, 32, 32, 28, 24, 16, 8',
  same(d.counts, counts(BODY)) && same(d.counts, [32, 32, 32, 32, 28, 24, 16, 8]) && same(d.solid, d.counts) && same(d.labels, BODY.map((_, i) => `iteration ${i + 1}`))
  && d.counters['iterations run'] === `8 of ${LOOP.stmts[1].passes}` && d.counters['lanes left waiting'] === `${loopIdle} lane-steps` && d.headline === 'iteration 8: 8 lanes work, 24 have finished and wait',
  JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

await press('7. Order');
await seek(350);
d = await drawn();
check('part 7 first: the two paths take turns, as the model interleaves them', d.parts.length >= 2 && same(d.parts, MIXED.issues.slice(1, 1 + d.parts.length).map((x) => x.part)) && d.parts[0] === 'A' && d.parts[1] === 'B' && d.barrier === null, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 7 ends: A and B interleaved, the same 6 warp-steps, then a warp barrier before C',
  same(d.parts, [...MIXED.issues.slice(1, -1).map((x) => x.part), 'op']) && same(d.parts, ['A', 'B', 'A', 'B', 'A', 'B', 'op']) && d.barrier === 6
  && d.counters['warp-steps for A and B'] === '6, in any order' && MIXED.spent === BRANCH.spent && d.headline === 'a step that needs every lane comes after a warp barrier',
  JSON.stringify(d));
check('and says who documents what: NVIDIA may mix, AMD documents lockstep, Apple does not say', same(d.side, ['NVIDIA: the two paths may be mixed', 'AMD: lockstep is documented', 'Apple: not documented']), JSON.stringify(d.side));
if (out) await page.screenshot({ path: out + '-part7.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());
await press('Play again');
await pause(300);
check('playing again starts from part 1', (await current()) === 1 && (await mainButton()) === 'Pause');
await press('Pause');

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('4. Pass B');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part4-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at 390 px the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
// Nothing may hide behind a sideways scroll inside a bench either: the last beat button included.
const hidden = await page.evaluate(() => [...document.querySelectorAll('.bench > .body')].map((b) => ({ bench: b.parentElement.getAttribute('aria-label') ?? b.parentElement.querySelector('h4')?.textContent, scroll: b.scrollWidth, client: b.clientWidth })).filter((x) => x.scroll > x.client + 1));
check('at 390 px no bench on the page scrolls sideways inside itself (the beat buttons need the shared .scene .seg wrap)', hidden.length === 0, JSON.stringify(hidden));
// On a narrow screen the counters and the headline leave the drawing and are shown as text under it.
const readout = await page.evaluate((S) => {
  const r = document.querySelector(S + ' .stage [data-readout="scene"]');
  return r && {
    headline: r.querySelector('[data-readout-headline]')?.textContent.trim() ?? '',
    items: Object.fromEntries([...r.querySelectorAll('[data-readout-item]')].map((e) => [e.dataset.readoutItem, e.querySelector('b').textContent.trim()])),
    px: parseFloat(getComputedStyle(r).fontSize),
    inSvg: !!document.querySelector(S + ' .stage svg [data-headline], ' + S + ' .stage svg [data-counter]'),
  };
}, S);
check('at 390 px part 4\'s headline and counters are text under the drawing, at a readable size, with the model\'s numbers',
  !!readout && readout.headline === `then C, with all ${W} lanes again` && readout.items['warp-steps spent'] === String(BRANCH.spent) && readout.items['lanes left waiting'] === `${BRANCH.idle} lane-steps` && readout.px >= 12 && !readout.inSvg,
  JSON.stringify(readout));
await press('7. Order');
await seek(1000);
const sides = await page.$$eval(S + ' .stage [data-readout-note]', (es) => es.map((e) => e.textContent.trim()));
check('and part 7\'s three vendor lines are text too', same(sides, ['NVIDIA: the two paths may be mixed', 'AMD: lockstep is documented', 'Apple: not documented']), JSON.stringify(sides));
await press('4. Pass B');
await seek(1000);
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch04\/narration\/(manifest\.json|warp-branch-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
