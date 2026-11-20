// Browser check for the narrated animation "Hiding a wait" in section 4.6: six parts,
// captions are an option, the parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with the models it is computed from
// (src/sim/latency.ts and src/sim/occupancy.ts, imported here directly).
// It passes with or without content/ch04/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch04-scene-latency.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { PRESETS, busyFraction, estimate, runs, simulate, simulateSwitching, unitShares, utilisation, warpsNeeded } from '../src/sim/latency.ts';
import { BOOK, occupancy } from '../src/sim/occupancy.ts';

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
const pct = (v) => Math.round(v * 1000) / 10;
const S = 'section.scene[aria-label="Hiding a wait"]';
const N = 6;
const LABELS = ['One warp', 'Three warps', 'Enough warps', 'A free switch', 'The price', 'Two kernels'];

// The scene's cases, from the models (the same numbers LatencyScene.tsx uses).
const SCENE = { work: 2, latency: 6, slots: 1 };
const CYCLES = 24;
const NEEDED = warpsNeeded(SCENE.work, SCENE.latency);
const line = (warps) => simulate({ warps, ...SCENE }, CYCLES);
const busyIn = (tl, n) => Math.round(busyFraction(tl, 0, n) * n);
const ONE = line(1);
const THREE = line(3);
const ENOUGH = line(NEEDED);
const COSTLY = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, 2);
const FREE = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, 0);
const regsFit = (regs) => occupancy(BOOK, { threads: 2 * BOOK.width, regs, shared: 0 });
const LADDER = [1, 2, 4, 8, 16];

const res = await fetch(BASE + '/content/ch04/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch04/narration/latency-${n}.txt`)).text()));

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
/** The drawing: its data-* attributes and its visible text. */
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const data = Object.fromEntries(Object.entries(svg.dataset).map(([k, v]) => [k, Number.isNaN(Number(v)) ? v : Number(v)]));
  return { data, text: [...svg.querySelectorAll('text')].map((t) => t.textContent.trim()) };
}, S);
const headline = (d) => d.text[d.text.length - 2];
const has = (list, s) => list.some((x) => x.includes(s));

await goto(page, 'ch04/4.6');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`computes for ${SCENE.work} cycles`) && scripts[0].includes(`waits ${SCENE.latency} cycles`)
  && scripts[1].includes('busy for 6 cycles out of every 8') && estimate({ warps: 3, ...SCENE }) === 6 / 8
  && scripts[2].includes(`With ${NEEDED} warps`) && NEEDED === 4
  && scripts[4].includes(`${regsFit(32).residentWarps} warps fit when each thread uses 32 registers`) && scripts[4].includes(`only ${regsFit(64).residentWarps} warps fit`)
  && scripts[5].includes(`${PRESETS.computeHeavy.work} cycles of arithmetic`) && scripts[5].includes(`${warpsNeeded(PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency)} warps keep the unit busy`)
  && scripts[5].includes(`needs ${warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency)}`),
  scripts.join(' / '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}]/.test(s)), scripts.find((s) => /[<>=×%/_(){}]/.test(s)));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
// The captions choice is remembered per browser; this check starts from a clean profile.
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['latency-1'];
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
check('before playing, part 1 has drawn nothing and claims nothing', d.data.beat === 1 && d.data.shown === 0 && headline(d) === 'cycle 0: nothing issued yet', JSON.stringify(d.data) + ' ' + headline(d));
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
await press('1. One warp');
await seek(500);
d = await drawn();
check('part 1 midway: the cycles counted busy are the model\'s for the cycles drawn so far', d.data.shown > 0 && d.data.busy === busyIn(ONE, d.data.shown) && headline(d) === `execution unit busy for ${d.data.busy} of ${d.data.shown} cycles: ${pct(d.data.busy / d.data.shown)}%`, JSON.stringify(d.data));
await seek(1000);
d = await drawn();
const wait = runs(ONE.rows[0]).find((r) => r.state === 'wait');
check('part 1 ends: one warp keeps the unit busy 25% of the time, c / (c + L)', d.data.shown === CYCLES && d.data.busy === busyIn(ONE, CYCLES) && pct(d.data.busy / CYCLES) === pct(SCENE.work / (SCENE.work + SCENE.latency)) && has(d.text, 'load sent') && has(d.text, 'data back') && wait.start === SCENE.work && wait.length === SCENE.latency, JSON.stringify(d.data));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Three warps');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
d = await drawn();
check('part 2 ends: three warps, busy 75% as the script says', d.data.warps === 3 && d.data.busy === busyIn(THREE, CYCLES) && pct(d.data.busy / CYCLES) === 75, JSON.stringify(d.data));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Enough warps');
await seek(1000);
d = await drawn();
const first = runs(ENOUGH.rows[0]).find((r) => r.state === 'wait');
const lastWait = runs(ENOUGH.rows[NEEDED - 1]).find((r) => r.state === 'wait');
check('part 3 ends: 4 warps fill every cycle', d.data.warps === NEEDED && d.data.busy === CYCLES && busyIn(ENOUGH, CYCLES) === CYCLES && headline(d) === `${NEEDED} warps: busy for ${CYCLES} of ${CYCLES} cycles: 100%`, JSON.stringify(d.data) + ' ' + headline(d));
check('and marks the cycle where warp 0\'s data is back as the last warp starts to wait', d.data.meet === first.start + first.length && lastWait.start === d.data.meet && has(d.text, `cycle ${d.data.meet}`), JSON.stringify(d.data));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. A free switch');
await seek(600);
d = await drawn();
const execTo = (unit, n) => unit.slice(0, n).filter((u) => u.kind === 'exec').length;
check('part 4 midway: the SM has issued every cycle, the CPU side only the model\'s share', d.data['gpuExec'] === execTo(FREE, d.data.shown) && d.data['gpuExec'] === d.data.shown && d.data['cpuExec'] === execTo(COSTLY, d.data.shown) && d.data['cpuExec'] < d.data['gpuExec'], JSON.stringify(d.data));
check('and states the switching share the model gives (50%)', d.data['switchShare'] === pct(unitShares(COSTLY, 0, CYCLES).switch) && headline(d).includes(`${d.data['switchShare']}% of its cycles saving and loading`), headline(d));
check('every warp\'s state is drawn on the chip, the CPU\'s threads in memory', [0, 1, 2, 3].every((w) => has(d.text, `warp ${w}: PC, registers`)) && has(d.text, 'memory: the state of the threads that are not running'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The price');
await seek(500);
d = await drawn();
const A = regsFit(32);
check('part 5 first: 32 registers per thread fill all 64 warp slots of the book\'s register file', d.data.regs === 32 && d.data.warps === A.residentWarps && d.data.warps === 64 && d.data.slots === A.warpSlots && headline(d) === `32 registers per thread: 64 of 64 warp slots filled, 65,536 of 65,536 registers`, JSON.stringify(d.data) + ' ' + headline(d));
await seek(1000);
d = await drawn();
const B = regsFit(64);
check('part 5 ends: 64 registers per thread, 32 warps fit, as the occupancy model says', d.data.regs === 64 && d.data.warps === B.residentWarps && d.data.warps === 32 && d.data.fit === 32 && headline(d) === `64 registers per thread: 32 of 64 warp slots filled, 65,536 of 65,536 registers`, JSON.stringify(d.data) + ' ' + headline(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Two kernels');
const side = (preset, warps) => pct(utilisation({ warps, ...preset, slots: 1 }));
for (const v of [300, 600, 1000]) {
  await seek(v);
  d = await drawn();
  check(`part 6 at ${v / 10}%: ${d.data.rung} warps, the two utilisations are the model's`, LADDER.includes(d.data.rung) && d.data.left === side(PRESETS.computeHeavy, d.data.rung) && d.data.right === side(PRESETS.loadHeavy, d.data.rung)
    && has(d.text, `${d.data.rung} ${d.data.rung === 1 ? 'warp' : 'warps'}: unit busy ${d.data.left}%`), JSON.stringify(d.data));
}
check('part 6 ends at 16 warps, both busy, and says which needed how many', d.data.rung === 16 && d.data.left === 100 && d.data.right === 100 && headline(d) === 'busy with 2 warps on the left; 16 needed on the right', headline(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());
await press('Play again');
await pause(300);
check('playing again starts from part 1', (await current()) === 1 && (await mainButton()) === 'Pause');
await press('Pause');
await setBox('captions', false);

// ---- dark theme and a narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(200);
const colours = await page.$eval(S + ' .stage svg', (svg) => [...svg.querySelectorAll('[fill]')].map((e) => e.getAttribute('fill')).filter((f) => f !== 'none' && !f.startsWith('var(')));
check('the drawing uses theme variables only, so it follows dark mode', colours.length === 0, colours.join(','));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at 390 px the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
// Nothing may hide behind a sideways scroll inside a bench either: the last beat button included.
const hidden = await page.evaluate(() => [...document.querySelectorAll('.bench > .body')].map((b) => ({ bench: b.parentElement.getAttribute('aria-label') ?? b.parentElement.querySelector('h4')?.textContent, scroll: b.scrollWidth, client: b.clientWidth })).filter((x) => x.scroll > x.client + 1));
check('at 390 px no bench on the page scrolls sideways inside itself (the beat buttons need the shared .scene .seg wrap)', hidden.length === 0, JSON.stringify(hidden));
// On a narrow screen the headline and the line under it leave the drawing and are shown as text.
await press('5. The price');
await seek(1000);
const readout = await page.evaluate((S) => {
  const r = document.querySelector(S + ' .stage [data-readout="scene"]');
  const svg = document.querySelector(S + ' .stage svg');
  const [, , , h] = svg.getAttribute('viewBox').split(' ').map(Number);
  return r && {
    headline: r.querySelector('[data-readout-headline]')?.textContent.trim() ?? '',
    notes: [...r.querySelectorAll('[data-readout-note]')].map((e) => e.textContent.trim()),
    px: parseFloat(getComputedStyle(r).fontSize),
    viewHeight: h,
  };
}, S);
check('at 390 px the headline and the line under it are text under the drawing, at a readable size, and the drawing stops above them',
  !!readout && readout.headline === `64 registers per thread: 32 of 64 warp slots filled, 65,536 of 65,536 registers` && readout.notes.length === 1 && readout.px >= 12 && readout.viewHeight === 304, JSON.stringify(readout));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch04\/narration\/(manifest\.json|latency-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
