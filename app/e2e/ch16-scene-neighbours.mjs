// Browser check for the narrated animation "Waiting for everyone, or only for your
// neighbours" in section 16.3: five parts, captions are an option (off by default,
// remembered), the parts advance by themselves, it runs with the sound off, and every
// part's drawing agrees with the scheduler model (src/sim/wavesched.ts, imported here
// directly), as do the numbers the scripts say. It passes with or without
// content/ch16/narration/manifest.json: with no audio the player runs on estimated timings
// and shows each script regardless.
// Usage: node e2e/ch16-scene-neighbours.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { flagsAfter, schedule } from '../src/sim/wavesched.ts';

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
const S = 'section.scene[aria-label="Waiting for everyone, or only for your neighbours"]';
const N = 5;
const LABELS = ['Sixteen tiles', 'A relaunch per front', 'Flags', 'Four SMs', 'Fixed rows'];

// The model's numbers.
const front2 = schedule(4, 2, 'front');
const counter2 = schedule(4, 2, 'counter');
const front4 = schedule(4, 4, 'front');
const counter4 = schedule(4, 4, 'counter');
const rows = schedule(4, 2, 'rows', 'reversed');
const sixAt = counter2.doneAt[6];

const res = await fetch(BASE + '/content/ch16/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch16/narration/neighbours-${n}.txt`)).text()));

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
    beat: svg.querySelector('[data-beat]').dataset.beat,
    headline: svg.querySelector('[data-headline]').textContent.trim(),
    counter: svg.querySelector('[data-counter]')?.dataset.counter ?? null,
    counterText: svg.querySelector('[data-counter]')?.textContent.trim() ?? null,
    idle: svg.querySelector('[data-idle]')?.dataset.idle ?? null,
    lanes: [...svg.querySelectorAll('[data-lanes]')].map((g) => ({ S: Number(g.dataset.lanes), rule: g.dataset.rule, shown: Number(g.dataset.shown), finished: g.dataset.finished === '1' })),
    slots: [...svg.querySelectorAll('[data-lanes]')].map((g) => [...g.querySelectorAll('[data-slot]')].map((x) => `${x.dataset.slot}:${x.dataset.state}:${x.dataset.tile}`)),
    flags: [...svg.querySelectorAll('[data-tile][data-done="1"]')].map((x) => Number(x.dataset.tile)).sort((a, b) => a - b),
    barriers: svg.querySelectorAll('[data-barrier]').length,
    six: svg.querySelector('[data-mark="six"]')?.textContent.trim() ?? null,
    arrows: !!svg.querySelector('[data-arrows]'),
    notes: [...svg.querySelectorAll('[data-notes] text')].map((x) => x.textContent),
    text: svg.textContent,
  };
}, S);
const modelSlots = (sch, n) => {
  const o = [];
  for (let u = 0; u < n; u++) for (let s = 0; s < sch.S; s++) {
    const c = sch.timeline[Math.min(u, sch.timeline.length - 1)][s];
    o.push(`${u},${s}:${c.state}:${c.tile ?? ''}`);
  }
  return o;
};
const flagsOf = (sch, n) => [...flagsAfter(sch, n)].sort((a, b) => a - b);

await goto(page, 'ch16/16.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*[\]]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*[\]]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes('16 tiles') && scripts[0].includes('2 lanes')
  && scripts[1].includes(`the steps: ${front2.makespan}`) && front2.makespan === 10
  && scripts[2].includes(`At step ${sixAt + 1}, tile 6`) && scripts[2].includes(`takes ${counter2.makespan} steps`) && counter2.makespan === 9
  && scripts[3].includes('4 lanes') && scripts[3].includes(`take ${front4.makespan} steps`) && front4.makespan === 7 && counter4.makespan === 7
  && scripts[4].includes('rows 3 and 2') && rows.stuck && rows.timeline[0].map((c) => c.block).join() === '3,2',
  JSON.stringify({ f2: front2.makespan, c2: counter2.makespan, sixAt, f4: front4.makespan, c4: counter4.makespan }));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['neighbours-1'];
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
await press('1. Sixteen tiles');
await seek(100);
let d = await drawn();
check('part 1 starts: the grid, two empty lanes, no flags', d.beat === '1' && d.lanes.length === 1 && d.lanes[0].S === 2 && d.slots[0].length === 0 && d.flags.length === 0, JSON.stringify(d.lanes));
await seek(800);
d = await drawn();
check('part 1 later: arrows into tile 4, and front 2 (tiles 3, 4, 5) named', d.arrows && d.headline.includes('tiles 3, 4 and 5'), d.headline);
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. A relaunch per front');
await seek(450);
d = await drawn();
const mid = d.lanes[0].shown;
check('part 2 midway: the lanes show the model\'s first steps of a relaunch per front', mid > 2 && mid < 10 && JSON.stringify(d.slots[0]) === JSON.stringify(modelSlots(front2, mid)) && JSON.stringify(d.flags) === JSON.stringify(flagsOf(front2, mid)), `${mid} ${JSON.stringify(d.slots[0])}`);
await seek(1000);
d = await drawn();
check('part 2 ends: 10 steps, 4 empty lane-steps, a barrier after every front',
  d.lanes[0].shown === 10 && d.lanes[0].finished && d.counterText === '10 steps' && d.idle === '4' && d.barriers === front2.barriers.length && d.flags.length === 16
  && JSON.stringify(d.slots[0]) === JSON.stringify(modelSlots(front2, 10)), JSON.stringify({ c: d.counterText, idle: d.idle, b: d.barriers }));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Flags');
await seek(1000);
d = await drawn();
check('part 3 ends: 9 steps, the model\'s schedule, and tile 6 marked in the empty lane of step 4',
  d.lanes[0].shown === 9 && d.counterText === '9 steps' && JSON.stringify(d.slots[0]) === JSON.stringify(modelSlots(counter2, 9))
  && d.six === `tile 6 fills the empty lane of step ${sixAt + 1}` && sixAt === 3 && d.text.includes('a relaunch per front took 10'), JSON.stringify({ c: d.counterText, six: d.six }));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Four SMs');
await seek(1000);
d = await drawn();
check('part 4 ends: two charts of 4 lanes, both 7 steps, as the model says',
  d.lanes.length === 2 && d.lanes.every((l) => l.S === 4 && l.shown === 7 && l.finished) && d.counterText === 'both 7 steps'
  && JSON.stringify(d.slots[0]) === JSON.stringify(modelSlots(front4, 7)) && JSON.stringify(d.slots[1]) === JSON.stringify(modelSlots(counter4, 7)), JSON.stringify(d.lanes));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Fixed rows');
await seek(380);
d = await drawn();
check('part 5 first: rows 3 and 2 spin, nothing done, the chain of waits named',
  d.lanes[0].rule === 'rows' && d.slots[0].every((x) => x.includes(':spin:')) && d.slots[0].length >= 4 && d.counter === 'stuck' && d.flags.length === 0
  && d.notes.join(' | ') === 'row 3 waits for row 2 | row 2 waits for row 1 | row 1 has no lane, and never will', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5-stuck.png' });
await seek(1000);
d = await drawn();
check('part 5 ends: tiles from a counter complete in 9 steps', d.lanes[0].rule === 'counter' && d.lanes[0].finished && d.counterText === '9 steps' && d.flags.length === 16, JSON.stringify(d.lanes));
if (out) await page.screenshot({ path: out + '-part5-end.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Flags');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part3-light.png' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
if (out) await page.screenshot({ path: out + '-part3-dark.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at 390 px the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch16\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
