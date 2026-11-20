// Browser check for the narrated animation "One vote, one atomic" in section 12.3: six
// parts, captions are an option, parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with the models (src/sim/filter.ts and src/sim/warpvote.ts,
// imported directly). It passes with or without content/ch12/narration/manifest.json: with no
// audio the player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch12-scene-one-vote.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { SCENE, sceneKeep, serveOneByOne } from '../src/sim/filter.ts';
import { binary, vote } from '../src/sim/warpvote.ts';

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
const S = 'section.scene[aria-label="One vote, one atomic"]';
const N = 6;
const LABELS = ['Eight lanes', 'One atomic per key', 'Run it again', 'The vote', 'One atomic', 'Ranks and slots'];

// ---- the model
const KEEP = sceneKeep();
const PASSING = KEEP.flatMap((k, l) => (k ? [l] : []));
const RUNS = SCENE.seeds.map((s) => serveOneByOne(KEEP, SCENE.counter, s));
const V = vote({ width: 8, pass: KEEP, counter: SCENE.counter, form: 'book' });
const G = V.groups[0];
const keysOf = (lanes) => lanes.map((l) => String(SCENE.keys[l])).join(' ');
const ordinal = ['', 'then ', '', 'and '];

const res = await fetch(BASE + '/content/ch12/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch12/narration/one-vote-${n}.txt`)).text()));

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
  return {
    headline: q('[data-headline]')?.textContent.trim() ?? '',
    counter: q('[data-counter]') ? Number(q('[data-counter]').dataset.counter) : null,
    pass: [...svg.querySelectorAll('g[data-lane]')].map((g) => g.dataset.pass),
    out1: q('g[data-out="1"]')?.dataset.values ?? null,
    out2: q('g[data-out="2"]')?.dataset.values ?? null,
    queue: q('[data-queue]')?.dataset.queue ?? '',
    served: [...svg.querySelectorAll('[data-served]')].map((t) => t.dataset.served),
    mask: q('[data-mask]')?.dataset.mask ?? null,
    shuffle: [...svg.querySelectorAll('[data-shuffle]')].map((g) => Number(g.dataset.shuffle)),
    ranks: [...svg.querySelectorAll('[data-rank-lane]')].map((g) => [Number(g.dataset.rankLane), Number(g.dataset.rank)]),
  };
}, S);

await goto(page, 'ch12/12.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
const served = (run) => run.map((x, k) => `${ordinal[k]}lane ${x.lane} gets ${x.slot}`);
check('the numbers the scripts say are the models\': lanes, both service orders, the mask bits, leader, count, start, ranks and slots',
  scripts[0].includes(`Lanes ${PASSING.slice(0, -1).join(', ')} and ${PASSING.at(-1)} pass`) && scripts[0].includes(`below ${SCENE.threshold}`)
  && scripts[1].includes(`stands at ${SCENE.counter}`) && served(RUNS[0]).every((p) => scripts[1].includes(p))
  && scripts[2].includes(`lane ${RUNS[1][0].lane}, lane ${RUNS[1][1].lane}, lane ${RUNS[1][2].lane}, then lane ${RUNS[1][3].lane}`)
  && scripts[3].includes(`bits ${PASSING.slice(0, -1).join(', ')} and ${PASSING.at(-1)}`) && scripts[3].includes(`makes lane ${G.leader} the leader`) && scripts[3].includes(`gives ${G.count}`)
  && scripts[4].includes(`adds ${G.count}`) && scripts[4].includes(`gets back ${G.start}`) && scripts[4].includes(`from lane ${G.leader}`)
  && scripts[5].includes('ranks 0, 1, 2 and 3') && scripts[5].includes(`slots ${SCENE.counter} to ${SCENE.counter + PASSING.length - 1}`) && V.atomics === 1,
  JSON.stringify({ runs: RUNS.map((r) => r.map((x) => x.lane)), leader: G.leader, count: G.count, start: G.start }));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['one-vote-1'];
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
check('before playing, part 1 shows eight lanes and no verdicts yet', d.pass.length === 8 && d.pass.every((p) => p === ''), JSON.stringify(d.pass));
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

// ---- every part against the models
await press('1. Eight lanes');
await seek(1000);
d = await drawn();
check('part 1 ends: the model\'s keep flags, lanes 1, 2, 4, 5 pass', same(d.pass, KEEP.map((k) => (k ? '1' : '0'))) && d.headline === 'lanes 1, 2, 4, 5 pass: each needs one slot', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. One atomic per key');
await seek(450);
d = await drawn();
check('part 2 midway: two served, the counter at 42, the other two waiting', d.counter === SCENE.counter + 2 && d.queue === RUNS[0].slice(2).map((x) => x.lane).join(' ') && d.served.length === 2, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2-mid.png' });
await seek(1000);
d = await drawn();
check('part 2 ends: four atomics, the counter at 44, keys in run 1\'s order', d.counter === SCENE.counter + 4 && d.out1 === keysOf(RUNS[0].map((x) => x.lane)) && d.served.length === 4, JSON.stringify(d));

await press('3. Run it again');
await seek(1000);
d = await drawn();
check('part 3 ends: run 2 serves another order, same keys, and both rows are shown', d.out1 === keysOf(RUNS[0].map((x) => x.lane)) && d.out2 === keysOf(RUNS[1].map((x) => x.lane))
  && d.out1 !== d.out2 && [...d.out1.split(' ')].sort().join() === [...d.out2.split(' ')].sort().join() && d.headline === 'same keys, another order: the filter is unstable', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The vote');
await seek(1000);
d = await drawn();
check('part 4 ends: the mask 0011 0110, leader lane 1, 4 ones', d.mask === binary(G.mask, 8) && d.mask === '0011 0110' && d.headline === `leader lane ${G.leader}; the mask has ${G.count} ones`, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. One atomic');
await seek(1000);
d = await drawn();
check('part 5 ends: one atomic, 40 becomes 44, a shuffle to lanes 2, 4 and 5', d.counter === G.atomic.after && same(d.shuffle, PASSING.filter((l) => l !== G.leader)) && d.headline === `a shuffle hands ${G.start} to the other passing lanes`, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Ranks and slots');
await seek(1000);
d = await drawn();
const writers = V.lanes.filter((L) => L.writes);
check('part 6 ends: ranks 0 to 3, keys side by side in lane order, one atomic instead of four',
  same(d.ranks, writers.map((L) => [L.lane, L.rank])) && d.out1 === keysOf(writers.map((L) => L.lane)) && d.headline === 'slots 40 to 43, in lane order, after 1 atomic instead of 4', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('5. One atomic');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part5-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch12\/narration\/(manifest\.json|one-vote-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
