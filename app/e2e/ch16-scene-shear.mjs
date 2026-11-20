// Browser check for the narrated animation "Shearing a tile" in section 16.6: five parts,
// captions are an option (off by default, remembered), the parts advance by themselves,
// it runs with the sound off, and every part's drawing agrees with the tiling model
// (src/sim/wavetile.ts, imported here), as do the numbers the scripts say.
// It passes with or without content/ch16/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch16-scene-shear.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { activeThreads, busyPerStep, cellAt, launchPlan, loadsOf, producerOf, slotCell, summary } from '../src/sim/wavetile.ts';

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
const S = 'section.scene[aria-label="Shearing a tile"]';
const N = 5;
const LABELS = ['A square tile', 'Shear it', 'Every thread busy', 'Stored as a square', 'The whole table'];

// ---- the model's numbers
const SQ = { Lseq: 9, W: 3, scheme: 'square' };
const SH = { Lseq: 9, W: 3, scheme: 'sheared' };
const TILE = { row: 1, col: 1 };
const busySq = busyPerStep(3, 'square');
const busySh = busyPerStep(3, 'sheared');
const sumSq = summary(SQ);
const sumSh = summary(SH);
const inputsAt = (k) => loadsOf(SH, cellAt(SH, TILE, k, 1)).map((l) => {
  const p = producerOf(SH, l.r, l.q);
  const same = p && p.tile.row === 1 && p.tile.col === 1;
  return `${l.name}:${l.from}:${same ? k - p.dTile : 'launch'}`;
}).join(' ');
const lit = (g, k) => activeThreads(g, TILE, k).map((t) => cellAt(g, TILE, k, t)).map((c) => `${c.r},${c.q}`).join(' ');
const f4 = loadsOf(SH, cellAt(SH, TILE, 2, 2));
const diag = [0, 1, 2].map((qt) => [...new Set([0, 1, 2].map((rt) => { const c = slotCell(SH, TILE, rt, qt); return c.r + c.q; }))].join('/')).join(' ');

const res = await fetch(BASE + '/content/ch16/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch16/narration/shear-${n}.txt`)).text()));

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
  const g = svg.querySelector('[data-beat]');
  return {
    ...g.dataset,
    bars: svg.querySelector('[data-bars]')?.dataset.bars ?? null,
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    panels: [...svg.querySelectorAll('[data-panel]')].map((p) => ({ ...p.dataset })),
    inputTexts: [...svg.querySelectorAll('[data-input]')].map((t) => t.textContent.trim()),
    offsets: [...svg.querySelectorAll('[data-offset]')].map((t) => t.textContent.trim()),
    text: svg.textContent,
  };
}, S);

await goto(page, 'ch16/16.6');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*[\]]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*[\]]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`lights ${busySq[0]} cell, then ${busySq[1]}, then ${busySq[2]}, then ${busySq[3]}, then ${busySq[4]}`) && scripts[0].includes(`So 9 cells take ${busySq.length} steps`)
  && scripts[1].includes('Row 1 slides 1 cell to the left, and row 2 slides 2 cells')
  && scripts[2].includes(`all ${busySh[0]} threads work at every step`) && scripts[2].includes(`take ${busySh.length} steps`) && inputsAt(2) === 'n:tile:1 w:tile:1 nw:tile:2'
  && scripts[2].includes('north and west neighbours were finished 1 step earlier, and its north west neighbour 2 steps earlier')
  && scripts[3].includes('north input sits 1 row up and 1 column left') && scripts[3].includes('2 columns left') && f4.map((l) => l.slot.join(',')).join(' ') === '1,1 2,1 1,0'
  && scripts[4].includes(`Square tiles take ${sumSq.launches} launches, up to ${sumSq.mostTiles} tiles each`) && scripts[4].includes(`Sheared tiles take ${sumSh.launches}, at most ${sumSh.mostTiles} at a time`),
  JSON.stringify({ busySq, busySh, sq: [sumSq.launches, sumSq.mostTiles], sh: [sumSh.launches, sumSh.mostTiles], in2: inputsAt(2) }));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['shear-1'];
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
await press('1. A square tile');
await seek(500);
let d = await drawn();
check('part 1 halfway: step 2 of the square tile, 3 cells on one anti-diagonal', d.beat === '1' && d.step === '2' && d.busy === String(busySq[2]) && d.lit === lit(SQ, 2) && d.bars === busySq.slice(0, 3).join(','), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 1 ends: step 4, 1 busy and 2 idle, 9 cells in 5 steps', d.step === '4' && d.busy === '1' && d.lit === lit(SQ, 4) && d.bars === busySq.join(',') && d.headline.includes('9 cells take 5 steps'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Shear it');
await seek(0);
d = await drawn();
check('part 2 starts unsheared', d.beat === '2' && d.shift === '0.00' && d.lines === '0', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 2 ends: rows moved by their row number; tile columns on anti-diagonals r + q = 8, 9, 10', d.shift === '1.00' && d.lines === '1' && d.diagonals === diag && diag === '8 9 10', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Every thread busy');
await seek(100);
d = await drawn();
check('part 3 starts at step 0: all 3 busy; thread 1\'s inputs come from earlier launches', d.beat === '3' && d.step === '0' && d.busy === '3' && d.lit === lit(SH, 0) && d.inputs === inputsAt(0) && inputsAt(0).split(' ').every((x) => x.endsWith('global:launch'))
  && d.inputTexts.every((x) => x.endsWith('an earlier launch, global memory')), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 3 ends: step 2, north and west 1 step earlier, north-west 2, all from the tile',
  d.step === '2' && d.inputs === inputsAt(2) && d.bars === '3,3,3' && d.headline.includes('9 cells take 3 steps')
  && d.inputTexts.join(' | ') === 'north (4, 5): this tile, 1 step earlier | west (5, 4): this tile, 1 step earlier | north-west (4, 4): this tile, 2 steps earlier', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Stored as a square');
await seek(300);
d = await drawn();
check('part 4 before the reveal: no offsets shown yet', d.beat === '4' && d.shown === '' && d.offsets.length === 0, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4 ends: thread 2 at step 2 reads slots (1,1), (2,1) and (1,0): column − 1, − 1, − 2',
  d.focus === '2,2' && d.n === '1,1' && d.w === '2,1' && d.nw === '1,0' && d.shown === 'n w nw'
  && d.offsets.join(' | ') === 'load_n: r_tile − 1, q_tile − 1 | load_w: same row, q_tile − 1 | load_nw: r_tile − 1, q_tile − 2', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The whole table');
await seek(500);
d = await drawn();
const [pSq, pSh] = d.panels;
check('part 5 halfway: square tiles on their last launch, sheared tiles on launch 5 of 8 with the model\'s tiles',
  pSq.panel === 'square' && pSq.launch === '5' && pSq.total === '5' && pSq.tiles === String(launchPlan(SQ, 4).tiles.length)
  && pSh.panel === 'sheared' && pSh.launch === '5' && pSh.total === '8' && pSh.tiles === String(launchPlan(SH, 4).tiles.length), JSON.stringify(d.panels));
await seek(1000);
d = await drawn();
check('part 5 ends: 5 launches of up to 3 tiles against 8 of at most 2',
  d.panels[0].launch === '5' && d.panels[0].most === '3' && d.panels[1].launch === '8' && d.panels[1].most === '2' && d.headline.includes('5 launches against 8'), JSON.stringify(d.panels));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Every thread busy');
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
