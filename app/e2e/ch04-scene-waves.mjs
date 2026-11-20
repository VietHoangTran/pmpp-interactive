// Browser check for the narrated animation "One grid, two devices" in section 4.3:
// six parts, captions are an option, the parts advance by themselves, it runs with
// the sound off, and every part's drawing agrees with the scheduler model
// (src/sim/blocksched.ts, imported here directly).
// It passes with or without content/ch04/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch04-scene-waves.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  completionOrder,
  schedule,
  shuffledOrder,
  tail,
  unevenDurations,
  usage,
  waveSizes,
  waves,
} from '../src/sim/blocksched.ts';

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
const S = 'section.scene[aria-label="One grid, two devices"]';
const N = 6;
const LABELS = ['The list', 'Two slots', 'Four slots', 'Any order', 'The tail', 'Two ways out'];

const res = await fetch(BASE + '/content/ch04/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch04/narration/waves-${n}.txt`)).text()));

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
// What the stage draws: its visible text, and the blocks in each named part of the drawing.
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  // Drawn means not faded out: the earlier device in part 3 is shown dimmed (0.5) and still counts.
  const visible = (el) => {
    for (let e = el; e && e !== svg; e = e.parentElement) if (Number(e.getAttribute('opacity') ?? 1) < 0.45) return false;
    return true;
  };
  const blocks = (root) => [...root.querySelectorAll('g[data-block]')].filter(visible).map((g) => Number(g.dataset.block));
  const part = (name) => [...svg.querySelectorAll(`g[data-part="${name}"]`)];
  return {
    text: [...svg.querySelectorAll('text')].filter(visible).map((t) => t.textContent.trim()),
    pending: part('pending').map(blocks),
    running: part('running').map(blocks),
    history: part('history').map(blocks),
    devices: part('device').map((g) => Number(g.dataset.capacity)),
    outputs: part('output').map((g) => g.dataset.written === '1'),
    field: part('field').map((g) => ({ busy: Number(g.dataset.busy), slots: Number(g.dataset.slots), filled: [...g.querySelectorAll('rect')].filter((r) => r.getAttribute('fill') === 'var(--ink-2)').length }))[0] ?? null,
    all: blocks(svg),
  };
}, S);
const has = (list, s) => list.some((x) => x.includes(s));
const headline = (d) => d.text[d.text.length - 2];

// The scene's cases, from the model.
const SMALL = schedule({ blocks: FIG45.blocks, ...FIG45.small });
const LARGE = schedule({ blocks: FIG45.blocks, ...FIG45.large });
const SHUFFLE = shuffledOrder(FIG45.blocks, 3);
const SHUFFLED = schedule({ blocks: FIG45.blocks, ...FIG45.large, order: SHUFFLE });
const CAP = capacity(TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit);
const SIZES = waveSizes(TAIL_EXAMPLE.blocks, CAP);
const TAIL = tail(TAIL_EXAMPLE.blocks, CAP);
const EVEN = Math.ceil(waves(TAIL_EXAMPLE.blocks, CAP)) * CAP;
const unevenUse = (w) => usage(schedule({ ...TAIL_EXAMPLE, blocks: w * CAP, durations: unevenDurations(w * CAP) })).utilisation;
const pct = (v) => Math.round(v * 100);
const ids = (n) => [...Array(n).keys()];
// Blocks of a schedule that have not started in wave k (1-based) of a run of equal blocks.
const notStartedIn = (s, k) => s.spec.order.filter((b) => s.placements[b].start >= k);
const startedBy = (s, k) => ids(s.spec.blocks).filter((b) => s.placements[b].start < k);

await goto(page, 'ch04/4.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('scripts use digits for the numbers the model produces', scripts[1].includes('8 blocks, 2 at a time, make 4 waves') && scripts[2].includes('in 2 waves instead of 4') && scripts[4].includes(`${TAIL_EXAMPLE.blocks} blocks`) && scripts[4].includes(`runs ${CAP} at once`) && scripts[4].includes(`only ${TAIL.last} blocks for the third`) && scripts[5].includes(`a multiple of ${CAP}`)
  && waves(8, SMALL.capacity) === 4 && waves(8, LARGE.capacity) === 2 && TAIL.last === 132 && CAP === 264);
check('the script for the tail says "half", which is the model\'s idle share', scripts[4].includes('Half of the slots sit empty') && TAIL.idleShare === 0.5);
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
const audible = !!manifest?.clips?.['waves-1'];
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
check('before playing, part 1 shows the grid as a list of blocks that have not started', has(d.text, `${FIG45.blocks} blocks have not started`) && has(d.text, 'not started') && d.pending[0].length >= 1 && d.pending[0].every((b, i) => b === i), JSON.stringify(d.pending));
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
await press('1. The list');
await seek(1000);
d = await drawn();
check('part 1 ends with all 8 blocks on the list, in launch order, and the question about slots', same(d.pending[0], ids(FIG45.blocks)) && has(d.text, 'how many slots?') && has(d.text, 'each needs a free slot on an SM for all of its threads'), JSON.stringify(d.pending));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Two slots');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(500);
d = await drawn();
let m = /^wave (\d) of (\d)$/.exec(headline(d));
check('part 2 midway: a small device of 2 slots is running a wave of the model\'s 4', !!m && Number(m[2]) === waves(FIG45.blocks, SMALL.capacity) && same(d.devices, [SMALL.capacity]) && has(d.text, `a small device: ${SMALL.capacity} slots`), headline(d));
let k = m ? Number(m[1]) : 1;
check('the list holds exactly the blocks the model has not started in that wave', same(d.pending[0], notStartedIn(SMALL, k)) && same([...d.history[0]].sort((a, b) => a - b), startedBy(SMALL, k)), JSON.stringify([k, d.pending[0], d.history[0]]));
check('and the two blocks of that wave are the ones in the slots', same([...d.running[0]].sort((a, b) => a - b), ids(FIG45.blocks).filter((b) => SMALL.placements[b].start === k - 1)) && d.running[0].length === SMALL.capacity, JSON.stringify(d.running[0]));
await seek(1000);
d = await drawn();
check('part 2 ends: 8 blocks, 2 at a time, 4 waves, nothing left on the list', headline(d) === `${FIG45.blocks} blocks, ${SMALL.capacity} at a time: ${waves(FIG45.blocks, SMALL.capacity)} waves` && SMALL.makespan === 4 && d.pending[0].length === 0 && has(d.text, 'nothing left') && d.history[0].length === FIG45.blocks, headline(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Four slots');
await seek(250);
d = await drawn();
m = /^wave (\d) of (\d)$/.exec(headline(d));
k = m ? Number(m[1]) : 1;
check('part 3 early: the larger device runs the same list 4 at a time', !!m && Number(m[2]) === waves(FIG45.blocks, LARGE.capacity) && same(d.devices, [SMALL.capacity, LARGE.capacity]) && d.running[1].length === LARGE.capacity && same(d.pending[0], notStartedIn(LARGE, k)), JSON.stringify([headline(d), d.running, d.pending]));
await seek(1000);
d = await drawn();
check('part 3 ends: the same grid, 4 waves there and 2 waves here, kernel and launch unchanged', headline(d) === `the same grid: ${waves(8, SMALL.capacity)} waves there, ${waves(8, LARGE.capacity)} waves here` && LARGE.makespan === 2 && has(d.text, 'the kernel and the launch are unchanged') && d.history[0].length === 8 && d.history[1].length === 8, headline(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Any order');
await seek(400);
d = await drawn();
check('part 4 first: every block is drawn with its own barrier, and nothing between blocks', same(d.all, ids(FIG45.blocks)) && ids(8).every((b) => has(d.text, `block ${b}`)) && has(d.text, 'a barrier holds the threads of one block') && has(d.text, 'it stops at the edge of the block: nothing ties one block to another'), d.text.join('|'));
if (out) await page.screenshot({ path: out + '-part4a.png' });
await seek(1000);
d = await drawn();
const shuffledDone = completionOrder(SHUFFLED);
check('part 4 ends: the list is handed out in the model\'s shuffled order, which is not the launch order', headline(d) === `handed out as ${SHUFFLE.join(' ')}: ${SHUFFLE.length} of ${SHUFFLE.length} outputs written` && !same(SHUFFLE, ids(8)) && same([...SHUFFLE].sort((a, b) => a - b), ids(8)), headline(d));
check('and every part of the output is still written', d.outputs.length === 8 && d.outputs.every(Boolean) && ids(8).every((b) => has(d.text, `part ${b}`)) && has(d.text, 'the same outputs as before, whatever the order') && same([...shuffledDone].sort((a, b) => a - b), ids(8)), JSON.stringify(d.outputs));
await seek(760);
d = await drawn();
const written = d.outputs.filter(Boolean).length;
check('part 4 midway: the outputs written are those of the blocks the model has finished by then', [0, SHUFFLED.capacity, 8].includes(written) && has(d.text, `handed out as ${SHUFFLE.join(' ')}: ${written} of 8 outputs written`)
  && same(d.outputs.map((x, b) => (x ? b : -1)).filter((b) => b >= 0), [...shuffledDone.slice(0, written)].sort((a, b) => a - b)), JSON.stringify(d.outputs));
if (out) await page.screenshot({ path: out + '-part4b.png' });

await press('5. The tail');
await seek(300);
d = await drawn();
check('part 5 early: the first wave fills all 264 slots of 132 SMs', headline(d) === `wave 1 of ${SIZES.length}: ${CAP} of ${CAP} slots busy` && d.field.slots === CAP && d.field.busy === CAP && d.field.filled === CAP && has(d.text, `${TAIL_EXAMPLE.units} SMs with ${TAIL_EXAMPLE.slotsPerUnit} slots each`), JSON.stringify([headline(d), d.field]));
await seek(1000);
d = await drawn();
check('part 5 ends in the third wave with 132 of 264 slots busy: half the device dark', headline(d) === `wave ${SIZES.length} of ${SIZES.length}: ${TAIL.last} of ${CAP} slots busy` && d.field.busy === TAIL.last && d.field.filled === TAIL.last && TAIL.last === 132 && SIZES.length === 3, JSON.stringify([headline(d), d.field]));
check('and states the book\'s arithmetic: 660 at 264 at once is 2.5 waves, 132 slots idle', has(d.text, `${TAIL_EXAMPLE.blocks} blocks at ${CAP} at once is ${waves(TAIL_EXAMPLE.blocks, CAP)} waves: the last leaves ${TAIL.idle} slots idle`) && waves(TAIL_EXAMPLE.blocks, CAP) === 2.5, d.text.join('|'));
check('the blocks in each wave are the model\'s: 264, 264, 132', SIZES.every((n, i) => has(d.text, `wave ${i + 1}`) && d.text.includes(String(n))) && same(SIZES, [264, 264, 132]), d.text.join('|'));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Two ways out');
await seek(200);
d = await drawn();
check('part 6 first: 792 blocks, a multiple of 264, keep every slot busy', /^792 blocks = 3 × 264: wave \d, 264 of 264 slots busy$/.test(headline(d)) && EVEN === 792 && d.field.busy === CAP && tail(EVEN, CAP).idle === 0, headline(d));
await seek(400);
d = await drawn();
check('in its last wave too: no dark slots', headline(d) === `${EVEN} blocks = ${waves(EVEN, CAP)} × ${CAP}: wave 3, ${CAP} of ${CAP} slots busy` && d.field.filled === CAP, headline(d));
if (out) await page.screenshot({ path: out + '-part6a.png' });
await seek(650);
d = await drawn();
m = /^uneven block times: (\d+) of 264 slots busy$/.exec(headline(d));
check('part 6 then makes block times uneven: the slots busy are counted from the model\'s schedule', !!m && Number(m[1]) === d.field.busy && d.field.busy === d.field.filled && d.field.busy <= CAP && has(d.text, `the same ${EVEN} blocks, now taking 1 to 4 time units each`), headline(d));
await seek(1000);
d = await drawn();
const one = unevenUse(1);
const ten = unevenUse(10);
check('part 6 ends with the model\'s comparison: uneven blocks use more of the device in 10 waves than in 1', headline(d) === `uneven blocks: ${pct(one)}% of the capacity used in 1 wave, ${pct(ten)}% in 10` && ten > one && d.text.includes(`${pct(one)}%`) && d.text.includes(`${pct(ten)}%`), headline(d));
if (out) await page.screenshot({ path: out + '-part6b.png' });

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

// ---- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at 390 px the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
// Nothing may hide behind a sideways scroll inside a bench either: the last beat button included.
const hidden = await page.evaluate(() => [...document.querySelectorAll('.bench > .body')].map((b) => ({ bench: b.parentElement.getAttribute('aria-label') ?? b.parentElement.querySelector('h4')?.textContent, scroll: b.scrollWidth, client: b.clientWidth })).filter((x) => x.scroll > x.client + 1));
check('at 390 px no bench on the page scrolls sideways inside itself (the beat buttons need the shared .scene .seg wrap)', hidden.length === 0, JSON.stringify(hidden));
// On a narrow screen the headline and the line under it leave the drawing and are shown as text.
await press('5. The tail');
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
  !!readout && readout.headline.length > 10 && readout.notes.length === 1 && readout.px >= 12 && readout.viewHeight === 304, JSON.stringify(readout));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch04\/narration\/(manifest\.json|waves-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
