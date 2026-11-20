// Browser check for the narrated animation "One GPU page fault" in section C.8: five parts,
// captions are an option, the parts advance by themselves, it runs with the sound off, and
// every part's drawing is a frame of the model's trace (faultStory in
// src/sim/managedmem.ts, imported here directly). It passes with or without
// content/appC/narration/manifest.json.
// Usage: APP_URL=http://localhost:5702 node e2e/appC-scene-page-fault.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FAULT_PAGE, faultStory } from '../src/sim/managedmem.ts';

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
const S = 'section.scene[aria-label="One GPU page fault"]';
const N = 5;
const LABELS = ['The fault', 'Unmap', 'The move', 'Map and replay', 'In reverse'];

// The model's trace, frame by frame.
const run = faultStory();
const [, , gpuTouch, again, , cpuTouch] = run.steps;
const fr = (i) => {
  const f = run.frames[i];
  const p = f.view.m.pages[FAULT_PAGE];
  return { e: f.ev.e, proc: f.ev.proc ?? '', loc: p.loc, cpu: p.cpuMap ? 'yes' : 'no', gpu: p.gpuMap ? 'yes' : 'no', c: f.counters };
};

const res = await fetch(BASE + '/content/appC/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appC/narration/page-fault-${n}.txt`)).text()));

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
  const d = svg.dataset;
  return {
    beat: Number(d.beat), e: d.event, proc: d.proc, loc: d.loc, cpu: d.cpuMap, gpu: d.gpuMap,
    c: { h2d: Number(d.h2d), d2h: Number(d.d2h), gpuFaults: Number(d.gpuFaults), cpuFaults: Number(d.cpuFaults) },
    shown: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((c) => [c.dataset.counter, Number(c.querySelector('[data-value]').textContent)])),
    access: svg.querySelector('[data-access]')?.dataset.access ?? null,
    maps: Object.fromEntries([...svg.querySelectorAll('[data-map]')].map((m) => [m.dataset.map, m.dataset.on])),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    note: svg.querySelector('[data-note]')?.textContent.trim() ?? '',
  };
}, S);
const is = (d, f, extra = true) =>
  d.e === f.e && d.proc === f.proc && d.loc === f.loc && d.cpu === f.cpu && d.gpu === f.gpu && d.c.h2d === f.c.h2d && d.c.d2h === f.c.d2h
  && d.c.gpuFaults === f.c.gpuFaults && d.c.cpuFaults === f.c.cpuFaults && d.shown['to GPU'] === f.c.h2d && d.shown['CPU faults'] === f.c.cpuFaults
  && d.maps.cpu === f.cpu && d.maps.gpu === f.gpu && extra;

await goto(page, 'appC/C.8');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the scripts tell the model\'s order: fault, unmap from the CPU, move, map and replay, then the reverse',
  same(gpuTouch.events.map((e) => e.e), ['fault', 'unmap', 'move', 'map', 'replay']) && same(cpuTouch.events.map((e) => e.e), ['fault', 'unmap', 'move', 'map', 'replay'])
  && scripts[0].includes('page fault') && scripts[1].includes("The CPU's mapping for it is removed") && scripts[2].includes('travels over the interconnect')
  && scripts[3].includes('The load that faulted is replayed') && scripts[4].includes('runs in reverse'));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['page-fault-1'];
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
const movedOn = await slider();
check('with sound off the animation runs and the audio stays silent', movedOn > 10 && (await audio()).paused, JSON.stringify({ movedOn, audio: await audio() }));
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

// ---- every part's drawing is a frame of the model's trace
await press('1. The fault');
await seek(0);
let d = await drawn();
check('part 1 starts: the page in CPU memory, mapped by the CPU only, no fault yet',
  d.loc === 'cpu' && d.cpu === 'yes' && d.gpu === 'no' && d.c.gpuFaults === 0 && d.headline.includes('about to load from p2'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 1 ends on the model\'s fault frame: GPU access, no GPU mapping, the load waiting',
  is(d, fr(gpuTouch.start), d.access === 'waiting' && d.proc === 'gpu'), JSON.stringify({ d, want: fr(gpuTouch.start) }));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Unmap');
await seek(1000);
d = await drawn();
check('part 2 ends on the unmap frame: neither processor holds a mapping, the load still waiting',
  is(d, fr(gpuTouch.start + 1), d.cpu === 'no' && d.gpu === 'no' && d.access === 'waiting'), JSON.stringify({ d, want: fr(gpuTouch.start + 1) }));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. The move');
await seek(450);
d = await drawn();
check('part 3 midway: the page is crossing the interconnect towards GPU memory', d.e === 'move' && d.loc === 'moving' && d.headline.includes('crosses the interconnect (to GPU memory)') && d.access === 'waiting', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });
await seek(1000);
d = await drawn();
check('part 3 ends on the move frame: in GPU memory, not yet mapped, 1 page to the GPU',
  is(d, fr(gpuTouch.start + 2), d.loc === 'gpu' && d.c.h2d === 1 && d.note.includes('proportional to the page size')), JSON.stringify({ d, want: fr(gpuTouch.start + 2) }));

await press('4. Map and replay');
await seek(1000);
d = await drawn();
check('part 4 ends on the second kernel\'s local load: mapped by the GPU, done, still one fault',
  is(d, fr(again.start), d.e === 'local' && d.access === 'done' && d.gpu === 'yes' && d.c.gpuFaults === 1), JSON.stringify({ d, want: fr(again.start) }));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. In reverse');
await seek(500);
d = await drawn();
check('part 5 midway: the CPU side of the same sequence is under way', d.proc === 'cpu' || d.e === 'move' || d.e === 'wait', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 5 ends on the CPU\'s replay: back in CPU memory, mapped by the CPU, one fault and one move each way',
  is(d, fr(cpuTouch.end - 1), d.e === 'replay' && d.proc === 'cpu' && d.c.h2d === 1 && d.c.d2h === 1 && d.c.cpuFaults === 1 && d.access === 'done'), JSON.stringify({ d, want: fr(cpuTouch.end - 1) }));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. The move');
await seek(500);
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

const expected404 = (u) => /\/content\/appC\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
