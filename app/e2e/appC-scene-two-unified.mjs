// Browser check for the narrated animation "Two things called unified memory" in section
// C.6: six parts, captions are an option, the parts advance by themselves, it runs with the
// sound off, and every part's drawing agrees with the model (src/sim/managedmem.ts,
// imported here directly). It passes with or without content/appC/narration/manifest.json:
// with no audio the player runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5702 node e2e/appC-scene-two-unified.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FIG_PAGES, moved, runFig } from '../src/sim/managedmem.ts';

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
const S = 'section.scene[aria-label="Two things called unified memory"]';
const N = 6;
const LABELS = ['First touch', 'Limited mode', 'Full mode', 'Back to the CPU', 'One memory', 'Side by side'];

const bulk = runFig('managed', 'bulk').run.counters;
const full = runFig('managed', 'migrate').run.counters;
const one = runFig('managed', 'one').run.counters;
const explicit = runFig('explicit', 'migrate').run.counters;

const res = await fetch(BASE + '/content/appC/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appC/narration/two-unified-${n}.txt`)).text()));

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
  const where = [...svg.querySelectorAll('[data-page]')].map((g) => g.dataset.where);
  return {
    beat: Number(svg.dataset.beat),
    where,
    counters: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((c) => [c.dataset.counter, Number(c.querySelector('[data-value]').textContent)])),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    note: svg.querySelector('[data-note]')?.textContent.trim() ?? '',
    locked: !!svg.querySelector('[data-locked]'),
    waited: svg.dataset.waited ?? null,
    modes: Object.fromEntries([...svg.querySelectorAll('[data-mode]')].map((g) => [g.dataset.mode, Number(g.dataset.moved)])),
  };
}, S);
const all = (arr, v) => arr.length === FIG_PAGES && arr.every((x) => x === v);

await goto(page, 'appC/C.6');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the model\'s: 8 pages, 8 over and 8 back, 16 crossed in both CUDA modes and none on the Mac',
  scripts[0].includes(`each of the ${FIG_PAGES} pages`) && scripts[1].includes(`all ${FIG_PAGES} of them`) && bulk.h2d === FIG_PAGES
  && scripts[3].includes(`${full.h2d} pages over, ${full.d2h} pages back`) && moved(explicit) === moved(full)
  && scripts[5].includes(`${moved(bulk)} pages crossed in both CUDA modes, and none on the Mac`) && moved(bulk) === moved(full) && moved(one) === 0,
  JSON.stringify([bulk, full, one]));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['two-unified-1'];
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

// ---- every part's drawing against the model
await press('1. First touch');
await seek(0);
let d = await drawn();
check('part 1 starts: nothing placed, the allocation headline', all(d.where, 'none') && d.headline.startsWith('cudaMallocManaged'), JSON.stringify(d));
await seek(500);
d = await drawn();
const placed = d.where.filter((w) => w === 'cpu').length;
check('part 1 midway: some pages placed, all of them in CPU memory', placed > 0 && placed < FIG_PAGES && d.where.every((w) => w === 'cpu' || w === 'none') && d.headline.includes(`${placed} of ${FIG_PAGES} pages`), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 1 ends: all 8 pages in CPU memory, nothing moved', all(d.where, 'cpu') && d.counters['to GPU'] === 0 && d.note.includes('first touched'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Limited mode');
await seek(1000);
d = await drawn();
check('part 2 ends: every page moved at the launch, no GPU faults, the CPU locked out', all(d.where, 'gpu') && d.counters['to GPU'] === bulk.h2d && d.counters['GPU faults'] === 0 && d.locked && d.headline.includes(`(${FIG_PAGES} of ${FIG_PAGES})`), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Full mode');
await seek(30);
d = await drawn();
check('part 3 starts: the kernel runs and nothing has moved', all(d.where, 'cpu') && d.headline === 'full: the kernel starts, nothing has moved', JSON.stringify(d));
await seek(500);
d = await drawn();
const onGpu = d.where.filter((w) => w === 'gpu').length;
check('part 3 midway: pages fault over one at a time, the counters agree with the picture',
  onGpu > 0 && onGpu < FIG_PAGES && d.counters['to GPU'] === onGpu && d.counters['GPU faults'] >= onGpu && d.counters['GPU faults'] <= onGpu + 1, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 3 ends: 8 GPU faults, 8 pages moved, all in GPU memory', all(d.where, 'gpu') && d.counters['GPU faults'] === full.gpuFaults && d.counters['to GPU'] === full.h2d, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Back to the CPU');
await seek(1000);
d = await drawn();
check('part 4 ends: every page back in CPU memory, 8 each way, the same 16 as the explicit copies',
  all(d.where, 'cpu') && d.counters['to CPU'] === full.d2h && d.counters['CPU faults'] === full.cpuFaults && d.note === `${moved(full)} pages crossed in all. Two explicit copies move ${moved(explicit)}.`, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. One memory');
await seek(1000);
d = await drawn();
check('part 5 ends: every page in the one memory, nothing moved, the wait done', all(d.where, 'one') && d.waited === 'yes' && d.counters['to GPU'] === 0 && d.note.startsWith('Nothing moved.'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Side by side');
await seek(1000);
d = await drawn();
check('part 6: 16, 16 and 0 pages crossed, from the three runs', d.modes['CUDA, limited'] === moved(bulk) && d.modes['CUDA, full'] === moved(full) && d.modes['Apple silicon'] === moved(one), JSON.stringify(d.modes));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Full mode');
await seek(600);
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
