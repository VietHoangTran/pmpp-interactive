// Browser check for the narrated animation "Turn the corner, then pad" in section 6.4:
// six parts, captions are an option, the parts advance by themselves, it runs with the
// sound off, and every part's drawing agrees with the models (global sectors from
// src/sim/coalesce.ts, bank passes from src/sim/banks.ts, imported here directly).
// It passes with or without content/ch06/narration/manifest.json.
// Usage: node e2e/ch06-scene-turn-and-pad.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { tileLoad } from '../src/sim/coalesce.ts';
import { tileAccess } from '../src/sim/banks.ts';

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
const S = 'section.scene[aria-label="Turn the corner, then pad"]';
const N = 6;
const LABELS = ['Top edge', 'Turned', '32 passes', 'One column more', 'The price', 'Other GPUs'];

const load = (a) => tileLoad({ layout: 'col', arrangement: a, T: 32, bx: 0, ph: 0, Width: 1024, warpWidth: 32 }).warps[0].access;
const store = (pitch, warpWidth = 32, banks = 32) => tileAccess({ T: 32, pitch, orientation: 'x-first', warpWidth, banks });

const res = await fetch(BASE + '/content/ch06/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch06/narration/turn-and-pad-${n}.txt`)).text()));

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
  const n = svg.querySelector('[data-arrangement]');
  const sh = svg.querySelector('[data-pitch]');
  const b = svg.querySelector('[data-passes]');
  return {
    arrangement: n?.dataset.arrangement || null,
    sectors: n?.dataset.sectors ? Number(n.dataset.sectors) : null,
    pitch: sh ? Number(sh.dataset.pitch) : null,
    passes: b ? Number(b.dataset.passes) : null,
    banksUsed: b ? Number(b.dataset.banksUsed) : null,
    banks: b ? Number(b.dataset.banks) : null,
    served: b ? Number(b.dataset.served) : null,
    dots: svg.querySelectorAll('circle').length,
    lines: [...svg.querySelectorAll('[data-line]')].map((t) => t.textContent.trim()),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
  };
}, S);

await goto(page, 'ch06/6.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the models\': 32 sectors, then 4; 32 passes and 31 idle banks; one pass at 33 words; 128 bytes',
  scripts[0].includes(`needs ${load('ch5').transactions} sectors`) && scripts[1].includes(`so ${load('turned').transactions} sectors serve`)
  && scripts[2].includes(`${store(32).passes} passes, while ${32 - store(32).banksUsed} banks sit idle`) && scripts[3].includes('33 words') && store(33).passes === 1
  && scripts[4].includes(`which is ${32 * 4} bytes per tile`) && scripts[4].includes('a shift to the left by 5')
  && scripts[5].includes('64 banks and 64 lanes') && scripts[5].includes('16 lanes per cycle') && scripts[5].includes('lab 6.3'),
  JSON.stringify([load('ch5').transactions, load('turned').transactions, store(32).passes]));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['turn-and-pad-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));

// ---- sound off, parts advance
await setBox('sound', false);
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
await seek(970);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself, with its caption', (await current()) === 2 && (await caption()) === scripts[1], `part ${await current()}`);
await press('Pause');

// ---- every part's drawing against the models
await press('1. Top edge');
await seek(1000);
let d = await drawn();
check('part 1 ends: chapter 5\'s mapping, 32 sectors for one warp', d.arrangement === 'ch5' && d.sectors === load('ch5').transactions && d.sectors === 32, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Turned');
await seek(1000);
d = await drawn();
check('part 2 ends: turned, 4 sectors, and the warp writes a column of a 32-wide shared tile', d.arrangement === 'turned' && d.sectors === 4 && d.pitch === 32 && d.headline.includes('writes a column'), JSON.stringify(d));

await press('3. 32 passes');
await seek(650);
d = await drawn();
const S32 = store(32);
check('part 3 midway: one bank, 32 passes, some already served', d.passes === S32.passes && d.passes === 32 && d.banksUsed === 1 && d.served > 0 && d.served < 32 && d.dots === 32 && d.headline.startsWith(`pass ${d.served + 1} of 32`), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });
await seek(1000);
d = await drawn();
check('part 3 ends: all 32 passes served', d.served === 32, JSON.stringify(d));

await press('4. One column more');
await seek(1000);
d = await drawn();
check('part 4 ends: rows of 33 words, 32 banks, one pass', d.pitch === 33 && d.passes === store(33).passes && d.passes === 1 && d.banksUsed === 32 && d.headline === '32 banks, 1 pass', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The price');
await seek(1000);
d = await drawn();
check('part 5 ends: 128 bytes per tile, one shift against a shift and an add, and the misprint parsed',
  d.lines.length === 4 && d.lines[0] === 'unused: 32 floats × 4 bytes = 128 bytes per tile' && d.lines[1].includes('i << 5, one shift') && d.lines[2].includes('(i << 5) + i, a shift and an add') && d.lines[3].includes('i >> (5 + i)'),
  JSON.stringify(d.lines));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Other GPUs');
await seek(300);
d = await drawn();
check('part 6 first: 64 banks and 64 lanes, rows of 32 words, the model\'s count', d.banks === 64 && d.pitch === 32 && d.passes === store(32, 64, 64).passes && d.passes === 16 && d.dots === 64, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 6 ends: padded, the count changes again, and who documents what', d.pitch === 33 && d.passes === store(33, 64, 64).passes && d.lines.join('|') === 'NVIDIA: 32 banks, documented rule|AMD: 16 lanes per cycle, counts differ|Apple: nothing documented', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. 32 passes');
await seek(650);
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

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
