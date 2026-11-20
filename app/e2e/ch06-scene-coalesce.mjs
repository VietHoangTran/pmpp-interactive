// Browser check for the narrated animation "One instruction, thirty-two addresses" in
// section 6.1: six parts, captions are an option, the parts advance by themselves, it
// runs with the sound off, and every part's drawing agrees with the coalescing model
// (src/sim/coalesce.ts, imported here directly).
// It passes with or without content/ch06/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch06-scene-coalesce.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { exprAccess, tileLoad } from '../src/sim/coalesce.ts';

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
const S = 'section.scene[aria-label="One instruction, thirty-two addresses"]';
const N = 6;
const LABELS = ['Addresses', 'Four sectors', 'k advances', 'Column by column', 'Eight times', 'Turn the corner'];

// The model's numbers for the scene's warp: Width 1024, 32 lanes, NVIDIA's 32-byte sectors.
const P = (k) => ({ Width: 1024, k, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0 });
const ROW = exprAccess('n-row', P(0), 32);
const COL = exprAccess('n-col', P(0), 32);
const turn = (a) => tileLoad({ layout: 'col', arrangement: a, T: 32, bx: 0, ph: 0, Width: 1024, warpWidth: 32 }).warps[0].access;
const GAP = COL.lanes[1].byte - COL.lanes[0].byte;

const res = await fetch(BASE + '/content/ch06/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch06/narration/coalesce-${n}.txt`)).text()));

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
const num = (s) => (s == null || s === '' ? null : Number(String(s).replace(/,/g, '')));
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const g = svg.querySelector('[data-layout]');
  const t = svg.querySelector('[data-arrangement]');
  return {
    layout: g?.dataset.layout ?? null,
    k: g ? Number(g.dataset.k) : null,
    transactions: g?.dataset.transactions ?? t?.dataset.transactions ?? null,
    arrangement: t?.dataset.arrangement ?? null,
    sectors: svg.querySelectorAll('[data-sector]').length,
    counters: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((c) => [c.dataset.counter, c.querySelector('[data-value]').textContent.trim()])),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    note: svg.querySelector('[data-note]')?.textContent.trim() ?? '',
    text: svg.textContent,
  };
}, S);

await goto(page, 'ch06/6.1');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the model\'s: 4 sectors, 4096 bytes apart, 32 transactions, a factor of 8, 4 sectors after the turn',
  scripts[1].includes('exactly 4 sectors') && ROW.transactions === 4
  && scripts[3].includes(`${GAP} bytes apart`) && scripts[3].includes(`That is ${COL.transactions} transactions`) && COL.transactions === 32
  && scripts[4].includes(`a factor of ${COL.bytesFetched / COL.bytesNeeded}`)
  && scripts[5].includes(`and ${turn('turned').transactions} sectors come back`),
  JSON.stringify([ROW.transactions, GAP, COL.transactions, turn('turned').transactions]));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['coalesce-1'];
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
await press('1. Addresses');
await seek(1000);
let d = await drawn();
check('part 1 ends: row-major, k = 0, lane 0 to N[0] … lane 31 to N[31], no sectors yet',
  d.layout === 'row' && d.k === 0 && d.sectors === 0 && d.note.includes(`lane 31 → N[${ROW.lanes[31].element}]`) && d.headline === 'k = 0: each lane computes k·Width + col', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Four sectors');
await seek(1000);
d = await drawn();
check('part 2 ends: 4 sectors, 128 bytes fetched, 128 needed',
  num(d.transactions) === ROW.transactions && d.sectors === 4 && num(d.counters['bytes fetched']) === 128 && num(d.counters['bytes needed']) === 128 && d.text.includes('one 128-byte line = 4 sectors'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. k advances');
await seek(450);
d = await drawn();
const kMid = d.k;
check('part 3 midway: k has advanced and fetched equals needed', kMid > 0 && kMid < 7 && d.counters['bytes fetched'] === d.counters['bytes needed'] && num(d.counters['bytes fetched']) === 128 * (kMid + 1), JSON.stringify(d));
check('part 3 headline: 128 bytes of the next row, a whole row (4,096 bytes) further on, not the adjacent bytes',
  d.headline === `k = ${kMid}: 128 bytes from the next row of N, 4,096 bytes further on`, d.headline);
await seek(1000);
d = await drawn();
const r7 = exprAccess('n-row', P(7), 32);
check('part 3 ends at k = 7: the eighth row, 4 sectors, 1,024 bytes fetched and needed',
  d.k === 7 && num(d.transactions) === 4 && d.text.includes(`bytes ${(r7.segments[0] * 32).toLocaleString('en-US')} to`) && num(d.counters['bytes fetched']) === 1024 && num(d.counters['bytes needed']) === 1024, JSON.stringify(d));

await press('4. Column by column');
await seek(1000);
d = await drawn();
check('part 4 ends: column-major, 32 sectors, 1,024 bytes fetched for 128 needed, 4,096 bytes apart',
  d.layout === 'col' && d.sectors === 32 && num(d.transactions) === COL.transactions && num(d.counters['bytes fetched']) === COL.bytesFetched && num(d.counters['bytes needed']) === COL.bytesNeeded
  && d.text.includes(`${GAP.toLocaleString('en-US')} bytes apart`), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Eight times');
await seek(1000);
d = await drawn();
let f = 0, nd = 0;
for (let k = 0; k <= 7; k++) { const a = exprAccess('n-col', P(k), 32); f += a.bytesFetched; nd += a.bytesNeeded; }
check('part 5 ends at k = 7: 8,192 bytes fetched for 1,024 needed, and the cache caveat',
  d.k === 7 && num(d.counters['bytes fetched']) === f && num(d.counters['bytes needed']) === nd && f === 8 * nd && d.headline === 'k = 7: fetched 8 times what the warp needed' && d.note.includes('a cache might keep it'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Turn the corner');
await seek(300);
d = await drawn();
check('part 6 first: chapter 5\'s mapping reads the top edge, 32 sectors', d.arrangement === 'ch5' && num(d.transactions) === turn('ch5').transactions && num(d.transactions) === 32, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 6 ends: corner turned, 4 sectors, every byte used, and the column write handed to 6.4',
  d.arrangement === 'turned' && num(d.transactions) === turn('turned').transactions && num(d.transactions) === 4 && d.text.includes('100 % of fetched bytes used') && d.text.includes('section 6.4'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('5. Eight times');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part5-light.png' });
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
