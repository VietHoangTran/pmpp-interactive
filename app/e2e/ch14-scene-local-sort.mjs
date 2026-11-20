// Browser check for the narrated animation "Sort locally, then write in runs" in section
// 14.6: six parts, captions are an option, the parts advance by themselves, it runs with
// the sound off, the scripts say the model's numbers, and every part's drawing agrees with
// src/sim/radixlocal.ts and src/sim/radixcost.ts (imported here directly).
// It passes with or without content/ch14/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch14-scene-local-sort.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const M = await import('../src/sim/radixlocal.ts');
const RC = await import('../src/sim/radixcost.ts');

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
const S = 'section.scene[aria-label="Sort locally, then write in runs"]';
const N = 6;
const LABELS = ['Direct scatter', 'Wider digits', 'Sort locally', 'The table', 'Write in runs', 'What it buys'];

// ---- the model's numbers, computed the way the scene computes them
const DIRECT = M.directPass(M.BOOK_KEYS, 0, 1, 4);
const LOCAL = M.localPass(M.BOOK_KEYS, 0, 1, 4);
const WKEYS = M.uniformKeys(4096, 1);
const warpAt = (r) => M.warpStores(M.directPass(WKEYS, 0, r, 256), 32, 4, 32);
const W1 = warpAt(1);
const PICK = W1.findIndex((w) => w.units.length === Math.round(M.summarise(W1).meanUnits));
const WARP = { 1: W1[PICK], 2: warpAt(2)[PICK], 8: warpAt(8)[PICK] };
const U = RC.PROBLEMS[0];
const bars = Object.fromEntries([1, 2, 4, 8].map((r) => [r, {
  direct: RC.sampledDirect(U, 32, r, 256).meanUnits,
  local: RC.sampled({ dist: 'uniform', keyBits: 32, rowLength: U.rowLength, r, threads: 256, c: 1 }).meanUnits,
}]));
const one = (v) => (Math.round(v * 10) / 10).toFixed(1);
const row = (keys) => keys.map((k) => M.bin(k, 4)).join(' ');

const res = await fetch(BASE + '/content/ch14/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch14/narration/local-sort-${n}.txt`)).text()));

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
  const warp = svg.querySelector('[data-warp]');
  const view = svg.querySelector('[data-view]');
  const rowKeys = (name) => [...svg.querySelectorAll(`[data-row="${name}"] [data-cell]`)].map((c) => c.dataset.key).filter(Boolean).join(' ');
  return {
    view: view?.dataset.view ?? (warp ? 'warp' : null),
    warp: warp ? { r: Number(warp.dataset.r), runs: Number(warp.dataset.runs), sectors: Number(warp.dataset.sectors), min: Number(warp.dataset.min), boxes: warp.querySelectorAll('[data-sector]').length } : null,
    counters: Object.fromEntries([...svg.querySelectorAll('[data-counter]')].map((c) => [c.dataset.counter, c.querySelector('[data-value]').textContent.trim()])),
    blocksDone: view?.dataset.blocksDone ?? null,
    table: view?.dataset.table ?? null,
    scanned: view?.dataset.scanned ?? null,
    block2: view?.dataset.block2 ?? null,
    bars: Object.fromEntries([...svg.querySelectorAll('[data-bars]')].map((b) => [b.dataset.bars, [b.dataset.direct, b.dataset.local]])),
    input: rowKeys('input'),
    local: rowKeys('local'),
    output: rowKeys('output'),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    note: svg.querySelector('[data-note]')?.textContent.trim() ?? '',
  };
}, S);

await goto(page, 'ch14/14.6');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*:;]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*:;]/.test(s)));
const st = M.starts(LOCAL.blocks.map((b) => b.counts), 'digit-major');
const b2 = LOCAL.writes.filter((w) => w.block === 2).map((w) => w.addr);
check('part 1 says 4 to 7 sectors against 4, the model\'s bounds for a one-bit warp store',
  scripts[0].includes('4 to 7 sectors of 32 bytes') && scripts[0].includes('costs 4') && W1.every((w) => w.units.length >= 4 && w.units.length <= 7 && w.minUnits === 4));
check('part 4 says the starts the model computes: 0, 2, 4, 5 and 8, 10, 12, 15',
  scripts[3].includes(`${st.map((x) => x[0]).slice(0, 3).join(', ')} and ${st[3][0]}`) && scripts[3].includes(`${st.map((x) => x[1]).slice(0, 3).join(', ')} and ${st[3][1]}`), JSON.stringify(st));
check('part 5 says block 2 writes position 4, then 12, 13 and 14', scripts[4].includes(`position ${b2[0]}`) && scripts[4].includes(`at ${b2[1]}, ${b2[2]} and ${b2[3]}`), b2.join(','));
check('part 6 says the sampled averages: about 16 to 6 at 4 bits, 30 to 22 at 8 bits',
  scripts[5].includes(`from about ${Math.round(bars[4].direct)} sectors to ${Math.round(bars[4].local)} at 4 bits`) && scripts[5].includes(`from ${Math.round(bars[8].direct)} to ${Math.round(bars[8].local)} at 8 bits`),
  JSON.stringify(bars));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['local-sort-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
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
await press('1. Direct scatter');
await seek(300);
let d = await drawn();
check('part 1, first half: Fig. 14.5 on the book\'s keys, block 0 writes 0, 8, 1, 9',
  d.view === 'direct-book' && d.input === row(M.BOOK_KEYS) && d.output === row(DIRECT.out) && d.headline.endsWith(DIRECT.writes.slice(0, 4).map((w) => w.addr).join(', ')), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1a.png' });
await seek(1000);
d = await drawn();
check('part 1 ends: one warp at 1 bit, its runs and sectors from the model',
  d.warp?.r === 1 && d.warp.runs === WARP[1].runs && d.warp.sectors === WARP[1].units.length && d.warp.boxes === WARP[1].units.length && d.warp.min === 4
  && d.counters.sectors === String(WARP[1].units.length) && d.counters.minimum === '4', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1b.png' });

await press('2. Wider digits');
await seek(200);
d = await drawn();
check('part 2, first: the same warp at 2 bits', d.warp?.r === 2 && d.warp.runs === WARP[2].runs && d.warp.sectors === WARP[2].units.length, JSON.stringify(d.warp));
await seek(1000);
d = await drawn();
check('part 2 ends: 8 bits, nearly a sector per key', d.warp?.r === 8 && d.warp.runs === WARP[8].runs && d.warp.sectors === WARP[8].units.length && WARP[8].units.length >= 24, JSON.stringify(d.warp));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Sort locally');
await seek(1000);
d = await drawn();
check('part 3 ends: all four blocks sorted locally, Fig. 14.8\'s rows',
  d.view === 'local-sort' && d.blocksDone === '4' && d.local === row(LOCAL.blocks.flatMap((b) => b.sorted)) && d.local === '1100 0110 0011 1001 1000 1010 1111 0101 0110 1001 1011 1101 0100 1010 0000 0111', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The table');
await seek(400);
d = await drawn();
check('part 4 midway: the table is there, the scan not yet', d.table === LOCAL.table.join(' ') && d.scanned === '', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4 ends: table 2 2 1 3 2 2 3 1, scan 0 2 4 5 8 10 12 15 (Fig. 14.9)', d.table === '2 2 1 3 2 2 3 1' && d.scanned === LOCAL.scanned.join(' ') && d.scanned === '0 2 4 5 8 10 12 15', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Write in runs');
await seek(1000);
d = await drawn();
check('part 5 ends: block 2 writes 4, 12, 13, 14 and the output is iteration 1 of Fig. 14.4',
  d.block2 === b2.join(' ') && d.output === row(M.passInput(M.BOOK_KEYS, 1, 1)), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. What it buys');
await seek(1000);
d = await drawn();
check('part 6 ends: the bars are the sampled averages for r = 1, 2, 4, 8',
  [1, 2, 4, 8].every((r) => d.bars[r]?.[0] === one(bars[r].direct) && d.bars[r]?.[1] === one(bars[r].local)), JSON.stringify(d.bars));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- dark theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await press('5. Write in runs');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part5-dark.png' });
await press('1. Direct scatter');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part1-dark.png' });
await setBox('captions', false);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch14\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u) || /\/api\/lab-file\?path=ch14%2Flab2_multibit/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
