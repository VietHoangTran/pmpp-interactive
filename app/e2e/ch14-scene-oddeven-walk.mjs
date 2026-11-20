// Browser check for the narrated animation "Why odd-even sort needs N steps" in section 14.2:
// six parts, the scripts (35 to 60 words, nothing read aloud that is a symbol, numbers that
// match the model), captions as an option (off by default, remembered; shown regardless for a
// part with no audio), parts that advance by themselves, sound off, and every part's drawing
// against src/sim/oddeven.ts. Passes with or without content/ch14/narration/manifest.json.
// Usage: node e2e/ch14-scene-oddeven-walk.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FIG_14_1, LARGEST_FIRST, bookBound, nLogN, runToSorted, step, walk } from '../src/sim/oddeven.ts';

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
const S = 'section.scene[aria-label="Why odd-even sort needs N steps"]';
const N = 6;
const LABELS = ['Even step', 'Odd step', 'Worst case', 'The bill', 'The flag', 'When to stop'];

// The model's numbers.
const FIG = walk(FIG_14_1, 2);
const WORST = runToSorted(LARGEST_FIRST, 'ours');
const REV = step([9, 8, 7, 6, 5, 4, 3, 2], 0);
const TRAP4 = runToSorted([1, 3, 2, 4], 'ours');

const res = await fetch(BASE + '/content/ch14/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch14/narration/oddeven-walk-${n}.txt`)).text()));

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
  const q = (sel, k) => svg.querySelector(sel)?.dataset[k] ?? null;
  return {
    beat: q('[data-beat]', 'beat'),
    values: q('[data-row="data"]', 'values'),
    threads: q('[data-threads]', 'threads'),
    swapped: [...svg.querySelectorAll('[data-pair][data-swapped="yes"]')].map((p) => p.dataset.pair),
    eight: q('[data-eight]', 'eight'),
    counter: q('[data-counter]', 'counter'),
    big: q('[data-big]', 'big'),
    small: q('[data-small]', 'small'),
    writers: q('[data-writers]', 'writers'),
    atomic: q('[data-writers]', 'atomic'),
    verdict: q('[data-verdict]', 'verdict'),
    flags: q('[data-flags]', 'flags'),
    mark: svg.querySelector('[data-mark]')?.dataset.mark ?? null,
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
  };
}, S);

await goto(page, 'ch14/14.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*[\]]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*[\]]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`On an even step, ${FIG[0].comparisons} threads`) && scripts[0].includes('Two of the pairs are out of order') && FIG[0].swaps === 2
  && scripts[1].includes(`only ${FIG[1].comparisons} threads have a pair`)
  && scripts[2].includes(`needs ${WORST.sortedAfter} steps`)
  && scripts[3].includes('For 1,024 values that is about half a million comparisons, against about 10 thousand') && bookBound(1024) === 524288 && nLogN(1024) === 10240
  && scripts[5].includes('Take 1, 3, 2, 4') && TRAP4.rows[0].swaps === 0,
  JSON.stringify({ fig: FIG.map((s) => [s.comparisons, s.swaps]), worst: WORST.sortedAfter }));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['oddeven-walk-1'];
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
await seek(985);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part's drawing against the model
const fin = async (label, name) => {
  await press(label);
  await seek(1000);
  if (out) await page.screenshot({ path: `${out}-${name}.png` });
  return drawn();
};
await press('1. Even step');
await seek(100);
let d = await drawn();
check('part 1 starts on Fig. 14.1\'s first row', d.beat === '1' && d.values === FIG_14_1.join(' '), JSON.stringify(d));
d = await fin('1. Even step', 'part1');
check('part 1 ends: 4 threads, pairs (4,5) and (6,7) swapped, the row after the even step',
  d.values === FIG[0].after.join(' ') && d.threads === '4' && d.swapped.join() === '4-5,6-7' && d.headline.includes(`${FIG[0].comparisons} threads compared, ${FIG[0].swaps} swapped`), JSON.stringify(d));
d = await fin('2. Odd step', 'part2');
check('part 2 ends: 3 threads, pairs (1,2) and (5,6) swapped, Fig. 14.1\'s third row',
  d.values === FIG[1].after.join(' ') && d.threads === '3' && d.swapped.join() === '1-2,5-6', JSON.stringify(d));
await press('3. Worst case');
await seek(400);
d = await drawn();
const midSteps = Number(/step (\d+)/.exec(d.headline)?.[1] ?? 0);
check('part 3 midway: the 8 is one place along per step taken', midSteps > 1 && Number(d.eight) >= midSteps - 1 && Number(d.eight) <= midSteps, JSON.stringify(d));
d = await fin('3. Worst case', 'part3');
check('part 3 ends: sorted after 7 steps, the 8 at the end', d.values === WORST.rows[WORST.sortedAfter - 1].after.join(' ') && d.eight === '7' && d.headline.startsWith(`sorted after ${WORST.sortedAfter} steps`), JSON.stringify(d));
await press('4. The bill');
await seek(450);
d = await drawn();
check('part 4: the counter totals the model\'s comparisons for the worst case', d.counter === String(WORST.comparisons), JSON.stringify(d));
d = await fin('4. The bill', 'part4');
check('part 4 ends: N = 1,024 costs 524,288 against 10,240', d.big === '524288' && d.small === '10240' && d.headline === 'N = 1,024: about 524,288 comparisons against 10,240', JSON.stringify(d));
await press('5. The flag');
await seek(400);
d = await drawn();
check('part 5 first: every swapping thread writes the flag, a data race', d.writers === String(REV.swaps) && d.atomic === 'no' && d.verdict === 'fail', JSON.stringify(d));
d = await fin('5. The flag', 'part5');
check('part 5 ends: the store made atomic', d.atomic === 'yes' && d.verdict === 'ok' && d.headline.startsWith('atomicExch(hasChanged, 1u)'), JSON.stringify(d));
await press('6. When to stop');
await seek(320);
d = await drawn();
check('part 6 after step 1: quiet, flag 0, unsorted, marked as a failure', d.flags === '0' && d.mark === 'fail' && d.values === '1 3 2 4', JSON.stringify(d));
d = await fin('6. When to stop', 'part6');
check('part 6 ends: flags as the model, sorted, stop after an even and an odd quiet step',
  d.flags === TRAP4.rows.map((r) => (r.flag ? 1 : 0)).join(' ') && d.mark === 'ok' && d.values === '1 2 3 4' && d.headline.startsWith(`stop after ${TRAP4.launched} steps`), JSON.stringify(d));

const expected404 = (u) => /\/content\/ch14\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
