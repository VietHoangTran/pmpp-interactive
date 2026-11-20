// Browser check for the narrated animation "Searching for the split" in section 13.3: six
// parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the co-rank
// model (src/sim/corank.ts, imported here directly), as do the numbers the scripts say.
// It passes with or without content/ch13/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch13-scene-corank.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { coRankTrace, knownRange, mergeByThreads, mergeStable, partition } from '../src/sim/corank.ts';

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
const S = 'section.scene[aria-label="Searching for the split"]';
const N = 6;
const LABELS = ['Three threads', 'A first guess', 'Halve the gap', 'The answer', 'Every thread at once', 'Why the tests are lopsided'];

// The model's numbers.
const A = [1, 7, 8, 9, 10];
const B = [7, 10, 10, 12];
const T3 = coRankTrace(3, A, B);
const [s0, s1, s2] = T3.steps;
const BOOK = partition(A, B, 3, 3);
const SWAP = partition(A, B, 3, 3, 'swapped');
const tags = (c) => c.map((e) => e.src + e.idx).join(' ');
const C_BOOK = mergeStable(A, B);
const C_SWAP = mergeByThreads(A, B, SWAP);
const MOVED = C_SWAP.flatMap((e, p) => (e.src !== C_BOOK[p].src || e.idx !== C_BOOK[p].idx ? [e.src + e.idx] : []));

const res = await fetch(BASE + '/content/ch13/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch13/narration/corank-search-${n}.txt`)).text()));

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
    window: svg.querySelector('[data-window]')?.dataset.window ?? null,
    slices: svg.querySelector('[data-slices]')?.dataset.slices ?? null,
    order: svg.querySelector('[data-order]')?.dataset.order ?? null,
    badge: svg.querySelector('[data-badge]')?.textContent.trim() ?? null,
    tests: [...svg.querySelectorAll('[data-test]')].map((t) => `${t.dataset.test}: ${t.textContent.trim()}`),
    fails: [...svg.querySelectorAll('[data-mark="fail"]')].map((c) => c.dataset.cell),
    oks: [...svg.querySelectorAll('[data-mark="ok"]')].map((c) => c.dataset.cell),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    text: svg.textContent,
  };
}, S);

await goto(page, 'ch13/13.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*[\]]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*[\]]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes('at 0, 3, and 6') && BOOK.map((x) => x.kCurr).join() === '0,3,6'
  && scripts[1].includes(`i is ${s0.i} and j is ${s0.j}`) && scripts[1].includes(`which holds ${s0.t1.left}`) && scripts[1].includes(`which holds ${s0.t1.right}`)
  && scripts[2].includes(`i drops by ${s0.delta}, to ${s1.i}`) && scripts[2].includes(`j rises by ${s0.delta}, to ${s1.j}`) && scripts[2].includes(`which holds ${s1.t2.left}`) && scripts[2].includes(`which holds ${s1.t2.right}`)
  && scripts[3].includes(`j drops to ${s2.j}, and i rises to ${s2.i}`) && scripts[3].includes(`${T3.candidates} candidates shrank to 1 in ${T3.steps.length} iterations`)
  && scripts[4].includes(`thread 1 at ${BOOK[1].iCurr} and ${BOOK[1].jCurr}, thread 2 at ${BOOK[2].iCurr} and ${BOOK[2].jCurr}, and the end is ${BOOK[2].iNext} and ${BOOK[2].jNext}`)
  && scripts[5].includes('B\'s first 10 lands ahead of A\'s 10') && MOVED.join(' ') === 'B1 A4',
  JSON.stringify({ s0, s1: [s1.i, s1.j], s2: [s2.i, s2.j], MOVED }));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['corank-search-1'];
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
await press('1. Three threads');
await seek(1000);
let d = await drawn();
check('part 1 ends: C cut at 0, 3 and 6, and the question of where the inputs start',
  d.beat === '1' && d.text.includes('thread 1: k = 3') && d.text.includes('thread 2: k = 6') && d.text.includes('where does thread 1 start in A?'), JSON.stringify(d.headline));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. A first guess');
await seek(200);
d = await drawn();
check('part 2 starts at the initial state (3, 0, 0, 0) with i + j = 3 and four candidates, tests not yet run',
  [d.i, d.j, d.ilow, d.jlow].join() === [s0.i, s0.j, s0.iLow, s0.jLow].join() && d.badge === 'i + j = 3 + 0 = 3' && d.window === '0-3' && d.tests.length === 0, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 2 ends: A[2] = 8 > B[0] = 7 in the failure colour, i too high, three candidates left',
  d.verdict === 'i-too-high' && d.tests.join() === 'fail: A[2] = 8 > B[0] = 7: i is too high' && d.fails.join() === 'A2,B0' && d.window === '0-2', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Halve the gap');
await seek(200);
d = await drawn();
check('part 3 mid-move: the badge still reads the old state, sum 3', d.badge === 'i + j = 3 + 0 = 3', d.badge);
await seek(560);
d = await drawn();
check('part 3 after the move: (1, 2, 0, 0), first test passes in green', [d.i, d.j, d.ilow, d.jlow].join() === '1,2,0,0' && d.badge === 'i + j = 1 + 2 = 3' && d.tests.join() === 'ok: A[0] = 1 ≤ B[2] = 10: first test passes', JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 3 ends: B[1] = 10 ≥ A[1] = 7, j too high, one candidate left',
  d.verdict === 'j-too-high' && d.tests[1] === 'fail: B[1] = 10 ≥ A[1] = 7: j is too high' && d.window === '2-2', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The answer');
await seek(1000);
d = await drawn();
check('part 4 ends: (2, 1, 1, 0), both tests green, co-rank 2, 4 candidates to 1 in 3 iterations',
  [d.i, d.j, d.ilow, d.jlow].join() === '2,1,1,0' && d.verdict === 'done' && d.oks.length === 4 && d.fails.length === 0
  && d.tests.join(' | ') === 'ok: A[1] = 7 ≤ B[1] = 10: first test passes | ok: B[0] = 7 < A[2] = 8: second test passes'
  && d.headline === `co-rank of 3: i = 2, j = 1; ${T3.candidates} candidates to 1 in ${T3.steps.length} iterations`
  && JSON.stringify(knownRange(T3, 2)) === '[2,2]', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Every thread at once');
await seek(1000);
d = await drawn();
const want5 = BOOK.map((x) => `${x.iCurr},${x.jCurr}`).join(' ') + ` | ${BOOK[2].iNext},${BOOK[2].jNext}`;
check('part 5 ends: the partition (0,0) (2,1) (5,1) and the end (5,4); thread 1 only A, thread 2 only B',
  d.slices === want5 && want5 === '0,0 2,1 5,1 | 5,4' && d.text.includes('thread 1: A[2..5) and 0 of B') && d.text.includes('thread 2: 0 of A and B[1..4)'), JSON.stringify(d.slices));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Why the tests are lopsided');
await seek(200);
d = await drawn();
check('part 6 first: the book\'s tests, C in stable order with tags', d.rule === 'book' && d.order === tags(C_BOOK) && d.fails.length === 0, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 6 ends: swapped tests, keys sorted, B1 ahead of A4 marked in the failure colour',
  d.rule === 'swapped' && d.order === tags(C_SWAP) && d.fails.length === 2 && d.headline.includes('B1 now comes before A4'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('4. The answer');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part4-light.png' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
if (out) await page.screenshot({ path: out + '-part4-dark.png' });
await setBox('captions', false);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch13\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
