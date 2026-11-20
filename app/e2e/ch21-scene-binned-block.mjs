// Browser check for the narrated animation "A block walks its neighbourhood" in section
// 21.5: six parts, captions are an option (off by default, remembered), it runs with the
// sound off, parts advance by themselves, the scripts' numbers are the model's, and every
// part's drawing agrees with src/sim/cutoff.ts (imported inside the page).
// It passes with or without content/ch21/narration/manifest.json.
// Usage: node e2e/ch21-scene-binned-block.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

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
const S = 'section.scene[aria-label="A block walks its neighbourhood"]';
const N = 6;
const LABELS = ['The list', 'Stage a bin', 'Test and diverge', 'Down the list', 'Tally', 'Overflow'];

const res = await fetch(BASE + '/content/ch21/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch21/narration/binned-block-${n}.txt`)).text()));

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
  return {
    ...svg.dataset,
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    staged: svg.querySelector('[data-staged]')?.dataset.staged ?? null,
    lanes: [...svg.querySelectorAll('[data-lane]')].map((g) => g.dataset.outcome),
    overflow: svg.querySelectorAll('[data-overflow]').length,
    timeline: !!svg.querySelector('[data-timeline]'),
    tally: svg.querySelector('[data-tally]')?.textContent.replace(/\s+/g, ' ') ?? '',
    text: svg.textContent.replace(/\s+/g, ' '),
  };
}, S);

await goto(page, 'ch21/21.5');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// The model's numbers, computed in the page from the same module the scene draws from.
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/cutoff.ts');
  const { world, binning, walk } = m.sceneCase();
  const L = walk.list;
  let div = null;
  for (const e of L) {
    const k = e.slots.findIndex((s, j) => s >= 0 && m.diverges(m.acceptMasks(e, j, 32)[0]));
    const d = e.slots.findIndex((s) => s < 0);
    if (k >= 0 && d >= 0) { div = { entry: e.k, k, d, row: e.outcome[k] }; break; }
  }
  const k4 = Math.min(L.length - 1, 1 + Math.floor(Math.min(1, Math.max(0, (0.5 - 0.08) / (0.95 - 0.08))) * (L.length - 1)));
  return {
    list: L.length, threads: walk.points.length, cap: m.SCENE.capacity, atoms: world.atoms.length,
    tested: walk.tested, dummy: walk.dummyTests, accepted: walk.accepted, pct: Math.round((walk.accepted / walk.tested) * 100), direct: walk.directPairs,
    overflow: binning.overflow.length, staged0: L[0].slots.join(','), div,
    k4, cum4: L[k4].cumTested, acc4: L[k4].cumAccepted,
    inBox: L.every((e) => e.inBox),
  };
});
const fmt = (v) => v.toLocaleString('en-US');

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check('the numbers the scripts say are the model\'s',
  scripts[0].includes(`that is ${M.list} bins`) && M.inBox
  && scripts[1].includes(`All ${M.threads} threads`) && scripts[1].includes(`the same ${M.cap} slots`)
  && scripts[4].includes(`made ${M.tested} tests, ${M.dummy} of them on dummy atoms, and accepted ${M.accepted}, about ${M.pct} percent`)
  && scripts[4].includes(`these ${M.threads} points with all ${M.atoms} atoms: ${M.direct} pairs`)
  && scripts[5].includes(`${M.overflow} atoms did not fit their bins`),
  JSON.stringify(M));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['binned-block-1'];
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
await press('1. The list');
await seek(400);
let d = await drawn();
check('part 1 midway: some of the bins have lit up', Number(d.lit) > 0 && Number(d.lit) < M.list, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 1 ends: 21 bins, a list of relative offsets', Number(d.lit) === M.list && d.headline.startsWith(`${M.list} bins`), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Stage a bin');
await seek(1000);
d = await drawn();
check('part 2 ends: the first entry\'s slots staged (dummies included) and the barrier', d.entry === '0' && d.staged === M.staged0 && d.text.includes('barrier: wait until the whole bin is in shared memory') && d.lanes.length === 0, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Test and diverge');
await seek(300);
d = await drawn();
check('part 3 first: one staged atom, the model\'s accept and reject per lane, and the warp splits',
  M.div && d.entry === String(M.div.entry) && d.slot === String(M.div.k) && JSON.stringify(d.lanes) === JSON.stringify(M.div.row) && d.lanes.includes('accept') && d.lanes.includes('reject'), JSON.stringify({ d: d.lanes, m: M.div }));
await seek(1000);
d = await drawn();
check('part 3 then: a dummy slot, tested by all 16 lanes', d.slot === String(M.div.d) && d.lanes.length === M.threads && d.lanes.every((o) => o === 'dummy'), JSON.stringify(d.lanes));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Down the list');
await seek(500);
d = await drawn();
check('part 4 midway: the running counts are the model\'s at that entry', d.entry === String(M.k4) && Number(d.tested) === M.cum4 && Number(d.accepted) === M.acc4, JSON.stringify({ d, M: [M.k4, M.cum4, M.acc4] }));
await seek(1000);
d = await drawn();
check('part 4 ends on the last entry with every test counted', d.entry === String(M.list - 1) && Number(d.tested) === M.tested && Number(d.accepted) === M.accepted && Number(d.dummy) === M.dummy, JSON.stringify(d));

await press('5. Tally');
await seek(1000);
d = await drawn();
check('part 5: tests, dummies, accepted and the share, beside direct summation',
  d.tally.includes(`tests: ${fmt(M.tested)} (${fmt(M.dummy)} on dummy atoms)`) && d.tally.includes(`accepted: ${M.accepted} (${M.pct} percent)`) && d.tally.includes(`${M.threads} × ${M.atoms} = ${fmt(M.direct)} pairs`), d.tally);
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Overflow');
await seek(1000);
d = await drawn();
check('part 6: the model\'s 3 overflow atoms are marked, and the host pass follows the kernel', d.overflow === M.overflow && M.overflow === 3 && d.timeline && d.text.includes(`${M.overflow} overflow atoms of ${M.atoms}`), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('3. Test and diverge');
await seek(300);
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

const expected404 = (u) => /\/content\/ch21\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
