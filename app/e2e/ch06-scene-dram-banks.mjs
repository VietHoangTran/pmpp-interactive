// Browser check for the narrated animation "Why one bank is not enough" in section 6.2:
// six parts, captions are an option, the parts advance by themselves, it runs with the
// sound off, and every part's drawing agrees with the model it is computed from
// (src/sim/dram.ts, imported here directly).
// It passes with or without content/ch06/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch06-scene-dram-banks.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { banksNeeded, busBusy, busGaps, channelTimeline, inFlight } from '../src/sim/dram.ts';

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
const pct = (v) => Math.round(v * 1000) / 10;
const S = 'section.scene[aria-label="Why one bank is not enough"]';
const N = 6;
const LABELS = ['One bank', 'Mostly idle', 'A second bank', 'R + 1 banks', 'A conflict', 'In flight'];

// The scene's case, from the model (the numbers DramBanksScene.tsx uses: the book's illustration, p. 134).
const R = 20;
const GBS = 16;
const SPAN = 3 * (R + 1) + 1;
const CONFLICT_AT = 30;
const COUNT_AT = 50.5;
const FULL = banksNeeded(R);
const ONE = channelTimeline({ R, banks: 1 });
const TWO = channelTimeline({ R, banks: 2 });
const FULL_TL = channelTimeline({ R, banks: FULL });
const CONFLICT = channelTimeline({ R, banks: FULL, forced: [CONFLICT_AT] });
const GAP = busGaps(CONFLICT, R, 84).reduce((s, [a, e]) => s + e - a, 0);

const res = await fetch(BASE + '/content/ch06/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch06/narration/dram-banks-${n}.txt`)).text()));

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
/** The drawing: its data-* attributes and its visible text. */
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const data = Object.fromEntries(Object.entries(svg.dataset).map(([k, v]) => [k, Number.isNaN(Number(v)) ? v : Number(v)]));
  return { data, text: [...svg.querySelectorAll('text')].map((t) => t.textContent.trim()) };
}, S);
const headline = (d) => d.text[d.text.length - 2];
const has = (list, s) => list.some((x) => x.includes(s));

await goto(page, 'ch06/6.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check("the numbers the scripts say are the model's",
  scripts[1].includes(`${R} times as long as the burst`) && scripts[1].includes(`out of every ${R + 1} time slots`)
  && scripts[1].includes(`about ${pct(ONE.utilisation)} percent`) && scripts[1].includes(`about ${(Math.round(ONE.utilisation * GBS * 100) / 100).toFixed(2)} of them`)
  && scripts[1].includes(`${GBS} gigabytes per second`)
  && scripts[3].includes(`With ${FULL} banks`) && pct(FULL_TL.utilisation) === 100 && scripts[3].includes('100 percent')
  && scripts[4].includes(`dry for ${GAP} burst times`)
  && scripts[5].includes(`${inFlight(FULL_TL, COUNT_AT)} requests must be in flight`),
  scripts.join(' / '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}]/.test(s)), scripts.find((s) => /[<>=×%/_(){}]/.test(s)));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['dram-banks-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check("switching captions on shows part 1's script, without audio tags", (await hasCaption()) && (await caption()) === scripts[0] && !(await caption()).includes('['));
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
check('before playing, part 1 has drawn no burst yet', d.data.beat === 1 && d.data.busy === 0 && headline(d) === 'the bank is reaching into its cell array; nothing on the bus yet', JSON.stringify(d.data) + ' ' + headline(d));
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
await press('1. One bank');
await seek(500);
d = await drawn();
check("part 1 midway: the bursts counted are the model's for the time drawn so far", d.data.until > R && d.data.busy === Math.round(busBusy(ONE, 0, d.data.until) * d.data.until) && headline(d) === `the bus has carried data for ${d.data.busy} of ${Math.floor(d.data.until)} burst times`, JSON.stringify(d.data) + ' ' + headline(d));
await seek(1000);
d = await drawn();
check('part 1 ends: 3 bursts in 64 burst times, one every 21', d.data.until === SPAN && d.data.busy === 3 && d.data.banks === 1 && d.data.r === R, JSON.stringify(d.data));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Mostly idle');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
d = await drawn();
check('part 2: 4.8 % busy, 0.76 of 16 GB/s, as the book says (p. 134)', d.data.util === pct(ONE.utilisation) && d.data.util === 4.8 && d.data.gb === 0.76 && headline(d) === 'one bank: bus busy 1 of every 21 burst times, 4.8%' && has(d.text, '0.76 GB/s'), JSON.stringify(d.data) + ' ' + headline(d));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. A second bank');
await seek(1000);
d = await drawn();
check('part 3 ends: two banks, 9.5 %, twice one bank', d.data.banks === 2 && d.data.util === pct(TWO.utilisation) && d.data.util === 9.5 && headline(d) === 'two banks: bursts go out back to back, bus busy 9.5%', JSON.stringify(d.data) + ' ' + headline(d));
check('and both gauges are drawn', has(d.text, 'bus busy, one bank') && has(d.text, 'bus busy, two banks'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. R + 1 banks');
await seek(500);
d = await drawn();
check('part 4 midway: banks being added, utilisation the model\'s k/21', d.data.banks > 2 && d.data.banks < FULL && d.data.util === pct(channelTimeline({ R, banks: d.data.banks }).utilisation) && d.data.util === pct(d.data.banks / (R + 1)), JSON.stringify(d.data));
await seek(1000);
d = await drawn();
check('part 4 ends: 21 banks, R + 1, the bus 100 % busy', d.data.banks === FULL && d.data.util === 100 && d.data.bound === 100 && headline(d) === '21 banks, R + 1: the bursts tile the bus, busy 100%', JSON.stringify(d.data) + ' ' + headline(d));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. A conflict');
await seek(1000);
d = await drawn();
const hit = CONFLICT.accesses[CONFLICT_AT];
check('part 5 ends: the conflict leaves the bus idle for 20 burst times', d.data.gap === GAP && GAP === R && d.data.waited === R && d.data.bank === hit.bank && has(d.text, `bus idle for ${R} burst times`), JSON.stringify(d.data));
check('and names the requests and the bank', headline(d) === `bank conflict: request ${CONFLICT_AT} waited for bank ${hit.bank} until time ${hit.start}`, headline(d));
const failRects = await page.$$eval(S + ' .stage svg rect', (rs) => rs.filter((r) => (r.getAttribute('fill') || '').includes('--fail') || (r.getAttribute('stroke') || '').includes('--fail')).length);
check('the conflicting pair and the gap are drawn in the failure colour', failRects >= 3, String(failRects));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. In flight');
await seek(400);
d = await drawn();
check('part 6: 21 requests in flight at the marked moment, one per bank', d.data.inflight === inFlight(FULL_TL, COUNT_AT) && d.data.inflight === FULL && has(d.text, `${FULL} requests in flight`), JSON.stringify(d.data));
await seek(1000);
d = await drawn();
check('part 6 ends on the two levers', headline(d) === 'each one is a load some warp issued: more warps, or more bytes per load', headline(d));
if (out) await page.screenshot({ path: out + '-part6.png' });

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

// ---- dark theme and a narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(200);
const colours = await page.$eval(S + ' .stage svg', (svg) => [...svg.querySelectorAll('[fill], [stroke]')].flatMap((e) => [e.getAttribute('fill'), e.getAttribute('stroke')]).filter((f) => f && f !== 'none' && !f.startsWith('var(')));
check('the drawing uses theme variables only, so it follows dark mode', colours.length === 0, colours.join(','));
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|dram-banks-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
