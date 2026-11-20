// Browser check for the narrated animation "Why an asynchronous copy needs pinned memory"
// in section 23.3: six parts, captions are an option (off by default, remembered), parts
// advance by themselves, it runs with the sound off, and every part's drawing agrees with
// the model (src/sim/chunkpipe.ts, pinnedScene(), imported directly). The numbers the
// scripts say are the model's too. It passes with or without content/ch23/narration/manifest.json.
// Usage: node e2e/ch23-scene-pinned.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { pinnedScene } from '../src/sim/chunkpipe.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const S = 'section.scene[aria-label="Why an asynchronous copy needs pinned memory"]';
const N = 6;
const LABELS = ['Pages', 'DMA', 'Staging', 'Pageable', 'Pinned', 'Today'];
const M = pinnedScene();

const res = await fetch(BASE + '/content/ch23/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch23/narration/pinned-${n}.txt`)).text()));

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
const svgText = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
const headline = () => page.$eval(S + ' [data-headline]', (e) => e.textContent.trim());

await goto(page, 'ch23/23.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
check("the numbers the scripts say are the model's: 4 chunks, 16 pageable (the one-piece time), 12 operations, 10 pinned",
  scripts[3].includes('4 chunks') && scripts[3].includes(`takes ${M.pageable.total} units`) && M.pageable.total === M.pageable.serial
  && scripts[4].includes(`all ${M.pinned.ops.length} operations`) && scripts[4].includes(`${M.pinned.total} units instead of ${M.pageable.total}`),
  JSON.stringify({ pageable: M.pageable.total, pinned: M.pinned.total, ops: M.pinned.ops.length }));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['pinned-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check("switching captions on shows part 1's script", (await hasCaption()) && (await caption()) === scripts[0]);
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

// ---- every part against the model
await press('1. Pages');
await seek(0);
let pages = await page.$$eval(S + ' [data-page]', (g) => g.length);
let frames = await page.$$eval(S + ' [data-frame]', (g) => g.map((e) => e.dataset.state));
check(`part 1 starts: ${M.pages} pages, each in its own frame; the frame it will move to is free`,
  pages === M.pages && frames.filter((x) => x === 'mine').length === M.pages && frames[M.moved.to] === 'free' && frames.length === M.frameCount, JSON.stringify(frames));
await seek(1000);
frames = await page.$$eval(S + ' [data-frame]', (g) => g.map((e) => e.dataset.state));
check(`part 1 ends: the OS has moved page ${M.moved.page} to frame ${M.moved.to}, and its old frame is free`,
  frames[M.moved.to] === 'mine' && frames[M.frames[M.moved.page]] === 'free' && (await headline()).includes(`moves page ${M.moved.page} to frame ${M.moved.to}`), JSON.stringify(frames));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. DMA');
await seek(300);
check('part 2: the engine reads a frame by its physical address', (await headline()).includes('by its physical address'), await headline());
await seek(1000);
frames = await page.$$eval(S + ' [data-frame]', (g) => g.map((e) => e.dataset.state));
check('part 2 ends: a frame reassigned mid-copy is marked as not yours', frames.includes('reassigned') && (await svgText()).includes('not yours'), JSON.stringify(frames));

await press('3. Staging');
await seek(1000);
const st = await page.$eval(S + ' [data-staging]', (e) => ({ pieces: Number(e.dataset.staging), returns: Number(e.dataset.returns) }));
const drawnPieces = await page.$$eval(S + ' [data-piece]', (g) => g.length);
check(`part 3 ends: ${M.plan.pieces.length} pieces through the staging buffers, the call returning at ${M.plan.returnsAt}, against ${M.plan.pinnedTime} from pinned memory`,
  st.pieces === M.plan.pieces.length && st.returns === M.plan.returnsAt && drawnPieces === M.plan.pieces.length && !!(await page.$(S + ' [data-returned]'))
  && (await svgText()).includes(`${M.plan.pinnedTime} for the same bytes`), JSON.stringify(st));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Pageable');
await seek(1000);
let tl = await page.$eval(S + ' [data-timeline]', (e) => ({ which: e.dataset.timeline, total: Number(e.dataset.total) }));
let ops = await page.$$eval(S + ' [data-op]', (g) => g.length);
const spans = await page.$$eval(S + ' [data-host]', (g) => g.map((e) => e.dataset.host));
check(`part 4 ends: pageable, ${M.pageable.total}, every operation drawn, the host held by every copy`,
  tl.which === 'pageable' && tl.total === M.pageable.total && ops === M.pageable.ops.length
  && spans.filter((x) => x === 'staging').length === M.pageable.ops.filter((o) => o.kind !== 'kernel').length
  && (await headline()).includes(`${M.pageable.total}: the same as doing it in one piece (${M.pageable.serial})`), JSON.stringify({ tl, ops, spans }));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Pinned');
await seek(400);
check('part 5 midway: the host queued everything at once', (await page.$eval(S + ' [data-issued]', (e) => Number(e.dataset.issued))) === M.pinned.ops.length);
await seek(1000);
tl = await page.$eval(S + ' [data-timeline]', (e) => ({ which: e.dataset.timeline, total: Number(e.dataset.total) }));
ops = await page.$$eval(S + ' [data-op]', (g) => g.length);
check(`part 5 ends: pinned, ${M.pinned.total}, both totals side by side`,
  tl.which === 'pinned' && tl.total === M.pinned.total && ops === M.pinned.ops.length
  && (await svgText()).includes(`pageable ${M.pageable.total}, pinned ${M.pinned.total}`), JSON.stringify({ tl, ops }));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Today');
await seek(1000);
const cols = await page.$$eval(S + ' [data-col]', (g) => g.map((e) => e.dataset.col));
const t6 = await svgText();
check('part 6 ends: CUDA, HIP and Apple silicon, each with its fact', JSON.stringify(cols) === JSON.stringify(['CUDA', 'HIP', 'Apple silicon'])
  && t6.includes('cudaMallocHost') && t6.includes('hipHostMalloc') && t6.includes('nothing to'), JSON.stringify(cols));

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('5. Pinned');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part5-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
