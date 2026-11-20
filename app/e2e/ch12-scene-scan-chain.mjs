// Browser check for the narrated animation "The scan chain orders the blocks" in section
// 12.7: six parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing shows what the model
// computes (src/sim/inplace.ts sceneRuns(), pinned in inplace.test.ts).
// It passes with or without content/ch12/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-scene-scan-chain.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { sceneRuns } from '../src/sim/inplace.ts';

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const S = 'section.scene[aria-label="The scan chain orders the blocks"]';
const N = 6;
const LABELS = ['Who overwrites whom', 'Numbers from a counter', 'Publish and wait', 'Write after the flag', 'Numbers from blockIdx', 'Each backend'];
const R = sceneRuns();

const res = await fetch(BASE + '/content/ch12/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch12/narration/scan-chain-${n}.txt`)).text()));
for (let n = 1; n <= N; n++) {
  const words = scripts[n - 1].split(' ').length;
  check(`script ${n} is 35 to 60 words`, words >= 35 && words <= 60, String(words));
}

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const setBox = async (word, on) => {
  await page.evaluate((S, word, on) => {
    const b = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (b.checked !== on) b.click();
  }, S, word, on);
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((S, label) => [...document.querySelectorAll(S + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
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
const svgText = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
const rows = (gantt) => page.$$eval(`${S} [data-gantt="${gantt}"] [data-hw]`, (gs) => gs.map((g) => ({ hw: Number(g.dataset.hw), num: g.dataset.num, phase: g.dataset.phase })));
const arrayVals = () => page.$$eval(S + ' g[data-slot]', (gs) => gs.map((g) => g.dataset.value));
const expectRows = (frame) => frame.blocks.map((b) => ({ hw: b.hw, num: b.num == null ? '' : String(b.num), phase: b.phase }));

await goto(page, 'ch12/12.7');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('scan-chain-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()));
  await setBox('captions', true);
  check("switching captions on shows part 1's script", (await caption()) === scripts[0]);
} else {
  check("with no audio yet, part 1's script is shown regardless", (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'none');
  check('there is a captions option, off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
  await setBox('captions', true);
}
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(400);
check('the captions choice is remembered after a reload', await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
await press('1. Who overwrites whom');
await seek(300);
const some = (await page.$$(S + ' line[data-arrow]')).length;
check('part 1, part way: some but not all arrows drawn', some > 0 && some < R.overlap.writes.length, String(some));
await seek(1000);
const kinds = await page.$$eval(S + ' line[data-arrow]', (ls) => ls.map((l) => [Number(l.dataset.arrow), l.dataset.kind]));
check('part 1: one arrow per kept key, kinds as the model', JSON.stringify(kinds) === JSON.stringify(R.overlap.writes.map((w) => [w.key, w.kind])), JSON.stringify(kinds));
const red = R.overlap.writes.filter((w) => w.kind === 'earlier');
check(`part 1: ${red.length} red arrows, listed with whom they overwrite`, (await page.$$(S + ' [data-part="red-list"] text')).length === red.length && (await svgText()).includes(`${red.length} of ${R.overlap.writes.length} writes land where an earlier block still has to read`));
check('part 1: the first red write is k4 over k2, read by block 0', (await svgText()).includes('k4 (block 1) → slot 2, over k2: block 0 must read k2 first'));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Numbers from a counter');
await seek(0);
check('part 2, start: nothing has started', (await svgText()).includes('nothing has started') && JSON.stringify(await rows('numbers from a counter')) === JSON.stringify(expectRows(R.dynamic.frames[0])));
await seek(1000);
check('part 2, end: blockIdx 2 is number 0, blockIdx 1 is number 1, blockIdx 0 waits for a slot', JSON.stringify(await rows('numbers from a counter')) === JSON.stringify(expectRows(R.dynamic.frames[1])) && (await svgText()).includes('blockIdx 2 starts and takes number 0'));
check('part 2: the resident slots hold blockIdx 2 and 1', JSON.stringify(await page.$$eval(S + ' [data-slot-holder]', (g) => g.map((x) => x.dataset.slotHolder))) === JSON.stringify(['2', '1']));

await press('3. Publish and wait');
await seek(1000);
check('part 3, end: #0 publishes while #1 waits (frame 4 of the model)', JSON.stringify(await rows('numbers from a counter')) === JSON.stringify(expectRows(R.dynamic.frames[4])) && (await svgText()).includes('number 1 waits for number 0'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Write after the flag');
await seek(1000);
const last = R.dynamic.frames[R.dynamic.frames.length - 1];
check('part 4, end: every block done, as the model', JSON.stringify(await rows('numbers from a counter')) === JSON.stringify(expectRows(last)));
const vals = await arrayVals();
check('part 4, end: the array holds the stable filter k1 k3 k4 k6 k8 k10 k11', JSON.stringify(vals.slice(0, 7)) === JSON.stringify(last.array.slice(0, 7).map(String)) && JSON.stringify(vals.slice(0, 7)) === JSON.stringify(['1', '3', '4', '6', '8', '10', '11']), JSON.stringify(vals));
check('part 4, end: finished with no key overwritten before it was read', R.dynamic.hazards === 0 && (await svgText()).includes('finished: the stable filter, and no key overwritten before it was read'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Numbers from blockIdx');
await seek(1000);
const lastB = R.blockIdx.frames[R.blockIdx.frames.length - 1];
check('part 5, end: 2 and 1 wait, 0 never starts, as the model', JSON.stringify(await rows('numbers from blockIdx')) === JSON.stringify(expectRows(lastB)) && R.blockIdx.outcome === 'starved');
const t5 = await svgText();
check('part 5, end: drawn as starved, with who spins on whom', t5.includes('starved') && t5.includes('spins on #1') && t5.includes('spins on #0') && t5.includes('never starts'), t5.slice(0, 300));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Each backend');
await seek(1000);
const t6 = await svgText();
check('part 6: both outcomes side by side', t6.includes('numbers from a counter') && t6.includes('numbers from blockIdx') && t6.includes('finished') && t6.includes('starved'));
check('part 6: one line each for CUDA, HIP and Metal', (await page.$$(S + ' [data-backend]')).length === 3 && t6.includes('unverified'));

// ---------------------------------------------------------------- playback
await setBox('sound', false);
await press('1. Who overwrites whom');
await seek(950);
await press('Resume');
let moved = false;
for (let i = 0; i < 40 && !moved; i++) {
  await pause(250);
  moved = (await current()) === 2;
}
check('with the sound off, part 1 runs out and part 2 starts by itself', moved, String(await current()));
check("and part 2's script is the caption", (await caption()) === scripts[1]);
await pause(800);
check('the animation clock runs in part 2', (await slider()) > 0, String(await slider()));
await press('Pause');
const held = await slider();
await pause(600);
check('pause holds the position', Math.abs((await slider()) - held) <= 2);

// Phone width: the stage scrolls inside its frame, the page does not.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));

// The lab 12.5 listing on the same page may not be written yet; its 404 is not this scene's.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
