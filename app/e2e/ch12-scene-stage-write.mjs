// Browser check for the narrated animation "Gather, then write together" in section 12.6: six
// parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing shows the numbers of the
// model (src/sim/compactstore.ts sceneData(), pinned in compactstore.test.ts: the book's
// sixteen keys, offsets 0..7 for the eight kept keys; direct scatter 4 stores and 4 segments,
// 64 bytes for 32 useful; staged 2 stores and 2 segments; coarsened by two, 1 chunk of 16 keys
// and still 2 stores; each block keeps 4 keys for 8 threads).
// It passes with or without content/ch12/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-scene-stage-write.mjs [screenshot-prefix] [--dark]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const S = 'section.scene[aria-label="Gather, then write together"]';
const N = 6;
const LABELS = ['Sixteen keys', 'Direct scatter', 'Stage, then copy out', 'The tally', 'Coarsen by two', 'When staging pays'];

const res = await fetch(BASE + '/content/ch12/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch12/narration/stage-write-${n}.txt`)).text()));

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
const svg = (name) => page.$eval(S + ' .stage svg', (e, name) => e.getAttribute('data-' + name), name);
const svgText = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
const written = () => page.$$eval(S + ' [data-out]', (gs) => gs.filter((g) => g.dataset.written === '1').length);
const touched = () => page.$$eval(S + ' [data-segment]', (gs) => gs.filter((g) => g.dataset.touched === '1').length);

await goto(page, 'ch12/12.6');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('stage-write-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()));
  await setBox('captions', true);
  check('switching captions on shows part 1\'s script', (await caption()) === scripts[0]);
} else {
  check('with no audio yet, part 1\'s script is shown regardless', (await hasCaption()) && (await caption()) === scripts[0], await caption());
  check('there is a captions option, off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
  await setBox('captions', true);
}
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(400);
check('the captions choice is remembered after a reload', await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
await press('1. Sixteen keys');
await seek(1000);
check('part 1: sixteen keys, eight kept, their offsets 0 to 7', (await page.$$(S + ' [data-key]')).length === 16 && (await svg('offsets')) === '8' &&
  JSON.stringify(await page.$$eval(S + ' [data-offset]', (t) => t.map((x) => Number(x.dataset.offset)))) === JSON.stringify([0, 1, 2, 3, 4, 5, 6, 7]));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Direct scatter');
await seek(450);
const mid = Number(await svg('stores'));
check('part 2, part way: some but not all of the 4 stores', mid > 0 && mid < 4, String(mid));
await seek(1000);
check('part 2: 4 store instructions touching 4 segments', (await svg('stores')) === '4' && (await svg('segments')) === '4' && (await page.$$(S + ' [data-store]')).length === 4);
check('part 2: every output slot written, both 16-byte segments touched', (await written()) === 8 && (await touched()) === 2);
check('part 2: the stores have 2, 2, 3 and 1 lanes', (await svgText()).includes('block 0, warp 0, 2 lanes') && (await svgText()).includes('block 1, warp 0, 3 lanes') && (await svgText()).includes('block 1, warp 1, 1 lane'));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Stage, then copy out');
await seek(500);
check('part 3, middle: all 8 keys staged, the barrier drawn, nothing written yet', (await svg('placed')) === '8' && !!(await page.$(S + ' [data-barrier]')) && (await written()) === 0);
await seek(1000);
check('part 3: 2 store instructions, 2 segments', (await svg('stores')) === '2' && (await svg('segments')) === '2' && (await written()) === 8);
check('part 3: each store is 4 lanes of one block', (await svgText()).includes('block 0, warp 0, 4 lanes') && (await svgText()).includes('block 1, warp 0, 4 lanes'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The tally');
await seek(1000);
check('part 4: 4 stores become 2', (await svg('before')) === '4' && (await svg('after')) === '2');
check('part 4: 64 bytes for 32 useful, then 32 for 32', (await svgText()).includes('4 segments, 64 bytes moved for 32 useful') && (await svgText()).includes('2 segments, 32 bytes moved for 32 useful'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Coarsen by two');
await seek(1000);
check('part 5: one block, one chunk, sixteen keys per block-level scan, still 2 stores', (await svg('chunks')) === '1' && (await svg('keys')) === '16' && (await svg('stores')) === '2' && (await svgText()).includes('8 threads, 2 keys each'));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. When staging pays');
await seek(500);
check('part 6: 4 kept keys for 8 threads, so CUB would scatter directly', (await svg('kept')) === '4' && (await svg('threads')) === '8' && (await svgText()).includes('CUB would scatter directly'));

// ---------------------------------------------------------------- playing
await press('1. Sixteen keys');
check('the caption follows the part', (await caption()) === scripts[0]);
await setBox('sound', false);
await press('Play');
await pause(1500);
const s1 = await slider();
check('it plays with the sound off: the position advances', s1 > 20, String(s1));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2);
await seek(985);
await press('Resume');
await pause(1800);
check('at the end of a part the next one starts by itself', (await current()) === 2, String(await current()));
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

const expected404 = (u) => /\/content\/ch12\/narration\/(manifest\.json|stage-write-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
