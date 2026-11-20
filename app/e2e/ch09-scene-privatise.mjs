// Browser check for the narrated animation "Privatise, then commit" in section 9.4: five
// parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing shows the numbers of the
// model (src/sim/histpriv.ts sceneData(), pinned in histpriv.test.ts: 14 threads on bin 7
// before, 5 on the busiest private bin, at most 3 committers per public bin, 24 global
// atomics before and 11 after).
// It passes with or without content/ch09/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-scene-privatise.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
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
const S = 'section.scene[aria-label="Privatise, then commit"]';
const N = 5;
const LABELS = ['One public histogram', 'A copy per block', 'Count locally', 'Barrier, then commit', 'The bill'];
// The model's numbers (histpriv.test.ts, "the narrated scene").
const TOTAL = [1, 1, 1, 3, 0, 0, 4, 14];
const PER_BLOCK = [
  [0, 1, 0, 1, 0, 0, 1, 5],
  [0, 0, 0, 2, 0, 0, 2, 4],
  [1, 0, 1, 0, 0, 0, 1, 5],
];

const res = await fetch(BASE + '/content/ch09/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch09/narration/privatise-${n}.txt`)).text()));

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
const attr = (sel, name) => page.$eval(S + ' ' + sel, (e, name) => e.getAttribute(name), name);
const publicBins = () => page.$$eval(S + ' [data-public-bin]', (gs) => gs.map((g) => g.dataset.value));
const copy = (k) => page.$$eval(S + ` [data-copy="${k}"] [data-copy-bin]`, (gs) => gs.map((g) => g.dataset.value));

await goto(page, 'ch09/9.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('five parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('privatise-'));
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
await press('1. One public histogram');
await seek(400);
const mid = Number(await attr('[data-contenders]', 'data-contenders'));
check('part 1, part way: some but not all of the 14 threads drawn', mid > 0 && mid < 14, String(mid));
await seek(1000);
check('part 1: 14 threads from all 3 blocks aim at bin 7', (await attr('[data-contenders]', 'data-contenders')) === '14' && (await svgText()).includes('threads aiming at bin 7: 14 from 3 blocks'));
check('part 1: the public bins end at the histogram of the 24 pixels', JSON.stringify((await publicBins()).map(Number)) === JSON.stringify(TOTAL), JSON.stringify(await publicBins()));
check('part 1: 24 pixels drawn', (await page.$$(S + ' [data-pixel]')).length === 24);
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. A copy per block');
await seek(150);
check('part 2, early: the copies hold leftovers', (await copy(0)).every((v) => v === '?'), JSON.stringify(await copy(0)));
await seek(1000);
check('part 2: every bin of every copy is zero, then barrier 1', (await copy(0)).every((v) => v === '0') && (await copy(2)).every((v) => v === '0') && (await svgText()).includes('barrier 1'));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Count locally');
await seek(1000);
check('part 3: each copy holds its block\'s counts', JSON.stringify([await copy(0), await copy(1), await copy(2)].map((c) => c.map(Number))) === JSON.stringify(PER_BLOCK));
check('part 3: the busiest private bin has 5 contenders, down from 14', (await attr('[data-contenders]', 'data-contenders')) === '5' && (await svgText()).includes('(was 14)'));
check('part 3: the public bins are untouched', (await publicBins()).every((v) => v === '0'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Barrier, then commit');
await seek(50);
check('part 4 opens on barrier 2', (await svgText()).includes('barrier 2'));
await seek(1000);
check('part 4: 11 commits, one per non-zero bin of each block', (await attr('[data-commits]', 'data-commits')) === '11' && (await page.$$(S + ' [data-commit]')).length === 11);
check('part 4: bin 7 gets one commit from each of the 3 blocks', (await page.$$(S + ' [data-commit$="-7"]')).length === 3 && (await svgText()).includes('at most 3'));
check('part 4: the public bins end right', JSON.stringify((await publicBins()).map(Number)) === JSON.stringify(TOTAL));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The bill');
await seek(1000);
check('part 5: 24 global atomics become 11', (await attr('[data-bill]', 'data-bill')) === '24-11' && (await attr('[data-before]', 'data-before')) === '24' && (await attr('[data-after]', 'data-after')) === '11' && (await svgText()).includes('24 global atomics become 11, plus 3 zeroing passes and 3 commits'));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---------------------------------------------------------------- playing
await press('1. One public histogram');
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

const expected404 = (u) => /\/content\/ch09\/narration\/(manifest\.json|privatise-\d\.mp3)$/.test(u) || /\/api\/lab-file\?path=ch09%2Flab2_privatised/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
