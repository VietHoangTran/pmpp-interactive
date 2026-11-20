// Browser check for the narrated animation "Sorting one bit at a time" in section 14.4: six
// parts, captions are an option (off by default, remembered), the parts advance by themselves,
// it runs with the sound off, and every part's drawing shows what the model computes
// (src/sim/radixsort.ts: lsdSort and parallelPass on the book's sixteen keys).
// It passes with or without content/ch14/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch14-scene-lsd-passes.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_KEYS, bin, lsdSort, parallelPass } from '../src/sim/radixsort.ts';

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
const S = 'section.scene[aria-label="Sorting one bit at a time"]';
const N = 6;
const LABELS = ['Pass 1', 'Pass 2', 'One pass inherits', 'Passes 3 and 4', 'One unstable pass', 'A stable partition'];
const RUN = lsdSort(BOOK_KEYS, 4, 1);
const BAD = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 4 });
const PP = parallelPass(BOOK_KEYS, 0, 4);
const show4 = (ks) => ks.map((k) => bin(k, 4));

const res = await fetch(BASE + '/content/ch14/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch14/narration/lsd-passes-${n}.txt`)).text()));
for (let n = 1; n <= N; n++) {
  const words = scripts[n - 1].split(' ').length;
  check(`script ${n} is 35 to 60 words`, words >= 35 && words <= 60, String(words));
  check(`script ${n} reads no symbols aloud`, !/[=<>&|^~*#/\\+]/.test(scripts[n - 1]), scripts[n - 1]);
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
const headline = () => page.$eval(S + ' [data-headline]', (e) => e.textContent);
const row = (which) => page.$$eval(`${S} g[data-row="${which}"]`, (gs) => gs.map((g) => g.dataset.key));
const arrows = () => page.$$eval(S + ' line[data-arrow]', (ls) => ls.map((l) => [Number(l.dataset.arrow), Number(l.dataset.to)]));
const sortedBy = () => page.$eval(S + ' [data-sorted-by]', (e) => Number(e.dataset.sortedBy)).catch(() => null);
const runs = () => page.$$eval(S + ' [data-run]', (gs) => gs.map((g) => g.dataset.run));

await goto(page, 'ch14/14.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('lsd-passes-'));
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
const P = RUN.passes;
await press('1. Pass 1');
await seek(300);
const early = await arrows();
const bit0 = (i) => BOOK_KEYS[i] & 1;
check('part 1, part way: some arrows, all of them from keys whose bit 0 is 0', early.length > 0 && early.length < 16 && early.every(([i]) => bit0(i) === 0), JSON.stringify(early));
await seek(1000);
check('part 1, end: 16 arrows to the model\'s destinations', JSON.stringify((await arrows()).sort((a, b) => a[0] - b[0])) === JSON.stringify(P[0].dst.map((d, i) => [i, d])));
check("part 1, end: the output row is pass 1's", JSON.stringify(await row('bottom')) === JSON.stringify(show4(P[0].output)));
check('part 1, end: buckets "bit 0 = 0" and "bit 0 = 1", in order by the low bit', JSON.stringify(await runs()) === JSON.stringify(['bit 0 = 0', 'bit 0 = 1']) && (await sortedBy()) === 1);
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Pass 2');
await seek(1000);
check("part 2, end: the input is pass 1's output and the output pass 2's", JSON.stringify(await row('top')) === JSON.stringify(show4(P[1].input)) && JSON.stringify(await row('bottom')) === JSON.stringify(show4(P[1].output)));
check('part 2, end: four runs ending 00, 01, 10, 11, in order by the low two bits', JSON.stringify(await runs()) === JSON.stringify(['ends in 00', 'ends in 01', 'ends in 10', 'ends in 11']) && (await sortedBy()) === 2);

await press('3. One pass inherits');
await seek(200);
check('part 3, start: 0101 before 0100 in the input', (await headline()).includes('0101 comes before 0100') && (await page.$eval(S + ' [data-pair-pos]', (e) => e.dataset.pairPos)) === `${BOOK_KEYS.indexOf(0b0100)},${BOOK_KEYS.indexOf(0b0101)}`);
await seek(1000);
const pos = (p, id) => p.outIds.indexOf(id);
check('part 3, end: after pass 2, 0100 still ahead of 0101, at the model\'s positions', (await page.$eval(S + ' [data-pair-pos]', (e) => e.dataset.pairPos)) === `${pos(P[1], 12)},${pos(P[1], 6)}` && pos(P[1], 12) < pos(P[1], 6) && (await headline()).includes('stable pass keeps 0100 ahead'));
check('part 3: the pair is drawn joined across the three rows', (await page.$$(S + ' line[data-pair]')).length === 4);
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Passes 3 and 4');
await seek(400);
check("part 4, part way: pass 3's output, in order by the low three bits", JSON.stringify(await row('bottom')) === JSON.stringify(show4(P[2].output)) && (await sortedBy()) === 3);
await seek(1000);
check('part 4, end: sorted, seven zeros and nine ones', JSON.stringify(await row('bottom')) === JSON.stringify(show4(RUN.result)) && (await sortedBy()) === 4 && JSON.stringify(await runs()) === JSON.stringify(['7 keys with bit 3 = 0', '9 keys with bit 3 = 1']) && (await headline()).includes('sorted'));

await press('5. One unstable pass');
await seek(1000);
check("part 5, end: the unstable pass's output, as the model", JSON.stringify(await row('bottom')) === JSON.stringify(show4(BAD.result)));
check('part 5: the nine reversed arrows are drawn as failures', (await page.$$(S + ' line[stroke="var(--fail)"]')).length === 9);
check(`part 5: "not sorted: ${bin(BAD.result[7], 4)} lands before ${bin(BAD.result[8], 4)}"`, (await headline()).includes(`not sorted: ${bin(BAD.result[7], 4)} lands before ${bin(BAD.result[8], 4)}`));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. A stable partition');
await seek(1000);
const cells = (name) => page.$$eval(`${S} [data-scan-row="${name}"] text[data-cell]`, (ts) => ts.map((t) => t.textContent));
check("part 6: Fig. 14.6's bits", JSON.stringify(await cells('bit 0')) === JSON.stringify(PP.bits.map(String)));
check("part 6: Fig. 14.6's ones before", JSON.stringify(await cells('ones before')) === JSON.stringify(PP.threads.map((t) => String(t.onesBefore))));
check("part 6: Fig. 14.6's destinations", JSON.stringify(await cells('destination')) === JSON.stringify(PP.threads.map((t) => String(t.dst))));
check('part 6: the total, 8, from entry 16', (await svgText()).includes('total ones: 8'));

// ---------------------------------------------------------------- playback
await setBox('sound', false);
await press('1. Pass 1');
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
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

// A lab file on the same page that is not written yet answers 404; that is not this scene's.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
