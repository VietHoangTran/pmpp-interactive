// Browser check for the narrated animation "Why one busy bin sets the pace" in section
// 9.3: six parts, captions are an option (off by default, remembered), the parts advance
// by themselves, it runs with the sound off, and every part's bars and finish lines
// agree with the model (src/sim/atomicrate.ts, imported here directly).
// It passes with or without content/ch09/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-scene-atomic-wait.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { SCENE, SKEWED4, UNIFORM4, sceneBins, sceneLoads, sceneLockstep, sceneOneBin, scenePrivatised } from '../src/sim/atomicrate.ts';

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
const S = 'section.scene[aria-label="Why one busy bin sets the pace"]';
const N = 6;
const LABELS = ['Loads overlap', 'One address', 'Without the atomic', 'Many bins', 'Shorter links', 'Next: privatise'];

const res = await fetch(BASE + '/content/ch09/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch09/narration/atomic-wait-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('six scripts of 35 to 60 words', words.every((w) => w >= 35 && w <= 60), JSON.stringify(words));

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
/** Rows drawn: [start, end] per bar, and the finish line. */
const drawn = () =>
  page.$eval(S + ' .stage svg', (svg) => ({
    rows: [...svg.querySelectorAll('[data-row]')].map((g) => [Number(g.dataset.start), Number(g.dataset.end)]),
    finish: svg.querySelector('[data-finish]') ? Number(svg.querySelector('[data-finish]').dataset.finish) : null,
    finishShown: svg.querySelector('[data-finish]') ? Number(svg.querySelector('[data-finish]').getAttribute('opacity')) : 0,
    ghosts: [...svg.querySelectorAll('[data-ghost]')].map((g) => Number(g.dataset.ghost)),
    value: svg.querySelector('[data-bin-value]') ? Number(svg.querySelector('[data-bin-value]').dataset.binValue) : null,
    lost: svg.querySelector('[data-bin-value]') ? Number(svg.querySelector('[data-bin-value]').dataset.lost) : null,
  }));
const same = (tl) => (d) => JSON.stringify(d.rows) === JSON.stringify(tl.bars.map((b) => [b.start, b.end])) && d.finish === tl.end;

await goto(page, 'ch09/9.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('atomic-wait-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()));
  await setBox('captions', true);
  check('switching captions on shows part 1\'s script', (await caption()) === scripts[0]);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector(S);
  await pause(400);
  check('the captions choice is remembered after a reload', await hasCaption());
} else {
  check('with no audio yet, part 1\'s script is shown regardless', (await hasCaption()) && (await caption()) === scripts[0], await caption());
  check('there is a captions option, off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
  await setBox('captions', true);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector(S);
  await pause(400);
  check('the captions choice is remembered after a reload', await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked));
}
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
await press('1. Loads overlap');
await seek(1000);
let d = await drawn();
check('part 1: eight overlapping loads, done at one latency plus seven issue slots', same(sceneLoads())(d) && d.finish === SCENE.dram + 7 * SCENE.issue && d.finishShown === 1, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. One address');
await seek(450);
d = await drawn();
check('part 2, halfway: the bin has counted some but not all of the 8 atomics', d.value > 0 && d.value < 8 && d.finishShown === 0, JSON.stringify(d.value));
await seek(1000);
d = await drawn();
check('part 2: eight atomics in a row, done at 8 × 400 = 3200 cycles, the bin at 8', same(sceneOneBin())(d) && d.finish === 3200 && d.value === 8, JSON.stringify(d));
check('part 2: the loads\' finish is kept as a ghost line', d.ghosts.includes(sceneLoads().end));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Without the atomic');
await seek(1000);
d = await drawn();
const lock = sceneLockstep();
check('part 3: plain lockstep, done early, the bin at 1 and 7 updates lost', same(lock)(d) && d.value === 1 && d.lost === 7 && d.finish < 600, JSON.stringify(d));
check('part 3: every lane reads 0 and writes 1', (await svgText()).includes('0→1') && (await svgText()).includes('7 updates lost'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Many bins');
await seek(400);
d = await drawn();
check('part 4, first half: uniform over 4 bins, chains of 2', same(sceneBins(UNIFORM4))(d) && (await svgText()).includes('uniform: 8 updates spread over 4 bins'), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4, second half: skewed, the chain of 5 finishes at 2000', same(sceneBins(SKEWED4))(d) && d.finish === 2000 && d.ghosts.includes(sceneBins(UNIFORM4).end), JSON.stringify(d));
check('part 4: rows name the bin each thread hits', (await svgText()).includes('thread 4 → bin 0') && (await svgText()).includes('thread 7 → bin 3'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Shorter links');
await seek(1000);
d = await drawn();
check('part 5: the same skewed chains in the cache, ten times shorter, DRAM as a ghost', same(sceneBins(SKEWED4, SCENE.cache))(d) && d.finish === 200 && d.ghosts.includes(2000), JSON.stringify(d));
check('part 5: the cache figure is labelled illustrative', (await svgText()).includes('illustrative'));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Next: privatise');
await seek(1000);
d = await drawn();
const priv = scenePrivatised();
check('part 6: private chains, then two commits, done long before 3200', same(priv)(d) && d.value === 8 && d.ghosts.includes(3200) && d.finish < 1000, JSON.stringify(d));
check('part 6: private copies drawn in the shared-memory colour', await page.$$eval(S + ' .stage svg rect', (rs) => rs.some((r) => r.getAttribute('fill') === 'var(--shared)')));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---------------------------------------------------------------- playing
await press('1. Loads overlap');
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

// ---------------------------------------------------------------- narrow and dark
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways scrolling at 400 px', wide <= 1, String(wide));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-dark.png' });
}

const expected404 = (u) => /\/content\/ch09\/narration\/(manifest\.json|atomic-wait-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
