// Browser check for the narrated animation "Blocks finish in any order" in section 10.9:
// five parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the model
// (src/sim/multiblock.ts, imported here directly).
// It passes with or without content/ch10/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-scene-commit.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { f32, f32hex, f32short, halvingLevels, sceneData, segments } from '../src/sim/multiblock.ts';

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
const S = 'section.scene[aria-label="Blocks finish in any order"]';
const N = 5;
const LABELS = ['Segments', 'Commit', 'Run it again', 'Second pass', 'The cost'];
const D = sceneData();

const res = await fetch(BASE + '/content/ch10/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch10/narration/commit-order-${n}.txt`)).text()));

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

await goto(page, 'ch10/10.9');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('five parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('commit-order-'));
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
await press('1. Segments');
await seek(1000);
let t = await svgText();
check('part 1: 8 segments of 8, the last with 4 real and 4 past the end',
  t.includes('block 7') && t.includes('starts at 56') && t.includes(`${D.seg.lastReal} real,`) && t.includes(`${D.seg.pastEnd} past the end`) && t.includes(`ceil(${D.seg.N} / ${D.seg.perBlock}) = ${D.seg.blocks} blocks`), t.slice(0, 300));
check('part 1: 64 slots drawn, 4 of them hatched', (await page.$$eval(S + ' .stage svg rect[fill="url(#co-hatch)"]', (r) => r.length)) === 4);
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Commit');
await seek(500);
const midCommits = await page.$$eval(S + ' [data-commit]', (e) => e.length);
check('part 2, halfway: some commits but not all', midCommits > 0 && midCommits < 8, String(midCommits));
await seek(1000);
const commits = await page.$$eval(S + ' [data-commit]', (e) => e.map((x) => x.textContent));
check('part 2: the 8 commits follow run 1\'s seeded order', commits.length === 8 && commits.every((c, k) => c.startsWith(`commit ${k + 1}: block ${D.runA.order[k]} adds ${f32short(D.sums[D.runA.order[k]])}, total ${f32short(D.runA.running[k])}`)), JSON.stringify(commits.slice(0, 2)));
check('part 2: the total box ends on run 1\'s total', (await page.$eval(S + ' [data-total]', (e) => e.dataset.total)) === f32hex(D.runA.total));
check('part 2 says the order is a seeded shuffle, not a timing model', (await svgText()).includes('not a timing model'));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Run it again');
await seek(300);
const runB = await page.$$eval(S + ' [data-commit]', (e) => e.map((x) => x.textContent));
check('part 3: run 2 commits in its own order', runB.length > 0 && runB.every((c, k) => c.startsWith(`commit ${k + 1}: block ${D.runB.order[k]} `)), JSON.stringify(runB.slice(0, 2)));
await seek(1000);
t = await svgText();
check('part 3: both totals in hex, different, beside the float64 reference',
  t.includes(f32hex(D.runA.total)) && t.includes(f32hex(D.runB.total)) && f32hex(D.runA.total) !== f32hex(D.runB.total) && t.includes(`nearest float32 ${f32hex(f32(D.ref))}`), t.slice(0, 400));
const bold = await page.$$eval(S + ' [data-compare] tspan[font-weight="700"]', (e) => e.map((x) => x.textContent));
check('part 3: only the differing hex digits are marked', bold.length === 2 && bold[0] !== bold[1], JSON.stringify(bold));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Second pass');
await seek(1000);
const levels = await page.$$eval(S + ' [data-level]', (gs) => gs.map((g) => [...g.querySelectorAll(':scope > g text')].map((x) => x.textContent)));
const want = halvingLevels(D.sums).map((l) => l.map(f32short));
check('part 4: every level of the second pass is the model\'s halving tree', JSON.stringify(levels) === JSON.stringify(want), JSON.stringify(levels));
check('part 4: the result is the two-pass total, the same for both runs', (await page.$eval(S + ' [data-final]', (e) => e.dataset.final)) === f32hex(D.twoPass.total) && (await svgText()).includes('same every run'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. The cost');
await seek(1000);
check('part 5: one float per block, 489 for a million logits at 1024 threads', Number(await page.$eval(S + ' [data-million]', (e) => e.dataset.million)) === segments(1_000_000, 1024).blocks && (await svgText()).includes('1,956 bytes'));
check('part 5 names llama.cpp\'s two-dispatch Metal argmax', (await svgText()).includes('two dispatches'));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---------------------------------------------------------------- playing
await press('1. Segments');
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

const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|commit-order-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
