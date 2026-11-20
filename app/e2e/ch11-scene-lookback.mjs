// Browser check for the narrated animation "Waiting for the block before you" in section
// 11.9: six parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the model
// (src/sim/lookback.ts, imported here directly).
// It passes with or without content/ch11/narration/manifest.json: with no audio the player
// runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-scene-lookback.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { flagsAt, phaseAt, sceneRuns, simulate } from '../src/sim/lookback.ts';

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
const S = 'section.scene[aria-label="Waiting for the block before you"]';
const N = 6;
const LABELS = ['Six blocks, three slots', 'One flag', 'The chain', 'Without the counter', 'Decoupled', 'Who promises progress'];
const R = sceneRuns();
const WIDE = simulate({ G: 32, R: 32, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'started' });

const res = await fetch(BASE + '/content/ch11/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch11/narration/lookback-${n}.txt`)).text()));

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
const rowsNow = () => page.$$eval(S + ' [data-row]', (g) => g.map((x) => ({ hw: Number(x.dataset.row), phase: x.dataset.phase })));
const flagsNow = () => page.$$eval(S + ' [data-flag]', (g) => g.map((x) => x.dataset.state));
const links = () => page.$$eval(S + ' [data-link]', (g) => g.map((x) => x.dataset.link));

await goto(page, 'ch11/11.9');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('lookback-'));
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
const single = R.single;
const firstPublish = single.events.find((e) => e.kind === 'inclusive' && e.idx === 0).t;
const b1Read = single.blocks.find((b) => b.idx === 1).spans.find((s) => s.phase === 'look').from;

await press('1. Six blocks');
await seek(0);
let rows = await rowsNow();
check('part 1, start: six rows, the first three scanning, the rest queued',
  rows.length === 6 && rows.slice(0, 3).every((r) => r.phase === 'local') && rows.slice(3).every((r) => r.phase === 'queued'), JSON.stringify(rows));
check('part 1: three of three slots resident', (await page.$eval(S + ' [data-resident]', (e) => e.dataset.resident)) === '3');
check('part 1: queued blocks show no claimed index yet', (await svgText()).includes('blockIdx 3 (queued)'));
await seek(1000);
check('part 1, end: no flag is set yet', (await flagsNow()).every((f) => f === 'none'));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. One flag');
await seek(1000);
check('part 2, end: flags as the model at its last round', JSON.stringify(await flagsNow()) === JSON.stringify(flagsAt(single, b1Read + 2)), JSON.stringify(await flagsNow()));
check('part 2: the message from index 0 to index 1 is drawn', !!(await page.$(S + ' [data-message="0-1"]')));
let t = await svgText();
check('part 2: store then release, acquire then read', t.includes('store sum, then release the flag') && t.includes('acquire the flag, then read the sum'));
check(`part 2: index 1 spins ${single.blocks.find((b) => b.idx === 1).wait} round(s)`, t.includes(`index 1 spins ${single.blocks.find((b) => b.idx === 1).wait} round`));
await seek(Math.round(1000 * ((firstPublish - 1 - (firstPublish - 2)) / (b1Read + 2 - (firstPublish - 2))) * 0.8 + 50));
const hw1 = single.blocks.find((b) => b.idx === 1).hw;
check('part 2, before block 0 publishes: index 1 is spinning', (await rowsNow()).find((r) => r.hw === hw1)?.phase === 'spin', JSON.stringify(await rowsNow()));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. The chain');
await seek(1000);
t = await svgText();
const l3 = await links();
check('part 3: five links, each index waiting on the one before', l3.length === 5 && l3.every((l) => { const [a, b] = l.split('-').map(Number); return b === a + 1; }), JSON.stringify(l3));
check('part 3: headline and the 32-block chain from the model', t.includes(`a chain of ${single.criticalPath} links for 6 blocks`) && t.includes(`with 32 blocks all resident: ${WIDE.criticalPath} links, finished after ${WIDE.time} rounds`), t.slice(-300));
check('part 3: every block done', (await rowsNow()).every((r) => r.phase === 'done'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Without the counter');
await seek(1000);
rows = await rowsNow();
const D = R.deadlock;
check('part 4: blocks 0, 1, 2 never start; 3, 4, 5 spin', rows.filter((r) => r.hw < 3).every((r) => r.phase === 'queued') && rows.filter((r) => r.hw >= 3).every((r) => r.phase === phaseAt(D.blocks[r.hw], D.time) && r.phase === 'spin'), JSON.stringify(rows));
check('part 4: the deadlock banner', !!(await page.$(S + ' [data-banner="fail"]')) && (await svgText()).includes('deadlock'));
check('part 4: who waits on whom, from the model', (await svgText()).includes('3 waits on 2 (not started) · 4 waits on 3 · 5 waits on 4'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Decoupled');
// The part maps p from 0.03 to 0.75 onto rounds 0 to the run's end; 0.645 lands just after round 17,
// when indices 3 to 5 have published their own totals and not yet their prefixes.
await seek(645);
const t5 = Math.floor(((0.645 - 0.03) / 0.72) * R.decoupled.time);
const f5 = await flagsNow();
check(`part 5, round ${t5}: flags as the model, some showing only an own total`, JSON.stringify(f5) === JSON.stringify(flagsAt(R.decoupled, t5)) && f5.includes('own'), JSON.stringify(f5));
await seek(1000);
const Dc = R.decoupled;
t = await svgText();
check('part 5: waits, extra additions and chains from the model',
  t.includes(`spinning: ${single.waitTotal} rounds single, ${Dc.waitTotal} decoupled; ${Dc.repeated} extra additions`) && t.includes(`chain ${single.criticalPath} links single, ${Dc.criticalPath} decoupled; finished after ${single.time} rounds single, ${Dc.time} decoupled`), t.slice(-300));
const l5 = await links();
check('part 5: some blocks skip back past their neighbour', l5.some((l) => { const [a, b] = l.split('-').map(Number); return b > a + 1; }), JSON.stringify(l5));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Who promises');
await seek(1000);
const M = R.metal;
rows = await rowsNow();
check('part 6: the block waited for is resident but not run', rows.some((r) => r.phase === 'starved') && rows.filter((r) => r.phase === 'starved').every((r) => phaseAt(M.blocks[r.hw], M.time) === 'starved'), JSON.stringify(rows));
t = await svgText();
check('part 6: banner and the three backends', !!(await page.$(S + ' [data-banner="fail"]')) && t.includes('CUDA: claimed index is safe') && t.includes('HIP: unverified') && t.includes('Metal: unsafe'));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---------------------------------------------------------------- playing
await press('1. Six blocks');
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

// Light theme and phone width, for the screenshots.
if (out) {
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
  await press('4. Without the counter');
  await seek(1000);
  await page.screenshot({ path: out + '-light-part4.png' });
  await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
  await pause(300);
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/api\/lab-file\?path=ch11%2Flab5_/.test(u) || /\/content\/ch11\/narration\/(manifest\.json|lookback-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (missing narration audio and a lab file not yet written are allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
