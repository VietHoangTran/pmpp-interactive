// Browser check for the narrated animation "Ten rounds of Philox" in section X1.7: six
// parts, captions are an option (off by default, remembered), with no audio each part's
// script is shown regardless, the parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with src/sim/philox.ts (run here in Node).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-scene-philox-rounds.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const P = await import('../src/sim/philox.ts');
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
const S = 'section.scene[aria-label="Ten rounds of Philox"]';
const N = 6;
const LABELS = ['Counter and key', 'One round', 'Key bump', 'Spreading', 'Round ten', 'Every row at once'];

// ---------------------------------------------------------------- the model
const A = P.m5Input(0n, 0n, 0n);
const B = P.m5Input(0n, 1n, 0n);
const TA = P.philoxTrace(A.ctr, A.key);
const TB = P.philoxTrace(B.ctr, B.key);
const AV = P.avalanche(A.ctr, B.ctr, A.key);

const res = await fetch(BASE + '/content/x1/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x1/narration/philox-rounds-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const capBox = () => page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
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
const primary = async () => {
  await page.$eval(S + ' button.primary', (b) => b.click());
  await pause(150);
};
const primaryText = () => page.$eval(S + ' button.primary', (b) => b.textContent.trim());
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
const wordsOf = (tag) => page.$$eval(`${S} [data-word="${tag}"], ${S} [data-word^="${tag}-"]`, (e) => e.map((x) => x.dataset.value));
const diffsOf = (tag) => page.$$eval(S + ` [data-word^="${tag}-"]`, (e) => e.map((x) => x.dataset.diff === '1'));

await goto(page, 'x1/X1.7');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('philox-rounds-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()));
  await setBox('captions', true);
  check("switching captions on shows part 1's script", (await caption()) === scripts[0]);
} else {
  check("with no audio yet, part 1's script is shown regardless", (await hasCaption()) && (await caption()) === scripts[0], await caption());
  check('there is a captions option, off by default', !(await capBox()));
  await setBox('captions', true);
}
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(400);
check('the captions choice is remembered after a reload', await capBox());
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
await press('1. Counter and key');
await seek(1000);
check('part 1: row 0 counter is (step lo, step hi, stream lo, stream hi) = 0', JSON.stringify(await wordsOf('a')) === JSON.stringify(A.ctr.map(P.hex)));
check('part 1: row 1 differs only in the stream word', JSON.stringify(await diffsOf('b')) === JSON.stringify([false, false, true, false]) && JSON.stringify(await wordsOf('b')) === JSON.stringify(B.ctr.map(P.hex)));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. One round');
await seek(1000);
const r1 = TB[3]; // round 4 of row 1 (ROUND_SHOWN)
check('part 2: the two products split into halves as the model computes them',
  JSON.stringify([await wordsOf('hi0'), await wordsOf('lo0'), await wordsOf('hi1'), await wordsOf('lo1')].flat()) === JSON.stringify([r1.p0.hi, r1.p0.lo, r1.p1.hi, r1.p1.lo].map(P.hex)));
check("part 2: the words out are round 4's output for row 1", JSON.stringify(await wordsOf('out')) === JSON.stringify(r1.out.map(P.hex)), JSON.stringify(await wordsOf('out')));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Key bump');
await seek(1000);
check('part 3: the bumped key is (0x9e3779b9, 0xbb67ae85)', JSON.stringify([...(await wordsOf('key2'))]) === JSON.stringify(TA[1].key.map(P.hex)));
check('part 3: after round 1, one word differs, and it is word 1', JSON.stringify(await diffsOf('b')) === JSON.stringify(AV.masks[0]) && AV.differing[0] === 1 && AV.masks[0][1]);
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Spreading');
const seen = [];
for (const v of [0, 300, 520, 760, 1000]) {
  await seek(v);
  const r = await page.$eval(S + ' [data-round]', (e) => [Number(e.dataset.round), Number(e.dataset.differing)]);
  seen.push(r);
  check(`part 4 at ${v / 10}%: round ${r[0]}, the left and right words are the model's, ${r[1]} differ`,
    JSON.stringify(await wordsOf('a')) === JSON.stringify(TA[r[0] - 1].out.map(P.hex)) && JSON.stringify(await diffsOf('b')) === JSON.stringify(AV.masks[r[0] - 1]) && r[1] === AV.differing[r[0] - 1]);
}
check('part 4: rounds 2 to 5 shown in turn, ending with all four words different', seen[0][0] === 2 && seen[seen.length - 1][0] === 5 && seen[seen.length - 1][1] === 4, JSON.stringify(seen));
check('part 4: the bars read 1, 1, 2, 3, 4', (await page.$eval(S + ' [data-bars]', (e) => e.dataset.bars)) === '1,1,2,3,4');
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Round ten');
await seek(1000);
check("part 5: row 0 after round 10 is Random123's zero vector", JSON.stringify(await wordsOf('a')) === JSON.stringify(P.KATS[0].expected.map(P.hex)) && (await page.$eval(S + ' [data-kat-match]', (e) => e.dataset.katMatch)) === '1');
check("part 5: u is the model's 24-bit u", Number(await page.$eval(S + ' g[data-u]', (e) => e.dataset.u)) === P.m5U(0n, 0n, 0n));
let t = await svgText();
check('part 5: the top 24 bits are printed', t.includes((P.KATS[0].expected[0] >>> 8).toLocaleString('en-US')), t.slice(0, 200));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Every row');
await seek(1000);
const us = await page.$$eval(S + ' g[data-row]', (g) => g.map((x) => Number(x.dataset.u)));
check('part 6: eight rows, each u from its own stream', us.length === 8 && us.every((u, s) => u === P.m5U(0n, BigInt(s), 0n)), JSON.stringify(us));
t = await svgText();
check('part 6: CUDA, HIP, Metal and CPU are named', ['CUDA', 'HIP', 'Metal', 'CPU'].every((b) => t.includes(b)));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---------------------------------------------------------------- playing
await press('3. Key bump');
check('the caption follows the part', (await caption()) === scripts[2]);
await setBox('sound', false);
check('the button offers to start this part', ['Play', 'Resume'].includes(await primaryText()), await primaryText());
await primary();
await pause(1500);
const s1 = await slider();
check('it plays with the sound off: the position advances', s1 > 10, String(s1));
check('while playing the button says Pause', (await primaryText()) === 'Pause');
await primary();
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2);
await seek(990);
await primary();
await pause(1600);
check('at the end of a part the next one starts by itself', (await current()) === 4, String(await current()));
await primary();

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
