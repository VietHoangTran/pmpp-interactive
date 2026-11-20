// Browser check for the narrated animation "One row through the chain" in section X1.1:
// six parts, captions are an option (off by default, remembered), with no audio each
// part's script is shown regardless, the parts advance by themselves, it runs with the
// sound off, and every part's drawing agrees with src/sim/chain.ts (sceneChain(), run in
// the page through the app's module server).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-scene-chain-row.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const S = 'section.scene[aria-label="One row through the chain"]';
const N = 6;
const LABELS = ['Penalties', 'Top-k', 'Sort, softmax', 'Top-p', 'Min-p', 'Draw'];

const res = await fetch(BASE + '/content/x1/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x1/narration/chain-row-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));

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
/** The play / pause / resume button. */
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

await goto(page, 'x1/X1.1');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const D = await page.evaluate(async () => {
  const m = await import('/src/sim/chain.ts');
  const r = m.sceneChain();
  const snap = (st) => r.snapshots.find((s) => s.stage === st);
  const pen = snap('penalties').logits;
  const ids = (a) => a.flatMap((x, i) => (x ? [i] : []));
  const counts = m.SCENE_ROW.logits.map((_, i) => m.SCENE_ROW.context.filter((t) => t === i).length);
  const topkSorted = m.rankDesc(pen, snap('top-k').alive);
  const tkp = snap('top-k').probs;
  const before = topkSorted.map((_, j) => topkSorted.slice(0, j).reduce((s, i) => s + tkp[i], 0));
  return {
    pen: pen.map((v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2))),
    counts,
    topk: ids(snap('top-k').alive),
    topkSorted,
    topkPc: topkSorted.map((i) => `${Math.round(tkp[i] * 100)}%`),
    before: before.map((b) => b.toFixed(2)),
    topp: ids(snap('top-p').alive),
    minp: ids(snap('min-p').alive),
    th: snap('min-p').threshold.toFixed(4),
    u: r.draw.u.toFixed(4),
    token: r.token,
    order: r.draw.order,
    finalPc: r.draw.probs.map((p) => `${Math.round(p * 100)}%`),
    settings: m.SCENE_SETTINGS,
  };
});

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('chain-row-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()));
  await setBox('captions', true);
  check("switching captions on shows part 1's script", (await caption()) === scripts[0]);
} else {
  check("with no audio yet, part 1's script is shown regardless", (await hasCaption()) && (await caption()) === scripts[0], await caption());
  check('there is a captions option, off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
  await setBox('captions', true);
}
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(400);
check('the captions choice is remembered after a reload', await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
await press('1. Penalties');
await seek(1000);
const vals = await page.$$eval(S + ' [data-token] [data-value]', (e) => e.map((x) => x.dataset.value));
check('part 1: the twelve bars end on the penalised logits', JSON.stringify(vals) === JSON.stringify(D.pen), JSON.stringify(vals));
const counts = await page.$$eval(S + ' [data-count]', (e) => e.map((x) => [Number(x.closest('[data-token]').dataset.token), Number(x.dataset.count)]));
check('part 1: "used N" under every token of the context', JSON.stringify(counts) === JSON.stringify(D.counts.flatMap((c, i) => (c ? [[i, c]] : []))), JSON.stringify(counts));
let t = await svgText();
check('part 1: token 7 was first and is now second', t.includes('token 7 was first; now token 2 is first and token 7 is second'), t.slice(0, 200));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Top-k');
await seek(1000);
const op = await page.$$eval(S + ' g[data-token]', (gs) => gs.map((g) => [Number(g.dataset.token), Number(g.getAttribute('opacity'))]));
check(`part 2: the ${D.topk.length} survivors stay lit, the rest fade where they stand`,
  op.length === 12 && op.every(([i, o]) => (D.topk.includes(i) ? o === 1 : o < 0.5)), JSON.stringify(op));
t = await svgText();
check('part 2: tokens 3 and 10 tie at the cut and the lower id stays', t.includes('tokens 3 and 10 tie at the cut (0.70): the lower id, 3, stays'), t.slice(0, 220));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Sort, softmax');
await seek(500);
check('part 3, halfway: the bars are moving into descending order', (await page.$$eval(S + ' g[data-token]', (g) => g.length)) === D.topk.length);
await seek(1000);
const order3 = await page.$$eval(S + ' g[data-token]', (gs) => gs.map((g) => Number(g.dataset.token)));
t = await svgText();
check('part 3: five probability bars in descending order, as the model ranks them', JSON.stringify(order3) === JSON.stringify(D.topkSorted), JSON.stringify(order3));
check("part 3: each bar shows its probability among the five", D.topkPc.every((p) => t.includes(p)) && t.includes('add up to 1.00'), t.slice(0, 200));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Top-p');
await seek(1000);
const bef = await page.$$eval(S + ' [data-before]', (e) => e.map((x) => x.dataset.before));
check('part 4: the mass before each token, as the model computes it', JSON.stringify(bef) === JSON.stringify(D.before), JSON.stringify(bef));
check(`part 4: p = ${D.settings.p} is marked on the strip`, (await page.$eval(S + ' [data-pline]', (e) => Number(e.dataset.pline))) === D.settings.p);
const dark4 = await page.$$eval(S + ' g[data-token]', (gs) => gs.filter((g) => Number(g.getAttribute('opacity')) < 1).map((g) => Number(g.dataset.token)));
check('part 4: only the token top-p drops goes dark', JSON.stringify(dark4) === JSON.stringify(D.topk.filter((i) => !D.topp.includes(i))), JSON.stringify(dark4));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Min-p');
await seek(300);
const midLine = await page.$eval(S + ' [data-threshold] text', (e) => Number(e.textContent));
await seek(1000);
check('part 5: the threshold line comes down to max + ln 0.1', (await page.$eval(S + ' [data-threshold]', (e) => e.dataset.threshold)) === D.th && midLine > Number(D.th), `${midLine} ${D.th}`);
t = await svgText();
check(`part 5: token 9 falls and ${D.minp.length} are left`, t.includes('token 9 (0.80) falls') && t.includes(`${D.minp.length} left`), t.slice(0, 260));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Draw');
await seek(200);
check('part 6, early: no u yet', (await page.$(S + ' [data-u]')) === null);
await seek(1000);
check("part 6: u is the model's Philox u", (await page.$eval(S + ' [data-u]', (e) => e.dataset.u)) === D.u);
const segs = await page.$$eval(S + ' [data-segment]', (e) => e.map((x) => Number(x.dataset.segment)));
check('part 6: the survivors end to end in draw order', JSON.stringify(segs) === JSON.stringify(D.order), JSON.stringify(segs));
t = await svgText();
check(`part 6: u lands in token ${D.token}, which is written out`, t.includes(`lands in token ${D.token}: token ${D.token} is written out`) && D.finalPc.every((p) => t.includes(p)), t.slice(0, 200));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---------------------------------------------------------------- playing
await press('3. Sort, softmax');
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
