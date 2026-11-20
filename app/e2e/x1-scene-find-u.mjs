// Browser check for the narrated animation "Finding u in the scan" in section X1.8: six
// parts, captions are an option (off by default, remembered), with no audio each part's
// script is shown regardless, the parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with src/sim/draw.ts (run here in Node).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-scene-find-u.mjs [--dark] [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const D = await import('../src/sim/draw.ts');

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
const S = 'section.scene[aria-label="Finding u in the scan"]';
const N = 6;
const LABELS = ['The target', 'Chunks', 'Scan and flag', 'The vote', 'Near a boundary', 'The two ends'];

// ---------------------------------------------------------------- the model
const K = D.sceneDraw();
const crossing = K.chunks.find((c) => c.crosses);
const BND = D.sceneBoundary();
const FB = D.fallbackDraw();
const gt = D.inverseCdf(D.GE_GT.weights, D.GE_GT.u, { rule: 'gt' }).index;
const ge = D.inverseCdf(D.GE_GT.weights, D.GE_GT.u, { rule: 'ge' }).index;

const res = await fetch(BASE + '/content/x1/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x1/narration/find-u-${n}.txt`)).text()));
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
const short = (v) => {
  const x = Math.fround(v);
  for (let p = 1; p <= 9; p++) if (Math.fround(Number(x.toPrecision(p))) === x) return Number(x.toPrecision(p)).toString();
  return x.toPrecision(9);
};

await goto(page, 'x1/X1.8');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('find-u-'));
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
await press('1. The target');
await seek(1000);
check('part 1: forty survivors in one bar', (await page.$$eval(S + ' [data-survivor]', (e) => e.length)) === D.SCENE_WEIGHTS.length);
check("part 1: the target is the kernel's u x total", Number(await page.$eval(S + ' [data-target]', (e) => e.dataset.target)) === K.target);
check(`part 1: the token is survivor ${K.index}, marked on the bar`,
  Number(await page.$eval(S + ' [data-token]', (e) => e.dataset.token)) === K.index &&
  JSON.stringify(await page.$$eval(S + ' [data-state="win"]', (e) => e.map((x) => Number(x.dataset.survivor)))) === JSON.stringify([K.index]));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Chunks');
await seek(1000);
const ch = await page.$$eval(S + ' [data-chunk]', (e) => e.map((x) => [Number(x.dataset.chunk), x.dataset.crosses, Number(x.dataset.agg)]));
check('part 2: chunk 0 skipped, chunk 1 crosses, aggregates as the model', JSON.stringify(ch) === JSON.stringify(K.chunks.map((c) => [c.c, c.crosses ? '1' : '0', c.aggAfter])), JSON.stringify(ch));
const skipped = await page.$$eval(S + ' [data-state="skip"]', (e) => e.map((x) => Number(x.dataset.survivor)));
check('part 2: the skipped chunk is greyed, survivors 0 to 15', JSON.stringify(skipped) === JSON.stringify(Array.from({ length: 16 }, (_, i) => i)), JSON.stringify(skipped));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Scan and flag');
await seek(1000);
const cells = await page.$$eval(S + ' [data-index]', (e) => e.filter((x) => x.dataset.cum).map((x) => [Number(x.dataset.index), Number(x.dataset.cum), x.dataset.flag]));
const want = crossing.lanes.flatMap((l) => l.cum.map((c, j) => [l.first + j, c, l.flags[j] ? '1' : '0']));
check("part 3: each lane's sums and flags in chunk 1 are the model's", JSON.stringify(cells) === JSON.stringify(want), JSON.stringify(cells.slice(0, 4)));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The vote');
await seek(1000);
const ballots = await page.$$eval(S + ' [data-ballot]', (e) => e.map((x) => [Number(x.dataset.bits), Number(x.dataset.first)]));
check('part 4: one ballot per warp, with the lowest flagged lane', JSON.stringify(ballots) === JSON.stringify(crossing.ballots.map((b, k) => [b, crossing.warpFirst[k]])), JSON.stringify(ballots));
check(`part 4: survivor ${K.index} is written (warp ${K.winner.warp}, lane ${K.winner.lane})`, Number(await page.$eval(S + ' [data-winner]', (e) => e.dataset.winner)) === K.index);
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Near a boundary');
await seek(1000);
const ord = await page.$$eval(S + ' [data-order]', (e) => e.map((x) => [x.dataset.order, Number(x.dataset.cum), Number(x.dataset.index)]));
check('part 5: the two orders\' sums differ and pick neighbouring tokens, as the model finds',
  JSON.stringify(ord) === JSON.stringify([['loop', BND.loop.cum, BND.loop.index], ['kernel', BND.kernel.cum, BND.kernel.index]]) && BND.loop.cum !== BND.kernel.cum, JSON.stringify(ord));
let t = await svgText();
check('part 5: u is printed', t.includes(short(BND.u)), t.slice(0, 120));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. The two ends');
await seek(1000);
const z = await page.$eval(S + ' [data-end="zero"]', (e) => [Number(e.dataset.gt), Number(e.dataset.ge)]);
check('part 6: at u = 0, > picks survivor 1 and >= picks the zero weight 0', z[0] === gt && z[1] === ge && gt === 1 && ge === 0, JSON.stringify(z));
const f = await page.$eval(S + ' [data-end="fallback"]', (e) => [Number(e.dataset.index), e.dataset.fallback]);
check(`part 6: the fallback row falls back to survivor ${FB.index}`, f[0] === FB.index && f[1] === '1' && FB.fallback, JSON.stringify(f));
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---------------------------------------------------------------- playing
await press('3. Scan and flag');
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
