// Browser check for the narrated animation "Narrowing in on the k-th key" in section X1.4:
// six parts, captions are an option (off by default, remembered), with no audio each part's
// script is shown regardless, the parts advance by themselves, it runs with the sound off,
// and every part's drawing agrees with src/sim/topkselect.ts (selectTrace on SCENE_ROW).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-scene-radix-select.mjs [--dark] [screenshot-prefix]
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
const M = await import('../src/sim/topkselect.ts');

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
const S = 'section.scene[aria-label="Narrowing in on the k-th key"]';
const N = 6;
const LABELS = ['Keys', 'Top-byte histogram', 'Walk the bins', 'Only the prefix works', 'The k-th key', 'The gather'];

// The model, as the scene runs it: 64 logits, k = 12, one block of 16 threads, one 32-lane warp.
const tr = M.selectTrace(M.SCENE_ROW, M.SCENE_K, 16, 32);
const P = tr.passes;
const binHex = (b) => b.toString(16).toUpperCase().padStart(2, '0');
const nz = (h) => h.map((c, b) => [b, c]).filter(([, c]) => c > 0).reverse();
const statesAfter = (n) => Array.from({ length: tr.n }, (_, c) => M.statusAfter(tr, n, c));

const res = await fetch(BASE + '/content/x1/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x1/narration/radix-select-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('the scripts name the numbers the model gives', scripts[2].includes(`holds ${P[0].walk.steps[0].count} keys`) && scripts[2].includes(`next holds ${P[0].inBin}`)
  && scripts[2].includes(`need ${P[0].needOut} of its ${P[0].inBin}`) && scripts[3].includes(`Only ${P[1].candidates} lanes`)
  && scripts[4].includes(`need only ${tr.need} of them`) && /Ten keys lie strictly above/.test(scripts[4]) && tr.greater === 10
  && /Three keys are equal/.test(scripts[4]) && tr.equal === 3, JSON.stringify(scripts.slice(2, 5)));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const captionsBox = () => page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
const setBox = async (word, on) => {
  await page.evaluate((S, word, on) => {
    const b = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (b.checked !== on) b.click();
  }, S, word, on);
  await pause(120);
};
const part = async (n) => {
  await page.evaluate((S, n) => [...document.querySelectorAll(S + ' .seg button')][n - 1].click(), S, n);
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
const headline = () => page.$eval(S + ' [data-headline]', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const cells = () => page.$$eval(S + ' svg g[data-col]', (gs) => gs.map((g) => ({ col: Number(g.dataset.col), state: g.dataset.state, slot: g.dataset.slot, text: g.textContent.replace(/\s+/g, ' ').trim() })));
const bins = () => page.$$eval(S + ' svg g[data-bin]', (gs) => gs.map((g) => [Number(g.dataset.bin), Number(g.dataset.count)]));
const walk = () => page.$$eval(S + ' svg [data-walk]', (es) => es.map((e) => e.dataset.walk));
const lines = () => page.$$eval(S + ' svg [data-line]', (es) => es.map((e) => e.textContent.replace(/\s+/g, ' ').trim()));
const shot = async (name) => {
  if (out) await (await page.$(S)).screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'x1/X1.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);

// ---------------------------------------------------------------- the player
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
check('six parts with the planned labels', JSON.stringify(await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('radix-select-'));
if (anyAudio) {
  check('captions are off by default', !(await hasCaption()) && !(await captionsBox()));
  await setBox('captions', true);
  check("switching captions on shows part 1's script", (await caption()) === scripts[0]);
} else {
  check("with no audio yet, part 1's script is shown regardless", (await hasCaption()) && (await caption()) === scripts[0], await caption().catch(() => ''));
  check('there is a captions option, off by default', !(await captionsBox()));
  await setBox('captions', true);
}
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(400);
check('the captions choice is remembered after a reload', await captionsBox());
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---------------------------------------------------------------- each part against the model
// Part 1: logits, then keys with the top byte in bold.
await part(1);
await seek(0);
let c = await cells();
check('part 1, start: 64 cells showing logits', c.length === 64 && (await headline()) === `64 logits; we want the ${M.SCENE_K} largest`);
await seek(1000);
c = await cells();
const keyText = (col) => M.hex(tr.keys[col]).slice(2).match(/../g).join(' ');
check('part 1, end: every cell shows its 32-bit key from the model', c.every((x) => x.text === keyText(x.col)), JSON.stringify(c.slice(0, 3)));
check('part 1, end: the histogram is empty and nothing is decided', (await bins()).length === 0 && c.every((x) => x.state === 'open'));
await shot('part1');

// Part 2: pass 1, round by round, then the full top-byte histogram.
await part(2);
await seek(50);
c = await cells();
const r0 = P[0].rounds[0];
check('part 2, round 1: thread t reads column t and all 16 lanes do the atomic', c.filter((x) => x.col < 16).every((x) => x.state === (r0.lanes[x.col].match ? 'active' : 'idle')) && r0.lanes.every((l) => l.match)
  && (await headline()).startsWith('round 1: thread t reads column t + 0; 16 of 16 lanes'));
await seek(500);
const r2 = P[0].rounds[1];
check('part 2, round 3: the histogram so far is the model\'s after round 2', JSON.stringify(await bins()) === JSON.stringify(nz(r2.histAfter)), JSON.stringify(await bins()));
await seek(1000);
check('part 2, end: the full histogram of byte 1', JSON.stringify(await bins()) === JSON.stringify(nz(P[0].hist)));
check('part 2, end: one read of the row counted all 64 keys', (await headline()) === `pass 1 done: ${P[0].candidates} keys counted in one read of the row` && P[0].candidates === 64);
await shot('part2');

// Part 3: the walk from the top bin.
await part(3);
await seek(1000);
const wantWalk = P[0].walk.steps.map((s) => `${s.bin}:${s.count}:${s.running}`);
check('part 3: the walk visits C1 then C0, running 6 then 19', JSON.stringify(await walk()) === JSON.stringify(wantWalk) && wantWalk.join() === '193:6:6,192:13:19', JSON.stringify(await walk()));
c = await cells();
const s1 = statesAfter(1);
check('part 3: cells decided in, out and open as after pass 1 (6, 45, 13)', c.every((x) => x.state === s1[x.col]) && c.filter((x) => x.state === 'in').length === 6 && c.filter((x) => x.state === 'out').length === 45);
check('part 3: the headline gives the split and what is still needed', (await headline()) === `${P[0].above} in, ${P[0].below} out, ${P[0].inBin} undecided in bin ${binHex(P[0].bin)}: ${P[0].needOut} still needed`, await headline());
await shot('part3');

// Part 4: pass 2, only lanes matching the prefix work.
await part(4);
await seek(300); // counting(1, 0.356): round 2
c = await cells();
const rd = P[1].rounds[1];
check('part 4, round 2: lanes matching prefix C0 are active, the rest idle', c.filter((x) => x.col >= 16 && x.col < 32).every((x) => x.state === (rd.lanes[x.col - 16].match ? 'active' : 'idle'))
  && rd.lanes.some((l) => l.match) && rd.lanes.some((l) => !l.match), JSON.stringify(c.slice(16, 32).map((x) => x.state)));
check('part 4, round 2: the headline says grey lanes skip the branch', (await headline()).endsWith('grey lanes skip the branch'), await headline());
await seek(780);
check('part 4, before the walk: only 13 keys match the prefix', (await headline()) === `pass 2: only ${P[1].candidates} keys match prefix C0; every round split the warp` && P[1].rounds.every((r) => r.warps[0].diverged));
await seek(1000);
c = await cells();
const s2 = statesAfter(2);
check('part 4, end: pass 2\'s histogram and the states after it (8 in, 8 open)', JSON.stringify(await bins()) === JSON.stringify(nz(P[1].hist)) && c.every((x) => x.state === s2[x.col]) && c.filter((x) => x.state === 'in').length === 8);
await shot('part4');

// Part 5: passes 3 and 4, then K*.
await part(5);
await seek(200);
check('part 5, first half: pass 3 leaves 7 keys in bin 00', (await headline()) === `pass 3: ${P[2].inBin} keys left in bin ${binHex(P[2].bin)}, ${P[2].needOut} still needed` && P[2].inBin === 7);
await seek(1000);
check('part 5, end: K* with 10 keys above and 2 of 3 ties needed', (await headline()) === `K* = ${M.hex(tr.prefix)}: ${tr.greater} keys above it, ${tr.need} of its ${tr.equal} ties needed` && tr.exact, await headline());
check('part 5, end: one line per pass, from the model', JSON.stringify(await lines()) === JSON.stringify(P.map((ps) => `pass ${ps.index + 1}: bin ${binHex(ps.bin)}, ${ps.above} more in, ${ps.needOut} still needed`)), JSON.stringify(await lines()));
c = await cells();
const s4 = statesAfter(4);
check('part 5, end: 10 in, 3 undecided (the ties at K*)', c.every((x) => x.state === s4[x.col]) && c.filter((x) => x.state === 'open').map((x) => x.col).join() === '0,22,23');
await shot('part5');

// Part 6: the gather and the bytes.
await part(6);
await seek(200);
check('part 6, early: no bars yet', (await page.$(S + ' [data-bars]')) === null);
await seek(1000);
const outCols = await page.$$eval(S + ' [data-out-col]', (es) => es.map((e) => Number(e.dataset.outCol)));
check('part 6: 12 output slots hold the model\'s columns in column order', JSON.stringify(outCols) === JSON.stringify(tr.out.map((o) => o.col)), JSON.stringify(outCols));
c = await cells();
check('part 6: the tie at column 23 is cut, 0 and 22 kept with their slots', c.find((x) => x.col === 23).state === 'dropped' && c.find((x) => x.col === 0).slot === '0'
  && c.find((x) => x.col === 22).slot === String(tr.out.findIndex((o) => o.col === 22)));
check('part 6: the bars compare 20 with 80 bytes per key', !!(await page.$(S + ' [data-bars]')) && (await headline()).endsWith(`read ${M.SELECT_PER_KEY} bytes per key, a sort moves ${M.SORT_PER_KEY}`) && M.SORT_PER_KEY / M.SELECT_PER_KEY === 4);
await shot('part6');

// ---------------------------------------------------------------- playing
await part(3);
check('the caption follows the part', (await caption()) === scripts[2]);
await setBox('sound', false);
check('the button offers to start this part', ['Play', 'Resume'].includes(await primaryText()), await primaryText());
await primary();
await pause(1500);
const pos = await slider();
check('it plays with the sound off: the position advances', pos > 10, String(pos));
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

// Phone width: the scene scales, the page does not scroll sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await shot('narrow');

// Lab X1.1's files are loaded on this page by their planned path; until the lab exists they answer 404.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
