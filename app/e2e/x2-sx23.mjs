// Browser check for section X2.3: "Where each weight lives", "Unpack one word", "Blocks end
// to end", the byte-5 prediction, the narrated scene "One block, packed, shipped and
// unpacked" and three questions. Model values are recomputed in the page from
// app/src/sim/quant.ts through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx23.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
// Leave no trace in the real progress file, and start from no answers so the prediction is fresh.
page.on('request', (r) =>
  r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : '{"version":1,"sections":{},"answers":{},"review":{}}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const setSelect = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const setRange = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} input${aria ? `[aria-label="${aria}"]` : '[type=range]'}`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};

await goto(page, 'x2/X2.3');
const L = bench('Where each weight lives');
await page.waitForSelector(L);

const M = await page.evaluate(async () => {
  const q = await import('/src/sim/quant.ts');
  const b = q.quantizeQ4_0(q.SCENE_BLOCK);
  const qs = b.bytes.slice(2);
  const y = q.sceneActivations();
  const c1 = q.unpackTrace('cuda32', qs, 1, y);
  const m2 = q.unpackTrace('metal16', qs, 2, y);
  const dot = q.blockDotQ4_0(b, y, q.SCENE_BLOCK);
  return {
    codes: b.codes, qs, d16: q.fp16FromBits(b.dBits),
    c1mask: c1[1].out.map((o) => [o.weight, o.value]), c1shift: c1[2].out.map((o) => [o.weight, o.value]), c1sub: c1[3].out.map((o) => o.value),
    m2masked: m2.slice(1, 5).map((s) => s.out[0].value), m2prod: m2[5].out.map((o) => o.value),
    mlx13: q.placeOf('mlx4', 13), folded: dot.folded, exact: dot.exact,
    align4: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => q.qsLoads(i, 4).every((l) => l.ok)),
  };
});

// ---------------- where each weight lives
let r = await text(L + ' [data-part="readout"]');
check('Q4_0: weight 5 is byte 5, low nibble, sharing with 21', r.includes('Weight 5') && r.includes('byte 5, low nibble') && r.includes('weight 21'), r);
check('Q4_0: byte 5 is highlighted with weights 5 and 21', (await page.$$eval(L + ' [data-w][aria-pressed="true"]', (e) => e.map((x) => +x.dataset.w))).join(',') === '5,21');
await page.$eval(L + ' [data-b="9"]', (e) => e.click());
await pause(80);
r = await text(L + ' [data-part="readout"]');
check('Q4_0: byte 9 holds weights 9 and 25 with the model’s codes', r.includes('weight 9 in the low nibble') && r.includes('weight 25 in the high nibble') && r.includes(`(${M.codes[25]})`), r);
check('Q4_0: byte 9 shows the packed hex', r.includes('0x' + M.qs[9].toString(16).toUpperCase().padStart(2, '0')), r);
await press(L, 'MLX 4-bit');
await page.$eval(L + ' [data-w="13"]', (e) => e.click());
await pause(80);
r = await text(L + ' [data-part="readout"]');
check('MLX: weight 13 is word 1, bits 20 to 23, beside weight 12', r.includes('uint32 word 1, bits 20 to 23') && r.includes('weight 12'), r);
await press(L, 'llama.cpp Q4_K chunk');
check('Q4_K chunk shows 64 weights and 32 bytes', (await page.$$(L + ' [data-w]')).length === 64 && (await page.$$(L + ' [data-b]')).length === 32);
await page.$eval(L + ' [data-w="40"]', (e) => e.click());
await pause(80);
r = await text(L + ' [data-part="readout"]');
check('Q4_K: weight 40 is byte 8, high nibble, with weight 8', r.includes('byte 8, high nibble') && r.includes('weight 8.'), r);
if (out) await page.screenshot({ path: out + '-where.png' });

// ---------------- prediction: byte 5
const P = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('weight 5 shares its byte')));
await (await P.$('input[type=number]')).type('6');
await (await P.$('button.primary')).click();
await pause(150);
const pr = await P.evaluate((e) => e.textContent);
check('byte-5 prediction reveals 21 and names MLX’s order for a neighbour guess', pr.includes('Weight 21') && pr.includes('MLX'), pr);

// ---------------- unpack one word
const U = bench('Unpack one word');
await page.$eval(U, (e) => e.scrollIntoView());
const outRows = () => page.$$eval(U + ' [data-part="out"] tr[data-weight]', (rs) => rs.map((x) => [+x.dataset.weight, +x.dataset.value]));
check('CUDA, word 1: load step draws 32 bits', (await page.$$(U + ' [data-bit]')).length === 32);
await press(U, 'Step');
check('CUDA mask gives weights 4..7 with the model codes', JSON.stringify(await outRows()) === JSON.stringify(M.c1mask), JSON.stringify(await outRows()));
await press(U, 'Step');
check('CUDA shift gives weights 20..23', JSON.stringify(await outRows()) === JSON.stringify(M.c1shift), JSON.stringify(await outRows()));
await press(U, 'Step');
check('minus 8 gives the signed codes', JSON.stringify((await outRows()).map((x) => x[1])) === JSON.stringify(M.c1sub));
await press(U, 'Metal: 16-bit word');
check('Metal, word 2: load step draws 16 bits', (await page.$$(U + ' [data-bit]')).length === 16);
const masked = [];
for (let k = 0; k < 4; k++) {
  await press(U, 'Step');
  masked.push((await outRows())[0][1]);
}
check('Metal masks give code × 1, 256, 16, 4096 (model)', JSON.stringify(masked) === JSON.stringify(M.m2masked), JSON.stringify(masked));
await press(U, 'Step');
check('Metal pre-divided products are exactly code × activation (model)', JSON.stringify((await outRows()).map((x) => x[1])) === JSON.stringify(M.m2prod));
check('the Metal idiom never shifts', !(await text(U + ' .narrate')).includes('>>'));
if (out) await page.screenshot({ path: out + '-unpack.png' });

// ---------------- blocks end to end
const A = bench('Blocks end to end');
await page.$eval(A, (e) => e.scrollIntoView());
const loads = () => page.$$eval(A + ' [data-load]', (e) => e.map((x) => x.dataset.ok));
check('block 0, 4-byte loads: all 4 misaligned', (await loads()).join() === 'no,no,no,no', (await loads()).join());
await setRange(A, 'block index', 1);
check('block 1, 4-byte loads: all aligned', (await loads()).every((x) => x === 'yes'));
check('the readout lists the model’s aligned blocks', (await text(A + ' [data-part="align"]')).includes(M.align4.flatMap((o, i) => (o ? [i] : [])).join(', ')));
await press(A, '2');
await setRange(A, 'block index', 6);
check('2-byte loads: all 8 aligned for block 6', (await loads()).length === 8 && (await loads()).every((x) => x === 'yes'));
await press(A, '16');
await setRange(A, 'block index', 7);
check('16-byte load: aligned only for block 7', (await loads()).join() === 'yes');
await setRange(A, 'block index', 3);
check('16-byte load: misaligned for block 3', (await loads()).join() === 'no');

// ---------------- the scene
const SC = 'section.scene[aria-label="One block, packed, shipped and unpacked"]';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let k = 1; k <= 6; k++) scripts.push(strip(await (await fetch(`${BASE}/content/x2/narration/packed-block-${k}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every scene script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('no symbols read aloud in the scripts', scripts.every((s) => !/[=×÷<>&|%]/.test(s)));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const capBox = () => page.evaluate((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked, SC);
check('captions are off by default', (await capBox()) === false);
const hasManifest = (await fetch(`${BASE}/content/x2/narration/manifest.json`)).ok;
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent).then(strip).catch(() => null);
const goBeat = async (k) => {
  await page.evaluate((SC, k) => [...document.querySelectorAll(SC + ' .seg button')][k - 1].click(), SC, k);
  await pause(120);
  await setRange(SC, null, 1000);
};
const svg = () => page.$eval(SC + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
await goBeat(1);
if (!hasManifest) check('without audio the script shows as a caption', (await page.evaluate((SC) => document.querySelector(SC + ' .caption')?.textContent ?? '', SC)).includes('Thirty-two weights'));
let s = await svg();
check('beat 1: d = -0.8 / −8 and its FP16 value from the model', s.includes('d = -0.8 / −8 = 0.100') && s.includes(M.d16.toPrecision(6)), s);
await goBeat(2);
s = await svg();
check('beat 2: weight 21 clipped to 15', s.includes('weight 21') && s.includes('clipped to 15'), s);
await goBeat(3);
s = await svg();
check('beat 3: every packed byte matches the model', M.qs.every((b) => s.includes('0x' + b.toString(16).toUpperCase().padStart(2, '0'))), s);
check('beat 3: 18 bytes against 128', s.includes('18 bytes') && s.includes('128 bytes'));
await goBeat(4);
const qsAt = await page.$$eval(SC + ' [data-block] text', (t) => t.map((x) => x.textContent));
check('beat 4: quants start at 2, 20, 38, 56', ['2', '20', '38', '56'].every((v) => qsAt.includes(v)), qsAt.join(','));
await goBeat(5);
const sm = await page.$$eval(SC + ' [data-masked]', (t) => t.map((x) => +x.dataset.masked));
check('beat 5: four masked values from the model', JSON.stringify(sm) === JSON.stringify(M.m2masked), JSON.stringify(sm));
await goBeat(6);
s = await svg();
check('beat 6: folded sum and float32 sum from the model', s.includes(M.folded.toFixed(4)) && s.includes(M.exact.toFixed(4)), s);
if (out) await page.screenshot({ path: out + '-scene.png' });
// Parts advance by themselves; it plays with the sound off.
await page.evaluate((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), SC);
await goBeat(2);
await setRange(SC, null, 997);
await page.$eval(SC + ' button.primary', (b) => b.click());
await pause(900);
const cur = await page.$$eval(SC + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
check('playing from the end of part 2 moves on to part 3', cur === 3, String(cur));
await page.$eval(SC + ' button.primary', (b) => b.click());
// Captions switch is remembered.
await page.evaluate((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), SC);
await pause(100);
check('captions on shows the script', (await caption())?.startsWith('Now the codes pair up') ?? false, String(await caption()));
check('the captions choice is remembered', (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');
await page.evaluate(() => localStorage.setItem('captions', 'off'));

// ---------------- questions
const ask = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  if (typeof value === 'number' && (await q.$('input[type=number]'))) await (await q.$('input[type=number]')).type(String(value));
  else await (await q.$$('input[type=radio]'))[value.idx].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-byte9 accepts 25', await ask('high nibble of quant byte 9', 25));
check('q-mlx13: word 1, bits 20 to 23', await ask('where is weight 13 of a group', { idx: 0 }));
check('q-align: blocks 1 and 3', await ask('Whose quants can be read as', { idx: 0 }));
check('q-int4b: undocumented nibble order', await ask('MSL now has', { idx: 0 }));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
