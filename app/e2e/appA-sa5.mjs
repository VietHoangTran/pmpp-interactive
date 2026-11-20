// Browser check for appendix A, section A.5: the decode-ceiling explorer, the "near the top
// of E4M3" prediction, the MX block explorer, the narrated scene "Quantising one block", the
// C++ listing and the questions. Every number it expects comes from the models the page
// runs on (src/sim/mxblock.ts, src/sim/minifloat.ts), imported here directly.
// Usage: node e2e/appA-sa5.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

// The models import their siblings without a file extension (as the app's bundler allows).
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
const { fmt } = await import('../src/lib/reports.ts');
const { E4M3, E5M2 } = await import('../src/sim/minifloat.ts');
const { SCENE_BLOCK, bitsPerWeight, blockBytes, convert, decodeCeiling, maxTree, quantizeBlock, sampleBlock, sceneValues } = await import('../src/sim/mxblock.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const B = (t) => `section.bench[aria-label="${t}"]`;
const D = B('Decode ceiling by weight format');
const M = B('One MX block');
const S = 'section.scene[aria-label="Quantising one block"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const clickText = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(120);
  return ok;
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);

await goto(page, 'appA/A.5');
await page.waitForSelector(D);

// ---------------------------------------------------------------- decode ceiling
const paramsAt = (s) => Number((0.5e9 * (70e9 / 0.5e9) ** (s / 1000)).toPrecision(2));
const headline = () => page.$eval(D + ' .ceiling-head', (e) => ({ text: e.textContent.trim(), v: Number(e.dataset.ceiling) }));
let h = await headline();
const want7 = decodeCeiling(7e9, bitsPerWeight('mxfp4'), 3350);
check('decode ceiling: 7 billion MXFP4 weights on the book\'s 3.35 TB/s', h.text.startsWith(`${fmt(want7)} tokens per second`) && Math.abs(h.v - want7) < 1e-9, h.text);
const rowCeil = () => page.$$eval(D + ' tbody tr', (rows) => rows.map((r) => [r.dataset.format, ...[...r.querySelectorAll('td')].map((td) => td.textContent.trim())]));
let rows = await rowCeil();
const okRows = (rows, params, gbs) => rows.every(([f, , bits, , , ceil]) => bits === fmt(bitsPerWeight(f)) && ceil === fmt(decodeCeiling(params, bitsPerWeight(f), gbs)));
check('every format\'s bits per weight and ceiling are the model\'s', okRows(rows, 7e9, 3350), JSON.stringify(rows));
const mx = rows.find((r) => r[0] === 'mxfp4');
check('MXFP4 over BF16: 3.76 times, 4.25 bits, 3.76 FLOP/B (q-ceiling)', mx[6] === '3.76' && mx[2] === '4.25' && mx[4] === '3.76', JSON.stringify(mx));
check('the scales cost 0.25 bit (MX) and 0.5 bit (16-element FP4)', rows.find((r) => r[0] === 'mxfp8')[2] === '8.25' && rows.find((r) => r[0] === 'nvfp4')[2] === '4.5');
await setRange(D + ' input[type=range]', 1000);
rows = await rowCeil();
check('70 billion parameters: the table follows the model', okRows(rows, paramsAt(1000), 3350) && (await text(D + ' label.field')).includes('70 billion'), JSON.stringify(rows[1]));
await setSelect(D + ' select[aria-label="weight format"]', 'bf16');
h = await headline();
check('selecting BF16 moves the headline to the BF16 ceiling', Math.abs(h.v - decodeCeiling(70e9, 16, 3350)) < 1e-9, h.text);
const reports = await (await fetch(BASE + '/api/reports')).json();
const yours = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) => {
  const r = reports[b].results.filter((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution').flatMap((x) => x.bench).find((x) => x.name === 'roofline');
  return r && typeof r.bandwidth_gb_per_s === 'number' ? [{ b, gbs: r.bandwidth_gb_per_s }] : [];
});
if (yours.length) {
  await setSelect(D + ' select[aria-label="device"]', yours[0].b);
  h = await headline();
  check(`your device (${yours[0].b}, lab 5.1): the ceiling uses its measured bandwidth`, Math.abs(h.v - decodeCeiling(70e9, 16, yours[0].gbs)) < 1e-6, h.text);
} else {
  check('no lab 5.1 measurement here: only the book\'s example and custom are offered', (await page.$$eval(D + ' select[aria-label="device"] option', (o) => o.length)) === 2);
}
await setSelect(D + ' select[aria-label="device"]', 'custom');
await page.$eval(D + ' input[type=number]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '100');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
h = await headline();
check('a custom 100 GB/s gives the model\'s ceiling', Math.abs(h.v - decodeCeiling(70e9, 16, 100)) < 1e-9, h.text);
if (out) await page.$eval(D, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-ceiling.png' });

// ---------------------------------------------------------------- predict: the top of E4M3
const p = await question('460 is converted to E4M3');
await (await p.$('input[type=number]')).type('460');
await (await p.$('button.primary')).click();
await pause(200);
const reveal = await page.$$eval('[data-reveal="e4m3"] tbody tr', (rs) => rs.map((r) => ({ x: Number(r.dataset.input), sat: r.querySelector('[data-sat]').dataset.sat, nosat: r.querySelector('[data-nosat]').dataset.nosat })));
const want = [[448, E4M3], [460, E4M3], [464, E4M3], [470, E4M3], [57344, E5M2], [61440, E5M2]].map(([x, f]) => ({ x, sat: String(convert(x, f, 'saturate').value), nosat: String(convert(x, f, 'ieee').value) }));
check('the reveal shows each conversion as the model computes it', JSON.stringify(reveal) === JSON.stringify(want), JSON.stringify(reveal));
check('460 → 448 either way; 470 → NaN without saturation; 61,440 → infinity in E5M2', want[1].sat === '448' && want[1].nosat === '448' && want[3].nosat === 'NaN' && want[5].nosat === 'Infinity');
check('the reveal answers the prediction it was given', (await p.evaluate((e) => e.textContent)).includes('You predicted 460') && (await p.evaluate((e) => e.textContent)).includes('460 itself is not an E4M3 value'));

// ---------------------------------------------------------------- the MX block explorer
const svgData = () => page.$eval(M + ' svg', (s) => ({ ...s.dataset }));
const model = (seed, outlier, opts) => quantizeBlock(sampleBlock(seed, 32, 0.25, outlier, 7), opts);
const MX4 = { elem: 'E2M1', k: 32, scale: 'ocp-floor' };
let d = await svgData();
let q = model(1, 0, MX4);
check('MX block, default: MXFP4, no outlier; lost and clamped counts are the model\'s', Number(d.zeroed) === q.zeroed && Number(d.clamped) === q.clamped && d.scales === String(q.blocks[0].scale), JSON.stringify(d));
check('the panel names the format and its 4.25 bits per weight', (await text(M)).includes('MXFP4 · 4.25 bits per weight'));
await setRange(M + ' input[aria-label="outlier"]', 40);
d = await svgData();
q = model(1, 40, MX4);
check('an outlier of 40 sets the scale and zeroes the small weights', Number(d.zeroed) === q.zeroed && q.zeroed >= 25 && d.scales === String(q.blocks[0].scale) && q.blocks[0].scaleExp === 3, JSON.stringify(d));
check('the summary chip says how many were lost', (await text(M + ' [data-summary]')).includes(`${q.zeroed} of 32 lost to 0`));
const lostDrawn = await page.$$eval(M + ' svg g[data-lost="1"]', (g) => g.length);
check('the drawing marks exactly those weights as lost', lostDrawn === q.zeroed, `${lostDrawn} vs ${q.zeroed}`);
if (out) await page.$eval(M, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-mx-outlier.png' });
await clickText(M, '16');
d = await svgData();
q = model(1, 40, { ...MX4, k: 16 });
const blockRows = await page.$$eval(M + ' tbody tr[data-block]', (r) => r.length);
check('16 per scale: two blocks; the outlier-free half keeps its weights', Number(d.zeroed) === q.zeroed && blockRows === 2 && q.blocks[1].zeroed <= 2 && d.scales === q.blocks.map((b) => b.scale).join(','), JSON.stringify(d));
await clickText(M, '32');
await setSelect(M + ' select[aria-label="element format"]', 'E4M3');
d = await svgData();
q = model(1, 40, { ...MX4, elem: 'E4M3' });
check('E4M3 elements keep every small weight', Number(d.zeroed) === 0 && q.zeroed === 0 && (await text(M)).includes('MXFP8'), JSON.stringify(d));
await setSelect(M + ' select[aria-label="element format"]', 'E2M1');
await clickText(M, '16');
await setSelect(M + ' select[aria-label="scale rule"]', 'e4m3');
d = await svgData();
q = model(1, 40, { elem: 'E2M1', k: 16, scale: 'e4m3' });
check('the E4M3 scale rule, 16 per scale: NVIDIA\'s format, scales as the model rounds them', d.scales === q.blocks.map((b) => b.scale).join(',') && Number(d.zeroed) === q.zeroed && (await text(M)).includes("NVIDIA's 16-element FP4"), JSON.stringify(d));
await setSelect(M + ' select[aria-label="scale rule"]', 'cublas-ceil');
await clickText(M, '32');
d = await svgData();
q = model(1, 40, { ...MX4, scale: 'cublas-ceil' });
check('the cuBLAS rule clamps nothing', Number(d.clamped) === 0 && q.clamped === 0 && Number(d.zeroed) === q.zeroed, JSON.stringify(d));
await setSelect(M + ' select[aria-label="scale rule"]', 'ocp-floor');
await clickText(M, 'New weights');
d = await svgData();
q = model(2, 40, MX4);
check('new weights: seed 2, recomputed', d.seed === '2' && Number(d.zeroed) === q.zeroed && d.scales === String(q.blocks[0].scale), JSON.stringify(d));
await page.$eval(M + ' details summary', (e) => e.click());
await pause(150);
const el = await page.$$eval(M + ' [data-elements] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td')].map((t) => t.textContent.trim())));
const sh = (v, dd) => (v === 0 ? (Object.is(v, -0) ? '−0' : '0') : String(Number(v.toPrecision(dd))).replace(/-/g, '−'));
check('all 32 elements listed; element 7 decodes to the model\'s value', el.length === 32 && el[7][4] === sh(q.decoded[7], 4), JSON.stringify(el[7]));

// ---------------------------------------------------------------- the scene
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const res = await fetch(BASE + '/content/appA/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appA/narration/mx-block-${n}.txt`)).text()));
check('six scripts of 35 to 60 words', scripts.every((s) => { const w = s.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(' ').length).join(' '));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const seek = (v) => setRange(S + ' input[type=range]', v);
const drawn = () => page.$eval(S + ' .stage svg', (s) => ({ ...s.dataset, text: [...s.querySelectorAll('text')].map((t) => t.textContent) }));
const labels = await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.trim()));
check('six parts with their names', labels.join('|') === '1. Thirty-two weights|2. Find the largest|3. Choose the scale|4. Divide and snap|5. Store|6. Decode', labels.join('|'));
check('captions switch exists and is off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
if (manifest?.clips?.['mx-block-1']) check('with audio and captions off, no caption', !(await hasCaption()));
else check('with no audio, part 1 shows its script anyway', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'none');
await setBox('captions', true);
check('captions on: part 1\'s script without tags', (await caption()) === scripts[0]);
await setBox('sound', false);
const V = sceneValues();
const Q32 = quantizeBlock(V, MX4);
const Q16 = quantizeBlock(V, { ...MX4, k: 16 });
d = await drawn();
check('part 1: all 32 weights; the outlier is the model\'s 12 at element 7', d.beat === '1' && Number(d.outlier) === SCENE_BLOCK.outlier && d.text.some((t) => t.includes('32 weights')));
await press('Play');
await pause(1500);
const moved = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('with sound off it plays on estimated timings', moved > 20, String(moved));
await press('Pause');
await seek(1000);
d = await drawn();
check('part 1 ends naming the outlier', d.text.some((t) => t.includes('element 7: 12')));
await press('2. Find the largest');
await seek(1000);
d = await drawn();
const tree = maxTree(V);
check('part 2 ends after 5 rounds on element 7 (the model\'s max tree)', d.round === String(tree.length - 1) && d.winner === String(SCENE_BLOCK.at) && tree.length - 1 === 5, JSON.stringify(d));
await seek(400);
d = await drawn();
check('part 2 midway: the survivors are one tree level', tree.some((l) => String(l.length) === d.survivors) && Number(d.survivors) < 32, JSON.stringify(d));
await press('3. Choose the scale');
check('jumping to part 3 shows its script', (await caption()) === scripts[2]);
await seek(1000);
d = await drawn();
check('part 3: exponent 1, X = 2, UE8M0 byte 128, as the model and the script say', d.exp === String(Q32.blocks[0].scaleExp) && d.scale === '2' && d.code === '128' && scripts[2].includes('2 to the power 1, which is 2'), JSON.stringify(d));
await press('4. Divide and snap');
await seek(1000);
d = await drawn();
check('part 4 ends: 30 small weights on 0, the outlier exactly on 6', d.zeroed === String(Q32.zeroed) && Q32.zeroed === 30 && d.top === '6', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene-4.png' });
await press('5. Store');
await seek(1000);
d = await drawn();
check('part 5: 17 bytes against 64 in BF16', d.mx === String(blockBytes(4, 32, 1)) && d.bf === '64' && d.scaleByte === '80', JSON.stringify(d));
await press('6. Decode');
await seek(300);
d = await drawn();
check('part 6, first half: one block of 32, 30 lost', d.k === '32' && d.zeroed === String(Q32.zeroed), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 6, second half: two blocks of 16, the outlier-free half loses none', d.k === '16' && d.exps === Q16.blocks.map((b) => b.scaleExp).join(',') && Q16.blocks[1].zeroed === 0 && d.text.some((t) => t.includes('loses 0')), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene-6.png' });
await seek(975);
await press('Resume');
await pause(2500);
check('after the last part it offers to play again', (await page.$eval(S + ' button.primary', (e) => e.textContent.trim())) === 'Play again');
await setBox('captions', false);

// ---------------------------------------------------------------- the listing
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check('the C++ listing is labs/appA/snippets/mx_quantize.h', code.includes('inline MxBlock mx_quantize(const float* v)') && code.includes('std::ilogb(amax) - kE2M1MaxExp'));

// ---------------------------------------------------------------- questions
const answerNum = async (needle, v) => {
  const qq = await question(needle);
  await (await qq.$('input[type=number]')).type(String(v));
  await (await qq.$('button.primary')).click();
  await pause(150);
  return (await qq.evaluate((e) => e.textContent)).includes('Correct.');
};
const answerChoice = async (needle, idx) => {
  const qq = await question(needle);
  await (await qq.$$('input[type=radio]'))[idx].click();
  await (await qq.$('button.primary')).click();
  await pause(150);
  return (await qq.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-ceiling: 3.2 (16 over 4 + 32/32 bits) is accepted', await answerNum('batch-1 decode ceiling rise', 3.2));
check('q-mxfp4-bits: 4.125 (blocks of 64) is accepted', await answerNum('across 64 weights instead of 32', 4.125));
check('neither answer is printed on the page before its question',
  await page.evaluate(() => { const t = document.querySelector('.prose')?.textContent ?? ''; return !t.slice(0, t.indexOf('decode ceiling rise')).includes('3.2 ×') && !t.slice(0, t.indexOf('across 64 weights')).includes('4.125'); }));
check('q-outlier: 0 is right', await answerChoice('one weight of 100 and 31 weights near 0.3', 0));
check('q-h100: TF32 with sparsity is right', await answerChoice('What is the 989', 1));
check('q-fnuz: halve, NaN at 0x80, 240 is right', await answerChoice('FNUZ E4M3. What happens', 0));
check('answers were saved through PUT /api/progress (intercepted)', puts > 0, String(puts));

// ---------------------------------------------------------------- dark theme, narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(250);
const hard = await page.$$eval(`${M} svg [fill], ${S} svg [fill], ${M} svg [stroke], ${S} svg [stroke]`, (els) =>
  els.flatMap((e) => ['fill', 'stroke'].map((a) => e.getAttribute(a))).filter((v) => v && v !== 'none' && !v.startsWith('var(')),
);
check('drawings use theme variables only', hard.length === 0, hard.join(','));
if (out) {
  await page.$eval(M, (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-mx-dark.png' });
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  benches: [...document.querySelectorAll('.bench > .body')].filter((b) => b.scrollWidth > b.clientWidth + 1).map((b) => b.parentElement.getAttribute('aria-label')),
}));
check('at 390 px the page does not scroll sideways and no bench scrolls inside itself', narrow.page && narrow.benches.length === 0, JSON.stringify(narrow));
if (out) {
  await page.$eval(M, (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-mx-narrow.png' });
  await page.$eval(D, (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-ceiling-narrow.png' });
}

const expected404 = (u) => /\/content\/appA\/narration\/(manifest\.json|mx-block-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
