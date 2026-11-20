// Browser check for section 15.6: the prediction, the bank explorer (A and B strips in
// both layouts, both bank counts and both widths, the worst case over every access,
// the conflict-free lists, the project's W-tile store with both load mappings,
// picking a bank), the erratum hunt (the checked snippet, the question, the misread
// panel for several thread rows) and two more questions. Expected values come from
// src/sim/gemmbanks.ts, imported inside the page through the app's own module server.
// Usage: node e2e/ch15-s156.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
/** Run fn(model, ...args) inside the page against src/sim/gemmbanks.ts. */
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/gemmbanks.ts');
    return (0, eval)(src)(m, ...args);
  }, fn.toString(), args);
const E = bench('Banks for one strip read');
const pick = async (label, value, scope = E) => {
  const ok = await page.evaluate((scope, label, value) => {
    const f = [...document.querySelector(scope).querySelectorAll('label.field')].find((l) => l.childNodes[0]?.textContent.trim().startsWith(label));
    const s = f?.querySelector('select');
    if (!s) return false;
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, scope, label, value);
  await pause(80);
  return ok;
};
const slide = async (aria, value) => {
  await page.$eval(`${E} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const press = async (label, scope = E) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const shown = () => page.evaluate((E) => {
  const root = document.querySelector(E);
  return {
    degree: Number(root.querySelector('svg[data-degree]').dataset.degree),
    worst: Number(root.querySelector('[data-worst]').dataset.worst),
    words: [...root.querySelectorAll('g[data-bank]')].map((g) => Number(g.dataset.words)),
    selected: Number(root.querySelector('[data-selected]').dataset.selected),
    free: root.querySelector('[data-free]')?.dataset.free ?? null,
  };
}, E);
const strip = (spec) => model((m, spec) => {
  const r = m.stripRead(spec);
  const { warp, i, p, ...rest } = spec;
  return { degree: r.degree, worst: m.stripWorst(rest).degree, words: r.perBank.map((b) => b.words.length) };
}, spec);
const wst = (spec) => model((m, spec) => {
  const r = m.wStore(spec);
  const { warp, step, ...rest } = spec;
  return { degree: r.degree, worst: m.wStoreWorst(rest).degree, words: r.perBank.map((b) => b.words.length) };
}, spec);
const same = (got, want) => got.degree === want.degree && got.worst === want.worst && JSON.stringify(got.words) === JSON.stringify(want.words);

await goto(page, 'ch15/15.6');
await page.waitForSelector(E);

// ---- prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('smallest leading dimension')));
await (await pq.$('input[type=number]')).type('9');
await (await pq.$('button.primary')).click();
await pause(150);
const first = await model((m) => m.freeLds('remap', 32, 32)[0]);
check('prediction reveal: the model’s first free leading dimension is 9', first === 9 && (await pq.evaluate((e) => e.textContent)).includes('It is 9, as you said'));

// ---- the explorer, the book's case
const A = { layout: 'remap', strip: 'A', ld: 8, banks: 32, W: 32, warp: 0, i: 0, p: 0 };
let got = await shown();
let want = await strip(A);
check('default is Fig. 15.11(a): 8-way, all in bank 0', same(got, want) && got.degree === 8 && got.words[0] === 8, JSON.stringify([got, want]));
const bank0 = await text(E + ' [data-selected]');
check('bank 0 lists words 0, 32, ..., 224 with four lanes each', [0, 32, 64, 96, 128, 160, 192, 224].every((w) => bank0.includes(`word ${w} = A_s[${w / 8}][0]`)) && bank0.includes('lanes 0–3'), bank0);
if (out) { await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-banks-a.png' }); }
await press('Fig. 15.11(b)');
got = await shown();
want = await strip({ ...A, ld: 9 });
check('Fig. 15.11(b): ld 9, no conflict, banks 0, 4, ..., 28', same(got, want) && got.degree === 1 && [0, 4, 8, 12, 16, 20, 24, 28].every((b) => got.words[b] === 1), JSON.stringify(got.words));
await pick('Step i', 1);
got = await shown();
check('second elements (i = 1): banks 1, 5, ..., 29', [1, 5, 9, 13, 17, 21, 25, 29].every((b) => got.words[b] === 1));
await pick('Step i', 0);

// Leading dimensions on 32 and 64 banks (ch15-facts E1).
for (const [banks, lds] of [[32, { 10: 2, 12: 4, 16: 8 }], [64, { 10: 1, 12: 2, 16: 8 }]]) {
  await pick('Banks', banks);
  for (const [ld, d] of Object.entries(lds)) {
    await slide('A_s leading dimension', ld);
    got = await shown();
    want = await strip({ ...A, ld: Number(ld), banks });
    check(`§15.5 A strip, ld ${ld}, ${banks} banks: worst ${d === 1 ? 'none' : d + '-way'}`, got.worst === d && same(got, want), JSON.stringify([got.worst, want.worst]));
  }
}
await pick('Banks', 32);
await slide('A_s leading dimension', 8);
await press('List every conflict-free leading dimension for this setting');
got = await shown();
check('conflict-free list on 32 banks: every odd ld', got.free === (await model((m) => m.freeLds('remap', 32, 32).join(','))) && got.free === '9,11,13,15,17', got.free);
await pick('Banks', 64);
await press('List every conflict-free leading dimension for this setting');
got = await shown();
check('conflict-free list on 64 banks: ld not divisible by 4', got.free === '9,10,11,13,14,15,17', got.free);
await pick('Banks', 32);

// Every access of a kind is a shifted copy of the first (the index is affine in warp, i
// and the register), so the worst case equals the case shown: check that on a few.
for (const [ld, p] of [[12, 1], [10, 5]]) {
  await slide('A_s leading dimension', ld);
  await pick('Register', p);
  got = await shown();
  check(`ld ${ld}, register ${p}: this access is as bad as the worst`, got.degree === got.worst, JSON.stringify(got));
}
await pick('Register', 0);
await slide('A_s leading dimension', 8);

// Width 64.
await pick('Warp width', 64);
for (const [banks, d] of [[32, 8], [64, 4]]) {
  await pick('Banks', banks);
  got = await shown();
  want = await strip({ ...A, W: 64, banks });
  check(`64-wide wavefront, A strip, ld 8, ${banks} banks: ${d}-way`, got.worst === d && same(got, want), JSON.stringify([got.worst, want.worst]));
}
await pick('Banks', 32);
await pick('Warp width', 32);

// B strips and the early layout.
await press('B strip');
got = await shown();
want = await strip({ ...A, strip: 'B' });
check('§15.5 B strip: no conflict, 4 distinct words', got.worst === 1 && same(got, want) && got.words.reduce((a, b) => a + b, 0) === 4, JSON.stringify(got.words));
await pick('Thread layout', 'coarse');
got = await shown();
want = await strip({ ...A, strip: 'B', layout: 'coarse' });
check('§15.3/§15.4 B strip: 4-way on 32 banks', got.worst === 4 && same(got, want), JSON.stringify([got.worst, want.worst]));
await pick('Banks', 64);
got = await shown();
check('§15.3/§15.4 B strip: 2-way on 64 banks', got.worst === 2, String(got.worst));
await pick('Banks', 32);
await press('A strip');
got = await shown();
want = await strip({ ...A, layout: 'coarse' });
check('§15.3/§15.4 A strip: 2-way at ld 8', got.worst === 2 && same(got, want), JSON.stringify([got.worst, want.worst]));
await slide('A_s leading dimension', 9);
got = await shown();
check('§15.3/§15.4 A strip: none at ld 9', got.worst === 1, String(got.worst));
await pick('Thread layout', 'remap');
await slide('A_s leading dimension', 8);

// The project's W tile.
const WS = { mapping: 'scalar', pitch: 128, banks: 32, W: 32, warp: 0, step: 0 };
await press('W tile, scalar loads');
got = await shown();
want = await wst(WS);
check('W tile, scalar loads, pitch 128: 8-way', got.worst === 8 && same(got, want), JSON.stringify([got.worst, want.worst]));
if (out) { await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-banks-w128.png' }); }
await slide('B_s pitch', 129);
got = await shown();
check('W tile, scalar loads, pitch 129: up to 4-way', got.worst === 4 && same(got, await wst({ ...WS, pitch: 129 })), String(got.worst));
await slide('B_s pitch', 132);
got = await shown();
check('W tile, scalar loads, pitch 132: free', got.worst === 1 && same(got, await wst({ ...WS, pitch: 132 })), String(got.worst));
await press('List every conflict-free pitch for this setting');
got = await shown();
check('conflict-free pitches 128 to 148: 132, 140, 148', got.free === '132,140,148', got.free);
await press('W tile, float4 loads');
await slide('B_s pitch', 128);
got = await shown();
check('W tile, float4 loads, pitch 128: 2-way', got.worst === 2 && same(got, await wst({ ...WS, mapping: 'float4' })), String(got.worst));
await slide('B_s pitch', 132);
got = await shown();
check('W tile, float4 loads, pitch 132: free', got.worst === 1, String(got.worst));
// Pick a bank by clicking its bar.
await page.$eval(`${E} g[data-bank="7"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(80);
check('clicking bank 7 selects it', (await shown()).selected === 7);
await press('A strip');

// ---- the erratum hunt
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const snippet = fs.readFileSync(path.join(repo, 'labs/ch15/snippets/s156_padded_offset.cu'), 'utf8').split('\n');
const code = await page.evaluate(() => {
  const fig = [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('mm_tiled_padded'));
  fig.scrollIntoView({ block: 'start' });
  return {
    lines: [...fig.querySelectorAll('.ln')].map((l) => ({ n: Number(l.dataset.n), t: l.textContent.replace(/\n$/, '') })),
    hl: [...fig.querySelectorAll('.ln.hl')].map((l) => Number(l.dataset.n)),
    status: fig.querySelector('.status')?.textContent ?? '',
  };
});
check('hunt listing is the snippet file, with its own line numbers', code.lines.length > 10 && code.lines.every((l) => snippet[l.n - 1] === l.t), JSON.stringify(code.lines.slice(0, 2)));
check('hunt listing is marked front-end-checked', code.status.startsWith('front-end-checked'), code.status);
check('before answering, the three A_s lines are highlighted', JSON.stringify(code.hl) === JSON.stringify([25, 27, 30]), JSON.stringify(code.hl));
check('no misread panel before the question is answered', !(await page.$(bench('What a thread row reads with the old offset'))));
const hq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Erratum hunt')));
await (await hq.$$('input[type=radio]'))[0].click();
await (await hq.$('button.primary')).click();
await pause(150);
let hv = await hq.evaluate((e) => e.textContent);
check('a wrong pick is marked and explained', hv.includes('Not quite.') && hv.includes('still 8 floats'), hv.slice(0, 200));
await (await hq.$$('input[type=radio]'))[1].click();
await (await hq.$('button.primary')).click();
await pause(200);
hv = await hq.evaluate((e) => e.textContent);
check('the mm offset line is the right answer', hv.includes('Correct.'), hv.slice(0, 200));
const M = bench('What a thread row reads with the old offset');
check('after answering, the misread panel appears', !!(await page.$(M)));
const hl2 = await page.evaluate(() => [...[...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('mm_tiled_padded')).querySelectorAll('.ln.hl')].map((l) => Number(l.dataset.n)));
check('…and only line 30 stays highlighted', JSON.stringify(hl2) === '[30]', JSON.stringify(hl2));
const mis = () => page.$eval(M + ' [data-misread]', (p) => ({ at: p.dataset.misread, right: Number(p.dataset.right), pad: Number(p.dataset.padding), cells: p.closest('section').querySelectorAll('rect[data-read="wrong"]').length }));
for (const ty of [1, 15, 0]) {
  await pick('Thread row ty', ty, M);
  const g = await mis();
  const w = await model((m, ty) => { const r = m.wrongOffsetReads(ty); return { at: `${r[0].row},${r[0].col}`, right: r.filter((x) => x.right).length, pad: r.filter((x) => x.padding).length, wrong: new Set(r.filter((x) => !x.right).map((x) => x.word)).size }; }, ty);
  check(`thread row ${ty}: starts at ${w.at}, ${w.right} of 64 right, ${w.pad} from the padding`, g.at === w.at && g.right === w.right && g.pad === w.pad && g.cells === w.wrong, JSON.stringify([g, w]));
}
if (out) { await page.$eval(M, (e) => e.scrollIntoView({ block: 'start' })); await pick('Thread row ty', 1, M); await page.screenshot({ path: out + '-misread.png' }); }

// ---- two more questions
const askQ = async (needle, act) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await act(q);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
const bytes = await model((m) => m.sharedBytes(9).total);
let r = await askQ('How many bytes of shared memory', async (q) => (await q.$('input[type=number]')).type(String(bytes)));
check(`numeric question: ${bytes} bytes is correct`, bytes === 8704 && r.includes('Correct.'), r.slice(0, 120));
r = await askQ('Which B_s pitch makes that store', async (q) => (await q.$$('input[type=radio]'))[2].click());
check('choice question: pitch 132 is correct', r.includes('Correct.'), r.slice(0, 120));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
