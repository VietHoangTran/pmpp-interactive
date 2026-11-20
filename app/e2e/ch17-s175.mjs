// Browser check for section 17.5, "Buying regularity": the explorer "Pad, spill or sort"
// driven through every control and compared with src/sim/sparsescale.ts (loaded inside
// the page through the dev server, since it imports other models), the two predictions and
// the buttons in their reveals, every question, the narrated animation "Routing tokens to
// experts" compared with src/sim/sparseroute.ts (imported here directly), the libraries
// table, margin-note placement, and the page at phone width. It passes with or without
// narration audio: with none, the player runs on estimated timings and shows each script.
// Usage: node e2e/ch17-s175.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { route, countsAfter, gather } from '../src/sim/sparseroute.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const num = (s) => Number(String(s).replace(/[,%]/g, ''));
const X = '[data-explorer="pad-spill-sort"]';
const attr = (name) => page.$eval(X, (e, n) => e.dataset[n], name);
const outText = (sel) => page.$eval(`${X} ${sel}`, (e) => e.textContent.trim());
const cell = async (format, key) => num(await page.$eval(`${X} tr[data-format="${format}"] [data-out="${key}"]`, (e) => e.textContent));
const warpCell = async (id, key) => num(await page.$eval(`${X} tr[data-warps="${id}"] [data-out="${key}"]`, (e) => e.textContent));
const clickIn = async (scope, label) => {
  const ok = await page.evaluate(
    (scope, label) => {
      const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(120);
  return ok;
};
const setRange = async (aria, v) => {
  await page.$eval(`${X} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const setSelect = async (aria, v) => {
  await page.$eval(`${X} select[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(150);
};
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const setNumber = async (handle, value) => {
  await handle.evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));
};
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  await setNumber(await q.$('input[type=number]'), value);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const q = await qHandle(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const predict = async (needle, value) => {
  const q = await qHandle(needle);
  await setNumber(await q.$('input[type=number]'), value);
  await (await q.$('button.primary')).click();
  await pause(200);
  return q.evaluate((e) => e.textContent);
};

/** The model's numbers for the explorer's current state, computed in the page. */
const model = () =>
  page.evaluate(async (X) => {
    const m = await import('/src/sim/sparsescale.ts');
    const d = document.querySelector(X).dataset;
    const presetId = d.preset;
    // rebuild the params from what the explorer shows
    const shape = document.querySelector(X + ' select[aria-label="Shape of the row lengths"]').value;
    const rows = Number(d.rows);
    const mean = Number(document.querySelector(X + ' input[aria-label="Mean non-zeros per row"]').value);
    const longCount = Number(document.querySelector(X + ' input[aria-label="Number of long rows"]').value);
    const longLen = Number(document.querySelector(X + ' input[aria-label="Length of the long rows"]').value);
    const params = presetId ? m.preset(presetId).params : { shape, rows, mean, longCount, longLen };
    const len = m.rowLengths(params);
    const w = { value: Number(d.value), index: Number(d.index) };
    const K = Number(d.k);
    const S = Number(d.s);
    const sorted = d.sorted === '1';
    const width = Number(d.width);
    const st = m.storage(len, m.colsOf(params), w, K, S, sorted);
    const wt = m.warps(len, width, K);
    const pick = (s) => ({ numbers: s.numbers, bytes: s.bytes, pads: s.pads, coo: s.coo });
    const seg = { nvidia: 32, amd64: 64, amd128: 128 }[d.model];
    const traffic = seg ? ['csr', 'ell', 'jds'].map((f) => m.traffic(f, len, width, seg, w)) : null;
    const lanes = (trips, warp, t) => m.laneStates(trips, width, warp, t).filter((x) => x === 'work').length;
    const sortedLen = m.sortedLengths(len);
    return {
      nnz: len.reduce((a, b) => a + b, 0),
      max: Math.max(...len),
      stores: Object.fromEntries(Object.entries(st.stores).map(([k, v]) => [k, pick(v)])),
      warps: Object.fromEntries(Object.entries(wt).map(([k, v]) => [k, { spent: v.spent, useful: v.useful, efficiency: v.efficiency, longest: v.longest }])),
      traffic: traffic?.map((t) => ({ format: t.format, requested: t.requested, distinct: t.distinct })),
      grid: [0, 3, 9].map((t) => [lanes(len, 0, t), lanes(len, 1, t), lanes(sortedLen, 0, t), lanes(sortedLen, 1, t)]),
      fig: { ell: m.ell([...m.FIG1713_LENGTHS], { value: 4, index: 4 }), k2: m.hybrid([...m.FIG1713_LENGTHS], 2, { value: 4, index: 4 }) },
      book: { csr: m.csr(m.rowLengths(m.preset('book').params), { value: 4, index: 4 }).numbers, ell: m.ell(m.rowLengths(m.preset('book').params), { value: 4, index: 4 }).numbers },
      hk: m.hybrid([...Array(1000).fill(16), 400, 400], 16, { value: 4, index: 4 }).coo,
      amort: m.amortiseIterations(30, 0.6),
    };
  }, X);

/** The storage table, the warps table and the traffic table agree with the model. */
const agrees = async (label) => {
  const M = await model();
  check(`${label}: non-zeros and longest row`, num(await attr('nnz')) === M.nnz && num(await attr('max')) === M.max, `${await attr('nnz')} ${await attr('max')} vs ${M.nnz} ${M.max}`);
  let ok = true;
  let detail = '';
  for (const f of ['dense', 'csr', 'ell', 'hybrid', 'jds', 'sell']) {
    const n = await cell(f, 'numbers');
    const b = await cell(f, 'bytes');
    if (n !== M.stores[f].numbers || Math.abs(b - Math.round(M.stores[f].bytes)) > 0) {
      ok = false;
      detail += `${f}: ${n}/${b} vs ${M.stores[f].numbers}/${M.stores[f].bytes}; `;
    }
    if (f !== 'dense' && (await cell(f, 'pads')) !== M.stores[f].pads) {
      ok = false;
      detail += `${f} pads; `;
    }
  }
  check(`${label}: every storage row matches the model`, ok, detail);
  ok = true;
  detail = '';
  for (const id of ['original', 'sorted', 'hybrid']) {
    const spent = await warpCell(id, 'spent');
    const useful = await warpCell(id, 'useful');
    const longest = await warpCell(id, 'longest');
    if (spent !== M.warps[id].spent || useful !== M.warps[id].useful || longest !== M.warps[id].longest) {
      ok = false;
      detail += `${id}: ${spent}/${useful}/${longest} vs ${JSON.stringify(M.warps[id])}; `;
    }
  }
  check(`${label}: every warps row matches the model`, ok, detail);
  if (M.traffic) {
    const rows = await page.$$eval(`${X} tr[data-traffic]`, (r) =>
      r.map((x) => ({ format: x.dataset.traffic, requested: Number(x.querySelector('[data-out=requested]').textContent.replace(/,/g, '')), distinct: Number(x.querySelector('[data-out=distinct]').textContent.replace(/,/g, '')) })),
    );
    check(`${label}: requested and distinct segments match the model`, JSON.stringify(rows) === JSON.stringify(M.traffic), JSON.stringify(rows) + ' vs ' + JSON.stringify(M.traffic));
  }
  return M;
};

await goto(page, 'ch17/17.5');
await page.waitForSelector(X);

// ---------------------------------------------------------------- the book's example, predicted first
let M = await model();
check('the explorer opens on the book\'s example', (await attr('preset')) === 'book' && num(await attr('rows')) === 1000, await attr('preset'));
check("CSR stores 21,001 numbers and ELL 400,000, as the model and E1 say", M.book.csr === 21001 && M.book.ell === 400000 && (await cell('csr', 'numbers')) === 21001 && (await cell('ell', 'numbers')) === 400000);
check('the summary gives the ratio 19.05', (await outText('[data-out=ratio]')) === '19.05', await outText('[data-out=ratio]'));
let t = await predict('how many times as many numbers does ELL store', 20);
check('p-book1000 commits and reveals 19.05, rounded to 20 by the book', t.includes('You predicted 20 ×.') && t.includes('19.05') && t.includes('rounds it'), t.slice(0, 300));
await agrees('book example');

// K: drag to the typical row length
await setRange('Hybrid ELL width K', 10);
M = await agrees('book example, K = 10');
check('at K = 10 the hybrid keeps 190 pads and moves 190 entries to COO', M.stores.hybrid.pads === 190 && M.stores.hybrid.coo === 190 && (await outText('[data-out=hyb-coo]')) === '190', JSON.stringify(M.stores.hybrid));
check("the hybrid's ELL part runs at most 10 iterations", (await warpCell('hybrid', 'longest')) === 10);

// lane grid at three iterations
const grid = async () => page.$$eval(`${X} [data-grid-row]`, (g) => g.map((x) => Number(x.dataset.work)));
const want = M.grid;
let gridOk = true;
for (const [k, it] of [0, 3, 9].entries()) {
  await setRange('Iteration shown in the lane grid', it);
  const got = await grid();
  if (JSON.stringify(got) !== JSON.stringify(want[k])) gridOk = false;
}
check('the lane grid shows the model\'s working lanes at iterations 0, 3 and 9', gridOk, JSON.stringify(want));
if (out) await (await page.$('section.bench[aria-label="Pad, spill or sort"]')).screenshot({ path: out + '-explorer-book.png' });

// the load button in the reveal restores the preset
await setRange('Hybrid ELL width K', 3);
check('the reveal button reloads the book\'s example', await clickIn('[data-reveal="ch17/17.5/p-book1000"]', "Load the book's example in the explorer"));
check('after reloading, K is back at the longest row', num(await attr('k')) === 200, await attr('k'));

// ---------------------------------------------------------------- spread: sorting and slices
check('preset spread', await clickIn(X, 'Spread, 1 to 31'));
const unsortedSell = (await model()).stores.sell.pads;
await setSelect('Slice size', 32);
const slicedPads = await cell('sell', 'pads');
await page.$eval(`${X} input[aria-label="Sort rows longest first"]`, (e) => e.click());
await pause(150);
M = await agrees('spread, S = 32, sorted');
check('slices of 32 cut the padding, and sorting first cuts it further', slicedPads < unsortedSell && M.stores.sell.pads < slicedPads, `${unsortedSell} → ${slicedPads} → ${M.stores.sell.pads}`);
check('JDS (sorted) spends fewer warp-iterations than row order on spread rows', M.warps.sorted.spent < M.warps.original.spent, JSON.stringify(M.warps));
const effO = await warpCell('original', 'efficiency');
const effS = await warpCell('sorted', 'efficiency');
check('sorting raises the share of lane time that works', effS > effO, `${effO} → ${effS}`);
if (out) await (await page.$('section.bench[aria-label="Pad, spill or sort"]')).screenshot({ path: out + '-explorer-spread.png' });

// ---------------------------------------------------------------- uniform: ELL has no pads, CSR does not coalesce
check('preset uniform', await clickIn(X, 'Uniform, 16 per row'));
M = await agrees('uniform');
check('uniform rows: ELL has no pad slots', M.stores.ell.pads === 0 && (await cell('ell', 'pads')) === 0);
check('uniform rows: CSR requests 8 times the segments ELL does', M.traffic[0].requested === 8 * M.traffic[1].requested, JSON.stringify(M.traffic));

// widths: 64 lanes, 2-byte values, AMD's 128-byte lines
await setSelect('Warp width', 'w64');
check('the warp width follows the choice', num(await attr('width')) === 64);
check('2-byte values', await clickIn(X, '2 B'));
await setSelect('Segment model', 'amd128');
await agrees('uniform, 64 lanes, 2-byte values, 128-byte lines');
await setSelect('Segment model', 'metal');
check('Apple: no documented unit, the traffic panel says so and counts nothing', (await page.$eval(`${X} [data-panel=traffic]`, (e) => e.dataset.hidden)) === '1' && (await page.$$(`${X} tr[data-traffic]`)).length === 0);
await setSelect('Segment model', 'nvidia');
await setSelect('Warp width', 'book');

// 4-bit dense against 4-bit values and 2-byte indices at 25% density
await clickIn(X, '4 bits');
const twoByteIdx = await page.evaluate((X) => {
  const g = document.querySelector(X + ' span[aria-label="Bytes per index"]');
  const b = [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === '2 B');
  b.click();
  return true;
}, X);
await pause(120);
await setSelect('Rows', 256);
await setRange('Mean non-zeros per row', 64);
M = await agrees('256 rows, mean 64, 4-bit values, 2-byte indices');
const csrPct = await cell('csr', 'pct');
check('at 25% kept, CSR with 4-bit values reads more bytes than 4-bit dense', twoByteIdx && csrPct > 100 && M.stores.csr.bytes > M.stores.dense.bytes, String(csrPct));
check('editing a control leaves the preset', (await attr('preset')) === '');

// long rows controls
await setSelect('Shape of the row lengths', 'long');
await page.$eval(`${X} input[aria-label="Number of long rows"]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '2');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
await setRange('Length of the long rows', 200);
M = await agrees('long rows, 2 of 200');
check('two long rows set the ELL width', M.max === 200 && (await cell('ell', 'numbers')) === 2 * 256 * 200);

// skewed and power presets agree with the model too
check('preset skewed', await clickIn(X, 'Skewed: 1 row in 512 is 16× longer'));
await agrees('skewed');
check('preset power law', await clickIn(X, 'Power law (ours)'));
await agrees('power law');

// ---------------------------------------------------------------- Fig. 17.13, predicted first
t = await predict('Cut the ELL part to width 2', 3);
check('p-fig1713 reveals 3 pads, 5 COO entries, from rows 1 and 6, and the "row 2" slip', t.includes('3: rows 0, 4 and 7') && t.includes('3 from row 1 and 2 from row 6') && t.includes('"row 2"') && t.includes('You had it.'), t.slice(0, 500));
check('the reveal button shows Fig. 17.13 at K = 2', await clickIn('[data-reveal="ch17/17.5/p-fig1713"]', 'Show Fig. 17.13 at K = 2 in the explorer'));
M = await agrees('Fig. 17.13 at K = 2');
check('the explorer now holds Fig. 17.13 at K = 2: 3 pads and 5 COO, pure ELL 22 pads', (await attr('preset')) === 'fig1713' && num(await attr('k')) === 2 && (await cell('hybrid', 'pads')) === M.fig.k2.pads && (await cell('hybrid', 'pads')) === 3 && M.stores.hybrid.coo === 5 && (await cell('ell', 'pads')) === M.fig.ell.pads && M.fig.ell.pads === 22);
check('rows control is fixed at 8 for the figure', (await page.$eval(`${X} select[aria-label="Rows"]`, (e) => e.disabled)) === true);
if (out) await (await page.$('section.bench[aria-label="Pad, spill or sort"]')).screenshot({ path: out + '-explorer-fig1713.png' });

// ---------------------------------------------------------------- questions
t = await answerNumeric('How many entries go to the COO part', 800);
check('hybrid K: 800 is wrong', t.includes('Not quite. The answer is 768 entries.'), t.slice(0, 300));
t = await answerNumeric('How many entries go to the COO part', M.hk);
check('hybrid K: 768 is correct, pure ELL 384,000 pads', t.includes('Correct.') && t.includes('384,000'), t.slice(0, 400));
t = await answerChoice('which vector must be put back', 0);
check('sort-y: x is wrong, with the reason', t.includes('Not quite.') && t.includes('not unknowns'), t.slice(0, 400));
t = await answerChoice('which vector must be put back', 1);
check('sort-y: y with the row array is correct', t.includes('Correct.') && t.includes('y[row[r]]'), t.slice(0, 400));
t = await answerChoice('Which test lets thread', 0);
check('jds-loop: CSR\'s shape is wrong', t.includes('Not quite.') && t.includes('one pointer per row'), t.slice(0, 400));
t = await answerChoice('Which test lets thread', 3);
check('jds-loop: r < iterPtr[t+1] - iterPtr[t] is correct', t.includes('Correct.') && t.includes('our answer, not the book'), t.slice(0, 500));
t = await answerChoice('Why can JDS not do the same', 1);
check('jds-align: iterPtr is correct, with the 32-byte rule', t.includes('Correct.') && t.includes('multiple of 8'), t.slice(0, 500));
t = await answerNumeric('paid for itself', Math.round(M.amort));
check('amortise: 75 is correct', t.includes('Correct.') && t.includes('30 / 0.4 = 75'), t.slice(0, 300));
t = await answerNumeric('at which position of the expert-sorted order', 4);
check('moe-offset: 4 is wrong, the answer is 1', t.includes('Not quite. The answer is 1.'), t.slice(0, 300));
t = await answerNumeric('at which position of the expert-sorted order', 1);
check('moe-offset: 1 is correct, expert 0 holds slots 1 and 4', t.includes('Correct.') && t.includes('slots 1 and 4'), t.slice(0, 400));
t = await answerChoice('What does that buy', 2);
check('moe-atomic: a fixed order inside each slice is correct', t.includes('Correct.') && t.includes('undefined'), t.slice(0, 400));
const rightAt = await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelector('.opts')).map((q) => [...q.querySelectorAll('.opts label')].findIndex((l) => l.classList.contains('right'))));
check('the right option sits in different written positions', new Set(rightAt).size >= 3, JSON.stringify(rightAt));

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="Routing tokens to experts"]';
const R = route();
const LABELS = ['Pairs', 'Count', 'Scan', 'Scatter', 'Run', 'Put back'];
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 6; n++) {
  const r = await fetch(`${BASE}/content/ch17/narration/route-${n}.txt`);
  scripts.push(r.ok ? strip(await r.text()) : '');
}
const words = scripts.map((s) => s.split(' ').filter(Boolean).length);
check('six scripts of 35 to 60 words', words.every((w) => w >= 35 && w <= 60), JSON.stringify(words));
check('the scripts name no symbols to be read aloud', scripts.every((s) => !/[=+×→<>]/.test(s)), scripts.join(' | ').slice(0, 200));
const res = await fetch(BASE + '/content/ch17/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const parts = await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '')));
check('six parts with their labels', JSON.stringify(parts) === JSON.stringify(LABELS), JSON.stringify(parts));
const captionBox = await page.$eval(S, (s) => [...s.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
check('captions are off by default', captionBox === false);
const goPart = async (k) => {
  await page.evaluate((S, k) => document.querySelectorAll(S + ' .seg button')[k].click(), S, k);
  await pause(120);
};
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const beat = () => page.$eval(S + ' .stage svg [data-beat]', (g) => ({ ...g.dataset }));
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
if (!manifest) check('without audio, part 1 shows its script', (await caption()) === scripts[0], String(await caption()));
// part 1: the opening frame is not empty, and at the end all pairs are drawn
let b = await beat();
check('part 1 opens with the first token\'s two pairs', b.beat === '1' && Number(b.shown) === R.topk, JSON.stringify(b));
await seek(1000);
b = await beat();
const matrixCells = await page.$$eval(S + ' [data-matrix] rect', (r) => r.filter((x) => x.getAttribute('fill') !== 'var(--cell)').length);
check('part 1 ends with all 16 pairs in the matrix', Number(b.shown) === 16 && matrixCells === 16, `${b.shown} ${matrixCells}`);
if (out) await (await page.$(S)).screenshot({ path: out + '-scene-1.png' });
// part 2: counts build up to the model's
await goPart(1);
await seek(300);
const mid = await page.$eval(S + ' [data-counts]', (e) => e.dataset.counts);
const kMid = Math.min(16, Math.floor(Math.max(0, Math.min(1, (0.3 - 0.08) / (0.72 - 0.08))) * 16.999));
check('part 2 mid-way shows the counts after the pairs counted so far', mid === countsAfter(R, kMid).join(','), `${mid} vs ${countsAfter(R, kMid)}`);
await seek(1000);
check('part 2 ends with the model\'s counts and expert 2 skipped', (await page.$eval(S + ' [data-counts]', (e) => e.dataset.counts)) === R.counts.join(',') && (await page.$eval(S + ' .stage', (e) => e.textContent)).includes('skip its GEMM'));
// part 3: offsets
await goPart(2);
await seek(1000);
check('part 3 ends with the offsets = rowPtrs', (await page.$eval(S + ' [data-offsets]', (e) => e.dataset.offsets)) === R.offsets.join(','), R.offsets.join(','));
// part 4: scatter
await goPart(3);
await seek(1000);
b = await beat();
check('part 4 lands the pairs in the stable scatter\'s order', b.order === gather(R.pairs, R.scatter.order).map((p) => p.slot).join(','), b.order);
const xs = await page.$$eval(S + ' g[data-pair] rect', (r) => r.map((x) => Number(x.getAttribute('x'))));
const ordered = R.scatter.order.map((i) => xs[i]);
check('in part 4, positions along the strip follow the expert order', ordered.every((x, k) => k === 0 || x > ordered[k - 1]), JSON.stringify(ordered));
if (out) await (await page.$(S)).screenshot({ path: out + '-scene-4.png' });
// part 5: padded slices and the GEMMs
await goPart(4);
await seek(1000);
b = await beat();
check('part 5 pads slices to the tile and skips the empty expert', b.starts === R.slices.starts.join(',') && Number(b.pads) === R.slices.pads && b.skip === R.empty.join(','), JSON.stringify(b));
const padBoxes = await page.$$eval(S + ' g[data-pads]:not([data-beat]) rect', (r) => r.length);
check(`part 5 draws ${R.slices.pads} pad slots`, padBoxes === R.slices.pads, String(padBoxes));
if (out) await (await page.$(S)).screenshot({ path: out + '-scene-5.png' });
// part 6: back in token order, combined
await goPart(5);
await seek(1000);
b = await beat();
check('part 6 returns every output to its token\'s slot and combines pairs', b.back === R.pairs.map((p) => p.slot).join(',') && (await page.$eval(S + ' [data-combine]', (e) => e.dataset.combine)) === '1');
const combineText = await page.$$eval(S + ' [data-combine] text', (t) => t.map((x) => x.textContent));
check('part 6 labels each token with its two experts', JSON.stringify(combineText) === JSON.stringify(R.tokens ? Array.from({ length: R.tokens }, (_, k) => `t${k} = E${R.pairs[2 * k].expert} + E${R.pairs[2 * k + 1].expert}`) : []), JSON.stringify(combineText));
if (out) await (await page.$(S)).screenshot({ path: out + '-scene-6.png' });
// captions switch is remembered
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), S);
await pause(100);
check('captions on: the switch is remembered in this browser', (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');
check('captions on: part 6 shows its script', (await caption()) === scripts[5], String(await caption()));
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), S);
// play runs on (sound off so no audio is needed)
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), S);
await goPart(0);
await page.evaluate((S) => document.querySelector(S + ' button.primary').click(), S);
await pause(1500);
const pos = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('Play advances the animation with the sound off', pos > 0, String(pos));
await page.evaluate((S) => document.querySelector(S + ' button.primary').click(), S);

// ---------------------------------------------------------------- notes, table, links
const placement = await page.$$eval('.prose > aside.note', (a) => a.map((x) => x.previousElementSibling?.tagName ?? 'none'));
check('every margin note follows a paragraph', placement.every((t) => t === 'P'), placement.join(','));
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
for (const [what, needle] of [
  ['HYB removed (A5, B3)', 'A5, B3'],
  ['Sliced ELL and LRB (A6, B4)', 'A6, B4'],
  ['preprocess (A8)', 'A8)'],
  ['llama.cpp and mlx-lm (D5 to D7)', 'D5, D6, D7'],
  ['the FLOPS claim unverified (A12)', 'A12'],
])
  check(`a margin note for ${what}`, notes.some((t) => t.includes(needle)), notes.join(' | ').slice(0, 200));
check('the explorer and the scene state what they leave out', notes.filter((t) => t.startsWith('What this model leaves out')).length === 2);
const libs = await page.$$eval('table[data-libraries] tbody tr', (r) => r.map((x) => x.textContent));
check('the libraries table has five rows, with HYB removed in CUDA 11.0', libs.length === 5 && libs.some((t) => t.includes('removed in CUDA 11.0')), JSON.stringify(libs).slice(0, 200));
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
check('the page links back to 17.2', links.includes('#/ch17/17.2'));

await pause(700);
const last = saved.at(-1) ?? '';
check(
  'answers and predictions were saved through PUT /api/progress (intercepted)',
  ['p-book1000', 'p-fig1713', 'q-hybrid-k', 'q-sort-y', 'q-jds-loop', 'q-jds-align', 'q-amortise', 'q-moe-offset', 'q-moe-atomic'].every((id) => last.includes('ch17/17.5/' + id)),
  last.slice(0, 300),
);

// ---------------------------------------------------------------- dark theme and phone width
if (out) {
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  await clickIn(X, "The book's 1000 × 1000");
  await pause(200);
  await (await page.$('section.bench[aria-label="Pad, spill or sort"]')).screenshot({ path: out + '-explorer-dark.png' });
  await goPart(4);
  await seek(600);
  await (await page.$(S)).screenshot({ path: out + '-scene-5-dark.png' });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
}
await page.setViewport({ width: 390, height: 860, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('no horizontal scroll at 390 px', overflow <= 0, String(overflow));
const benchW = await page.$eval('section.bench[aria-label="Pad, spill or sort"]', (e) => e.getBoundingClientRect().width);
check('the explorer fits the phone width', benchW <= 390, String(benchW));
if (out) {
  await (await page.$('section.bench[aria-label="Pad, spill or sort"]')).screenshot({ path: out + '-explorer-narrow.png' });
  await (await page.$(S)).screenshot({ path: out + '-scene-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
