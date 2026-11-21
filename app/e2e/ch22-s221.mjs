// Browser check for section 22.1: the question before the first explorer, "Which scan for this
// hardware?" (N, P, the charging rule, the warp width, the presets, the winner, the crossover
// readout, the plot), "Bin capacity and the overflow list" (distribution, atoms, S, the
// capacity slider and its three buttons, memory, overflow, work bars, the bin map), the
// other four questions, the Today note, links and the page at 390 px.
// Expected values come from the models: src/sim/binning.ts is imported here directly;
// src/sim/algoselect.ts imports its neighbours without file extensions, which Node cannot
// resolve on its own, so it is imported inside the page through the app's module server.
// Usage: APP_URL=http://localhost:<port> node e2e/ch22-s221.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { BOOK_SHARE, bestCapacity, binWith, capacityForShare, countsFor, work } from '../src/sim/binning.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const int = (v) => Math.round(v).toLocaleString('en-US');
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const A = 'section.bench[aria-label="Which scan for this hardware?"]';
const B = 'section.bench[aria-label="Bin capacity and the overflow list"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label || b.textContent.trim().startsWith(label));
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(90);
  return done;
};
const setInput = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(90);
};
const setSelect = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(90);
};
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerChoice = async (part, option) => {
  const h = await q(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return ok && norm(await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};

await goto(page, 'ch22/22.1');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
await page.waitForSelector(A);
const model = (n, p, charge, warp) =>
  page.evaluate(async (n, p, charge, warp) => {
    const m = await import('/src/sim/algoselect.ts');
    const r = m.compare(n, p, charge, warp);
    return {
      rows: Object.fromEntries(m.ALGOS.map((a) => [a, { time: r[a].time, adds: r[a].additions }])),
      net: m.networkWinner(r),
      bkUpTo: m.brentKungUpTo(n, charge, warp),
      ksFrom: m.koggeStoneFrom(n, charge, warp),
      points: m.curve(n, charge, warp).length,
    };
  }, n, p, charge, warp);
const table = () => page.$$eval(A + ' tbody tr', (rows) => Object.fromEntries(rows.map((r) => [r.dataset.row, { time: Number(r.dataset.time), adds: Number(r.dataset.adds) }])));
const agrees = (got, want) => Object.keys(want).every((k) => got[k] && got[k].time === want[k].time && got[k].adds === want[k].adds);

// ---- the question comes first, before the explorer
const order = await page.evaluate((A) => {
  const qEl = [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('can keep only 32 adders busy'));
  const bench = document.querySelector(A);
  return !!qEl && !!(qEl.compareDocumentPosition(bench) & Node.DOCUMENT_POSITION_FOLLOWING);
}, A);
check('q-scan-units sits before the scan explorer', order);
const m1024 = await model(1024, 32, 'adds', 32);
let v = await answerChoice('can keep only 32 adders busy', 'Kogge-Stone, because it takes 10 steps');
check('q-scan-units: Kogge-Stone is wrong, with the reason', v && v.includes('Not quite.') && v.includes('up to 32 rounds'), v);
v = await answerChoice('can keep only 32 adders busy', 'Brent-Kung, despite');
check(`q-scan-units: Brent-Kung is right, ${m1024.rows['brent-kung'].time} against ${m1024.rows['kogge-stone'].time}, Kogge-Stone only from ${m1024.ksFrom}`,
  v && v.includes('Correct.') && v.includes(`${m1024.rows['kogge-stone'].time} in all`) && v.includes(`: ${m1024.rows['brent-kung'].time}.`) && v.includes(`from ${m1024.ksFrom}`), v);

// ---- explorer 1: defaults are the question's case
check('default: N = 1,024, P = 32', (await text(A + ' [data-value="n"]')) === '1,024' && (await text(A + ' [data-value="p"]')) === '32');
let t = await table();
check('N = 1024, P = 32, additions: Brent-Kung 71, Kogge-Stone 289, as the model says', agrees(t, m1024.rows) && t['brent-kung'].time === 71 && t['kogge-stone'].time === 289, JSON.stringify(t));
check('the winner chip says Brent-Kung, 71 against 289', (await attr(A + ' [data-readout="winner"]', 'data-network')) === 'brent-kung' && (await text(A + ' [data-value="network-times"]')).includes("71 time units against Kogge-Stone's 289"));
check(`the crossover readout: Brent-Kung up to ${m1024.bkUpTo}, Kogge-Stone from ${m1024.ksFrom}`,
  (await text(A + ' [data-value="bk-upto"]')) === int(m1024.bkUpTo) && (await text(A + ' [data-value="ks-from"]')) === int(m1024.ksFrom) && m1024.bkUpTo === 511 && m1024.ksFrom === 768);
check('the coarsened scan is the fastest of the three here (68)', (await attr(A + ' [data-readout="winner"]', 'data-overall')) === 'coarsened' && t.coarsened.time === 68);
const curves = await page.$$eval(A + ' polyline[data-curve]', (ps) => ps.map((p) => [p.dataset.curve, p.getAttribute('points').trim().split(/\s+/).length]));
check('the plot draws three curves with one point per power of two of P', curves.length === 3 && curves.every(([, k]) => k === m1024.points), JSON.stringify(curves));
check('the crossover is marked on the plot', (await page.$(A + ' [data-mark="ks-from"]')) !== null && (await page.$(A + ' [data-region="bk"]')) !== null);
await shot('scan-default', A);

// P = 1024: Kogge-Stone wins, 10 against 19
await setInput(A + ' input[aria-label="Execution units, as a power of two"]', 10);
t = await table();
const m1024b = await model(1024, 1024, 'adds', 32);
check('P = 1,024: Kogge-Stone 10 against Brent-Kung 19, and wins', (await text(A + ' [data-value="p"]')) === '1,024' && agrees(t, m1024b.rows) && t['kogge-stone'].time === 10 && t['brent-kung'].time === 19 && (await attr(A + ' [data-readout="winner"]', 'data-network')) === 'kogge-stone', JSON.stringify(t));
// P = 512: a tie
await setInput(A + ' input[aria-label="Execution units, as a power of two"]', 9);
check('P = 512: the two networks tie', (await attr(A + ' [data-readout="winner"]', 'data-network')) === 'tie' && (await text(A + ' [data-value="network-winner"]')) === 'a tie');
await shot('scan-512', A);

// whole warps, warp 64, a small N
await press(A, 'whole warps');
const widthOptions = await page.$$eval(A + ' select[aria-label="warp width"] option', (os) => os.map((o) => o.value));
check('the warp width is enabled under whole-warp charging and offers 64', !(await page.$eval(A + ' select[aria-label="warp width"]', (s) => s.disabled)) && widthOptions.includes('w64'), widthOptions.join(','));
await setSelect(A + ' select[aria-label="warp width"]', 'w64');
await setInput(A + ' input[aria-label="Elements, as a power of two"]', 6);
await setInput(A + ' input[aria-label="Execution units, as a power of two"]', 3);
t = await table();
const mw = await model(64, 8, 'warps', 64);
check('N = 64, P = 8, whole warps of 64: the table agrees with the model', (await text(A + ' [data-value="n"]')) === '64' && (await text(A + ' [data-value="p"]')) === '8' && agrees(t, mw.rows), JSON.stringify({ t, want: mw.rows }));
check('  and the crossover readout too', (await attr(A + ' [data-readout="crossover"]', 'data-bk-upto')) === String(mw.bkUpTo ?? '') && (await attr(A + ' [data-readout="crossover"]', 'data-ks-from')) === String(mw.ksFrom ?? ''));
check('  and the charge note explains whole warps', (await text(A + ' [data-readout="charge"]')).includes('charged all its lanes'));

// a preset
check('back to counting additions', await press(A, 'its additions'));
await press(A, '16,384 elements, 1,024 units');
t = await table();
const mbig = await model(16384, 1024, 'adds', 64);
check('preset 16,384 elements on 1,024 units: the table agrees with the model', (await text(A + ' [data-value="n"]')) === '16,384' && (await text(A + ' [data-value="p"]')) === '1,024' && agrees(t, mbig.rows), JSON.stringify({ t, want: mbig.rows }));
await press(A, '1,024 elements, 32 units');
check('preset 1,024 elements on 32 units brings back 71 against 289', agrees(await table(), m1024.rows));

// ---- explorer 2: bin capacity
const capSel = B + ' input[aria-label="Bin capacity"]';
const counts0 = countsFor('pocket', 4096);
const full0 = Math.max(...counts0);
const readBins = async () => ({
  capacity: Number(await text(B + ' [data-value="capacity"]')),
  overflow: Number((await text(B + ' [data-value="overflow"]')).replace(/,/g, '')),
  slots: Number((await text(B + ' [data-value="slots"]')).replace(/,/g, '')),
  dummies: Number((await text(B + ' [data-value="dummies"]')).replace(/,/g, '')),
  share: Number(await attr(B + ' [data-readout="overflow"]', 'data-share')),
  device: Number(await attr(B + ' [data-bar="device"]', 'data-units')),
  host: Number(await attr(B + ' [data-bar="host"]', 'data-units')),
  outlined: (await page.$$(B + ' rect[data-over]')).length,
  bottleneck: await text(B + ' [data-readout="bottleneck"]'),
});
const agreesBins = (got, counts, c, S) => {
  const b = binWith(counts, c);
  const w = work(counts, c, S);
  return got.capacity === c && got.overflow === b.overflow && got.slots === b.slots && got.dummies === b.dummies && Math.abs(got.share - b.overflowShare) < 1e-12 &&
    got.device === Math.round(w.deviceTime) && got.host === Math.round(w.hostTime) && got.outlined === b.overflowingBins;
};
let g = await readBins();
check(`default: one dense pocket, 4,096 atoms, sized for the fullest bin (${full0}), nothing overflows`, (await text(B + ' [data-value="atoms"]')) === '4,096' && g.capacity === full0 && g.overflow === 0 && g.share === 0 && agreesBins(g, counts0, full0, 20), JSON.stringify(g));
check('  and most of the slots are dummies, and the chip says within 3%', g.dummies > g.slots / 2 && (await text(B + ' [data-readout="overflow"]')).includes("within the book's 3%"));
check('  and the full-sizing readout says this is that sizing', (await text(B + ' [data-readout="full-sizing"]')).includes('This is the sizing for the fullest bin'));
await shot('bins-full', B);

// reduce the capacity: overflow appears
const cMid = Math.floor(full0 / 3);
await setInput(capSel, cMid);
g = await readBins();
check(`capacity ${cMid}: a non-zero overflow share, as the model says`, g.overflow > 0 && g.share > 0 && agreesBins(g, counts0, cMid, 20), JSON.stringify(g));
check('  and fewer slots than sizing for the fullest bin', g.slots < 256 * full0 && (await text(B + ' [data-readout="full-sizing"]')).includes('times as many'));

// the book's 3%
const c3 = capacityForShare(counts0, BOOK_SHARE);
check(`the 3% button shows ${c3}`, await press(B, `smallest under 3% (${c3})`));
g = await readBins();
check(`smallest capacity under 3%: ${c3}, share at most 3%`, g.capacity === c3 && g.share <= BOOK_SHARE && g.share > 0 && agreesBins(g, counts0, c3, 20), JSON.stringify(g));

// too small: the host becomes the bottleneck
await setInput(capSel, 5);
g = await readBins();
const w5 = work(counts0, 5, 20);
check('capacity 5: the host pass takes longer than the kernel, and the chip says so', w5.hostTime > w5.deviceTime && agreesBins(g, counts0, 5, 20) && g.bottleneck.includes("host's serial pass takes longer"), JSON.stringify(g));
await shot('bins-small', B);

// the lowest total
const best20 = bestCapacity(counts0, 20);
check(`the lowest-total button shows ${best20}`, await press(B, `lowest total time (${best20})`));
g = await readBins();
check(`lowest total time at S = 20: capacity ${best20}, below both other choices`, g.capacity === best20 && agreesBins(g, counts0, best20, 20) &&
  work(counts0, best20, 20).serial < work(counts0, full0, 20).serial && work(counts0, best20, 20).serial <= work(counts0, c3, 20).serial, JSON.stringify(g));

// a faster device: overflow becomes dearer
await setInput(B + ' input[aria-label="Device speed-up over the host"]', 30);
const best1000 = bestCapacity(counts0, 1000);
check('S slider at its top reads 1000', (await text(B + ' [data-value="speedup"]')) === '1000');
check(`at S = 1000 the lowest total moves up to ${best1000}`, best1000 > best20 && (await press(B, `lowest total time (${best1000})`)) && (await readBins()).capacity === best1000);
g = await readBins();
check('  and the readout agrees with the model at S = 1000', agreesBins(g, counts0, best1000, 1000), JSON.stringify(g));

// another distribution and size
await press(B, 'uniform');
await setInput(B + ' input[aria-label="Atoms"]', 8192);
const countsU = countsFor('uniform', 8192);
const fullU = Math.max(...countsU);
check(`uniform, 8,192 atoms: the fullest-bin button shows ${fullU}`, (await text(B + ' [data-value="atoms"]')) === '8,192' && (await press(B, `fullest bin (${fullU})`)));
g = await readBins();
check('  and sized for it nothing overflows, as the model says', g.overflow === 0 && agreesBins(g, countsU, fullU, 1000), JSON.stringify(g));
await press(B, 'clustered');
const countsC = countsFor('clustered', 8192);
const cC = Math.min(fullU, Math.max(...countsC));
g = await readBins();
check('clustered, 8,192 atoms: the capacity carries over (clamped) and the readout agrees', agreesBins(g, countsC, cC, 1000), JSON.stringify(g));
await shot('bins-clustered', B);

// ---- the other questions
v = await answerChoice('Can you use it?', 'Yes, as they are');
check('q-sort-logits: raw bits is wrong, with the negative-float reason', v && v.includes('Not quite.') && v.includes('Negative floats'), v);
v = await answerChoice('Can you use it?', 'Yes, after mapping');
check('q-sort-logits: the transform is right; the explanation names CUB and chapter 14', v && v.includes('Correct.') && v.includes('CUB') && v.includes('chapter 14'), v);
v = await answerChoice('plus an overflow list cost?', 'Nothing at all');
check('q-overflow: dropping atoms is wrong (p. 531)', v && v.includes('Not quite.') && v.includes('p. 531'), v);
v = await answerChoice('plus an overflow list cost?', 'A short serial pass');
check('q-overflow: the serial pass is right; the explanation points at the host bottleneck', v && v.includes('Correct.') && v.includes('host the bottleneck'), v);
v = await answerChoice('which two does cutoff summation trade', 'Accuracy, given up a little');
check('q-cutoff-axis: accuracy for steps; the explanation points to 22.4', v && v.includes('Correct.') && v.includes('22.4'), v);
v = await answerChoice('Why a measured threshold', 'It is a hardware limit');
check('q-two-kernels: a hardware limit is wrong, with the M3 reason', v && v.includes('Not quite.') && v.includes('M3'), v);
v = await answerChoice('Why a measured threshold', 'Which algorithm wins depends');
check('q-two-kernels: hardware and format decide; the explanation names the dispatch function', v && v.includes('Correct.') && v.includes('dispatch function'), v);

// ---- notes, links
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const today = norm(await page.$eval('.prose .note.today', (e) => e.textContent));
check('the Today note gives the llama.cpp paths, the 8-token limit, the RTX 4090 tuning and the check date', today.includes('ggml-cuda/') && today.includes('ggml-metal/') && today.includes('mul_mv_ext') && today.includes('tuned on RTX 4090') && today.includes('2026-11-20'), today);
check('two "what this model leaves out" notes, one per explorer', (await page.$$('.prose .note.caveat')).length === 2);
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch22/22.2', '#/ch22/22.4', '#/ch22/labs', '#/ch14/14.4']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch22/22.1');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone-scan', A);
await shot('phone-bins', B);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
