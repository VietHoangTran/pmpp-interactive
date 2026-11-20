// Browser check for section 21.5 (cutoff binning): the two predictions, the explorer's
// cutoff view (neighbourhood counts, both radii, clicking blocks and points, the staged-atom
// mask, capacity, doubling the box, the 3D readout of the book's case), and the questions.
// Expected values come from src/sim/cutoff.ts, imported inside the page through the app's
// module server. Progress is neither read from nor written to the real file.
// Usage: node e2e/ch21-s215.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1,"sections":{},"answers":{},"review":{}}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const B = 'section.bench[aria-label="Cutoff and bins"]';
const V = B + ' [data-view="cutoff"]';
const data = () => page.$eval(V, (e) => ({ ...e.dataset }));
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, B, label);
  await pause(150);
  return ok;
};
const setRange = async (aria, v) => {
  await page.$eval(`${V} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(200);
};
const setSelect = async (label, v) => {
  await page.evaluate((scope, label, v) => {
    const s = [...document.querySelectorAll(scope + ' label.field')].find((l) => l.querySelector('select') && l.textContent.trim().startsWith(label)).querySelector('select');
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(v));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, V, label, v);
  await pause(200);
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerChoice = async (needle, index) => {
  const q = await question(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
const typeAnswer = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await question(needle)).evaluate((e) => e.textContent);
};
const model = (fn, arg) => page.evaluate(async (src, arg) => {
  const m = await import('/src/sim/cutoff.ts');
  return new Function('m', 'arg', 'return (' + src + ')(m, arg)')(m, arg);
}, fn.toString(), arg);

await goto(page, 'ch21/21.5');
await page.waitForSelector(V);

// ---- predictions and the first question
let t = await typeAnswer('By what factor does the work of direct summation grow?', 64);
check('p-volume reveal: 64, and 8 for the cutoff', t.includes('You predicted 64') && t.includes('64.') && t.includes('eight-fold'), t.slice(0, 200));
t = await answerChoice('Why not give each GPU thread one atom and do the same?', 0);
check('q-atom-centric: the scatter answer is accepted', t.includes('Correct.'));
t = await typeAnswer('How many bins does the 2D neighbourhood contain?', 25);
check('p-bins reveal: 21, and why 25 is too many', t.includes('21.') && t.includes('Twenty-five takes the whole 5 × 5 square'), t.slice(0, 300));

// ---- the neighbourhood
const exp = await model((m) => {
  const R = (c, mode) => m.superRadius(c * 4, 4, 2, mode) / 4;
  return {
    at2: m.counts(2, 2),
    full: m.counts(R(1.3, 'full'), 2),
    book2d: m.counts(R(3, 'half'), 2),
    r: Number(R(2 - Math.SQRT2 / 2, 'half').toFixed(3)),
    at24: m.counts(R(1.7, 'half'), 2),
  };
});
await press('R = 2 bin widths');
let d = await data();
check('R = 2 bin widths: 21 bins, 5 fully inside, 16 partly (Fig. 21.14\'s 9 + 12 is not what R = 2 gives)',
  Number(d.list) === 21 && Number(d.full) === 5 && Number(d.partial) === 16 && exp.at2.total === 21 && exp.at2.full === 5 && Number(d.radius) === exp.r && exp.r === 2, JSON.stringify(d));
check('the readout states R and the counts', (await text(V + ' [data-readout="list"]')).includes('2.00 bin widths') && (await text(V + ' [data-readout="list"]')).includes('21 bins: 5 fully inside, 16 partly'));
check('the drawing shades 21 bins (5 fully covered) around the chosen block',
  (await page.$$(V + ' [data-cover="full"]')).length === 5 && (await page.$$(V + ' [data-cover="partial"]')).length === 16);
await setRange('cutoff in bin widths', 1.3);
await setSelect('Super-circle radius', 'full');
d = await data();
check('Fig. 21.13\'s radius (the whole diagonal) gives a longer list', Number(d.list) === exp.full.total && exp.full.total > 21, JSON.stringify(d));
await setSelect('Super-circle radius', 'half');
await setRange('cutoff in bin widths', 3);
d = await data();
check('cutoff 3 bin widths (12 Å on 4 Å bins): the 2D list of the book\'s case has 61 bins', Number(d.list) === exp.book2d.total && exp.book2d.total === 61, JSON.stringify(d));
await setRange('cutoff in bin widths', 1.7);
d = await data();
check('cutoff 1.70 (R ≈ 2.41, q-fully\'s case): 25 bins, 9 fully inside', Number(d.list) === exp.at24.total && Number(d.full) === exp.at24.full && exp.at24.total === 25 && exp.at24.full === 9, JSON.stringify(d));
await press('R = 2 bin widths');
if (out) await page.screenshot({ path: out + '-r2.png' });

// ---- the chosen block, thread and staged atom
const st = await model((m) => {
  const w = m.makeWorld(8, 8, 4, 128, 1);
  const cap = m.capacityFor(w, 0.03);
  const bin = m.binAtoms(w, cap);
  const cutoff = (2 - Math.SQRT2 / 2) * 4;
  const R = m.superRadius(cutoff, 4, 2, 'half') / 4;
  const walk = m.blockWalk(w, bin, cutoff, R, 3, 3);
  const t5 = walk.list.reduce((s, x) => { for (const row of x.outcome) { s.scanned++; if (row[5] === 'accept') s.accepted++; } return s; }, { scanned: 0, accepted: 0 });
  const g = m.runGrid(w, bin, cutoff, R, false);
  const walk2 = m.blockWalk(w, bin, cutoff, R, 6, 1);
  const split = (ww) => { let real = 0, n = 0; for (const e of walk.list) e.slots.forEach((s, k) => { if (s >= 0) { real++; if (m.acceptMasks(e, k, ww).some(m.diverges)) n++; } }); return { real, n }; };
  return { split32: split(32), cap, overflow: bin.overflow.length, dummy: bin.dummy, tested: walk.tested, accepted: walk.accepted, t5, tests: g.tests, direct: g.directPairs, max: m.maxOccupancy(w), tested2: walk2.tested, inBox2: walk2.list.filter((x) => x.inBox).length };
});
d = await data();
check('default capacity is the smallest with under 3 % overflow', Number(d.capacity) === st.cap && Number(d.overflow) === st.overflow && st.overflow <= 0.03 * 128, JSON.stringify(d));
check('block (3, 3): the model\'s tests and accepted pairs', (await text(V + ' [data-readout="block"]')).startsWith(`${st.tested.toLocaleString('en-US')} tests`) && (await text(V + ' [data-readout="block"]')).includes(`${st.accepted} accepted`));
check('the chosen thread\'s scanned and accepted atoms', Number(d.threadScanned) === st.t5.scanned && Number(d.threadAccepted) === st.t5.accepted, JSON.stringify({ d, t5: st.t5 }));
check('whole-grid pairs: direct against binned', Number(d.direct) === st.direct && Number(d.tests) === st.tests);
check('the mask panel opens on an atom that splits warp 0', !!(await page.$(V + ' [data-warp="0"][data-diverges="yes"]')));
const lanes = await page.$$eval(V + ' [data-warp="0"] svg', (x) => x.length);
check('a block of 4 × 4 points is one warp of 16 lanes here', lanes === 16);
check('the mask panel counts, from the model, how many staged real atoms split a warp',
  Number((await page.$eval(V + ' [data-mask]', (e) => e.dataset.split))) === st.split32.n && Number((await page.$eval(V + ' [data-mask]', (e) => e.dataset.real))) === st.split32.real &&
  (await text(V + ' [data-readout="split"]')).includes(`${st.split32.n} of ${st.split32.real} staged real atoms split at least one warp of 32 lanes`), JSON.stringify(st.split32));
// group width: 8 × 8-point bins are 64 threads, two warps of 32 or one wavefront of 64
await setSelect('Bin side', 8);
check('64 threads at 32 lanes: two warps of 32', (await page.$$(V + ' [data-warp]')).length === 2 && (await page.$$eval(V + ' [data-warp="1"] svg', (x) => x.length)) === 32);
await setSelect('Group width', '64');
check('64 threads at 64 lanes (AMD Instinct): one wavefront of 64', (await page.$$(V + ' [data-warp]')).length === 1 && (await page.$$eval(V + ' [data-warp="0"] svg', (x) => x.length)) === 64 &&
  (await text(V + ' [data-mask]')).includes('wavefront 0') && Number(await page.$eval(V + ' [data-mask]', (e) => e.dataset.groupWidth)) === 64);
const appleOpt = await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [data-mask] option')].find((o) => o.value === 'apple').textContent, V);
check('the Apple option names a width recorded on this Mac, or says none is recorded', /Apple: \d+ lanes, from lab [\d.]+ on this Mac/.test(appleOpt) || appleOpt.includes('not recorded here yet'), appleOpt);
await setSelect('Group width', '32');
await setSelect('Bin side', 4);
await press('R = 2 bin widths');
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [data-mask] .seg button')].find((b) => b.textContent.includes('dummy'))?.click(), V);
await pause(150);
check('a dummy slot: tested by every lane, adds 0', (await text(V + ' [data-mask]')).includes('dummy: tested, adds 0'));
if (out) await page.screenshot({ path: out + '-mask.png' });
// click another bin: the block moves
const where = await page.$eval(V + ' [data-bin="6,1"]', (e) => { e.scrollIntoView({ block: 'center' }); return null; });
void where;
await pause(150);
const r61 = await page.$eval(V + ' [data-bin="6,1"]', (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
await page.mouse.click(r61.x, r61.y);
await pause(200);
check('clicking bin (6, 1) chooses that block; neighbours outside the box are skipped',
  (await text(V)).includes('absolute coordinates for block (6, 1)') && (await text(V + ' [data-readout="block"]')).startsWith(`${st.tested2.toLocaleString('en-US')} tests`) && (await text(V + ' [data-readout="list"]')).includes(`${21 - st.inBox2} of them lie outside the box`));
await page.$eval(V + ' input[type=checkbox]', (e) => e.click());
await pause(120);
check('absolute coordinates: offset (2, 1) of block (6, 1) is bin (8, 2), outside, and (−1, −1) is (5, 0)',
  await page.evaluate((sel) => {
    const bs = [...document.querySelectorAll(sel + ' [aria-label="The list"] button')].map((b) => ({ t: b.textContent, dis: b.disabled }));
    return bs.some((b) => b.t === '(8, 2)' && b.dis) && bs.some((b) => b.t === '(5, 0)' && !b.dis);
  }, V));

// ---- capacity
await setRange('bin capacity', st.max);
d = await data();
check('capacity at the largest bin: no overflow, more dummies', Number(d.overflow) === 0 && Number(d.dummy) === 64 * st.max - 128, JSON.stringify(d));
await setRange('bin capacity', 1);
d = await data();
const cap1 = await model((m) => { const w = m.makeWorld(8, 8, 4, 128, 1); const b = m.binAtoms(w, 1); return { overflow: b.overflow.length, dummy: b.dummy }; });
check('capacity 1: the model\'s overflow and dummy counts, and the readout says over 3 %',
  Number(d.overflow) === cap1.overflow && Number(d.dummy) === cap1.dummy && (await text(V + ' [data-readout="capacity"]')).includes('over the book\'s 3 %'), JSON.stringify({ d, cap1 }));

// ---- double the box
const beforeDirect = Number((await data()).direct);
await press('Double the box');
d = await data();
check('double the box: 16 × 16 bins, 512 atoms, direct pairs × 16, the same capacity',
  Number(d.bins) === 16 && Number(d.atoms) === 512 && Number(d.direct) === beforeDirect * 16 && Number(d.capacity) === 1, JSON.stringify(d));
t = await text(V + ' [data-readout="doubled"]');
const m = /binned tests ×([\d.]+) \(tests per point ×([\d.]+)\)/.exec(t);
const dbl = await model((m) => {
  const R = 2;
  const cutoff = (2 - Math.SQRT2 / 2) * 4;
  const a = m.makeWorld(8, 8, 4, 128, 1);
  const b = m.makeWorld(16, 16, 4, 512, 1);
  const ga = m.runGrid(a, m.binAtoms(a, 1), cutoff, R, false);
  const gb = m.runGrid(b, m.binAtoms(b, 1), cutoff, R, false);
  return { tests: (gb.tests / ga.tests).toFixed(1), perPoint: ((gb.tests / 4096) / (ga.tests / 1024)).toFixed(2) };
});
check('against the box before: points ×4, atoms ×4, direct ×16; binned tests about ×4 (the model\'s ratio), tests per point about the same',
  t.includes('points ×4, atoms ×4, direct pairs ×16') && m && m[1] === dbl.tests && m[2] === dbl.perPoint && Number(m[1]) > 4 && Number(m[1]) < 5 && Number(m[2]) < 1.2, JSON.stringify({ t, dbl }));
if (out) await page.screenshot({ path: out + '-doubled.png' });
await press('Back to 8 × 8 bins');
check('and back', Number((await data()).bins) === 8);

// ---- 3D readout
const b3 = await model((m) => {
  const n = (c, b, mode) => m.counts(m.superRadius(c, b, 3, mode) / b, 3);
  return { h: n(12, 4, 'half'), f: n(12, 4, 'full'), s: Math.round(m.acceptedShare(12, 4, 3) * 100), h8: n(8, 4, 'half') };
});
t = await text(V + ' [data-readout="book3d"]');
check('3D, the book\'s case: 461 bins (147 fully inside), 667 with the whole diagonal, about 25 %',
  t.includes(`${b3.h.total} bins (${b3.h.full} fully inside)`) && b3.h.total === 461 && b3.h.full === 147 && t.includes(`whole diagonal: ${b3.f.total}`) && b3.f.total === 667 && t.includes(`about ${b3.s} %`) && b3.s === 25, t);
await setSelect('Cutoff', 8);
check('an 8 Å cutoff recomputes the 3D list', (await text(V + ' [data-readout="book3d"]')).includes(`${b3.h8.total} bins`));

// ---- the scene is there; the remaining questions
check('the binned-block scene is on the page', !!(await page.$('section.scene[aria-label="A block walks its neighbourhood"]')));
t = await answerChoice('a super circle of radius 2.4 b', 0);
check('q-fully (R = 2.4 b): the figure\'s 21 with 9 is marked wrong and explained', t.includes('Not quite.') && t.includes('2.12 b') && t.includes('25'));
t = await answerChoice('Exercise 5. How does control divergence arise', 0);
check('q-ex5: accepted; the explanation sends the reader to the computed count, not a claim', t.includes('Correct.') && t.includes('the mask panel counts them') && !t.includes('most bins split most warps'));
t = await answerChoice('Why do the atoms move from constant memory to global memory', 0);
check('q-global: accepted', t.includes('Correct.'));
t = await answerChoice('raising it until nothing overflows', 0);
check('q-capacity: accepted', t.includes('Correct.'));
t = await typeAnswer('how many slots is that?', 1528);
check('q-vllm-pad: 1,528 slots', t.includes('Correct.') && t.includes('1,528'));
check('links to chapter 9\'s summary and to sections 22.1 and 22.2',
  !!(await page.$('a[href="#/ch09/wrap"]')) && !!(await page.$('a[href="#/ch22/22.1"]')) && !!(await page.$('a[href="#/ch22/22.2"]')));
const tg = await page.evaluate(() => [...document.querySelectorAll('.note.today')].map((n) => n.textContent).find((x) => x.includes('threadgroup_barrier')) ?? '');
check('the Metal note reads threadgroup memory from the device report (or says how to get it), not a typed number',
  /your Mac reports [\d,]+ bytes per threadgroup/.test(tg) || tg.includes('python3 labs/pmpp.py doctor'), tg);

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-light.png' });
}
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('at 400 px the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (e) => /content\/ch21\/narration\/(manifest\.json|[\w-]+\.mp3)/.test(e) || /status of 404/.test(e);
const real = errors.filter((e) => !expected404(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0, real.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
