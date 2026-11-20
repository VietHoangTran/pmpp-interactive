// Browser check for section 11.10: the exercise 7 listing, the "Up-sweep, down-sweep"
// stepper (both inputs, ranges and values, the read/write readout, the Fig. 11.20 heat map
// with both shadings and its pointer readout), the "Inside one warp" explorer, the narrated
// scene "Up-sweep and down-sweep" (scripts, captions, playback, every part's drawing against
// the model) and the questions. Expected values come from src/sim/prefixnet.ts, imported
// here directly. The scene passes with or without content/ch11/narration/manifest.json.
// Usage: node e2e/ch11-s1110.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  EX6,
  FIG116,
  FIG_11_20_PRINTED,
  additions,
  brentKung,
  finalAfter,
  koggeStone,
  needs,
  printedDiffers,
  rangeLabel,
  ranges,
  readWriteOverlap,
  reverseLevels,
  run,
  warpView,
} from '../src/sim/prefixnet.ts';

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (title) => `section.bench[aria-label="${title}"]`;
const S = bench('Up-sweep, down-sweep');
const WB = bench('Inside one warp');
const SC = 'section.scene[aria-label="Up-sweep and down-sweep"]';
const press = async (scope, label, exact = true) => {
  const done = await page.evaluate((scope, label, exact) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => (exact ? b.textContent.trim() === label : b.textContent.trim().includes(label)));
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label, exact);
  await pause(90);
  return done;
};
const setRange = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(120);
};
const setSelect = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(120);
};
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const cells = (net) => page.$$eval(`${S} svg[data-net="${net}"] g[data-part="cell"]`, (gs) => gs.map((g) => ({ label: g.dataset.label, final: g.dataset.final === '1' })));
const counter = (name) => text(`${S} [data-counter="${name}"]`).then(Number);
const labelOf = (r) => (r.lo === r.hi ? `x${r.lo}` : `x${r.lo}..x${r.hi}`);

await goto(page, 'ch11/11.10');
await page.waitForSelector(S);

// ---- the exercise 7 listing comes from the snippet file
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('brent_kung_scan_kernel')), { timeout: 15000 });
const listing = await page.evaluate(() => {
  const fig = [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('brent_kung_scan_kernel'));
  return {
    status: fig.querySelector('.status')?.textContent ?? '',
    hl: [...fig.querySelectorAll('.ln.hl')].map((l) => l.textContent.trim()),
  };
});
check('listing is labs/ch11/snippets/s1110_brent_kung.cu, emulator- and front-end-checked', listing.status.includes('emulator-checked and front-end-checked'), listing.status);
check('its three barriers are highlighted', listing.hl.length === 3 && listing.hl.every((l) => l === '__syncthreads();'), JSON.stringify(listing.hl));

// ---- stepper: 16 positions, ranges
const bk16 = brentKung(16);
const ks16 = koggeStone(16);
check('stepper starts at step 0 of 7 on 16 positions', (await text(`${S} [data-readout="step"]`)) === 'step 0 of 7');
for (let k = 0; k < 2; k++) await press(S, 'Step');
const ov = await page.$eval(`${S} [data-readout="overlap"]`, (e) => ({ bk: Number(e.dataset.bk), ks: Number(e.dataset.ks), text: e.textContent }));
check('step 2: Brent-Kung writes nothing another adder reads; Kogge-Stone overwrites 12 positions', ov.bk === 0 && ov.ks === readWriteOverlap(ks16.steps[1]).length && ov.text.includes('2 to 13'), JSON.stringify(ov));
for (let k = 2; k < 4; k++) await press(S, 'Step');
let c = await cells('brent-kung');
check('after the reduction tree: 15 additions in 4 steps', (await counter('bk-adds')) === 15 && (await counter('bk-steps')) === 4);
check('after the reduction tree positions 0, 1, 3, 7 and 15 are final', same(c.flatMap((x, i) => (x.final ? [i] : [])), finalAfter(bk16, 4)), JSON.stringify(c));
check('cells show the ranges the model holds', same(c.map((x) => x.label), ranges(bk16)[4].map((r) => (r.lo === r.hi ? `x${r.lo}` : `x${r.lo}..x${r.hi}`))), JSON.stringify(c.map((x) => x.label)));
check('Kogge-Stone is done: 49 additions in 4 steps, all final', (await counter('ks-adds')) === 49 && (await counter('ks-steps')) === 4 && (await cells('kogge-stone')).every((x) => x.final));
check('narration names the final positions', (await text(`${S} [data-readout="narrate"]`)).includes('Final now: 0, 1, 3, 7 and 15.'));
const curRow = () => page.$$eval(`${S} g[data-part="needs-row"][data-current="1"]`, (gs) => gs.map((g) => g.dataset.label));
check('the Initial row of Fig. 11.20 is current after the reduction tree', same(await curRow(), ['Initial']));
await press(S, 'Step');
check('step 5 is reverse level 1: position 7 into 11', (await text(`${S} [data-readout="narrate"]`)).includes('(7 into 11)') && same(await curRow(), ['Level 1']));
await press(S, 'Step');
await press(S, 'Step');
c = await cells('brent-kung');
check('after 7 steps: 26 additions, every position final', (await counter('bk-adds')) === 26 && (await counter('bk-steps')) === 7 && c.every((x) => x.final));
check('Step is disabled at the end', !(await press(S, 'Step')));
if (out) await page.screenshot({ path: out + '-stepper.png' });
await press(S, 'values');
c = await cells('brent-kung');
check('values: Fig. 11.6\'s sixteen values scan to the book\'s printed row', same(c.map((x) => Number(x.label)), run(bk16, FIG116)[7]) && same(c.map((x) => Number(x.label)), [2, 3, 6, 7, 7, 11, 12, 14, 14, 17, 18, 20, 25, 28, 29, 31]));
check('Kogge-Stone\'s values agree', same((await cells('kogge-stone')).map((x) => Number(x.label)), run(ks16, FIG116)[4]));
await press(S, 'Back');
check('Back goes to step 6', (await counter('bk-steps')) === 6);

// ---- the Fig. 11.20 heat map
const levels16 = reverseLevels(16);
const shades = () => page.$$eval(`${S} g[data-part="needs-row"]`, (rows) => rows.map((r) => [...r.querySelectorAll('g[data-part="need-cell"]')].map((g) => Number(g.dataset.shade))));
check('shading from the ranges equals the model\'s needs', same(await shades(), levels16.map((l) => l.needs)));
const pick = async (row, pos) => {
  await page.$eval(`${S} g[data-part="need-cell"][data-row="${row}"][data-pos="${pos}"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(100);
  return text(`${S} [data-readout="needs"]`);
};
let r = await pick(0, 14);
check('pointing at Initial 14 lists 13 (x12..x13), 11 (x8..x11) and 7 (x0..x7)', r.includes('still needs 3 partial sums: 13 (x12..x13), 11 (x8..x11) and 7 (x0..x7)'), r);
r = await pick(1, 13);
check('Level 1, position 13 needs only 11 (x0..x11)', r.includes('still needs 1 partial sum: 11 (x0..x11)'), r);
await press(S, 'as printed');
check('as printed: the book\'s shading', same(await shades(), FIG_11_20_PRINTED.map((x) => [...x])));
const flagged = await page.$$eval(`${S} g[data-part="need-cell"][data-differs="1"]`, (gs) => gs.map((g) => [Number(g.dataset.row), Number(g.dataset.pos)]));
check('three cells are flagged: Initial 9 and 11, Level 1 9', same(flagged, printedDiffers().map((d) => [d.row, d.col])), JSON.stringify(flagged));
r = await pick(0, 9);
check('the readout says the printed cell is shaded one level more than its ranges imply', r.includes('one level more than its ranges imply') && r.includes('7 (x0..x7)'), r);
if (out) await page.screenshot({ path: out + '-printed.png' });
await press(S, 'from the ranges');

// ---- the network cells point into the heat map too
await page.$eval(`${S} svg[data-net="brent-kung"] g[data-part="cell"][data-pos="14"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(100);
r = await text(`${S} [data-readout="needs"]`);
check('pointing at a Brent-Kung cell at step 6 reads Level 2', r.startsWith('Level 2, position 14') && r.includes('13 (x0..x13)'), r);

// ---- exercise 6 on 8 positions
await press(S, '8: exercise 6');
check('switching input restarts at step 0 of 5', (await text(`${S} [data-readout="step"]`)) === 'step 0 of 5');
for (let k = 0; k < 4; k++) await press(S, 'Step');
c = await cells('brent-kung');
check('exercise 6 after reverse level 1: position 5 holds 28', same(c.map((x) => Number(x.label)), run(brentKung(8), EX6)[4]) && Number(c[5].label) === 28, JSON.stringify(c.map((x) => x.label)));
await press(S, 'Step');
check('exercise 6 ends at 4 10 17 18 20 28 33 35', same((await cells('brent-kung')).map((x) => Number(x.label)), [4, 10, 17, 18, 20, 28, 33, 35]));
check('8 positions: no printed shading switch', !(await page.$$eval(`${S} button`, (bs) => bs.some((b) => b.textContent.trim() === 'as printed'))));
await press(S, 'ranges');
const lv8 = reverseLevels(8);
check('8 positions: heat map has the model\'s rows and needs', same(await shades(), lv8.map((l) => l.needs)));

// ---- Play runs by itself
await press(S, 'Restart');
await press(S, 'Play');
await pause(2100);
const playedTo = await counter('bk-steps');
check('Play advances the stepper by itself', playedTo >= 2, String(playedTo));
await press(S, 'Pause');

// ---- inside one warp
const warpRows = (net) => page.$$eval(`${WB} svg[data-net="${net}"] g[data-part="lane-row"]`, (gs) => gs.map((g) => ({ adds: Number(g.dataset.adds), idle: Number(g.dataset.idle), shown: g.dataset.shown === '1', filled: [...g.querySelectorAll('rect')].filter((x) => x.getAttribute('fill') !== 'none').length })));
const warpTable = () => page.$$eval(`${WB} tbody tr`, (rows) => Object.fromEntries(rows.map((x) => [x.dataset.row, { adds: Number(x.dataset.adds), steps: Number(x.dataset.steps), slots: Number(x.dataset.slots), idle: Number(x.dataset.idle), spent: Number(x.querySelector('[data-cell="spent"]').textContent) }])));
for (const W of [32, 64]) {
  if (W === 64) await setSelect(`${WB} select[aria-label="warp width"]`, 'w64');
  const ks = warpView('kogge-stone', W);
  const bk = warpView('brent-kung', W);
  await setRange(`${WB} input[aria-label="Steps elapsed"]`, ks.finishesAt);
  const kr = await warpRows('kogge-stone');
  const br = await warpRows('brent-kung');
  check(`${W} lanes: lane rows are the model's adders, filled lane by lane`, same(kr.map((x) => x.adds), ks.steps.map((s) => s.adds)) && same(br.map((x) => x.adds), bk.steps.map((s) => s.adds)) && kr.concat(br).every((x) => x.filled === x.adds && x.adds + x.idle === W));
  const st = await page.$$eval(`${WB} [data-part="status"]`, (s) => s.map((x) => x.textContent.trim()));
  check(`${W} lanes: at step ${ks.finishesAt} Kogge-Stone is done and Brent-Kung is not`, st[0] === `done after ${ks.finishesAt} steps` && st[1] === `step ${ks.finishesAt} of ${bk.finishesAt}`, JSON.stringify(st));
  check(`${W} lanes: only rows up to step ${ks.finishesAt} are drawn solid`, br.filter((x) => x.shown).length === ks.finishesAt);
  const tb = await warpTable();
  check(`${W} lanes: table agrees with the model (${ks.slots} against ${bk.slots} lane slots)`, tb['kogge-stone'].slots === ks.slots && tb['brent-kung'].slots === bk.slots && tb['kogge-stone'].adds === ks.additions && tb['brent-kung'].adds === bk.additions && tb['brent-kung'].idle === bk.idleSlots && tb['brent-kung'].spent === ks.finishesAt * W, JSON.stringify(tb));
  const says = await text(`${WB} [data-readout="warp"]`);
  check(`${W} lanes: the readout states the saving and the cost`, says.includes(`saves ${ks.additions - bk.additions} additions`) && says.includes(`${bk.slots} lane slots on it against ${ks.slots}`), says);
}
if (out) await page.screenshot({ path: out + '-warp.png' });

// ---- the narrated scene
const N = 6;
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let k = 1; k <= N; k++) scripts.push(strip(await (await fetch(`${BASE}/content/ch11/narration/bk-sweeps-${k}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}+−#*]/.test(s)));
const res = await fetch(BASE + '/content/ch11/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(200);
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await text(SC + ' footer')).includes('No narration manifest found'));
const capBox = () => page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('captions switch exists and is off by default', !(await (await capBox()).evaluate((e) => e.checked)));
const seek = (v) => setRange(SC + ' input[type=range]', v);
const part = async (k) => {
  await page.evaluate((sel, k) => document.querySelectorAll(sel + ' .seg button')[k].click(), SC, k);
  await pause(120);
  await seek(1000);
};
const drawn = () => page.evaluate((SC) => {
  const svg = document.querySelector(SC + ' .stage svg');
  const vis = (el) => { for (let e = el; e && e !== svg; e = e.parentElement) if (Number(e.getAttribute('opacity') ?? 1) < 0.45) return false; return true; };
  const all = (sel) => [...svg.querySelectorAll(sel)].filter(vis);
  return {
    headline: svg.querySelector('[data-part="headline"]').textContent,
    steps: all('g[data-part="step"]').map((g) => Number(g.dataset.step)),
    lastAdders: (() => { const s = all('g[data-part="step"]').at(-1); return s ? [...s.querySelectorAll('g[data-part="adder"]')].map((a) => [Number(a.dataset.src), Number(a.dataset.dst)]) : []; })(),
    cells: all('g[data-part="cell"]').map((g) => ({ range: g.dataset.range, final: g.dataset.final === '1' })),
    needs: all('g[data-part="need"]').map((g) => Number(g.dataset.needs)),
    counter: (() => { const c = svg.querySelector('[data-part="counter"]'); return { adds: c.dataset.adds, steps: c.dataset.steps, text: c.textContent }; })(),
    compare: all('g[data-part="compare-row"]').map((g) => [g.dataset.name, Number(g.dataset.adds), Number(g.dataset.steps)]),
    warps: all('g[data-part="warp"]').map((g) => ({ net: g.dataset.net, shown: Number(g.dataset.shown), finishes: Number(g.dataset.finishes), adds: [...g.querySelectorAll('g[data-part="warp-step"]')].map((s) => Number(s.dataset.adds)) })),
  };
}, SC);
const R = ranges(bk16);
const rangesAt = (k) => R[k].map((x) => `${x.lo}..${x.hi}`);
const finals = (d) => d.cells.flatMap((x, i) => (x.final ? [i] : []));
for (let k = 0; k < N; k++) {
  await part(k);
  const cap = (await page.$(SC + ' .caption')) ? await text(SC + ' .caption') : '';
  const audioFor = manifest?.clips?.[`bk-sweeps-${k + 1}`];
  if (!audioFor) check(`part ${k + 1}: with no audio its script is shown`, cap === scripts[k], cap);
}
await part(0);
let d = await drawn();
check('part 1: step 1 drawn, 8 adders into the odd positions from their left neighbours', same(d.steps, [1]) && same(d.lastAdders, bk16.steps[0].adders.map((a) => [a.src, a.dst])) && same(d.cells.map((x) => x.range), rangesAt(1)), JSON.stringify(d));
await part(1);
d = await drawn();
check('part 2: the reduction tree, 15 additions in 4 steps, positions 0, 1, 3, 7 and 15 final', same(d.steps, [1, 2, 3, 4]) && d.counter.adds === '15' && d.counter.steps === '4' && same(finals(d), finalAfter(bk16, 4)), JSON.stringify(d.counter));
await part(2);
d = await drawn();
check('part 3: level 1 adds 7 into 11; the needs strip is the model\'s after it', same(d.lastAdders, [[7, 11]]) && same(d.needs, needs(R[5])) && same(d.cells.map((x) => x.range), rangesAt(5)));
await part(3);
d = await drawn();
check('part 4: level 2 adds 3, 7, 11 into 5, 9, 13', same(d.lastAdders, [[3, 5], [7, 9], [11, 13]]) && same(d.needs, needs(R[6])));
await part(4);
d = await drawn();
check('part 5: all 16 final; 26 additions in 7 steps against 49 in 4', d.cells.every((x) => x.final) && d.counter.adds === String(additions(bk16)) && same(d.compare, [['Brent-Kung', 26, 7], ['Kogge-Stone', 49, 4]]), JSON.stringify(d.compare));
if (out) await page.screenshot({ path: out + '-scene5.png' });
await part(5);
d = await drawn();
const w32 = { ks: warpView('kogge-stone', 32), bk: warpView('brent-kung', 32) };
check('part 6: 32 lanes, Kogge-Stone done in 5 steps, Brent-Kung in 9, rows from the model', same(d.warps.map((w) => [w.net, w.shown, w.finishes]), [['kogge-stone', 5, 5], ['brent-kung', 9, 9]]) && same(d.warps[1].adds, w32.bk.steps.map((s) => s.adds)) && same(d.warps[0].adds, w32.ks.steps.map((s) => s.adds)), JSON.stringify(d.warps));
if (out) await page.screenshot({ path: out + '-scene6.png' });
// Playback: from part 1, with the sound off, the clock moves and the main button pauses.
await part(0);
await seek(0);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), SC);
const mainLabel = await text(SC + ' button.primary');
await page.$eval(SC + ' button.primary', (b) => b.click());
let pos = 0;
for (let i = 0; i < 30 && pos === 0; i++) {
  await pause(150);
  pos = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
}
check('Play runs the scene by itself', mainLabel === 'Play' && pos > 0 && (await text(SC + ' button.primary')) === 'Pause', `${mainLabel} ${pos}`);
await page.$eval(SC + ' button.primary', (b) => b.click());
await pause(100);
await (await capBox()).click();
await pause(100);
check('captions switch on and are remembered in the browser', await page.evaluate(() => localStorage.getItem('captions') === 'on'));

// ---- questions
const question = (needle) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), needle);
const answer = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
const choice = async (needle, optionText) => {
  const q = await question(needle);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), optionText);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-ex6 accepts 28', await answer('Book exercise 6', 28));
check('q-ops accepts 2036', await answer('How many additions does Brent-Kung perform', 2036));
check('q-steps accepts 19', await answer('And in how many steps', 19));
check('q-needs accepts 11', await answer('From which position does it receive', 11));
check('q-warp: idle lanes are occupied anyway', await choice('why does the book keep Kogge-Stone', 'the lanes Brent-Kung leaves idle are occupied anyway'));
check('q-ex7 rejects "fewer steps"', !(await choice('Why is one barrier per iteration enough', 'Brent-Kung has fewer steps')));
check('q-ex7: disjoint reads and writes', await choice('Why is one barrier per iteration enough', 'no position written by one thread is read by another'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
