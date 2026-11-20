// Browser check for section 6.2, "Hiding memory latency": the bus calculator, the channel
// timeline explorer, the dealer, the phase-by-phase replay with its 8 x 8 prediction, the
// bytes-in-flight explorer and the questions. Every number asserted comes from
// src/sim/dram.ts, imported here directly, or is the book's worked figure.
// The narrated animation has its own check: e2e/ch06-scene-dram-banks.mjs.
// Usage: node e2e/ch06-s62.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  AMD_CDNA_L2, BUS_PRESETS, FIG_6_9, busBandwidth, busesFor, bytesInFlight, channelTimeline, elementsToFill, phaseChannels,
} from '../src/sim/dram.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from a clean one so the predictions are unanswered.
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress'))
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : '{"version":1}' });
  r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (v) => Math.round(v * 1000) / 10;
const r2 = (v) => Math.round(v * 100) / 100;
const B = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const outOf = (scope, name) => page.$eval(`${B(scope)} [data-out="${name}"]`, (e) => ({ text: e.textContent.replace(/\s+/g, ' ').trim(), value: e.dataset.value }));
const click = async (scope, label, exact = true) => {
  const done = await page.evaluate((sel, label, exact) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => (exact ? b.textContent.trim() === label : b.textContent.includes(label)));
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B(scope), label, exact);
  await pause(80);
  return done;
};
const setInput = async (scope, label, v) => {
  await page.evaluate((sel, label, v) => {
    const el = document.querySelector(`${sel} input[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, B(scope), label, v);
  await pause(80);
};
const setSelect = async (scope, label, v) => {
  await page.evaluate((sel, label, v) => {
    const el = document.querySelector(`${sel} select[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, B(scope), label, v);
  await pause(80);
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumber = async (needle, v) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, optionStart) => {
  const q = await question(needle);
  await q.evaluate((e, s) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(s)).querySelector('input').click(), optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch06/6.2');
await page.waitForSelector(B('Bus bandwidth'));

// ---- the figure of channels and banks
check('Fig. 6.7 is redrawn: four channels, four banks each', (await page.$$eval('figure svg text', (t) => t.filter((x) => /^channel \d: controller$/.test(x.textContent)).length)) === 4
  && (await page.$$eval('figure svg text', (t) => t.filter((x) => /^b\d$/.test(x.textContent)).length)) === 16);

// ---- bus calculator
const BUS = 'Bus bandwidth';
const ddr = BUS_PRESETS.ddr5;
const per = busBandwidth(ddr.widthBits, ddr.clockGHz, ddr.ddr);
let o = await outOf(BUS, 'per');
check('DDR5-6400: one 64-bit bus at 3.2 GHz, double data rate, gives 51.2 GB/s', Number(o.value) === r2(per) && r2(per) === 51.2, JSON.stringify(o));
o = await outOf(BUS, 'need');
check('a CPU at 200 GB/s needs 4 such buses', o.text === String(busesFor(200, per)) && o.text === '4', o.text);
await click(BUS, '1000: a GPU (p. 133)');
o = await outOf(BUS, 'need');
check('a GPU at 1000 GB/s needs 20', o.text === String(busesFor(1000, per)) && o.text === '20', o.text);
await click(BUS, "one HBM3E stack (the book's inputs)");
o = await outOf(BUS, 'total');
const hbm = BUS_PRESETS.hbm3e;
check("the book's HBM3E inputs give 1254.4 GB/s", Number(o.value) === r2(busBandwidth(hbm.widthBits, hbm.clockGHz, hbm.ddr, hbm.buses)) && Number(o.value) === 1254.4, JSON.stringify(o));
check('and the erratum is said beside it', (await text(`${B(BUS)} [data-out="erratum"]`)).includes('the book prints 1 TB/s'));
await setInput(BUS, 'Number of buses', 8);
o = await outOf(BUS, 'total');
check('the bus count field drives the total', Number(o.value) === r2(busBandwidth(64, 4.9, true, 8)), JSON.stringify(o));
check('the erratum goes away when the inputs are no longer the book\'s', !(await page.$(`${B(BUS)} [data-out="erratum"]`)));
await click(BUS, '1 (single)');
o = await outOf(BUS, 'per');
check('single data rate halves a bus', Number(o.value) === r2(busBandwidth(64, 4.9, false)), JSON.stringify(o));
if (out) await (await page.$(B(BUS))).screenshot({ path: out + '-bus.png' });

check('q-buses accepts 20', (await answerNumber('64-bit DDR5-6400 buses', 20)).includes('Correct.'));
check('q-hbm accepts 1254.4', (await answerNumber('HBM3E example', 1254.4)).includes('Correct.'));

// ---- channel timeline explorer
const CH = 'One channel, its banks, in time';
const tlOut = async () => ({ util: Number((await outOf(CH, 'util')).value), gbs: Number((await outOf(CH, 'gbs')).value), bound: Number((await outOf(CH, 'bound')).value), need: (await outOf(CH, 'need')).text });
let t = await tlOut();
const one = channelTimeline({ R: 20, banks: 1 });
check("the book's case: R = 20, one bank, 4.8 %", t.util === pct(one.utilisation) && t.util === 4.8, JSON.stringify(t));
check('a 16 GB/s channel delivers 0.76 GB/s', t.gbs === r2(one.utilisation * 16) && t.gbs === 0.76, JSON.stringify(t));
check('R + 1 = 21 banks fill the bus', t.need === '21');
const bars = await page.$$eval(`${B(CH)} svg rect`, (rs) => rs.filter((r) => r.getAttribute('fill') === 'var(--global)').length);
check('the drawing shows the bursts of the model\'s timeline', bars > 0 && (await page.$eval(`${B(CH)} svg`, (s) => s.dataset.util)) === String(pct(one.utilisation)));
await setInput(CH, 'Banks', 2);
t = await tlOut();
check('two banks double it: 9.5 %', t.util === pct(channelTimeline({ R: 20, banks: 2 }).utilisation) && t.util === 9.5, JSON.stringify(t));
for (const banks of [7, 15]) {
  await setInput(CH, 'Banks', banks);
  t = await tlOut();
  check(`${banks} banks: utilisation is the book's bound, ${banks}/21`, t.util === pct(channelTimeline({ R: 20, banks }).utilisation) && t.util === t.bound && t.util === pct(banks / 21), JSON.stringify(t));
}
await setInput(CH, 'Banks', 21);
t = await tlOut();
check('21 banks: 100 %', t.util === 100 && t.gbs === 16, JSON.stringify(t));
check('and the verdict says exactly enough', (await text(`${B(CH)} [data-out="verdict"]`)).startsWith('Exactly enough.'));
await setInput(CH, 'Requests sent to a busy bank', 20);
t = await tlOut();
const conflicted = channelTimeline({ R: 20, banks: 21, conflictShare: 0.2, seed: 1 });
check('sending 20 % of requests to a busy bank opens gaps: the model\'s figure, below the bound', t.util === pct(conflicted.utilisation) && t.util < 100 && t.bound === 100, JSON.stringify(t));
check('and the verdict counts the conflicts', (await text(`${B(CH)} [data-out="verdict"]`)).includes(`${conflicted.accesses.filter((a) => a.conflict).length} conflicts`));
await click(CH, 'Draw other conflicts');
t = await tlOut();
check('a new draw gives the model\'s figure for the next seed', t.util === pct(channelTimeline({ R: 20, banks: 21, conflictShare: 0.2, seed: 2 }).utilisation), JSON.stringify(t));
await setInput(CH, 'Latency-to-burst ratio R', 30);
await setInput(CH, 'Requests sent to a busy bank', 0);
t = await tlOut();
check('R = 30 needs 31 banks; 21 give 21/31', t.need === '31' && t.util === pct(21 / 31), JSON.stringify(t));
await click(CH, "The book's case: R = 20, one bank, 16 GB/s");
t = await tlOut();
check('the preset returns to the book\'s case', t.util === 4.8 && t.gbs === 0.76, JSON.stringify(t));
if (out) await (await page.$(B(CH))).screenshot({ path: out + '-timeline.png' });

check('q-util accepts 4 GB/s', (await answerNumber('can move 32 GB/s', 4)).includes('Correct.'));
check('q-banks accepts 31', (await answerNumber('latency-to-burst ratio is 30', 31)).includes('Correct.'));
check('q-banks rejects 30 and gives the answer', (await (async () => {
  const q = await question('latency-to-burst ratio is 30');
  const inp = await q.$('input[type=number]');
  await inp.click({ clickCount: 3 });
  await inp.type('30');
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
})()).includes('The answer is 31'));

// ---- the dealer
const D = 'Dealing an array to channels and banks';
const cell = (c, b) => page.$eval(`${B(D)} [data-cell="${c},${b}"]`, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const fig = [['0,0', 'M[0..1]'], ['0,1', 'M[8..9]'], ['1,0', 'M[2..3]'], ['1,1', 'M[10..11]'], ['2,0', 'M[4..5]'], ['2,1', 'M[12..13]'], ['3,0', 'M[6..7]'], ['3,1', 'M[14..15]']];
const got = [];
for (const [k] of fig) got.push(await cell(...k.split(',')));
check('the dealer reproduces Fig. 6.9', fig.every(([, want], i) => got[i].endsWith(want)), got.join(' | '));
check('16 elements touch 4 of 4 channels', (await outOf(D, 'channels')).text === '4');
check('every channel gets a turn from 8 elements, every bank from 16', (await outOf(D, 'fill-channels')).text === String(elementsToFill(FIG_6_9).channels) && (await text(B(D))).includes('every bank of every channel from 16'));
await setSelect(D, 'Array length', 32);
check('32 elements wrap round: channel 0, bank 0 holds M[0..1] and M[16..17]', (await cell(0, 0)).endsWith('M[0..1], M[16..17]'), await cell(0, 0));
await click(D, 'AMD CDNA: 256-byte interleave across 32 L2 channels');
check('AMD preset: 32 channel cards', (await page.$$(`${B(D)} [data-channel]`)).length === 32);
check('32 floats sit in one channel under a 256-byte interleave', (await outOf(D, 'channels')).text === '1' && (await cell(0, 0)).includes('M[0..31]'), await cell(0, 0));
check('it takes 2048 floats to give all 32 channels a turn', (await outOf(D, 'fill-channels')).text === String(elementsToFill(AMD_CDNA_L2).channels) && (await outOf(D, 'fill-channels')).text === '2048');
await setSelect(D, 'Array length', 2048);
check('2048 floats touch all 32', (await outOf(D, 'channels')).text === '32');
if (out) await (await page.$(B(D))).screenshot({ path: out + '-dealer.png' });
await click(D, 'Fig. 6.9: 8-byte bursts, 4 channels, 2 banks');
check('the Fig. 6.9 preset restores the toy', (await outOf(D, 'channels')).text === '4' && (await cell(3, 1)).endsWith('M[14..15]'));
check('q-onechannel: one channel', (await answerChoice('256 bytes at a time', 'One.')).includes('Correct.'));

// ---- the phase-by-phase replay and the 8 x 8 prediction
const T = 'The tiled multiplication, phase by phase';
const four = phaseChannels(4, 2);
const lit = () => page.$$eval(`${B(T)} [data-lit="yes"]`, (e) => e.map((x) => x.textContent.trim()));
o = await outOf(T, 'phase-channels');
check('4 x 4, phase 0: channels 0 and 2 (Fig. 6.11 with Fig. 6.9)', o.value === four[0].channels.join(',') && o.value === '0,2', JSON.stringify(o));
check('the loaded cells are M[0], M[1], M[4], M[5], M[8], M[9], M[12], M[13]', JSON.stringify(await page.$$eval(`${B(T)} [data-out="grid"] span.hot`, (s) => s.map((x) => Number(x.dataset.i)))) === JSON.stringify([0, 1, 4, 5, 8, 9, 12, 13]));
const rowsOf = () => page.$$eval(`${B(T)} table tr`, (trs) => trs.map((tr) => [...tr.querySelectorAll('td')].map((td) => td.textContent.trim())));
let rows = await rowsOf();
check('blocks 0,0 and 0,1 load M[0], M[1], M[4], M[5]; blocks 1,0 and 1,1 load M[8], M[9], M[12], M[13]',
  JSON.stringify(rows) === JSON.stringify([['Blocks 0,0 and 0,1', 'M[0], M[1], M[4], M[5]', 'ch. 0, 2'], ['Blocks 1,0 and 1,1', 'M[8], M[9], M[12], M[13]', 'ch. 0, 2']]), JSON.stringify(rows));
await click(T, 'phase 1');
o = await outOf(T, 'phase-channels');
check('phase 1: channels 1 and 3', o.value === '1,3' && JSON.stringify(await lit()) === JSON.stringify(['channel 1', 'channel 3']), JSON.stringify(o));
const locked = await page.$eval(B(T), (b) => [...b.querySelectorAll('button')].find((x) => x.textContent.trim() === '8 × 8').disabled);
check('8 x 8 is locked until the prediction is committed', locked);
const P = await question('all sixteen blocks running');
await (await P.$('input[type=number]')).type('4');
await (await P.$('button.primary')).click();
await pause(200);
const reveal = await page.$eval('[data-out="p-8x8"]', (e) => e.textContent.replace(/\s+/g, ' ').trim());
check('the reveal gives 1 channel per phase, channel 0, then 1, then 2, then 3', reveal.startsWith('1.') && reveal.includes('channel 0, then 1, then 2, then 3'), reveal);
check('and says the derivation is ours', (await P.evaluate((e) => e.textContent)).includes('notes item 8'));
check('8 x 8 opens after the commit', await click(T, '8 × 8'));
const eight = phaseChannels(8, 2);
for (const ph of [0, 1, 2, 3]) {
  await click(T, `phase ${ph}`);
  o = await outOf(T, 'phase-channels');
  check(`8 x 8, phase ${ph}: only channel ${ph}`, o.value === eight[ph].channels.join(',') && o.value === String(ph), JSON.stringify(o));
}
check('the grid has 64 cells; a phase loads two columns of M, 16 elements', (await page.$$(`${B(T)} [data-out="grid"] > span`)).length === 64 && (await page.$$(`${B(T)} [data-out="grid"] > span.hot`)).length === 16);
rows = await rowsOf();
check('and the four rows of blocks each load from channel 3 in phase 3', rows.length === 4 && rows.every((r) => r[2] === 'ch. 3') && rows[0][0] === 'Blocks 0,0 to 0,3', JSON.stringify(rows));
if (out) await (await page.$(B(T))).screenshot({ path: out + '-replay.png' });

check('q-occupancy: too few requests in flight', (await answerChoice('Its occupancy is 25 %', 'Too few requests in flight')).includes('Correct.'));

// ---- bytes in flight
const F = "Bytes in flight (ours, not the book's)";
await setSelect(F, 'Bandwidth to sustain', 'book');
let f = bytesInFlight(3350, 50, 4, 32);
check("the book's example at 50 ns: 167,500 bytes in flight", Number((await outOf(F, 'bytes')).value) === Math.round(f.bytes) && Math.round(f.bytes) === 167500);
check('41,875 four-byte loads, 1309 warp-wide loads of 32', Number((await outOf(F, 'loads')).value) === Math.round(f.loads) && Number((await outOf(F, 'warps')).value) === Math.round(f.warpLoads));
await click(F, '16 (float4)');
f = bytesInFlight(3350, 50, 16, 32);
check('16-byte loads: a quarter of the loads for the same bytes', Number((await outOf(F, 'loads')).value) === Math.round(f.loads) && Number((await outOf(F, 'bytes')).value) === 167500);
await click(F, '64');
f = bytesInFlight(3350, 50, 16, 64);
check('a 64-wide wavefront halves the warp-wide instructions', Number((await outOf(F, 'warps')).value) === Math.round(f.warpLoads));
await setInput(F, 'Latency in nanoseconds', 200);
f = bytesInFlight(3350, 200, 16, 64);
check('four times the latency, four times the bytes', Number((await outOf(F, 'bytes')).value) === Math.round(f.bytes) && Math.round(f.bytes) === 670000);
const reports = await (await fetch((process.env.APP_URL || 'http://localhost:5173') + '/api/reports')).json();
const mine = Object.entries(reports).filter(([b]) => b !== 'emu').flatMap(([b, run]) => run.results.filter((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution').flatMap((x) => x.bench).filter((r) => r.name === 'roofline' && r.bandwidth_gb_per_s).map((r) => [b, r.bandwidth_gb_per_s]));
const opts = await page.$$eval(`${B(F)} select option`, (os) => os.map((x) => x.value));
check('your lab 5.1 measurements are offered, and nothing else is invented', JSON.stringify(opts) === JSON.stringify([...mine.map(([b]) => b), 'book', 'custom']), JSON.stringify(opts));
if (mine.length) {
  await setSelect(F, 'Bandwidth to sustain', mine[0][0]);
  check('picking your measurement uses it', Number((await outOf(F, 'bytes')).value) === Math.round(bytesInFlight(mine[0][1], 200, 16, 64).bytes));
}
if (out) await (await page.$(B(F))).screenshot({ path: out + '-inflight.png' });

// ---- notes: every Today note says when it was checked, the Beyond note is there
const notes = await page.$$eval('.note', (ns) => ns.map((n) => n.className + ': ' + n.textContent));
check('Today notes carry their check date', notes.filter((n) => n.includes('today')).every((n) => n.includes('Checked 2026-11-20')) && notes.filter((n) => n.includes('today')).length >= 5);
check('the bytes-in-flight material is labelled as ours', notes.some((n) => n.includes('beyond') && n.includes("Little's law")));
check('the model says what it leaves out', notes.some((n) => n.includes('caveat') && n.includes('arrival order')));

// ---- margin notes sit beside the paragraph they belong to (CLAUDE.md), so they open no gaps
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph, never a bench, a listing or another note', misplaced.length === 0, misplaced.join(' | '));

// ---- dark theme and a narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(200);
const colours = await page.$$eval('.prose svg [fill], .prose svg [stroke]', (els) => els.flatMap((e) => [e.getAttribute('fill'), e.getAttribute('stroke')]).filter((c) => c && c !== 'none' && !c.startsWith('var(')));
check('the drawings use theme variables only', colours.length === 0, colours.join(','));
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await (await page.$(B(CH))).screenshot({ path: out + '-narrow-timeline.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|dram-banks-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
