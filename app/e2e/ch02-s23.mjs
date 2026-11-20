// Browser check for section 2.3, "Vector addition and the host stub": the two listings,
// the narrated animation "Three parts, two memories", the prediction, the boundary
// traffic explorer and the questions. Everything asserted is computed from the model
// the page runs on (src/sim/offload.ts) or read from the lab files it shows.
// Usage: node e2e/ch02-s23.mjs [screenshot-prefix]
// E2E_DARK=1 and E2E_WIDTH=<px> run the same checks in the dark theme or at another width.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { FLOAT_BYTES, N, boundaryTraffic, chainTimes, expected, program, run } from '../src/sim/offload.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const read = (rel) => fs.readFileSync(path.join(repo, rel), 'utf8');
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000, dark: process.env.E2E_DARK === '1' });
// Leave no trace in the real progress file.
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
// The page's own number and size formats (lib/reports fmt, S23 bytesText), for comparing with what it shows.
const fmt = (v) => (v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(Number(v.toPrecision(3))));
const bytesText = (b) => {
  if (b === 0) return '0 bytes';
  const units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB'];
  let v = b;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${fmt(v)} ${units[u]}`;
};
/** Set a range input the way a person would, so React sees the change. */
const setRange = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const question = (start) => page.evaluateHandle((s) => [...document.querySelectorAll('.q')].find((q) => q.querySelector('.prompt').textContent.includes(s)), start);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, start) => {
  await q.evaluate((e, s) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(s)).querySelector('input').click(), start);
  await pause(50);
  await q.evaluate((e) => e.querySelector('button.primary').click());
  await pause(120);
};

await goto(page, 'ch02/2.3');
await page.waitForSelector('section.scene');
await page.waitForSelector(bench('Boundary traffic'));
await page.waitForFunction(() => document.querySelectorAll('figure.code').length >= 2 && ![...document.querySelectorAll('figure.code pre')].some((p) => p.textContent.includes('// loading')));

// ---------------------------------------------------------------- the two listings
const figs = await page.$$eval('figure.code', (fs_) => fs_.map((f) => ({
  text: f.querySelector('pre').textContent,
  first: Number(f.querySelector('.ln').dataset.n),
  status: f.querySelector('.status')?.textContent ?? '',
})));
const seq = read('labs/ch02/snippets/vecadd_sequential.cu');
check('the sequential listing is labs/ch02/snippets/vecadd_sequential.cu, whole', figs[0].text.trim() === seq.trim() && figs[0].first === 1, figs[0].text.slice(0, 80));
check('it is labelled front-end-checked', figs[0].status === 'front-end-checked', figs[0].status);
const starter = read('labs/ch02/lab1_vecadd/starter/kernel.cu').replace(/\n$/, '').split('\n');
const at = starter.findIndex((l) => /^int vecAdd\(/.test(l));
check(`the starter listing is lab 2.1's host function, from line ${at + 1} of the file`, figs[1].first === at + 1 && figs[1].text.trim() === starter.slice(at).join('\n').trim(), `${figs[1].first}`);
check('the starter still has its three parts to write', ['TODO 3: Part 1', 'TODO 4: Part 2', 'TODO 6: Part 3'].every((s) => figs[1].text.includes(s)));
const pageText = await text('.page');
check('the corrected Fig. 2.5 declaration is the tested one', pageText.includes('float *d_A *d_B, *d_C;') && pageText.includes(program('cuda').head[2].trim()));

// ---------------------------------------------------------------- the narrated animation
const SC = 'section.scene';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const scripts = [1, 2, 3, 4, 5, 6].map((k) => strip(read(`content/ch02/narration/round-trip-${k}.txt`)));
const manifest = await fetch(BASE + '/content/ch02/narration/manifest.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), SC, label);
  await pause(150);
};
const seek = async (v) => {
  await setRange(SC + ' input[type=range]', v);
  await pause(80);
};
const drawn = () => page.$$eval(SC + ' .stage svg text', (ts) => ts.map((x) => x.textContent.trim()).filter(Boolean));
const has = (list, s) => list.some((x) => x.includes(s));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), SC, word);
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(200);
check('the animation is "Three parts, two memories", in six parts', (await page.$eval(SC, (e) => e.getAttribute('aria-label'))) === 'Three parts, two memories' && (await page.$$(SC + ' .seg button')).length === 6);
check('six scripts, each 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(' '));
check('it has a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
await (await box('captions')).click();
await pause(120);
check('switching captions on is remembered in the browser', (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');

const cuda = program('cuda');
const end = run(cuda);
const LABELS = ['Host only', 'Second memory', 'Copy in', 'Launch', 'Copy back', 'The bill'];
const narrow = Number(process.env.E2E_WIDTH || 1440) <= 704;
check(`the drawing uses the ${narrow ? 'stacked' : 'side-by-side'} layout at this width`,
  (await page.$eval(SC + ' .stage svg', (e) => e.dataset.layout)) === (narrow ? 'narrow' : 'wide'));
// On a phone the shared player's row of part buttons does not wrap (reported to the integrator); the drawing itself fits.
if (!narrow) check('the part buttons fit in the player, with no sideways scroll', await page.$eval(SC + ' .seg', (e) => e.scrollWidth <= e.closest('section').clientWidth));
check('the drawing fits in the player at this width', await page.$eval(SC + ' .stage', (e) => e.scrollWidth <= e.clientWidth + 1));
const captionsOk = [];
const frames = {};
for (let k = 0; k < 6; k++) {
  await press(`${k + 1}. ${LABELS[k]}`);
  captionsOk.push((await page.$eval(SC + ' .caption', (e) => e.textContent.trim())) === scripts[k]);
  await seek(1000);
  frames[k + 1] = await drawn();
  await shot(`scene-${k + 1}`, SC);
}
check('each part shows its own script as the caption', captionsOk.every(Boolean), JSON.stringify(captionsOk));
const sums = expected().map(String);
check(`part 1: one host thread adds ${N} pairs and no byte leaves host memory`,
  has(frames[1], `${N} additions, and no byte left host memory`) && sums.every((v) => frames[1].includes(v)) && has(frames[1], 'nothing of ours yet'), frames[1].join('|'));
const afterAlloc = run(cuda, 3);
check('part 2: three allocations, every cell unknown, and three pointers the host cannot read through',
  frames[2].filter((x) => x === '?').length === 3 * N && afterAlloc.device.C.cells.every((c) => c === 'uncleared')
  && has(frames[2], 'A_d: device   B_d: device   C_d: device') && has(frames[2], 'cannot read through'), frames[2].join('|'));
check(`part 3: ${run(cuda, 5).bytesMoved} bytes across before any addition`,
  has(frames[3], `${run(cuda, 5).bytesMoved} bytes across, and no addition has happened yet`) && has(frames[3], `bytes across: ${run(cuda, 5).bytesMoved}`), frames[3].join('|'));
check(`part 5: ${end.bytesMoved} bytes across, ${end.frees} free calls, and the caller holds the sums`,
  has(frames[5], `bytes across: ${end.bytesMoved}`) && has(frames[5], `free calls: ${end.frees}`) && has(frames[5], 'the caller holds C_h') && sums.every((v) => frames[5].includes(v)), frames[5].join('|'));
const shared = run(program('metal', 'buffers'));
check(`part 6 ends on one shared memory: ${shared.bytesMoved} bytes, ${shared.copies} copies, ${shared.frees} free calls, one wait`,
  has(frames[6], `${shared.bytesMoved} bytes across, ${shared.copies} copies, ${shared.frees} free calls, and still one wait before the read`) && has(frames[6], 'buffer C') && has(frames[6], 'One memory, shared by the CPU and the GPU'), frames[6].join('|'));
// The lane labels are the model's calls, in order.
const lanes = frames[5].filter((x) => /^(alloc|copy|launch|free)/.test(x) && x.length < 10);
check('the lanes are labelled with the ten calls of the stub, in order', JSON.stringify(lanes) === JSON.stringify(['alloc A', 'alloc B', 'alloc C', 'copy A', 'copy B', 'launch', 'copy C', 'free A', 'free B', 'free C']), lanes.join('|'));

// Mid-beat states: who waits for whom.
await press('3. Copy in');
await seek(300);
let f = await drawn();
const mid = Number((f.find((x) => x.startsWith('bytes across: ')) ?? '').replace('bytes across: ', ''));
check('part 1, half way: the host is blocked inside cudaMemcpy while A crosses', has(f, 'host thread: blocked inside cudaMemcpy') && mid > 0 && mid < 4 * N && mid % FLOAT_BYTES === 0, `${mid} | ${f.join('|')}`);
await press('4. Launch');
await seek(650);
f = await drawn();
check('part 2: the host thread runs on while the device computes C_d', has(f, 'host thread: running') && has(f, 'device: vecAddKernel:') && has(f, 'the host has moved on'), f.join('|'));
await press('5. Copy back');
await seek(300);
f = await drawn();
check('part 3: the copy back keeps the host blocked until the kernel is done', has(f, 'host thread: blocked inside cudaMemcpy') && has(f, 'Part 3: the copy back waits for the kernel'), f.join('|'));
await press('6. The bill');
await seek(250);
f = await drawn();
check(`part 6 first states the bill: ${end.bytesMoved / N} bytes per element added`,
  has(f, `The bill: ${N} additions, ${end.bytesMoved} bytes across the boundary`) && has(f, `${end.bytesMoved / N} bytes moved for every element added, ${end.copies} copies, ${end.frees} free calls`), f.join('|'));
await seek(820);
f = await drawn();
check('with one shared memory the host still blocks, in waitUntilCompleted', has(f, 'host thread: blocked inside waitUntilCompleted') && has(f, 'bytes across: 0'), f.join('|'));

if (manifest) check('the manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await text(SC + ' footer')).includes('No narration manifest found'));
const soundBox = await box('sound');
if (await soundBox.evaluate((e) => e.checked)) await soundBox.click();
await press('1. Host only');
await press('Play');
await pause(1500);
const moved = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
await press('Pause');
check('pressing Play moves the animation, with the sound off', moved > 20, String(moved));
await (await box('captions')).click();

// ---------------------------------------------------------------- the prediction
const four = { transparent: boundaryTraffic(4, 'transparent', 'separate', 1).transfers.length, resident: boundaryTraffic(4, 'resident', 'separate', 1).transfers.length };
let q = await question('Four elementwise operations in a chain');
await (await q.$('input[type=number]')).type(String(four.resident));
await q.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
let t = await qText(q);
check(`p-crossings: a prediction of ${four.resident} is told it is ${four.transparent}, and that ${four.resident} is the resident count`,
  four.transparent === 12 && four.resident === 6 && t.includes(`You predicted ${four.resident} transfers.`) && t.includes(`It is ${four.transparent}`) && t.includes('when the data stays on the device') && t.includes(`it is ${four.resident}`), t);

// ---------------------------------------------------------------- the explorer
const BT = bench('Boundary traffic');
const cell = (k) => page.$eval(`${BT} [data-cell="${k}"]`, (e) => e.textContent.trim());
const arrows = (dir) => page.$$eval(`${BT} [data-transfer${dir ? `="${dir}"` : ''}]`, (a) => a.length);
const clickIn = async (label) => {
  const ok = await page.evaluate((sel, label) => {
    const b = [...document.querySelector(sel).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, BT, label);
  await pause(80);
  return ok;
};
const table = async (k, n) => {
  const tt = boundaryTraffic(k, 'transparent', 'separate', n);
  const rr = boundaryTraffic(k, 'resident', 'separate', n);
  const want = {
    'in-transparent': String(tt.transfersIn), 'in-resident': String(rr.transfersIn),
    'out-transparent': String(tt.transfersOut), 'out-resident': String(rr.transfersOut),
    'across-transparent': bytesText(tt.bytesAcross), 'across-resident': bytesText(rr.bytesAcross),
    'kernel-transparent': bytesText(tt.kernelBytes), 'kernel-resident': bytesText(rr.kernelBytes),
    'ratio-transparent': fmt(tt.ratio), 'ratio-resident': fmt(rr.ratio),
  };
  const got = {};
  for (const key of Object.keys(want)) got[key] = await cell(key);
  return { ok: JSON.stringify(got) === JSON.stringify(want), got, want };
};
await page.$eval(BT, (e) => e.scrollIntoView({ block: 'start' }));
const n0 = 2 ** 20;
let tb = await table(3, n0);
check('default: 3 operations of 2^20 floats, every number in the table from the model', tb.ok, JSON.stringify(tb.got));
check('default picture: 9 arrows, 6 in and 3 out, and the three operations of the chain',
  (await arrows()) === 9 && (await arrows('in')) === 6 && (await arrows('out')) === 3
  && JSON.stringify(await page.$$eval(`${BT} [data-op]`, (o) => o.map((x) => x.dataset.op))) === JSON.stringify(['add bias', 'scale', 'add residual']));
check('the summary line names the chain and its traffic', (await text(`${BT} [data-out="traffic-summary"]`)).includes(`add bias, scale, add residual: 9 tensor-sized transfers, ${bytesText(9 * FLOAT_BYTES * n0)}`));
await shot('traffic-default', BT);
await clickIn('data stays on the device between operations');
check('resident: k + 2 = 5 arrows, 4 in and 1 out', (await arrows()) === 5 && (await arrows('in')) === 4 && (await arrows('out')) === 1);
await setRange(`${BT} input[aria-label="Operations in the chain"]`, 4);
check('four operations kept resident: 6 arrows, as the prediction said', (await arrows()) === four.resident);
await clickIn('every operation is a transparent offload');
check('four operations as round trips: 12 arrows', (await arrows()) === four.transparent);
const counts = [];
for (const k of [1, 2, 5, 8]) {
  await setRange(`${BT} input[aria-label="Operations in the chain"]`, k);
  const tr = await arrows();
  await clickIn('data stays on the device between operations');
  const re = await arrows();
  await clickIn('every operation is a transparent offload');
  tb = await table(k, n0);
  counts.push([k, tr, re, tb.ok]);
}
check('for 1, 2, 5 and 8 operations: 3k arrows as round trips, k + 2 resident, and the table agrees', counts.every(([k, tr, re, ok]) => tr === 3 * k && re === k + 2 && ok), JSON.stringify(counts));
check('past three operations the chain repeats by layer', (await page.$$eval(`${BT} [data-op]`, (o) => o.map((x) => x.dataset.op)))[3] === 'add bias 2');
await setRange(`${BT} input[aria-label="Elements per tensor, as a power of two"]`, 28);
tb = await table(8, 2 ** 28);
check('n = 2^28: the byte rows scale and the ratios do not', tb.ok && tb.got['ratio-transparent'] === '1' && tb.got['across-transparent'] === bytesText(24 * 2 ** 30), JSON.stringify(tb.got));
await shot('traffic-8', BT);
await clickIn('one shared memory (Metal on Apple silicon, data created in shared buffers)');
const sh = { t: boundaryTraffic(8, 'transparent', 'shared', 2 ** 28), r: boundaryTraffic(8, 'resident', 'shared', 2 ** 28) };
check('one shared memory: no arrows, 0 bytes across in both columns, the same kernel bytes',
  (await arrows()) === 0 && (await page.$(`${BT} svg[data-picture="shared"]`)) !== null
  && (await cell('across-transparent')) === '0 bytes' && (await cell('across-resident')) === '0 bytes'
  && sh.t.bytesAcross === 0 && (await cell('kernel-transparent')) === bytesText(sh.t.kernelBytes) && (await cell('kernel-resident')) === bytesText(sh.r.kernelBytes));
await shot('traffic-shared', BT);

// The reader's own lab 2.3 measurement, or what to run.
const reports = await fetch(BASE + '/api/reports').then((r) => r.json());
const rows = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) =>
  (reports[b].results ?? []).filter((x) => x.lab === 'ch02/lab3_elementwise' && x.variant === 'solution').flatMap((x) => x.bench)
    .filter((r) => r.name === 'chain').map((r) => ({ b, t: chainTimes(r) })).filter((x) => x.t));
const measured = await page.$eval(`${BT} [data-out="chain-measured"]`, (e) => ({ found: Number(e.dataset.found), text: e.textContent }));
if (rows.length)
  check(`the measured chain from lab 2.3 is shown as measured (${rows.length} row(s))`, measured.found === rows.length
    && rows.every((r) => measured.text.includes(`${fmt(r.t.resident * 1e3)} ms`) && measured.text.includes(`${fmt(r.t.roundTrips * 1e3)} ms`))
    && measured.text.includes('Measured on your machine, not predicted'), measured.text);
else check('with no lab 2.3 measurement the explorer says what to run and shows no time', measured.found === 0 && measured.text.includes('python3 labs/pmpp.py run 2.3') && !/\d ms/.test(measured.text), measured.text);

// ---------------------------------------------------------------- the questions
q = await question('the stub allocates three device arrays');
await pick(q, 'All three.');
check('q-output-only: "all three" is refused with the wasted transfer named', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('one more transfer of n floats'));
await pick(q, 'A_d and B_d only.');
check('q-output-only: A_d and B_d is accepted', (await qText(q)).includes('Correct.'));
q = await question('On Apple silicon you keep the signature');
await pick(q, 'None, unless the arrays are page-aligned');
check('q-interface: none, unless wrapped or created in buffers', (await qText(q)).includes('Correct.') && (await qText(q)).includes('newBufferWithBytesNoCopy'));
q = await question('You may offload exactly one of two host functions');
await pick(q, 'The matrix product');
check('q-pays: the matrix product, and the explanation counts 12 bytes per addition', (await qText(q)).includes('Correct.') && (await qText(q)).includes('one per 12 bytes') && boundaryTraffic(1, 'transparent', 'separate', 1000).bytesAcross / 1000 === 12);

// ---------------------------------------------------------------- the page as a whole
const all = await text('.page');
check('the page states its book pages', ['§2.3, p. 26', '§2.3, p. 27', '§2.3, p. 28'].every((s) => all.includes(s)));
check('vendor statements name their facts items', ['ch02-facts.md, A4 and A5', 'ch02-facts.md, D4', 'B1 and B3', 'C2 and C3'].every((s) => all.includes(s)));
check('each simulation says what it leaves out', (await page.$$('.note.caveat')).length === 2);
check('answers and predictions were sent to the (intercepted) progress API', puts > 0, String(puts));
const expected404 = (u) => /\/content\/ch02\/narration\/(manifest\.json|round-trip-\d\.mp3)$/.test(u);
const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
