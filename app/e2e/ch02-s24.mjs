// Browser check for section 2.4, "Device memory and data transfer": the device memory
// table, the stub listing, the call-by-call stepper on CUDA, HIP and Metal (both
// interfaces), the prediction that unlocks Metal, the bug hunt (six rounds), the two
// listings that would not compile, the CHECK macro and the questions. Everything
// asserted is computed from the model the page runs on (src/sim/offload.ts), from
// the lab files it shows, or from the reports the server hands the page.
// Usage: node e2e/ch02-s24.mjs [screenshot-prefix]
// E2E_DARK=1 and E2E_WIDTH=<px> run the same checks in the dark theme or at another width.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { DEFECTS, expected, outcome, phaseTimes, program, result, run, showCell } from '../src/sim/offload.ts';

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
const fmt = (v) => (v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(Number(v.toPrecision(3))));
/** Click the first enabled button in `scope` (a CSS selector) whose text is `label`. */
const clickIn = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(70);
  return ok;
};
const disabled = (scope, label) => page.evaluate((scope, label) => [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label)?.disabled, scope, label);
const question = (start) => page.evaluateHandle((s) => [...document.querySelectorAll('.q')].find((q) => q.querySelector('.prompt').textContent.includes(s)), start);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, start) => {
  await q.evaluate((e, s) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(s)).querySelector('input').click(), start);
  await pause(50);
  await q.evaluate((e) => e.querySelector('button.primary').click());
  await pause(120);
};

await goto(page, 'ch02/2.4');
await page.waitForSelector(bench('The round trip, call by call'));
await page.waitForFunction(() => document.querySelector('figure.code pre') && !document.querySelector('figure.code pre').textContent.includes('// loading'));
const reports = await fetch(BASE + '/api/reports').then((r) => r.json());
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');

// ---------------------------------------------------------------- your devices, the stub
const FIELDS = ['totalGlobalMem', 'hasUnifiedMemory', 'recommendedMaxWorkingSetSize'];
const wantRows = real.flatMap((b) => FIELDS.filter((k) => reports[b].device?.[k] !== undefined).map((k) => [b, k]));
const memRows = await page.$eval('[data-your-memory]', (e) => ({ n: Number(e.dataset.yourMemory), text: e.textContent }));
check(`the device table shows what the reports hold (${wantRows.length} row(s)), and nothing else`, memRows.n === wantRows.length
  && wantRows.every(([b, k]) => memRows.text.includes(k) && (k === 'hasUnifiedMemory' || memRows.text.includes(Number(reports[b].device[k]).toLocaleString('en-US')))), memRows.text);
const page0 = await text('.page');
check('the H100 sizes appear only as the book\'s example', page0.includes("The book's example of a size is the NVIDIA H100, with 80 GB or 94 GB") && (page0.match(/80 GB/g) ?? []).length === 2, String((page0.match(/80 GB/g) ?? []).length));
const stubFile = read('labs/ch02/snippets/vecadd_stub.cu');
const stubFig = await page.$eval('figure.code', (f) => ({ text: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' }));
check('the stub listing is labs/ch02/snippets/vecadd_stub.cu, whole, front-end-checked', stubFig.text.trim() === stubFile.trim() && stubFig.status.startsWith('front-end-checked'), stubFig.status);
const L = stubFile.split('\n');
const lineOf = (s) => L.findIndex((l) => l.includes(s)) + 1;
const lastOf = (s) => L.length - [...L].reverse().findIndex((l) => l.includes(s));
const where = await text('[data-out="stub-lines"]');
check('the prose points at the lines of Parts 1 and 3 and of the launch, as they are in the file',
  where.includes(`lines ${lineOf('int size = ')} to ${lastOf('MemcpyHostToDevice')} and ${lineOf('MemcpyDeviceToHost')} to ${lastOf('Free(')}`) && where.includes(`launch on line ${lineOf('<<<')}`), where);

// ---------------------------------------------------------------- the questions on cudaMalloc, cudaFree, cudaMemcpy
let q = await question('You want an array of');
await pick(q, 'v');
check('exercise 5: a count of elements is refused', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('wants bytes'));
await pick(q, 'v * sizeof(int)');
check('exercise 5: v * sizeof(int)', (await qText(q)).includes('Correct.'));
q = await question('is to point at');
await pick(q, '(void **) &A_d');
check('exercise 6: (void **) &A_d, and the cast is optional today', (await qText(q)).includes('Correct.') && (await qText(q)).includes('plain &A_d compiles too'));
q = await question('Why does');
await pick(q, 'Allocation has to write a new address');
check('the out-parameter question, with exercise 8 in the explanation', (await qText(q)).includes('Correct.') && (await qText(q)).includes('cudaError_t'));
q = await question('Copy 3000 bytes');
await pick(q, 'cudaMemcpy(A_h, A_d, 3000, cudaMemcpyHostToDevice);');
check('exercise 7: swapped pointers are called undefined behaviour, not an error', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('undefined behaviour'));
await pick(q, 'cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);');
check('exercise 7: destination, source, bytes, kind', (await qText(q)).includes('Correct.'));

// ---------------------------------------------------------------- A: the round trip, call by call
const RT = bench('The round trip, call by call');
const cellsOf = (row) => page.$$eval(`${RT} [data-row="${row}"] span`, (s) => s.map((x) => x.textContent.trim()));
const cell = (out_) => page.$eval(`${RT} [data-out="${out_}"] td:last-child`, (e) => e.textContent.trim());
const refersTo = (name) => page.$eval(`${RT} [data-pointer="${name}"] td:nth-child(2)`, (e) => e.textContent.trim());
const step = () => page.$eval(`${RT} [data-stub]`, (e) => Number(e.dataset.step));
const readNow = async () => {
  await clickIn(RT, 'Read C now');
  return page.$eval(`${RT} [data-out="read-now"]`, (e) => ({ status: e.dataset.status, text: e.textContent }));
};
/** Every panel of the stepper against the model at the current step. */
const agrees = async (p) => {
  const k = await step();
  const s = run(p, k);
  const bad = [];
  const eq = (name, got, want) => { if (String(got) !== String(want)) bad.push(`${name}: ${got} vs ${want}`); };
  eq('bytes', await cell('bytes'), s.bytesMoved);
  eq('copies', await cell('copies'), s.copies);
  eq('frees', await cell('frees'), s.frees);
  eq('api calls', await cell('api-calls'), s.apiCalls);
  eq('host thread', await cell('host-thread'), k > 0 && s.hostThread === 'blocked' ? 'blocked until the call returns' : 'running');
  eq('queue', await cell('device-queue'), s.queue.length ? s.queue.join(', ') : 'nothing');
  eq('ran', await cell('device-ran'), s.ran.length ? s.ran.join(', ') : 'nothing');
  for (const a of ['A', 'B', 'C']) {
    const d = s.device[a];
    const name = p.backend === 'metal' ? `buffer ${a}` : `${a}_d`;
    if (d && !d.freed) eq(name, (await cellsOf(name)).join(' '), d.cells.map(showCell).join(' '));
    if (s.host) eq(`${a}_h`, (await cellsOf(`${a}_h`)).join(' '), s.host[a].map(showCell).join(' '));
  }
  for (const ptr of s.pointers) eq(ptr.name, await refersTo(ptr.name), ptr.refersTo);
  if (k > 0) {
    const hl = await page.$$eval(`${RT} .code .ln.hl`, (ls) => ls.map((l) => l.textContent.trim()));
    eq('highlighted line', hl.join(' / '), p.calls[k - 1].code.join(' / '));
  }
  return bad;
};

await page.$eval(RT, (e) => e.scrollIntoView({ block: 'start' }));
check('Metal is locked until the prediction is committed', (await disabled(RT, 'Metal, shared buffers')) === true && (await text(RT)).includes('Metal unlocks when'));
const cuda = program('cuda');
const mismatches = [];
const reads = [];
for (let k = 1; k <= cuda.calls.length; k++) {
  await clickIn(RT, 'Step');
  const bad = await agrees(cuda);
  if (bad.length) mismatches.push(`step ${k}: ${bad.join('; ')}`);
  if (k === 6 || k === 7) reads.push(await readNow());
  if (k === 6) await shot('stepper-cuda-launch', RT);
}
check(`CUDA: every panel at each of the ${cuda.calls.length} calls agrees with the model`, mismatches.length === 0 && (await step()) === cuda.calls.length, mismatches.join(' | '));
check('CUDA, right after the launch: "Read C now" says the result cannot be read from here', reads[0].status === 'impossible' && run(cuda, 6).hostRead.status === 'impossible' && reads[0].text.includes('host code cannot read device memory'), JSON.stringify(reads[0]));
check('CUDA, after the copy back: C_h is the result', reads[1].status === 'valid' && reads[1].text.includes('the result'));
check('CUDA ends with the caller holding A + B, 12n bytes across, 3 copies and 3 frees',
  (await text(`${RT} [data-out="verdict"]`)).includes('The caller holds A + B.') && (await cell('bytes')) === String(12 * cuda.n) && (await cell('frees')) === '3');
await clickIn(RT, 'Back');
check('Back goes one call back', (await step()) === cuda.calls.length - 1);
await page.$eval(`${RT} button[data-group="copy in"]`, (b) => b.click());
await pause(70);
check('a step group jumps to its last call', (await step()) === cuda.calls.findLastIndex((c) => c.group === 'copy in') + 1 && (await agrees(cuda)).length === 0);
await clickIn(RT, 'Restart');
check('Restart goes back to before the first call', (await step()) === 0 && (await text(`${RT} [data-out="says"]`)).includes('Nothing has run yet.'));

await clickIn(RT, 'HIP');
const hip = program('hip');
await clickIn(RT, 'Run to the end');
const hipCode = await page.$eval(`${RT} .code pre`, (e) => e.textContent);
check('HIP: the same ten calls with hip names, the same result and bytes', (await step()) === hip.calls.length && hipCode.includes('hipMalloc((void**)&A_d, size);') && !hipCode.includes('cudaMalloc') && (await agrees(hip)).length === 0);
await page.$eval(`${RT} button[data-group="free"]`, (b) => b.click());
await pause(70);
check('HIP: the free step says that hipFree synchronises first', (await text(`${RT} [data-out="says"]`)).includes('hipFree performs an implicit hipDeviceSynchronize() first'));

// The prediction unlocks Metal.
const mBuf = run(program('metal', 'buffers'));
q = await question('The CUDA stub copies data three times');
await (await q.$('input[type=number]')).type('0');
await q.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
let t = await qText(q);
check(`p-copies: ${mBuf.copies} copies and ${mBuf.frees} free calls, and the wait remains`, mBuf.copies === 0 && mBuf.frees === 0 && t.includes('It makes 0, against 3 for CUDA.') && t.includes('As you said.') && t.includes('0 free calls, against 3') && t.includes('What is left is the wait'), t);
check('Metal is unlocked', (await disabled(RT, 'Metal, shared buffers')) === false);
await clickIn(RT, 'Metal, shared buffers');
const metal = program('metal', 'buffers');
const commit = metal.calls.findIndex((c) => c.id === 'launch') + 1;
const wait = metal.calls.findIndex((c) => c.id === 'wait') + 1;
const metalBad = [];
let metalReads = [];
for (let k = 1; k <= metal.calls.length; k++) {
  await clickIn(RT, 'Step');
  const bad = await agrees(metal);
  if (bad.length) metalBad.push(`step ${k}: ${bad.join('; ')}`);
  if (k === commit || k === wait) metalReads.push(await readNow());
  if (k === commit) await shot('stepper-metal-commit', RT);
}
check(`Metal, data in buffers: every panel at each of the ${metal.calls.length} calls agrees with the model`, metalBad.length === 0, metalBad.join(' | '));
check('Metal draws one memory and no host arrays', (await page.$(`${RT} [data-memory="shared"]`)) !== null && (await page.$(`${RT} [data-memory="device"]`)) === null && (await page.$(`${RT} [data-row="A_h"]`)) === null);
check('Metal, after commit: the read works and is not the result', metalReads[0].status === 'stale' && metalReads[0].text.includes('readable, and not the result') && run(metal, commit).hostRead.status === 'stale', JSON.stringify(metalReads[0]));
check('Metal, after the wait: the contents are the result', metalReads[1].status === 'valid');
check('Metal with the data in buffers ends with 0 bytes copied, 0 copies, 0 free calls, and no free step',
  (await cell('bytes')) === '0' && (await cell('copies')) === '0' && (await cell('frees')) === '0' && (await page.$(`${RT} button[data-group="free"]`)) === null);
await clickIn(RT, 'caller hands me host arrays');
const metalH = program('metal', 'host-arrays');
await clickIn(RT, 'Run to the end');
const mh = run(metalH);
check(`Metal with host arrays: the copies come back (${mh.copies} copies, ${mh.bytesMoved} bytes), still no free call`,
  (await agrees(metalH)).length === 0 && (await cell('copies')) === String(mh.copies) && mh.copies === 3 && (await cell('frees')) === '0' && result(mh).join() === expected().join());
await shot('stepper-metal-host-arrays', RT);

// The reader's own lab 2.2 phase times, where a run exists.
const phasesRow = (b) => {
  const rows = (reports[b]?.results ?? []).filter((x) => x.lab === 'ch02/lab2_timing' && x.variant === 'solution').flatMap((x) => x.bench).filter((r) => r.name === 'phases');
  return rows.length ? rows[rows.length - 1] : null;
};
const shown = () => page.$$eval(`${RT} [data-measured]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.measured, x.textContent.trim()])));
const metalRow = phasesRow('metal');
if (metalRow) {
  const times = phaseTimes(metalRow);
  const groups = new Set(metalH.calls.map((c) => c.group));
  const want = Object.fromEntries(Object.entries(times).filter(([g]) => groups.has(g)).map(([g, s]) => [g, `${fmt(s * 1e3)} ms`]));
  const got = await shown();
  check('Metal: the lab 2.2 phase times sit beside their step groups, as measured', JSON.stringify(got) === JSON.stringify(want) && (await text(`${RT} [data-out="phases-note"]`)).includes('measured, not modelled'), `${JSON.stringify(got)} vs ${JSON.stringify(want)}`);
} else check('Metal: with no lab 2.2 run no time is shown', Object.keys(await shown()).length === 0);
await clickIn(RT, 'CUDA');
if (!phasesRow('cuda')) check('CUDA: with no lab 2.2 run on CUDA no time is shown', Object.keys(await shown()).length === 0 && (await page.$(`${RT} [data-out="phases-note"]`)) === null);

// ---------------------------------------------------------------- questions after the stepper
q = await question('A launch covers only the first 768');
await pick(q, 'Zeros on both');
check('q-not-cleared: "zeros on both" is refused with NVIDIA\'s words', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('the memory is not cleared'));
await pick(q, 'On CUDA whatever that device memory held before.');
check('q-not-cleared: CUDA stale, Metal zeros', (await qText(q)).includes('Correct.'));
q = await question('Metal, shared buffers. The host commits');
await pick(q, 'Possibly old or partly written data.');
check('q-shared-wait: it must wait for the command buffer', (await qText(q)).includes('Correct.') && (await qText(q)).includes('waitUntilCompleted'));

// ---------------------------------------------------------------- B: break the stub
const BS = bench('Break the stub');
const emuSeen = (o) => (o.stoppedAt != null ? `stopped: ${o.finding.kind}` : o.result.map(showCell).join(' '));
const signature = Object.fromEntries(DEFECTS.map((d) => [d, `${emuSeen(outcome(d, 'emu'))} | ${outcome(d, 'cuda').result.map(showCell).join(' ')}`]));
check('what a test sees identifies the defect: seven different symptoms', new Set(Object.values(signature)).size === DEFECTS.length);
await page.$eval(BS, (e) => e.scrollIntoView({ block: 'start' }));
const seenDefects = [];
const roundBad = [];
for (let r = 0; r < 6; r++) {
  const sym = await page.$eval(`${BS} [data-out="hunt-symptom"]`, (e) => `${e.dataset.emu} | ${e.dataset.device}`);
  const d = DEFECTS.find((x) => signature[x] === sym);
  if (!d) { roundBad.push(`round ${r + 1}: no defect has symptom ${sym}`); break; }
  seenDefects.push(d);
  if ((await text(`${BS} [data-out="hunt-round"]`)).trim() !== `Round ${r + 1} of 6.`) roundBad.push(`round label ${r + 1}`);
  if (r === 0) await shot('hunt-round', BS);
  // Round 2 is answered wrongly on purpose.
  const answer = r === 1 ? DEFECTS.find((x) => x !== d) : d;
  await page.$eval(`${BS} label[data-defect="${answer}"] input`, (i) => i.click());
  await pause(40);
  await clickIn(BS, 'Check');
  const v = await page.$eval(`${BS} [data-out="hunt-verdict"]`, (e) => ({ right: e.dataset.right, text: e.textContent }));
  if (v.right !== (answer === d ? 'yes' : 'no')) roundBad.push(`round ${r + 1}: verdict ${v.right}`);
  const classes = await page.$$eval(`${BS} tr[data-env]`, (rows) => Object.fromEntries(rows.map((x) => [x.dataset.env, x.dataset.class])));
  const want = Object.fromEntries(['cuda', 'hip', 'emu', 'metal'].map((e) => [e, outcome(d, e).klass]));
  if (JSON.stringify(classes) !== JSON.stringify(want)) roundBad.push(`round ${r + 1} (${d}): classes ${JSON.stringify(classes)} vs ${JSON.stringify(want)}`);
  if (r === 0) {
    // Step the broken stub in the emulator and as documented.
    await clickIn(BS, 'in the course emulator');
    await clickIn(BS, 'Run to the end');
    const emu = outcome(d, 'emu');
    const at = await page.$eval(`${BS} [data-stub]`, (e) => Number(e.dataset.step));
    if (at !== (emu.stoppedAt ?? emu.program.calls.length)) roundBad.push(`round 1 emulator steps: ${at}`);
    if (emu.stoppedAt != null && !(await text(`${BS} [data-out="says"]`)).includes('AddressSanitizer stops the run here')) roundBad.push('round 1: no AddressSanitizer stop shown');
    await clickIn(BS, 'as a CUDA device is documented');
    await clickIn(BS, 'Run to the end');
    const dev = outcome(d, 'cuda');
    const holds = dev.result.join() === expected().join();
    const verdict = await text(`${BS} [data-out="verdict"]`).catch(() => '');
    const wantVerdict = dev.stoppedAt != null ? '' : holds && !dev.finding ? 'The caller holds A + B.' : holds ? 'and the stub broke a rule on the way' : 'The caller does not hold A + B.';
    if (!verdict.includes(wantVerdict)) roundBad.push(`round 1 device verdict: ${verdict} vs ${wantVerdict}`);
    if (!(await page.$eval(`${BS} .code`, (e) => e.textContent)).includes(dev.program.calls.find((c) => c.changed)?.code[0] ?? '')) roundBad.push('round 1: the changed line is not in the listing');
    await shot('hunt-stepped', BS);
  }
  await clickIn(BS, r < 5 ? 'Next round' : 'See the score');
}
check('six rounds, each symptom identified, each verdict and outcome table from the model', roundBad.length === 0 && seenDefects.length === 6 && new Set(seenDefects).size === 6, roundBad.join(' | ') || seenDefects.join(','));
const final = await page.$eval(`${BS} [data-out="hunt-final"]`, (e) => ({ score: Number(e.dataset.score), text: e.textContent }));
const emuStops = DEFECTS.filter((d) => outcome(d, 'emu').stoppedAt != null).length;
check(`the score is 5 of 6 after one wrong answer, and the summary counts ${emuStops} emulator stops`, final.score === 5 && final.text.includes('in 5 of 6 rounds') && emuStops === 2 && final.text.includes('stopped exactly two') && DEFECTS.every((d) => outcome(d, 'cuda').errorCode === null), final.text);
await clickIn(BS, 'Play six more');
check('"Play six more" starts again at round 1 with a score of 0', (await text(`${BS} [data-out="hunt-round"]`)).trim() === 'Round 1 of 6.' && (await text(`${BS} [data-out="hunt-score"]`)).trim() === '0');

// ---------------------------------------------------------------- C: two listings that would not compile
const TL = bench('Two listings that would not compile');
const token = async (listing, key) => {
  await page.$eval(`${TL} [data-listing="${listing}"] button[data-token="${key}"]`, (b) => b.click());
  await pause(70);
  return page.$eval(`${TL} [data-out="listing-${listing}"]`, (e) => e.textContent);
};
await page.$eval(TL, (e) => e.scrollIntoView({ block: 'start' }));
check('the macro is hidden until both tokens are found', !(await text('.page')).includes('CHECK, from lab 2.1'));
t = await token('fig25', 'comma');
check('Fig. 2.5: the comma that is there is not the answer', t.includes('Not this token.'), t);
t = await token('fig25', 'dB');
check("Fig. 2.5: the missing comma, clang's message and the corrected line", t.includes('That is the one.') && t.includes("error: expected ';' at end of declaration") && t.includes(program('cuda').head[2].trim()), t);
t = await token('check', 'err');
check('the sidebar: the declaration is not the answer', t.includes('Not this token.'), t);
t = await token('check', 'error');
check("the sidebar: 'error' was never declared, with clang's message", t.includes('That is the one.') && t.includes("use of undeclared identifier 'error'") && t.includes('if (err != cudaSuccess) {'), t);
await shot('listings', TL);
await page.waitForFunction(() => [...document.querySelectorAll('figure.code .tabs button')].some((b) => b.textContent === 'CHECK, from lab 2.1'));
const macroFig = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.querySelector('.tabs button')?.textContent === 'CHECK, from lab 2.1');
  return { text: f.querySelector('pre').textContent, first: Number(f.querySelector('.ln').dataset.n) };
});
const sol = read('labs/ch02/lab1_vecadd/solution/kernel.cu').split('\n');
const m0 = sol.findIndex((l) => /^#define CHECK\(/.test(l));
check(`with both found, the CHECK macro of lab 2.1's solution appears, from line ${m0 + 1}`, macroFig.first === m0 + 1 && macroFig.text.startsWith('#define CHECK(call)') && macroFig.text.includes('err != cudaSuccess') && macroFig.text.includes('return 1;'), JSON.stringify(macroFig).slice(0, 120));

// ---------------------------------------------------------------- the page as a whole
const all = await text('.page');
check('the page states its book pages', ['§2.4, p. 28', '§2.4, pp. 28–30', '§2.4, pp. 30–31', '§2.4, p. 32', '§2.4, p. 29'].every((s) => all.includes(s)));
check('vendor statements name their facts items', ['ch02-facts.md, A12', 'A2 and A3', 'ch02-facts.md, A4', 'ch02-facts.md, A6', 'D1 to D4', 'ch02-facts.md, C5', 'A16'].every((s) => all.includes(s)));
check('the Metal table has the rows of the facts file', ['Where a copy returns', 'newBufferWithBytesNoCopy', 'waitUntilCompleted'].every((s) => all.includes(s)));
check('each simulation says what it leaves out', (await page.$$('.note.caveat')).length === 2);
check('answers and predictions were sent to the (intercepted) progress API', puts > 0, String(puts));
const real404 = missing;
const realErrors = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors', realErrors.length === 0 && real404.length === 0, [...realErrors, ...real404].join(' | '));

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
