// Browser check for section 2.7, "Compilation": the three-backend toolchain stepper,
// the caller-by-callee classification, the Metal call table, the two lab listings
// and the questions. Everything asserted is computed from the model the page runs on
// (src/sim/toolchain.ts) or cut from the lab files on disk the way the page cuts them.
// Usage: node e2e/ch02-s27.mjs [screenshot-prefix]
//   E2E_DARK=1, E2E_LIGHT=1 and E2E_WIDTH=<px> run the same checks in either theme or at another width.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { tabStatus as sharedTabStatus } from './labstatus.mjs';
import {
  BACKENDS,
  BACKEND_NAME,
  CALLEES,
  CALLERS,
  DYNAMIC_PARALLELISM,
  METAL_FNS,
  PROGRAM,
  QUALIFIER_TEXT,
  canRunOnNewArchitecture,
  compilesOnUsersMachine,
  functionSource,
  legalCall,
  metalCall,
  placement,
  stages,
} from '../src/sim/toolchain.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000, dark: process.env.E2E_DARK === '1' });
// E2E_LIGHT=1 forces the light theme even when the operating system is set to dark.
if (process.env.E2E_LIGHT === '1') await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
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
const exists = async (sel) => (await page.$(sel)) !== null;
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const press = async (sel) => {
  await page.$eval(sel, (e) => e.click());
  await pause(80);
};
const setValue = async (sel, value) => {
  await page.$eval(
    sel,
    (e, value) => {
      const proto = e.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(e, value);
      e.dispatchEvent(new Event(e.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    },
    String(value),
  );
  await pause(80);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, label) => {
  const found = await q.evaluate((e, label) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().startsWith(label));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, label);
  await (await q.$('button.primary')).click();
  await pause(150);
  return found;
};

await goto(page, 'ch02/2.7');
const FOLLOW = 'Follow one source file';
const WHO = 'Who may call it, who runs it';
await page.waitForSelector(bench(FOLLOW));
await page.waitForFunction(() => document.querySelectorAll('figure.code pre').length >= 2, { timeout: 15000 });
await pause(300);

// ---------------------------------------------------------------- the prediction before the stepper
const future = { cuda: canRunOnNewArchitecture('cuda', { ptx: true }), hip: canRunOnNewArchitecture('hip', { archs: ['a'], newArch: 'b' }), metal: canRunOnNewArchitecture('metal') };
let q = await question('did not exist when you built it');
check('q-future comes before the stepper and is labelled as a prediction', (await q.evaluate((e) => e.querySelector('.kind').textContent)) === 'Predict first' &&
  (await page.evaluate((sel) => { const qq = [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('did not exist when you built it')); return !!(qq.compareDocumentPosition(document.querySelector(sel)) & Node.DOCUMENT_POSITION_FOLLOWING); }, bench(FOLLOW))));
await pick(q, 'On all three.');
check('q-future: "all three" is refused because HIP compiles ahead of time', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('HIP compiles ahead of time'));
await pick(q, 'On CUDA, when the binary carries PTX, and on Metal.');
let t = await qText(q);
check('q-future: CUDA with PTX and Metal; the explanation is the model\'s three verdicts', t.includes('Correct.') && t.includes(future.cuda.why) && t.includes(future.metal.why) && t.includes(future.hip.why) && future.cuda.runs === 'yes' && future.hip.runs === 'no' && future.metal.runs === 'yes', t);

// ---------------------------------------------------------------- A: follow one source file
const F = bench(FOLLOW);
const fo = (name) => page.$eval(`${F} [data-out="${name}"]`, (e) => e.textContent.trim());
const factList = (ids) => ids.filter((f) => !f.startsWith('Fig.') && !f.startsWith('p.')).join(', ');
const shownStage = async () => ({
  id: await page.$eval(`${F} [data-follow]`, (e) => e.dataset.stage),
  title: await fo('title'), text: await fo('text'), when: await fo('when'), tool: await fo('tool'), facts: await fo('facts'),
  boxes: await page.$$eval(`${F} [data-box]`, (bs) => bs.map((b) => ({ id: b.dataset.box, state: b.dataset.state, chips: [...b.querySelectorAll('[data-chip]')].map((c) => `${c.dataset.chip}:${c.dataset.copy}`) }))),
});
const wantStage = (s) => ({
  id: s.id, title: s.title + (/[.?!]$/.test(s.title) ? '' : '.'), text: s.text, when: s.when === 'build' ? 'build time' : 'run time', tool: s.tool, facts: `(${factList(s.facts)})`,
  boxes: s.boxes.map((b) => ({ id: b.id, state: b.state ?? 'plain', chips: b.chips.map((c) => `${c.fn}:${c.copy ?? 'source'}`) })),
});
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
/** Steps through every stage with "Next stage" and compares each with the model. */
const walk = async (list) => {
  const bad = [];
  for (let k = 0; k < list.length; k++) {
    if (k > 0) await page.evaluate((sel) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === 'Next stage').click(), F);
    await pause(60);
    const got = await shownStage();
    if (!same(got, wantStage(list[k]))) bad.push(JSON.stringify({ got, want: wantStage(list[k]) }));
  }
  return bad;
};
const placementRows = () => page.$$eval(`${F} table[data-table="placement"] tbody tr`, (rs) => rs.map((r) => ({ fn: r.dataset.fn, n: Number(r.dataset.placements), cells: [...r.querySelectorAll('td')].map((c) => c.textContent.trim()) })));
const placementOk = async (backend) => {
  const rows = await placementRows();
  return rows.length === PROGRAM.length && PROGRAM.every((f, i) => {
    const p = placement(f, backend);
    const r = rows[i];
    return r.fn === f.name && r.n === p.length && r.cells[1] === (backend === 'metal' ? p.map((x) => x.file).join(' and ') : QUALIFIER_TEXT[f.quals]) &&
      r.cells[2] === p.map((x) => x.compiler).join('; and again by ') + (p.length > 1 ? ' (two compiled functions)' : '');
  });
};
const usersMachine = async () => page.$eval(`${F} [data-out="users-machine"]`, (b) => b.parentElement.textContent.trim());
const UM = { always: 'always: every kernel is finished there', 'when-needed': 'only when it has to', never: 'never: the binary holds finished machine code' };

await page.$eval(F, (e) => e.scrollIntoView({ block: 'start' }));
check('the stepper opens on CUDA at stage 1', (await page.$eval(`${F} [data-follow]`, (e) => e.dataset.backend)) === 'cuda' && (await page.$eval(`${F} [data-follow]`, (e) => e.dataset.stage)) === 'source');
let bad = await walk(stages('cuda', { cubinFits: true }));
check('CUDA, a cubin that fits: six stages, each with its tool, its text, its boxes and where each function is', bad.length === 0, bad[0]);
check('CUDA: the split puts add1 on both sides, main and vecAdd on the host, the kernel on the device', stages('cuda')[1].boxes.map((b) => b.chips.map((c) => c.fn).join(',')).join(' | ') === 'main,vecAdd,add1 | vecAddKernel,add1');
check('CUDA: the placement table gives add1 two compiled functions', await placementOk('cuda'));
check('CUDA: device code is compiled on the user\'s machine only when needed', (await usersMachine()).includes(UM[compilesOnUsersMachine('cuda')]));
await shot('follow-cuda-run', F);
await press(`${F} [data-in="cubin-fits"]`);
let got = await shownStage();
const jit = stages('cuda', { cubinFits: false });
check('CUDA, no cubin for this GPU: the last stage becomes the driver compiling the PTX', same(got, wantStage(jit[5])) && got.tool === 'the device driver' && got.boxes.some((b) => b.id === 'jit'), JSON.stringify(got));
await press(`${F} [data-stage-choice="fatbin"]`);
check('a stage can be chosen directly', same(await shownStage(), wantStage(jit[4])));
await press(`${F} [data-stage-choice="split"]`);
await shot('follow-cuda-split', F);

// HIP.
await press(`${F} [data-backend-choice="hip"]`);
check('switching to HIP starts again at stage 1', (await page.$eval(`${F} [data-follow]`, (e) => e.dataset.stage)) === 'source');
const hipNone = stages('hip', { archs: [], deviceArch: '' });
bad = await walk(hipNone);
check('HIP with no architecture typed: five stages, the device pass asks for a name', bad.length === 0 && hipNone[2].title === 'Device passes: no architecture named yet' && (await fo('title')) === 'Run time: which GPU?', bad[0]);
await setValue(`${F} [data-in="archs"]`, 'archA, archB, archC');
await setValue(`${F} [data-in="device-arch"]`, 'archB');
await press(`${F} [data-stage-choice="source"]`);
const hipRuns = stages('hip', { archs: ['archA', 'archB'], deviceArch: 'archB' });
bad = await walk(hipRuns);
check('HIP, archA and archB named (a third name is ignored), running on archB: one device pass each, and it runs', bad.length === 0 && hipRuns[2].title === 'Device passes: archA, archB' && hipRuns[4].title === 'Run time on archB: runs', bad[0]);
await setValue(`${F} [data-in="device-arch"]`, 'archZ');
got = await shownStage();
const hipNo = stages('hip', { archs: ['archA', 'archB'], deviceArch: 'archZ' });
check('HIP on an architecture not in the list: not runnable, the missing code marked as a failure', same(got, wantStage(hipNo[4])) && got.boxes.find((b) => b.id === 'missing')?.state === 'fail', JSON.stringify(got));
check('HIP: the placement table names amdclang++ passes', await placementOk('hip'));
check('HIP: device code is never compiled on the user\'s machine', (await usersMachine()).includes(UM[compilesOnUsersMachine('hip')]));
await shot('follow-hip-fail', F);

// Metal.
await press(`${F} [data-backend-choice="metal"]`);
bad = await walk(stages('metal', { metalLibrary: 'run' }));
check('Metal, library made at run time (the labs\' route): four stages, three of them at run time', bad.length === 0, bad[0]);
await press(`${F} [data-metal-library="build"]`);
await press(`${F} [data-stage-choice="library"]`);
got = await shownStage();
const metalBuild = stages('metal', { metalLibrary: 'build' });
check('Metal, library made at build time: xcrun metal makes a .metallib', same(got, wantStage(metalBuild[1])) && got.tool === 'xcrun metal' && got.when === 'build time', JSON.stringify(got));
await press(`${F} [data-stage-choice="pipeline"]`);
got = await shownStage();
check('Metal: whichever route, the pipeline state finishes compilation for this GPU at run time', same(got, wantStage(metalBuild[3])) && got.when === 'run time' && got.boxes.find((b) => b.id === 'pipeline')?.state === 'ok');
check('Metal: add1 is written in both files, by hand', await placementOk('metal') && (await placementRows()).find((r) => r.fn === 'add1').cells[1] === 'launch.mm and kernel.metal');
check('Metal: compilation always finishes on the user\'s machine', (await usersMachine()).includes(UM[compilesOnUsersMachine('metal')]));
const firstMetal = stages('metal')[0];
await press(`${F} [data-stage-choice="source"]`);
check('Metal: the helper chip says it is written again in the kernel file', firstMetal.boxes[1].title === 'kernel.metal' && (await text(`${F} [data-box="dev-src"] [data-chip="add1"]`)).includes('written again here'));
await shot('follow-metal-source', F);

// ---------------------------------------------------------------- q-first-call
q = await question('the first launch of a kernel takes far longer');
await pick(q, 'The first launch runs the kernel twice');
check('q-first-call: "runs the kernel twice" is refused', (await qText(q)).includes('Not quite.'));
await pick(q, 'The first call into the runtime pays for start-up work');
t = await qText(q);
check('q-first-call: lazy initialisation and pipeline creation, with the habit that follows', t.includes('Correct.') && t.includes('untimed run first'), t);

// ---------------------------------------------------------------- B: who may call it, who runs it
const W = bench(WHO);
const answerFor = (caller, callee) => {
  const r = legalCall(caller, callee, 'cuda');
  return r.book === 'illegal' ? 'error' : r.book === 'dynamic-parallelism' ? 'new-grid' : r.executedBy;
};
const mark = (caller, callee) => page.$eval(`${W} td[data-cell="${caller}:${callee}"]`, (e) => e.dataset.mark);
const choose = (caller, callee, v) => setValue(`${W} td[data-cell="${caller}:${callee}"] select`, v);
const feedback = () => page.$eval(`${W} [data-out="feedback"]`, (e) => e.textContent.trim());
await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
check('a 3 by 4 grid of open cells', (await page.$$(`${W} td[data-cell]`)).length === CALLERS.length * CALLEES.length && (await page.$$eval(`${W} td[data-cell]`, (cs) => cs.every((c) => c.dataset.mark === 'open'))));
await choose('kernel', 'none', 'caller-device-thread');
check('a kernel calling an unqualified helper, answered "run by the calling device thread": marked no, with a hint and not the answer',
  (await mark('kernel', 'none')) === 'wrong' && (await feedback()).includes('Not this one.') && (await feedback()).includes('This call does not compile.'), await feedback());
await choose('kernel', 'none', 'error');
const plain = legalCall('kernel', 'none', 'cuda');
check('the same cell answered "compile error": right, and the reason names the default', (await mark('kernel', 'none')) === 'right' && (await feedback()).includes(plain.why) && plain.why.includes('No qualifier means __host__'), await feedback());
await choose('host-fn', 'global', 'caller-host-thread');
check('host calls a kernel, "run by the calling host thread": no, it compiles but someone else runs it', (await mark('host-fn', 'global')) === 'wrong' && (await feedback()).includes('It compiles, but that is not who executes the callee.'), await feedback());
await choose('kernel', 'global', 'error');
const dp = BACKENDS.map((b) => `${BACKEND_NAME[b]}: ${DYNAMIC_PARALLELISM[b].text} (${DYNAMIC_PARALLELISM[b].fact})`).join('. ') + '.';
check('a kernel calling a kernel: "depends", with the three backends\' status', (await mark('kernel', 'global')) === 'depends' && (await page.$eval(`${W} [data-out="dp"]`, (e) => e.textContent.trim())) === dp, await feedback());
for (const caller of CALLERS) for (const callee of CALLEES) await choose(caller, callee, answerFor(caller, callee));
const marks = await page.$$eval(`${W} td[data-cell]`, (cs) => cs.map((c) => [c.dataset.cell, c.dataset.mark]));
check('every cell answered from the model is right, or "depends" for the two dynamic-parallelism cells',
  marks.every(([k, m]) => { const [caller, callee] = k.split(':'); return m === (legalCall(caller, callee, 'cuda').book === 'dynamic-parallelism' ? 'depends' : 'right'); }) && marks.filter(([, m]) => m === 'depends').length === 2, JSON.stringify(marks));
check('all twelve settled: the one rule and the default are spelled out', (await text(`${W} [data-out="score"]`)) === '12 of 12 cells settled' && (await text(`${W} [data-out="rule"]`)).includes('a function with no qualifier is a host function'));
await shot('who-done', W);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === 'Start again').click(), W);
await pause(80);
check('Start again clears every cell', (await text(`${W} [data-out="score"]`)) === '0 of 12 cells settled' && !(await exists(`${W} [data-out="rule"]`)));

// ---------------------------------------------------------------- q-plain-helper and q-declare-twice
q = await question('which is declared with no qualifier');
await pick(q, 'It works.');
check('q-plain-helper: "small functions are inlined" is refused', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Inlining happens after'));
await pick(q, 'A compile error, at the call');
check('q-plain-helper: a compile error, explained by the model\'s rule', (await qText(q)).includes('Correct.') && (await qText(q)).includes(plain.why));
q = await question('writes every helper twice');
await pick(q, 'Put __host__ and __device__ on one definition.');
t = await qText(q);
check('q-declare-twice (book exercise 10): one definition with both qualifiers, two compiled versions', t.includes('Correct.') && t.includes(`generates ${placement(PROGRAM.find((f) => f.name === 'add1'), 'cuda').length} versions`), t);

// ---------------------------------------------------------------- the listings
const lab = (rel) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8');
const reports = await (await fetch(BASE + '/api/reports')).json();
const passed = (backend, labId) => {
  const rs = (reports[backend]?.results ?? []).filter((x) => x.lab === labId && x.variant === 'solution' && x.tests.length);
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((tt) => tt.pass));
};
// LabKit's solutionTabStatus, from the shared helper, with this machine's runs and its CUDA
// front-end check (GET /api/frontend-check).
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
const tabStatus = (labId, file) => sharedTabStatus(reports, labId, file, frontend);
const join = (src, names) => names.map((n) => functionSource(src, n)).sort((a, b) => a.firstLine - b.firstLine).map((f) => f.text).join('\n\n');
const figure = (tabs) => page.evaluateHandle((tabs) => [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].map((b) => b.textContent).join('|') === tabs), tabs);
const lines = (fig) => fig.evaluate((f) => [...f.querySelectorAll('pre .ln')].map((l) => l.textContent.replace(/\n$/, '')));
const status = (fig) => fig.evaluate((f) => f.querySelector('.status')?.textContent ?? '');
const helper = await figure('CUDA and HIP|Metal');
const L3 = 'ch02/lab3_elementwise';
check('the shared helper is shown with a CUDA and HIP tab and a Metal tab', !!(await helper.evaluate((f) => !!f)));
check('CUDA and HIP tab: scaleShift, scaleShiftOnHost and scaleShiftKernel from lab 2.3\'s kernel.cu',
  same(await lines(helper), join(lab(L3 + '/solution/kernel.cu'), ['scaleShift', 'scaleShiftOnHost', 'scaleShiftKernel']).split('\n')));
const wantCu = tabStatus(L3, 'kernel.cu');
check('its status follows this machine\'s runs, in the fixed vocabulary', (await status(helper)) === wantCu, `${await status(helper)} / ${wantCu}`);
await helper.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal').click());
await pause(100);
const metalLines = await lines(helper);
check('Metal tab: scale_shift written again, and the kernel that calls it', same(metalLines, join(lab(L3 + '/solution/kernel.metal'), ['scale_shift', 'scale_shift_k']).split('\n')) && metalLines.some((l) => l.includes('static inline float scale_shift(')));
check('the Metal status is "verified on Metal" only with a passing run here', (await status(helper)) === tabStatus(L3, 'kernel.metal'), await status(helper));
const pipe = await figure('Metal host');
const mk = functionSource(lab('ch02/lab1_vecadd/solution/launch.mm'), 'makeVecAddPipeline');
const pipeLines = await pipe.evaluate((f) => [...f.querySelectorAll('pre .ln')].map((l) => [Number(l.dataset.n), l.textContent.replace(/\n$/, '')]));
check('makeVecAddPipeline is shown from lab 2.1\'s launch.mm with the file\'s line numbers',
  same(pipeLines.map((x) => x[1]), mk.text.split('\n')) && pipeLines[0][0] === mk.firstLine && pipeLines.at(-1)[0] === mk.lastLine, JSON.stringify(pipeLines.slice(0, 2)));
check('its status is "verified on Metal" only with a passing run here', (await status(pipe)) === tabStatus('ch02/lab1_vecadd', 'launch.mm'));

// ---------------------------------------------------------------- the Metal call table and q-metal-helper
const mcells = await page.$$eval('table[data-table="metal-calls"] td[data-metal]', (cs) => cs.map((c) => [c.dataset.metal, c.dataset.result]));
check('the Metal table: nine cells, each the model\'s answer',
  mcells.length === 9 && METAL_FNS.every((a) => METAL_FNS.every((b) => mcells.find(([k]) => k === `${a}:${b}`)?.[1] === metalCall(a, b).result)), JSON.stringify(mcells));
check('in it, the host dispatches a kernel and a kernel can call a plain MSL function, not another kernel',
  metalCall('host', 'kernel').result === 'dispatch' && metalCall('kernel', 'msl').result === 'call' && metalCall('kernel', 'kernel').result === 'no' &&
  (await text('table[data-table="metal-calls"] td[data-metal="host:kernel"]')).startsWith('not a call: dispatched'));
q = await question('You port both to');
await pick(q, 'Mark the function kernel');
check('q-metal-helper: "mark it kernel" is refused', (await qText(q)).includes('Not quite.'));
await pick(q, 'No documented way to build one function for both.');
t = await qText(q);
check('q-metal-helper: write it twice, share constants and structs through a header', t.includes('Correct.') && t.includes('one plain header'), t);

// ---------------------------------------------------------------- the page as a whole
const whole = await text('.page');
check('the page cites the book\'s pages and figure and the facts items', whole.includes('§2.7, pp. 39–40') && whole.includes('Fig. 2.14') && whole.includes('Fig. 2.11') && whole.includes('notes/today/ch02-facts.md, A14'));
check('dynamic parallelism by backend: CDP2 on CUDA, not on HIP, nothing documented on Metal',whole.includes('CDP2') && whole.includes('HIP does not support it') && whole.includes('Metal documents nothing like it'));
check('the stepper says what it leaves out', (await page.$$('.note.caveat')).length === 1);
check('answers went to the (intercepted) progress API', puts > 0, String(puts));
const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors, nothing missing', real.length === 0 && missing.length === 0, [...real, ...missing].join(' | '));
if (out) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `${out}-page.png`, fullPage: true });
}

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
