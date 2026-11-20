// Browser check for section 6.3, "Vector loads and stores": lab 6.2's reference solution
// in three tabs with statuses read from the reports, the chunk planner, the speed-up
// prediction and its reveal, and the questions. Every number asserted comes from
// src/sim/vectorize.ts, imported here directly, or is the book's worked figure.
// The reveal is checked twice: against this machine's reports as they are, and against a
// small fixture served in place of /api/reports (clearly not a measurement; it only shows
// that the reveal reads lab 6.2's "vecadd" rows).
// Usage: node e2e/ch06-s63.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { instructions, plan, widestVector } from '../src/sim/vectorize.ts';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const LAB = 'ch06/lab2_vector';
const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
let fixture = null; // when set, served in place of /api/reports
await page.setRequestInterception(true);
page.on('request', (r) => {
  // Leave no trace in the real progress file, and start from a clean one so the prediction is unanswered.
  if (r.url().includes('/api/progress'))
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : '{"version":1}' });
  if (fixture && r.url().includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture) });
  r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const P = 'section.bench[aria-label="Who owns which floats"]';
const outOf = (name) => page.$eval(`${P} [data-out="${name}"]`, (e) => ({ text: e.textContent.replace(/\s+/g, ' ').trim(), value: e.dataset.value }));
const click = async (label, scope = P) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setInput = async (label, v) => {
  await page.evaluate((sel, label, v) => {
    const el = document.querySelector(`${sel} input[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, P, label, v);
  await pause(80);
};
const setSelect = async (label, v) => {
  await page.evaluate((sel, label, v) => {
    const el = document.querySelector(`${sel} select[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, P, label, v);
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
// The project's rule for a reference solution's status (src/lib/reports.ts labStatus).
const passes = (reports, backend) => {
  const rs = reports[backend]?.results.filter((x) => x.lab === LAB && x.variant === 'solution' && x.tests.length) ?? [];
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
};
const statusOf = () => page.$eval('.code .tabs .status', (e) => e.textContent.trim());
const tab = async (label) => {
  await page.evaluate((l) => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent.trim() === l).click(), label);
  await pause(100);
};

await goto(page, 'ch06/6.3');
await page.waitForSelector(P);

// ---- lab 6.2's reference solution
const exists = ['kernel.cu', 'kernel.metal', 'launch.mm'].every((f) => fs.existsSync(path.join(REPO, 'labs', LAB, 'solution', f)));
check('lab 6.2\'s three solution files exist on disk', exists);
await page.waitForSelector('.code .tabs button', { timeout: 10000 });
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('the listing has CUDA, Metal kernel and Metal host tabs', JSON.stringify(tabs) === JSON.stringify(['CUDA', 'Metal kernel', 'Metal host']), tabs.join('|'));
const disk = (f) => fs.readFileSync(path.join(REPO, 'labs', LAB, 'solution', f), 'utf8').replace(/\n$/, '');
const shown = () => page.$eval('.code pre', (e) => e.textContent.replace(/\n$/, ''));
check('the CUDA tab is the file on disk, with the vector kernel', (await shown()) === disk('kernel.cu') && (await shown()).includes('((const float4*)x)[i]'));
check('the float4 lines are highlighted', (await page.$$('.code .ln.hl')).length >= 4);
const reports = await (await fetch(BASE + '/api/reports')).json();
const wantCu = (passes(reports, 'emu') ? 'emulator-checked' : 'unverified: no passing emulator run on this machine') + '; ' + (passes(reports, 'cuda') ? 'verified on CUDA' : 'unverified on CUDA hardware');
const wantMetal = passes(reports, 'metal') ? 'verified on Metal' : 'unverified: no passing Metal run on this machine';
check('the CUDA status is read from the reports, in the project vocabulary', (await statusOf()) === wantCu, `${await statusOf()} vs ${wantCu}`);
await tab('Metal kernel');
check('the Metal tab declares float4 buffers instead of casting', (await shown()) === disk('kernel.metal') && (await shown()).includes('device const float4* x4'));
check('the Metal status is read from the reports', (await statusOf()) === wantMetal, `${await statusOf()} vs ${wantMetal}`);
await tab('Metal host');
check('the Metal host tab binds each buffer twice', (await shown()) === disk('launch.mm') && (await shown()).includes('atIndex:3'));
check('the caption says how the solution differs from the page', (await page.$eval('.code .caption', (e) => e.textContent)).includes('never uses its parameter'));
await tab('CUDA');
if (out) await (await page.$('.code')).screenshot({ path: out + '-code.png' });

// ---- the chunk planner
const state = async () => ({
  threads: Number((await outOf('threads')).text), blocks: Number((await outOf('blocks')).text), chunks: Number((await outOf('chunks')).text),
  tail: (await outOf('tail')).value, idle: Number((await outOf('idle')).text), fewer: Number((await outOf('fewer')).value), widest: Number((await outOf('widest')).text),
});
const roles = () => page.$$eval(`${P} [data-thread]`, (es) => es.map((e) => ({ t: Number(e.dataset.thread), role: e.dataset.role, bad: e.dataset.bad === 'yes' })));
const matches = (s, n, b, w) => {
  const p = plan(n, b, w);
  return s.threads === p.threads && s.blocks === p.blocks && s.chunks === p.chunks && s.tail === String(p.tailThread ?? 'none') && s.idle === p.idle
    && s.fewer === Math.round(instructions(n, w).fewerLoadsPct * 100) / 100;
};
let s = await state();
check('n = 37, 8 per block, float4: the model\'s plan (16 threads, tail thread 9, 6 idle)', matches(s, 37, 8, 4) && s.tail === '9' && s.idle === 6 && s.threads === 16, JSON.stringify(s));
let r = await roles();
check('the drawing marks thread 9 as the tail and 6 idle threads', r.find((x) => x.t === 9)?.role === 'tail' && r.filter((x) => x.role === 'idle').length === 6 && r.filter((x) => x.role === 'vector').length === 9, JSON.stringify(r));
await click('n = 1003, 256 threads per block');
s = await state();
check('n = 1003, 256 per block: thread 250 has the tail, 5 threads idle, one block', matches(s, 1003, 256, 4) && s.tail === '250' && s.idle === 5 && s.blocks === 1 && s.chunks === 251, JSON.stringify(s));
check('with a tail, slightly less than 75 % fewer loads (74.78)', s.fewer === 74.78, String(s.fewer));
r = await roles();
check('the long array is drawn from near its end: thread 250 tail, threads 251 to 255 idle', r.find((x) => x.t === 250)?.role === 'tail' && JSON.stringify(r.filter((x) => x.role === 'idle').map((x) => x.t)) === JSON.stringify([251, 252, 253, 254, 255]) && (await page.$eval(`${P} [data-out="drawing"]`, (e) => e.textContent)).includes('the drawing starts at thread'), JSON.stringify(r.slice(-8)));
if (out) await (await page.$(P)).screenshot({ path: out + '-planner-1003.png' });
await setInput('Elements n', 1024);
s = await state();
check('n = 1024: no tail, no idle thread, exactly 75 % fewer load instructions', matches(s, 1024, 256, 4) && s.tail === 'none' && s.idle === 0 && s.fewer === 75, JSON.stringify(s));
await setInput('Elements n', 1025);
s = await state();
check('n = 1025: one element more costs a whole extra block', matches(s, 1025, 256, 4) && s.blocks === 2 && s.tail === '256' && s.idle === 255, JSON.stringify(s));
await click('n = 37, 8 threads per block');
check('the alignment counter starts at 8 floats for an aligned base', (await state()).widest === widestVector(0, 4) && (await state()).widest === 8);
await click('4');
s = await state();
r = await roles();
let align = (await outOf('align')).text;
check('a base 4 bytes off (x + 1): every float4 access is misaligned and the widest PyTorch would use is 1', s.widest === 1 && align.includes('not allowed') && r.filter((x) => x.role === 'vector').every((x) => x.bad), align);
await click('8');
align = (await outOf('align')).text;
check('8 bytes off: float4 still not allowed, float2 would be', (await state()).widest === 2 && align.includes('not allowed'), align);
await click('2 (float2)');
align = (await outOf('align')).text;
s = await state();
check('switching to float2 at 8 bytes off makes every access legal', !align.includes('not allowed') && (await roles()).every((x) => !x.bad) && matches(s, 37, 8, 2), align);
await click('4 (float4)');
await click('16');
align = (await outOf('align')).text;
check('16 bytes off: float4 is legal again; widest 4', (await state()).widest === 4 && !align.includes('not allowed'), align);
await click('0');
await click('1 (float)');
s = await state();
check('one float per access: no tail, and the scalar launch\'s idle threads', matches(s, 37, 8, 1) && s.tail === 'none' && s.idle === 3 && s.fewer === 0, JSON.stringify(s));
await click('4 (float4)');
await setSelect('Threads per block', 4);
s = await state();
check('4 threads per block: 3 blocks for 10 chunks, 2 idle', matches(s, 37, 4, 4) && s.blocks === 3 && s.idle === 2, JSON.stringify(s));
if (out) await (await page.$(P)).screenshot({ path: out + '-planner.png' });

// ---- questions
check('q-tail accepts 250', (await answerNumber('handles the partial last chunk', 250)).includes('Correct.'));
check('q-instr accepts 75', (await answerNumber('percent fewer load instructions', 75)).includes('Correct.'));
check('q-when: the streaming kernel near its bandwidth', (await answerChoice('help least', 'A simple streaming kernel')).includes('Correct.'));
check('q-align: no', (await answerChoice('a valid access', 'No: x + 1')).includes('Correct.'));
check('q-align explains a wrong pick', (await answerChoice('a valid access', 'Yes: cudaMalloc')).includes('the address you read at is what has to be'));
check('q-metal-index: a wrong pick explains itself', (await answerChoice('Before crediting the 16-byte loads', 'Nothing: wider loads')).includes('differ in a second way'));
const qmi = await answerChoice('Before crediting the 16-byte loads', 'Whether the uint index held the baseline back');
check('q-metal-index: re-measure with int indexing, with Apple\'s uint quote', qmi.includes('Correct.') && qmi.includes('wrapping characteristics of uint'), qmi);
check('q-int4 accepts 32', (await answerNumber('4-bit quantised weights', 32)).includes('Correct.'));

// ---- the prediction, against this machine's reports
const speedRows = Object.entries(reports).filter(([b]) => b !== 'emu').flatMap(([b, run]) => run.results.filter((x) => x.lab === LAB && x.variant === 'solution').flatMap((x) => x.bench).filter((row) => row.name === 'vecadd' && row.variant === 'vec4').map((row) => [b, row.speedup_vs_scalar]));
const Pq = await question('how many times faster is the float4 kernel');
await (await Pq.$('input[type=number]')).type('1.5');
await (await Pq.$('button.primary')).click();
await pause(400);
const measured = await Pq.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((x) => [x.dataset.backend, x.dataset.measured, x.textContent]));
if (speedRows.length)
  check('the reveal shows lab 6.2\'s measured speed-up on this machine', measured.length === speedRows.length && measured.every(([b], i) => b === speedRows[i][0]), JSON.stringify(measured));
else
  check('with no lab 6.2 benchmark on this machine, the reveal says what to run', measured.length === 1 && measured[0][1] === 'none' && measured[0][2].includes('python3 labs/pmpp.py run 6.2 --solution --bench'), JSON.stringify(measured));

// ---- the reveal against a fixture (not a measurement)
fixture = {
  metal: {
    generated: '2026-11-20T00:00:00', device: { name: 'fixture, not a device' },
    results: [{
      lab: LAB, variant: 'solution', backend: 'metal', tile: 0, built: true, test_rc: 0, tests: [{ case: 'fixture', pass: true }],
      bench: [
        { name: 'vecadd', variant: 'scalar', n: 1000, seconds: 3, gb_per_s: 4, speedup_vs_scalar: 1 },
        { name: 'vecadd', variant: 'coarse4', n: 1000, seconds: 2, gb_per_s: 6, speedup_vs_scalar: 1.5 },
        { name: 'vecadd', variant: 'vec4', n: 1000, seconds: 1.5, gb_per_s: 8, speedup_vs_scalar: 2 },
      ],
    }],
  },
};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await pause(400);
const Pf = await question('how many times faster is the float4 kernel');
await (await Pf.$('input[type=number]')).type('1.5');
await (await Pf.$('button.primary')).click();
await pause(400);
const fx = await Pf.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((x) => [x.dataset.backend, x.dataset.measured, x.textContent.replace(/\s+/g, ' ')]));
check('the reveal reads the vec4 row\'s speed-up and names the coarsened kernel\'s beside it', fx.length === 1 && fx[0][0] === 'metal' && fx[0][1] === '2' && fx[0][2].includes('the coarsened kernel without float4 measured 1.5 ×') && fx[0][2].includes('The measurement is 1.33 times your prediction.'), JSON.stringify(fx));
await page.waitForSelector('.code .tabs .status');
await tab('Metal kernel');
check('a passing Metal run turns the Metal status into "verified on Metal"', (await statusOf()) === 'verified on Metal', await statusOf());
// ---- and with no reports at all: the reveal names the command that benchmarks the reference solution
fixture = {};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await pause(400);
const Pn = await question('how many times faster is the float4 kernel');
await (await Pn.$('input[type=number]')).type('1.5');
await (await Pn.$('button.primary')).click();
await pause(400);
const none = await Pn.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((x) => [x.dataset.measured, x.textContent.replace(/\s+/g, ' ')]));
check('no reports: the reveal says to run the solution with --solution --bench, or plain --bench once your kernels pass',
  none.length === 1 && none[0][0] === 'none' && none[0][1].includes('python3 labs/pmpp.py run 6.2 --solution --bench') && none[0][1].includes('run 6.2 --bench once your own kernels pass'), JSON.stringify(none));
fixture = null;
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await pause(400);

// ---- notes
const notes = await page.$$eval('.note', (ns) => ns.map((n) => n.className + ': ' + n.textContent));
check('Today notes carry their check date', notes.filter((n) => n.includes('today')).length >= 3 && notes.filter((n) => n.includes('today')).every((n) => n.includes('Checked 2026-11-20')));
check('the packed-weights bridge is labelled as ours', notes.some((n) => n.includes('beyond') && n.includes('ours')));
check('the planner says what it leaves out', notes.some((n) => n.includes('caveat') && n.includes('counts instructions')));
check('the unverified Metal claim is labelled unverified', notes.some((n) => n.includes('Unverified: that such a cast becomes one 16-byte load')));

// ---- margin notes sit beside the paragraph they belong to (CLAUDE.md), so they open no gaps
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph, never a bench, a listing or another note', misplaced.length === 0, misplaced.join(' | '));

// ---- dark theme and a narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) {
  await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
