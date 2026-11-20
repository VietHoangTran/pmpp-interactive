// Browser check for section 11.3: the Fig. 11.5 listing, the two-buffer stepper (the
// three kernels in several orders, the Fig. 11.4 rows, the missing copy), the outcome
// table, the "where is the answer?" prediction and its replay, the measured prediction,
// and questions. Expected values come from src/sim/ksblock.ts, imported directly.
// Usage: node e2e/ch11-s113.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { EXERCISE_1, FIG_11_3, FIG_11_5, SCHEDULES, kernelSource, levels, resultBuffer, run, sharedBytes } from '../src/sim/ksblock.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
// Later in the script a stand-in report replaces /api/reports (test data only, never shown to the reader).
let fakeReports = false;
const FAKE = {
  testgpu: { generated: 'test', device: { name: 'a stand-in device' }, results: [{ lab: 'ch11/lab1_blockscan', variant: 'solution', backend: 'testgpu', tile: 0, built: true, test_rc: 0, tests: [],
    bench: [{ name: 'scan', kernel: 'two_barrier', block: 256, seconds: 0.01 }, { name: 'scan', kernel: 'two_barrier', block: 1024, seconds: 0.003 }, { name: 'scan', kernel: 'double_buffer', block: 1024, seconds: 0.002 }] }] },
};
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  if (fakeReports && r.method() === 'GET' && r.url().endsWith('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(FAKE) });
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (t) => `section.bench[aria-label="${t}"]`;
const click = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const select = async (scope, aria, value) => {
  await page.$eval(`${scope} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(100);
};
const cells = (scope) =>
  page.$$eval(scope + ' svg[data-part="ks-grid"] g[data-part="cell"]', (gs) =>
    gs.map((g) => ({ row: Number(g.dataset.row), pos: Number(g.dataset.pos), value: g.dataset.value, range: g.dataset.range, state: g.dataset.state })),
  );
const rowValues = (cs, row) => cs.filter((c) => c.row === row).sort((a, b) => a.pos - b.pos).map((c) => c.value);

await goto(page, 'ch11/11.3');
await page.waitForSelector(bench('Two buffers, one barrier'));

// ---- the listing
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('double_buffer_scan_kernel')));
const fig = await page.$$eval('.code', (cs) => cs.map((c) => ({ text: c.textContent, n: c.querySelectorAll('.ln').length, hl: [...c.querySelectorAll('.ln.hl')].map((l) => l.dataset.n) })).find((c) => c.text.includes('Fig. 11.5 from labs/ch11/snippets')));
const lines = kernelSource(FIG_11_5).line;
check('Fig. 11.5 is shown from the snippet file, 23 lines numbered as on the page', !!fig && fig.n === 23, JSON.stringify(fig?.n));
check('the barrier (line 9) and the copy (line 14) are marked', !!fig && same(fig.hl, [...lines.barrier1, ...lines.copy].map(String)), JSON.stringify(fig?.hl));

// ---- two buffers, one barrier
const T = bench('Two buffers, one barrier');
const end = async () => {
  await click(T, 'Run to the end');
  return page.$eval(T + ' [data-readout="outcome"]', (e) => ({ ok: e.dataset.correct === '1', text: e.textContent }));
};
await click(T, 'Step');
let n = await page.$eval(T + ' [data-readout="narrate"]', (e) => e.textContent);
check('step 1: thread 0 loads x0 into buffer1_s', n.includes('Thread 0 loads x0 = 4 into buffer1_s[0]'), n);
await click(T, 'Run to next barrier');
n = await page.$eval(T + ' [data-readout="narrate"]', (e) => e.textContent);
check('run to the next barrier ends on the barrier opening', n.startsWith('The barrier opens'), n);
let o = await end();
check('Fig. 11.5, thread 0 first: correct, output = the scan of exercise 1', o.ok && o.text.includes('output = ' + run({ ...FIG_11_5, threads: 8 }, EXERCISE_1, 'lockstep').reference.join(' ')), o.text);
let cs = await cells(T);
const L = levels(EXERCISE_1);
check('the rows are Fig. 11.4: each iteration\'s buffer holds the model\'s level', [0, 1, 2, 3].every((r) => same(rowValues(cs, r), L[r].values.map(String))), JSON.stringify([1, 2, 3].map((r) => rowValues(cs, r))));
const labels = await page.$$eval(T + ' svg[data-part="ks-grid"] > g > text', (t) => t.map((x) => x.textContent).filter((x) => x.startsWith('buffer')));
check('rows alternate buffer1_s, buffer2_s, buffer1_s, buffer2_s', same(labels, ['buffer1_s', 'buffer2_s', 'buffer1_s', 'buffer2_s']), JSON.stringify(labels));
const copies = await page.$$eval(T + ' line[data-part="arrow"][data-kind="copy"]', (ls) => ls.length);
check('7 copy arrows: 1, 2 and 4 idle threads (Fig. 11.4)', copies === 7, String(copies));
check('every thread ends with inBuffer_s on buffer2_s', (await page.$$eval(T + ' table.threads tbody tr td:nth-child(3)', (t) => t.map((x) => x.textContent))).every((x) => x === '2 / 1'));
for (const s of SCHEDULES) {
  await select(T, 'Thread order', s);
  o = await end();
  check(`Fig. 11.5 in the order "${s}": correct`, o.ok, o.text);
}
await select(T, 'Kernel', 'no-copy');
await select(T, 'Thread order', 'in-order');
const noCopy = run({ ...FIG_11_5, copy: false, threads: 8 }, EXERCISE_1, 'in-order');
o = await end();
check(`copy left out: positions ${noCopy.wrong.join(', ')} are wrong, as the model says`, !o.ok && o.text.includes(`${noCopy.wrong.length} of 8 outputs wrong`) && o.text.includes(`positions ${noCopy.wrong.join(', ')}`), o.text);
cs = await cells(T);
check('the slot nobody copied into is drawn as stale in the iteration-0 row', cs.some((c) => c.row === 1 && c.pos === 0 && c.state === 'stale'));
check('it offers its first failure, a missing copy', await click(T, 'Go to the first failure'));
n = await page.$eval(T + ' [data-readout="narrate"]', (e) => e.textContent);
check('the failure says the idle thread did not copy its value across', n.includes('copy missing') && n.includes('did not copy its value across'), n);
for (const s of ['lockstep', 'reversed']) {
  await select(T, 'Thread order', s);
  o = await end();
  check(`copy left out, order "${s}": wrong in the same positions`, !o.ok && o.text.includes(`positions ${noCopy.wrong.join(', ')}`), o.text);
}
if (out) await page.screenshot({ path: out + '-nocopy.png' });
await select(T, 'Kernel', 'single');
await select(T, 'Thread order', 'in-order');
o = await end();
check('one buffer, barrier 1 only, thread 0 first: wrong', !o.ok && !run({ ...FIG_11_3, barrier2: false, threads: 8 }, EXERCISE_1, 'in-order').correct, o.text);
const src = await page.$eval(T + ' .code pre', (e) => e.textContent);
check('and the code shown is no_barrier2_scan_kernel', src.includes('no_barrier2_scan_kernel'));

// ---- outcomes
const O = bench('Three kernels, every order');
const table = await page.$$eval(O + ' tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)));
const cfgs = [FIG_11_5, { ...FIG_11_5, copy: false }, { ...FIG_11_3, barrier2: false }];
const want = cfgs.map((c) => SCHEDULES.map((s) => (run({ ...c, threads: 8 }, EXERCISE_1, s).correct ? 'correct' : 'wrong')));
check('the outcome table is the model\'s, cell by cell', same(table, want), JSON.stringify(table));
check('two buffers right in every order, no copy wrong in every order', table[0].every((c) => c === 'correct') && table[1].every((c) => c === 'wrong'));

// ---- where is the answer
const pb = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('which array holds the result: 1')));
await (await pb.$('input[type=number]')).type('1');
await (await pb.$('button.primary')).click();
await pause(150);
let rb = await page.$eval('[data-readout="buffer"]', (e) => ({ res: Number(e.dataset.result), text: e.textContent }));
check('the replay: 8 swaps for 256 threads end on buffer1_s, as the model says', rb.res === resultBuffer(256) && rb.res === 1 && rb.text.includes('runs 8 iterations') && rb.text.includes('You had it.'), rb.text);
await page.evaluate(() => [...document.querySelectorAll('[data-readout="buffer"] button')].find((b) => b.textContent === '512').click());
await pause(100);
rb = await page.$eval('[data-readout="buffer"]', (e) => ({ res: Number(e.dataset.result), text: e.textContent }));
check('512 threads: 9 swaps, buffer2_s', rb.res === resultBuffer(512) && rb.res === 2 && rb.text.includes('runs 9 iterations'), rb.text);

// ---- measured prediction
const pd = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('two-barrier kernel will the double-buffered kernel')));
await (await pd.$('input[type=number]')).type('1.3');
await (await pd.$('button.primary')).click();
await page.waitForSelector('[data-readout="measured"]');
const m = await page.$eval('[data-readout="measured"]', (e) => ({ found: e.dataset.found, text: e.textContent }));
check('the measured reveal either reports lab 11.1 or says there is no measurement yet, with no expected value', (m.found === '0' && m.text.includes('No measurement of lab 11.1')) || (m.found === '1' && /ran [\d.]+× as fast/.test(m.text)), JSON.stringify(m));

// The same reveal against the stand-in report (rows shaped like lab 11.1's: "scan" rows
// with a kernel key and a block size), to check that it reads the 1024-thread rows.
fakeReports = true;
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await pause(400);
const pd2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('two-barrier kernel will the double-buffered kernel')));
const committed = await pd2.$('input[type=number]');
if (committed) {
  await committed.type('1.5');
  await (await pd2.$('button.primary')).click();
}
await page.waitForSelector('[data-readout="measured"][data-found="1"]', { timeout: 5000 }).catch(() => null);
const m2 = await page.$eval('[data-readout="measured"]', (e) => e.textContent).catch(() => '');
check('with a report, the reveal compares the 1024-thread rows: 0.003 s against 0.002 s is 1.5×', m2.includes('a stand-in device') && m2.includes('1.5×'), m2);
fakeReports = false;

// ---- questions
const ask = async (text, value) => {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), text);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-bytes accepts 8192', (await ask('How many bytes of shared memory', sharedBytes(1024, 'double'))).includes('Correct.'));
const qb = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('For a 512-thread block')));
await qb.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim() === 'buffer2_s').querySelector('input').click());
await (await qb.$('button.primary')).click();
await pause(120);
check('q-buffer accepts buffer2_s', (await qb.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
