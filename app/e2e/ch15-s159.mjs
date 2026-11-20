// Browser check for section 15.9: the "What can this target use?" explorer against the
// facts table it runs on (src/sim/gemmhw.ts, imported here directly), the prediction of
// your M3 kernel against the vendor library with its reveal, and the questions.
// Usage: node e2e/ch15-s159.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FEATURES, TARGETS, summarize } from '../src/sim/gemmhw.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const flat = (s) => s.replace(/\s+/g, ' ').trim();
const HB = 'section.bench[aria-label="What can this target use?"]';

const pick = async (label, id) => {
  await page.select(`${HB} select[aria-label="${label}"]`, id);
  await pause(80);
};
const cellOf = (feature, col) =>
  page.$eval(`${HB} [data-feature="${feature}"] [data-col="${col}"]`, (td) => ({
    status: td.querySelector('[data-status]').dataset.status,
    text: td.textContent.replace(/\s+/g, ' ').trim(),
  }));
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch15/15.9');
await page.waitForSelector(HB);

// ---------------------------------------------------------------- every target, every feature, against the model
const options = await page.$$eval(`${HB} select[aria-label="Target"] option`, (os) => os.map((o) => o.value));
check('every target in the model is offered', TARGETS.every((t) => options.includes(t.id)) && options.length === TARGETS.length, JSON.stringify(options));
let allMatch = true;
const mismatches = [];
for (const t of TARGETS) {
  await pick('Target', t.id);
  const s = summarize(t.id);
  for (const f of FEATURES) {
    const c = await cellOf(f.id, 'a');
    const m = s.cells[f.id];
    const ok = c.status === m.status && c.text.includes(flat(m.what).slice(0, 40)) && m.items.every((i) => c.text.includes(i));
    if (!ok) { allMatch = false; mismatches.push(`${t.id}/${f.id}`); }
  }
  const tk = await page.$(`${HB} [data-toolkit]`);
  if (t.backend === 'CUDA') {
    const text = tk ? flat(await tk.evaluate((e) => e.textContent)) : '';
    if (!text.includes(flat(s.toolkit.what).slice(0, 30))) { allMatch = false; mismatches.push(`${t.id}/toolkit`); }
  } else if (tk) { allMatch = false; mismatches.push(`${t.id}/toolkit shown`); }
}
check('every cell for every target shows the model\'s status, text and facts items', allMatch, mismatches.join(', '));

// ---------------------------------------------------------------- the discoveries the task names
await pick('Target', 'cuda-9.0a');
await pick('Compare with', 'cuda-10.0');
let a = await cellOf('bigMma', 'a');
let b = await cellOf('bigMma', 'b');
check('sm_90a has WGMMA; 10.0 has tcgen05 with TMEM and says WGMMA code does not run there',
  a.status === 'yes' && a.text.includes('WGMMA') && b.status === 'yes' && b.text.includes('TMEM') && b.text.includes('WGMMA is not available'), JSON.stringify([a, b]));
await shot('hw-90a-vs-100', HB);
await pick('Target', 'cuda-9.0');
a = await cellOf('bigMma', 'a');
check('plain sm_90 cannot use WGMMA', a.status === 'no' && a.text.includes('Requires sm_90a'));
await pick('Target', 'cuda-12');
a = await cellOf('bigMma', 'a');
check('RTX 50 (12.x): no TMEM', a.status === 'no' && a.text.includes('No TMEM'));
check('RTX 50 still has LDGSTS and TMA by compute capability', (await cellOf('asyncCopy', 'a')).status === 'yes' && (await cellOf('bulkCopy', 'a')).status === 'yes');
await pick('Target', 'cuda-7.0');
check('Volta: tensor cores, but not a CUDA 13 target', flat(await page.$eval(`${HB} [data-toolkit]`, (e) => e.textContent)).includes('does not list sm_70'));
await pick('Target', 'hip-rdna3');
await pick('Compare with', 'hip-cdna4');
a = await cellOf('warpMma', 'a');
b = await cellOf('warpMma', 'b');
check('RDNA3 has WMMA builtins despite the features table; CDNA4 has MFMA without xf32',
  a.status === 'yes' && a.text.includes('wmma') && a.text.includes('No MFMA') && b.text.includes('does not support xf32'), JSON.stringify([a, b]));
check('no documented asynchronous copy on HIP', (await cellOf('asyncCopy', 'a')).status === 'undocumented' && (await cellOf('asyncCopy', 'b')).status === 'undocumented');
await pick('Target', 'metal-apple7-m4');
await pick('Compare with', 'metal-apple7');
check('Metal 4 adds MPP matmul2d; without Metal 4 there is none', (await cellOf('bigMma', 'a')).status === 'yes' && (await cellOf('bigMma', 'b')).status === 'no');
check('Metal documents no asynchronous copy', (await cellOf('asyncCopy', 'a')).status === 'undocumented' && (await cellOf('asyncCopy', 'a')).text.includes('no "async"'));
const metalText = flat(await page.$eval(`${HB} table`, (e) => e.textContent));
check('the Metal columns never call Apple\'s matrix interfaces tensor cores', !/tensor cores?(?!")/i.test(metalText.replace(/not "tensor cores"/g, '')), metalText.slice(0, 200));
await shot('hw-metal', HB);
await pick('Compare with', '');
check('choosing "nothing" leaves one column', (await page.$$(`${HB} [data-col="b"]`)).length === 0);

// Differences are marked when two targets are compared.
await pick('Target', 'cuda-9.0a');
await pick('Compare with', 'cuda-12');
const differing = await page.$$eval(`${HB} [data-differs]`, (es) => es.map((e) => e.dataset.feature));
const expectDiff = FEATURES.filter((f) => summarize('cuda-9.0a').cells[f.id].status !== summarize('cuda-12').cells[f.id].status).map((f) => f.id);
check('rows whose status differs are marked "differs", as the model says', JSON.stringify(differing) === JSON.stringify(expectDiff) && expectDiff.includes('bigMma'), JSON.stringify(differing));

// At phone width the table becomes one card per feature, with the same cells.
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(250);
check('at phone width the comparison is cards, not a table', !!(await page.$(`${HB} [data-hw-cards]`)) && !(await page.$(`${HB} [data-hw-table]`)));
const narrowCell = await cellOf('bigMma', 'b');
check('the cards show the model\'s cells', narrowCell.status === summarize('cuda-12').cells.bigMma.status && narrowCell.text.includes('No TMEM'));
check('nothing makes the page scroll sideways', (await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)) <= 0);
await shot('hw-narrow', HB);
await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
await pause(250);
await pick('Compare with', '');

// ---------------------------------------------------------------- your M3 against the library
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('percentage of the vendor library')));
await (await pq.$('input[type=number]')).type('40');
await (await pq.$('button.primary')).click();
await page.waitForFunction((el) => el.querySelector('[data-measured]') && el.querySelector('[data-measured]').dataset.measured !== 'loading', {}, pq);
const reports = await (await fetch(BASE + '/api/reports')).json();
const hasRows = Object.entries(reports).some(([bk, run]) => bk !== 'emu' && bk !== 'race' && run.results.some((r) => r.lab.startsWith('ch15/lab6_') && r.bench.some((x) => x.name === 'm3')));
const measured = await pq.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((p) => ({ v: p.dataset.measured, b: p.dataset.backend, text: p.textContent })));
const revealText = await pq.evaluate((e) => e.textContent);
check('the prediction is committed and remembered', revealText.includes('You predicted 40'));
if (hasRows) {
  check('with lab 15.6 rows the reveal shows a percentage per backend', measured.length > 0 && measured.every((m) => Number(m.v) > 0 && m.b), JSON.stringify(measured));
  // The value: m3 over vendor on the 1024 -> 4096 layer at batch 256 (not the 4096 -> 4096 one),
  // from the reader's rows where their work passed, else the reference solution's.
  const LAB = 'ch15/lab6_linear';
  const pass = (run, v) => { const rs = run.results.filter((x) => x.lab === LAB && x.variant === v && x.tests.length); return rs.length && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) && run.work_state?.[LAB] !== 'untouched'; };
  const want = Object.entries(reports).filter(([bk]) => bk !== 'emu' && bk !== 'race').flatMap(([bk, run]) => {
    const rows = (v) => run.results.filter((x) => x.lab === LAB && x.variant === v).flatMap((x) => x.bench);
    const set = pass(run, 'work') && rows('work').length ? rows('work') : rows('solution');
    const at = (n) => set.find((r) => r.name === n && r.batch === 256 && r.in === 1024 && r.out === 4096);
    return at('m3') && at('vendor') ? [{ b: bk, v: (100 * at('m3').gflops) / at('vendor').gflops }] : [];
  });
  check(`  the percentage is m3 over vendor on the 1024 -> 4096 layer at batch 256 (${want.map((w) => `${w.b} ${w.v.toFixed(1)}`).join(', ')})`,
    want.length === measured.length && want.every((w) => { const m = measured.find((x) => x.b === w.b); return m && Math.abs(Number(m.v) - w.v) / w.v < 0.01; }), JSON.stringify(measured));
}
else check('without lab 15.6 rows the reveal says so and names the command', measured.length === 1 && measured[0].v === 'none' && measured[0].text.includes('pmpp.py run 15.6'), JSON.stringify(measured));
await shot('predict', '.q:last-of-type');

// ---------------------------------------------------------------- questions (the correct option is listed first)
const choice = async (needle, idx) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$$('.opts label'))[idx].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
check('q-wgmma: sm_90a code does not run on 10.0', (await choice('built for sm_90a with WGMMA', 0)).includes('Correct.'));
check('q-tmem-rtx: "tcgen05 on both Blackwell families" is wrong', (await choice('largest tensor-core multiply each GPU offers', 2)).includes('Not quite.'));
check('q-ldgsts: check the target and the SASS, then use the primitives', (await choice('still shows the shared-memory fills going through the register file', 0)).includes('Correct.'));
check('q-tma-2d: a 2D tile needs a tensor map', (await choice('fetch the 128 × 8 tile of A through TMA', 0)).includes('Correct.'));
check('q-progression: register pressure', (await choice('progression from WMMA to WGMMA to TMEM', 0)).includes('Correct.'));
check('q-fp32-default: TF32 can be ruled out', (await choice('Which explanation can you rule out without measuring', 0)).includes('Correct.'));
check('q-rdna: "MFMA everywhere" is wrong', (await choice('must use matrix instructions on a CDNA GPU and on an RDNA3 one', 1)).includes('Not quite.'));
check('q-handwrite: when the library lacks the operation', (await choice('write your own GEMM rather than call', 0)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
