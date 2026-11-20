// Browser check for the chapter 17 labs page: status table, the race line, the reader's
// own status, the two predictions against this machine's measurements, the results table,
// the ratio table and the sorted-rows line, the reference-solution toggle, links and margin
// notes. Everything expected is computed here from /api/reports and /api/lab-file, so the
// check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch17-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that every
// prediction is uncommitted whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') saved.push(r.postData() ?? '');
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, LabKit.tsx)
const LAB = { num: '17.1', id: 'ch17/lab1_spmv', dir: 'lab1_spmv' };
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const MATRICES = ['uniform', 'spread', 'spread_sorted', 'skewed'];
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const benchRows = (run, id, variant) => results(run, id, variant).flatMap((x) => x.bench);
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const solutionText = (reports, b, id) => {
  const s = labStatus(reports[b], id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : 'no run on this machine';
};
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, variant: mine.length ? 'work' : 'solution', rows }] : [];
  });
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const spmv = (set, matrix, format) => set.rows.find((r) => r.name === 'spmv' && r.matrix === matrix && r.format === format);

await goto(page, 'ch17/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Lab"', h1.trim().endsWith('Lab'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 17.1', 'python3 labs/pmpp.py run 17.1 --bench', 'python3 labs/pmpp.py run 17.1 --backend emu',
  'python3 labs/pmpp.py run 17.1 --solution', 'python3 labs/pmpp.py reset 17.1 --yes', 'python3 labs/pmpp.py run 17.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the lab is called optional and adds no milestone', text0.includes('It is optional') && text0.includes('nothing to the through-line project'));
check('both kernels and both launchers are named, with the Metal functions',
  ['spmvCsrKernel', 'launchSpmvCsr', 'spmvEllKernel', 'launchSpmvEll', 'spmv_csr', 'spmv_ell', 'encodeSpmvCsr', 'encodeSpmvEll'].every((s) => text0.includes(s)));
check('the differences from the page are stated: raw pointers, accumulate, stride, -1 pads',
  text0.includes('Raw pointers, not structs.') && text0.includes('The ELL kernel accumulates.') && text0.includes('A stride, not numRows.') && text0.includes('Pads hold column −1 and value 0.'));
check('the tests name what each check catches: y0, the 128 sentinels, the NaN before x[0], the alignment rows',
  text0.includes('y0 + A·x') && text0.includes('128 sentinel elements') && text0.includes('NaN in front of x[0]') && text0.includes('steps by numRows instead of stride'));
check('the tolerance is scaled, never relative-only near zero', text0.includes('1e-4 × (1 + |y0| + Σ|a·x|)') && text0.includes('never relative-only near zero'));
check('Metal specifics: atomic_float, one encoder in order, no NaN guard, the dispatch counter',
  text0.includes('device atomic_float*') && text0.includes('runs them in order') && text0.includes('Metal gets no NaN guard') && text0.includes('puts a counter in place of the encoder'));
check('the four benchmark matrices and the four formats are described',
  MATRICES.every((m) => text0.includes(m)) && ['ell_aligned', 'hybrid at K = 16'].every((s) => text0.includes(s)));
check('the two unrecorded guesses are asked', text0.includes('without recording') && text0.includes('will ell_aligned beat ell?'));
check('the ragged-batch bridge is labelled as ours', text0.includes('A ragged batch is the same problem') &&
  (await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('ragged-batch reading is ours')))));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (no GB/s or measured ms outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.sorted && !p.dataset.raceStatus && !p.dataset.measured).some((p) => /\d+(\.\d+)? GB\/s|\d+\.\d+ ms\b/.test(p.textContent)))));

// ---- status table of the reference solution, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b, LAB.id);
  check(`status ${b}: "${want}"`, c && c.status === want && c.text === want, JSON.stringify(c));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    c && (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
}
if (reports.metal) check('the run line names the Metal device and the time of the run', text0.includes(`metal (${reports.metal.device.name}), ${reports.metal.generated.replace('T', ' ')}`));
await shot('status', 'table[data-table="solution-status"]');

// ---- race flavour line
{
  const run = reports.race;
  const sol = labStatus(run, LAB.id, 'solution');
  const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  check(`the race line reads the reports: solution ${run ? sol : 'no run'}`, race.s === sol &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked on this flavour') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified there')), JSON.stringify(race));
}

// ---- status of the reader's own work
{
  const row = await page.$$eval(`[data-work="${LAB.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return row[b].includes('no run on this machine');
    const rs = results(run, LAB.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, LAB.id, 'work');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check('status of your work matches the reports on every backend', ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const sets = preferred(reports, LAB.id);
const expected = {
  uniform: sets.flatMap((s) => {
    const a = spmv(s, 'uniform', 'csr')?.seconds;
    const b = spmv(s, 'uniform', 'ell')?.seconds;
    return a && b ? [{ backend: s.backend, value: a / b, hyb: null }] : [];
  }),
  skewed: sets.flatMap((s) => {
    const a = spmv(s, 'skewed', 'ell')?.seconds;
    const b = spmv(s, 'skewed', 'csr')?.seconds;
    const h = spmv(s, 'skewed', 'hybrid')?.seconds;
    return a && b ? [{ backend: s.backend, value: a / b, hyb: h ? fmt(b / h) : null }] : [];
  }),
};
const predictions = [
  { key: 'uniform', id: 'ch17/labs/p-lab1-uniform', find: 'CSR time divided by ELL time on your GPU', guess: 1.5 },
  { key: 'skewed', id: 'ch17/labs/p-lab1-skewed', find: 'ELL time divided by CSR time', guess: 3 },
];
for (const p of predictions) {
  const el = (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find)).asElement();
  if (!el) { check(`${p.key}: prediction block present`, false); continue; }
  check(`${p.key}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured and the times behind it`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && / ms/.test(s.text)), JSON.stringify(shown));
    if (p.key === 'skewed')
      check('  skewed: the reveal sets the hybrid beside CSR, from the reports',
        want.every((w) => { const s = shown.find((x) => x.backend === w.backend); return w.hyb ? s.text.includes(`CSR time / hybrid time is ${w.hyb}`) : s.text.includes('no hybrid row'); }), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 17.1 --bench'), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('both predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results table: every timing row in the reports, and nothing invented
const allRows = real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
  return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
});
if (allRows.length) {
  const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  const pct = (r) => (r.format === 'csr' || r.pad_fraction == null ? '' : `${fmt(r.pad_fraction * 100, 3)}%`);
  const head = await page.$$eval(`table[data-results="${LAB.num}"] thead th`, (ths) => ths.map((t) => t.textContent.trim()));
  check('results table columns', head.join('|') === 'Backend|Kernel|Matrix|Format|Width|Stride|Pad|COO entries|Matrix MB|Time, ms|GB/s, useful bytes', head.join('|'));
  check(`results table has the ${allRows.length} rows from the reports, with matrix, format, width, stride, pad, COO, MB, time and GB/s`,
    shown.length === allRows.length && allRows.every((r, i) => shown[i][2] === r.matrix && shown[i][3] === r.format &&
      shown[i][4] === String(r.width ?? '') && shown[i][5] === String(r.stride ?? '') && shown[i][6] === pct(r) &&
      shown[i][7] === fmt(r.coo_entries) && shown[i][8] === fmt(r.matrix_bytes / 1e6) && shown[i][9] === fmt(r.seconds * 1000) && shown[i][10] === fmt(r.gbps)),
    JSON.stringify(shown.slice(0, 3)));
} else {
  const empty = await page.$eval(`[data-results-empty="${LAB.num}"]`, (e) => e.textContent).catch(() => '');
  check('no benchmark rows in the reports, and the empty state says what to run', empty.includes('./verify.sh'), empty);
}
await shot('results', `section.bench[aria-label="Lab ${LAB.num} on this machine"]`);

// ---- the ratio table and the sorted-rows line answer the two unrecorded guesses
{
  const withCsr = sets.flatMap((s) => MATRICES.filter((m) => spmv(s, m, 'csr')).map((m) => ({ s, m })));
  if (withCsr.length) {
    const shown = await page.$$eval('table[data-ratios] tbody tr', (trs) => trs.map((tr) => ({
      backend: tr.dataset.backend, matrix: tr.dataset.matrix,
      ell: tr.querySelector('[data-col="ell"]').textContent, al: tr.querySelector('[data-col="ell_aligned"]').textContent, hyb: tr.querySelector('[data-col="hybrid"]').textContent,
    })));
    const ratio = (s, m, f) => { const a = spmv(s, m, 'csr')?.seconds; const b = spmv(s, m, f)?.seconds; return a && b ? fmt(a / b) : ''; };
    check(`ratio table: CSR time over ell, ell_aligned and hybrid for ${withCsr.length} (backend, matrix) pairs, from the reports`,
      shown.length === withCsr.length && withCsr.every(({ s, m }, i) => shown[i].backend === s.backend && shown[i].matrix === m &&
        shown[i].ell === ratio(s, m, 'ell') && shown[i].al === ratio(s, m, 'ell_aligned') && shown[i].hyb === ratio(s, m, 'hybrid')), JSON.stringify(shown));
    const sorted = await page.$$eval('[data-sorted]', (ps) => ps.map((p) => ({ b: p.dataset.sorted, r: p.dataset.sortedRatio, text: p.textContent })));
    const want = sets.flatMap((s) => {
      const a = spmv(s, 'spread', 'csr')?.seconds;
      const b = spmv(s, 'spread_sorted', 'csr')?.seconds;
      return a && b ? [{ b: s.backend, r: fmt(a / b) }] : [];
    });
    check(`sorted rows: CSR on spread over CSR on spread_sorted (${want.map((w) => `${w.b} ${w.r}`).join(', ')})`,
      sorted.length === want.length && want.every((w) => sorted.some((x) => x.b === w.b && x.r === w.r && x.text.includes('Above 1, sorting the rows made CSR faster'))), JSON.stringify(sorted));
  } else {
    check('no ratios without benchmark rows, and the page says what to run', (await page.$eval('[data-ratios="none"]', (e) => e.textContent).catch(() => '')).includes('run 17.1 --bench'));
  }
}

// ---- reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
const P = `p[data-solution="${LAB.num}"] button`;
check('toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
await page.click(P);
await page.waitForSelector('figure.code .tabs button');
const files = {};
for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch17/${LAB.dir}/solution/${f}`);
check('the three solution files exist in the repository', Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('tabs are CUDA, Metal kernel, Metal host', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
const st = (b) => labStatus(reports[b], LAB.id, 'solution');
const got = [st('cuda') === 'pass' ? 'verified on CUDA' : '', st('hip') === 'pass' ? 'verified on HIP' : '', st('emu') === 'pass' ? 'emulator-checked' : ''].filter(Boolean);
const lacking = ['cuda', 'hip'].filter((b) => st(b) !== 'pass').map((b) => NAME[b]);
const cuStatus = got.length ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified (no run on this machine)';
const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : st('metal') === 'fail' ? 'unverified (failed on this machine)' : 'unverified (no run on this machine)';
const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
for (let i = 0; i < 3; i++) {
  await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
  const [f, status] = order[i];
  await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
  await pause(60);
  const shown = await page.$eval('figure.code pre', (e) => e.textContent);
  const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
  check(`${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
}
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
await pause(80);
check('tier colours are applied: the arrays in global memory, the sum and indices in registers',
  (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', `p[data-solution="${LAB.num}"]`);
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to pages 17.2 and 17.5 and to the summary', ['#/ch17/17.2', '#/ch17/17.5', '#/ch17/wrap'].every((l) => links.includes(l)), links.join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph or another note', misplaced.length === 0, misplaced.join(', '));
const stacked = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => a.previousElementSibling?.tagName === 'ASIDE').length);
check('no two margin notes stack after one paragraph', stacked === 0, String(stacked));

// ---- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
