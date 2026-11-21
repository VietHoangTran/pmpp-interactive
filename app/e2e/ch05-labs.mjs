// Browser check for the chapter 5 labs page: status table, predictions against
// this machine's measurements, results tables and the reference-solution toggles.
// Everything expected is computed here from /api/reports and /api/lab-file, so the
// check holds on any machine, with or without runs.
// Usage: node e2e/labs.mjs [screenshot-prefix] [--dark]
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
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
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts)
const LABS = [
  { num: '5.1', id: 'ch05/lab1_roofline', dir: 'lab1_roofline' },
  { num: '5.2', id: 'ch05/lab2_tiled', dir: 'lab2_tiled' },
  { num: '5.3', id: 'ch05/lab3_gemm', dir: 'lab3_gemm' },
  { num: '5.4', id: 'ch05/lab4_racehunt', dir: 'lab4_racehunt' },
  { num: '5.5', id: 'ch05/lab5_dynamic', dir: 'lab5_dynamic' },
];
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  // An untouched starter fails by design: the page reports it as "not started".
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const benchRows = (run, id, variant) => results(run, id, variant).flatMap((x) => x.bench);
const solutionText = (reports, b, id) => {
  const s = labStatus(reports[b], id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : 'no run on this machine';
};
// One set of benchmark rows per real backend: the reader's own if it passed and was timed, else the reference.
const preferred = (reports, id) =>
  Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, variant: mine.length ? 'work' : 'solution', rows }] : [];
  });
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};

await goto(page, 'ch05/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- commands and backends
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 5.2', 'python3 labs/pmpp.py run 5.2 --bench', 'python3 labs/pmpp.py run 5.2 --backend emu',
  'python3 labs/pmpp.py run 5.2 --solution', 'python3 labs/pmpp.py reset 5.2 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 5 labs x 4 backends', cells.length === 20, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  const ok = mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]);
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`, ok, JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
if (reports.metal) {
  const verified = LABS.filter((l) => labStatus(reports.metal, l.id, 'solution') === 'pass').map((l) => l.num);
  check(`Metal results from this machine's reports are in the table (verified on Metal: labs ${verified.join(', ') || 'none'})`,
    verified.every((n) => cells.find((c) => c.lab === n && c.backend === 'metal').text === 'verified on Metal') && verified.length > 0);
  check('the run line names the Metal device and the time of the run',
    text0.includes(`metal (${reports.metal.device.name}), ${reports.metal.generated.replace('T', ' ')}`));
} else console.log('note  no Metal run on this machine; the Metal-specific assertions were skipped');
await shot('status', 'table[data-table="solution-status"]');

// ---- status of the reader's own work
for (const l of LABS) {
  const row = await page.$$eval(`[data-work="${l.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return row[b].includes('no run on this machine');
    const rs = results(run, l.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, l.id, 'work');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const row = (set, name, key, value) => set.rows.find((r) => r.name === name && (key == null || r[key] === value));
const expected = {
  ceiling: preferred(reports, LABS[0].id).flatMap((s) => {
    const r = row(s, 'roofline');
    return r?.naive_gflops && r?.naive_model_ceiling_gflops ? [{ backend: s.backend, value: r.naive_gflops / r.naive_model_ceiling_gflops }] : [];
  }),
  speedup: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = row(s, 'tiled', 'tile', 16);
    return r?.speedup_vs_naive ? [{ backend: s.backend, value: r.speedup_vs_naive }] : [];
  }),
  batch: preferred(reports, LABS[2].id).flatMap((s) => {
    const a = row(s, 'linear_layer', 'batch', 256)?.gflops;
    const b = row(s, 'linear_layer', 'batch', 1)?.gflops;
    return a && b ? [{ backend: s.backend, value: a / b }] : [];
  }),
  // Lab 5.5 against lab 5.2 at T = 16 and at the matrix width lab 5.2's benchmark chose.
  dynamic: preferred(reports, LABS[4].id).flatMap((s) => {
    const f = preferred(reports, LABS[1].id).find((x) => x.backend === s.backend);
    const fix = f && row(f, 'tiled', 'tile', 16);
    const dyn = fix && s.rows.find((r) => r.name === 'tiled_dynamic' && r.tile === 16 && r.width === fix.width);
    const a = dyn?.gflops;
    const b = fix?.gflops;
    return a && b ? [{ backend: s.backend, value: a / b, width: fix.width }] : [];
  }),
};
const predictions = [
  { key: 'ceiling', lab: '5.1', find: 'what multiple of that ceiling', guess: 2 },
  { key: 'speedup', lab: '5.2', find: 'one sixteenth', guess: 16 },
  { key: 'batch', lab: '5.3', find: 'at batch 256', guess: 30 },
  { key: 'dynamic', lab: '5.5', find: 'fixed at compile time', guess: 0.9 },
];
for (const p of predictions) {
  const q = await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find);
  const el = q.asElement();
  if (!el) { check(`lab ${p.lab}: prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  let committed;
  if (input) {
    check(`lab ${p.lab}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
    await input.focus();
    await input.type(String(p.guess));
    await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
    await pause(150);
    committed = p.guess;
  } else {
    // Already committed in the reader's real progress file: read what they committed.
    committed = Number(/You predicted ([-\d.eE]+)/.exec(await el.evaluate((e) => e.textContent))?.[1]);
    console.log(`note  lab ${p.lab}: a prediction of ${committed} was already committed in the progress file; using it`);
  }
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(committed, w.value)) && s.text.includes(`You predicted`) === false;
    });
    check(`lab ${p.lab}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with the prediction of ${committed}: "${compare(committed, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    if (p.key === 'dynamic')
      check('  lab 5.5: the two kernels are compared at one matrix width, the one lab 5.2 chose',
        want.every((w) => shown.find((x) => x.backend === w.backend)?.text.includes(`both at width ${w.width}`)), JSON.stringify(shown));
    check(`  lab ${p.lab}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution|lab 5\.5 from)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet') && shown[0].text.includes('./verify.sh'), JSON.stringify(shown));
  }
  check(`  lab ${p.lab}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${committed}`));
  if (p.key === 'ceiling') await shot('predict-lab1', 'h2:nth-of-type(4)');
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const tables = [
  { lab: LABS[0], keep: (r) => r.name === 'roofline', cell: (r) => fmt(r.naive_gflops) },
  { lab: LABS[1], keep: (r) => r.name === 'naive' || r.name === 'tiled', cell: (r) => fmt(r.gflops) },
  { lab: LABS[2], keep: (r) => r.name === 'linear_layer', cell: (r) => fmt(r.rows_per_s) },
  { lab: LABS[4], keep: (r) => r.name === 'tiled_dynamic', cell: (r) => fmt(r.gflops) },
];
for (const t of tables) {
  const rows = sets(t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => shown[i].includes(t.cell(r))), JSON.stringify(shown));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
check('lab 5.4 says it has no benchmark', (await body()).includes('This lab has no benchmark'));
await shot('lab2', 'section.bench[aria-label="Lab 5.2 on this machine"]');

// ---- reference solutions: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch05/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  // LabKit's solutionTabStatus, from the shared helper (it adds this machine's CUDA front-end check).
  const cuStatus = tabStatus(reports, l.id, 'kernel.cu', frontend);
  const metalStatus = tabStatus(reports, l.id, 'kernel.metal', frontend);
  const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
  for (let i = 0; i < 3; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const [f, status] = order[i];
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
    await pause(60);
    const shown = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  if (l.num === '5.2') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 5.2: tier colours are applied to the real file', (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-global')).length > 0);
    await shot('solution-lab2', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
