// Browser check for the chapter 6 labs page: status table, the race-backend line, six
// predictions against this machine's measurements, results tables, the milestone line and
// the reference-solution toggles. Everything expected is computed here from /api/reports
// and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch06-labs.mjs [screenshot-prefix] [--dark]
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that no
// prediction is already committed whatever the reader has done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, LabKit)
const LABS = [
  { num: '6.1', id: 'ch06/lab1_coalescing', dir: 'lab1_coalescing' },
  { num: '6.2', id: 'ch06/lab2_vector', dir: 'lab2_vector' },
  { num: '6.3', id: 'ch06/lab3_banks', dir: 'lab3_banks' },
  { num: '6.4', id: 'ch06/lab4_linear', dir: 'lab4_linear' },
  { num: '6.5', id: 'ch06/lab5_doublebuf', dir: 'lab5_doublebuf' },
];
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
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
const solutionText = (reports, b, id) => {
  const s = labStatus(reports[b], id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : 'no run on this machine';
};
// realBackends(): every backend in the reports except 'emu'.
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu');
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows }] : [];
  });
const allRows = (reports, id) =>
  real(reports).flatMap((b) => [...(labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : []), ...benchRows(reports[b], id, 'solution')]);
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const where = (rows, name, w) => rows.find((r) => r.name === name && Object.entries(w).every(([k, v]) => r[k] === v));

await goto(page, 'ch06/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure and standard sections
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Labs" of chapter 6', h1.includes('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions',
  'Lab 6.1: coalescing on your device', 'Lab 6.2: vector loads, with the boundaries done right', 'Lab 6.3: provoke a bank conflict, then pad it away',
  'Lab 6.4: the linear layer in the framework layout, corner-turned (milestone M2)', 'Lab 6.5: double buffering, with a race detector watching'];
check('the page has its sections in order, one per lab', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 6.1', 'python3 labs/pmpp.py run 6.1 --bench', 'python3 labs/pmpp.py run 6.1 --backend emu',
  'python3 labs/pmpp.py run 6.1 --solution', 'python3 labs/pmpp.py reset 6.1 --yes', 'python3 labs/pmpp.py run 6.5 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the race flavour of the emulator is described', text0.includes('runs the emulated threads concurrently under ThreadSanitizer'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 5 labs x 4 backends', cells.length === 20, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`,
    mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the reader's own work
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

// ---- the race backend line for lab 6.5
const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent }));
const raceSol = labStatus(reports.race, LABS[4].id, 'solution');
const raceWork = labStatus(reports.race, LABS[4].id, 'work');
check(`lab 6.5: the race-backend line reads the reports (solution ${raceSol}, your work ${raceWork})`,
  race.s === raceSol && race.w === raceWork &&
  (!reports.race ? race.text.includes('No run on the race backend') : raceSol === 'pass' ? race.text.includes('emulator-checked under both flavours') : true), JSON.stringify(race));
if (reports.race && raceSol === 'pass') {
  const n = results(reports.race, LABS[4].id, 'solution').flatMap((x) => x.tests).length;
  check(`  the line counts the ${n} race-backend tests`, race.text.includes(`all ${n} tests`));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const ms = (r) => `${fmt(r.seconds * 1000)} ms`;
const expected = {
  stride2: preferred(reports, LABS[0].id).flatMap((s) => {
    const r = where(s.rows, 'strided', { stride: 2 });
    return r?.relative_to_stride1 != null ? [{ backend: s.backend, value: r.relative_to_stride1 * 100 }] : [];
  }),
  vec4: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = where(s.rows, 'vecadd', { variant: 'vec4' });
    return r?.speedup_vs_scalar != null ? [{ backend: s.backend, value: r.speedup_vs_scalar, detail: ms(r) }] : [];
  }),
  probe: preferred(reports, LABS[2].id).flatMap((s) => {
    const r = where(s.rows, 'bank_probe', { stride: 32 });
    return r?.relative_to_stride1 != null ? [{ backend: s.backend, value: r.relative_to_stride1 }] : [];
  }),
  pad: preferred(reports, LABS[2].id).flatMap((s) => {
    const a = where(s.rows, 'transpose', { tile: 32, pitch: 32 });
    const b = where(s.rows, 'transpose', { tile: 32, pitch: 33 });
    return a && b ? [{ backend: s.backend, value: a.seconds / b.seconds }] : [];
  }),
  linear: preferred(reports, LABS[3].id).flatMap((s) => {
    const r = where(s.rows, 'linear', { variant: 'tuned', tile: 16, batch: 256 });
    return r?.speedup_vs_baseline != null ? [{ backend: s.backend, value: r.speedup_vs_baseline }] : [];
  }),
  barrier: preferred(reports, LABS[4].id).flatMap((s) => {
    const r = where(s.rows, 'matmul', { variant: 'double_buffered', tile: 16 });
    return r?.speedup_vs_two_barriers != null ? [{ backend: s.backend, value: r.speedup_vs_two_barriers }] : [];
  }),
};
const predictions = [
  { key: 'stride2', lab: '6.1', find: 'At stride 2, what percentage of stride 1', guess: 50 },
  { key: 'vec4', lab: '6.2', find: '75% fewer load instructions', guess: 2 },
  { key: 'probe', lab: '6.3', find: 'How many times longer per access', guess: 32 },
  { key: 'pad', lab: '6.3 (padding)', find: 'its time at pitch 32 divided by its time at pitch 33', guess: 1.5 },
  { key: 'linear', lab: '6.4', find: 'at batch 256, T = 16', guess: 3 },
  { key: 'barrier', lab: '6.5', find: 'One barrier per phase instead of two', guess: 1.2 },
];
const predictBlocks = await page.$$eval('.q', (qs) => qs.filter((e) => e.querySelector('.kind')?.textContent === 'Predict first').length);
check('the page has six predictions, none committed at the start', predictBlocks === 6 && (await page.$$('[data-measured]')).length === 0, String(predictBlocks));
for (const p of predictions) {
  const q = await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find);
  const el = q.asElement();
  if (!el) { check(`lab ${p.lab}: prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  check(`lab ${p.lab}: nothing is revealed before a prediction is committed`, !!input && (await el.$('[data-measured]')) === null);
  if (!input) continue;
  await input.focus();
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && (!w.detail || s.text.includes(w.detail));
    });
    check(`lab ${p.lab}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  lab ${p.lab}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet'), JSON.stringify(shown));
  }
  check(`  lab ${p.lab}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
  if (p.key === 'stride2') {
    const reading = await el.evaluate((e) => e.textContent);
    check('  lab 6.1: the reading gives the sector model\'s two thirds for this figure, with its reason', reading.includes('two thirds') && reading.includes('stride 2 moves 12 for 8 used'));
    await shot('predict-lab1', 'h2:nth-of-type(4)');
  }
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const tables = [
  { lab: LABS[0], keep: () => true, cell: (r) => (r.name === 'simd_width' ? String(r.width) : fmt(r.gb_per_s)) },
  { lab: LABS[1], keep: (r) => r.name === 'vecadd', cell: (r) => fmt(r.speedup_vs_scalar) },
  { lab: LABS[2], keep: () => true, cell: (r) => (r.name === 'simd_width' ? String(r.width) : r.name === 'bank_probe' ? `${fmt(r.ns_per_access)} ns per access` : `${fmt(r.gb_per_s)} GB/s`) },
  { lab: LABS[3], keep: (r) => r.name === 'linear', cell: (r) => fmt(r.gflops) },
  { lab: LABS[4], keep: (r) => r.name === 'matmul', cell: (r) => fmt(r.gflops) },
];
for (const t of tables) {
  const rows = allRows(reports, t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => shown[i].some((c) => c.includes(t.cell(r)))), JSON.stringify(shown.slice(0, 3)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
const lab1 = allRows(reports, LABS[0].id);
if (lab1.some((r) => r.name === 'transpose')) {
  const shown = await page.$$eval('table[data-results="6.1"] tbody tr', (trs) => trs.map((tr) => tr.children[2].textContent));
  check('lab 6.1: the two transposes are named by which side is strided', shown.some((s) => s.includes('rows read (strided stores)')) && shown.some((s) => s.includes('rows written (strided loads)')));
}
await shot('lab4', 'section.bench[aria-label="Lab 6.4 on this machine"]');

// ---- milestone M2
const m2 = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
const passM2 = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], LABS[3].id, 'work') === 'pass');
check(`milestone M2 status is read from the reports (${passM2.join(',') || 'none'})`, m2.v === (passM2.join(',') || 'none'), JSON.stringify(m2));
check('the milestone names the layout decision and the signature chapter 15 builds on', text0.includes('launchLinear(X, W, Y, batch, in, out)') && text0.includes('(out, in) row-major'));

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch06/${l.dir}/solution/${f}`);
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
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  if (l.num === '6.4') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 6.4: the padded W tile is coloured as shared memory, X/W/Y as global', (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-global')).length > 0);
    await shot('solution-lab4', `p[data-solution="${l.num}"]`);
  }
  if (l.num === '6.2') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 6.2: no identifier colours (x, y, z would colour every .x field)', (await page.$$('figure.code .v-global, figure.code .v-shared, figure.code .v-reg')).length === 0);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch06/6.1', '#/ch06/6.3', '#/ch06/6.4', '#/ch06/6.7', '#/ch06/6.9', '#/ch06/wrap']);
check(`links go to routes of this chapter (${links.join(', ')})`, links.length >= 5 && links.every((l) => known.has(l)), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch06/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="solution-status"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
