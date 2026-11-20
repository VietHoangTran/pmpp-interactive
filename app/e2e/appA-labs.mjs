// Browser check for the appendix A labs page: commands (with the lab named by its
// directory, since the runner does not resolve "A.1"), the status table and the race
// line, the one lab block, its three predictions, the probe and accuracy tables, the
// timed results table and the reference-solution toggle. Everything expected is computed
// here from /api/reports and /api/lab-file, so the check holds on any machine, with or
// without runs.
// Usage: node e2e/appA-labs.mjs [screenshot-prefix] [--dark]
// Environment: APP_URL (default http://localhost:5173), E2E_WIDTH (default 1440).
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every
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

// ---- the same reading of the reports that the page makes (lib/reports.ts, components/LabKit.tsx)
const LAB = { num: 'appA/lab1', id: 'appA/lab1_sumerror', dir: 'lab1_sumerror', name: 'summation error' };
const RUN = 'python3 labs/pmpp.py run appA/lab1';
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
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
// One set of benchmark rows per real backend: the reader's own if it passed and was timed, else the reference.
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows }] : [];
  });
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  if (!(measured > 0)) return 'The measurement is not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};

await goto(page, 'appA/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- page structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions', 'Lab A.1: summation error: order, accumulator width and compensation', 'What this lab feeds'];
check('the page has its standard sections and the lab block, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const h3 = await page.$$eval('.prose h3', (hs) => hs.map((h) => h.textContent.trim()));
check('the lab block has its three parts: testing, predicting, measuring', JSON.stringify(h3) === JSON.stringify(['Testing', 'Before you benchmark, predict', 'What your machine measured']), JSON.stringify(h3));
check('the page title is "Labs"', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- commands, backends, what is ours
const text0 = await body();
for (const c of ['./verify.sh', RUN, `${RUN} --bench`, `${RUN} --backend emu`, `${RUN} --backend race`, `${RUN} --solution`, 'python3 labs/pmpp.py reset appA/lab1 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the page says why the lab is named appA/lab1 and not A.1', text0.includes('pmpp.py run A.1 does not work') && text0.includes('The lessons call it lab A.1'));
check('the three files the reader edits are named', ['kernel.cu', 'kernel.metal', 'launch.mm'].every((f) => text0.includes(`labs/${LAB.id}/work/${f}`)));
const accs = await page.$$eval('table[data-table="accumulators"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
check('the accumulator table has Metal, CUDA, HIP and the emulators, and the emulators have no 16-bit types',
  accs.map((r) => r[0]).join(',') === 'Metal,CUDA,HIP,emu, race' && accs[3][2] === 'none' && accs[3][3] === 'none' && accs[2][3].includes('__hip_bfloat16'), JSON.stringify(accs));
check('the 16-bit CUDA branches are called unverified and not front-end-checked', text0.includes('They are unverified, and not even front-end-checked'));
check('what the tests cannot see includes the GPU compiler and the hint, without giving the fix away',
  text0.includes('What a GPU compiler does to your loop') && text0.includes('The README has a hint, folded away') && !text0.includes('math_mode(safe)'));
const notes = await page.$$eval('.prose aside.note', (as) => as.map((a) => ({ cls: a.className, text: a.textContent })));
check('what is ours is labelled (kernels, probe, bounds, measurements)', notes.some((n) => n.cls.includes('beyond') && n.text.includes('Appendix A shows no code')));
check('vendor facts are Today notes citing the facts file (A13, A16, B1, B5, C1, C3)',
  ['A13', 'A16', 'B1', 'B5', 'C1', 'C3'].every((id) => notes.some((n) => n.cls.includes('today') && n.text.includes(id))), JSON.stringify(notes.map((n) => n.text.slice(0, 60))));
check('no "could not load", no "being written" and no em dash anywhere on the page', !/could not load|being written|—/i.test(text0));

// ---- links lead to sections that exist
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/appA/start', ...['A.1', 'A.2', 'A.3', 'A.4', 'A.5', 'A.6', 'A.7'].map((s) => '#/appA/' + s), '#/appA/labs', '#/appA/wrap']);
check(`links go to sections that exist (${links.join(', ')})`, links.length >= 4 && links.every((l) => known.has(l)), links.join(', '));

// ---- status table of the reference solution, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b, LAB.id);
  check(`status ${b}: ${want}`, c && c.status === want && c.text === want, JSON.stringify(c));
}
check('the status table names the lab by its directory', (await page.$eval('table[data-table="solution-status"] tbody td', (e) => e.textContent)) === `Lab ${LAB.num}, ${LAB.name}`);
{
  const s = labStatus(reports.race, LAB.id, 'solution');
  const n = results(reports.race, LAB.id, 'solution').flatMap((x) => x.tests).length;
  const line = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  const want = s === 'pass' ? `passed all ${n} tests with no race reported` : s === 'fail' ? 'failed: unverified there' : 'No run of this lab on the race backend';
  check(`race line matches the reports: "${want}"`, line.s === s && line.text.includes(want), JSON.stringify(line));
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
    if (!rs.length) return row[b].includes('not started');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check('status of your work matches the reports on every backend', ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const sets = preferred(reports, LAB.id);
const acc = (rows, m, a, d) => rows.find((r) => r.name === 'accuracy' && r.method === m && r.acc === a && r.data === d);
const predictions = [
  {
    id: 'appA/labs/p-lab1-half-loop', find: 'What will your device\'s kernel return?', guess: 1024,
    want: sets.flatMap((s) => { const r = acc(s.rows, 'serial', 'half', 'uniform'); return r && r.result != null ? [{ backend: s.backend, value: r.result }] : []; }),
    missing: 'which only Metal builds on this machine',
  },
  {
    id: 'appA/labs/p-lab1-kahan-cost', find: 'how many times longer does the Kahan kernel take', guess: 2,
    want: sets.flatMap((s) => {
      const t = s.rows.find((r) => r.name === 'time' && r.method === 'tree' && r.acc === 'float');
      const k = s.rows.find((r) => r.name === 'time' && r.method === 'kahan' && r.acc === 'float');
      return t?.seconds && k?.seconds ? [{ backend: s.backend, value: k.seconds / t.seconds }] : [];
    }),
    missing: './verify.sh',
  },
  {
    id: 'appA/labs/p-lab1-coarse', find: 'how many ULPs off will the plain float tree be', guess: 3,
    want: sets.flatMap((s) => { const r = s.rows.find((x) => x.name === 'coarse' && x.method === 'tree' && x.coarse === 4096); return r && r.ulps != null ? [{ backend: s.backend, value: r.ulps }] : []; }),
    missing: './verify.sh',
  },
];
const findPredict = (find) =>
  page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.replace(/\s+/g, ' ').includes(find)), find);
const commit = async (el, guess) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
check('the page has three predictions', (await page.$$eval('.q .kind', (ks) => ks.filter((k) => k.textContent === 'Predict first').length)) === 3);
for (const p of predictions) {
  const el = (await findPredict(p.find)).asElement();
  if (!el) { check(`${p.id}: prediction block present`, false); continue; }
  check(`${p.id}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  if (p.want.length) {
    const ok = p.want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`${p.id}: measured value from /api/reports is shown (${p.want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === p.want.length, JSON.stringify(shown));
    check(`  ${p.id}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.id}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(p.missing), JSON.stringify(shown));
  }
  check(`  ${p.id}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
{
  await pause(700); // the app saves 400 ms after the last change
  const stored = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  check('the three predictions were sent to /api/progress under their ids, ungraded, and none reached the real file',
    predictions.every((p) => stored[p.id]?.value === p.guess && stored[p.id]?.correct === undefined), JSON.stringify(stored).slice(0, 400));
}
await shot('predict', '.q');

// ---- the probe table
{
  const want = sets.flatMap((s) => { const p = s.rows.find((r) => r.name === 'probe'); return p ? [{ backend: s.backend, p }] : []; });
  const shown = await page.$$eval('table[data-table="probe"] tbody tr', (trs) => trs.map((tr) => ({ backend: tr.dataset.backend, cells: [...tr.children].map((td) => td.textContent.trim()) })));
  const yes = (v) => v === true || v === 1 || v === 'true';
  if (want.length)
    check(`the probe table has one row per backend with a probe (${want.map((w) => w.backend).join(', ')}), worded from the row`,
      shown.length === want.length && want.every((w, i) => shown[i].backend === w.backend &&
        shown[i].cells[1] === (w.p.rounding === 'nearest' ? 'rounds to nearest' : 'rounds toward zero') &&
        shown[i].cells[2] === (yes(w.p.keeps_denormals) ? 'kept' : 'flushed to zero') &&
        shown[i].cells[3] === (yes(w.p.fuses_mul_add) ? 'fused, one rounding' : 'two roundings')), JSON.stringify(shown));
  else check('no probe row, and the empty state says what to run', (await page.$eval('[data-probe-empty]', (e) => e.textContent)).includes(`${RUN} --bench`));
}

// ---- the accuracy table: every cell from the reports
{
  const DATA = ['ones', 'uniform', 'mixed', 'mixed-asc', 'mixed-desc', 'cancel', 'softmax', 'big-small'];
  const withAcc = sets.filter((s) => s.rows.some((r) => r.name === 'accuracy'));
  if (withAcc.length) {
    for (const s of withAcc) {
      const shown = await page.$$eval(`table[data-table="accuracy"][data-backend="${s.backend}"] tbody tr`, (trs) =>
        trs.map((tr) => ({ row: tr.dataset.row, cells: Object.fromEntries([...tr.querySelectorAll('td[data-cell]')].map((td) => [td.dataset.cell, td.textContent.trim()])) })));
      const bad = [];
      for (const m of ['serial', 'tree', 'kahan']) for (const a of ['float', 'half', 'bfloat']) {
        const row = shown.find((x) => x.row === `${m}/${a}`);
        for (const d of DATA) {
          const r = acc(s.rows, m, a, d);
          const want = !r ? '' : r.result == null ? 'overflow' : (r.ulps == null ? '' : fmt(r.ulps)) + (r.matches_model === false ? ' *' : '');
          if (!row || row.cells[d] !== want) bad.push(`${m}/${a}/${d}: ${row?.cells[d]} vs ${want}`);
        }
      }
      check(`accuracy table on ${s.backend}: 9 rows x 8 data sets, every cell the reported ULPs (or "overflow")`, shown.length === 9 && bad.length === 0, bad.slice(0, 4).join(' | '));
    }
  } else {
    check('no accuracy rows, and the empty state says what to run', (await page.$eval('[data-accuracy-empty]', (e) => e.textContent)).includes(`${RUN} --bench`));
  }
  await shot('accuracy', 'section.bench[aria-label="Error by order, accumulator and data"]');
}

// ---- the timed results table: every coarse and time row in the reports, nothing invented
{
  const all = real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
    return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
  }).filter((r) => r.name === 'coarse' || r.name === 'time');
  if (all.length) {
    const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const ok = shown.length === all.length && all.every((r, i) => {
      const want = [r.name, r.method, r.acc, String(r.n), String(r.coarse), fmt(r.seconds * 1000), fmt(r.gb_per_s), r.ulps == null ? '' : fmt(r.ulps)];
      return JSON.stringify(shown[i].slice(2)) === JSON.stringify(want);
    });
    check(`the results table has the ${all.length} coarse and time rows from the reports, with their figures`, ok, JSON.stringify(shown.slice(0, 2)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${LAB.num}"]`, (e) => e.textContent).catch(() => '');
    check('no timed rows in the reports, and the empty state says what to run', empty.includes('./verify.sh'), empty);
  }
  await shot('results', `section.bench[aria-label="Lab ${LAB.num} on this machine"]`);
}

// ---- reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
{
  const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
  const P = `p[data-solution="${LAB.num}"] button`;
  check('the toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `appA/${LAB.dir}/solution/${f}`);
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
  check('  tier colours are applied to the real file (global, shared, registers)',
    (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  await shot('solution', `p[data-solution="${LAB.num}"]`);
  await page.click(P);
  await pause(80);
  check('the toggle hides it again', (await page.$$('figure.code')).length === 0);
}

// ---- margin notes follow text, never a bench, a table or a code block
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- a link to another section works
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/appA/A.6').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Algorithm considerations'), { timeout: 10000 });
check('the link to section A.6 opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Algorithm considerations'));

// ---- phone width: tables scroll inside their own boxes, the page does not scroll sideways
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'appA/labs');
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="accumulators"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
