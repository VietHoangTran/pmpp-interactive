// Browser check for the chapter 2 labs page: commands, status table, the four lab blocks,
// predictions against this machine's measurements, results tables and the
// reference-solution toggles. Everything expected is computed here from /api/reports and
// /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: node e2e/ch02-labs.mjs [screenshot-prefix] [--dark]
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, components/LabKit.tsx)
const LABS = [
  { num: '2.1', id: 'ch02/lab1_vecadd', dir: 'lab1_vecadd', name: 'vector addition' },
  { num: '2.2', id: 'ch02/lab2_timing', dir: 'lab2_timing', name: 'where the time goes' },
  { num: '2.3', id: 'ch02/lab3_elementwise', dir: 'lab3_elementwise', name: 'elementwise ops (M0)' },
  { num: '2.4', id: 'ch02/lab4_bughunt', dir: 'lab4_bughunt', name: 'bug hunt' },
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
// One set of benchmark rows per backend other than emu: the reader's own if it passed and was timed, else the reference.
const preferred = (reports, id) =>
  Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows }] : [];
  });
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const largest = (rows, name, sync) =>
  rows.filter((r) => r.name === name && (sync === undefined || r.sync === undefined || r.sync === sync)).reduce((best, r) => (best && best.n >= r.n ? best : r), undefined);

await goto(page, 'ch02/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- page structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions',
  'Lab 2.1: vector addition, end to end', 'Lab 2.2: where does the time go?', 'Lab 2.3: the elementwise operations a transformer needs',
  'Lab 2.4: bug hunt, a stub that almost works', 'What these labs leave for later'];
check('the page has its standard sections and one block per lab, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the page title is "Labs"', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- commands, backends, what is ours
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 2.1', 'python3 labs/pmpp.py run 2.1 --bench', 'python3 labs/pmpp.py run 2.1 --backend emu',
  'python3 labs/pmpp.py run 2.1 --solution', 'python3 labs/pmpp.py reset 2.1 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the page says none of these labs is built for the race flavour', text0.includes('None of these labs is built for race'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('every lab names the files the reader edits', LABS.every((l) => ['kernel.cu', 'kernel.metal', 'launch.mm'].every((f) => text0.includes(`labs/${l.id}/work/${f}`))));
check('lab 2.1 sends the reader to the p. 32 erratum before writing CHECK', text0.includes('That snippet does not compile as printed') && text0.includes('does not call exit()'));
check('lab 2.3 is named as milestone M0, with the project conventions',
  text0.includes('milestone M0 of the course project') && text0.includes('row-major float tensors with the batch first') && text0.includes('launch<Name>'));
check('what is ours is labelled (the Metal files, the project)', (await page.$$eval('.prose aside.note.beyond', (a) => a.length)) >= 2);
check('the timing traps are a Today note citing the facts file', (await page.$$eval('.prose aside.note', (as) => as.map((a) => a.textContent))).some((t) => t.includes('A3, A5, B1') && t.includes('hipFree')));
check('no "could not load" and no "being written" anywhere on the page', !/could not load|being written/i.test(text0));

// ---- links lead to sections that exist
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch02/start', '#/ch02/2.1', '#/ch02/2.2', '#/ch02/2.3', '#/ch02/2.4', '#/ch02/2.5', '#/ch02/2.6', '#/ch02/2.7', '#/ch02/labs', '#/ch02/wrap', '#/ch03/labs']);
check(`links go to sections that exist (${links.join(', ')})`, links.length >= 4 && links.every((l) => known.has(l)), links.join(', '));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 4 labs x 4 backends', cells.length === 16, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  const ok = mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]);
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`, ok, JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
if (reports.metal) {
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
    if (!rs.length) return row[b].includes('not started');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const ms = (s) => `${fmt(s * 1000)} ms`;
const expected = {
  speedup: preferred(reports, LABS[0].id).flatMap((s) => {
    const r = largest(s.rows, 'end_to_end');
    return r && r.speedup_vs_cpu != null ? [{ backend: s.backend, value: r.speedup_vs_cpu, words: [ms(r.cpu_s), ms(r.offload_s), `n = ${r.n}`] }] : [];
  }),
  share: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = largest(s.rows, 'phases', 1);
    if (!r || !r.total_s) return [];
    const k = r.kernel_s ?? r.gpu_s;
    return k == null ? [] : [{ backend: s.backend, value: (100 * k) / r.total_s, words: [ms(k), ms(r.total_s)] }];
  }),
  resident: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = largest(s.rows, 'phases', 1);
    const res = r && s.rows.find((x) => x.name === 'resident' && x.n === r.n);
    return r && res && r.total_s && res.seconds ? [{ backend: s.backend, value: r.total_s / res.seconds, words: [ms(r.total_s), ms(res.seconds)] }] : [];
  }),
  nosync: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = largest(s.rows, 'phases', 0);
    return r && r.sync === 0 && r.nosync_kernel_pct != null ? [{ backend: s.backend, value: r.nosync_kernel_pct, words: [ms(r.kernel_s)] }] : [];
  }),
  chain: preferred(reports, LABS[2].id).flatMap((s) => {
    const r = s.rows.find((x) => x.name === 'chain');
    return r && r.ratio != null ? [{ backend: s.backend, value: r.ratio, words: [ms(r.roundtrips_s), ms(r.resident_s), `${r.rows} × ${r.d}`] }] : [];
  }),
};
const predictions = [
  { key: 'speedup', lab: '2.1', id: 'ch02/labs/p-lab1-speedup', find: 'the time of the plain CPU loop of Fig. 2.4', guess: 2, cmd: 'run 2.1 --bench' },
  { key: 'share', lab: '2.2', id: 'ch02/labs/p-lab2-share', find: 'what percent of one host-array call is the kernel', guess: 50, cmd: 'run 2.2 --bench' },
  { key: 'resident', lab: '2.2', id: 'ch02/labs/p-lab2-resident', find: 'divided by the time\n            of the same addition', alt: 'of the same addition on data already on the device', guess: 3, cmd: 'run 2.2 --bench' },
  { key: 'nosync', lab: '2.2', id: 'ch02/labs/p-lab2-nosync', find: 'With the synchronise after the launch removed', guess: 10, cmd: 'machine with CUDA or ROCm' },
  { key: 'chain', lab: '2.3', id: 'ch02/labs/p-lab3-chain', find: 'as three separate round trips', guess: 3, cmd: 'run 2.3 --bench' },
];
const findPredict = (find) =>
  page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.replace(/\s+/g, ' ').includes(find.replace(/\s+/g, ' '))), find);
const commit = async (el, guess) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
check('the page has six predictions', (await page.$$eval('.q .kind', (ks) => ks.filter((k) => k.textContent === 'Predict first').length)) === 6);
for (const p of predictions) {
  const el = (await findPredict(p.alt ?? p.find)).asElement();
  if (!el) { check(`lab ${p.lab} (${p.key}): prediction block present`, false); continue; }
  check(`lab ${p.lab} (${p.key}): nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && w.words.every((x) => s.text.includes(x));
    });
    check(`lab ${p.lab} (${p.key}): measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) with its figures and compared with the prediction of ${p.guess}`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  lab ${p.lab} (${p.key}): the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet') && shown[0].text.includes(p.cmd), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
  if (p.key === 'speedup') await shot('predict-lab1', 'section.bench[aria-label="Lab 2.1 on this machine"]');
}

// ---- lab 2.4: the prediction is a count; the reveal reads how the reader's runs failed
{
  const runs = Object.keys(reports).flatMap((b) => {
    const rs = results(reports[b], LABS[3].id, 'work').filter((x) => x.built && x.test_rc != null);
    if (!rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    const bad = tests.filter((t) => !t.pass);
    const returned = bad.filter((t) => /^scalePrefix returned -?[1-9]/.test(t.detail ?? '')).length;
    const stopped = rs.some((x) => x.test_rc !== 0 && x.test_rc !== 1);
    return [{ backend: b, total: tests.length, failed: bad.length, self: returned + (stopped ? 1 : 0), untouched: reports[b].work_state?.[LABS[3].id] === 'untouched' }];
  });
  const el = (await findPredict('how many will the GPU report by itself')).asElement();
  check('lab 2.4: prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 1);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  if (runs.length) {
    const ok = runs.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === String(w.self) && s.bold === String(w.self) && s.text.includes(`${w.failed} of ${w.total} tests failed`) &&
        s.text.includes(w.untouched ? 'with work/ as shipped' : 'after work/ had been edited');
    });
    check(`lab 2.4: the reveal reads each run of your work (${runs.map((r) => `${r.backend}: ${r.failed} of ${r.total} failed, ${r.self} announced`).join('; ')})`,
      ok && shown.length === runs.length, JSON.stringify(shown));
  } else {
    check('lab 2.4: no run of the work in the reports, and the reveal says what to run',
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 2.4'), JSON.stringify(shown));
  }
  check('  lab 2.4: the committed count is echoed', (await el.evaluate((e) => e.textContent)).includes('You predicted 1 of the defects'));
  check('  lab 2.4: the reveal points to "After the hunt" and says a test counts failures, not defects',
    (await el.evaluate((e) => e.textContent)).includes('After the hunt') && (await el.evaluate((e) => e.textContent)).includes('counts failing tests, not defects'));
  await pause(700); // the app saves 400 ms after the last change
  const stored = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  check('the six predictions were sent to /api/progress under their ids, with the values committed, and none reached the real file',
    predictions.every((p) => stored[p.id]?.value === p.guess) && stored['ch02/labs/p-lab4-reported']?.value === 1 &&
    Object.values(stored).every((a) => a.correct === undefined), JSON.stringify(stored).slice(0, 400));
  await shot('predict-lab4', 'section.bench[aria-label="Lab 2.4 on this machine"]');
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const msCell = (v) => (v == null ? '' : fmt(v * 1000));
const tables = [
  { lab: LABS[0], keep: (r) => r.name === 'end_to_end', cells: (r) => [String(r.n), msCell(r.cpu_s), msCell(r.offload_s), fmt(r.speedup_vs_cpu)] },
  {
    lab: LABS[1], keep: (r) => r.name === 'phases' || r.name === 'resident' || r.name === 'cpu',
    cells: (r) => [String(r.n), msCell(r.name === 'phases' ? r.total_s : r.seconds),
      ...(r.name === 'phases' ? ['wrap_s', 'encode_s', 'wait_s', 'readback_s', 'gpu_s', 'alloc_s', 'h2d_s', 'kernel_s', 'd2h_s', 'free_s'].filter((k) => r[k] != null).map((k) => msCell(r[k])) : []),
      r.name === 'cpu' ? 'CPU loop' : r.name === 'resident' ? 'data already on the device' : 'host arrays'],
  },
  {
    lab: LABS[2], keep: (r) => r.name === 'op' || r.name === 'chain',
    cells: (r) => (r.name === 'chain'
      ? [`blockTail, ${r.rows} × ${r.d}`, msCell(r.resident_s), msCell(r.roundtrips_s), fmt(r.ratio)]
      : [`${r.kernel} alone, data on the device`, msCell(r.seconds), fmt(r.gb_per_s)]),
  },
];
for (const t of tables) {
  const rows = sets(t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => t.cells(r).every((c) => shown[i].some((cell) => cell === c || cell.startsWith(c)))), JSON.stringify(shown.slice(0, 3)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
// Lab 2.2's columns follow the backends that have rows: Metal's phases or the CUDA family's.
{
  const rows = sets(LABS[1].id);
  const heads = await page.$$eval('table[data-results="2.2"] thead th', (ths) => ths.map((t) => t.textContent.trim())).catch(() => []);
  if (rows.length) {
    const metal = rows.some((r) => r.wrap_s != null);
    const cuda = rows.some((r) => r.alloc_s != null);
    check(`lab 2.2: columns follow the rows (${metal ? 'Metal phases' : ''}${metal && cuda ? ' and ' : ''}${cuda ? 'CUDA phases' : ''})`,
      heads.includes('Wait, ms') === metal && heads.includes('GPU time, ms') === metal && heads.includes('Allocate, ms') === cuda && heads.includes('Kernel, ms') === cuda, heads.join(' | '));
  }
}
check('lab 2.4 says it has no benchmark and that the program is meant to fail as shipped',
  (await page.$eval('section.bench[aria-label="Lab 2.4 on this machine"]', (e) => e.textContent)).includes('This lab has no benchmark') &&
  (await body()).includes('As shipped, the program is meant to fail'));
await shot('lab2', 'section.bench[aria-label="Lab 2.2 on this machine"]');
await shot('lab3', 'section.bench[aria-label="Lab 2.3 on this machine"]');

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch02/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  const st = (b) => labStatus(reports[b], l.id, 'solution');
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
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`  lab ${l.num}: tier colours are applied to the real file`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  if (l.num === '2.1') await shot('solution-lab1', `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- margin notes follow text, never a bench or a code block
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- phone width: tables scroll inside their own boxes, the page does not scroll sideways
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(200);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'section.bench[aria-label="Lab 2.2 on this machine"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
