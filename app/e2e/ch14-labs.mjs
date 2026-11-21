// Browser check for the chapter 14 labs page: commands, the status table, the race line of each
// lab, the four predictions against this machine's measurements, the three results tables and
// the three reference-solution toggles. Everything expected is computed here from /api/reports
// and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch14-labs.mjs [screenshot-prefix] [--dark]
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
const LABS = [
  { num: '14.1', id: 'ch14/lab1_twosorts', dir: 'lab1_twosorts', name: 'odd-even and one-bit radix' },
  { num: '14.2', id: 'ch14/lab2_multibit', dir: 'lab2_multibit', name: 'multi-bit, local sort' },
  { num: '14.3', id: 'ch14/lab3_sortrows', dir: 'lab3_sortrows', name: 'sorted rows (M5)' },
];
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };

// The wording of LabKit's solutionTabStatus for a kernel.cu tab: what passed (CUDA, HIP, then the
// emulator), then what failed, then the CUDA-family backends with no run; "unverified: no run on
// this machine" when nothing passed or failed.
const cuTabStatus = (st) => {
  const fam = ['cuda', 'hip', 'emu'];
  const WHERE = { cuda: 'CUDA', hip: 'HIP', emu: 'the emulator' };
  const and = (xs) => (xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));
  const passed = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${WHERE[b]}`));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const none = ['cuda', 'hip'].filter((b) => st(b) === 'none').map((b) => WHERE[b]);
  return !passed.length && !failed.length
    ? 'unverified: no run on this machine'
    : [passed.join(', '), failed.length ? `unverified on ${and(failed)}: failed on this machine` : '', none.length ? `unverified on ${none.join(' and ')}` : '']
        .filter(Boolean).join('; ');
};
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
const row = (set, name, where = {}) => set.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch14/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 30000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Labs"', h1.trim().endsWith('Labs'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 14.1', 'python3 labs/pmpp.py run 14.1 --bench', 'python3 labs/pmpp.py run 14.1 --backend emu',
  'python3 labs/pmpp.py run 14.1 --solution', 'python3 labs/pmpp.py reset 14.1 --yes', 'python3 labs/pmpp.py run 14.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is explained for this chapter, with "evidence, not proof"', text0.includes('ThreadSanitizer') && text0.includes('evidence, not proof'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the provided files are named, in labs/ch14/shared-g11/, and labelled as ours',
  text0.includes('labs/ch14/shared-g11/') && (await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes('provided scan and baseline are ours')))));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
check('the three labs appear in order, lab 14.3 marked as milestone M5',
  ['Lab 14.1: two first sorts, odd-even and one-bit radix', 'Lab 14.2: multi-bit radix sort with local sorting',
    'Lab 14.3: sorted candidates, every row descending and stable (milestone M5)'].every((t, i, a) => h2.indexOf(t) >= 0 && (i === 0 || h2.indexOf(t) > h2.indexOf(a[i - 1]))), JSON.stringify(h2));
check('lab 14.1 lists the five traps of the pages', ['The N + 1 scan.', 'The "zero bucket" slip.', 'The unsigned wrap.', 'The stopping rule.', 'The flag race.'].every((t) => text0.includes(t)));
check('lab 14.1 says why a pass is three launches (gridExclusiveScan is never defined)', text0.includes('The book never') && text0.includes('defines that function') && text0.includes('So a pass here is three launches'));
check('lab 14.2 says it is exercises 1 to 3 and names its three launches', text0.includes("does the book's exercises 1, 2 and 3 at once") && text0.includes('radixLocalSort') && text0.includes('radixScatterLocal'));
check('lab 14.3 states the key, NaN and tie rule, and why bits not floats',
  text0.includes('+NaN before everything and −NaN after everything') && text0.includes('equal values by increasing column') && text0.includes('never perform a float operation'));
check('lab 14.3 gives the library route per backend, with the unstable CUB call named', text0.includes('StableSortPairsDescending') && text0.includes('documented as not stable') && text0.includes('argSort'));
check('the llama.cpp HIP reading is labelled as ours', text0.includes('by our reading') &&
  (await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes('Our reading of llama.cpp')))));
check('the milestone is described with the project conventions and lab 11.6', text0.includes('milestone M5') && text0.includes('row-major float32, batch first') && text0.includes("lab 11.6's top-p keep mask"));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (no GB/s, KiB or measured ms outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.measured && !p.dataset.race).some((p) => /\d+(\.\d+)? (GB\/s|KiB|KB)\b|\d+\.\d+ ms\b/.test(p.textContent)))));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 3 labs x 4 backends', cells.length === 12, String(cells.length));
for (const l of LABS)
  for (const b of BACKENDS) {
    const c = cells.find((x) => x.lab === l.num && x.backend === b);
    const want = solutionText(reports, b, l.id);
    check(`status ${l.num} ${b}: "${want}"`, c && c.status === want && c.text === want &&
      (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
  }
if (reports.metal) check('the run line names the Metal device and the time of the run', text0.includes(`metal (${reports.metal.device.name}), ${reports.metal.generated.replace('T', ' ')}`));
await shot('status', 'table[data-table="solution-status"]');

// ---- race lines, one per lab
for (const l of LABS) {
  const run = reports.race;
  const sol = labStatus(run, l.id, 'solution');
  // The shared RaceStatus line (components/LabKit.tsx).
  const race = await page.$eval(`[data-race-lab="${l.num}"]`, (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent })).catch(() => null);
  check(`race line of lab ${l.num} reads the reports: solution ${run ? sol : 'no run'}`, race && race.s === sol && race.w === labStatus(run, l.id, 'work') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified: failed on this machine')), JSON.stringify(race));
}

// ---- status of the reader's own work
for (const l of LABS) {
  const r = await page.$$eval(`[data-work="${l.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return r[b].includes('no run on this machine');
    const rs = results(run, l.id, 'work');
    const total = rs.reduce((k, x) => k + x.tests.length, 0);
    const passed = rs.reduce((k, x) => k + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, l.id, 'work');
    if (rs.some((x) => !x.built)) return r[b].includes('does not build');
    if (s === 'none') return r[b].includes('not started');
    return r[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(r));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const [L1, L2, L3] = LABS;
const N22 = 4194304;
const VOCAB = 151936;
const expected = {
  oddeven: preferred(reports, L1.id).flatMap((s) => {
    const a = row(s, 'oddeven', { n: 4096 })?.seconds;
    const rx = row(s, 'radix1', { n: 4096 });
    const b = rx?.wall_seconds ?? rx?.seconds;
    return a && b ? [{ backend: s.backend, value: a / b, detail: ['n = 4,096', `${row(s, 'oddeven', { n: 4096 }).steps.toLocaleString('en-US')} steps`] }] : [];
  }),
  cpu1: preferred(reports, L1.id).flatMap((s) => {
    const a = row(s, 'cpu_stable_sort', { n: N22 })?.seconds;
    const b = row(s, 'radix1', { n: N22 })?.seconds;
    return a && b ? [{ backend: s.backend, value: a / b, detail: [`${n(N22)} pairs`, `${fmt(b * 1000)} ms`] }] : [];
  }),
  cpu2: preferred(reports, L2.id).flatMap((s) => {
    const cpu = s.rows.find((r) => r.name === 'cpu_stable_sort' && r.seconds);
    if (!cpu) return [];
    const best = s.rows.filter((r) => r.name === 'local' && r.seconds && r.n === cpu.n).reduce((x, r) => (!x || r.seconds < x.seconds ? r : x), undefined);
    return best ? [{ backend: s.backend, value: cpu.seconds / best.seconds, detail: [`r = ${best.radix_bits}, c = ${best.keys_per_thread}`, `${n(cpu.n)} pairs`] }] : [];
  }),
  batch1: preferred(reports, L3.id).flatMap((s) => {
    const g = row(s, 'sort_rows', { rows: 1, cols: VOCAB })?.seconds;
    const c = row(s, 'cpu_stable_sort_rows', { rows: 1, cols: VOCAB })?.seconds;
    const g64 = row(s, 'sort_rows', { rows: 64, cols: VOCAB })?.seconds;
    const c64 = row(s, 'cpu_stable_sort_rows', { rows: 64, cols: VOCAB })?.seconds;
    return g && c ? [{ backend: s.backend, value: c / g, detail: ['1 × 151,936', g64 && c64 ? `at 64 × 151,936 the ratio is ${fmt(c64 / g64)}` : 'no batch-64 row'] }] : [];
  }),
};
const predictions = [
  { key: 'oddeven', id: 'ch14/labs/p-lab1-oddeven', find: 'how many times longer than your one-bit radix sort', guess: 100, lab: '14.1' },
  { key: 'cpu1', id: 'ch14/labs/p-lab1-cpu', find: 'how many times faster than one CPU thread\'s std::stable_sort will your one-bit radix sort be', guess: 5, lab: '14.1' },
  { key: 'cpu2', id: 'ch14/labs/p-lab2-cpu', find: 'will your fastest configuration be', guess: 20, lab: '14.2' },
  { key: 'batch1', id: 'ch14/labs/p-lab3-batch1', find: 'At batch 1 and 151,936 columns', guess: 2, lab: '14.3' },
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
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && w.detail.every((d) => s.text.includes(d));
    });
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) with its figures, and compared with ${p.guess}`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(`python3 labs/pmpp.py run ${p.lab} --bench`), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all four predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');
check('lab 14.2 points at its two predictions on pages 14.6 and 14.7', await page.$$eval('.prose p', (ps) => ps.some((p) => p.textContent.includes('This lab has three predictions') &&
  [...p.querySelectorAll('a')].some((a) => a.getAttribute('href') === '#/ch14/14.6') && [...p.querySelectorAll('a')].some((a) => a.getAttribute('href') === '#/ch14/14.7'))));

// ---- results tables: every timing row in the reports, and nothing invented
const WHAT = {
  oddeven: 'odd-even sort, whole host loop', radix1: 'one-bit radix sort', cpu_stable_sort: 'std::stable_sort, one CPU thread',
  onebit_global: 'one bit, direct scatter (provided)', local: 'local sort', sort_rows: 'GPU sort', cpu_stable_sort_rows: 'std::stable_sort, one CPU thread',
};
for (const l of LABS) {
  const rows = real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], l.id, 'work') === 'pass' ? benchRows(reports[b], l.id, 'work') : [];
    return [...mine, ...benchRows(reports[b], l.id, 'solution')];
  });
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${l.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const timeCol = l.num === '14.1' ? 5 : l.num === '14.2' ? 7 : 5;
    check(`lab ${l.num}: results table has the ${rows.length} rows from the reports, each with what ran and its time`,
      shown.length === rows.length && rows.every((r, i) => shown[i][2] === WHAT[r.name] && shown[i][timeCol] === fmt(r.seconds * 1000)), JSON.stringify(shown.slice(0, 2)));
    if (l.num === '14.2') {
      const loc = rows.findIndex((r) => r.name === 'local');
      if (loc >= 0) check('lab 14.2: r, c and passes are shown for a local row', shown[loc][4] === String(rows[loc].radix_bits) && shown[loc][5] === String(rows[loc].keys_per_thread) && shown[loc][6] === String(rows[loc].passes), JSON.stringify(shown[loc]));
    }
    if (l.num === '14.3') check('lab 14.3: rows and cols are shown', rows.every((r, i) => shown[i][3] === String(r.rows) && shown[i][4] === n(r.cols)), JSON.stringify(shown.slice(0, 2)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${l.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${l.num}: no benchmark rows, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
  }
}
await shot('results', 'section.bench[aria-label="Lab 14.1 on this machine"]');

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch14/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  const st = (b) => labStatus(reports[b], l.id, 'solution');
  const cuStatus = cuTabStatus(st);
  const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : st('metal') === 'fail' ? 'unverified: failed on this machine' : 'unverified: no run on this machine';
  const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
  for (let i = 0; i < 3; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const [f, status] = order[i];
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 30000 });
    await pause(60);
    const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shownCode, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`lab ${l.num}: tier colours are applied (global and register at least)`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  if (l.num !== '14.1') check(`lab ${l.num}: shared-memory identifiers are coloured`, (await page.$$('figure.code .v-shared')).length > 0);
  await shot(`solution-${l.num}`, `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to 14.4, 14.6, 14.7, 14.8, 14.9 and 10.10', ['#/ch14/14.4', '#/ch14/14.6', '#/ch14/14.7', '#/ch14/14.8', '#/ch14/14.9', '#/ch10/10.10'].every((h) => links.includes(h)), links.join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

// ---- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: `${out}-narrow.png` });

for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
