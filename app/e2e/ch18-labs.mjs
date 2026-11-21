// Browser check for the chapter 18 labs page: commands, status table, the race line, the
// reader's own status, the three predictions against this machine's measurements, the results
// table, the reference-solution toggle, links and margin notes. Everything expected is
// computed here from /api/reports and /api/lab-file, so the check holds on any machine, with
// or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch18-labs.mjs [screenshot-prefix] [--dark]
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
const LAB = { num: '18.1', id: 'ch18/lab1_frontier', dir: 'lab1_frontier' };
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
const row = (set, name, graph) => set.rows.find((r) => r.name === name && r.graph === graph);

await goto(page, 'ch18/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and content
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Lab"', h1.trim().endsWith('Lab'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 18.1', 'python3 labs/pmpp.py run 18.1 --bench', 'python3 labs/pmpp.py run 18.1 --backend emu',
  'python3 labs/pmpp.py run 18.1 --solution', 'python3 labs/pmpp.py reset 18.1 --yes', 'python3 labs/pmpp.py run 18.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat is stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the lab is called optional and adds no milestone', text0.includes('It is optional') && text0.includes('nothing to the through-line project'));
check('part A and part B functions are named, with the Metal functions',
  ['visitVertexAtomically', 'bfsPrivatisedKernel', 'bfsCooperativeKernel', 'launchBfsCooperative', 'launchBfsPrivatised', 'visit_vertex', 'bfs_privatised'].every((s) => text0.includes(s)));
check('Metal has no part B, and says why', text0.includes('Metal has no part B') && text0.includes('no grid barrier, no cooperative launch and no occupancy query'));
check('the differences from the page: line 29, UINT_MAX, hoisted shared, six arguments and 32-bit ranks, guard words, raw pointers',
  ['Line 29 is an atomic store.', 'UINT_MAX for the sentinel.', 'Shared variables at the top.', 'Six arguments, 32-bit ranks.', 'Guard words on the public frontier.', 'Raw pointers.'].every((s) => text0.includes(s)));
check('the backends table names the HIP legacy calls and the Metal weak CAS loop',
  await page.$eval('table[data-table="backends"]', (t) => ['atomicCAS', 'atomicExch', 'atomic_compare_exchange_weak_explicit', 'hip_cooperative_groups.h'].every((s) => t.textContent.includes(s))));
check('the weak-CAS question is labelled unverified', text0.includes('fail a weak compare-and-swap spuriously is not documented: unverified'));
check('the test graphs are described, with the trap and the fan-out', ['stale-expected trap', 'Fan-out', 'hub of degree 5,000', '8 blocks of 128 threads'].every((s) => text0.includes(s)));
check('what each backend can see is stated (emu blind to a missing CAS, race names both lines)',
  text0.includes('cannot see a missing compare-and-swap') && text0.includes('names both source lines'));
check('the two benchmark graphs and the five searches are described',
  ['grid', 'rmat', '2,046 levels'].every((s) => text0.includes(s)) && ['push', 'frontier', 'privatised', 'cooperative', 'floor'].every((s) => text0.includes(s)));
check('the R-MAT generator and the floor are labelled as ours',
  await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('R-MAT generator'))));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (no ms or µs outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.raceStatus && !p.dataset.measured).some((p) => /\d+(\.\d+)? ?(ms|µs|us)\b/.test(p.textContent)))));

// ---- status table of the reference solution, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b, LAB.id);
  check(`status ${b}: "${want}"`, c && c.status === want && c.text === want, JSON.stringify(c));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- race flavour line
{
  const run = reports.race;
  const sol = labStatus(run, LAB.id, 'solution');
  const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  check(`the race line reads the reports: solution ${run ? sol : 'no run'}`, race.s === sol &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified: failed on this machine')), JSON.stringify(race));
}

// ---- status of the reader's own work
{
  const r = await page.$$eval(`[data-work="${LAB.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return r[b].includes('no run on this machine');
    const rs = results(run, LAB.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, LAB.id, 'work');
    if (rs.some((x) => !x.built)) return r[b].includes('does not build');
    if (s === 'none') return r[b].includes('not started');
    return r[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check('status of your work matches the reports on every backend', ok, JSON.stringify(r));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const sets = preferred(reports, LAB.id);
const ratio = (s, a, b, graph) => {
  const x = row(s, a, graph)?.seconds;
  const y = row(s, b, graph)?.seconds;
  return x && y ? [{ backend: s.backend, value: x / y }] : [];
};
const expected = {
  floor: sets.flatMap((s) => ratio(s, 'floor', 'privatised', 'grid')),
  rmat: sets.flatMap((s) => ratio(s, 'frontier', 'privatised', 'rmat')),
  coop: sets.flatMap((s) => ratio(s, 'privatised', 'cooperative', 'grid')),
};
const predictions = [
  { key: 'floor', id: 'ch18/labs/p-lab1-floor', find: 'the share of your privatised search', guess: 0.5, missing: 'run 18.1 --bench' },
  { key: 'rmat', id: 'ch18/labs/p-lab1-rmat', find: 'how many times faster privatised is than frontier', guess: 2, missing: 'run 18.1 --bench' },
  { key: 'coop', id: 'ch18/labs/p-lab1-coop', find: 'how many times faster cooperative is', guess: 3, missing: 'Metal has no part B' },
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
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured and the times behind it`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && / ms/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(p.missing), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('the three predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results table: every timing row in the reports, and nothing invented
const allRows = real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
  return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
});
if (allRows.length) {
  const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  const head = await page.$$eval(`table[data-results="${LAB.num}"] thead th`, (ths) => ths.map((t) => t.textContent.trim()));
  check('results table columns', head.join('|') === 'Backend|Kernel|Search|Graph|Vertices|Edges|Levels|Time, ms|µs per level|Million edges/s', head.join('|'));
  check(`results table has the ${allRows.length} rows from the reports, with graph, sizes, levels, time, µs per level and edges per second`,
    shown.length === allRows.length && allRows.every((r, i) => shown[i][2] === r.name && shown[i][3] === r.graph && shown[i][4] === fmt(r.vertices) &&
      shown[i][5] === fmt(r.edges) && shown[i][6] === String(r.levels) && shown[i][7] === fmt(r.seconds * 1000) && shown[i][8] === fmt(r.us_per_level) && shown[i][9] === fmt(r.medges_per_s)),
    JSON.stringify(shown.slice(0, 3)));
  check('every row is one of the five searches on one of the two graphs',
    allRows.every((r) => ['push', 'frontier', 'privatised', 'cooperative', 'floor'].includes(r.name) && ['grid', 'rmat'].includes(r.graph)));
} else {
  const empty = await page.$eval(`[data-results-empty="${LAB.num}"]`, (e) => e.textContent).catch(() => '');
  check('no benchmark rows in the reports, and the empty state says what to run', empty.includes('./verify.sh'), empty);
}
await shot('results', `section.bench[aria-label="Lab ${LAB.num} on this machine"]`);

// ---- reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
const P = `p[data-solution="${LAB.num}"] button`;
check('toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
await page.click(P);
await page.waitForSelector('figure.code .tabs button');
const files = {};
for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch18/${LAB.dir}/solution/${f}`);
check('the three solution files exist in the repository', Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('tabs are CUDA, Metal kernel, Metal host', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
const st = (b) => labStatus(reports[b], LAB.id, 'solution');
const got = [st('cuda') === 'pass' ? 'verified on CUDA' : '', st('hip') === 'pass' ? 'verified on HIP' : '', st('emu') === 'pass' ? 'emulator-checked' : ''].filter(Boolean);
const lacking = ['cuda', 'hip'].filter((b) => st(b) !== 'pass').map((b) => NAME[b]);
const cuStatus = got.length ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified: no run on this machine';
const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : st('metal') === 'fail' ? 'unverified: failed on this machine' : 'unverified: no run on this machine';
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
check('tier colours are applied: frontiers in global memory, the private queue in shared, indices in registers',
  (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', `p[data-solution="${LAB.num}"]`);
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to page 18.6', links.includes('#/ch18/18.6'), links.join(' '));
for (const l of links) {
  await goto(page, l.slice(2));
  const s = await page.evaluate(() => ({ h1: !!document.querySelector('.page h1'), writing: document.querySelector('.page').textContent.includes('This section is being written') }));
  check(`link ${l} opens a written page`, s.h1 && !s.writing, JSON.stringify(s));
}
await goto(page, 'ch18/labs');
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
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
