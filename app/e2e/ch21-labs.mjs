// Browser check for the chapter 21 labs page: status table, the race line, the two
// predictions against this machine's measurements, the results table and the
// reference-solution toggle. Everything expected is computed here from /api/reports and
// /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch21-labs.mjs [screenshot-prefix] [--dark]
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
const LAB = { num: '21.1', id: 'ch21/lab1_dcs', dir: 'lab1_dcs' };
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

await goto(page, 'ch21/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Lab"', h1.trim().endsWith('Lab'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 21.1', 'python3 labs/pmpp.py run 21.1 --bench', 'python3 labs/pmpp.py run 21.1 --backend emu',
  'python3 labs/pmpp.py run 21.1 --solution', 'python3 labs/pmpp.py reset 21.1 --yes', 'python3 labs/pmpp.py run 21.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the lab is called optional and adds no milestone', text0.includes('It is optional') && text0.includes('nothing to the through-line project'));
check('the differences heading claims no blanket testing; tested items say how', text0.includes('Where the lab differs from the printed listings:') && !text0.includes('each difference tested') &&
  text0.includes('runs the printed line and shows it failing') && text0.includes('Tested: the slice offset'));
check('the group width is offered under "Apple", not as "yours"; AMD\'s documented lines are named in the strided reading',
  text0.includes('offer Metal\'s recorded width under "Apple"') && !text0.includes('offers it as yours'));
check('the differences from the printed listings are named (line 05, int k, bounds, cudaMemcpyToSymbol, legacy atomic)',
  text0.includes("Fig. 21.8's line 05 adds an unscaled") && text0.includes('int k') && text0.includes('Fig. 21.5 has no atom count') &&
  text0.includes('it is cudaMemcpyToSymbol') && text0.includes('cuda::atomic_ref'));
check('the page says the emulator cannot see a missing atomic and sends the reader to race',
  text0.includes('it cannot see a missing atomic') && text0.includes('passes there') && text0.includes('ThreadSanitizer'));
check('the Metal notes: atomic_float, dispatchThreadgroups, dispatch_threads_per_threadgroup, Apple7',
  text0.includes('device atomic_float*') && text0.includes('dispatchThreadgroups') && text0.includes('[[dispatch_threads_per_threadgroup]]') && text0.includes('Apple7'));
check('the HIP note names -munsafe-fp-atomics and says the build does not set it', text0.includes('does not pass -munsafe-fp-atomics'));
check('the differently named baseline array is explained (atoms_adj, emulator link)', text0.includes('atoms_adj') && text0.includes('would collide when the files'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
{
  const run = reports.race;
  const sol = labStatus(run, LAB.id, 'solution');
  const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  check(`the race line reads the reports: solution ${run ? sol : 'no run'}`, race.s === (run ? sol : 'none') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked on this flavour') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified there')), JSON.stringify(race));
}

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
const row = (set, name, where = {}) => set.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));
const expected = {
  scatter: preferred(reports, LAB.id).flatMap((s) => {
    const sc = row(s, 'scatter', { nx: 256, chunk: 256 })?.seconds;
    const ga = row(s, 'gather', { nx: 256, chunk: 256 })?.seconds;
    return sc && ga ? [{ backend: s.backend, value: sc / ga, also: [] }] : [];
  }),
  strided: preferred(reports, LAB.id).flatMap((s) => {
    const r = (chunk) => {
      const a = row(s, 'coarse_adjacent', { nx: 1024, chunk })?.seconds;
      const t = row(s, 'coarse_strided', { nx: 1024, chunk })?.seconds;
      return a && t ? a / t : null;
    };
    const v = r(64);
    return v ? [{ backend: s.backend, value: v, also: [256, 512, 1024].filter((c) => r(c)).map((c) => `${fmt(r(c))} at chunk ${c.toLocaleString('en-US')}`) }] : [];
  }),
};
const predictions = [
  { key: 'scatter', id: 'ch21/labs/p-scatter', find: 'how many times faster than your scatter kernel', guess: 20 },
  { key: 'strided', id: 'ch21/labs/p-strided', find: 'how many times faster than the provided adjacent kernel', guess: 1.5 },
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
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && w.also.every((a) => s.text.includes(a));
    });
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured and the times behind it`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && / ms/.test(s.text)), JSON.stringify(shown));
    if (p.key === 'strided') check('  strided: the reveal lists the other three chunk sizes', want.every((w) => w.also.length === 3), JSON.stringify(want));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 21.1 --bench'), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
{
  const t = (await body()).replace(/\s+/g, ' ');
  check('the strided reading calls the shrinking gap our own unverified prediction, and says Apple documents no coalescing rule',
    t.includes('our own prediction, unverified') && t.includes('Apple documents no coalescing rule'));
  check('the strided reading names AMD\'s documented lines (B3, E2) instead of saying no document predicts HIP',
    t.includes('AMD documents grouping into 64- or 128-byte lines') && !t.includes('On Metal and HIP no document predicts'));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('the two predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results table: every timing row in the reports, and nothing invented
const LABEL = {
  scatter: 'scatter (Fig. 21.5)',
  gather: 'gather (Fig. 21.6)',
  coarse_adjacent: 'adjacent (provided, Fig. 21.8 corrected)',
  coarse_strided: 'strided (Fig. 21.10)',
};
const sets = real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
  return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
});
if (sets.length) {
  const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  check(`results table has the ${sets.length} rows from the reports, with kernel, comparison, slice, chunk, time, Gpairs/s and group width`,
    shown.length === sets.length && sets.every((r, i) =>
      shown[i][2] === LABEL[r.name] && shown[i][3] === r.comparison && shown[i][4] === `${r.nx} × ${r.ny}` && shown[i][6] === String(r.chunk) &&
      shown[i][7] === fmt(r.seconds * 1000) && shown[i][8] === fmt(r.pairs_per_s / 1e9) && shown[i][9] === String(r.group_width)),
    JSON.stringify(shown.slice(0, 3)));
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
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch21/${LAB.dir}/solution/${f}`);
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
check('tier colours are applied: atoms constant, energygrid global, the sums in registers',
  (await page.$$('figure.code .v-const')).length > 0 && (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', `p[data-solution="${LAB.num}"]`);
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to page 21.2', links.includes('#/ch21/21.2'), links.join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph or another note', misplaced.length === 0, misplaced.join(', '));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
