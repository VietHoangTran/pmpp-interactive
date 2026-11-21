// Browser check for the chapter 16 labs page: commands, status table, the race line, the
// three predictions against this machine's measurements, the square-padding line, the
// results table and the reference-solution toggle. Everything expected is computed here from
// /api/reports and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch16-labs.mjs [screenshot-prefix] [--dark]
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

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
const LAB = { num: '16.1', id: 'ch16/lab1_wavefront', dir: 'lab1_wavefront' };
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

await goto(page, 'ch16/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Lab"', h1.trim().endsWith('Lab'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 16.1', 'python3 labs/pmpp.py run 16.1 --bench', 'python3 labs/pmpp.py run 16.1 --backend emu',
  'python3 labs/pmpp.py run 16.1 --solution', 'python3 labs/pmpp.py reset 16.1 --yes', 'python3 labs/pmpp.py run 16.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the lab is called optional and adds no milestone', text0.includes('It is optional') && text0.includes('nothing to the through-line project'));
check('every function the reader writes is named, in CUDA and Metal',
  ['max4', 'sqLoadN', 'sqLoadW', 'sqLoadNW', 'sqStoreTile', 'swSquareKernel', 'launchSwSquare', 'hyInitTile', 'hyLoadN', 'hyLoadW', 'hyLoadNW',
    'hyStoreTile', 'swHyperKernel', 'launchSwHyper', 'sw_square', 'sw_hyper', 'encodeSwSquare', 'encodeSwHyper'].every((f) => text0.includes(f)));
check('the files listed are the work/ copies', (await page.$$eval('.prose li code', (cs) => cs.map((c) => c.textContent)))
  .filter((t) => t.startsWith('labs/ch16/lab1_wavefront/work/')).length === 3);
check('the baseline is named as provided and ours', text0.includes('A baseline is provided') && text0.includes('ours is in labs/ch16/shared/'));
check('launch counts are stated: 2·L_seq − 1, 2T − 1, 3T − 1', text0.includes('2·L_seq − 1 launches') && text0.includes('2T − 1 launches') && text0.includes('3T − 1'));
check('Metal specifics: tile passed in, mem_threadgroup and why, multiple of 16, one encoder',
  text0.includes('threadgroup int* swTile [[threadgroup(0)]]') && text0.includes('mem_flags::mem_threadgroup') &&
  text0.includes('written by earlier dispatches') && text0.includes('multiple of 16 bytes') && text0.includes('runs its dispatches in order'));
check('the differences from the page are listed: int arithmetic, exact fronts, stride padding, q >= 1, constants',
  text0.includes('All index arithmetic is int') && text0.includes('through firstRow') && text0.includes('through stride') &&
  text0.includes('The sheared store tests q >= 1') && text0.includes('never defined'));
const shapes = await page.$$eval('table[data-table="test-shapes"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
check('the test-shape table matches the lab README', JSON.stringify(shapes) === JSON.stringify([
  ['emu', '1, 2, 7, 8, 9, 31, 33, 50', '3, 4, 5, 8, 16'], ['race', '9, 20', '4, 8'], ['metal, cuda, hip', '1, 31, 32, 33, 100, 1000, 2049', '8, 16, 32, 64']]), JSON.stringify(shapes));
check('the hidden defect is explained: 0xCD on the emulator, and it may pass on a GPU',
  text0.includes('0xCD') && text0.includes('the same defect may pass there'));
check('the planted defects are listed', text0.includes('instead of <= for the sheared tiles') && text0.includes('without its q_tile == 0 case') && text0.includes('one tile short in a front'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
// (The benchmark's own stopping rule, "at least 20 ms", is a parameter of the driver, not a measurement.)
check('no device figure is typed into the prose (no measured ms or GCUPS outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.squarePad && !p.dataset.raceStatus).some((p) => /\d+\.\d+ ms\b|\d+(\.\d+)? GCUPS/.test(p.textContent)))));

// ---- the README states the same hidden-defect warning
const readme = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch16/lab1_wavefront/README.md')).then((r) => (r.ok ? r.text() : '')));
if (readme) check('the lab README says the missing hyInitTile may pass on a GPU', readme.includes('may pass there') && readme.includes('0xCD'));
else console.log('note  /api/lab-file does not serve README.md; README checked by hand');

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
const row = (set, name, where = {}) => set.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));
const ratio = (num, den) => preferred(reports, LAB.id).flatMap((s) => {
  const a = num(s)?.seconds;
  const b = den(s)?.seconds;
  return a && b ? [{ backend: s.backend, value: a / b }] : [];
});
const expected = {
  baseline: ratio((s) => row(s, 'baseline'), (s) => row(s, 'square', { tile: 32, pad: 0 })),
  hyper: ratio((s) => row(s, 'hyper', { tile: 32, pad: 0 }), (s) => row(s, 'square', { tile: 32, pad: 0 })),
  pad: ratio((s) => row(s, 'hyper', { tile: 32, pad: 1 }), (s) => row(s, 'hyper', { tile: 32, pad: 0 })),
};
const predictions = [
  { key: 'baseline', id: 'ch16/labs/p-lab1-baseline', find: 'How many times faster than the baseline', guess: 4 },
  { key: 'hyper', id: 'ch16/labs/p-lab1-hyper', find: "the sheared kernel's time divided by the square kernel's time", guess: 0.7 },
  { key: 'pad', id: 'ch16/labs/p-lab1-pad', find: 'with one gap per tile row divided by its time without', guess: 1 },
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
    check(`  ${p.key}: the reveal says whose kernel was measured, the times behind it and the table size`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && / ms/.test(s.text) && /L_seq \d/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 16.1 --bench'), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all three predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- which way padding moved the square kernel
{
  const want = ratio((s) => row(s, 'square', { tile: 32, pad: 1 }), (s) => row(s, 'square', { tile: 32, pad: 0 }));
  const shown = await page.$$eval('[data-square-pad]', (ps) => ps.map((p) => ({ b: p.dataset.squarePad, ratio: p.dataset.ratio ?? '', text: p.textContent })));
  if (want.length)
    check(`square padding line: ${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}`,
      want.every((w) => shown.some((s) => s.b === w.backend && s.ratio === fmt(w.value) && s.text.includes(fmt(w.value)))), JSON.stringify(shown));
  else check('no square rows, and the page says what to run', shown.length === 1 && shown[0].b === 'none' && shown[0].text.includes('run 16.1 --bench'));
}

// ---- results table: every timing row in the reports, and nothing invented
const sets = real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
  return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
});
if (sets.length) {
  const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  const label = (r) => (r.name === 'baseline' ? 'baseline (provided)' : r.name === 'square' ? 'square tiles' : r.name === 'hyper' ? 'sheared tiles' : r.name);
  const tiled = (r, k) => (r.name === 'baseline' ? '–' : String(r[k]));
  check(`results table has the ${sets.length} rows from the reports: variant, W, pad, launches, L_seq, time, GCUPS, over baseline`,
    shown.length === sets.length && sets.every((r, i) => shown[i][2] === label(r) && shown[i][3] === tiled(r, 'tile') && shown[i][4] === tiled(r, 'pad') &&
      shown[i][5] === r.launches.toLocaleString('en-US') && shown[i][6] === r.lseq.toLocaleString('en-US') && shown[i][7] === fmt(r.seconds * 1000) &&
      shown[i][8] === fmt(r.gcups) && shown[i][9] === fmt(r.speedup_vs_baseline)), JSON.stringify(shown.slice(0, 2)));
  const launchesOk = sets.every((r) => {
    if (r.name === 'baseline') return r.launches === 2 * r.lseq - 1;
    const T = Math.ceil(r.lseq / r.tile);
    return r.launches === (r.name === 'square' ? 2 * T - 1 : 3 * T - 1);
  });
  check('every row\'s launch count is 2·L_seq − 1, 2T − 1 or 3T − 1, as the page says', launchesOk);
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
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch16/${LAB.dir}/solution/${f}`);
check('the three solution files exist in the repository', Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('tabs are CUDA, Metal kernel, Metal host', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
// LabKit's solutionTabStatus, from the shared helper (it adds this machine's CUDA front-end check).
const cuStatus = tabStatus(reports, LAB.id, 'kernel.cu', frontend);
const metalStatus = tabStatus(reports, LAB.id, 'kernel.metal', frontend);
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
check('tier colours are applied: H global, the tile shared, indices in registers',
  (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', `p[data-solution="${LAB.num}"]`);
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to pages 16.3 and 16.6', links.includes('#/ch16/16.3') && links.includes('#/ch16/16.6'), links.join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));
check('the bank figures are labelled as ours', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('E1'))));

// ---- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: `${out}-narrow.png` });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
