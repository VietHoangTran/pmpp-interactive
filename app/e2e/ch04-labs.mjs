// Browser check for the chapter 4 labs page: status table, predictions against this
// machine's measurements, the barrier experiment, results tables, the milestone line and
// the reference-solution toggles. Everything expected is computed here from /api/reports
// and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch04-labs.mjs [screenshot-prefix] [--dark] [--narrow]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const narrow = process.argv.includes('--narrow');
const { browser, page, errors } = await open({ width: narrow ? 390 : 1440, height: 1000 });
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, components/LabKit.tsx)
const LABS = [
  { num: '4.1', id: 'ch04/lab1_launchconfig', dir: 'lab1_launchconfig' },
  { num: '4.2', id: 'ch04/lab2_divergence', dir: 'lab2_divergence' },
  { num: '4.3', id: 'ch04/lab3_barrierhunt', dir: 'lab3_barrierhunt' },
  { num: '4.4', id: 'ch04/lab4_blocksize', dir: 'lab4_blocksize' },
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
// Real backends are every one but emu (LabKit's realBackends); race has no bench rows.
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

await goto(page, 'ch04/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- commands, backends, headings
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 4.2', 'python3 labs/pmpp.py run 4.2 --bench', 'python3 labs/pmpp.py run 4.2 --backend emu',
  'python3 labs/pmpp.py run 4.2 --solution', 'python3 labs/pmpp.py reset 4.2 --yes', 'python3 labs/pmpp.py run 4.3 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent));
for (const t of ['Lab 4.1: ask the device, then size the launch', 'Lab 4.2: what a divergent warp costs', 'Lab 4.3: barrier hunt', 'Lab 4.4: block size on a kernel that waits on memory'])
  check(`heading present: ${t}`, h2.includes(t), h2.join(' | '));
check('no literal "being written" placeholder', !text0.includes('being written'));
check('the page never presents 32 as the warp width without saying whose', !/warps? (are|is) 32/i.test(text0));

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
  const verified = LABS.filter((l) => labStatus(reports.metal, l.id, 'solution') === 'pass').map((l) => l.num);
  check(`Metal results from this machine's reports are in the table (verified on Metal: labs ${verified.join(', ') || 'none'})`,
    verified.every((n) => cells.find((c) => c.lab === n && c.backend === 'metal').text === 'verified on Metal') && verified.length > 0);
} else console.log('note  no Metal run on this machine; the Metal-specific assertions were skipped');
await shot('status', 'table[data-table="solution-status"]');

// ---- the race backend line for lab 4.3
{
  const s = labStatus(reports.race, LABS[2].id, 'solution');
  const line = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  const n = results(reports.race, LABS[2].id, 'solution').flatMap((x) => x.tests).length;
  check(`lab 4.3: race backend line reads the reports (${s})`, line.s === s &&
    (s === 'pass' ? line.text.includes(`passed all ${n} tests on the race backend`) && line.text.includes('emulator-checked') : s === 'fail' ? line.text.includes('unverified there') : line.text.includes('no run on the race backend')), JSON.stringify(line));
}

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

// ---- milestone M1: labs 3.3 and 4.1 of the reader's own work
{
  const both = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], 'ch03/lab3_naive_gemm', 'work') === 'pass' && labStatus(reports[b], LABS[0].id, 'work') === 'pass');
  const m = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
  check(`milestone M1 line reads labs 3.3 and 4.1 from the reports (${both.join(',') || 'none'})`, m.v === (both.join(',') || 'none') &&
    (both.length ? m.text.includes('M1 is done there') : m.text.includes('do not both pass') || m.text.includes('No lab results')), JSON.stringify(m));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const row = (set, pred) => set.rows.find(pred);
const isMixed = (r) => String(r.pattern ?? '').startsWith('half of each');
const expected = {
  speedup: preferred(reports, LABS[0].id).flatMap((s) => {
    const v = row(s, (r) => r.name === 'speedup' && r.batch === 1)?.derived_over_fixed;
    const a = row(s, (r) => r.name === 'linear' && r.config === 'derived' && r.batch === 1)?.seconds;
    const b = row(s, (r) => r.name === 'linear' && r.config === 'fixed16' && r.batch === 1)?.seconds;
    return v && a && b ? [{ backend: s.backend, value: v }] : [];
  }),
  width: preferred(reports, LABS[0].id).flatMap((s) => {
    const q = s.rows.filter((r) => r.name === 'queried');
    const r = q.find((x) => x.device === 0) ?? q[0];
    const v = s.backend === 'metal' ? r?.threadExecutionWidth : r?.warpSize;
    return v ? [{ backend: s.backend, value: v }] : [];
  }),
  divcost: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = row(s, (x) => x.name === 'branch' && isMixed(x));
    return r?.ratio_to_sorted && r.seconds && r.sorted_seconds ? [{ backend: s.backend, value: r.ratio_to_sorted }] : [];
  }),
  select: preferred(reports, LABS[1].id).flatMap((s) => {
    const sel = row(s, (x) => x.name === 'select' && isMixed(x));
    const br = row(s, (x) => x.name === 'branch' && isMixed(x));
    return sel?.ratio_to_branch && sel.seconds && br?.seconds ? [{ backend: s.backend, value: sel.ratio_to_branch }] : [];
  }),
  blocksize: preferred(reports, LABS[3].id).flatMap((s) => {
    const r = row(s, (x) => x.name === 'summary');
    return r?.slowest_over_fastest ? [{ backend: s.backend, value: r.slowest_over_fastest }] : [];
  }),
};
const predictions = [
  { key: 'speedup', lab: '4.1', find: 'fixed 16 × 16 blocks. How many times as fast', guess: 1.5 },
  { key: 'width', lab: '4.1', find: 'the width your device will report', guess: 32 },
  { key: 'divcost', lab: '4.2', find: 'mixed half and half inside every warp', guess: 2 },
  { key: 'select', lab: '4.2', find: 'computing both paths and selecting', guess: 1.2 },
  { key: 'blocksize', lab: '4.4', find: 'slowest block size take as the fastest', guess: 1.3 },
];
for (const p of predictions) {
  const q = await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find);
  const el = q.asElement();
  if (!el) { check(`lab ${p.lab} (${p.key}): prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  check(`lab ${p.lab} (${p.key}): nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
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
    check(`lab ${p.lab} (${p.key}): measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  lab ${p.lab} (${p.key}): the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet') && shown[0].text.includes(`pmpp.py run ${p.lab}`), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
  if (p.key === 'divcost') await shot('predict-lab2', 'section.bench[aria-label="Lab 4.2 on this machine"]');
}

// ---- the barrier experiment: a four-way prediction, then the last GPU run of work/
{
  const OUT = ['Neither fails: both block sizes pass every launch.', 'Only the block four warps wide fails.', 'Only the block one warp wide fails.', 'Both fail.'];
  const el = await page.$('[data-predict="ch04/labs/p-lab3-barrier"]');
  check('lab 4.3: the barrier prediction is present with four outcomes', !!el && (await el.$$('input[type=radio]')).length === 4);
  const commit = async () => (await el.evaluateHandle((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction'))).asElement();
  check('lab 4.3: commit is disabled until an outcome is picked', await (await commit()).evaluate((b) => b.disabled));
  check('lab 4.3: nothing is revealed before committing', (await el.$('[data-measured]')) === null);
  const pick = 1;
  await (await el.$$('input[type=radio]'))[pick].click();
  await (await commit()).evaluate((b) => b.click());
  await pause(150);
  const txt = await el.evaluate((e) => e.textContent);
  check('lab 4.3: the committed outcome is echoed', txt.includes(`You predicted: ${OUT[pick]}`));
  const runs = ['metal', 'cuda', 'hip'].flatMap((b) => {
    const run = reports[b];
    if (!run) return [];
    const tests = results(run, LABS[2].id, 'work').flatMap((x) => x.tests);
    const one = tests.filter((t) => t.case.includes('(1 x width)'));
    const four = tests.filter((t) => t.case.includes('(4 x width)'));
    if (!one.length || !four.length) return [];
    const outcome = (one.some((t) => !t.pass) ? 2 : 0) + (four.some((t) => !t.pass) ? 1 : 0);
    return [{ backend: b, outcome, untouched: run.work_state?.[LABS[2].id] === 'untouched', one, four }];
  });
  const shown = await el.$$eval('[data-measured]', (ds) => ds.map((d) => ({ m: d.dataset.measured, b: d.dataset.backend ?? '', text: d.textContent })));
  if (runs.length) {
    const ok = runs.every((r) => {
      const s = shown.find((x) => x.b === r.backend);
      const right = r.untouched ? s?.text.includes('not the experiment yet') : s?.text.includes(r.outcome === pick ? 'That is the outcome you predicted.' : 'That is not the outcome you predicted.');
      const counts = [...r.one, ...r.four].filter((t) => !t.pass).map((t) => /wrong in \d+ of \d+ launches/.exec(t.detail ?? '')?.[0]).filter(Boolean);
      return s && Number(s.m) === r.outcome && s.text.includes(OUT[r.outcome]) && right && counts.every((c) => s.text.includes(c));
    });
    check(`lab 4.3: the last GPU run of work/ is read case by case (${runs.map((r) => `${r.backend}: ${OUT[r.outcome]}${r.untouched ? ', untouched starter' : ''}`).join('; ')})`, ok && shown.length === runs.length, JSON.stringify(shown));
  } else {
    check('lab 4.3: no GPU run of work/, and the reveal says what to run', shown.length === 1 && shown[0].m === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 4.3'), JSON.stringify(shown));
  }
  check('lab 4.3: the reveal says when the run is the experiment', txt.includes('It is the experiment only if'));
  await shot('barrier', '[data-predict="ch04/labs/p-lab3-barrier"]');
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const tables = [
  { lab: LABS[0], keep: (r) => r.name === 'linear', cell: (r) => fmt(r.gflops) },
  { lab: LABS[1], keep: (r) => ['branch', 'select', 'loop'].includes(r.name), cell: (r) => fmt(r.ratio_to_sorted) },
  { lab: LABS[3], keep: (r) => r.name === 'sweep', cell: (r) => fmt(r.lookups_per_s / 1e6, 4) },
];
for (const t of tables) {
  const rows = sets(t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => shown[i].includes(t.cell(r))), JSON.stringify(shown.slice(0, 3)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
{
  const rows = sets(LABS[0].id).filter((r) => r.name === 'linear');
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="4.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check('lab 4.1: block and grid are shown as x × y from the rows', rows.every((r, i) => shown[i].includes(`${r.block_x} × ${r.block_y}`) && shown[i].includes(`${r.grid_x} × ${r.grid_y}`)), JSON.stringify(shown.slice(0, 2)));
  }
}
check('lab 4.3 says it has no benchmark', (await body()).includes('This lab has no benchmark'));
await shot('lab4', 'section.bench[aria-label="Lab 4.4 on this machine"]');

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch04/${l.dir}/solution/${f}`);
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
  if (l.num === '4.4') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 4.4: tier colours are applied to the real file', (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
    await shot('solution-lab4', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links to other sections resolve to routes of this chapter
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check(`cross-links point at chapter routes (${links.join(', ')})`, links.length > 0 && links.every((h) => /^#\/(ch0[345]\/(start|labs|wrap|\d\.\d)|review)$/.test(h)), links.join(' '));

const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(`no sideways page scroll at ${narrow ? 390 : 1440} px`, overflow <= 1, String(overflow));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
