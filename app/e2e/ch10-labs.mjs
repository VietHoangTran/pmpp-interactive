// Browser check for the chapter 10 labs page: status table, race lines, the reader's own
// status, five predictions against this machine's measurements, results tables, the M4
// line and the reference-solution toggles. Everything expected is computed here from
// /api/reports and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch10-labs.mjs [screenshot-prefix] [--dark] [--narrow]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const narrow = process.argv.includes('--narrow');
const { browser, page, errors } = await open({ width: narrow ? 390 : 1440, height: 1000 });
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
  { num: '10.1', id: 'ch10/lab1_sum', dir: 'lab1_sum' },
  { num: '10.2', id: 'ch10/lab2_reducehunt', dir: 'lab2_reducehunt' },
  { num: '10.3', id: 'ch10/lab3_softmax_argmax', dir: 'lab3_softmax_argmax' },
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

await goto(page, 'ch10/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- commands, backends, headings
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 10.1', 'python3 labs/pmpp.py run 10.1 --bench', 'python3 labs/pmpp.py run 10.1 --backend emu',
  'python3 labs/pmpp.py run 10.1 --solution', 'python3 labs/pmpp.py reset 10.1 --yes', 'python3 labs/pmpp.py run 10.2 --backend race', 'python3 labs/pmpp.py run 10.3'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is described, and why lab 10.2 needs it', text0.includes('ThreadSanitizer') && text0.includes('without an atomic'));
check('the page says what a Mac cannot test about HIP warps', text0.includes('0xffffffff') && text0.includes('Check both by') && text0.includes('reading your code'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent));
for (const t of ['Lab 10.1: one sum, three kernels', 'Lab 10.2: reduction hunt', 'Lab 10.3: softmax and argmax over logits (M4)'])
  check(`heading present: ${t}`, h2.includes(t), h2.join(' | '));
check('no literal "being written" placeholder', !/being written/i.test(text0));
check('the page never presents 32 as the warp width without saying whose', !/warps? (are|is) 32/i.test(text0));
check('the provided files are named where they live', text0.includes('labs/ch10/shared/inplace_sum.cu') && text0.includes('labs/ch10/shared/rowwise_naive.cu'));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 3 labs x 4 backends', cells.length === 12, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  const ok = mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]);
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`, ok, JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the race lines, one per lab
for (const l of LABS) {
  const run = reports.race;
  const s = labStatus(run, l.id, 'solution');
  const n = results(run, l.id, 'solution').flatMap((x) => x.tests).length;
  const line = await page.$eval(`[data-race-lab="${l.num}"]`, (e) => ({ s: e.dataset.raceStatus, text: e.textContent })).catch(() => null);
  const work = results(run, l.id, 'work');
  const races = work.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
  const ok = line && line.s === s &&
    (s === 'pass' ? line.text.includes(`passed all ${n} tests with no race reported`) && line.text.includes('emulator-checked') : s === 'fail' ? line.text.includes('unverified there') : true) &&
    (!work.length || (races ? line.text.includes(`${races} race`) : true));
  check(`lab ${l.num}: race line reads the reports (solution ${s}; your work ${races} race report(s))`, !!ok, JSON.stringify(line));
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
    if (!rs.length) return row[b].includes('not started');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- milestone M4: lab 10.3 of the reader's own work
{
  const done = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], LABS[2].id, 'work') === 'pass');
  const m = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
  check(`milestone M4 line reads lab 10.3 from the reports (${done.join(',') || 'none'})`, m.v === (done.join(',') || 'none') &&
    (done.length ? m.text.includes('M4 is done there') : m.text.includes('does not pass on any backend') || m.text.includes('No lab results')), JSON.stringify(m));
  check('the milestone names M4, the toy model and M10', text0.includes('milestone M4 of the through-line project') && text0.includes('lookup') && text0.includes('M10'));
}

// ---- predictions
const predictBlock = async (find) =>
  (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), find)).asElement();
const commit = async (el, guess) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
const shownOf = (el) => el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
const row = (set, pred) => set.rows.find(pred);

// Ratio predictions (LabPrediction): the measured value and the comparison sentence.
const ratios = [
  {
    key: 'p-lab1-warp', lab: '10.1', find: 'how many times as fast as the simple kernel', guess: 2,
    want: preferred(reports, LABS[0].id).flatMap((s) => {
      const a = row(s, (r) => r.name === 'rows' && r.kernel === 'simple')?.seconds;
      const b = row(s, (r) => r.name === 'rows' && r.kernel === 'warp')?.seconds;
      return a && b ? [{ backend: s.backend, value: a / b }] : [];
    }),
  },
  {
    key: 'p-lab3-naive', lab: '10.3', find: 'how many times as fast as one thread per row', guess: 50,
    want: preferred(reports, LABS[2].id).flatMap((s) => {
      const a = row(s, (r) => r.name === 'softmax' && r.kernel === 'naive' && r.rows === 1 && r.cols === 128256)?.seconds;
      const b = row(s, (r) => r.name === 'softmax' && r.kernel === 'block' && r.rows === 1 && r.cols === 128256)?.seconds;
      return a && b ? [{ backend: s.backend, value: a / b }] : [];
    }),
  },
];
for (const p of ratios) {
  const el = await predictBlock(p.find);
  if (!el) { check(`lab ${p.lab} (${p.key}): prediction block present`, false); continue; }
  check(`lab ${p.lab} (${p.key}): nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
  const shown = await shownOf(el);
  if (p.want.length) {
    const ok = p.want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`lab ${p.lab} (${p.key}): measured value from /api/reports is shown (${p.want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === p.want.length, JSON.stringify(shown));
    check(`  lab ${p.lab} (${p.key}): the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
    if (p.key === 'p-lab3-naive') check('  lab 10.3: the batch-64 ratio is in the detail, which answers the second question', shown.every((s) => s.text.includes('at batch 64 the ratio is')), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(`pmpp.py run ${p.lab} --bench`), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}

// The coarsening factor and the determinism count (shared with sections 10.10 and 10.9).
{
  const el = await predictBlock('Which factor will be fastest on your device');
  check('lab 10.1 (p-coarse): the prediction says it is the one of section 10.10', !!el && (await el.evaluate((e) => e.textContent)).includes('The same prediction as in section 10.10'));
  await commit(el, 8);
  const want = preferred(reports, LABS[0].id).flatMap((s) => {
    const rows = s.rows.filter((r) => r.name === 'any' && isFinite(r.coarse) && isFinite(r.gb_per_s));
    if (!rows.length) return [];
    const best = rows.reduce((a, b) => (b.gb_per_s > a.gb_per_s ? b : a));
    const mine = rows.find((r) => r.coarse === 8);
    return [{ backend: s.backend, best: best.coarse, pct: mine ? Math.round((mine.gb_per_s / best.gb_per_s) * 100) : null }];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 10.1 (p-coarse): fastest factor from the reports (${want.map((w) => `${w.backend} C = ${w.best}`).join(', ')}), and the pick's share of it`,
      want.every((w) => {
        const s = shown.find((x) => x.backend === w.backend);
        return s && Number(s.measured) === w.best && s.text.includes(`C = ${w.best}`) && (w.best === 8 ? s.text.includes('That is the factor you picked.') : s.text.includes(`${w.pct}% of the fastest`));
      }) && shown.length === want.length, JSON.stringify(shown));
  else check('lab 10.1 (p-coarse): no sweep in the reports, and the reveal says what to run', shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('pmpp.py run 10.1 --bench'), JSON.stringify(shown));
}
{
  const el = await predictBlock('how many distinct results, as bit patterns');
  check('lab 10.1 (p-distinct): the prediction says it is the one of section 10.9', !!el && (await el.evaluate((e) => e.textContent)).includes('The same prediction as in section 10.9'));
  await commit(el, 1);
  const want = preferred(reports, LABS[0].id).flatMap((s) => {
    const r = row(s, (x) => x.name === 'determinism');
    return r && isFinite(r.distinct) ? [{ backend: s.backend, d: r.distinct, runs: r.runs }] : [];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 10.1 (p-distinct): the count from the reports (${want.map((w) => `${w.backend} ${w.d} of ${w.runs}`).join(', ')}) and the prediction`,
      want.every((w) => {
        const s = shown.find((x) => x.backend === w.backend);
        return s && Number(s.measured) === w.d && s.bold === String(w.d) && s.text.includes(`in ${w.runs} runs`) && s.text.includes(w.d === 1 ? 'That is what you predicted.' : 'You predicted 1.');
      }) && shown.length === want.length, JSON.stringify(shown));
  else check('lab 10.1 (p-distinct): no count in the reports, and the reveal says what to run', shown.length === 1 && shown[0].measured === 'none', JSON.stringify(shown));
  check('lab 10.1 (p-distinct): the reading points at the second pass', (await el.evaluate((e) => e.textContent)).includes('cub::DeviceReduce'));
  await shot('predict-distinct', '[data-results="10.1"]');
}

// Lab 10.2: how many of the GPU's nine cases the shipped kernel gets right, then every backend's run.
{
  const el = await predictBlock('how many of the nine will come out right');
  check('lab 10.2 (p-lab2-gpu): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 3);
  const shown = await shownOf(el);
  const runs = ['metal', 'cuda', 'hip', 'emu', 'race'].flatMap((b) => {
    const rs = results(reports[b], LABS[1].id, 'work');
    if (!reports[b] || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [{ backend: b, passed: tests.filter((t) => t.pass).length, total: tests.length, untouched: reports[b].work_state?.[LABS[1].id] === 'untouched',
      aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1 && !(b === 'race' && x.test_rc === 66)), asan: rs.some((x) => (x.log ?? '').includes('AddressSanitizer')),
      races: rs.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0) }];
  });
  if (runs.length) {
    const ok = runs.every((r) => {
      const s = shown.find((x) => x.backend === r.backend);
      if (!s || Number(s.measured) !== r.passed) return false;
      if (['metal', 'cuda', 'hip'].includes(r.backend))
        return s.text.includes(`${r.passed} of ${r.total} cases`) && (r.untouched ? s.text.includes(r.passed === 3 ? 'That is your prediction.' : 'You predicted 3.') : s.text.includes('not the experiment any more'));
      if (r.aborted && r.total === 0) return s.text.includes('stopped before its first case finished') && (!r.asan || s.text.includes('AddressSanitizer'));
      return s.text.includes(`${r.passed} of ${r.total} cases right`) && (!r.races || s.text.includes(`ThreadSanitizer reported ${r.races} race`));
    });
    check(`lab 10.2 (p-lab2-gpu): every backend's last run of work/ is read (${runs.map((r) => `${r.backend} ${r.passed}/${r.total}${r.aborted ? ' stopped' : ''}${r.races ? `, ${r.races} races` : ''}`).join('; ')})`, ok, JSON.stringify(shown));
    const gpu = runs.find((r) => ['metal', 'cuda', 'hip'].includes(r.backend));
    if (gpu && gpu.passed < gpu.total) {
      const txt = await el.evaluate((e) => e.textContent);
      check('  lab 10.2: the failing GPU cases are listed with the driver\'s launch counts', /wrong in \d+ of 20 launches/.test(txt));
    }
  } else check('lab 10.2 (p-lab2-gpu): no run of work/, and the reveal says what to run', shown.length >= 1 && shown[0].text.includes('pmpp.py run 10.2'), JSON.stringify(shown));
  check('lab 10.2: the reading says a passing case proves nothing about undefined reads', (await el.evaluate((e) => e.textContent)).includes('has not shown that the kernel is correct'));
  await shot('predict-hunt', '[data-predict], .q');
}
await pause(700); // the app saves 400 ms after the last change
{
  const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  const ids = ['ch10/labs/p-lab1-warp', 'ch10/10.10/p-coarse', 'ch10/10.9/p-distinct', 'ch10/labs/p-lab2-gpu', 'ch10/labs/p-lab3-naive'];
  check('all five predictions were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const tables = [
  { lab: LABS[0], keep: (r) => ['rows', 'any', 'determinism'].includes(r.name), cell: (r) => (r.name === 'determinism' ? `${r.distinct} of ${r.runs}` : fmt(r.gb_per_s)) },
  { lab: LABS[2], keep: (r) => ['softmax', 'argmax'].includes(r.name), cell: (r) => fmt(r.seconds * 1000) },
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
  const rows = sets(LABS[0].id).filter((r) => r.name === 'rows');
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="10.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check('lab 10.1: row kernels are named with their figures, and floats = rows × 2 × block', rows.every((r, i) => shown[i].join('|').includes('rows: ') && shown[i].includes((r.rows * 2 * r.block).toLocaleString('en-US'))) &&
      shown.some((s) => s.join('|').includes('simple (Fig. 10.5, provided)')), JSON.stringify(shown.slice(0, 2)));
  }
}
check('lab 10.2 says it has no benchmark', (await body()).includes('This lab has no benchmark'));
await shot('lab3', 'section.bench[aria-label="Lab 10.3 on this machine"]');

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch10/${l.dir}/solution/${f}`);
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
  check(`  lab ${l.num}: tier colours are applied to the real file`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0);
  if (l.num === '10.3') await shot('solution-lab3', `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links to other sections resolve to routes of this chapter
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check(`cross-links point at chapter routes (${links.join(', ')})`, links.length > 0 && links.every((h) => /^#\/(ch10\/(start|labs|wrap|10\.\d+)|review)$/.test(h)), links.join(' '));

const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(`no sideways page scroll at ${narrow ? 390 : 1440} px`, overflow <= 1, String(overflow));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
