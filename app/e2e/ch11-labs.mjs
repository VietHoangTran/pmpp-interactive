// Browser check for the chapter 11 labs page: status table, race lines, the reader's own
// status, six predictions against this machine's measurements, results tables, the M5 line
// and the reference-solution toggles. Everything expected is computed here from
// /api/reports and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch11-labs.mjs [screenshot-prefix] [--dark] [--narrow]
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
const ALL = ['kernel.cu', 'kernel.metal', 'launch.mm'];
const LABS = [
  { num: '11.1', id: 'ch11/lab1_blockscan', dir: 'lab1_blockscan', sources: ALL },
  { num: '11.2', id: 'ch11/lab2_coarsened', dir: 'lab2_coarsened', sources: ALL },
  { num: '11.3', id: 'ch11/lab3_scanhunt', dir: 'lab3_scanhunt', sources: ALL },
  { num: '11.4', id: 'ch11/lab4_global', dir: 'lab4_global', sources: ALL },
  { num: '11.5', id: 'ch11/lab5_singlepass', dir: 'lab5_singlepass', sources: ['kernel.cu'] },
  { num: '11.6', id: 'ch11/lab6_topp', dir: 'lab6_topp', sources: ALL },
];
const L = Object.fromEntries(LABS.map((l) => [l.num, l]));
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const KERNEL = { metal: 'kernel.metal', emu: 'kernel.cu', cuda: 'kernel.cu', hip: 'kernel.cu' };
const noSource = (l, b) => (l.sources.includes(KERNEL[b]) ? null : b === 'metal' ? 'no Metal files in this lab' : 'no kernel.cu in this lab');
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
const solutionText = (reports, b, l) => {
  const s = labStatus(reports[b], l.id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : noSource(l, b) ?? 'no run on this machine';
};
// Real backends are every one but emu and race (LabKit's realBackends).
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
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
const kernelRow = (set, kernel, where = {}) => set?.rows.find((r) => r.kernel === kernel && Object.entries(where).every(([k, v]) => r[k] === v));

await goto(page, 'ch11/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- commands, backends, headings
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 11.1', 'python3 labs/pmpp.py run 11.1 --bench', 'python3 labs/pmpp.py run 11.1 --backend emu',
  'python3 labs/pmpp.py run 11.1 --solution', 'python3 labs/pmpp.py reset 11.1 --yes', 'python3 labs/pmpp.py run 11.3 --backend race', 'python3 labs/pmpp.py run 11.5 --backend emu'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is described, and all six labs are built for it', text0.includes('ThreadSanitizer') && text0.includes('all six labs here are built for it'));
check('the page says what a Mac cannot test about HIP warps', text0.includes('0xffffffff') && text0.includes('Check both by reading your code'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('lab 11.5 is said to have no Metal version, with the reason', text0.includes('Lab 11.5 has kernel.cu only') && text0.includes('undefined behavior'));
check('lab 11.5\'s HIP dependency (libhipcxx) and its unverified status are stated', text0.includes('libhipcxx') && text0.includes('no ROCm machine has built it yet'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent));
for (const t of ['Lab 11.1: one block scan, three ways', 'Lab 11.2: coarsening and register tiling', 'Lab 11.3: scan race hunt',
  'Lab 11.4: a scan of any length, hierarchically', 'Lab 11.5: single-pass scan with lookback (CUDA family, optional)', 'Lab 11.6: top-p keep mask (milestone M5, part 1)'])
  check(`heading present: ${t}`, h2.includes(t), h2.join(' | '));
check('no literal "being written" placeholder', !/being written/i.test(text0));
check('the page never presents 32 as the warp width without saying whose', !/warps? (are|is) 32/i.test(text0));
check('the provided files are named where they live',
  ['labs/ch11/shared-g11/bs_two_barrier.cu', 'labs/ch11/shared-g11/bs_threadmajor.cu', 'labs/ch11/shared-g13/global_copy.cu', 'labs/ch11/shared-g12/'].every((s) => text0.includes(s)));
{
  // The page's own prose: paragraphs and lists outside the tables and readouts built from the reports.
  const prose = await page.$$eval('.prose > p:not([data-race-lab]):not([data-milestone]), .prose > ul, .prose > ol', (es) => es.map((e) => e.textContent).join(' '));
  const device = String(reports.metal?.device?.name ?? reports.cuda?.device?.name ?? 'no device');
  check(`no device name is written into the prose (${device})`, !prose.includes(device) && !/Apple M\d|H100/.test(prose));
}

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 6 labs x 4 backends', cells.length === 24, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l));
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
    const rs = results(run, l.id, 'work');
    if (!rs.length && noSource(l, b)) return row[b].includes(noSource(l, b));
    if (!run) return row[b].includes('no run on this machine');
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

// ---- milestone M5, part 1: lab 11.6 of the reader's own work
{
  const done = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], L['11.6'].id, 'work') === 'pass');
  const m = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
  check(`milestone M5 line reads lab 11.6 from the reports (${done.join(',') || 'none'})`, m.v === (done.join(',') || 'none') &&
    (done.length ? m.text.includes('part 1 of M5 is done there') : m.text.includes('does not pass on any backend') || m.text.includes('No lab results')), JSON.stringify(m));
  check('the milestone names M5, chapters 12 and 14 and module X1', text0.includes('first part of milestone M5') && text0.includes('Chapter 12') && text0.includes('chapter 14') && text0.includes('module X1'));
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
const N26 = 2 ** 26;

// Ratio predictions: the measured value and the comparison sentence.
const best = (rows, d, n) => {
  const rates = rows.filter((r) => r.kernel === d && r.n === n && r.seconds > 0).map((r) => r.n / r.seconds);
  return rates.length ? Math.max(...rates) : undefined;
};
const ratios = [
  {
    key: 'p-lab1-warp', lab: '11.1', find: 'how many times as fast as the two-barrier kernel', guess: 2,
    want: preferred(reports, L['11.1'].id).flatMap((s) => {
      const a = kernelRow(s, 'two_barrier', { block: 1024 })?.seconds;
      const b = kernelRow(s, 'warp_shuffle', { block: 1024 })?.seconds;
      return a && b ? [{ backend: s.backend, value: a / b }] : [];
    }),
  },
  {
    key: 'p-rss', lab: '11.4', find: 'The same prediction as in section 11.9', guess: 1.33,
    want: preferred(reports, L['11.4'].id).flatMap((s) => {
      const ssa = best(s.rows, 'scan_scan_add', N26);
      const rss = best(s.rows, 'reduce_scan_scan', N26);
      return ssa && rss ? [{ backend: s.backend, value: rss / ssa }] : [];
    }),
  },
  {
    key: 'p-lab6-copy', lab: '11.6', find: 'what share of your copy bandwidth will your kernel', guess: 60,
    want: preferred(reports, L['11.6'].id).flatMap((s) => {
      const r = kernelRow(s, 'yours', { rows: 64, cols: 128256 });
      return r && isFinite(r.share_of_copy) ? [{ backend: s.backend, value: r.share_of_copy * 100 }] : [];
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
    if (p.key === 'p-lab6-copy') check('  lab 11.6: the 1-row share is in the detail', shown.every((s) => s.text.includes('at 1 row of 128,256 the share is')), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(`pmpp.py run ${p.lab} --bench`), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}

// Lab 11.2: the fastest factor of the register-tiled kernel at block 256.
{
  const el = await predictBlock('Which factor will be fastest on your device');
  check('lab 11.2 (p-lab2-best): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 4);
  const want = preferred(reports, L['11.2'].id).flatMap((s) => {
    const rows = s.rows.filter((r) => r.kernel === 'register_tiled' && r.block === 256 && isFinite(r.coarse) && isFinite(r.seconds));
    if (!rows.length) return [];
    const b = rows.reduce((a, x) => (x.seconds < a.seconds ? x : a));
    const mine = rows.find((r) => r.coarse === 4);
    return [{ backend: s.backend, best: b.coarse, pct: mine ? Math.round((mine.seconds / b.seconds) * 100) : null }];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 11.2 (p-lab2-best): fastest factor from the reports (${want.map((w) => `${w.backend} ${w.best}`).join(', ')}), and the pick against it`,
      want.every((w) => {
        const s = shown.find((x) => x.backend === w.backend);
        return s && Number(s.measured) === w.best && s.text.includes(`the fastest factor was ${w.best}`) && (w.best === 4 ? s.text.includes('That is the factor you picked.') : s.text.includes(`${w.pct}% of the fastest time`));
      }) && shown.length === want.length, JSON.stringify(shown));
  else check('lab 11.2 (p-lab2-best): no sweep in the reports, and the reveal says what to run', shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('pmpp.py run 11.2 --bench'), JSON.stringify(shown));
  check('lab 11.2: the reading says to read the spread before the winner', (await el.evaluate((e) => e.textContent)).includes('Read the spread before the winner'));
}

// Lab 11.3: how many defects the GPU shows; the reveal reads every backend's run of work/.
{
  const el = await predictBlock('how many of the three defects will make a test fail');
  check('lab 11.3 (p-lab3-gpu): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 2);
  const shown = await shownOf(el);
  const runs = ['metal', 'cuda', 'hip', 'emu', 'race'].flatMap((b) => {
    const rs = results(reports[b], L['11.3'].id, 'work');
    if (!reports[b] || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [{ backend: b, passed: tests.filter((t) => t.pass).length, total: tests.length, races: rs.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0) }];
  });
  if (runs.length) {
    const ok = runs.every((r) => {
      const s = shown.find((x) => x.backend === r.backend);
      if (!s || Number(s.measured) !== r.passed) return false;
      if (['metal', 'cuda', 'hip'].includes(r.backend)) return s.text.includes(`${r.passed} of ${r.total} cases`);
      return s.text.includes(`${r.passed} of ${r.total} cases right`) && (!r.races || s.text.includes(`ThreadSanitizer reported ${r.races} race`));
    });
    check(`lab 11.3 (p-lab3-gpu): every backend's last run of work/ is read (${runs.map((r) => `${r.backend} ${r.passed}/${r.total}${r.races ? `, ${r.races} races` : ''}`).join('; ')})`, ok, JSON.stringify(shown));
  } else check('lab 11.3 (p-lab3-gpu): no run of work/, and the reveal says what to run', shown.length >= 1 && shown[0].text.includes('pmpp.py run 11.3'), JSON.stringify(shown));
  const txt = await el.evaluate((e) => e.textContent);
  check('lab 11.3: the reveal echoes the prediction and says the reports count cases, not defects', txt.includes('You predicted that 2 of the three defects') && txt.includes('count cases, not defects'));
}

// Lab 11.5 against lab 11.4 on the same CUDA or HIP device.
{
  const el = await predictBlock("How many times faster than lab 11.4's scan-scan-add");
  check('lab 11.5 (p-lab5-single): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 1.5);
  const four = preferred(reports, L['11.4'].id);
  const want = preferred(reports, L['11.5'].id).flatMap((s) => {
    const other = four.find((f) => f.backend === s.backend);
    const pairs = s.rows.filter((r) => r.kernel === 'single_pass' && r.n === N26).flatMap((r) => {
      const ssa = kernelRow(other, 'scan_scan_add', { block: r.block, n: N26 })?.seconds;
      return ssa ? [{ block: r.block, v: ssa / r.seconds }] : [];
    });
    const main = pairs.find((p) => p.block === 256) ?? pairs[0];
    return main ? [{ backend: s.backend, value: main.v }] : [];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 11.5 (p-lab5-single): ratio from both labs' rows (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')})`,
      want.every((w) => shown.some((s) => s.backend === w.backend && s.measured === fmt(w.value) && s.text.includes(compare(1.5, w.value)))), JSON.stringify(shown));
  else check('lab 11.5 (p-lab5-single): no CUDA or HIP rows here, and the reveal says where to run both labs',
    shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('CUDA and HIP devices') && shown[0].text.includes('pmpp.py run 11.4 --bench'), JSON.stringify(shown));
  check('lab 11.5: the reading names the chain of p. 279', (await el.evaluate((e) => e.textContent)).includes('p. 279'));
}
await pause(700); // the app saves 400 ms after the last change
{
  const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  const ids = ['ch11/labs/p-lab1-warp', 'ch11/labs/p-lab2-best', 'ch11/labs/p-lab3-gpu', 'ch11/11.9/p-rss', 'ch11/labs/p-lab5-single', 'ch11/labs/p-lab6-copy'];
  check('all six predictions were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
}
check('the predictions asked in sections 11.3, 11.6 and 11.8 are pointed to there', ['11.3', '11.6', '11.8'].every((s) => text0.includes(`ection ${s}`) || text0.includes(` ${s} asks`)) &&
  (await page.$$eval('.prose a', (as) => as.map((a) => a.getAttribute('href')))).filter((h) => ['#/ch11/11.3', '#/ch11/11.6', '#/ch11/11.8'].includes(h)).length >= 3);

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const ms = (r) => fmt(r.seconds * 1000);
const tables = [
  { lab: L['11.1'], keep: (r) => r.name === 'scan', cell: ms },
  { lab: L['11.2'], keep: (r) => r.name === 'scan', cell: ms },
  { lab: L['11.4'], keep: (r) => r.name === 'global_scan', cell: ms },
  { lab: L['11.5'], keep: (r) => r.name === 'global_scan', cell: ms },
  { lab: L['11.6'], keep: (r) => r.name === 'top_p_mask', cell: ms },
];
for (const t of tables) {
  const rows = sets(t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their times`,
      shown.length === rows.length && rows.every((r, i) => shown[i].includes(t.cell(r))), JSON.stringify(shown.slice(0, 3)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
{
  const rows = sets(L['11.1'].id).filter((r) => r.name === 'scan');
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="11.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check('lab 11.1: kernels are named for the reader (provided, (a), (b))', shown.some((s) => s.includes('two barriers (Fig. 11.3, provided)')) && shown.some((s) => s.includes('warp shuffle, (b)')));
  }
  const rows4 = sets(L['11.4'].id).filter((r) => r.name === 'global_scan' && r.kernel !== 'copy');
  if (rows4.length) {
    const shown = await page.$$eval('table[data-results="11.4"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check("lab 11.4: the book's bytes per element are printed beside each design", rows4.every((r) => shown.some((s) => s.includes(fmt(r.book_bytes_per_element, 4)))), JSON.stringify(shown.slice(0, 2)));
  }
}
check('lab 11.3 says it has no benchmark', (await body()).includes('This lab has no benchmark'));
await shot('lab6', 'section.bench[aria-label="Lab 11.6 on this machine"]');

// ---- reference solutions: hidden by default, then the real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
const TAB = { 'kernel.cu': 'CUDA', 'kernel.metal': 'Metal kernel', 'launch.mm': 'Metal host' };
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of l.sources)
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch11/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the solution files exist in the repository (${l.sources.join(', ')})`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are ${l.sources.map((f) => TAB[f]).join(', ')}`, tabs.join('|') === l.sources.map((f) => TAB[f]).join('|'), tabs.join('|'));
  const st = (b) => labStatus(reports[b], l.id, 'solution');
  const got = [st('cuda') === 'pass' ? 'verified on CUDA' : '', st('hip') === 'pass' ? 'verified on HIP' : '', st('emu') === 'pass' ? 'emulator-checked' : ''].filter(Boolean);
  const lacking = ['cuda', 'hip'].filter((b) => st(b) !== 'pass').map((b) => NAME[b]);
  const cuStatus = got.length ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified (no run on this machine)';
  const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : st('metal') === 'fail' ? 'unverified (failed on this machine)' : 'unverified (no run on this machine)';
  for (let i = 0; i < l.sources.length; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const f = l.sources[i];
    const status = f === 'kernel.cu' ? cuStatus : metalStatus;
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
    await pause(60);
    const shown = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`  lab ${l.num}: tier colours are applied to the real file`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0);
  if (l.num === '11.5') check('  lab 11.5: the caption says it has no Metal files', (await page.$eval('figure.code figcaption', (e) => e.textContent)).includes('no kernel.metal or launch.mm'));
  if (l.num === '11.6') await shot('solution-lab6', `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links, margin notes, width
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check(`cross-links point at chapter routes (${links.join(', ')})`, links.length > 0 && links.every((h) => /^#\/(ch11\/(start|labs|wrap|11\.\d+)|review)$/.test(h)), links.join(' '));
check('no two margin notes are stacked after one paragraph', (await page.evaluate(() => document.querySelectorAll('.prose > .note + .note').length)) === 0);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(`no sideways page scroll at ${narrow ? 390 : 1440} px`, overflow <= 1, String(overflow));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
