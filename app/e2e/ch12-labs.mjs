// Browser check for the chapter 12 labs page: status table, race lines, the reader's own
// status, eight predictions against this machine's measurements, results tables, the M5
// line and the reference-solution toggles. Everything expected is computed here from
// /api/reports and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch12-labs.mjs [screenshot-prefix] [--dark] [--narrow]
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

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
  { num: '12.1', id: 'ch12/lab1_unstable', dir: 'lab1_unstable', files: ['kernel.cu', 'kernel.metal', 'launch.mm'] },
  { num: '12.2', id: 'ch12/lab2_filterhunt', dir: 'lab2_filterhunt', files: ['kernel.cu', 'kernel.metal', 'launch.mm'] },
  { num: '12.3', id: 'ch12/lab3_stable', dir: 'lab3_stable', files: ['kernel.cu', 'kernel.metal', 'launch.mm'] },
  { num: '12.4', id: 'ch12/lab4_candidates', dir: 'lab4_candidates', files: ['kernel.cu', 'kernel.metal', 'launch.mm'] },
  { num: '12.5', id: 'ch12/lab5_inplace', dir: 'lab5_inplace', files: ['kernel.cu'] },
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
const noSource = (l, b) => (b === 'metal' && !l.files.includes('kernel.metal') ? 'no Metal files in this lab' : null);
const solutionText = (reports, b, l) => {
  const s = labStatus(reports[b], l.id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : noSource(l, b) ?? 'no run on this machine';
};
// Real backends are every one but the two emulators (LabKit's realBackends).
const REAL = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const preferred = (reports, id) =>
  REAL(reports).flatMap((b) => {
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

await goto(page, 'ch12/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- commands, backends, headings
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 12.1', 'python3 labs/pmpp.py run 12.1 --bench', 'python3 labs/pmpp.py run 12.1 --backend emu',
  'python3 labs/pmpp.py run 12.1 --solution', 'python3 labs/pmpp.py reset 12.1 --yes', 'python3 labs/pmpp.py run 12.2 --backend race',
  'python3 labs/pmpp.py run 12.3 --backend emu', 'python3 labs/pmpp.py run 12.4', 'python3 labs/pmpp.py run 12.5 --backend emu'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is described, and why these labs need it', text0.includes('ThreadSanitizer') && text0.includes('without an atomic') && text0.includes("lab 12.5's look-back"));
check('the page says what a Mac cannot test about 64-lane wavefronts', text0.includes('Neither emulator runs a warp wider than 32 lanes') && text0.includes('only keys 32 to 63 of every 64'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('lab 12.5 is named as the exception to the three-file layout, with the reason', text0.includes('Lab 12.5 is the exception') && text0.includes('Apple documents no guarantee'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent));
for (const t of ['Lab 12.1: the unstable filter three ways', 'Lab 12.2: filter hunt', 'Lab 12.3: a stable filter by scan', 'Lab 12.4: candidate compaction (M5)', 'Lab 12.5: in-place stable filter in a single pass (optional)'])
  check(`heading present: ${t}`, h2.includes(t), h2.join(' | '));
check('no literal "being written" placeholder', !/being written/i.test(text0));
check('the page never presents 32 as the warp width without saying whose', !/warps? (are|is) 32/i.test(text0));
check('the provided files are named where they live',
  ['labs/ch12/lab1_unstable/provided/copy_keys.cu', 'labs/ch12/shared-g12/scan_u32.cu', 'labs/ch12/lab4_candidates/provided/rowwise_serial.cu'].every((p) => text0.includes(p)));
for (const p of ['ch12/lab1_unstable/provided/copy_keys.cu', 'ch12/shared-g12/scan_u32.cu', 'ch12/lab4_candidates/provided/rowwise_serial.cu']) {
  const ok = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => r.ok), p);
  check(`  the provided file exists: labs/${p}`, ok);
}
check('the libhipcxx sentence matches the build (linked when CMake finds it, <cuda/atomic> unverified)', text0.includes('libhipcxx::libhipcxx when CMake finds it') && text0.includes('no ROCm machine has built this course') && !text0.includes('then compiles is unverified') && !text0.includes('builds on any ROCm') && !text0.includes('does not link libhipcxx'));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 5 labs x 4 backends', cells.length === 20, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l));
  const ok = mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]);
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`, ok, JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
check('lab 12.5 has no Metal files, and the table says so', cells.find((c) => c.lab === '12.5' && c.backend === 'metal')?.text === 'no Metal files in this lab');
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

// ---- milestone M5: lab 12.4 of the reader's own work
{
  const done = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], LABS[3].id, 'work') === 'pass');
  const m = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
  check(`milestone M5 line reads lab 12.4 from the reports (${done.join(',') || 'none'})`, m.v === (done.join(',') || 'none') &&
    (done.length ? m.text.includes('the compaction piece of M5 is in place there') : m.text.includes('does not pass on any backend') || m.text.includes('No lab results')), JSON.stringify(m));
  check('the milestone names M5, lab 9.3, lab 10.3 and module X1', ['Milestone M5 of the through-line project', 'Lab 9.3', "lab 10.3's row max", 'Module X1'].every((s) => text0.includes(s)));
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
const sec = (r) => (r && isFinite(r.seconds) ? r.seconds : null);

// Ratio predictions (LabPrediction): the measured value and the comparison sentence.
const lab3B = preferred(reports, LABS[2].id);
const ratios = [
  {
    key: 'p-lab1-agg', lab: '12.1', find: 'warp-aggregated kernel\'s time will be what', guess: 2,
    want: preferred(reports, LABS[0].id).flatMap((s) => {
      const a = sec(row(s, (r) => r.name === 'atomic' && r.fraction === 0.5));
      const w = sec(row(s, (r) => r.name === 'warp' && r.fraction === 0.5));
      return a && w ? [{ backend: s.backend, value: a / w }] : [];
    }),
  },
  {
    key: 'p-lab3-staged', lab: '12.3', find: "part A's time divided by part B's time", guess: 3,
    want: lab3B.flatMap((s) => {
      const a = sec(row(s, (r) => r.name === 'flags' && r.fraction === 0.5));
      const b = sec(row(s, (r) => r.name === 'staged' && r.fraction === 0.5 && r.coarse === 4));
      return a && b ? [{ backend: s.backend, value: a / b }] : [];
    }),
  },
  {
    key: 'p-lab4-batch1', lab: '12.4', find: 'provided thread-per-row time divided by your block-per-row time', guess: 20,
    want: preferred(reports, LABS[3].id).flatMap((s) => {
      const f = (k) => sec(row(s, (r) => r.name === 'compaction' && r.kernel === k && r.rows === 1 && r.cols === 128256 && r.fraction === 0.01));
      return f('serial') && f('block') ? [{ backend: s.backend, value: f('serial') / f('block') }] : [];
    }),
  },
  {
    key: 'p-lab4-fraction', lab: '12.4', find: 'block-per-row time at 50 percent kept divided by its time at 1', guess: 1.5,
    want: preferred(reports, LABS[3].id).flatMap((s) => {
      const f = (fr) => sec(row(s, (r) => r.name === 'compaction' && r.kernel === 'block' && r.rows === 64 && r.cols === 128256 && r.fraction === fr));
      return f(0.5) && f(0.01) ? [{ backend: s.backend, value: f(0.5) / f(0.01) }] : [];
    }),
  },
  {
    key: 'p-lab5-singlepass', lab: '12.5', find: "lab 12.3 part B's time", guess: 1.5,
    want: preferred(reports, LABS[4].id).flatMap((s) => {
      const b = lab3B.find((x) => x.backend === s.backend);
      const tB = b && sec(row(b, (r) => r.name === 'staged' && r.fraction === 0.5 && r.coarse === 4));
      const t1 = sec(row(s, (r) => r.name === 'singlepass' && r.fraction === 0.5 && r.coarse === 4 && r.inplace === false));
      return tB && t1 ? [{ backend: s.backend, value: tB / t1 }] : [];
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
    check(`  lab ${p.lab} (${p.key}): the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution|lab 12\.5 from)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(`pmpp.py run ${p.lab} --bench`), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}

// Lab 12.1: the fraction at which the plain atomic is furthest behind the privatised kernel.
{
  const el = await predictBlock('furthest behind the privatised one');
  check('lab 12.1 (p-lab1-worst): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  const guess = 0.5;
  await commit(el, guess);
  const want = preferred(reports, LABS[0].id).flatMap((s) => {
    const rs = s.rows.filter((r) => r.name === 'atomic' && isFinite(r.fraction)).flatMap((r) => {
      const p = sec(row(s, (x) => x.name === 'private' && x.fraction === r.fraction));
      return sec(r) && p ? [{ f: r.fraction, ratio: r.seconds / p }] : [];
    });
    if (!rs.length) return [];
    const worst = rs.reduce((a, b) => (b.ratio > a.ratio ? b : a));
    const mine = rs.reduce((a, b) => (Math.abs(b.f - guess) < Math.abs(a.f - guess) ? b : a));
    return [{ backend: s.backend, worst, mine }];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 12.1 (p-lab1-worst): the fraction from the reports (${want.map((w) => `${w.backend} ${w.worst.f}, ratio ${fmt(w.worst.ratio)}`).join(', ')}) and the ratio at the pick`,
      want.every((w) => {
        const s = shown.find((x) => x.backend === w.backend);
        return s && Number(s.measured) === w.worst.f && s.bold === String(w.worst.f) && s.text.includes(`${fmt(w.worst.ratio)} times as long`) &&
          (w.worst.f === guess ? s.text.includes('That is the fraction you picked.') : s.text.includes(`the ratio was ${fmt(w.mine.ratio)}`));
      }) && shown.length === want.length, JSON.stringify(shown));
  else check('lab 12.1 (p-lab1-worst): no sweep in the reports, and the reveal says what to run', shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('pmpp.py run 12.1 --bench'), JSON.stringify(shown));
  check('lab 12.1 (p-lab1-worst): the reading cites the book\'s model', (await el.evaluate((e) => e.textContent)).includes('§12.2, p. 291'));
}

// Lab 12.3: the fastest coarsening factor at half kept.
{
  const el = await predictBlock('which coarsening factor of part B');
  check('lab 12.3 (p-lab3-coarse): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  const guess = 4;
  await commit(el, guess);
  const want = lab3B.flatMap((s) => {
    const rs = s.rows.filter((r) => r.name === 'staged' && r.fraction === 0.5 && isFinite(r.coarse) && isFinite(r.seconds));
    if (!rs.length) return [];
    const best = rs.reduce((a, b) => (b.seconds < a.seconds ? b : a));
    const mine = rs.find((r) => r.coarse === guess);
    return [{ backend: s.backend, best, mine }];
  });
  const shown = await shownOf(el);
  if (want.length)
    check(`lab 12.3 (p-lab3-coarse): fastest factor from the reports (${want.map((w) => `${w.backend} coarse ${w.best.coarse}`).join(', ')}) and the pick's time against it`,
      want.every((w) => {
        const s = shown.find((x) => x.backend === w.backend);
        return s && Number(s.measured) === w.best.coarse && s.text.includes(`coarse ${w.best.coarse}`) &&
          (w.best.coarse === guess ? s.text.includes('That is the factor you picked.') : s.text.includes(`${fmt(w.mine.seconds / w.best.seconds)} times the fastest`));
      }) && shown.length === want.length, JSON.stringify(shown));
  else check('lab 12.3 (p-lab3-coarse): no sweep in the reports, and the reveal says what to run', shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('pmpp.py run 12.3 --bench'), JSON.stringify(shown));
}

// Lab 12.2: how many defects emu catches (answer 2), then every backend's run of the shipped work.
{
  const el = await predictBlock('how many will');
  check('lab 12.2 (p-lab2-emu): prediction block present, nothing revealed before committing', !!el && (await el.$('[data-answer]')) === null);
  await commit(el, 3);
  const txt = await el.evaluate((e) => e.textContent);
  check('lab 12.2 (p-lab2-emu): the answer is 2, and the guess of 3 is echoed', txt.includes('emu caught 2') && txt.includes('You predicted 3.') && (await el.$('[data-answer="2"]')) !== null, txt.slice(0, 300));
  const shown = await shownOf(el);
  const runs = ['emu', 'race', 'metal', 'cuda', 'hip'].flatMap((b) => {
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
      if (r.untouched && !s.text.includes('still as shipped, with all three defects')) return false;
      if (r.aborted && r.total === 0) return s.text.includes('stopped before its first case finished') && (!r.asan || s.text.includes('AddressSanitizer'));
      return s.text.includes(`${r.passed} of ${r.total} cases right`) && (!r.aborted || s.text.includes('then the run stopped')) && (!r.races || s.text.includes(`ThreadSanitizer reported ${r.races} race`));
    });
    check(`lab 12.2 (p-lab2-emu): every backend's last run of work/ is read (${runs.map((r) => `${r.backend} ${r.passed}/${r.total}${r.aborted ? ' stopped' : ''}${r.asan ? ' asan' : ''}${r.races ? `, ${r.races} races` : ''}`).join('; ')})`, ok, JSON.stringify(shown));
  } else check('lab 12.2 (p-lab2-emu): no run of work/, and the reveal says what to run', shown.length >= 1 && shown[0].text.includes('pmpp.py run 12.2'), JSON.stringify(shown));
  check('lab 12.2: the reading says a passing GPU case proves nothing', txt.includes('not that the kernel is correct'));
  await shot('predict-hunt', '[data-answer]');
}
await pause(700); // the app saves 400 ms after the last change
{
  const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  const ids = ['p-lab1-worst', 'p-lab1-agg', 'p-lab2-emu', 'p-lab3-staged', 'p-lab3-coarse', 'p-lab4-batch1', 'p-lab4-fraction', 'p-lab5-singlepass'].map((s) => 'ch12/labs/' + s);
  check('all eight predictions were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => REAL(reports).flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const tables = [
  { lab: LABS[0], keep: () => true, cell: (r) => fmt(r.seconds * 1000) },
  { lab: LABS[2], keep: () => true, cell: (r) => fmt(r.seconds * 1000) },
  { lab: LABS[3], keep: (r) => r.name === 'compaction', cell: (r) => fmt(r.seconds * 1000) },
  { lab: LABS[4], keep: (r) => r.name === 'singlepass', cell: (r) => fmt(r.seconds * 1000) },
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
  const rows = sets(LABS[0].id);
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="12.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check('lab 12.1: filters are named with their figures', shown.some((s) => s.includes('plain atomic (Fig. 12.2)')) && shown.some((s) => s.includes('privatised (Fig. 12.6)')) && shown.some((s) => s.includes('copy (provided)')), JSON.stringify(shown.slice(0, 2)));
  }
  const rows3 = sets(LABS[2].id).filter((r) => r.name === 'staged');
  if (rows3.length) {
    const shown = await page.$$eval('table[data-results="12.3"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check('lab 12.3: the model\'s bytes per key and the copy-speed equivalent are both shown', shown.some((s) => s.includes('part A, three steps')) && shown.some((s) => s.includes('part B, coarse 4')) &&
      rows3.every((r) => shown.some((s) => s.includes(fmt(r.model_bytes_per_key)) && s.includes(fmt(r.copy_equivalent_bytes_per_key)))), JSON.stringify(shown.slice(0, 2)));
  }
}
check('lab 12.2 says it has no benchmark', (await body()).includes('This lab has no benchmark'));
await shot('lab4', 'section.bench[aria-label="Lab 12.4 on this machine"]');

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
  for (const f of l.files)
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch12/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the solution files exist in the repository (${l.files.join(', ')})`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are ${l.files.map((f) => TAB[f]).join(', ')}`, tabs.join('|') === l.files.map((f) => TAB[f]).join('|'), tabs.join('|'));
  // LabKit's solutionTabStatus, from the shared helper (it adds this machine's CUDA front-end check).
  const cuStatus = tabStatus(reports, l.id, 'kernel.cu', frontend);
  const metalStatus = tabStatus(reports, l.id, 'kernel.metal', frontend);
  for (let i = 0; i < l.files.length; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const f = l.files[i];
    const status = f === 'kernel.cu' ? cuStatus : metalStatus;
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
    await pause(60);
    const shown = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  if (l.num === '12.5') check('  lab 12.5: the caption says there is no Metal source', (await page.$eval('figure.code', (e) => e.textContent)).includes('no kernel.metal or launch.mm'));
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`  lab ${l.num}: tier colours are applied to the real file`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0);
  if (l.num === '12.4') await shot('solution-lab4', `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links to other sections resolve to routes that exist
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check(`cross-links point at chapter routes (${links.join(', ')})`, links.length > 0 && links.every((h) => /^#\/(ch12\/(start|labs|wrap|12\.\d)|ch10\/10\.10|ch11\/11\.9|x1\/X1\.[46]|review)$/.test(h)), links.join(' '));
check('no two margin notes in a row', (await page.evaluate(() => document.querySelectorAll('.prose > .note + .note').length)) === 0);

const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(`no sideways page scroll at ${narrow ? 390 : 1440} px`, overflow <= 1, String(overflow));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
