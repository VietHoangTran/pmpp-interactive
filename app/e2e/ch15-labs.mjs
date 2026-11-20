// Browser check for the chapter 15 labs page: status table, the race-backend table, eight
// predictions against this machine's measurements, results tables, what each device said
// about the compiled kernels, the ladder, the milestone line and the reference-solution
// toggles. Everything expected is computed here from /api/reports and /api/lab-file, so
// the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch15-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that no
// prediction is already committed whatever the reader has done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, LabKit)
const LABS = [
  { num: '15.1', dir: 'lab1_coarse' },
  { num: '15.2', dir: 'lab2_regtile' },
  { num: '15.3', dir: 'lab3_remap' },
  { num: '15.4', dir: 'lab4_padded' },
  { num: '15.5', dir: 'lab5_pipelined' },
  { num: '15.6', dir: 'lab6_linear' },
  { num: '15.7', dir: 'lab7_bughunt' },
].map((l) => ({ ...l, id: `ch15/${l.dir}` }));
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
// realBackends(): every backend in the reports except the two emulators.
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows, variant: mine.length ? 'work' : 'solution' }] : [];
  });
const allRows = (reports, id) =>
  real(reports).flatMap((b) => [...(labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : []), ...benchRows(reports[b], id, 'solution')]);
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const where = (rows, name, w) => rows.find((r) => r.name === name && Object.entries(w).every(([k, v]) => r[k] === v));

await goto(page, 'ch15/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure and standard sections
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Labs" of chapter 15', h1.includes('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions',
  'Lab 15.1: the 128 × 128 coarsened kernel', 'Lab 15.2: register tiling, the loops interchanged',
  'Lab 15.3: coalesced stores, four 4 × 4 sub-tiles and float4 stores', 'Lab 15.4: padding and vector loads',
  'Lab 15.5: double buffering and software pipelining', 'Lab 15.6: the fast linear layer (milestone M3)',
  'Lab 15.7: bug hunt, three defects in a fast linear layer'];
check('the page has its sections in order, one per lab', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 15.1', 'python3 labs/pmpp.py run 15.1 --bench', 'python3 labs/pmpp.py run 15.1 --backend emu',
  'python3 labs/pmpp.py run 15.1 --solution', 'python3 labs/pmpp.py reset 15.1 --yes', 'python3 labs/pmpp.py run 15.7 --backend race', 'python3 labs/pmpp.py run 15.7 --backend emu'])
  check(`command shown: ${c}`, text0.includes(c));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the vendor calls are unverified and can be switched off', text0.includes('never been built on a CUDA or ROCm machine (unverified)') && text0.includes('-DPMPP_CH15_VENDOR=OFF') && text0.includes('-DPMPP_CH15_VENDOR_BLAS=OFF'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim() })));
check('status table has 7 labs x 4 backends', cells.length === 28, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`,
    mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the race table
if (!reports.race) {
  check('no race run: the page says so', text0.includes('No run on the race backend on this machine yet.'));
} else {
  const race = await page.$$eval('tr[data-race-lab]', (trs) => trs.map((tr) => ({ lab: tr.dataset.raceLab, s: tr.dataset.raceSolution, w: tr.dataset.raceWork, cells: [...tr.children].map((td) => td.textContent) })));
  check('race table: one row per lab', race.length === 7, String(race.length));
  for (const [i, l] of LABS.entries()) {
    const r = race[i];
    const s = labStatus(reports.race, l.id, 'solution');
    const w = labStatus(reports.race, l.id, 'work');
    const tests = results(reports.race, l.id, 'solution').flatMap((x) => x.tests);
    const tsan = results(reports.race, l.id, 'work').reduce((n, x) => n + (x.tsan?.length ?? 0), 0);
    check(`race table, lab ${l.num}: solution ${s}, work ${w}, counts from the reports`,
      r && r.lab === l.num && r.s === s && r.w === w &&
      (s !== 'pass' || r.cells[1].includes(`emulator-checked; passes: ${tests.length} of ${tests.length} tests`)) &&
      (w !== 'fail' || r.cells[2].includes(tsan ? `${tsan} ThreadSanitizer report` : 'no race reported')),
      JSON.stringify(r));
  }
}

// ---- the reader's own work
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

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const ratio = (id, top, bottom, w) =>
  preferred(reports, id).flatMap((s) => {
    const a = where(s.rows, top, w);
    const b = where(s.rows, bottom, w);
    return a && b && a.gflops && b.gflops ? [{ backend: s.backend, value: a.gflops / b.gflops, detail: `${fmt(a.gflops)} against` }] : [];
  });
const CUBE = { m: 2048, k: 2048, n: 2048 };
const predictions = [
  { lab: '15.1', id: 'ch15/labs/p-lab1-speedup', find: 'Weigh three things', guess: 4, want: ratio(LABS[0].id, 'stage', 'tiled_5_3', CUBE) },
  { lab: '15.1 (batch 1)', id: 'ch15/labs/p-lab1-batch1', find: 'Count the blocks each one launches', guess: 0.5, want: ratio(LABS[0].id, 'stage', 'tiled_5_3', { m: 1, k: 1024, n: 4096 }) },
  { lab: '15.2', id: 'ch15/labs/p-lab2-speedup', find: 'Fig. 15.6 reads shared memory 1024 times', guess: 1.5, want: ratio(LABS[1].id, 'stage', 'previous', CUBE) },
  { lab: '15.3 (short K)', id: 'ch15/labs/p-lab3-shortk', find: 'On the short-K shape', guess: 1.2, want: ratio(LABS[2].id, 'stage', 'previous', { m: 4096, k: 128, n: 4096 }) },
  { lab: '15.3', id: 'ch15/labs/p-lab3-speedup', find: 'where 256 tile pairs share each store', guess: 1.1, want: ratio(LABS[2].id, 'stage', 'previous', CUBE) },
  { lab: '15.4', id: 'ch15/labs/p-lab4-speedup', find: 'what speed-up over lab 15.3 will you measure', guess: 1.3, want: ratio(LABS[3].id, 'stage', 'previous', CUBE) },
  { lab: '15.5', id: 'ch15/labs/p-lab5-speedup', find: 'the overlap is the compiler', guess: 1.2, want: ratio(LABS[4].id, 'stage', 'previous', CUBE) },
  { lab: '15.6', id: 'ch15/labs/p-lab6-batch1', find: 'your M3’s speed as a multiple of M2’s', guess: 0.5,
    want: preferred(reports, LABS[5].id).flatMap((s) => {
      const a = where(s.rows, 'm3', { batch: 1, in: 1024, out: 4096 });
      const b = where(s.rows, 'm2', { batch: 1, in: 1024, out: 4096 });
      return a && b ? [{ backend: s.backend, value: a.gflops / b.gflops, detail: `${fmt(a.gflops)} against` }] : [];
    }) },
];
const predictBlocks = await page.$$eval('.q', (qs) => qs.filter((e) => e.querySelector('.kind')?.textContent === 'Predict first').length);
check('the page has eight predictions, none committed at the start', predictBlocks === 8 && (await page.$$('[data-measured]')).length === 0, String(predictBlocks));
for (const p of predictions) {
  const q = await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find);
  const el = q.asElement();
  if (!el) { check(`lab ${p.lab}: prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  check(`lab ${p.lab}: nothing is revealed before a prediction is committed`, !!input && (await el.$('[data-measured]')) === null);
  if (!input) continue;
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  if (p.want.length) {
    const ok = p.want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && s.text.includes(w.detail);
    });
    check(`lab ${p.lab}: measured value from /api/reports is shown (${p.want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === p.want.length, JSON.stringify(shown));
    check(`  lab ${p.lab}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet'), JSON.stringify(shown));
  }
  check(`  lab ${p.lab}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await shot('predict-lab1', 'section.bench[aria-label="Lab 15.1 on this machine"]');

// ---- results tables: every benchmark row in the reports, and nothing invented
const STAGE = new Set(['tiled_5_3', 'previous', 'stage', 'vendor']);
for (const [i, l] of LABS.entries()) {
  if (l.num === '15.7') {
    const t = await page.$eval('section.bench[aria-label="Lab 15.7 on this machine"]', (e) => e.textContent);
    check('lab 15.7: no benchmark, and the box says so', t.includes('This lab has no benchmark') && (await page.$('table[data-results="15.7"]')) === null);
    continue;
  }
  const keep = i < 5 ? (r) => STAGE.has(r.name) : (r) => ['m2', 'm3', 'vendor'].includes(r.name);
  const rows = allRows(reports, l.id).filter(keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${l.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${l.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their GFLOP/s`,
      shown.length === rows.length && rows.every((r, j) => shown[j][shown[j].length - 1] === fmt(r.gflops)), JSON.stringify(shown.slice(0, 2)));
    if (i < 5) check(`  lab ${l.num}: the shape is shown as M × K × N`, rows.every((r, j) => shown[j].includes(`${r.m} × ${r.k} × ${r.n}`)));
    else check('  lab 15.6: batch, in and out are shown', rows.every((r, j) => shown[j].includes(String(r.batch)) && shown[j].includes(String(r.out))));
  } else {
    const empty = await page.$eval(`[data-results-empty="${l.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${l.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${l.num}"]`)) === null, empty);
  }
  // What the device said about the compiled kernel.
  const facts = await page.$eval(`[data-kernel-facts="${l.num}"]`, (e) => e.textContent);
  const frows = preferred(reports, l.id).flatMap((s) => s.rows.filter((r) => r.name === 'pipeline' || r.name === 'occupancy').map((r) => ({ s, r })));
  if (frows.length) {
    const ok = frows.every(({ s, r }) => {
      if (r.name === 'occupancy') return facts.includes(`${s.backend} (`) && (r.blocks_per_sm < 0 ? facts.includes('not available') : facts.includes(`fits ${r.blocks_per_sm} block`));
      const cap = r.max_threads_per_threadgroup ?? r.maxTotalThreadsPerThreadgroup;
      const bytes = r.static_threadgroup_bytes ?? r.staticThreadgroupMemoryLength;
      return facts.includes(`at most ${cap} threads per threadgroup`) && facts.includes(`${fmt(bytes)} bytes of static threadgroup memory`);
    });
    check(`lab ${l.num}: what the device said about the kernel is read from the ${frows[0].r.name} row`, ok, facts);
  } else check(`lab ${l.num}: no pipeline or occupancy row, and the line says so`, facts.includes('there is no such row on this machine yet'), facts);
}
await shot('lab6', 'section.bench[aria-label="Lab 15.6 on this machine"]');

// ---- the ladder at 2048 cubed
const ladderSets = (b, i) => preferred(reports, LABS[i].id).find((s) => s.backend === b);
const ladderBackends = [...new Set(LABS.slice(0, 5).flatMap((l) => preferred(reports, l.id).map((s) => s.backend)))];
if (!ladderBackends.length) {
  check('the ladder says there is no benchmark yet', (await page.$('[data-ladder="none"]')) !== null);
} else {
  for (const b of ladderBackends) {
    const got = await page.$$eval(`table[data-ladder="${b}"] tr[data-rung]`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const ok = [0, 1, 2, 3, 4].every((i) => {
      const s = ladderSets(b, i);
      const st = s && where(s.rows, 'stage', CUBE);
      const below = s && where(s.rows, i === 0 ? 'tiled_5_3' : 'previous', CUBE);
      return got[i][1] === (st ? fmt(st.gflops) : '') && got[i][2] === (st && below ? fmt(st.gflops / below.gflops) : '');
    });
    check(`ladder on ${b}: each rung's GFLOP/s and its ratio to the rung below in the same run`, got.length === 5 && ok, JSON.stringify(got));
  }
}
await shot('ladder', 'table[data-ladder]');

// ---- milestone M3
const m3 = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
const passM3 = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], LABS[5].id, 'work') === 'pass');
check(`milestone M3 status is read from the reports (${passM3.join(',') || 'none'})`, m3.v === (passM3.join(',') || 'none'), JSON.stringify(m3));
check('the milestone keeps M2\'s signature and conventions', text0.includes('launchLinear(X, W, Y, batch, in, out)') && text0.includes('batch first'));
check('the 15.6 section points to the batch-256 prediction in section 15.9', text0.includes('under “Your kernel against the library”'));

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch15/${l.dir}/solution/${f}`);
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
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`  lab ${l.num}: shared tiles coloured as shared memory, the matrices as global memory`, (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-global')).length > 0);
  if (l.num === '15.3') {
    // W is the warp width here, not a matrix: it must not be coloured as global memory.
    const wGlobal = await page.$$eval('figure.code .v-global', (s) => s.filter((x) => x.textContent === 'W').length);
    check('  lab 15.3: the width W is not coloured as a matrix', wGlobal === 0, String(wGlobal));
    await shot('solution-lab3', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch15/start', '#/ch15/15.2', '#/ch15/15.3', '#/ch15/15.4', '#/ch15/15.5', '#/ch15/15.6', '#/ch15/15.7', '#/ch15/15.8', '#/ch15/15.9']);
check(`links go to routes of this chapter (${links.join(', ')})`, links.length >= 8 && links.every((l) => known.has(l)), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch15/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="solution-status"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
