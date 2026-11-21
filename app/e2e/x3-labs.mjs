// Browser check for module X3's labs page: commands, the status table, lab X3.1's race line,
// the five predictions (four against this machine's measurements, one with its own reveal), the
// two results tables, lab X3.2's launch table (from sim/llamastep.ts), per-piece table and model
// line, and the two reference-solution toggles. Everything expected is computed here from
// /api/reports, /api/lab-file and the app's own model, so the check holds on any machine.
// Usage: APP_URL=http://localhost:<port> node e2e/x3-labs.mjs [screenshot-prefix] [--dark]
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
  { num: 'X3.1', id: 'x3/lab1_llama_pieces', dir: 'lab1_llama_pieces' },
  { num: 'X3.2', id: 'x3/lab2_tinyllama', dir: 'lab2_tinyllama' },
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
const allSets = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const ref = benchRows(reports[b], id, 'solution');
    return [...(mine.length ? [{ backend: b, variant: 'work', rows: mine }] : []), ...(ref.length ? [{ backend: b, variant: 'solution', rows: ref }] : [])];
  });
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const row = (set, name, where = {}) => set.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));
const piece = (set, batch, position, group) => set.rows.find((r) => r.name === 'piece' && r.batch === batch && r.position === position && r.group === group);

await goto(page, 'x3/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 30000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);
const M = await page.evaluate(async () => {
  const s = await import('/src/sim/llamastep.ts');
  const c = await import('/src/sim/ckptformat.ts');
  const g = c.LAB_GPU;
  const cfg = { d: g.dim, hidden: g.hidden_dim, layers: g.n_layers, heads: g.n_heads, kvHeads: g.n_kv_heads, vocab: g.vocab_size, head: g.dim / g.n_heads, seqLen: g.seq_len };
  const ls = s.stepLaunches(cfg, 0);
  return { layers: cfg.layers, count: ls.length, formula: s.launchCount(cfg.layers), jobs: ls.filter((l) => l.layer === null || l.layer === 0).map((l) => [l.n, l.job, l.fn, l.lab, l.exists]) };
});

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Labs"', h1.trim().endsWith('Labs'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run X3.1', 'python3 labs/pmpp.py run X3.1 --bench', 'python3 labs/pmpp.py run X3.1 --backend emu',
  'python3 labs/pmpp.py run X3.1 --solution', 'python3 labs/pmpp.py reset X3.1 --yes', 'python3 labs/pmpp.py run X3.1 --backend race', 'python3 labs/pmpp.py run X3.2 --backend emu'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is explained for lab X3.1, with "evidence, not proof", and why lab X3.2 has none',
  text0.includes('ThreadSanitizer') && text0.includes('evidence, not proof') && text0.includes('Lab X3.2 is not: your code there'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the module is labelled as ours', await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes('Module X3 is not in the book'))));
check('the provided pieces are named, and that M10 runs reference solutions', text0.includes('provided/pieces.cmake') && text0.includes('So M10 runs the reference pieces, not your own'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantLabs = ["Lab X3.1: the Llama pieces the course has not built, and a port review", "Lab X3.2: milestone M10, a Llama-style model end to end on the course's kernels"];
check('the two labs appear in order', wantLabs.every((t, i, a) => h2.indexOf(t) >= 0 && (i === 0 || h2.indexOf(t) > h2.indexOf(a[i - 1]))), JSON.stringify(h2));
check('lab X3.2 states the milestone with the project conventions', text0.includes('completes milestone M10') && text0.includes('row-major float32 with the batch first'));
check('M10 says how it runs with M9: the adapter, what it binds, and that your interface is unchanged',
  text0.includes('M10 also runs with M9') && text0.includes('launchQLinear') && text0.includes('provided/m9_linear') &&
  text0.includes('without changing your interface') && !text0.includes('without M9'));
check('the M10 test list names the M9 check', text0.includes('M9 in M10.') && text0.includes('same quantised weights'));
check('the real-checkpoint path is labelled unverified', text0.includes('the README describes the work, unverified'));
check('the no-download rule is stated', text0.includes('Nothing is downloaded by CMake, the script, the driver or the tests'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (outside the report-driven parts)',
  !(await page.$$eval('.prose > p, .prose > ul li', (ps) => ps.filter((p) => !p.dataset?.measured && !p.dataset?.race && !p.dataset?.model)
    .some((p) => /\d+(\.\d+)? (GB\/s|GiB|MB\/s)\b|\d+\.\d+ (ms|µs)\b|\d+ tokens\/s/.test(p.textContent)))));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 2 labs x 4 backends', cells.length === 8, String(cells.length));
for (const l of LABS)
  for (const b of BACKENDS) {
    const c = cells.find((x) => x.lab === l.num && x.backend === b);
    const want = solutionText(reports, b, l.id);
    check(`status ${l.num} ${b}: "${want}"`, c && c.status === want && c.text === want &&
      (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
  }
await shot('status', 'table[data-table="solution-status"]');

// ---- race line, lab X3.1
{
  const l = LABS[0];
  const run = reports.race;
  const sol = labStatus(run, l.id, 'solution');
  const race = await page.$eval(`[data-race="${l.num}"]`, (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent })).catch(() => null);
  check(`race line of lab ${l.num} reads the reports: solution ${run ? sol : 'no run'}`, race && race.s === sol && race.w === labStatus(run, l.id, 'work') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('is emulator-checked: all') : sol === 'none' ? race.text.includes('has no run on the race backend on this machine') : race.text.includes('is unverified: failed on this machine')), JSON.stringify(race));
  check('lab X3.2 has no race line', (await page.$('[data-race="X3.2"]')) === null);
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
const [L1, L2] = LABS;
const rms = (rows) => preferred(reports, L1.id).flatMap((s) => {
  const r = row(s, 'rmsnorm', { rows, d: 4096 });
  return r && r.lab51_gb_per_s ? [{ backend: s.backend, value: (r.gb_per_s / r.lab51_gb_per_s) * 100, detail: [`rows ${rows}, d 4096`, `lab 5.1's ${fmt(r.lab51_gb_per_s)} GB/s`] }] : [];
});
const expected = {
  rms1: rms(1),
  rms256: rms(256),
  ceiling: preferred(reports, L2.id).flatMap((s) => {
    const r = row(s, 'ceiling');
    return r && typeof r.fraction === 'number' ? [{ backend: s.backend, value: r.fraction * 100, detail: [`${fmt(r.tokens_per_s)} tokens/s at batch 1, position 16`, `ceiling of ${fmt(r.ceiling_tokens_per_s)}`] }] : [];
  }),
  attn: preferred(reports, L2.id).flatMap((s) => {
    const late = piece(s, 1, 512, 'attention');
    const early = piece(s, 1, 16, 'attention');
    return late ? [{ backend: s.backend, value: late.share * 100, detail: ['batch 1', early ? `at position 16 the share was ${fmt(early.share * 100)} %` : 'not recorded'] }] : [];
  }),
};
const predictions = [
  { key: 'rms1', id: 'x3/labs/p-lab1-rmsnorm-bw', find: 'At rows = 1, d = 4,096, what percentage', guess: 20, lab: 'X3.1' },
  { key: 'rms256', id: 'x3/labs/p-lab1-rmsnorm-bw-256', find: 'And at rows = 256, d = 4,096?', guess: 80, lab: 'X3.1' },
  { key: 'ceiling', id: 'x3/labs/p-lab2-bwfraction', find: 'percentage of the weight-bandwidth ceiling', guess: 50, lab: 'X3.2' },
  { key: 'attn', id: 'x3/labs/p-lab2-attn-share', find: 'what percentage of the summed per-piece time', guess: 10, lab: 'X3.2' },
];
const predictBlock = async (find) =>
  (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), find)).asElement();
const commit = async (el, v) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(v));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
for (const p of predictions) {
  const el = await predictBlock(p.find);
  if (!el) { check(`${p.key}: prediction block present`, false); continue; }
  check(`${p.key}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
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
// Lab X3.1's port prediction has its own reveal: what emu reported when the lab was built.
{
  const el = await predictBlock('how many will emu report in the code as shipped');
  check('port: prediction block present, nothing revealed before commit', !!el && (await el.$('[data-reveal="p-lab1-port-emu"]')) === null);
  if (el) {
    await commit(el, 3);
    const t = await el.$eval('[data-reveal="p-lab1-port-emu"]', (e) => e.textContent).catch(() => '');
    check('port: the reveal gives 2 of 3, says why a missing barrier passes emu and that race names it, and names no defect',
      t.includes('reported 2 of the three') && t.includes('You predicted 3.') && t.includes('is a missing barrier') &&
      t.includes('The race backend runs the threads at once and names both lines') && !/float4|0xCD|Defect by defect/.test(t), t);
    const clicked = await el.evaluate((e) => {
      const b = [...e.querySelectorAll('[data-reveal="p-lab1-port-emu"] button')].find((x) => x.textContent.includes('I have fixed all three'));
      if (!b) return false;
      b.click();
      return true;
    });
    await pause(150);
    const d = await el.$eval('[data-reveal-detail="p-lab1-port-emu"]', (e) => e.textContent).catch(() => '');
    check('port: after "I have fixed all three", the detail names (a), (b) and (c) and which backend saw each',
      clicked && d.includes('(a), the second stage reading slots no warp wrote') && d.includes('(c), the float4 path') &&
      d.includes('(b), no barrier before the next row') && d.includes('only (c) failed'), d);
  }
}
// The reveal's claims against this machine's emu and race runs of the untouched starter, when they exist.
{
  const st = (b) => results(reports[b], L1.id, 'work').flatMap((x) => x.tests);
  const untouched = (b) => reports[b]?.work_state?.[L1.id] === 'untouched';
  const emu = st('emu');
  if (emu.length && untouched('emu')) {
    const portFails = emu.filter((t) => !t.pass && t.case.startsWith('port:'));
    check('starter on emu: the port fails at d 64 (unwritten slots) and no several-rows case fails alone at d 4096',
      portFails.some((t) => t.case.includes('d 64')) && !emu.some((t) => !t.pass && t.case.includes('port: rows 9, d 4096')), JSON.stringify(portFails.map((t) => t.case)));
  } else console.log('note  no emu run of the untouched starter of lab X3.1; reveal not cross-checked');
  const race = results(reports.race, L1.id, 'work');
  if (race.length && untouched('race')) check('starter on race: ThreadSanitizer reported at least one race', race.some((x) => Array.isArray(x.tsan) && x.tsan.length > 0));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all five predictions were sent to /api/progress under their ids, and none reached the real file',
  [...predictions.map((p) => [p.id, p.guess]), ['x3/labs/p-lab1-port-emu', 3]].every(([id, g]) => ans[id]?.value === g), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results tables: every timing row in the reports, and nothing invented
{
  const rows = allSets(reports, L1.id).flatMap((s) => s.rows);
  const KERNEL = { rmsnorm: 'launchRmsNorm', rope: 'launchRope, mode 0', swiglu: 'launchSwiGLU' };
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="X3.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab X3.1: results table has the ${rows.length} rows from the reports, each with launch, rows, d, time in µs and counted GB/s`,
      shown.length === rows.length && rows.every((r, i) => shown[i][2] === KERNEL[r.name] && shown[i][3] === String(r.rows) && shown[i][4] === String(r.d) &&
        shown[i][6] === fmt(r.seconds * 1e6) && shown[i][7] === fmt(r.gb_per_s) && shown[i][8] === (r.lab51_gb_per_s ? fmt((r.gb_per_s / r.lab51_gb_per_s) * 100) : '')), JSON.stringify(shown.slice(0, 2)));
    const heads = await page.$$eval('table[data-results="X3.1"] th', (t) => t.map((x) => x.textContent));
    check('lab X3.1: no column head repeats', new Set(heads).size === heads.length, heads.join('|'));
  } else check('lab X3.1: no benchmark rows, and the empty state says what to run', (await page.$eval('[data-results-empty="X3.1"]', (e) => e.textContent).catch(() => '')).includes('./verify.sh'));
}
{
  const rows = allSets(reports, L2.id).flatMap((s) => s.rows.filter((r) => r.name === 'step'));
  const GEMM = { gemv: 'lab 22.1 gather', m2: 'lab 6.4 (M2)', m3: 'lab 15.6 (M3)' };
  if (rows.length) {
    const shown = await page.$$eval('table[data-results="X3.2"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab X3.2: results table has the ${rows.length} step rows from the reports, with kernel, batch, position, ms and tokens/s`,
      shown.length === rows.length && rows.every((r, i) => shown[i][2] === (GEMM[r.gemm] ?? r.gemm) && shown[i][4] === String(r.batch) && shown[i][5] === String(r.position) &&
        shown[i][6] === fmt(r.seconds * 1000) && shown[i][7] === fmt(r.tokens_per_s)), JSON.stringify(shown.slice(0, 2)));
  } else check('lab X3.2: no benchmark rows, and the empty state says what to run', (await page.$eval('[data-results-empty="X3.2"]', (e) => e.textContent).catch(() => '')).includes('./verify.sh'));
  const pieceRows = allSets(reports, L2.id).flatMap((s) => s.rows.filter((r) => r.name === 'piece'));
  if (pieceRows.length) {
    const keys = new Set(allSets(reports, L2.id).flatMap((s) => s.rows.filter((r) => r.name === 'piece').map((r) => `${s.backend}|${s.variant}|${r.gemm}|${r.batch}|${r.position}`)));
    const shown = await page.$$eval('table[data-table="pieces"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const a = pieceRows.find((r) => r.group === 'attention');
    check(`per-piece table: ${keys.size} rows, attention cell is ms (share %) from the reports`,
      shown.length === keys.size && shown.some((c) => c[3] === String(a.batch) && c[4] === String(a.position) && c[6] === `${fmt(a.seconds * 1000)} (${fmt(a.share * 100)})`), JSON.stringify(shown[0]));
  } else check('per-piece table: no rows, and it says what to run', (await page.$eval('[data-pieces-empty]', (e) => e.textContent).catch(() => '')).includes('--bench'));
  const models = allSets(reports, L2.id).flatMap((s) => (row(s, 'model') ? [`${s.backend}:${s.variant}`] : []));
  const ml = await page.$eval('[data-model]', (e) => ({ v: e.dataset.model, t: e.textContent }));
  const m0 = allSets(reports, L2.id).map((s) => row(s, 'model')).find(Boolean);
  check(`model line reads the reports: ${models.join(',') || 'none'}`, ml.v === (models.join(',') || 'none') &&
    (!m0 || (ml.t.includes(`d ${m0.d.toLocaleString('en-US')} with ${m0.heads} query heads over ${m0.kv_heads} KV heads`) && ml.t.includes(`${fmt(m0.weight_bytes / 1e6)} MB`))), JSON.stringify(ml));
}

// ---- lab X3.2's launch table, from the tested model
{
  const shown = await page.$$eval('table[data-table="launches"] tbody tr', (trs) => trs.map((tr) => ({ job: tr.dataset.launch, cells: [...tr.children].map((td) => td.textContent.trim()), bold: !!tr.querySelector('td:last-child b') })));
  const want = M.jobs;
  const body = shown.filter((r) => r.job !== 'layers');
  check(`launch table: embed, one layer's sixteen launches and the tail, ${want.length} rows from sim/llamastep.ts, each with its index, launch and lab`,
    body.length === want.length && want.every(([n, job, fn, lab, exists], i) => body[i].job === job && body[i].cells[0] === String(n) && body[i].cells[2] === fn && body[i].cells[3] === lab && body[i].bold === !exists),
    JSON.stringify(body.slice(0, 3)));
  const mid = shown.find((r) => r.job === 'layers');
  check(`launch table: the elided layers are 17 to ${16 * M.layers}, and the caption gives ${M.count} launches = 5 + 16 × ${M.layers}`,
    mid && mid.cells[0] === `17–${16 * M.layers}` && M.count === M.formula && M.count === 5 + 16 * M.layers &&
    (await body0()).includes(`(${M.layers} layers, ${M.count} launches)`) && (await body0()).includes('5 + 16 × layers'), JSON.stringify(mid));
}
async function body0() { return page.$eval('.prose', (e) => e.textContent); }

// ---- reference solutions: hidden by default, then the real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), `x3/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  // The same rules as LabKit's solutionTabStatus: what passed, then what failed ("unverified on
  // the emulator: failed on this machine"), then the CUDA-family backends with no run.
  const st = (b) => labStatus(reports[b], l.id, 'solution');
  const WHERE = { emu: 'the emulator', cuda: 'CUDA', hip: 'HIP' };
  const fam = ['cuda', 'hip', 'emu'];
  const got = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const lacking = ['cuda', 'hip'].filter((b) => st(b) === 'none').map((b) => NAME[b]);
  const cuStatus = !got.length && !failed.length
    ? 'unverified: no run on this machine'
    : [got.join(', '), failed.length ? `unverified on ${failed.join(' and ')}: failed on this machine` : '', lacking.length ? `unverified on ${lacking.join(' and ')}` : '']
        .filter(Boolean).join('; ');
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
  if (l.num === 'X3.1') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('lab X3.1: tier colours are applied (global, shared and register)',
      (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  }
  await shot(`solution-${l.dir}`, `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to the start page, X3.3, X3.4, X3.7, X3.8 and X3.9', ['#/x3/start', '#/x3/X3.3', '#/x3/X3.4', '#/x3/X3.7', '#/x3/X3.8', '#/x3/X3.9'].every((h) => links.includes(h)), links.join(' '));
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
await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });

for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
