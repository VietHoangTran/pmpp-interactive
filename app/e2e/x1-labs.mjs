// Browser check for module X1's labs page: commands, the status table, the race line of each
// lab, the six predictions (five against this machine's measurements, one with its own
// reveal), the three results tables, lab X1.3's stage table and tokens digest, and the four
// reference-solution toggles. Everything expected is computed here from /api/reports and
// /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/x1-labs.mjs [screenshot-prefix] [--dark]
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
  { num: 'x1/lab1', id: 'x1/lab1_topk', dir: 'lab1_topk', name: 'top-k by radix select (X1.1)' },
  { num: 'x1/lab2', id: 'x1/lab2_filters', dir: 'lab2_filters', name: 'top-p and min-p, composed and fused (X1.2)' },
  { num: 'x1/lab3', id: 'x1/lab3_sampler', dir: 'lab3_sampler', name: 'the M5 sampler (X1.3)' },
  { num: 'x1/lab4', id: 'x1/lab4_filterhunt', dir: 'lab4_filterhunt', name: 'filter hunt (X1.4)' },
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
const n = (v) => v.toLocaleString('en-US');
const V = 151936;

await goto(page, 'x1/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 30000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Labs"', h1.trim().endsWith('Labs'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run x1/lab1', 'python3 labs/pmpp.py run x1/lab1 --bench', 'python3 labs/pmpp.py run x1/lab1 --backend emu',
  'python3 labs/pmpp.py run x1/lab1 --solution', 'python3 labs/pmpp.py reset x1/lab1 --yes', 'python3 labs/pmpp.py run x1/lab1 --backend race',
  'python3 labs/pmpp.py run x1/lab4 --backend emu', 'python3 labs/pmpp.py run x1/lab4 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the page says why the labs are named by path', text0.includes('Name them by path') && text0.includes('x1/lab1 is lab X1.1'));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is explained, with "evidence, not proof"', text0.includes('ThreadSanitizer') && text0.includes('evidence, not proof'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the module is labelled as ours', await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes('The book has no sampling'))));
check('the provided copies are named with their tests', text0.includes('labs/x1/shared-g11/') && text0.includes('labs/x1/shared-g12/') && text0.includes('copies-g11.test.ts') && text0.includes('copies-g12.test.ts'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantLabs = ['Lab X1.1: top-k by radix select, measured against a full sort', 'Lab X1.2: top-p and min-p, composed from earlier labs, then fused',
  'Lab X1.3: milestone M5, a batched GPU sampler', 'Lab X1.4: filter hunt, three defects in a fused top-p and min-p kernel'];
check('the four labs appear in order', wantLabs.every((t, i, a) => h2.indexOf(t) >= 0 && (i === 0 || h2.indexOf(t) > h2.indexOf(a[i - 1]))), JSON.stringify(h2));
check('lab X1.3 states the milestone with the project conventions', text0.includes('completes milestone M5') && text0.includes('row-major float32 with the batch first'));
check('lab X1.3 says which draws must match', text0.includes('Tokens are compared only where the draw is decisive') && text0.includes('counted and printed, never failed'));
check('the Philox mapping is labelled unverified', text0.includes('that is unverified on a device'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (no GB/s, KiB or measured ms outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.measured && !p.dataset.raceLab && !p.dataset.digest).some((p) => /\d+(\.\d+)? (GB\/s|KiB)\b|\d+\.\d+ ms\b/.test(p.textContent)))));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 4 labs x 4 backends', cells.length === 16, String(cells.length));
for (const l of LABS)
  for (const b of BACKENDS) {
    const c = cells.find((x) => x.lab === l.num && x.backend === b);
    const want = solutionText(reports, b, l.id);
    check(`status ${l.num} ${b}: "${want}"`, c && c.status === want && c.text === want &&
      (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
  }
await shot('status', 'table[data-table="solution-status"]');

// ---- race lines, one per lab
for (const l of LABS) {
  const run = reports.race;
  const sol = labStatus(run, l.id, 'solution');
  const race = await page.$eval(`[data-race-lab="${l.num}"]`, (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent })).catch(() => null);
  check(`race line of lab ${l.num} reads the reports: solution ${run ? sol : 'no run'}`, race && race.s === sol && race.w === labStatus(run, l.id, 'work') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified')), JSON.stringify(race));
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
const ratio = (a, b) => (a && b ? a / b : undefined);
const expected = {
  sort: preferred(reports, L1.id).flatMap((s) => {
    const v = ratio(row(s, 'full_sort_then_first_k', { rows: 64, cols: V, k: 40 })?.seconds, row(s, 'topk_select', { rows: 64, cols: V, k: 40 })?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['64 × 151,936, k = 40'] }] : [];
  }),
  fused: preferred(reports, L2.id).flatMap((s) => {
    const v = ratio(row(s, 'filters_unfused', { rows: 64, cols: V })?.seconds, row(s, 'filters_fused', { rows: 64, cols: V })?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['GB/s counted', '64 × 151,936'] }] : [];
  }),
  share: preferred(reports, L2.id).flatMap((s) => {
    const r = row(s, 'chain_composed', { rows: 64, cols: V });
    const small = row(s, 'chain_composed', { rows: 64, cols: 40 });
    return r ? [{ backend: s.backend, value: r.sort_share * 100, detail: ['64 rows of K = 151,936', small ? `at K = 40 the sort's share is ${fmt(small.sort_share * 100)} %` : 'no K = 40 row'] }] : [];
  }),
  cpu: preferred(reports, L3.id).flatMap((s) => {
    const v = ratio(row(s, 'sample_step', { rows: 1, vocab: V })?.seconds, row(s, 'cpu_sample_step', { rows: 1, vocab: V })?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['1 × 151,936, top-k 40'] }] : [];
  }),
  batch: preferred(reports, L3.id).flatMap((s) => {
    const v = ratio(row(s, 'sample_step', { rows: 64, vocab: V })?.seconds, row(s, 'sample_step', { rows: 1, vocab: V })?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['at batch 64', 'vocabulary 151,936'] }] : [];
  }),
};
const predictions = [
  { key: 'sort', id: 'x1/labs/p-lab1-sort', find: "how many times faster than lab 14.3's full sort", guess: 4, lab: 'x1/lab1' },
  { key: 'fused', id: 'x1/labs/p-lab2-fused', find: 'how many times faster than the composed chain after the sort', guess: 2, lab: 'x1/lab2' },
  { key: 'share', id: 'x1/labs/p-lab2-sort', find: 'what percentage of the composed chain', guess: 50, lab: 'x1/lab2' },
  { key: 'cpu', id: 'x1/labs/p-lab3-cpu', find: "divided by one CPU thread's", guess: 0.5, lab: 'x1/lab3' },
  { key: 'batch', id: 'x1/labs/p-lab3-batch', find: 'How many times longer will a sampling step take', guess: 3, lab: 'x1/lab3' },
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
// Lab X1.4's prediction has its own reveal: what emu reported when the lab was built.
{
  const el = await predictBlock('how many will emu report');
  check('hunt: prediction block present, nothing revealed before commit', !!el && (await el.$('[data-reveal="p-lab4-emu"]')) === null);
  if (el) {
    await commit(el, 3);
    const t = await el.$eval('[data-reveal="p-lab4-emu"]', (e) => e.textContent).catch(() => '');
    check('hunt: the reveal gives 2 of 3, says why a race between warps passes emu and that race reports it, and names no defect',
      t.includes('reported 2 of the three') && t.includes('You predicted 3.') && t.includes('The one it missed is a race between warps') &&
      t.includes('The race backend reports such a pair of lines') && !/inclusive mass|carry added|missing barrier|defect [123]/i.test(t), t);
    const clicked = await el.evaluate((e) => {
      const b = [...e.querySelectorAll('[data-reveal="p-lab4-emu"] button')].find((x) => x.textContent.includes('I have fixed all three'));
      if (!b) return false;
      b.click();
      return true;
    });
    await pause(150);
    const d = await el.$eval('[data-reveal-detail="p-lab4-emu"]', (e) => e.textContent).catch(() => '');
    check('hunt: after "I have fixed all three", the detail names defects 1 and 2 and says defect 3 passed emu',
      clicked && d.includes('defect 1 (the inclusive mass)') && d.includes('defect 2 (the carry added after the decision)') &&
      d.includes('Defect 3, the missing barrier, passed every emu case'), d);
  }
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all six predictions were sent to /api/progress under their ids, and none reached the real file',
  [...predictions.map((p) => [p.id, p.guess]), ['x1/labs/p-lab4-emu', 3]].every(([id, g]) => ans[id]?.value === g), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results tables: every timing row in the reports, and nothing invented
const WHAT = {
  topk_select: 'your select (both kernels)', full_sort_then_first_k: 'lab 14.3 sort, first k', cpu_partial_sort: 'std::partial_sort, one CPU thread',
  sort_rows: 'lab 14.3 sort alone', chain_composed: 'launchFilterRows, sort included', filters_unfused: 'composed chain after the sort', filters_fused: 'fused kernel after the sort',
  sample_step: 'launchSampleRows', cpu_sample_step: 'CPU reference chain, one thread', sample_step_whole_row: 'launchSampleRows, top-k 0', cpu_sample_step_whole_row: 'CPU chain, top-k 0, one thread',
};
const TIME_COL = { 'x1/lab1': 6, 'x1/lab2': 5, 'x1/lab3': 6 };
for (const l of LABS.slice(0, 3)) {
  const rows = allSets(reports, l.id).flatMap((s) => s.rows);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${l.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const tc = TIME_COL[l.num];
    check(`lab ${l.num}: results table has the ${rows.length} rows from the reports, each with what ran and its time`,
      shown.length === rows.length && rows.every((r, i) => shown[i][2] === WHAT[r.name] && shown[i][tc] === fmt(r.seconds * 1000)), JSON.stringify(shown.slice(0, 2)));
    if (l.num === 'x1/lab1') check('lab x1/lab1: rows, cols and k are shown', rows.every((r, i) => shown[i][3] === String(r.rows) && shown[i][4] === n(r.cols) && shown[i][5] === String(r.k)));
    if (l.num === 'x1/lab2') {
      const c = rows.findIndex((r) => r.name === 'chain_composed');
      if (c >= 0) check('lab x1/lab2: the sort share is shown in percent', shown[c][8] === fmt(rows[c].sort_share * 100), JSON.stringify(shown[c]));
    }
    if (l.num === 'x1/lab3') check('lab x1/lab3: rows, vocab and top-k are shown', rows.every((r, i) => shown[i][3] === String(r.rows) && shown[i][4] === n(r.vocab) && shown[i][5] === String(r.top_k)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${l.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${l.num}: no benchmark rows, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
  }
}
const hunt = await page.$eval('section.bench[aria-label="Lab x1/lab4 on this machine"]', (e) => e.textContent).catch(() => '');
check('lab x1/lab4 has no benchmark, and says so', hunt.includes('This lab has no benchmark') && hunt.includes('meant to fail'), hunt.slice(0, 120));

// ---- lab X1.3's stage table and digest
const stageRows = allSets(reports, L3.id).flatMap((s) => s.rows.filter((r) => (r.name === 'sample_step' || r.name === 'sample_step_whole_row') && typeof r.stage_0_s === 'number'));
if (stageRows.length) {
  const shown = await page.$$eval('table[data-table="stages"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  check(`stage table: ${stageRows.length} rows, seven stages each, in ms from the reports`,
    shown.length === stageRows.length && stageRows.every((r, i) => [0, 1, 2, 3, 4, 5, 6].every((k) => shown[i][5 + k] === fmt(r[`stage_${k}_s`] * 1000))), JSON.stringify(shown[0]));
  const heads = await page.$$eval('table[data-table="stages"] th', (ths) => ths.map((t) => t.textContent));
  check('stage table heads name the seven stages of the driver', ['penalties, ms', 'top-k select, ms', 'sort of k, ms', 'softmax, ms', 'top-p, ms', 'min-p + combine, ms', 'draw, ms'].every((h) => heads.includes(h)), heads.join('|'));
} else check('stage table: no rows, and it says what to run', (await page.$eval('[data-stages-empty]', (e) => e.textContent).catch(() => '')).includes('--bench'));
// Full digest (devices) and core digest (every backend); a run from before the core existed wrote
// tokens_digest alone, over exactly the core's draws.
const digests = Object.keys(reports).flatMap((b) => ['solution', 'work'].flatMap((v) => {
  const t = results(reports[b], L3.id, v).flatMap((x) => x.tests).find((x) => typeof x.core_digest === 'string' || typeof x.tokens_digest === 'string');
  if (!t || !t.pass || (v === 'work' && reports[b].work_state?.[L3.id] === 'untouched')) return [];
  const legacy = typeof t.core_digest !== 'string';
  const core = legacy ? t.tokens_digest : t.core_digest;
  const coreDraws = legacy ? t.decisive_draws ?? 0 : t.core_draws ?? 0;
  const full = !legacy && typeof t.tokens_digest === 'string' && (t.decisive_draws ?? 0) > 0 ? t.tokens_digest : '';
  return coreDraws > 0 || full ? [{ b, v, full, core, fullDraws: full ? t.decisive_draws : null, coreDraws }] : [];
}));
const dshown = await page.$$eval('table[data-table="digest"] tbody tr', (trs) => trs.map((tr) => ({
  b: tr.dataset.digestBackend, v: tr.dataset.digestVariant, full: tr.dataset.full, core: tr.dataset.core,
  fullCell: tr.children[2].textContent.trim(), fullDraws: tr.children[3].textContent.trim(), coreCell: tr.children[4].textContent.trim(), coreDraws: tr.children[5].textContent.trim(),
})));
check(`digest table lists the ${digests.length} runs' digests in the reports, with the draws each covers`,
  dshown.length === digests.length && digests.every((x) => dshown.some((y) => y.b === x.b && y.v === x.v && y.full === x.full && y.core === x.core &&
    y.coreCell === x.core && y.coreDraws === String(x.coreDraws) && (x.full ? y.fullCell === x.full && y.fullDraws === String(x.fullDraws) : y.fullCell === 'core only'))),
  JSON.stringify(dshown));
const dstate = await page.$eval('[data-digest]', (e) => e.dataset.digest);
const fullDistinct = new Set(digests.filter((x) => x.full).map((x) => x.full)).size;
const coreDistinct = new Set(digests.map((x) => x.core)).size;
const dwant = digests.length ? (fullDistinct <= 1 && coreDistinct === 1 ? 'same' : 'differ') : 'none';
check(`digest sentence: ${dwant}`, dstate === dwant, dstate);
check('the lesson states the draws each digest covers', text0.includes('1,024 draws') && text0.includes('2 rows by 2 seeds by 2 steps, 8 draws'));
await shot('digest', 'table[data-table="digest"], [data-digest]');

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
    files[f] = await page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), `x1/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  const st = (b) => labStatus(reports[b], l.id, 'solution');
  // The words of LabKit's solutionTabStatus.
  const fam = ['cuda', 'hip', 'emu'];
  const WHERE = { cuda: 'CUDA', hip: 'HIP', emu: 'the emulator' };
  const and = (xs) => (xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));
  const passed = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const none = ['cuda', 'hip'].filter((b) => st(b) === 'none').map((b) => NAME[b]);
  const cuStatus = !passed.length && !failed.length
    ? 'unverified: no run on this machine'
    : [passed.join(', '), failed.length ? `unverified on ${and(failed)}: failed on this machine` : '', none.length ? `unverified on ${none.join(' and ')}` : ''].filter(Boolean).join('; ');
  const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : `unverified: ${st('metal') === 'fail' ? 'failed on this machine' : 'no run on this machine'}`;
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
  check(`lab ${l.num}: tier colours are applied (global, shared and register)`,
    (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  await shot(`solution-${l.dir}`, `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to X1.1, X1.3 to X1.8 and 10.10', ['#/x1/X1.1', '#/x1/X1.3', '#/x1/X1.4', '#/x1/X1.5', '#/x1/X1.6', '#/x1/X1.7', '#/x1/X1.8', '#/ch10/10.10'].every((h) => links.includes(h)), links.join(' '));
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
