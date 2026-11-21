// Browser check for module X2's labs page: commands, the status table, the race line of lab
// X2.3, lab 5.1's bandwidth line and the "% of lab 5.1" columns, the nine predictions (eight
// against this machine's measurements, one with its own reveal), the four results tables and
// the four reference-solution toggles. Everything expected is computed here from /api/reports
// and /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/x2-labs.mjs [screenshot-prefix] [--dark]
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
  { num: 'X2.1', id: 'x2/lab1_q8gemv', dir: 'lab1_q8gemv' },
  { num: 'X2.2', id: 'x2/lab2_q4gemv', dir: 'lab2_q4gemv' },
  { num: 'X2.3', id: 'x2/lab3_qhunt', dir: 'lab3_qhunt' },
  { num: 'X2.4', id: 'x2/lab4_m9', dir: 'lab4_m9' },
];
const [L1, L2, L3, L4] = LABS;
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
// Lab 5.1's bandwidth per backend, as sim/batching.ts measuredDevices reads it.
const bandwidth = (reports, b) => {
  const r = results(reports[b], 'ch05/lab1_roofline', 'solution').flatMap((x) => x.bench).find((x) => x.name === 'roofline');
  return r && r.compute_gflops && r.bandwidth_gb_per_s ? r.bandwidth_gb_per_s : null;
};

await goto(page, 'x2/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 30000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);
const M = await page.evaluate(async () => {
  const t = await import('/src/sim/qtraffic.ts');
  return { q8: t.byteRatio('q8_0'), q4: t.byteRatio('q4_0'), q4q8: t.byteRatio('q4_0', 'q8_0') };
});

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Labs"', h1.trim().endsWith('Labs'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run X2.1', 'python3 labs/pmpp.py run X2.1 --bench', 'python3 labs/pmpp.py run X2.1 --backend emu',
  'python3 labs/pmpp.py run X2.1 --solution', 'python3 labs/pmpp.py reset X2.1 --yes', 'python3 labs/pmpp.py run X2.3 --backend race',
  'python3 labs/pmpp.py run X2.3 --backend emu', 'python3 labs/pmpp.py run X2.3'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour is explained for lab X2.3, with "evidence, not proof"', text0.includes('ThreadSanitizer') && text0.includes('evidence, not proof'));
check('the page says no emulator catches a misaligned load, and to check by reading', text0.includes('One defect no emulator can see') && text0.includes('Check those kernels by reading'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the module is labelled as ours', await page.$$eval('.prose > aside.note.beyond', (as) => as.some((a) => a.textContent.includes("Ours, not the book's") && a.textContent.includes('no integer quantisation'))));
check('the shared copies, stand-ins and two quantiser headers are named', text0.includes('labs/x2/shared-g12/qpieces.cmake') && text0.includes('labs/x2/lab4_m9/provided/pieces.cmake') &&
  text0.includes('labs/x2/shared/qformat.h') && text0.includes('labs/x2/shared-g12/qformats.h'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantLabs = ['Bandwidth utilisation', 'Lab X2.1: an int8-weight GEMV for batch-1 decode', "Lab X2.2: a 4-bit GEMV on llama.cpp's Q4_0 blocks",
  'Lab X2.3: bug hunt on the integer path', 'Lab X2.4: milestone M9, the quantised decode step'];
check('the bandwidth section and the four labs appear in order', wantLabs.every((t, i, a) => h2.indexOf(t) >= 0 && (i === 0 || h2.indexOf(t) > h2.indexOf(a[i - 1]))), JSON.stringify(h2));
check('the byte ratios in the prose come from the model', text0.includes(`1.0625 for Q8_0, ${fmt(M.q8)}`) && text0.includes(`4 / 0.5625 = ${fmt(M.q4)}`) && text0.includes(`1.0625 / 0.5625 = ${fmt(M.q4q8)}`));
check('lab X2.4 states the milestone with the project conventions', text0.includes('completes milestone M9') && text0.includes('row-major float32 activations with the batch first'));
check('lab X2.4 says how M10 calls it', text0.includes('labs/x3/lab2_tinyllama/provided/m9_linear') && text0.includes('same quantised weights'));
check('lab X2.4 says the toy model says nothing about a trained model', text0.includes("says nothing about a trained model's"));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the prose (no GB/s, KiB or measured ms outside the report-driven parts)',
  !(await page.$$eval('.prose > p', (ps) => ps.filter((p) => !p.dataset.measured && !p.dataset.race).some((p) => /\d+(\.\d+)? (GB\/s|KiB)\b|\d+\.\d+ ms\b/.test(p.textContent)))));

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

// ---- race line of lab X2.3
{
  const run = reports.race;
  const sol = labStatus(run, L3.id, 'solution');
  const race = await page.$eval(`[data-race="${L3.num}"]`, (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent })).catch(() => null);
  check(`race line of lab X2.3 reads the reports: solution ${run ? sol : 'no run'}`, race && race.s === sol && race.w === labStatus(run, L3.id, 'work') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked on this flavour') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified there')), JSON.stringify(race));
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

// ---- lab 5.1's bandwidth line
const bws = real(reports).flatMap((b) => (bandwidth(reports, b) ? [`${b}:${fmt(bandwidth(reports, b))}`] : []));
const bwLine = await page.$eval('p[data-measured]', (e) => ({ v: e.dataset.measured, t: e.textContent }));
check(`the bandwidth line reads lab 5.1 from the reports: ${bws.join(',') || 'none'}`, bwLine.v === (bws.join(',') || 'none') &&
  (bws.length ? bws.every((x) => bwLine.t.includes(`${x.split(':')[1]} GB/s on ${x.split(':')[0]}`)) : bwLine.t.includes('No run of lab 5.1')), JSON.stringify(bwLine));

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const ratio = (a, b) => (a && b ? a / b : undefined);
const q = (s, kernel, batch) => row(s, 'qgemv', { kernel, batch, in: 4096, out: 4096 });
const st = (s, l, h, batch) => row(s, 'qstep', { format_layers: l, format_head: h, batch });
const pct = (b, gbs) => (bandwidth(reports, b) && gbs ? fmt((100 * gbs) / bandwidth(reports, b)) : null);
const expected = {
  q8: preferred(reports, L1.id).flatMap((s) => {
    const v = ratio(q(s, 'gather', 1)?.seconds, q(s, 'q8_0', 1)?.seconds);
    const p = pct(s.backend, q(s, 'q8_0', 1)?.weight_gb_per_s);
    return v ? [{ backend: s.backend, value: v, detail: [`the byte ratio is ${fmt(M.q8)}`, p ? `at ${p} % of lab 5.1's bandwidth` : 'no lab 5.1 run'] }] : [];
  }),
  q8b8: preferred(reports, L1.id).flatMap((s) => {
    const v = ratio(q(s, 'gather', 8)?.seconds, q(s, 'q8_0', 8)?.seconds);
    const one = ratio(q(s, 'gather', 1)?.seconds, q(s, 'q8_0', 1)?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['at batch 8', one ? `at batch 1 it was ${fmt(one)}` : 'no batch-1 row'] }] : [];
  }),
  fused: preferred(reports, L2.id).flatMap((s) => {
    const v = ratio(q(s, 'gather', 1)?.seconds, q(s, 'q4_0', 1)?.seconds);
    const q8 = ratio(q(s, 'q8_0', 1)?.seconds, q(s, 'q4_0', 1)?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: [`the byte ratio is ${fmt(M.q4)}`, q8 ? `over lab X2.1's Q8_0 it is ${fmt(q8)}, against a byte ratio of ${fmt(M.q4q8)}` : 'no Q8_0 row'] }] : [];
  }),
  unfused: preferred(reports, L2.id).flatMap((s) => {
    const v = ratio(q(s, 'gather', 1)?.seconds, q(s, 'q4_0_unfused', 1)?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['dequantise-then-gather'] }] : [];
  }),
  path: preferred(reports, L3.id).flatMap((s) => {
    const i = row(s, 'qhunt', { path: 'int8_q8_1', batch: 1, in: 4096, out: 4096 });
    const f = row(s, 'qhunt', { path: 'float_q4_0', batch: 1, in: 4096, out: 4096 });
    const i8 = row(s, 'qhunt', { path: 'int8_q8_1', batch: 8, in: 4096, out: 4096 });
    const f8 = row(s, 'qhunt', { path: 'float_q4_0', batch: 8, in: 4096, out: 4096 });
    const v = ratio(i?.seconds_total, f?.seconds_total);
    return v ? [{ backend: s.backend, value: v, detail: ['of which the quantise pass', i8 && f8 ? `at batch 8 the ratio is ${fmt(i8.seconds_total / f8.seconds_total)}` : 'no batch-8 rows'] }] : [];
  }),
  step: preferred(reports, L4.id).flatMap((s) => {
    const v = ratio(st(s, 'f32', 'f32', 1)?.seconds, st(s, 'q4_0', 'q4_0', 1)?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: [`the bytes model says about ${fmt(M.q4)}`] }] : [];
  }),
  head: preferred(reports, L4.id).flatMap((s) => {
    const v = ratio(st(s, 'f32', 'f32', 1)?.seconds, st(s, 'q4_0', 'f32', 1)?.seconds);
    return v ? [{ backend: s.backend, value: v, detail: ['the bytes model says 1.21'] }] : [];
  }),
  agree: preferred(reports, L4.id).flatMap((s) => {
    const r4 = st(s, 'q4_0', 'q4_0', 1);
    const r8 = st(s, 'q8_0', 'q8_0', 1);
    return r4 && typeof r4.agree_steps === 'number'
      ? [{ backend: s.backend, value: r4.agree_steps, detail: [`off by ${fmt(100 * r4.logits_rel_rms)} % relative RMS`, r8 ? `Q8_0 agreed for ${r8.agree_steps} steps` : ''] }]
      : [];
  }),
};
const predictions = [
  { key: 'q8', id: 'x2/labs/p-lab1-speedup', find: 'how many times faster than lab 22.1', guess: 3, lab: 'X2.1' },
  { key: 'q8b8', id: 'x2/labs/p-lab1-batch8', find: 'And at batch 8, same layer', guess: 3, lab: 'X2.1' },
  { key: 'fused', id: 'x2/labs/p-lab2-fused', find: 'your fused Q4_0 kernel be', guess: 6, lab: 'X2.2' },
  { key: 'unfused', id: 'x2/labs/p-lab2-unfused', find: 'the provided unfused path, dequantise then gather', guess: 0.5, lab: 'X2.2' },
  { key: 'path', id: 'x2/labs/p-lab3-path', find: "the integer path's time divided by the float path's", guess: 1, lab: 'X2.3' },
  { key: 'step', id: 'x2/labs/p-lab4-speedup', find: 'will the (Q4_0, Q4_0) step be', guess: 5, lab: 'X2.4' },
  { key: 'head', id: 'x2/labs/p-lab4-head', find: 'with only the layers quantised', guess: 1.2, lab: 'X2.4' },
  { key: 'agree', id: 'x2/labs/p-lab4-agree', find: 'Of 32 greedy steps', guess: 20, lab: 'X2.4' },
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
// Lab X2.3's hunt prediction has its own reveal: what emu reported when the lab was built.
{
  const el = await predictBlock('how many will emu report');
  check('hunt: prediction block present, nothing revealed before commit', !!el && (await el.$('[data-reveal="p-lab3-emu"]')) === null);
  if (el) {
    await commit(el, 3);
    const t = await el.$eval('[data-reveal="p-lab3-emu"]', (e) => e.textContent).catch(() => '');
    check('hunt: the reveal gives 4 of 4, says two need particular cases, and names no defect',
      t.includes('reported 4 of the four') && t.includes('You predicted 3.') && t.includes('only one constructed block reaches') &&
      t.includes('passed it on every case') && !/short|partial\[|2i \+ 1|defect [1234]/i.test(t), t);
    const clicked = await el.evaluate((e) => {
      const b = [...e.querySelectorAll('[data-reveal="p-lab3-emu"] button')].find((x) => x.textContent.includes('I have fixed all four'));
      if (!b) return false;
      b.click();
      return true;
    });
    await pause(150);
    const d = await el.$eval('[data-reveal-detail="p-lab3-emu"]', (e) => e.textContent).catch(() => '');
    check('hunt: after "I have fixed all four", the detail names each defect and what saw it',
      clicked && d.includes('defect 1 (activation words 2i and 2i + 1)') && d.includes('Defect 2 (the short block sum) failed only the worst-case block') &&
      d.includes('60,960') && d.includes('Defect 3') && d.includes('Defect 4 (the missing barrier)') && d.includes('partial[warp][b]') && d.includes('Metal passed all 8 cases'), d);
  }
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all nine predictions were sent to /api/progress under their ids, and none reached the real file',
  [...predictions.map((p) => [p.id, p.guess]), ['x2/labs/p-lab3-emu', 3]].every(([id, g]) => ans[id]?.value === g), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results tables: every timing row in the reports, and nothing invented
const KERNEL = { gather: 'F32 gather (lab 22.1)', q8_0: 'Q8_0 (lab X2.1)', q4_0_unfused: 'Q4_0, dequantise then gather (provided)', q4_0: 'Q4_0 fused (lab X2.2)' };
const FMT = { f32: 'F32', q8_0: 'Q8_0', q4_0: 'Q4_0' };
for (const l of LABS) {
  const sets = allSets(reports, l.id);
  const rows = sets.flatMap((s) => s.rows.map((r) => ({ ...r, backend: s.backend })));
  if (!rows.length) {
    const empty = await page.$eval(`[data-results-empty="${l.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${l.num}: no benchmark rows, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
    continue;
  }
  const shown = await page.$$eval(`table[data-results="${l.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  const util = (r) => (bandwidth(reports, r.backend) ? fmt((100 * r.weight_gb_per_s) / bandwidth(reports, r.backend)) : '');
  let ok;
  if (l === L1 || l === L2)
    ok = rows.every((r, i) => shown[i] && shown[i][2] === KERNEL[r.kernel] && shown[i][3] === String(r.batch) && shown[i][6] === fmt(r.seconds * 1000) &&
      shown[i][7] === fmt(r.weight_gb_per_s) && shown[i][8] === util(r) && shown[i][9] === fmt(r.speedup_vs_f32));
  else if (l === L3)
    ok = rows.every((r, i) => shown[i] && shown[i][2].startsWith(r.path === 'int8_q8_1' ? 'quantise to Q8_1' : 'fused float Q4_0 GEMV') &&
      shown[i][3] === String(r.batch) && shown[i][6] === fmt(r.seconds_quantise * 1000) && shown[i][8] === fmt(r.seconds_total * 1000));
  else
    ok = rows.every((r, i) => shown[i] && shown[i][2] === `${FMT[r.format_layers]} / ${FMT[r.format_head]}` && shown[i][3] === String(r.batch) &&
      shown[i][4] === fmt(r.seconds * 1000) && shown[i][6] === fmt(r.weight_bytes / 1e6) && shown[i][8] === util(r) && shown[i][10] === fmt(r.logits_rel_rms * 100) && shown[i][11] === String(r.agree_steps));
  check(`lab ${l.num}: results table has the ${rows.length} rows from the reports, with what ran, time${l === L3 ? '' : ' and % of lab 5.1'}`,
    ok && shown.length === rows.length, JSON.stringify(shown.slice(0, 2)));
}
await shot('results', `table[data-results="X2.2"], [data-results-empty="X2.2"]`);

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
    files[f] = await page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), `x2/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  const stt = (b) => labStatus(reports[b], l.id, 'solution');
  const got = [stt('cuda') === 'pass' ? 'verified on CUDA' : '', stt('hip') === 'pass' ? 'verified on HIP' : '', stt('emu') === 'pass' ? 'emulator-checked' : ''].filter(Boolean);
  const lacking = ['cuda', 'hip'].filter((b) => stt(b) !== 'pass').map((b) => NAME[b]);
  const cuStatus = got.length ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified (no run on this machine)';
  const metalStatus = stt('metal') === 'pass' ? 'verified on Metal' : stt('metal') === 'fail' ? 'unverified (failed on this machine)' : 'unverified (no run on this machine)';
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
  check(`lab ${l.num}: tier colours are applied to the CUDA file (global and register${l === L3 ? ', shared' : ''})`,
    (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0 && (l !== L3 || (await page.$$('figure.code .v-shared')).length > 0));
  await shot(`solution-${l.dir}`, `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to X2.1, X2.3, X2.4, X2.5, X2.6 and X2.7', ['#/x2/X2.1', '#/x2/X2.3', '#/x2/X2.4', '#/x2/X2.5', '#/x2/X2.6', '#/x2/X2.7'].every((h) => links.includes(h)), links.join(' '));
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

// Section pages with narrated scenes ask for a manifest and clips that are not recorded yet.
const real404 = errors.filter((e) => !/narration\/(manifest\.json|[a-z-]+-\d+\.mp3)/.test(e) && !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (the missing narration files of linked sections allowed)', real404.length === 0, real404.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
