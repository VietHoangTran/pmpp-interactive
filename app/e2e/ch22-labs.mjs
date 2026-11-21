// Browser check for the chapter 22 labs page: status table, the race line, the four
// predictions against this machine's measurements, the results tables, the bandwidth and
// batching summaries, the piece table and both reference-solution toggles. Everything
// expected is computed here from /api/reports, /api/lab-file and src/sim/batching.ts, so the
// check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch22-labs.mjs [screenshot-prefix] [--dark]
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
const L1 = { num: '22.1', id: 'ch22/lab1_gemv', dir: 'lab1_gemv' };
const L2 = { num: '22.2', id: 'ch22/lab2_batching', dir: 'lab2_batching' };
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

await goto(page, 'ch22/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
const B = await page.evaluate(async () => {
  const m = await import('/src/sim/batching.ts');
  return { gemms: Object.keys(m.M8_GEMMS) };
});
const kneeOf = (pts) => page.evaluate(async (pts) => (await import('/src/sim/batching.ts')).kneeOf(pts), pts);
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = async () => (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');

// ---- structure, commands and backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Labs"', h1.trim().endsWith('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
check('standard sections, both labs and the milestone, in order',
  JSON.stringify(h2) === JSON.stringify(['Running them', 'The four backends', 'Verification status of the reference solutions',
    'Lab 22.1: two ways to cut a decode-time matrix-vector product', 'Lab 22.2: the batching study (milestone M8)', 'Milestone M8']), JSON.stringify(h2));
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 22.1', 'python3 labs/pmpp.py run 22.1 --bench', 'python3 labs/pmpp.py run 22.1 --backend emu',
  'python3 labs/pmpp.py run 22.1 --solution', 'python3 labs/pmpp.py reset 22.1 --yes', 'python3 labs/pmpp.py run 22.2 --backend emu'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('lab 22.1: emu cannot see the plain +=, race can, Metal shows it only as wrong sums',
  text0.includes('the plain += passes there, every time') && text0.includes('ThreadSanitizer') && text0.includes('only as a wrong sum'));
check('lab 22.1: the Metal fix and the HIP caveat are named (atomic_float, relaxed order, -munsafe-fp-atomics)',
  text0.includes('device atomic_float*') && text0.includes('memory_order_relaxed') && text0.includes('-munsafe-fp-atomics'));
check('lab 22.1: no literal 32, and nothing on a Mac checks a 64-lane warp', text0.includes('Never write a literal 32') && text0.includes('nothing on a Mac checks a 64-lane warp'));
check('lab 22.2: the step table names every piece and its lab', (await page.$$eval('table[data-table="step"] tbody tr', (t) => t.length)) === 5 &&
  ['lab 6.4', 'lab 15.6', 'lab 2.3', 'lab 10.3', "lab 22.1's gather"].every((s) => text0.includes(s)));
check('lab 22.2: attention is left out and why; the probabilities are for M5', text0.includes('attention does not batch') && text0.includes('milestone M5 needs them'));
check('lab 22.2: the three things to weigh (2T, M3 flat to 128, elementwise grows) and the cache caveat',
  text0.includes('twice your threshold') && text0.includes('flat up to 128') && text0.includes('grow with the batch from the start') && text0.includes('Apple documents no cache sizes'));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 2 labs x 4 backends', cells.length === 8, String(cells.length));
for (const L of [L1, L2])
  for (const b of BACKENDS) {
    const c = cells.find((x) => x.lab === L.num && x.backend === b);
    const want = solutionText(reports, b, L.id);
    check(`status ${L.num} ${b}: "${want}"`, c && c.status === want && c.text === want, JSON.stringify(c));
    check(`  ${L.num} ${b}: a pass is green, a failure red, no run is neither`,
      c && (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
  }
await shot('status', 'table[data-table="solution-status"]');

// ---- the race line
{
  const run = reports.race;
  const sol = labStatus(run, L1.id, 'solution');
  const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, text: e.textContent }));
  check(`the race line reads the reports: solution ${run ? sol : 'no run'}`, race.s === (run ? sol : 'none') &&
    (!run ? race.text.includes('No run on the race backend') : sol === 'pass' ? race.text.includes('emulator-checked') : sol === 'none' ? race.text.includes('no run on the race backend yet') : race.text.includes('unverified')), JSON.stringify(race));
}

// ---- status of the reader's own work, both labs
for (const L of [L1, L2]) {
  const row = await page.$$eval(`[data-work="${L.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return row[b].includes('no run on this machine');
    const rs = results(run, L.id, 'work');
    const total = rs.reduce((k, x) => k + x.tests.length, 0);
    const passed = rs.reduce((k, x) => k + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, L.id, 'work');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${L.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const secs = (r) => (typeof r?.seconds === 'number' ? r.seconds : 0);
const gain = (s, inn, outn) => {
  const rows = s.rows.filter((r) => r.name === 'gemv' && r.in === inn && r.out === outn && r.batch === 1);
  const g = rows.find((r) => r.kernel === 'gather' && secs(r));
  const sp = rows.filter((r) => r.kernel === 'splitk' && secs(r)).sort((a, b) => secs(a) - secs(b));
  return g && sp.length ? { value: secs(g) / secs(sp[0]), also: [`S = ${sp[0].splits}`] } : null;
};
const steps = (s, gemm) => s.rows.filter((r) => r.name === 'step' && r.gemm === gemm && r.batch && secs(r)).sort((a, b) => a.batch - b.batch);
const expected = { narrow: [], square: [], knee: [], gain64: [] };
for (const s of preferred(reports, L1.id)) {
  const nw = gain(s, 4096, 16);
  if (nw) expected.narrow.push({ backend: s.backend, ...nw });
  const sq = gain(s, 4096, 4096);
  if (sq) expected.square.push({ backend: s.backend, ...sq });
}
const noKnee = [];
for (const s of preferred(reports, L2.id)) {
  const rows = steps(s, 'm3');
  const knee = rows.length ? await kneeOf(rows.map((r) => ({ batch: r.batch, seconds: secs(r) }))) : null;
  if (knee != null) expected.knee.push({ backend: s.backend, value: knee, also: [`at batch ${knee}`] });
  else if (rows.length) noKnee.push(s.backend);
  const b1 = rows.find((r) => r.batch === 1);
  const b64 = rows.find((r) => r.batch === 64);
  if (b1 && b64) expected.gain64.push({ backend: s.backend, value: (64 / secs(b64)) / (1 / secs(b1)), also: ['tokens per second at batch 1'] });
}
const predictions = [
  { key: 'narrow', id: 'ch22/labs/p-narrow', find: 'On the 4096 → 16 layer at batch 1', guess: 4, run: '22.1' },
  { key: 'square', id: 'ch22/labs/p-square', find: 'On the 4096 → 4096 layer at batch 1', guess: 1, run: '22.1' },
  { key: 'knee', id: 'ch22/labs/p-knee-m3', find: 'With M3 as the step', guess: 64, run: '22.2' },
  { key: 'gain64', id: 'ch22/labs/p-gain64', find: 'how many times the tokens per second of batch 1 will batch 64', guess: 30, run: '22.2' },
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
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else if (p.key === 'knee' && noKnee.length) {
    check('knee: rows without a knee are reported as a result, not as missing', shown.length === 1 && shown[0].text.includes('No knee in the measured range'), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(`python3 labs/pmpp.py run ${p.run} --bench`), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('the four predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results tables: every timing row in the reports, and nothing invented
const allRows = (id, name) => real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')].filter((r) => r.name === name);
});
{
  const want = allRows(L1.id, 'gemv');
  if (want.length) {
    const shown = await page.$$eval('table[data-results="22.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab 22.1 results: the ${want.length} gemv rows, with kernel, S, batch, layer, µs and W GB/s`,
      shown.length === want.length && want.every((r, i) =>
        shown[i][2] === (r.kernel === 'gather' ? 'gather' : 'split-K') && shown[i][3] === String(r.splits) && shown[i][4] === String(r.batch) &&
        shown[i][5] === `${r.in.toLocaleString('en-US')} → ${r.out.toLocaleString('en-US')}` && shown[i][6] === fmt(r.seconds * 1e6) && shown[i][7] === fmt(r.weight_gb_per_s)),
      JSON.stringify(shown.slice(0, 2)));
  } else {
    const empty = await page.$eval('[data-results-empty="22.1"]', (e) => e.textContent).catch(() => '');
    check('lab 22.1: no benchmark rows, and the empty state says what to run', empty.includes('./verify.sh'), empty);
  }
}
{
  const want = allRows(L2.id, 'step');
  if (want.length) {
    const shown = await page.$$eval('table[data-results="22.2"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab 22.2 results: the ${want.length} step rows, with GEMM, batch, ms, tokens/s, W GB/s and model FLOP/B`,
      shown.length === want.length && want.every((r, i) =>
        shown[i][2] === String(r.gemm).toUpperCase() && shown[i][3] === String(r.batch) && shown[i][4] === fmt(r.seconds * 1e3) &&
        shown[i][5] === fmt(r.tokens_per_s) && shown[i][6] === fmt(r.weight_gb_per_s) && shown[i][7] === fmt(r.model_flop_per_byte)),
      JSON.stringify(shown.slice(0, 2)));
  } else {
    const empty = await page.$eval('[data-results-empty="22.2"]', (e) => e.textContent).catch(() => '');
    check('lab 22.2: no benchmark rows, and the empty state says what to run', empty.includes('./verify.sh'), empty);
  }
}

// ---- the bandwidth table: best weight GB/s at batch 1 against lab 5.1's measured bandwidth
{
  const sets = preferred(reports, L1.id);
  const bwOf = (b) => benchRows(reports[b], 'ch05/lab1_roofline', 'solution').find((r) => r.name === 'roofline')?.bandwidth_gb_per_s;
  if (sets.length) {
    const shown = await page.$$eval('table[data-table="speed-of-light"] tbody tr', (trs) => trs.map((tr) => ({ layer: tr.dataset.layer, backend: tr.dataset.backend, gbs: tr.dataset.gbs, cells: [...tr.children].map((td) => td.textContent.trim()) })));
    const want = sets.flatMap((s) => [[4096, 4096], [4096, 16], [14336, 4096]].flatMap(([i, o]) => {
      const rows = s.rows.filter((r) => r.name === 'gemv' && r.in === i && r.out === o && r.batch === 1 && r.weight_gb_per_s);
      if (!rows.length) return [];
      const best = rows.reduce((a, b) => (secs(b) < secs(a) ? b : a));
      const bw = bwOf(s.backend);
      return [{ layer: `${i}-${o}`, backend: s.backend, gbs: fmt(best.weight_gb_per_s), share: bw ? `${fmt((100 * best.weight_gb_per_s) / bw)}%` : '' }];
    }));
    check(`bandwidth table: ${want.length} rows, the fastest kernel's W GB/s and its share of lab 5.1's bandwidth`,
      shown.length === want.length && want.every((w, i) => shown[i].layer === w.layer && shown[i].gbs === w.gbs && shown[i].cells[5] === w.share),
      JSON.stringify(shown.map((s) => s.cells)));
  } else {
    check('bandwidth table: no rows, and it says so', (await page.$('[data-sol="none"]')) !== null);
  }
}

// ---- the threshold line and the batching summary
{
  const devs = real(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline', 'solution').find((x) => x.name === 'roofline');
    return r?.compute_gflops && r?.bandwidth_gb_per_s ? [{ b, t: r.compute_gflops / r.bandwidth_gb_per_s }] : [];
  });
  const shown = await page.$$eval('[data-threshold]', (ps) => ps.map((p) => ({ t: p.dataset.threshold, b: p.dataset.backend ?? '' })));
  check(devs.length ? `the threshold line shows lab 5.1's threshold (${devs.map((d) => `${d.b} ${fmt(d.t)}`).join(', ')})` : 'no lab 5.1 run, and the threshold line says what to run',
    devs.length ? devs.every((d) => shown.some((s) => s.b === d.b && s.t === fmt(d.t))) : shown.length === 1 && shown[0].t === 'none', JSON.stringify(shown));
}
{
  const sets = preferred(reports, L2.id);
  const want = [];
  for (const s of sets)
    for (const gemm of B.gemms) {
      const rows = steps(s, gemm);
      if (!rows.length) continue;
      const knee = await kneeOf(rows.map((r) => ({ batch: r.batch, seconds: secs(r) })));
      const ones = s.rows.filter((r) => r.name === 'one_by_one' && r.gemm === gemm && secs(r));
      const common = ones.map((o) => ({ o, st: rows.find((x) => x.batch === o.batch) })).filter((x) => x.st && x.o.batch > 1).sort((a, b) => b.o.batch - a.o.batch)[0];
      want.push({ backend: s.backend, gemm, knee: knee == null ? 'none' : String(knee), one: common ? fmt(secs(common.o) / secs(common.st)) : 'none' });
    }
  if (want.length) {
    const shown = await page.$$eval('table[data-table="batching-summary"] tbody tr', (trs) => trs.map((tr) => ({ backend: tr.dataset.backend, gemm: tr.dataset.gemm, knee: tr.dataset.knee, one: tr.querySelector('[data-one-by-one]')?.dataset.oneByOne })));
    check(`batching summary: knees and batched-against-one-by-one per GEMM (${want.map((w) => `${w.gemm} knee ${w.knee}, ${w.one}×`).join('; ')})`,
      JSON.stringify(shown) === JSON.stringify(want), JSON.stringify(shown));
  } else {
    check('batching summary: no rows, and it says what to run', (await page.$('[data-batching="none"]')) !== null);
  }
  await shot('summary', '[data-table="batching-summary"], [data-batching]');
  // The piece table follows the GEMM buttons.
  const gemmsWithPieces = B.gemms.filter((g) => sets.some((s) => s.rows.some((r) => r.name === 'piece' && r.gemm === g)));
  if (gemmsWithPieces.length) {
    for (const g of gemmsWithPieces) {
      await page.evaluate((g) => [...document.querySelectorAll('[data-pieces] .seg button')].find((b) => b.textContent === g.toUpperCase()).click(), g);
      await pause(80);
      const s = sets.find((x) => x.rows.some((r) => r.name === 'piece' && r.gemm === g));
      const rows = s.rows.filter((r) => r.name === 'piece' && r.gemm === g && secs(r));
      const batches = [...new Set(rows.map((r) => r.batch))].sort((a, b) => a - b);
      const lin = rows.find((r) => r.piece === 'linear' && r.batch === batches[0]);
      const shown = await page.evaluate((b) => {
        const t = document.querySelector(`table[data-table="pieces"][data-backend="${b}"]`);
        return { pick: document.querySelector('[data-pieces]').dataset.pieces, heads: [...t.querySelectorAll('thead th')].slice(1).map((x) => x.textContent), linear: t.querySelector('tr[data-piece="linear"] td:nth-child(2)').textContent };
      }, s.backend);
      check(`piece table for ${g}: batches ${batches.join(', ')}, one layer's linear ${fmt(secs(lin) * 1e3)} ms at batch ${batches[0]}`,
        shown.pick === g && JSON.stringify(shown.heads) === JSON.stringify(batches.map((b) => `batch ${b}`)) && shown.linear === fmt(secs(lin) * 1e3), JSON.stringify(shown));
    }
  } else {
    check('piece table: absent when there are no piece rows', (await page.$('[data-pieces]')) === null);
  }
}

// ---- reference solutions: hidden by default, then the real files, with computed status
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const L of [L1, L2]) {
  const P = `p[data-solution="${L.num}"] button`;
  check(`${L.num}: no reference solution is shown before it is asked for`, (await page.$$('figure.code')).length === 0);
  check(`${L.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch22/${L.dir}/solution/${f}`);
  check(`${L.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`${L.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  // LabKit's solutionTabStatus, from the shared helper (it adds this machine's CUDA front-end check).
  const cuStatus = tabStatus(reports, L.id, 'kernel.cu', frontend);
  const metalStatus = tabStatus(reports, L.id, 'kernel.metal', frontend);
  const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
  for (let i = 0; i < 3; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const [f, status] = order[i];
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
    await pause(60);
    const shown = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`${L.num}: ${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  check(`${L.num}: tier colours are applied (global and registers)`, (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  await shot(`solution-${L.num}`, `p[data-solution="${L.num}"]`);
  await page.click(P);
  await pause(80);
  check(`${L.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links and margin notes
const known = await page.evaluate(async () => {
  const reg = await import('/src/chapters/registry.ts');
  const out = [];
  for (const id of reg.CHAPTER_IDS) {
    const ch = await reg.loadChapter(id);
    out.push(`#/${id}`, ...ch.sections.map((s) => `#/${id}/${s.id}`));
  }
  return out;
});
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to 22.2 and 22.5', links.includes('#/ch22/22.2') && links.includes('#/ch22/22.5'), links.join(' '));
check('every link goes to a known route', links.every((l) => known.includes(l)), links.filter((l) => !known.includes(l)).join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph or another note', misplaced.length === 0, misplaced.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await page.goto('about:blank');
await goto(page, 'ch22/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
