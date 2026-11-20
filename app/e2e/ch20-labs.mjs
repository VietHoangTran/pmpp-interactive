// Browser check for the chapter 20 labs page: commands, the race-backend table, the status
// table of the reference solutions, the reader's own status per lab, the lab 20.3 question,
// five predictions against this machine's measurements, every results table, the lab 20.5
// table beside Eq. 20.13, the milestone line and the reference-solution toggles. Everything
// expected is computed here from /api/reports and /api/lab-file, so the check holds on any
// machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch20-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that no
// prediction is already committed whatever the reader has done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') puts.push(r.postData() ?? '');
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
  { num: '20.1', dir: 'lab1_causal_softmax' },
  { num: '20.2', dir: 'lab2_flash' },
  { num: '20.3', dir: 'lab3_flash_hunt' },
  { num: '20.4', dir: 'lab4_decode' },
  { num: '20.5', dir: 'lab5_gqa_decode' },
].map((l) => ({ ...l, id: `ch20/${l.dir}` }));
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const num = (v) => (typeof v === 'number' && isFinite(v) ? v : undefined);
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
const allSets = (reports, id) =>
  real(reports).flatMap((b) => [...(labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : []), ...benchRows(reports[b], id, 'solution')]);
const row = (set, name, where = {}) => set?.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => num(r[k]) === v));
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};

await goto(page, 'ch20/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));

// ---- structure and commands
const text0 = await body();
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
check('one heading per lab, in order', ['Lab 20.1', 'Lab 20.2', 'Lab 20.3', 'Lab 20.4', 'Lab 20.5'].every((t, i, a) => h2.findIndex((h) => h.startsWith(t)) > (i ? h2.findIndex((h) => h.startsWith(a[i - 1])) : -1)), JSON.stringify(h2));
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 20.4', 'python3 labs/pmpp.py run 20.4 --bench', 'python3 labs/pmpp.py run 20.4 --backend emu',
  'python3 labs/pmpp.py run 20.4 --solution', 'python3 labs/pmpp.py reset 20.4 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat is stated', text0.includes('front-end-checked') && text0.includes('has never been compiled'));
check('the page says its numbers are measurements', text0.includes('None is an expected value'));
check('nothing on the page says it is being written', !/being written/i.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- race table
if (reports.race) {
  const rt = await page.$$eval('[data-race-lab]', (trs) => trs.map((t) => ({ lab: t.dataset.raceLab, sol: t.dataset.raceSolution, work: t.dataset.raceWork })));
  const want = LABS.map((l) => ({ lab: l.num, sol: labStatus(reports.race, l.id, 'solution'), work: labStatus(reports.race, l.id, 'work') }));
  check(`race table: ${want.map((w) => `${w.lab} ${w.sol}/${w.work}`).join('; ')}`, JSON.stringify(rt) === JSON.stringify(want), JSON.stringify(rt));
} else check('no race run: the page says so', (await page.$('[data-race="none"]')) !== null);

// ---- status table, cell by cell
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 5 labs x 4 backends', cells.length === 20, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`,
    mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]), JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the reader's own work
for (const l of LABS) {
  const r = await page.$$eval(`[data-work="${l.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return r[b].includes('no run on this machine');
    const rs = results(run, l.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, l.id, 'work');
    if (rs.some((x) => !x.built)) return r[b].includes('does not build');
    if (s === 'none') return r[b].includes('not started');
    return r[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(r));
}

// ---- lab 20.3's question
const qh = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes('Lab 20.3, defect 1')));
const pick = async (label) => {
  await qh.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o)).querySelector('input').click(), label);
  await pause(60);
  await (await qh.$('button.primary')).click();
  await pause(120);
  return qh.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? ''));
};
let v = await pick('emu: its fixed thread orders');
check('q-detector: emu is wrong, and the reason says a shuffle is where the whole warp meets', v.includes('Not quite.') && v.includes('whole warp meets'), v);
v = await pick('race: ThreadSanitizer');
check('q-detector: race is correct, citing facts A3', v.includes('Correct.') && v.includes('A3'), v);

// ---- predictions
const expected = {
  share: preferred(reports, LABS[0].id).flatMap((s) => {
    const r = row(s, 'unfused_attention', { n: 4096 });
    return num(r?.softmax_share) != null && num(r?.softmax_seconds) != null && num(r?.total_seconds) != null ? [{ backend: s.backend, value: 100 * r.softmax_share }] : [];
  }),
  ratio: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = row(s, 'attention', { n: 4096, b_c: 32 }) ?? row(s, 'attention', { n: 4096 });
    return num(r?.flash_seconds) && num(r?.unfused_seconds) ? [{ backend: s.backend, value: r.flash_seconds / r.unfused_seconds }] : [];
  }),
  ms: preferred(reports, LABS[3].id).flatMap((s) => {
    const r = row(s, 'decode', { batch: 1, len: 8192 });
    return num(r?.ms) != null && num(r?.gb_per_s) != null ? [{ backend: s.backend, value: r.ms }] : [];
  }),
  batch8: preferred(reports, LABS[3].id).flatMap((s) => {
    const a = num(row(s, 'decode', { batch: 1, len: 8192 })?.ms);
    const b = num(row(s, 'decode', { batch: 8, len: 8192 })?.ms);
    return a != null && b != null ? [{ backend: s.backend, value: b / a }] : [];
  }),
  speedup: preferred(reports, LABS[4].id).flatMap((s) => {
    const g = num(row(s, 'grouped', { batch: 1, h_kv: 8 })?.ms);
    const p = num(row(s, 'perhead', { batch: 1, h_kv: 8 })?.ms);
    return g != null && p != null ? [{ backend: s.backend, value: p / g }] : [];
  }),
};
const bandwidth = (b) => {
  const run = reports[b];
  const rows = [...benchRows(run, 'ch05/lab1_roofline', 'work'), ...benchRows(run, 'ch05/lab1_roofline', 'solution')];
  return num(rows.find((r) => r.name === 'roofline')?.bandwidth_gb_per_s);
};
const predictions = [
  { key: 'share', lab: '20.1', find: 'what percentage of the three-stage', guess: 20 },
  { key: 'ratio', lab: '20.2', find: 'your flash time divided by', guess: 0.5 },
  { key: 'ms', lab: '20.4', find: 'the time in milliseconds of', guess: 1 },
  { key: 'batch8', lab: '20.4', find: 'batch 8 as a multiple', guess: 8 },
  { key: 'speedup', lab: '20.5', find: 'how many times faster will your grouped', guess: 4 },
];
for (const p of predictions) {
  const el = (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find)).asElement();
  if (!el) { check(`lab ${p.lab} (${p.key}): prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  check(`lab ${p.lab} (${p.key}): nothing is revealed before a prediction is committed`, input !== null && (await el.$('[data-measured]')) === null);
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
    check(`lab ${p.lab} (${p.key}): measured ${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')} shown and compared with ${p.guess}`, ok && shown.length === want.length, JSON.stringify(shown));
    check(`  lab ${p.lab} (${p.key}): the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
    if (p.key === 'ms')
      for (const w of want) {
        const bw = bandwidth(w.backend);
        const t = shown.find((x) => x.backend === w.backend)?.text ?? '';
        check(`  lab 20.4: the reveal sets the GB/s beside lab 5.1's ${bw ? fmt(bw) + ' GB/s' : 'missing bandwidth'}`,
          bw ? t.includes(`lab 5.1's ${fmt(bw)} GB/s`) : t.includes('no lab 5.1 bandwidth'), t);
      }
  } else {
    check(`lab ${p.lab} (${p.key}): no measurement, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet'), JSON.stringify(shown));
  }
  check(`  lab ${p.lab} (${p.key}): the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}

// ---- results tables
const tables = [
  { lab: LABS[0], keep: (r) => r.name === 'unfused_attention', cell: (r) => `${fmt(100 * r.softmax_share)}%` },
  { lab: LABS[1], keep: (r) => r.name === 'attention', cell: (r) => fmt(r.flash_over_unfused) },
  { lab: LABS[3], keep: (r) => r.name === 'decode' || r.name === 'append', cell: (r) => fmt(r.seconds * 1000) },
  { lab: LABS[4], keep: (r) => r.name === 'perhead' || r.name === 'grouped', cell: (r) => fmt(r.seconds * 1000) },
];
for (const t of tables) {
  const rows = allSets(reports, t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => shown[i].includes(t.cell(r))), JSON.stringify(shown.slice(0, 2)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
  }
}
check('lab 20.3 says it has no benchmark', text0.includes('This lab has no benchmark: lab 20.2 times the same kernel.'));

// ---- lab 20.5 beside Eq. 20.13
const gq = await page.$$eval('[data-gqa-row]', (trs) => trs.map((t) => [...t.children].map((td) => td.textContent.trim())));
const gqWant = preferred(reports, LABS[4].id).flatMap((s) => s.rows.filter((r) => r.name === 'grouped' && num(r.speedup_vs_perhead) != null).map((r) => ({ s, r })));
if (gqWant.length)
  check(`lab 20.5: the intensity table has ${gqWant.length} rows, each with g_q : 1 and the measured ratio`,
    gq.length === gqWant.length && gqWant.every(({ r }, i) => gq[i][4] === `${r.g_q} : 1` && gq[i][5] === fmt(r.speedup_vs_perhead) && gq[i][3].endsWith(`: ${r.blocks}`)), JSON.stringify(gq));
else check('lab 20.5: no benchmark, and the intensity table says what to run', (await page.$('[data-gqa="none"]')) !== null);

// ---- milestones
const ms = await page.$eval('[data-milestone]', (e) => e.dataset.milestone);
const bs = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const passes = (id) => bs.filter((b) => labStatus(reports[b], id, 'work') === 'pass');
const m6 = passes(LABS[0].id).filter((b) => passes(LABS[1].id).includes(b));
const m7 = passes(LABS[3].id);
check(`milestones M6 (${m6.join(',') || 'none'}) and M7 (${m7.join(',') || 'none'}) are read from the reports`, ms === `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`, ms);

// ---- reference solutions
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch20/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
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
  if (l.num === '20.2') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 20.2: tier colours are applied to the real file', (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
    await shot('solution-lab2', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch20/20.3', '#/ch20/20.4', '#/ch20/20.5', '#/ch20/20.6', '#/ch20/20.7', '#/ch05/labs']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch20/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="solution-status"]');

// ---- what was saved (the progress file is written 400 ms after the last change)
await pause(600);
check('the lab 20.3 answer was saved through PUT /api/progress', puts.some((p) => p.includes('ch20/labs/q-detector')));
check('the five predictions were saved through PUT /api/progress',
  ['p-lab1-share', 'p-lab2-ratio', 'p-lab4-ms', 'p-lab4-batch8', 'p-lab5-speedup'].every((id) => puts.some((x) => x.includes('ch20/labs/' + id))));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
