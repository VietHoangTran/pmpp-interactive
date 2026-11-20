// Browser check for section 14.7: the "Radix width" explorer (every problem, key width and
// block size control, the book's 2-bit preset, the charts, the measured columns when lab
// 14.2 has run), the prediction and the four questions. Every number asserted is
// recomputed here from the models (src/sim/radixcost.ts, src/sim/radixlocal.ts).
// Usage: node e2e/ch14-s147.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const M = await import('../src/sim/radixlocal.ts');
const RC = await import('../src/sim/radixcost.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
// The page's number formats (src/lib/reports.ts fmt, and one decimal).
const fmt = (v, digits = 3) => (!isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString());
const dec = (v) => (Math.round(v * 10) / 10).toFixed(1);
const RW = 'Radix width';
const bench = `section.bench[aria-label="${RW}"]`;
const shot = async (name) => {
  if (!out) return;
  await page.$eval(bench, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const seg = async (group, text) => {
  const done = await page.evaluate((sel, group, text) => {
    const g = document.querySelector(sel + ` .seg[aria-label="${group}"]`);
    const b = g && [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === text);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, bench, group, text);
  await pause(150);
  return done;
};
const table = () => page.$$eval(bench + ' tr[data-r]', (rs) => rs.filter((r) => r.dataset.r !== 'ref').map((r) => ({
  r: Number(r.dataset.r),
  ...Object.fromEntries([...r.querySelectorAll('[data-col]')].map((c) => [c.dataset.col, c.textContent.trim()])),
  chosen: r.lastElementChild.textContent.trim(),
})));
const tableMatches = async (problem, bits, threads, label) => {
  const rows = RC.sweepR(problem, bits, threads);
  const t = await table();
  const bad = rows.filter((x, i) => {
    const p = t[i];
    return !p || p.r !== x.r || p.passes !== String(x.passes) || p.launches !== String(x.launches) || p.bytes !== fmt(x.traffic.perKey)
      || p.table !== x.tableEntries.toLocaleString('en-US') || p.ratio !== fmt(x.tableToKeys) || p.bucket !== fmt(x.keysPerBucket)
      || p.runs !== dec(x.local.meanRuns) || p.sectors !== dec(x.local.meanUnits);
  });
  check(label, bad.length === 0 && t.length === rows.length, JSON.stringify({ bad: bad.map((b) => b.r), page: t.find((p) => bad.some((b) => b.r === p.r)) }));
  return rows;
};

await goto(page, 'ch14/14.7');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.waitForSelector(bench);

// ---- defaults: 2^20 random keys, 32 bits, 256 threads
let rows = await tableMatches(RC.PROBLEMS[0], 32, 256, 'default table matches the model for every r (passes, launches, bytes, table, buckets, runs, sectors)');
const t0 = await table();
check('passes are ceil(32 / r): 32, 16, 11, 8, 7, 6, 5, 4, 2', t0.map((x) => x.passes).join(',') === '32,16,11,8,7,6,5,4,2');
check('three launches per pass', t0.every((x) => Number(x.launches) === 3 * Number(x.passes)));
const minR = rows.reduce((a, b) => (b.traffic.perKey < a.traffic.perKey ? b : a)).r;
check('the cheapest r in bytes is marked bold', await page.$eval(`${bench} tr[data-r="${minR}"] [data-col="bytes"] b`, () => true).catch(() => false), `r = ${minR}`);
check('runs per store climb with r up to 8', rows.slice(0, 8).every((x, i) => i === 0 || x.local.meanRuns > rows[i - 1].local.meanRuns));
check('at r = 8 the table is as long as the data, at 16 far longer', t0[7].ratio === '1' && Number(t0[8].ratio) > 100, `${t0[7].ratio} / ${t0[8].ratio}`);
check('library choices are marked: 8 for CUB OneSweep, rocPRIM and PyTorch MPS; 7 and 6 for CUB', t0[7].chosen.includes('CUB OneSweep') && t0[7].chosen.includes('PyTorch MPS') && t0[6].chosen.includes('multi-pass') && t0[5].chosen.includes('single tile'));
const ref = await page.$eval(bench + ' tr[data-r="ref"]', (r) => r.textContent);
check("section 14.5's one-bit sort is shown for reference", ref.includes(fmt(RC.directTraffic(RC.PROBLEMS[0], 32).perKey)), ref);
const charts = await page.$$eval(bench + ' [data-chart]', (cs) => cs.map((c) => c.dataset.chart));
check('three charts: bytes, table, sectors', ['bytes', 'table', 'sectors'].every((c) => charts.includes(c)), charts.join(','));
const pts = await page.$eval(bench + ' [data-chart="sectors"] [data-series="local design"]', (g) => g.querySelectorAll('circle').length);
check('the sectors chart has a point for each r', pts === RC.R_CHOICES.length, String(pts));
await shot('default');

// ---- other problems, key widths and block sizes
await seg('Problem', 'one 151,936-token row');
await tableMatches(RC.PROBLEMS[2], 32, 256, 'one softmax row: the table matches the model');
await seg('Key width', '16 bits (bf16)');
rows = await tableMatches(RC.PROBLEMS[2], 16, 256, '16-bit keys: the table matches the model');
check('16-bit keys: 2 passes at r = 8, one at 16', (await table())[7].passes === '2' && (await table())[8].passes === '1');
await seg('Threads per block', '1024');
await tableMatches(RC.PROBLEMS[2], 16, 1024, '16-bit keys, 1,024 threads: the table matches the model');
await seg('Problem', '64 such rows');
await tableMatches(RC.PROBLEMS[3], 16, 1024, '64 rows: the table matches the model');
await seg('Problem', '2²⁴ random keys');
await seg('Key width', '32 bits');
await seg('Threads per block', '128');
await tableMatches(RC.PROBLEMS[1], 32, 128, '2^24 keys, 128 threads: the table matches the model');
await shot('n24-128');

// ---- the book's 2-bit example
await seg('Preset', "the book's 2-bit example");
const local = await page.$$eval(bench + ' [data-local]', (cs) => cs.map((c) => c.textContent.trim()));
check('preset: Fig. 14.11\'s locally sorted rows', local.join(' | ') === '1100 1001 0110 0011 | 1000 0101 1010 1111 | 1001 1101 0110 1011 | 0100 0000 1010 0111', local.join(' | '));
const cells = await page.$$eval(bench + ' [data-cell]', (cs) => Object.fromEntries(cs.map((c) => [c.dataset.cell, c.querySelector('b').textContent])));
const starts = [0, 1, 2, 3].map((d) => [0, 1, 2, 3].map((b) => cells[`${d}-${b}`]).join(' '));
check('preset: Fig. 14.12\'s starts, 00 [0 1 2 2], 01 [4 5 6 8], 10 [8 9 10 11], 11 [12 13 14 15]', starts.join(' | ') === '0 1 2 2 | 4 5 6 8 | 8 9 10 11 | 12 13 14 15', starts.join(' | '));
const runs = await page.$eval(bench + ' [data-book-runs]', (e) => Number(e.dataset.bookRuns));
check('preset: a block\'s store makes 3.5 runs on average', runs === M.summarise(M.warpStores(M.localPass(M.BOOK_KEYS, 0, 2, 4), 32, 4, 32)).meanRuns && runs === 3.5);
await shot('book');
await seg('Preset', "the book's 2-bit example");
check('the preset can be closed again', (await page.$(bench + ' [data-book-example]')) === null);

// ---- measured columns (only when lab 14.2 has run here)
const reports = await (await fetch(BASE + '/api/reports')).json();
const devs = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').filter((b) => {
  const rs = reports[b].results.filter((x) => x.lab === RC.LAB2 && x.variant === 'solution').flatMap((x) => x.bench);
  return RC.R_CHOICES.some((r) => RC.measuredLocal(rs, r, 1) != null);
});
if (devs.length) {
  const heads = await page.$$eval(bench + ' thead th', (h) => h.map((x) => x.textContent.trim()));
  check('a measured column and chart per device with lab 14.2 rows', devs.every((d) => heads.includes(`${d}, ms`) && charts.includes(`measured-${d}`)), heads.join('|'));
} else {
  check('with no lab 14.2 rows the explorer says how to measure', (await page.$eval(bench + ' [data-measured="none"]', (e) => e.textContent)).includes('pmpp.py run 14.2'));
}

// ---- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which of those five')));
await (await pred.$('input[type=number]')).type('6');
await (await pred.$('button.primary')).click();
await pause(150);
const pt = await pred.evaluate((e) => e.textContent);
check(devs.length ? 'the r prediction reveals each device\'s fastest r' : 'the r prediction says how to measure', devs.length ? pt.includes('fastest at') : pt.includes('pmpp.py run 14.2'), pt);
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-passes accepts 3', (await answer('6-bit digit', RC.passes(16, 6))).includes('Correct.'));
check('q-table accepts 4,194,304', (await answer('How many entries does the bucket table have', 4194304)).includes('Correct.'));
check('q-2bit accepts 10 and rejects 8', (await answer("block 2's 10 bucket start", 8)).includes('The answer is 10'));
const six = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Why not use r = 16')));
await (await six.$$('input[type=radio]'))[1].click();
await (await six.$('button.primary')).click();
await pause(120);
check('q-sixteen: the shift distractor is marked wrong with its reason', (await six.evaluate((e) => e.textContent)).includes('A shift by 16 is as cheap as a shift by 1'));

// ---- themes and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await shot('dark');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
await shot('narrow');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
