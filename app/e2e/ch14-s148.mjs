// Browser check for section 14.8: "Fig. 14.13, two ways" (one and two keys per thread on
// the book's keys), the "Keys per thread" explorer (problem, digit width, block size, the
// shared-memory limit from a device report or typed), the prediction and the four
// questions. Every number asserted is recomputed here from the models
// (src/sim/radixcost.ts, src/sim/radixlocal.ts).
// Usage: node e2e/ch14-s148.mjs [screenshot-prefix]
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
const fmt = (v, digits = 3) => (!isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString());
const dec = (v) => (Math.round(v * 10) / 10).toFixed(1);
const big = (v) => v.toLocaleString('en-US');
const kib = (b) => (b % 1024 === 0 ? `${big(b / 1024)} KiB` : `${big(b)} B`);
const FIG = 'Fig. 14.13, two ways';
const KPT = 'Keys per thread';
const bench = (t) => `section.bench[aria-label="${t}"]`;
const shot = async (t, name) => {
  if (!out) return;
  await page.$eval(bench(t), (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const seg = async (t, group, text) => {
  const done = await page.evaluate((sel, group, text) => {
    const g = document.querySelector(sel + ` .seg[aria-label="${group}"]`);
    const b = g && [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === text);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, bench(t), group, text);
  await pause(150);
  return done;
};
const row = (keys) => keys.map((k) => M.bin(k, 4)).join(' ');

await goto(page, 'ch14/14.8');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.waitForSelector(bench(FIG));

// ---- Fig. 14.13 on the book's keys
const fig = async () => page.evaluate((sel) => {
  const svg = document.querySelector(sel + ' svg[data-fig]');
  return {
    c: Number(svg.dataset.fig),
    runs: svg.dataset.runs,
    local: [...svg.querySelectorAll('[data-local] text')].map((t) => t.textContent).join(' '),
    output: [...svg.querySelectorAll('[data-out] text')].map((t) => t.textContent).join(' '),
    arrows: svg.querySelectorAll('line[marker-end]').length,
    sum: Object.fromEntries([...document.querySelectorAll(sel + ' [data-sum]')].map((e) => [e.dataset.sum, e.textContent.trim()])),
  };
}, bench(FIG));
const expectFig = (c) => {
  const p = M.localPass(M.BOOK_KEYS, 0, 2, 4, c);
  const ws = M.warpStores(p, 32, 4, 32);
  return { p, ws, runs: ws.map((w) => w.runs).join(' '), total: ws.reduce((a, w) => a + w.runs, 0) };
};
let f = await fig();
let e = expectFig(2);
check('two keys per thread: Fig. 14.13\'s locally sorted rows', f.c === 2 && f.local === '1100 1000 1001 0101 0110 1010 0011 1111 0100 0000 1001 1101 0110 1010 1011 0111', f.local);
check('two keys per thread: the output is iteration 1 of Fig. 14.10', f.output === row(M.passInput(M.BOOK_KEYS, 2, 1)), f.output);
check('two keys per thread: 2 blocks, 8 table entries, 2 keys per bucket, 4 stores of 2 runs each',
  f.sum.blocks === '2' && f.sum.table === '8' && f.sum.bucket === '2' && f.sum.stores === '4' && f.runs === e.runs && e.runs === '2 2 2 2' && f.sum.runs === String(e.total), JSON.stringify(f));
check('sixteen arrows to the output', f.arrows === 16, String(f.arrows));
await shot(FIG, 'fig-c2');
await seg(FIG, 'Keys per thread', '1 (Fig. 14.11)');
f = await fig();
e = expectFig(1);
check('one key per thread: Fig. 14.11\'s rows, 4 blocks, 16 entries, 1 key per bucket, 14 runs over 4 stores',
  f.local === row(e.p.blocks.flatMap((b) => b.sorted)) && f.sum.blocks === '4' && f.sum.table === '16' && f.sum.bucket === '1' && f.sum.stores === '4'
  && f.runs === e.runs && f.sum.runs === String(e.total) && e.total === 14, JSON.stringify(f));
await shot(FIG, 'fig-c1');

// ---- keys per thread
const tbl = () => page.$$eval(bench(KPT) + ' tr[data-c]', (rs) => rs.map((r) => ({
  c: Number(r.dataset.c),
  over: r.dataset.over === '1',
  ...Object.fromEntries([...r.querySelectorAll('[data-col]')].map((x) => [x.dataset.col, x.textContent.replace(/\s+/g, ' ').trim()])),
})));
const matches = async (problem, r, threads, limit, label) => {
  const rows = RC.sweepC(problem, 32, r, threads);
  const t = await tbl();
  const bad = rows.filter((x, i) => {
    const p = t[i];
    const over = limit != null && x.sharedBytes > limit;
    return !p || p.c !== x.c || p.blocks !== big(x.blocks) || p.table !== big(x.tableEntries) || p.bucket !== fmt(x.keysPerBucket)
      || p.runs !== dec(x.local.meanRuns) || p.sectors.replace(/\s/g, '') !== `${dec(x.local.meanUnits)}(min${dec(x.local.meanMin)})`
      || p.over !== over || !p.smem.startsWith(kib(x.sharedBytes)) || (over && !p.smem.includes(`over by ${kib(x.sharedBytes - limit)}`));
  });
  check(label, !bad.length && t.length === rows.length, JSON.stringify({ bad: bad.map((b) => b.c), page: t.find((p) => bad.some((b) => b.c === p.c)) }));
  return rows;
};
const reports = await (await fetch(BASE + '/api/reports')).json();
const limits = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) => {
  const d = reports[b].device;
  const v = Number('maxThreadgroupMemoryLength' in d ? d.maxThreadgroupMemoryLength : d.sharedMemPerBlock);
  return isFinite(v) && v > 0 ? [{ b, v }] : [];
});
const first = limits[0]?.v ?? null;
let rows = await matches(RC.PROBLEMS[1], 8, 256, first, `default (2^24 keys, r = 8, 256 threads) matches the model${first ? `, checked against ${limits[0].b}'s ${first} bytes` : ''}`);
check('coarsening shrinks the table by c and lowers sectors per store', rows.every((x, i) => i === 0 || (x.tableEntries * 2 === rows[i - 1].tableEntries && x.local.meanUnits < rows[i - 1].local.meanUnits)));
if (first) check('the limit line names the reported figure', (await page.$eval(bench(KPT) + ' [data-limit-line]', (el) => el.dataset.limitLine)) === String(first));
else check('with no report the explorer asks for a typed limit', (await page.$(bench(KPT) + ' [data-limit="none"]')) !== null);
await shot(KPT, 'kpt-default');

// typed limit
await page.$eval(bench(KPT) + ' select[aria-label="Shared memory limit"]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, 'typed');
  el.dispatchEvent(new Event('change', { bubbles: true }));
});
await pause(120);
check('choosing "a number I type" with nothing typed checks nothing', (await page.$(bench(KPT) + ' [data-limit="none"]')) !== null && (await tbl()).every((x) => !x.over));
await page.type(bench(KPT) + ' input[aria-label="Your limit in bytes"]', '20000');
await pause(150);
await matches(RC.PROBLEMS[1], 8, 256, 20000, 'a typed limit of 20,000 bytes marks exactly the rows whose tile and counts exceed it');
check('with 20,000 bytes, c = 16 and 32 are over and c = 8 fits', (await tbl()).filter((x) => x.over).map((x) => x.c).join(',') === '16,32');
await shot(KPT, 'kpt-typed');

// other settings
await seg(KPT, 'Digit width r', '4');
await seg(KPT, 'Threads per block', '512');
await seg(KPT, 'Problem', '64 such rows');
await matches(RC.PROBLEMS[3], 4, 512, 20000, '64 rows, r = 4, 512 threads: matches the model');
await seg(KPT, 'Problem', 'one 151,936-token row');
await seg(KPT, 'Digit width r', '1');
await seg(KPT, 'Threads per block', '1024');
await matches(RC.PROBLEMS[2], 1, 1024, 20000, 'one row, r = 1, 1,024 threads: matches the model');

// measured column (only when lab 14.2 has run here)
const devs = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').filter((b) => {
  const rs = reports[b].results.filter((x) => x.lab === RC.LAB2 && x.variant === 'solution').flatMap((x) => x.bench);
  return RC.C_CHOICES.some((c) => RC.measuredLocal(rs, 1, c) != null);
});
const heads = await page.$$eval(bench(KPT) + ' thead th', (h) => h.map((x) => x.textContent.trim()));
check(devs.length ? 'a measured column per device with lab 14.2 rows' : 'with no lab 14.2 rows there is no measured column, and the page says how to measure',
  devs.length ? devs.every((d) => heads.includes(`${d}, ms`)) : !heads.some((h) => h.endsWith(', ms')) && (await page.$eval(bench(KPT) + ' [data-limit-line]', (el) => el.textContent)).includes('pmpp.py run 14.2'));

// ---- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('1, 2, 4 and 8 keys per thread')));
await (await pred.$('input[type=number]')).type('4');
await (await pred.$('button.primary')).click();
await pause(150);
const pt = await pred.evaluate((el) => el.textContent);
const devs8 = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').filter((b) => {
  const rs = reports[b].results.filter((x) => x.lab === RC.LAB2 && x.variant === 'solution').flatMap((x) => x.bench);
  return [1, 2, 4, 8].some((c) => RC.measuredLocal(rs, 8, c) != null);
});
check(devs8.length ? 'the c prediction reveals each device\'s fastest c' : 'the c prediction says how to measure', devs8.length ? pt.includes('fastest at') : pt.includes('pmpp.py run 14.2'), pt);
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((el) => el.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((el) => el.textContent);
};
check('q-fig accepts 2', (await answer('How many keys per thread does that imply', 2)).includes('Correct.'));
check('q-table accepts 262,144', (await answer('sixteen keys per thread', RC.tableEntries({ rows: 1, rowLength: 2 ** 22 }, 8, 256, 16))).includes('Correct.'));
check('q-smem accepts 16,384', (await answer('tile alone need', RC.tileBytes(256, 8))).includes('Correct.'));
const waves = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((el) => el.textContent.includes('run 40 of your blocks')));
await (await waves.$$('input[type=radio]'))[0].click();
await (await waves.$('button.primary')).click();
await pause(120);
check('q-waves: the first option is right', (await waves.evaluate((el) => el.textContent)).includes('Correct.'));
check('the closing bridge computes about 594 keys per thread for a 151,936-token row', (await page.evaluate(() => document.querySelector('.page').textContent)).includes(`about ${Math.ceil(RC.VOCAB / 256)} keys`));

// ---- themes and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await shot(FIG, 'fig-dark');
await shot(KPT, 'kpt-dark');
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
await shot(FIG, 'fig-narrow');
await shot(KPT, 'kpt-narrow');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
