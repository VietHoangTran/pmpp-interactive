// Browser check for section 7.3: the prediction, the "Ideal against what the kernel asks"
// explorer (every control), the "Your kernels against their ceilings" panel, and the
// questions. Expected numbers come from src/sim/convai.ts and from this machine's reports.
// Usage: node e2e/ch07-s73.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { basicIntensity, ceiling, constIntensity, idealIntensity, multipleOfCeiling, smallestComputeBoundWidth, threshold } from '../src/sim/convai.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const close = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
const fmt = (v) => (Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(Number(v.toPrecision(3))));
const BOOK = { peak: 66900, bw: 3350 };

const E = 'Ideal against what the kernel asks';
const WIDTHS = [1, 3, 5, 7, 9, 11, 13, 15];
const setWidth = (m) =>
  page.$eval(bench(E) + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, WIDTHS.indexOf(m));
const press = async (control, label) => {
  await page.evaluate((sel, control, label) => {
    const b = [...document.querySelectorAll(`${sel} [data-control="${control}"] button`)].find((x) => x.textContent.trim().startsWith(label));
    b.click();
  }, bench(E), control, label);
  await pause(80);
};
const rows = () =>
  page.$$eval(bench(E) + ' tr[data-row]', (rs) =>
    Object.fromEntries(rs.map((r) => [r.dataset.row, { ai: Number(r.dataset.ai), regime: r.dataset.regime, cells: [...r.querySelectorAll('td')].map((c) => c.textContent.trim()) }])),
  );
const first = () => page.$eval(bench(E) + ' [data-first]', (e) => Number(e.dataset.first));
async function expectState(name, { m, bytes, dims, hw }) {
  const r = await rows();
  const thr = threshold(hw);
  const want = { basic: basicIntensity(bytes), const: constIntensity(bytes), ideal: idealIntensity(m, bytes, dims) };
  const okAi = Object.entries(want).every(([k, v]) => close(r[k].ai, v));
  const okRegime = Object.entries(want).every(([k, v]) => r[k].regime === (v >= thr ? 'compute-bound' : 'memory-bound'));
  const okCeil = Object.entries(want).every(([k, v]) => r[k].cells[3] === fmt(ceiling(v, hw)));
  check(`${name}: intensities, regimes and ceilings match the model`, okAi && okRegime && okCeil, JSON.stringify(r));
  check(`${name}: smallest compute-bound width is ${smallestComputeBoundWidth(thr, bytes, dims)}`, (await first()) === smallestComputeBoundWidth(thr, bytes, dims), String(await first()));
  return r;
}

await goto(page, 'ch07/7.3');
await page.waitForSelector(bench(E));

// ---- the prediction ----
let q = await question('how many operations does that statement perform');
check('prediction hides its answer until committed', !(await q.evaluate((e) => e.textContent)).includes('0.25 FLOP/B'));
await (await q.$('input[type=number]')).type('30.25');
await (await q.$('button.primary')).click();
await pause(150);
let t = await q.evaluate((e) => e.textContent);
check('committing reveals 0.25 FLOP/B and reacts to a guess of the ideal', t.includes('You predicted 30.25') && t.includes('0.25 FLOP/B') && t.includes('thinking of the ideal') && t.includes('121 times'), t);

// ---- the explorer, the book's example device ----
check('threshold line on the book\'s device reads 20', (await page.$eval(bench(E) + ' [data-threshold]', (e) => e.dataset.threshold)) === '20');
let r = await expectState('default 5×5, float32, book', { m: 5, bytes: 4, dims: 2, hw: BOOK });
check('basic kernel 0.25 FLOP/B, ceiling 838 GFLOP/s', r.basic.ai === 0.25 && r.basic.cells[3] === '838', JSON.stringify(r.basic));
check('constant kernel 0.5 FLOP/B', r.const.ai === 0.5);
await setWidth(3);
await pause(80);
r = await expectState('3×3', { m: 3, bytes: 4, dims: 2, hw: BOOK });
check('3×3 ideal is 2.25 and memory-bound on the book\'s device', r.ideal.ai === 2.25 && r.ideal.regime === 'memory-bound');
await setWidth(11);
await pause(80);
r = await expectState('11×11', { m: 11, bytes: 4, dims: 2, hw: BOOK });
check('11×11 ideal is 30.25 and compute-bound', r.ideal.ai === 30.25 && r.ideal.regime === 'compute-bound');
check('the gap label is m² = 121', (await text(bench(E) + ' [data-gap]')).includes('×121'));
await setWidth(9);
await pause(80);
r = await rows();
check('9×9 ideal is 20.25, just compute-bound', r.ideal.ai === 20.25 && r.ideal.regime === 'compute-bound');
await setWidth(7);
await pause(80);
check('7×7 ideal is 12.25, memory-bound', (await rows()).ideal.regime === 'memory-bound');
if (out) await page.screenshot({ path: out + '-explorer.png' });

// fp16 doubles everything
await press('bytes', '2');
await setWidth(3);
await pause(80);
r = await expectState('3×3 at 2 bytes', { m: 3, bytes: 2, dims: 2, hw: BOOK });
check('fp16 doubles: basic 0.5, constant 1, 3×3 ideal 4.5', r.basic.ai === 0.5 && r.const.ai === 1 && r.ideal.ai === 4.5);
check('fp16: smallest compute-bound width falls to 7', (await first()) === 7);
await press('bytes', '1');
r = await expectState('3×3 at 1 byte', { m: 3, bytes: 1, dims: 2, hw: BOOK });
check('int8: basic 1 FLOP/B, 3×3 ideal 9', r.basic.ai === 1 && r.ideal.ai === 9);

// 1D
await press('bytes', '4');
await press('dims', '1D');
await setWidth(9);
await pause(80);
r = await expectState('1D, 9 taps', { m: 9, bytes: 4, dims: 1, hw: BOOK });
check('1D with 9 taps: ideal 2.25 (m/4)', r.ideal.ai === 2.25);
check('1D in float32 needs 81 taps on the book\'s device', (await first()) === 81 && (await text(bench(E) + ' [data-first]')).includes('beyond this chart'));
await press('dims', '2D');

// custom hardware
await page.select(bench(E) + ' select', 'custom');
await pause(80);
await page.$$eval(bench(E) + ' input[type=number]', (els, vals) => {
  els.forEach((el, i) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(vals[i]));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
}, [1000, 100]);
await pause(100);
await setWidth(5);
await pause(80);
await expectState('custom 1000 GFLOP/s, 100 GB/s, 5×5', { m: 5, bytes: 4, dims: 2, hw: { peak: 1000, bw: 100 } });
check('custom threshold reads 10', (await page.$eval(bench(E) + ' [data-threshold]', (e) => e.dataset.threshold)) === '10');

// the reader's own device, when lab 5.1 measured one
const reports = await (await fetch(BASE + '/api/reports')).json();
const gpus = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const lab51 = (b) => reports[b]?.results.find((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution')?.bench.find((x) => x.name === 'roofline');
const measured = gpus.filter((b) => lab51(b)?.compute_gflops && lab51(b)?.bandwidth_gb_per_s);
const offered = await page.$$eval(bench(E) + ' select option', (o) => o.map((x) => x.value));
check('hardware list offers a device only when lab 5.1 measured it', offered.join() === ['book', ...measured, 'custom'].join(), offered.join());
for (const b of measured) {
  await page.select(bench(E) + ' select', b);
  await pause(100);
  const hw = { peak: lab51(b).compute_gflops, bw: lab51(b).bandwidth_gb_per_s };
  await expectState(`your ${b} device (lab 5.1)`, { m: 5, bytes: 4, dims: 2, hw });
  check(`your ${b} threshold is lab 5.1's ratio`, (await page.$eval(bench(E) + ' [data-threshold]', (e) => e.dataset.threshold)) === fmt(threshold(hw)));
}

// ---- your kernels against their ceilings ----
const Y = 'Your kernels against their ceilings';
const convRows = (b, name) =>
  (reports[b]?.results.filter((x) => x.lab === 'ch07/lab1_constconv' && x.variant === 'solution').flatMap((x) => x.bench) ?? []).filter((x) => x.name === name);
const withConv = gpus.filter((b) => convRows(b, 'basic').length || convRows(b, 'const').length);
if (!withConv.length) {
  check('with no lab 7.1 run the panel says what to run', (await text(bench(Y))).includes('python3 labs/pmpp.py run 7.1 --solution --bench'));
} else {
  for (const b of withConv) {
    const panel = bench(Y) + ` [data-backend="${b}"]`;
    const radii = [...new Set([...convRows(b, 'basic'), ...convRows(b, 'const')].map((x) => x.radius))];
    const shown = await page.$$eval(panel + ' tr[data-radius]', (rs) => rs.map((x) => Number(x.dataset.radius)));
    check(`${b}: one row per measured radius`, shown.length === radii.length, `${shown} vs ${radii}`);
    const hw = lab51(b) ? { peak: lab51(b).compute_gflops, bw: lab51(b).bandwidth_gb_per_s } : null;
    if (hw) {
      const mults = await page.$$eval(panel + ' [data-multiple]', (e) => e.map((x) => Number(x.dataset.multiple)));
      const want = radii.sort((a, c) => a - c).flatMap((rad) =>
        ['basic', 'const'].flatMap((n) => {
          const row = convRows(b, n).find((x) => x.radius === rad);
          return row ? [multipleOfCeiling(row.gflops, row.model_intensity ?? (n === 'basic' ? 0.25 : 0.5), hw)] : [];
        }),
      );
      check(`${b}: each measurement is shown as a multiple of its model ceiling`, mults.length === want.length && mults.every((v, i) => Math.abs(v - want[i]) < 0.001), `${mults} vs ${want}`);
      const ratios = radii.flatMap((rad) => {
        const a = convRows(b, 'basic').find((x) => x.radius === rad);
        const c = convRows(b, 'const').find((x) => x.radius === rad);
        return a && c ? [c.gflops / a.gflops] : [];
      });
      if (ratios.length) {
        const span = `${Math.min(...ratios).toFixed(3)}-${Math.max(...ratios).toFixed(3)}`;
        check(`${b}: measured constant-over-basic ratio is shown beside the model's 2`, (await page.$eval(panel + ' [data-speedup]', (e) => e.dataset.speedup)) === span, span);
      }
      const above = want.some((v) => v > 1.02);
      check(`${b}: a measurement above its ceiling is called out`, !above || (await text(panel)).includes('above its ceiling'));
      const chips = await page.$$eval(panel + ' [data-multiple]', (e) => e.map((x) => x.className));
      check(`${b}: the multiple-of-ceiling chips are neutral, neither fail nor ok`, chips.every((c) => c === 'chip'), chips.join(' | '));
    }
  }
  if (out) {
    await page.$eval(bench(Y), (e) => e.scrollIntoView());
    await page.screenshot({ path: out + '-yours.png' });
  }
}

// ---- chapter 5's naive kernel against its ceiling: the reader's number, or none ----
const naive = gpus.flatMap((b) => {
  const r = lab51(b);
  return r && r.naive_gflops && r.naive_model_ceiling_gflops ? [(r.naive_gflops / r.naive_model_ceiling_gflops).toFixed(3)] : [];
});
const naiveAttr = await page.$eval('[data-lab51-naive]', (e) => e.dataset.lab51Naive);
const naiveText = (await text('[data-lab51-naive]')).replace(/\s+/g, ' ');
check('the chapter 5 comparison is this machine\'s lab 5.1 ratio, or no number at all',
  naive.length ? naiveAttr === naive.join(',') && naiveText.includes('times the ceiling its 0.25 FLOP/B gave it') : naiveAttr === '' && naiveText === 'lab 5.1 measures the same thing on your device',
  `${naiveAttr} | ${naiveText}`);
check('no device-specific multiple is written into the page', !(await text('.prose')).includes('about ten times'));

// ---- questions ----
q = await question('For which square filters could it be compute-bound');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(100);
t = await q.evaluate((e) => e.textContent);
check('a wrong choice is marked wrong with its reason', t.includes('Not quite.') && t.includes('12.25'), t);
await (await q.$$('input[type=radio]'))[2].click();
await (await q.$('button.primary')).click();
await pause(100);
check('9×9 and larger is the right answer', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('leave the store of');
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(100);
check('store question: the right reason is accepted', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('1D convolution with a 9-tap filter');
await (await q.$('input[type=number]')).type('2.25');
await (await q.$('button.primary')).click();
await pause(100);
check('1D question accepts 2.25', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('2-byte values');
await (await q.$('input[type=number]')).type('2.25');
await (await q.$('button.primary')).click();
await pause(100);
check('fp16 question rejects 2.25 and gives 4.5', (await q.evaluate((e) => e.textContent)).includes('The answer is 4.5'));
await pause(600);
const last = puts[puts.length - 1] ?? {};
check('answers are written to the progress file under their ids', last.answers?.['ch07/7.3/q-regime']?.correct === true && last.answers?.['ch07/7.3/q-1d']?.correct === true && last.answers?.['ch07/7.3/q-fp16']?.correct === false && last.answers?.['ch07/7.3/p-basic']?.value === 30.25, JSON.stringify(last.answers ?? {}).slice(0, 300));

// ---- narrow screen ----
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch07/7.3');
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways page scroll at 390 px', wide <= 0, `${wide}px`);
if (out) await page.screenshot({ path: out + '-390.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
