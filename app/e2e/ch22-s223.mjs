// Browser check for section 22.3 ("Application-level considerations: Amdahl's law"): the
// prediction, the module explorer (book preset and decode preset), the many-small-kernels
// timeline, the questions, and the page at phone width. Every number asserted is computed
// from the tested models (src/sim/amdahl.ts, src/sim/tasklanes.ts); the book's 16.8 and 20
// are also pinned as literals.
// Usage: node e2e/ch22-s223.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

// The models the page runs on. Node strips the types itself; on a Node too old for that,
// take the same modules as the dev server compiles them for the browser.
const load = async (path) => {
  try {
    return await import('../' + path);
  } catch {
    const code = await (await fetch(BASE + '/' + path)).text();
    return import('data:text/javascript;charset=utf-8,' + encodeURIComponent(code));
  }
};
const a = await load('src/sim/amdahl.ts');
const t = await load('src/sim/tasklanes.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that the
// prediction is not already committed.
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const question = (needle) =>
  page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
const setRange = async (sel, value) => {
  await page.$eval(
    sel,
    (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    value,
  );
  await pause(40);
};
const clickIn = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(60);
  return done;
};
const setCheck = async (sel, on) => {
  if ((await page.$eval(sel, (e) => e.checked)) !== on) await page.click(sel);
  await pause(60);
};

// The page's own number formats (src/lib/reports.ts fmt, and the helpers at the top of S223.tsx).
const fmt = (v, d = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(d)).toString();
const times = (v) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : 'unbounded');
const pct = (v) => fmt(v * 100, 3) + '%';
const dur = (v) => fmt(v, 3) + ' d';

await goto(page, 'ch22/22.3');
check('section heading is shown', (await text('.page h1')).includes("Amdahl's law"));
const M = bench('Which modules go on the GPU');
const K = bench('Many small kernels');
for (const b of [M, K]) await page.waitForSelector(b);
check('both interactives are on the page', (await page.$$('section.bench')).length === 2);

// ------------------------------------------------------------ prediction
let q = await question('made 50 times faster on the GPU');
check('the prediction hides its answer until committed', !(await qText(q)).includes(times(a.speedup(0.9, 50))));
await (await q.$('input[type=number]')).type('10');
await (await q.$('button.primary')).click();
await pause(150);
let tx = await qText(q);
check(
  'committing 10 reveals 8.47× without overlap, 10× overlapped and the ceiling, and reacts to the guess',
  tx.includes('You predicted 10') && tx.includes('8.47×') && tx.includes(times(a.speedup(0.9, 50))) &&
    tx.includes(times(a.overlapped(0.9, 50))) && tx.includes('overlapped figure'),
  tx,
);
await pause(600);
check('the prediction is written to the progress file under its id', puts.some((p) => p.answers?.['ch22/22.3/p-overlap']?.value === 10));

// ------------------------------------------------------------ explorer 1: modules
const mOut = (name) => text(`${M} [data-readout] [data-out="${name}"]`);
const card = (i) => `${M} [data-module="${i}"]`;
const place = async (i, where) => {
  const ok = await clickIn(`${card(i)} [data-place]`, where === 'device' ? 'GPU' : 'host');
  await pause(40);
  return ok;
};
const overlapBox = `${M} input[aria-label="host and device overlap"]`;
const sliderS = (name) => `${M} input[aria-label="speed-up of ${name} on the GPU"]`;
const sliderShare = (name) => `${M} input[aria-label="share of ${name}"]`;
const model = (mods, overlap) => a.moduleSpeedup(mods, overlap);

await clickIn(`${M} [data-presets]`, "The book's time step (pp. 537–538)");
await setCheck(overlapBox, false);
let r = model(a.MD_STEP, false);
check("book preset: 16.8× without overlap (the page's 17×), as the model gives", (await mOut('overall')) === '16.8×' && (await mOut('overall')) === times(r.speedup));
check('book preset: 5% on the host, 0.95% on the GPU, 5.95% in all, ceiling 20×',
  (await mOut('host')) === '5%' && (await mOut('device')) === '0.95%' && (await mOut('total')) === '5.95%' && (await mOut('ceiling')) === '20×');
await setCheck(overlapBox, true);
check('book preset with overlap: 20×, the GPU fully hidden', (await mOut('overall')) === '20×' && (await mOut('overall')) === times(model(a.MD_STEP, true).speedup) && (await mOut('hidden')) === '0.95%');
// The timeline bars follow the model's spans.
const spanLeft = (lane, i) => page.$eval(`${M} [data-lane="${lane}"] [data-span="${i}"]`, (e) => parseFloat(e.style.left));
{
  const rr = model(a.MD_STEP, true);
  const sp = rr.spans[2];
  const want = (sp.start / rr.total) * 100;
  check('overlapped bars: vibrational and rotational forces start beside the kernel, after the neighbour list',
    Math.abs((await spanLeft('host', 3)) - want) < 1e-6 && Math.abs((await spanLeft('device', 2)) - want) < 1e-6);
}

// Toggle one module and recompute.
await place(3, 'device');
let mods = a.MD_STEP.map((m, j) => (j === 2 ? { ...m, place: 'device' } : m));
check('moving vibrational and rotational forces to the GPU (4×): the model\'s 22.5×, and overlap hides nothing',
  (await mOut('overall')) === times(model(mods, true).speedup) && (await mOut('overall')) === '22.5×' && (await mOut('hidden')) === '0%');
await setCheck(overlapBox, false);
check('the same placement without overlap gives the same figure', (await mOut('overall')) === times(model(mods, false).speedup));
check('its ceiling is 1 / 3% = 33.3×', (await mOut('ceiling')) === times(model(mods, false).ceiling) && (await mOut('ceiling')) === '33.3×');
await place(3, 'host');

// Push the big kernel from 100x to 1,000x: little gain.
await setRange(sliderS('Non-bonded forces'), a.SPEEDUPS.indexOf(1000));
mods = a.MD_STEP.map((m, j) => (j === 1 ? { ...m, s: 1000 } : m));
check('non-bonded forces at 1,000×: only 19.6×', (await mOut('overall')) === '19.6×' && (await mOut('overall')) === times(model(mods, false).speedup));
// A share slider rebalances the others.
await setRange(sliderShare('Non-bonded forces'), 99);
mods = a.rebalance(mods, 1, 0.99);
check('setting the kernel\'s share to 99% scales the others and gives the model\'s figure',
  (await text(`${card(1)} [data-out="share"]`)) === pct(mods[0].share) && (await mOut('overall')) === times(model(mods, false).speedup) &&
    (await mOut('host')) === pct(model(mods, false).host));
const host = await page.$eval(`${card(1)} input[aria-label="speed-up of Neighbour list on the GPU"]`, (e) => e.disabled);
check('a host module\'s GPU speed-up slider is disabled', host === true);
if (out) await page.screenshot({ path: out + '-modules.png' });

// The decode preset.
await clickIn(`${M} [data-presets]`, 'A decode step (ours)');
check('decode preset loads at 1×, ceiling 7.14×', (await mOut('overall')) === '1×' && (await mOut('ceiling')) === times(model(a.DECODE_STEP, false).ceiling));
check('kernel launches cannot move to the GPU', await page.$eval(`${card(4)} [data-place] button:last-child`, (e) => e.disabled));
await setRange(sliderS('Projections and attention'), a.SPEEDUPS.indexOf(10));
await setRange(sliderS('Elementwise kernels'), a.SPEEDUPS.indexOf(10));
mods = a.DECODE_STEP.map((m, j) => (j < 2 ? { ...m, s: 10 } : m));
check('kernels 10× faster: 4.42× without overlap', (await mOut('overall')) === '4.42×' && (await mOut('overall')) === times(model(mods, false).speedup));
await setCheck(overlapBox, true);
check('with overlap 6.85×, and the ceiling stays at 7.14×', (await mOut('overall')) === times(model(mods, true).speedup) && (await mOut('ceiling')) === '7.14×');
await place(3, 'device');
mods = mods.map((m, j) => (j === 2 ? { ...m, place: 'device' } : m));
check('sampling on the GPU lifts the ceiling to 12.5× and the speed-up to 10.9×',
  (await mOut('ceiling')) === '12.5×' && (await mOut('overall')) === times(model(mods, true).speedup) && (await mOut('overall')) === '10.9×');

// ------------------------------------------------------------ explorer 2: many small kernels
const kOut = (name) => text(`${K} [data-readout] [data-out="${name}"]`);
const cmp = (key) => text(`${K} [data-compare] tr[data-key="${key}"] [data-cell="total"]`);
const KN = `${K} input[aria-label="number of kernels"]`;
const KF = `${K} input[aria-label="fill fraction"]`;
const KL = `${K} input[aria-label="launch cost over kernel duration"]`;
const seg = (name) => `${K} [data-seg="${name}"]`;
const sim = (o) => t.simulate({ k: 8, f: 0.25, d: 1, L: 0.2, ...o });

check('defaults: one stream, 8 kernels at 25%, L = 0.2 d: 8.2 d, one at a time',
  (await kOut('total')) === '8.2 d' && (await kOut('total')) === dur(sim({ mode: 'stream' }).total) && (await kOut('max')) === '1');
check('the timeline draws all 8 kernels and 8 launches', (await page.$$(`${K} svg [data-kernel]`)).length === 8 && (await page.$$(`${K} svg [data-host="launch"]`)).length === 8);
await setRange(KF, t.FILLS.indexOf(0.1));
await clickIn(seg('mode'), 'A stream each');
check('a stream each at f = 10%, L = 0.2 d: launches let only 5 overlap', (await kOut('max')) === '5' && (await kOut('total')) === dur(sim({ mode: 'streams', f: 0.1 }).total));
await setRange(KL, t.LAUNCH_RATIOS.indexOf(0.05));
const r8 = sim({ mode: 'streams', f: 0.1, L: 0.05 });
check('at L = 0.05 d all 8 run at once: 1.4 d, d plus the launches', (await kOut('max')) === '8' && r8.maxConcurrent === 8 && (await kOut('total')) === dur(r8.total) && (await kOut('total')) === '1.4 d');
await clickIn(seg('queues'), `${t.HIP_DEFAULT_QUEUES} (HIP's default)`);
const r4 = sim({ mode: 'streams', f: 0.1, L: 0.05, queues: 4 });
check('8 streams on 4 hardware queues: overlap stops at 4', (await kOut('max')) === '4' && r4.maxConcurrent === 4 && (await kOut('total')) === dur(r4.total));
check('the timeline shows 4 queue rows', (await page.$$eval(`${K} svg text`, (ts) => ts.filter((x) => /^queue \d$/.test(x.textContent)).length)) === 4);
if (out) await page.screenshot({ path: out + '-queues.png' });

// Full fill: no mode with a launch per kernel gains.
await setRange(KF, t.FILLS.indexOf(1));
check('at f = 100% a stream each is no faster than one stream', (await cmp('streams')) === (await cmp('stream')));
await setRange(KF, t.FILLS.indexOf(0.25));

// Launch cost above the kernel: the graph wins.
await setRange(KL, t.LAUNCH_RATIOS.indexOf(2));
const all = t.compare({ k: 8, f: 0.25, d: 1, L: 2 });
const by = Object.fromEntries(all.map((x) => [x.key, x.total]));
check('L = 2 d: the compare table shows the model\'s totals', (await cmp('stream')) === dur(by.stream) && (await cmp('graph')) === dur(by.graph) && (await cmp('streams')) === dur(by.streams));
check('L = 2 d: the graph (10 d) beats one stream (17 d) and every other mode', by.graph < by.stream && (await cmp('graph')) === '10 d' && (await cmp('stream')) === '17 d' && all.every((x) => x.key === 'graph' || x.total > by.graph));
await clickIn(seg('mode'), 'Graph');
check('graph mode: one launch on the host lane, the model\'s total', (await page.$$(`${K} svg [data-host="graph"]`)).length === 1 && (await kOut('total')) === dur(by.graph));
await clickIn(seg('mode'), 'One stream');
check('one stream at L = 2 d leaves the device idle while the host launches', (await kOut('idle')) === dur(sim({ mode: 'stream', L: 2 }).idleWhileLaunching));
await setRange(KL, t.LAUNCH_RATIOS.indexOf(0.2));

// Metal encoder with pairs: barriers as issued, then producers first.
await clickIn(seg('deps'), 'pairs');
await clickIn(seg('mode'), 'Metal encoder');
const issued = sim({ mode: 'encoder', pairs: true });
check('Metal encoder, pairs as issued: 4 barriers, the model\'s total',
  (await kOut('barriers')) === '4' && (await kOut('total')) === dur(issued.total) && (await page.$$(`${K} svg [data-barrier]`)).length === 4);
check('the host lane ends in a commit', (await page.$$(`${K} svg [data-commit]`)).length === 1);
await clickIn(seg('order'), 'producers first');
const grouped = sim({ mode: 'encoder', pairs: true, grouped: true });
check('producers first: 1 barrier and a shorter run', (await kOut('barriers')) === '1' && (await kOut('total')) === dur(grouped.total) && grouped.total < issued.total);
await clickIn(seg('dispatch'), 'serial');
check('serial encoder: one dispatch at a time', (await kOut('max')) === '1' && (await kOut('total')) === dur(sim({ mode: 'encoder', pairs: true, concurrent: false }).total));
if (out) await page.screenshot({ path: out + '-encoder.png' });

// ------------------------------------------------------------ questions
const answerNum = async (needle, v) => {
  const qq = await question(needle);
  await (await qq.$('input[type=number]')).type(String(v));
  await (await qq.$('button.primary')).click();
  await pause(120);
  return qText(qq);
};
const answerChoice = async (needle, idx) => {
  const qq = await question(needle);
  const radios = await qq.$$('input[type=radio]');
  await radios[idx].click();
  await (await qq.$('button.primary')).click();
  await pause(120);
  return qText(qq);
};
check('q-hip-queues: 4 is correct', (await answerNum('At most how many hardware queues', 4)).includes('Correct.'));
check('q-decode-kernels: 2.5 is correct', (await answerNum('By what factor do tokens per second rise', 2.5)).includes('Correct.'));
check('q-host-share: 5 is correct', (await answerNum('largest share of the original run time', 5)).includes('Correct.'));
check('q-when-concurrent: the idle-device option is correct', (await answerChoice('When does that make the step shorter', 0)).includes('Correct.'));
check('q-default-stream: a wrong pick is marked wrong', (await answerChoice('keeps independent kernels from overlapping', 2)).includes('Not quite.'));
check('q-default-stream: the legacy default stream and the plain encoder', (await answerChoice('keeps independent kernels from overlapping', 0)).includes('Correct.'));
check('q-metal-overlap: a concurrent encoder with a barrier before C', (await answerChoice('What does Apple document for this', 0)).includes('Correct.'));
check('q-graphs: short kernels make the launch cost a larger share', (await answerChoice('Why small batches', 0)).includes('Correct.'));
await pause(600);
check('answers are written to the progress file', puts.some((p) => p.answers?.['ch22/22.3/q-hip-queues']?.correct === true && p.answers?.['ch22/22.3/q-graphs']));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();

// ------------------------------------------------------------ phone width
{
  const { browser: b2, page: p2, errors: e2 } = await open({ width: 390, height: 900 });
  await p2.setRequestInterception(true);
  p2.on('request', (r) =>
    r.url().includes('/api/progress') && r.method() === 'PUT'
      ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
      : r.continue(),
  );
  await goto(p2, 'ch22/22.3');
  const over = await p2.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check('at 390 px the page does not scroll sideways', over <= 1, `${over}px too wide`);
  const svgW = await p2.$eval(`${K} svg`, (s) => s.getBoundingClientRect().width);
  check('at 390 px the timeline fits its bench', svgW <= 390, String(svgW));
  if (out) await p2.screenshot({ path: out + '-narrow.png', fullPage: true });
  check('no console or page errors at phone width', e2.length === 0, e2.join(' | '));
  await b2.close();
}

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
