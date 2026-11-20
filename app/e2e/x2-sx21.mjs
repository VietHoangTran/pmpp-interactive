// Browser check for section X2.1: the bytes-per-weight table, the decode roofline explorer
// (format, layer, activations, batch, hardware, your lab X2.1 / X2.2 diamonds), the unfused
// Predict, the "Fused or unfused" explorer, the narrated scene "Three ways to feed one GEMV"
// (five parts, captions, every drawing against qtraffic.ts), the bandwidth utilisation
// calculator, and three questions. Expected values come from src/sim/qtraffic.ts, imported
// inside the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx21.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const setSelect = async (scope, label, value) => {
  const ok = await page.evaluate((scope, label, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${label}"]`);
    if (!s) return false;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, scope, label, value);
  await pause(120);
  return ok;
};
const setInput = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const el = document.querySelector(`${scope} input[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
};
const clickButton = async (scope, text) => {
  const ok = await page.evaluate((scope, text) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === text);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, text);
  await pause(120);
  return ok;
};
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
async function answerChoice(text, index) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), index);
  await pause(60);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
async function answerNumber(text, value) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}

await goto(page, 'x2/X2.1');
await page.waitForSelector(bench('Decode GEMV roofline by weight format'));
const reports = await (await fetch(BASE + '/api/reports')).json();
const M = await page.evaluate(async (reports) => {
  const m = await import('/src/sim/qtraffic.ts');
  const L4 = m.layer('4096x4096');
  const dev = m.BOOK_DEVICE;
  const rows = m.qgemvRows(reports);
  const devs = m.measuredDevices(reports);
  return {
    bpw: Object.fromEntries(m.FORMAT_IDS.map((f) => [f, m.bytesPerWeight(f)])),
    q4: { i: m.gemv('q4_0', L4, 1, 2).intensity.toFixed(3), s: m.speedupOverF32('q4_0', L4, 1, 2, dev).toFixed(3) },
    q4b6: m.gemv('q4_0', L4, 6, 2).intensity.toFixed(3),
    q8ratio: m.byteRatio('q8_0').toFixed(3),
    knee: m.kneeBatch('q4_0', L4, 2, 20),
    kneeF32: m.kneeBatch('f32', L4, 4, 20),
    book: m.gemv('f16', m.layer('1024x4096'), 1, 2).intensity.toFixed(3),
    rel: Object.fromEntries(['f32', 'unfused-f32', 'unfused-f16', 'fused'].map((p) => [p, { q4: m.pipelineRelative(p, 'q4_0', L4, 1, 4).toFixed(3), q8: m.pipelineRelative(p, 'q8_0', L4, 1, 4).toFixed(3), bpw: m.pipelineBytesPerWeight(p, 'q4_0') }])),
    finish: Object.fromEntries(['f32', 'unfused-f32', 'fused'].map((p) => [p, m.pipelineRelative(p, 'q4_0', L4, 1, 4).toFixed(2)])),
    picks: Object.fromEntries(devs.map((d) => {
      const p = m.pickRuns(rows, d.backend, L4, 1, 'q8_0');
      const a = p.find((r) => r.format === 'q8_0');
      const b = p.find((r) => r.format === 'f32');
      return [d.backend, { n: p.length, sp: a && b ? (b.seconds / a.seconds).toFixed(3) : null }];
    })),
    pipes: Object.fromEntries(devs.map((d) => [d.backend, m.measuredPipelines(rows, d.backend, L4, 1)])),
    rows: rows.map((r) => ({ backend: r.backend, mbu: devs.find((d) => d.backend === r.backend) ? (100 * m.mbu(r.weightBytes, r.seconds, devs.find((d) => d.backend === r.backend).bw)).toFixed(1) : null })),
    devs: devs.map((d) => d.backend),
    hand: (100 * m.mbu(m.modelBytes(7e9, 'q4_0'), 1 / 30, 400)).toFixed(1),
  };
}, reports);

// ---------------------------------------------------------------- the table
const tbl = await page.$$eval('table[data-table="bpw"] tr[data-format]', (rs) => rs.map((r) => [r.dataset.format, Number(r.querySelector('[data-bpw]').textContent)]));
check('bytes-per-weight table rows match the model', tbl.length >= 10 && tbl.every(([f, b]) => b === M.bpw[f]), JSON.stringify(tbl));
check("the book's layer is stated at the model's intensity", (await page.$eval('.prose', (e) => e.textContent)).includes(`${M.book} FLOP/B`));

// ---------------------------------------------------------------- the roofline explorer
const R = bench('Decode GEMV roofline by weight format');
check('default: Q4_0, 4096 → 4096, 16-bit, batch 1, at the model intensity', (await attr(R + ' [data-intensity]', 'data-intensity')) === M.q4.i);
check('default speed-up over float32 matches the model', (await attr(R + ' [data-speedup]', 'data-speedup')) === M.q4.s);
check('knee batches: 6 for Q4_0 with 16-bit activations, 41 for float32', (await attr(R + ' [data-knee]', 'data-knee')) === String(M.knee) && (await attr(R + ' [data-knee-f32]', 'data-knee-f32')) === String(M.kneeF32));
check('two model marks on the plot: the format and float32', (await page.$$(R + ' [data-mark="model"]')).length === 1 && (await page.$$(R + ' [data-mark="ref"]')).length === 1);
check('at batch 1 the point is memory-bound', (await page.$eval(R + ' [data-panel="roofline"]', (e) => e.textContent)).includes('memory-bound'));
await setInput(R, 'Batch', 6);
check('at batch 6 the Q4_0 point crosses into compute-bound', (await attr(R + ' [data-intensity]', 'data-intensity')) === M.q4b6 && (await page.$eval(R + ' [data-panel="roofline"]', (e) => e.textContent)).includes('compute-bound'));
await setInput(R, 'Batch', 1);
await setSelect(R, 'Weight format', 'q8_0');
check('Q8_0: the byte ratio is 3.76', (await attr(R + ' [data-ratio]', 'data-ratio')) === M.q8ratio);
await setSelect(R, 'Hardware', 'custom');
await setInput(R, 'Peak', 3000);
await setInput(R, 'Bandwidth', 1000);
check('custom hardware: threshold 3 and Q8_0 at batch 1 is still memory-bound', (await page.$eval(R + ' svg', (e) => e.textContent)).includes('threshold 3') && (await page.$eval(R + ' [data-panel="roofline"]', (e) => e.textContent)).includes('memory-bound'));
if (M.devs.length) {
  const b = M.devs[0];
  await setSelect(R, 'Hardware', b);
  const n = (await page.$$(R + ' [data-mark="measured"]')).length;
  check(`your ${b} device: diamonds for this layer, format and batch and the float32 baseline (${M.picks[b].n})`, n === M.picks[b].n, String(n));
  if (M.picks[b].sp) check('your measured speed-up is shown beside the model', (await attr(R + ' [data-measured-speedup]', 'data-measured-speedup')) === M.picks[b].sp);
  await setInput(R, 'Batch', 2);
  check('at a batch your labs did not run, no diamonds and a note says which batches they ran', (await page.$$(R + ' [data-mark="measured"]')).length === 0 && !!(await page.$(R + ' [data-no-runs]')));
  await setInput(R, 'Batch', 1);
  if (out) { await page.$eval(R, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-roofline-measured.png' }); }
} else console.log('skip no lab 5.1 measurement on this machine: diamonds not checked');
await setSelect(R, 'Hardware', 'book');

// ---------------------------------------------------------------- unfused Predict and explorer
const pq = await qHandle('Dequantise Q4_0 weights to float32 in one kernel, then run lab 22.1');
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  const input = await pq.$('input[type=number]');
  await input.type('0.5');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const rev = await page.$eval('[data-reveal="unfused"]', (e) => e.textContent).catch(() => '');
check('the unfused prediction reveals about 2.14 times, slower than float32', rev.includes(`${Number(M.rel['unfused-f32'].q4).toFixed(2)} times`) && rev.includes('8.5625'), rev.slice(0, 160));

const P = bench('Fused or unfused');
await page.evaluate((P) => [...document.querySelectorAll(P + ' label.check')].forEach((l) => { const i = l.querySelector('input'); if (!i.checked) i.click(); }), P);
await pause(120);
let pipes = await page.$$eval(P + ' [data-pipeline]', (e) => e.map((x) => [x.dataset.pipeline, x.dataset.rel, Number(x.dataset.bpw)]));
check('all four pipelines show the model bytes and times (Q4_0)', pipes.length === 4 && pipes.every(([p, r, b]) => r === M.rel[p].q4 && b === M.rel[p].bpw), JSON.stringify(pipes));
await setSelect(P, 'Quantised format', 'q8_0');
pipes = await page.$$eval(P + ' [data-pipeline]', (e) => e.map((x) => [x.dataset.pipeline, x.dataset.rel]));
check('switching to Q8_0 recomputes every pipeline', pipes.every(([p, r]) => r === M.rel[p].q8), JSON.stringify(pipes));
await setSelect(P, 'Quantised format', 'q4_0');
const mp = Object.entries(M.pipes).filter(([, v]) => v);
if (mp.length) {
  const shown = await page.$$eval(P + ' [data-measured]', (e) => e.map((x) => [x.closest('[data-pipeline]').dataset.pipeline, x.dataset.measured, x.dataset.value]));
  const want = mp.flatMap(([b, v]) => Object.entries(v).map(([p, r]) => [p, b, r.toFixed(3)]));
  check("your lab X2.2 times appear beside the model's (Q4_0)", want.every((w) => shown.some((x) => JSON.stringify(x) === JSON.stringify(w))), JSON.stringify(shown));
} else console.log('skip no lab X2.2 runs on this machine: measured pipelines not checked');
if (out) { await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-pipelines.png' }); }

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="Three ways to feed one GEMV"]';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const mres = await fetch(BASE + '/content/x2/narration/manifest.json');
const manifest = mres.ok ? await mres.json() : null;
const scripts = [];
for (let n = 1; n <= 5; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x2/narration/three-traffics-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('every script is 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('no symbols read aloud in the scripts', scripts.every((s) => !/[%×→=<>_/*+]/.test(s)), scripts.find((s) => /[%×→=<>_/*+]/.test(s)) ?? '');
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const labels = await page.$$eval(S + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()));
check('five parts with the planned labels', JSON.stringify(labels) === JSON.stringify(['Float32', 'Q4_0 is short', 'Unfused', 'Fused', 'Side by side']), JSON.stringify(labels));
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('three-traffics-'));
const capOn = () => page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
if (anyAudio) check('captions are off by default', !(await capOn()) && (await caption()) === null);
else check("with no audio yet, part 1's script is shown regardless, and captions are an option", (await caption()) === scripts[0] && !(await capOn()), String(await caption()));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const part = async (label) => {
  await page.evaluate((S, label) => [...document.querySelectorAll(S + ' .seg button')].find((b) => b.textContent.includes(label)).click(), S, label);
  await pause(150);
};
const finishes = () => page.$$eval(S + ' [data-pipeline]', (e) => e.map((x) => [x.dataset.pipeline, x.dataset.finish]));
const svgText = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));

await part('1. Float32');
await seek(1000);
check('part 1: the float32 stream finishes at 1.00', JSON.stringify(await finishes()) === JSON.stringify([['f32', M.finish.f32]]) && (await svgText()).includes('done at 1.00'));
if (out) { await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-scene1.png' }); }
await part('2. Q4_0 is short');
await seek(1000);
const f2 = Object.fromEntries(await finishes());
check('part 2: the Q4_0 stream is the model length, about a seventh', f2.fused === M.finish.fused && (await svgText()).includes('about 1/7'), JSON.stringify(f2));
await part('3. Unfused');
await seek(500);
const mid = await svgText();
check('part 3, midway: the float32 copy is crossing the pipe', /kernel [12]: (write a float32 copy|read the float32 copy back)/.test(mid), mid.slice(0, 200));
await seek(1000);
check('part 3: the unfused path ends at the model time, past the float32 finish line',
  (await attr(S + ' [data-clock]', 'data-clock')) === M.finish['unfused-f32'] && (await svgText()).includes('past the float32 finish line'));
if (out) await page.screenshot({ path: out + '-scene3.png' });
await part('4. Fused');
await seek(1000);
check('part 4: unpacking sits inside the kernel and the fused stream ends at the model time', !!(await page.$(S + ' [data-unpack]')) && Object.fromEntries(await finishes()).fused === M.finish.fused);
await part('5. Side by side');
await seek(1000);
const t5 = await svgText();
check('part 5: three finishing times labelled with the model ratios', t5.includes(`${M.finish['unfused-f32']} × float32`) && t5.includes(`${(1 / Number(M.rel.fused.q4)).toFixed(2)} × faster`), t5.slice(0, 260));
if (out) await page.screenshot({ path: out + '-scene5.png' });
await part('2. Q4_0 is short');
if (!(await capOn())) await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), S);
await pause(100);
check('the caption follows the part', (await caption()) === scripts[1]);
await page.evaluate((S) => { const i = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'); if (i.checked) i.click(); }, S);
await page.$eval(S + ' button.primary', (b) => b.click());
await pause(1200);
const pos = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('it plays with the sound off: the position advances', pos > 10, String(pos));
await page.$eval(S + ' button.primary', (b) => b.click());

// ---------------------------------------------------------------- utilisation
const U = bench('Your bandwidth utilisation');
if (M.rows.length) {
  const v = await page.$eval(U + ' [data-mbu-run]', (e) => e.textContent);
  const want = M.rows[0].mbu;
  check('your first run: utilisation from its weight bytes, time and lab 5.1 bandwidth', want ? v.startsWith(want + ' %') : v.includes('run lab 5.1'), `${v} vs ${want}`);
} else check('with no runs it names the commands that produce them', (await page.$eval(U + ' [data-empty="runs"]', (e) => e.textContent)).includes('pmpp.py run X2.1 --bench'));
await setInput(U, 'Weights, billions', 7);
await setSelect(U, 'Format', 'q4_0');
await setInput(U, 'Tokens per second', 30);
await setInput(U, 'Peak bandwidth', 400);
check('by hand: 7 billion Q4_0 weights at 30 tokens per second on 400 GB/s is 29.5 %', (await attr(U + ' [data-mbu-hand]', 'data-mbu-hand')) === M.hand, await attr(U + ' [data-mbu-hand]', 'data-mbu-hand'));

// ---------------------------------------------------------------- questions
check('q-knee-batch accepts 6', (await answerNumber('smallest batch at which it reaches 20 FLOP/B', 6)).includes('Correct.'));
check('q-mbu accepts 29.5', (await answerNumber('model bandwidth utilisation, in percent', 29.5)).includes('Correct.'));
check('q-unfused: the float32 GEMV alone is faster', (await answerChoice('Dequantise Q4_0 weights to float32 in one kernel, then run the float32 GEMV, against', 0)).includes('Correct.'));
check('q-q8-ceiling: a wrong answer explains itself', (await answerChoice('Why is its speed-up ceiling over float32 3.76', 1)).includes('Not quite.'));

const expected404 = (u) => /\/content\/x2\/narration\/(manifest\.json|three-traffics-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
