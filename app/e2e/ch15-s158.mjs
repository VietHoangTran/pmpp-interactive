// Browser check for section 15.8: the barrier classification, the pipelining Gantt
// explorer, the narrated scene "Loading the next tile while computing this one", the
// stepper and outcome table of the pipelined loop, lab 15.5's listing and the
// questions. Expected values are computed by the model the page runs on
// (src/sim/gemmpipe.ts), imported directly. The scene passes with or without
// content/ch15/narration/manifest.json: with no audio the player runs on estimated
// timings and shows each script regardless.
// Usage: node e2e/ch15-s158.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import {
  HAZARD_LABEL, SCHEDULES, SCHEMES, VARIANTS, baselineBarriers, firstFailure, hazardsOf, roles, run, speedup, speedupLimit, timeline,
} from '../src/sim/gemmpipe.ts';

const out = process.argv[2];
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(50);
  return done;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
};
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(v));
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const choose = async (needle, label) => {
  const h = await q(needle);
  await h.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};

await goto(page, 'ch15/15.8');
await page.waitForSelector(bench('True or false dependence?'));

// ---------------------------------------------------------------- true or false?
const Q = bench('True or false dependence?');
const model = baselineBarriers(4);
const rows = await page.$$eval(Q + ' tr[data-barrier]', (r) => r.map((x) => x.dataset.dep));
check('seven barriers, labelled as the model has them (4 true, 3 false)', rows.length === 7 && rows.every((d, i) => d === model[i].dep), rows.join(' '));
const pick = async (i, d) => {
  await page.$eval(`${Q} tr[data-barrier="${i}"]`, (tr, d) => [...tr.querySelectorAll('button')].find((b) => b.textContent === d).click(), d);
  await pause(60);
};
await pick(1, 'true');
const fb = await text(`${Q} tr[data-barrier="1"]`);
check('a wrong label is marked, with the reason: the next load only overwrites', fb.includes('no: false') && fb.includes('only because it would overwrite A0 and B0'), fb);
for (const b of model) await pick(b.barrier, b.dep);
const score = await page.$eval(`${Q} [data-out="score"]`, (e) => [e.dataset.right, e.dataset.answered, e.textContent]);
check('all seven labelled correctly gives 7 of 7 and the summary', score[0] === '7' && score[1] === '7' && score[2].includes('7 of 7 right'), JSON.stringify(score));
if (out) await page.$eval(Q, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-quiz.png' });
const qf = await choose('which barriers guard a false dependence', 'The ones after each compute');
check('q-false: the barriers after each compute', qf.includes('Correct.'));

// ---------------------------------------------------------------- the Gantt explorer
const G = bench('Pipelining Gantt');
await page.$eval(G, (e) => e.scrollIntoView({ block: 'start' }));
/** What the explorer draws and tabulates, per scheme. */
const gantt = () =>
  page.$eval(G, (b) => ({
    drawn: [...b.querySelectorAll('svg g[data-scheme]')].map((g) => ({
      scheme: g.dataset.scheme,
      total: Number(g.dataset.total),
      boxes: [...g.querySelectorAll('rect[data-lane]')].map((r) => [r.dataset.lane, Number(r.dataset.tile), Number(r.dataset.start), Number(r.dataset.end), Number(r.dataset.buf)]),
      barriers: g.querySelectorAll('[data-barrier]').length,
    })),
    table: [...b.querySelectorAll('tbody tr[data-scheme]')].map((r) => [r.dataset.scheme, Number(r.dataset.total), r.dataset.speedup]),
    limit: b.querySelector('[data-out="limit"]').dataset.limit,
  }));
const same = (a, b) => Math.abs(a - b) < 1e-9;
const matches = (g, n, L, C, s) =>
  SCHEMES.every((sc, i) => {
    const m = timeline(sc, n, L, C, s);
    const d = g.drawn[i];
    const t = g.table[i];
    return d.scheme === sc && same(d.total, m.total) && d.boxes.length === m.boxes.length &&
      d.boxes.every((b, k) => b[0] === m.boxes[k].lane && b[1] === m.boxes[k].tile && same(b[2], m.boxes[k].start) && same(b[3], m.boxes[k].end) && b[4] === m.boxes[k].buf) &&
      d.barriers === m.barriers.length && t[0] === sc && same(t[1], m.total);
  });
let g = await gantt();
check("the book's figure by default: 8, 8 and 5 units, drawn box for box as the model has them", matches(g, 4, 1, 1, 0) && g.table.map((t) => t[1]).join() === '8,8,5' && g.table[2][2] === '1.6', JSON.stringify(g.table));
check('two barriers draw 7 barriers, double-buffered 4, pipelined 4', g.drawn.map((d) => d.barriers).join() === '7,4,4');
await page.$eval(`${G} button[data-checkpoint="slow-load"]`, (b) => b.click());
await pause(120);
g = await gantt();
check('loads 3 x compute, 8 pairs: 32 against 25, a speed-up of 1.28', matches(g, 8, 3, 1, 0) && g.table[2][1] === 25 && g.table[2][2] === String(speedup(8, 3, 1)), JSON.stringify(g.table));
check('the limit line gives (L + C) / max(L, C) = 1.33', g.limit === String(Number(speedupLimit(3, 1).toFixed(2))), g.limit);
if (out) await page.screenshot({ path: out + '-gantt-slow.png' });
await page.$eval(`${G} button[data-checkpoint="slow-compute"]`, (b) => b.click());
await pause(120);
g = await gantt();
check('compute 4 x loads: the model\'s totals again, and the limit is 1.25', matches(g, 8, 1, 4, 0) && g.limit === '1.25', JSON.stringify(g.table));
await page.$eval(`${G} button[data-checkpoint="book"]`, (b) => b.click());
await setRange(`${G} input[aria-label="barrier cost s"]`, 0.5);
await setRange(`${G} input[aria-label="tile pairs n"]`, 6);
g = await gantt();
check('with s = 0.5 and n = 6 the totals and boxes follow the model', matches(g, 6, 1, 1, 0.5), JSON.stringify(g.table));
await setRange(`${G} input[aria-label="load time L"]`, 4);
await setRange(`${G} input[aria-label="barrier cost s"]`, 0);
g = await gantt();
check('L = 4, C = 1: pipelining helps little (speed-up as the model gives)', matches(g, 6, 4, 1, 0) && g.table[2][2] === String(Number(speedup(6, 4, 1).toFixed(2))), JSON.stringify(g.table));
await page.$eval(`${G} button[data-checkpoint="book"]`, (b) => b.click());
await pause(80);
check('q-fig1513: 1.6', (await numeric('What is the speed-up of the pipelined schedule over the two-barrier one', 1.6)).includes('Correct.'));
const qu = await numeric('a load takes 3 units and a compute 1', 1.28);
check('q-unbalanced: 1.28, with 32 and 25', qu.includes('Correct.') && qu.includes('3 + 21 + 1 = 25'));

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="Loading the next tile while computing this one"]';
const N = 6;
const LABELS = ['One block alone', 'Two kinds of barrier', 'Two buffers', 'Rotate the loop', 'curr and next', 'Slow loads'];
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const res = await fetch(BASE + '/content/ch15/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch15/narration/pipeline-${n}.txt`)).text()));
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}[\]+:;]/.test(s)), scripts.find((s) => /[<>=×%/_(){}[\]+:;]/.test(s)));
const tb = timeline('two-barriers', 4, 1, 1).total, tp = timeline('pipelined', 4, 1, 1).total;
check(`script 4 tells the model's totals: 5 units instead of 8 (model: ${tp} and ${tb})`, tp === 5 && tb === 8 && scripts[3].includes('take 5 units instead of 8'));
const sb = timeline('two-barriers', 4, 3, 1).total, sp = timeline('pipelined', 4, 3, 1).total;
check(`script 6 tells the model's totals for slow loads: 13 instead of 16 (model: ${sp} and ${sb})`, sp === 13 && sb === 16 && scripts[5].includes('take 13 units instead of 16'));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['pipeline-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0]);
await (await box('captions')).click();
await pause(120);
check("switching captions on shows part 1's script", (await hasCaption()) && (await caption()) === scripts[0]);
await (await box('sound')).click();
await pause(80);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .row button.primary')][0].click(), S);
await pause(1500);
const pos = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('Play with the sound off runs the part on its own clock', pos > 0, String(pos));
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .row button.primary')][0].click(), S);
await pause(80);

const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const goPart = async (k) => {
  await page.evaluate((sel, k) => document.querySelectorAll(sel + ' .seg button')[k - 1].click(), S, k);
  await pause(120);
};
const frame = () =>
  page.$eval(S + ' .stage svg', (svg) => ({
    beat: Number(svg.dataset.beat),
    lanes: [...svg.querySelectorAll('g[data-scheme]')].map((g) => ({
      scheme: g.dataset.scheme,
      total: Number(g.dataset.total),
      boxes: [...g.querySelectorAll('rect[data-lane]')].map((r) => [r.dataset.lane, Number(r.dataset.tile), Number(r.dataset.start), Number(r.dataset.end), Number(r.dataset.buf)]),
      barriers: [...g.querySelectorAll('[data-barrier]')].map((b) => b.dataset.arrival === 'true'),
    })),
    roles: [...svg.querySelectorAll('[data-iter]')].map((r) => [Number(r.dataset.iter), Number(r.dataset.curr), Number(r.dataset.next)]),
    text: svg.textContent.replace(/\s+/g, ' '),
  }));
const lanesMatch = (d, scheme, L, C) => {
  const m = timeline(scheme, 4, L, C);
  return d.scheme === scheme && d.total === m.total && d.boxes.length === m.boxes.length &&
    d.boxes.every((b, k) => b[0] === m.boxes[k].lane && b[2] === m.boxes[k].start && b[3] === m.boxes[k].end && b[4] === m.boxes[k].buf);
};
await goPart(1);
await seek(1000);
let f = await frame();
check('part 1 draws the two-barrier loop of the model, complete at the end, half-busy lanes', f.beat === 1 && lanesMatch(f.lanes[0], 'two-barriers', 1, 1) && f.text.includes('8 units') && f.text.includes('memory busy 50%'));
await seek(300);
f = await frame();
check('part 1 halfway: only the boxes that have started are drawn', f.lanes[0].boxes.length > 0 && f.lanes[0].boxes.length < 8, String(f.lanes[0].boxes.length));
await goPart(2);
await seek(1000);
f = await frame();
check('part 2 labels the false dependences', f.beat === 2 && f.text.includes('false') && f.lanes[0].barriers.filter((a) => !a).length === 3);
await goPart(3);
await seek(1000);
f = await frame();
check('part 3 ends on the double-buffered loop: one barrier per iteration, still 8 units', f.beat === 3 && lanesMatch(f.lanes[0], 'double-buffered', 1, 1) && f.lanes[0].barriers.length === 4 && f.text.includes('still 8 units'));
await goPart(4);
await seek(1000);
f = await frame();
check('part 4 draws both loops from the model: 8 and 5 units, speed-up 1.6', f.beat === 4 && lanesMatch(f.lanes[0], 'two-barriers', 1, 1) && lanesMatch(f.lanes[1], 'pipelined', 1, 1) && f.text.includes('a speed-up of 1.6'));
if (out) await page.screenshot({ path: out + '-scene-4.png' });
await goPart(5);
await seek(1000);
f = await frame();
check('part 5: curr and next swap every iteration, as roles() gives', f.beat === 5 && f.roles.length === 3 && f.roles.every(([i, c, n]) => roles(i).curr === c && roles(i).next === n), JSON.stringify(f.roles));
check('part 5 ends on the last barrier: no tile left to load', f.text.includes('no tile is left to load'));
await seek(450);
f = await frame();
check('part 5 midway: a loop barrier that also frees a buffer', /barrier \d: tile \d has arrived, and buffer \d is free for tile \d/.test(f.text), f.text.slice(-120));
await goPart(6);
await seek(1000);
f = await frame();
check('part 6: slow loads, 16 against 13 units from the model', f.beat === 6 && lanesMatch(f.lanes[0], 'two-barriers', 3, 1) && lanesMatch(f.lanes[1], 'pipelined', 3, 1) && f.text.includes('a speed-up of ' + speedup(4, 3, 1).toFixed(2)));
check("part 6's caption is its script", (await caption()) === scripts[5]);
if (out) await page.screenshot({ path: out + '-scene-6.png' });
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await pause(500);
check('the captions choice is remembered after a reload', await (await box('captions')).evaluate((e) => e.checked));
await (await box('captions')).click();
await pause(80);

// ---------------------------------------------------------------- the stepper
const T = bench('Step through the pipelined loop');
await page.waitForSelector(T);
await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
const narr = () => text(T + ' .narrate');
const hl = () => text(T + ' .code .ln.hl');
const result = () => text(T + ' [data-k="result"]');
const toEnd = async () => {
  for (let i = 0; i < 40; i++) if (!(await clickText(T, 'Run to next barrier'))) break;
  for (let i = 0; i < 80; i++) if (!(await clickText(T, 'Step'))) break;
};
const cfg = (variant) => ({ threads: 4, tiles: 4, variant });
let tr = run(cfg('correct'), 'in-order');
await clickText(T, 'Step');
check('step 1 is thread 0 loading slot 0 of tile 0 into buffer 0, on line 1', (await narr()) === 'Thread 0, the prologue: loads slot 0 of tile 0 into buffer 0.' && (await hl()).startsWith('loadTile(buf[0], tiles, 0);'), await narr());
await clickText(T, 'Run to next barrier');
check('run to next barrier: the prologue barrier opens', (await narr()).startsWith('The barrier after the prologue opens'));
await clickText(T, 'Step');
const s1 = tr.steps.find((s) => s.op.kind === 'mm');
check(`the first mm adds up buffer 0 and finds tile 0 four times (model sum ${s1.sum})`, (await narr()).includes('adds up the 4 slots of buffer 0 and finds tiles 0, 0, 0, 0. Its sum is now 0004.') && (await hl()).includes('mm(buf[tile & 1], sum);'), await narr());
await toEnd();
check('the correct loop: every thread ends with 4444', (await result()).includes('Every thread ends with 4444.'));
check('the correct loop offers no failure to jump to', !(await clickText(T, 'Go to the first failure')));
if (out) await page.screenshot({ path: out + '-stepper-correct.png' });

await setSelect(T + ' select[aria-label="kernel"]', 'no-loop-barrier');
tr = run(cfg('no-loop-barrier'), 'in-order');
check('without the loop barrier there is a failure to jump to', await clickText(T, 'Go to the first failure'));
const v0 = tr.violations[0];
let n = await narr();
check(`the first failure is the model's: ${HAZARD_LABEL[v0.kind]} by thread ${v0.thread}`, n.includes(`Failure (${HAZARD_LABEL[v0.kind]}): ${v0.message}.`), n);
check('the bad slots are marked', (await page.$$(T + ' .mx.shared span.bad')).length >= 1);
if (out) await page.screenshot({ path: out + '-stepper-fail.png' });
await toEnd();
check('without the loop barrier, in order, the sums are wrong', (await result()).includes('end with a wrong sum'));
await setSelect(T + ' select[aria-label="thread order"]', 'lockstep');
await toEnd();
check('in lockstep the same broken loop gives 4444 everywhere', (await result()).includes('Every thread ends with 4444.'));
await setSelect(T + ' select[aria-label="thread order"]', 'in-order');
await setSelect(T + ' select[aria-label="kernel"]', 'no-epilogue');
check('a missing epilogue is not a race: nothing to jump to', !(await clickText(T, 'Go to the first failure')));
await toEnd();
const sums = await page.$$eval(T + ' tr[data-thread]', (r) => r.map((x) => Number(x.dataset.sum)));
check('a missing epilogue: every thread ends at 0444', (await result()).includes('0444, 0444, 0444, 0444') && sums.every((v) => v === 444), JSON.stringify(sums));
await setSelect(T + ' select[aria-label="kernel"]', 'single-buffer');
check('one buffer: only buf[0] is drawn', (await page.$$(T + ' .mx.shared')).length === 1);
tr = run(cfg('single-buffer'), 'in-order');
await clickText(T, 'Go to the first failure');
check('one buffer, in order: the first failure is an overwrite before the read', (await narr()).includes('Failure (overwritten before read)') && tr.violations[0].kind === 'overwritten-before-read');

// ---------------------------------------------------------------- the outcome table
const O = bench('Every thread order at once');
await page.$eval(O, (e) => e.scrollIntoView({ block: 'start' }));
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('as in Fig. 15.14')).querySelector('input').click(), O);
await clickText(O, 'Reveal the table');
check('predicting only the book\'s loop is right', (await text(O + ' [data-k="prediction"]')).startsWith('as you predicted'));
const table = await page.$$eval(O + ' tbody tr', (rs) => rs.map((r) => ({ v: r.dataset.variant, cells: [...r.querySelectorAll('td[data-outcome]')].map((c) => ({ o: c.dataset.outcome, k: c.dataset.kinds, t: c.textContent })) })));
check('every cell is the model\'s outcome and hazards',
  table.length === VARIANTS.length && table.every((row, i) => row.v === VARIANTS[i] && row.cells.every((c, k) => {
    const r = run(cfg(row.v), SCHEDULES[k]);
    return c.o === (r.correct ? 'correct' : 'wrong') && c.k === hazardsOf(r).join(' ');
  })), JSON.stringify(table.map((r) => r.cells.map((c) => c.o))));
const rowOf = (v) => table.find((r) => r.v === v).cells;
check('lockstep hides both barrier defects and the single buffer', ['no-loop-barrier', 'barrier-before-load', 'single-buffer'].every((v) => rowOf(v)[0].o === 'correct'));
check('the missing epilogue and the wrong tile are wrong everywhere, with no race', ['no-epilogue', 'wrong-tile'].every((v) => rowOf(v).every((c) => c.o === 'wrong' && c.k === '' && c.t.includes('no race'))));
check('the removed loop barrier shows both hazards in order', rowOf('no-loop-barrier')[1].k === 'read-before-write overwritten-before-read');
if (out) await page.screenshot({ path: out + '-table.png' });

// ---------------------------------------------------------------- lab 15.5's listing
const labCu = path.join(REPO, 'labs/ch15/lab5_pipelined/solution/kernel.cu');
const labMetal = path.join(REPO, 'labs/ch15/lab5_pipelined/solution/kernel.metal');
const haveLab = fs.existsSync(labCu) && fs.existsSync(labMetal);
check('lab 15.5\'s reference solution exists (kernel.cu and kernel.metal)', haveLab);
if (haveLab) {
  const fig = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('The reference solution of lab 15.5')));
  const figText = () => fig.evaluate((f) => f.querySelector('pre').textContent);
  const firstLine = fs.readFileSync(labCu, 'utf8').split('\n')[0];
  check('the CUDA tab is the lab file', (await figText()).startsWith(firstLine), firstLine);
  await fig.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal').click());
  await pause(80);
  const status = await fig.evaluate((f) => f.querySelector('.tabs .status')?.textContent ?? '');
  check('the Metal tab is the Metal file, its status in the fixed vocabulary', (await figText()).startsWith(fs.readFileSync(labMetal, 'utf8').split('\n')[0]) && /^(verified on Metal|no run on this machine|unverified: failed on this machine)$/.test(status), status);
}

// ---------------------------------------------------------------- questions
const k0 = await choose('What happens for K = 0', 'numTiles is 0, so numTiles - 1 is -1');
check('q-k0: "the loop never runs" is wrong, with the reason', k0.includes('Not quite.') && k0.includes('does not give -1'));
check('q-k0: the unsigned wrap is right', (await choose('What happens for K = 0', 'numTiles - 1 wraps to')).includes('Correct.'));
const bw = await choose('no barrier between', 'They use different buffers');
check('q-between: right, and the second job is labelled ours', bw.includes('Correct.') && bw.includes('our reading'));
check('q-warpspec: different warps, scheduled in hardware', (await choose('warp specialisation not depend', 'Different warps issue the loads')).includes('Correct.'));
const na = await choose('what fills the next buffer', 'An asynchronous copy engine');
check('q-no-async: the documented HIP intrinsic does not exist', na.includes('Not quite.') && na.includes('does not exist in the ROCm 10.2 compiler'));
check('q-no-async: ordinary loads, this software pipeline', (await choose('what fills the next buffer', 'Ordinary loads')).includes('Correct.'));
const sm = await numeric('How many bytes of shared memory per block', 17408);
check('q-smem: 17,408 bytes, and no device limit is printed in the text', sm.includes('Correct.') && sm.includes('Two pairs: 17,408') && !sm.includes('32 KB') && !sm.includes('48 KB'));
const rep = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const want = Object.entries(rep).filter(([b]) => b !== 'emu' && b !== 'race').flatMap(([b, r]) => {
  const v = Number(r.device.maxThreadgroupMemoryLength ?? r.device.sharedMemPerBlock);
  return isFinite(v) && v > 0 ? [[b, String(v)]] : [];
});
const shownLimits = await (await q('How many bytes of shared memory per block')).evaluate((e) => [...e.querySelectorAll('[data-smem-limit]')].map((x) => [x.dataset.backend ?? '', x.dataset.smemLimit]));
check(`q-smem: the per-block limit comes from the reports (${want.map((w) => w.join(' ')).join(', ') || 'none'})`,
  want.length ? JSON.stringify(shownLimits) === JSON.stringify(want) : shownLimits.length === 1 && shownLimits[0][1] === 'none', JSON.stringify(shownLimits));
check('answers went to the (intercepted) progress API', puts >= 1, String(puts));

// ---------------------------------------------------------------- the page's own claims
const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the p. 365 slip is pointed out', body.includes('Page 365 says A0 and B0 are loaded "from shared memory"'));
check('the stepper is tied to the snippet file', body.includes('labs/ch15/snippets/s158_pipeline_toy.cu'));
check('first-failure helper agrees with the page for one buffer', firstFailure(run(cfg('single-buffer'), 'in-order')) != null);

// ---------------------------------------------------------------- dark theme and narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(200);
if (out) {
  await page.$eval(G, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-gantt-dark.png' });
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) {
  await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-stepper.png' });
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-scene.png' });
}

const expected404 = (u) => /\/content\/ch15\/narration\/(manifest\.json|pipeline-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
