// Browser check for section 2.1: the read-set explorer (all eight operations, the
// blur radius, the partial sums), the decode-step sorting exercise, the narrated
// animation "Same loop, another order" and the four questions. What each one shows
// is compared with the model they run on (src/sim/depend.ts, imported directly).
// It passes with or without content/ch02/narration/manifest.json: with no audio
// the player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch02-s21.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  BINS,
  DECODE_ITEMS,
  OPS,
  OP_BODY,
  OP_NOTE,
  binCounts,
  classify,
  classifyDeps,
  combineReads,
  counts,
  defaults,
  reads,
  runSequential,
  runThreads,
  shape,
  stageItems,
  threadOrder,
  timesRead,
  underOrders,
} from '../src/sim/depend.ts';

const out = process.argv[2];
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
const num = (v) => v.toLocaleString('en-US');

await goto(page, 'ch02/2.1');
const E = '[data-explorer]';
await page.waitForSelector(E);
await page.$eval(E, (e) => e.closest('section.bench').scrollIntoView({ block: 'start' }));

// ================================================================ the explorer
const explorer = () =>
  page.evaluate((sel) => {
    const root = document.querySelector(sel);
    const o = (k) => root.querySelector(`[data-out="${k}"]`);
    const cells = {};
    const subs = {};
    for (const g of root.querySelectorAll('g[data-cell]')) {
      cells[g.dataset.cell] = g.dataset.state;
      const t = g.querySelectorAll('text');
      if (t.length > 1) subs[g.dataset.cell] = t[1].textContent;
    }
    return {
      op: root.dataset.op,
      body: o('body').textContent,
      inputs: Number(o('inputs').textContent),
      combines: o('combines') ? Number(o('combines').textContent) : null,
      outputsFirst: Number(o('outputs-first').textContent),
      outputsRed: o('outputs-first').classList.contains('t-fail'),
      bin: o('bin').dataset.bin,
      orders: o('orders').dataset.match,
      ordersText: o('orders').textContent,
      said: o('said').textContent,
      bad: o('said').classList.contains('bad'),
      cells,
      subs,
      chainArrows: root.querySelectorAll('svg path[marker-end]').length,
    };
  }, E);
const chooseOp = async (op) => {
  await page.$eval(`${E} button[data-op-choice="${op}"]`, (e) => e.click());
  await pause(60);
};
const clickCell = async (key) => {
  await page.$eval(`${E} g[data-cell="${key}"]`, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(40);
};
const setRadius = async (v) => {
  await page.$eval(`${E} input[type=range]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};

/** The state every cell should be in when output k is picked. */
function wantCells(op, k, p) {
  const s = shape(op, p);
  const r = reads(op, k, p);
  const via = new Set(r.combines.flatMap((c) => combineReads(op, c, p).indices));
  const want = {};
  s.inputs.forEach((a, ai) => {
    for (let i = 0; i < a.length; i++) {
      const isRead = r.inputs.some((x) => x.array === a.name && x.indices.includes(i));
      want[`in:${a.name}:${i}`] = isRead ? 'read' : ai === 0 && via.has(i) ? 'via' : 'idle';
    }
  });
  if (s.combine) for (let x = 0; x < s.combine.length; x++) want[`combine:${x}`] = r.combines.includes(x) ? 'needed' : 'idle';
  for (let o = 0; o < s.output.length; o++) want[`out:${o}`] = o === k ? 'picked' : r.chain.includes(o) ? 'chain' : 'idle';
  return want;
}
/** Everything the explorer shows for output k, against the model. Returns the disagreements. */
function disagreements(d, op, k, p) {
  const bad = [];
  const eq = (name, got, want) => {
    if (JSON.stringify(got) !== JSON.stringify(want)) bad.push(`${op} out ${k} ${name}: shown ${JSON.stringify(got)}, model ${JSON.stringify(want)}`);
  };
  const c = counts(op, k, p);
  const orders = underOrders(op, undefined, p);
  eq('op', d.op, op);
  eq('body', d.body, OP_BODY[op]);
  eq('cells', d.cells, wantCells(op, k, p));
  eq('inputs', d.inputs, c.inputs);
  eq('combines', d.combines, shape(op, p).combine ? c.combines : null);
  eq('outputs first', d.outputsFirst, c.outputsFirst);
  eq('outputs first in red', d.outputsRed, c.outputsFirst > 0);
  eq('bin', d.bin, classify(op));
  eq('orders', d.orders, orders.every((o) => o.matches) ? 'all' : 'some');
  eq('sentence flagged', d.bad, reads(op, k, p).chain.length > 0);
  eq('chain arrows', d.chainArrows, reads(op, k, p).chain.length);
  if (!shape(op, p).combine)
    for (const a of shape(op, p).inputs)
      for (let i = 0; i < a.length; i++) eq(`times ${a.name}[${i}] is read`, d.subs[`in:${a.name}:${i}`], String(timesRead(op, a.name, i, p)));
  return bad;
}

let radius = 1;
let chunk = 16;
const params = (op) => ({ ...defaults(op), radius, chunk });

let d = await explorer();
check('it opens on grayscale with pixel 3 picked: 3 inputs, no output needed first', d.op === 'grayscale' && d.cells['out:3'] === 'picked' && d.inputs === 3 && d.outputsFirst === 0 && d.said.includes('I[9], I[10] and I[11]'), d.said);
const labels = await page.$$eval(`${E} g[data-cell^="in:I:"] text:first-of-type`, (t) => t.slice(0, 6).map((x) => x.textContent).join(''));
check('the grayscale input is drawn interleaved, r g b r g b', labels === 'rgbrgb', labels);

const allBad = [];
let clicks = 0;
for (const op of OPS) {
  await chooseOp(op);
  const p = params(op);
  const first = Math.min(5, shape(op, p).output.length - 1);
  allBad.push(...disagreements(await explorer(), op, first, p).map((b) => 'on choosing: ' + b));
  for (let k = 0; k < shape(op, p).output.length; k++) {
    await clickCell(`out:${k}`);
    clicks++;
    allBad.push(...disagreements(await explorer(), op, k, p));
  }
}
check(`all eight operations, every output cell (${clicks} clicks): cells, counters, bin and thread orders agree with the model`, allBad.length === 0, allBad.slice(0, 3).join(' ; '));

// Blur: the radius widens the read set, and nothing else moves.
await chooseOp('blur');
await clickCell('out:8');
const blurBad = [];
const seen = [];
for (const r of [0, 1, 2, 3, 4]) {
  radius = r;
  await setRadius(r);
  d = await explorer();
  blurBad.push(...disagreements(d, 'blur', 8, params('blur')));
  seen.push(d.inputs);
  if (d.outputsFirst !== 0 || d.orders !== 'all') blurBad.push(`radius ${r}: an output is needed first`);
}
check('dragging the blur radius from 0 to 4 reads 1, 3, 5, 7, 9 inputs while "outputs needed first" stays 0', blurBad.length === 0 && seen.join() === '1,3,5,7,9', blurBad.slice(0, 2).join(' ; ') + ' ' + seen.join());
radius = 2;
await setRadius(2);
d = await explorer();
check('with radius 2 every interior input is read by 5 outputs (question q-blur-reads)', d.subs['in:in:8'] === '5' && timesRead('blur', 'in', 8, params('blur')) === 5 && d.subs['in:in:0'] === '3', JSON.stringify([d.subs['in:in:8'], d.subs['in:in:0']]));
await clickCell('out:0');
d = await explorer();
check('at the edge the read set is clipped', d.inputs === 3 && d.said.includes('in[0], in[1] and in[2]'), d.said);
if (out) await page.screenshot({ path: out + '-blur.png' });

// Mean: one output, built from partial sums that do not need each other.
await chooseOp('mean');
const meanBad = [];
for (const c of [16, 8, 4, 2]) {
  chunk = c;
  await page.$eval(`${E} button[data-chunk="${c}"]`, (e) => e.click());
  await pause(60);
  d = await explorer();
  meanBad.push(...disagreements(d, 'mean', 0, params('mean')));
  if (d.combines !== 16 / c || d.inputs !== 0) meanBad.push(`chunk ${c}: ${d.combines} combined values, ${d.inputs} inputs`);
}
check('the mean reads no input itself and needs 1, 2, 4, 8 partial sums as the chunk shrinks', meanBad.length === 0, meanBad.slice(0, 2).join(' ; '));
chunk = 4;
await page.$eval(`${E} button[data-chunk="4"]`, (e) => e.click());
await pause(60);
await clickCell('combine:2');
d = await explorer();
const cr = combineReads('mean', 2, params('mean'));
check('clicking a partial sum shows the inputs it is built from, and no other partial sum', d.cells['combine:2'] === 'picked' && d.inputs === cr.indices.length && cr.indices.every((i) => d.cells[`in:x:${i}`] === 'read') && Object.entries(d.cells).filter(([k, v]) => k.startsWith('in:') && v === 'read').length === 4 && d.said.includes('x[8] to x[11]') && d.outputsFirst === 0, d.said);
if (out) await page.screenshot({ path: out + '-mean.png' });

// Softmax: its own element plus one row sum.
await chooseOp('softmax');
await clickCell('out:11');
d = await explorer();
check('a softmax output reads 1 input itself and needs the sum of its row of 8', d.inputs === 1 && d.combines === 1 && d.cells['combine:1'] === 'needed' && d.cells['combine:0'] === 'idle' && Object.values(d.cells).filter((v) => v === 'via').length === 7 && d.bin === 'combine' && d.said.includes('built from 8 inputs'), d.said);
const softNote = await page.$eval(`${E} [data-out="body"]`, (e) => e.parentElement.textContent);
check('the softmax note says the explorer uses the plain formula and real kernels subtract the row maximum first', softNote.includes(OP_NOTE.softmax) && OP_NOTE.softmax.includes('row maximum'), softNote);

// The chain.
await chooseOp('inplace');
await clickCell('out:7');
d = await explorer();
const differ = underOrders('inplace', undefined, params('inplace')).filter((o) => !o.matches).length;
check('the in-place running sum is the one with red: output 7 needs 7 other outputs, in order', d.outputsFirst === 7 && d.outputsRed && d.bad && d.chainArrows === 7 && [0, 1, 2, 3, 4, 5, 6].every((k) => d.cells[`out:${k}`] === 'chain') && d.cells['out:8'] === 'idle' && d.said.includes('back to x[0]'), d.said);
check('and it is the one whose result depends on the order of the threads', d.orders === 'some' && d.ordersText.includes(`${differ} of 4 differ from the loop`) && differ > 0 && (await page.$eval(`${E} [data-out="orders"] .chip`, (e) => e.classList.contains('fail'))), d.ordersText);
if (out) await page.screenshot({ path: out + '-inplace.png' });
await chooseOp('neighbour');
await clickCell('out:7');
d = await explorer();
check('the out-of-place neighbour sum reads the same two neighbours and needs no output', d.inputs === 2 && d.outputsFirst === 0 && !d.bad && d.orders === 'all' && d.subs['in:x:6'] === '2', JSON.stringify([d.inputs, d.outputsFirst, d.orders]));

// Keyboard.
await page.$eval(`${E} g[data-cell="out:2"]`, (e) => e.focus());
const focused = await page.evaluate(() => document.activeElement?.getAttribute('data-cell'));
await page.keyboard.press('Enter');
await pause(60);
check('an output cell can be reached and picked from the keyboard', focused === 'out:2' && (await explorer()).cells['out:2'] === 'picked', String(focused));

// ================================================================ sort a decode step
const SORT = '[data-sort]';
const choose = async (id, bin) => {
  await page.$eval(`${SORT} [data-card="${id}"] button[data-choice="${bin}"]`, (e) => e.click());
  await pause(40);
};
const card = (id) =>
  page.$eval(`${SORT} [data-card="${id}"]`, (e) => ({
    solved: e.dataset.solved === 'true',
    feedback: e.querySelector('[data-feedback]')?.textContent ?? '',
    links: [...e.querySelectorAll('[data-feedback] a')].map((a) => a.getAttribute('href')),
    disabled: [...e.querySelectorAll('button[data-choice]')].filter((b) => b.disabled).length,
    body: e.querySelector('code').textContent,
  }));
const score = () => page.$eval(`${SORT} [data-sort-score]`, (e) => e.textContent.trim());

const cards = await page.$$eval(`${SORT} [data-card]`, (c) => c.map((x) => x.dataset.card));
check('eleven operations are offered, in the model’s order, with nothing placed', cards.join() === DECODE_ITEMS.map((i) => i.id).join() && (await score()) === '0 of 11 placed', cards.join());
check('each card has the four bins', (await page.$$eval(`${SORT} [data-card="residual"] button[data-choice]`, (b) => b.map((x) => x.dataset.choice))).join() === BINS.join());

const sortBad = [];
let placed = 0;
for (const it of DECODE_ITEMS) {
  const answer = classifyDeps(it.deps);
  const wrong = BINS.find((b) => b !== answer);
  await choose(it.id, wrong);
  let c = await card(it.id);
  if (c.solved || !c.feedback.startsWith('not this one')) sortBad.push(`${it.id}: wrong bin ${wrong} was not rejected (${c.feedback.slice(0, 60)})`);
  if (c.body !== it.body) sortBad.push(`${it.id}: body shown ${c.body}`);
  await choose(it.id, answer);
  placed++;
  c = await card(it.id);
  if (!c.solved || !c.feedback.startsWith('placed')) sortBad.push(`${it.id}: ${answer} was not accepted (${c.feedback.slice(0, 60)})`);
  if (c.disabled !== 3) sortBad.push(`${it.id}: ${c.disabled} buttons disabled after placing`);
  if ((await score()) !== `${placed} of 11 placed`) sortBad.push(`${it.id}: score ${await score()}`);
  for (const w of it.where.filter((x) => x !== 'ch02')) if (!c.links.includes('#/' + w)) sortBad.push(`${it.id}: no link to ${w}`);
  const size = it.id === 'softmax' ? 50000 : 4096;
  if (it.deps.reads(size) > 0 && !['cumprob'].includes(it.id) && !c.feedback.includes(num(it.deps.reads(size)))) sortBad.push(`${it.id}: feedback does not state ${num(it.deps.reads(size))} reads`);
  if (it.deps.combines > 0 && !c.feedback.includes(`${it.deps.combines} value`)) sortBad.push(`${it.id}: feedback does not state ${it.deps.combines} combined values`);
}
check('every item rejects a wrong bin, accepts the model’s bin, and its feedback states the model’s read set and the chapter', sortBad.length === 0, sortBad.slice(0, 3).join(' ; '));
const lin = await card('linear');
check('the linear layer is the one whose reads grow: 8,192 values at d = 4096', lin.feedback.includes('8,192') && DECODE_ITEMS.find((i) => i.id === 'linear').deps.reads(4096) === 8192, lin.feedback);
await choose('scale', 'chain');
check('a placed item cannot be moved', (await card('scale')).solved);
const tally = binCounts();
const doneText = await page.$eval(`${SORT} [data-sort-done]`, (e) => e.textContent).catch(() => '');
check('with all eleven placed the summary counts the bins from the model', doneText.includes(`${tally.elementwise} per element`) && doneText.includes(`${tally.manyreads} with many reads`) && doneText.includes(`${tally.combine} that combine first`) && doneText.includes(`${tally.chain} chains`) && tally.elementwise === 5, doneText);
const fwd = binCounts('forward');
check('the summary counts the forward pass alone, from the model, and names the loop as the loop around the step',
  doneText.includes(`Of the ${stageItems('forward').length} operations inside the forward pass, ${fwd.elementwise} are per element`) && fwd.elementwise === 5 &&
  doneText.includes('layer norm') && doneText.includes('the loop around the step'), doneText);
if (out) {
  await page.$eval(SORT, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-sorted.png' });
}
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === 'Start again').click(), SORT);
await pause(80);
check('Start again clears the exercise', (await score()) === '0 of 11 placed' && !(await page.$(`${SORT} [data-sort-done]`)) && !(await card('residual')).solved);

// ================================================================ the narrated animation
const S = 'section.scene';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const LABELS = ['Two loops', 'Loop 1 in order', 'Loop 1 backwards', 'Loop 2 in order', 'Loop 2 backwards', 'The test'];
const res = await fetch(BASE + '/content/ch02/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch02/narration/order-${n}.txt`)).text()));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const pressScene = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.trim());
/** The rows of 8 cells the stage is showing, by label, and every line of text on it. */
const stage = () =>
  page.$eval(S + ' .stage svg', (svg) => {
    const rows = {};
    for (const g of svg.querySelectorAll('g')) {
      const cells = g.querySelectorAll(':scope > g > text');
      if (cells.length !== 8) continue;
      const head = g.querySelectorAll(':scope > text');
      rows[head[0].textContent] = { cells: [...cells].map((t) => t.textContent), sub: head[1]?.textContent ?? null };
    }
    return { rows, text: [...svg.querySelectorAll('text')].map((t) => t.textContent) };
  });

await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const X = [3, 1, 7, 0, 4, 1, 6, 3];
const P8 = (op) => ({ ...defaults(op), n: 8 });
const want1 = runSequential('neighbour', { x: X }, P8('neighbour'));
const want2 = runSequential('inplace', { x: X }, P8('inplace'));
const rev = threadOrder('reversed', 8);
const rev2 = runThreads('inplace', { x: X }, rev, P8('inplace'));
const differ2 = rev2.filter((v, k) => v !== want2[k]).length;
const str = (a) => a.map(String);

check('six scripts exist, each 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(' '));
check('the scripts state the model’s numbers', scripts[3].includes('3, 4, 11') && scripts[3].includes('25') && want2.join() === '3,4,11,11,15,16,22,25' && scripts[4].includes(`first ${8 - differ2} results`) && scripts[4].includes(`other ${differ2}`), JSON.stringify([want2, differ2]));
const partLabels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', partLabels.length === 6 && LABELS.every((l, i) => partLabels[i] === `${i + 1}. ${l}`), partLabels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['order-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows the script of part 1, without audio tags', (await caption()) === scripts[0] && !(await caption()).includes('['));

let st = await stage();
check('part 1 shows the 8 numbers and both loop bodies from the model', st.rows.x.cells.join() === X.join() && st.text.some((t) => t.includes(OP_BODY.neighbour)) && st.text.some((t) => t.includes(OP_BODY.inplace)), st.text.join('|'));
await setBox('sound', false);
await pressScene('Play');
await pause(1500);
const moved = await slider();
check('with the sound off the animation runs', moved > 20, String(moved));
await pressScene('Pause');

await pressScene('2. Loop 1 in order');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
st = await stage();
check('part 2 ends with y equal to what the loop gives', st.rows.y.cells.join() === want1.join() && st.rows['what the loop gives'].cells.join() === want1.join() && st.rows.x.cells.join() === X.join() && st.text.includes('all 8 threads done: the same answer as the loop'), JSON.stringify(st.rows.y));
await pressScene('3. Loop 1 backwards');
await seek(500);
st = await stage();
const half1 = Array.from({ length: 8 }, (_, k) => (rev.slice(0, 4).includes(k) ? String(want1[k]) : ''));
check('part 3 midway: the last four threads have run first, and each already has the loop’s value', st.rows.y.cells.join() === half1.join() && st.text.some((t) => t.startsWith('threads run so far: 7 6 5 [4]')), JSON.stringify(st.rows.y.cells));
await seek(1000);
st = await stage();
check('part 3 ends with the same y, run backwards', st.rows.y.cells.join() === want1.join() && st.text.includes('all 8 threads done: the same answer as the loop'));

await pressScene('4. Loop 2 in order');
await seek(1000);
st = await stage();
check('part 4: in loop order the in-place loop gives the running total', st.rows['x in memory now'].cells.join() === want2.join() && st.rows['x at the start'].cells.join() === X.join() && st.text.includes('all 8 threads done: the same answer as the loop'), JSON.stringify(st.rows['x in memory now']));
check('part 4 says which write the last thread read', st.text.includes('thread 7 read what thread 6 wrote'), st.text.join('|'));
await pressScene('5. Loop 2 backwards');
await seek(500);
st = await stage();
const half2 = runThreads('inplace', { x: X }, rev.slice(0, 4), P8('inplace'));
check('part 5 midway: memory holds what the model gives after four threads', st.rows['x in memory now'].cells.join() === half2.join() && st.text.includes('thread 4 read x[3] before thread 3 ran'), JSON.stringify([st.rows['x in memory now'].cells, str(half2)]));
await seek(1000);
st = await stage();
check('part 5 ends with a different answer: the model’s count of wrong results', st.rows['x in memory now'].cells.join() === rev2.join() && st.text.includes(`all 8 threads done: ${differ2} of 8 results differ from the loop`) && differ2 === 6, st.text.join('|'));
if (out) await page.screenshot({ path: out + '-scene5.png' });
await pressScene('6. The test');
await seek(1000);
st = await stage();
check('part 6 sets the two backwards runs side by side', st.rows['loop 1, backwards'].cells.join() === want1.join() && st.rows['loop 2, backwards'].cells.join() === rev2.join() && st.rows['loop 2, backwards'].sub === `${differ2} of 8 differ` && st.rows['loop 2, in loop order'].cells.join() === want2.join(), JSON.stringify(st.rows));
check('and it ends on the test', st.text.includes('the test: does any thread read what another thread writes?') && (await caption()) === scripts[5]);
await setBox('captions', false);

// ================================================================ notes and questions
check('the explorer states what it leaves out, before the bench', await page.evaluate((sel) => {
  const note = [...document.querySelectorAll('.note.caveat')].find((n) => n.textContent.includes('which elements are read'));
  return !!note && !!(note.compareDocumentPosition(document.querySelector(sel)) & Node.DOCUMENT_POSITION_FOLLOWING);
}, E));
check('the mapping to a decode step is labelled as ours', (await page.$$eval('.note.beyond', (n) => n.map((x) => x.textContent))).some((t) => t.includes('The mapping to the operations of a language model is ours')));
check('there is no Today note: the section is backend-neutral and says so', (await page.$$('.note.today')).length === 0 && (await page.$eval('.prose', (e) => e.textContent)).includes('the same question on CUDA, HIP and Metal'));
check('every margin note directly follows text', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName))));

const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
let q = await question('safe to give every iteration');
await pick(q, 'The iterations read disjoint inputs');
check('q-criterion: disjoint reads is rejected as too strong', (await qText(q)).includes('far too strong'));
await pick(q, 'No iteration reads a location that another iteration writes');
check('q-criterion: the read half alone is rejected, with the m = x[i] counterexample', !(await qText(q)).includes('Correct.') && (await qText(q)).includes('Only half of it') && (await qText(q)).includes('m = x[i]'));
await pick(q, 'Each iteration writes only locations that no other iteration reads or writes');
check('q-criterion: the full read-write condition is accepted', (await qText(q)).includes('Correct.'));
q = await question('how many times is one input');
const input = await q.$('input[type=number]');
await input.type(String(timesRead('blur', 'in', 8, { ...defaults('blur'), radius: 2 })));
await (await q.$('button.primary')).click();
await pause(120);
check('q-blur-reads: the model’s count is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('5 neighbouring inputs'), (await qText(q)).slice(0, 200));
q = await question('appears global');
await pick(q, 'Each pixel can be given its own thread');
check('q-average: one shared total is rejected as a race', (await qText(q)).includes('is a race'));
await pick(q, 'Additions can be regrouped');
check('q-average: regrouping is accepted, with the rounding caution', (await qText(q)).includes('Correct.') && (await qText(q)).includes('not exactly associative'));
q = await question('Loop 1 is');
await pick(q, 'Both');
check('q-inplace: "both" is rejected', (await qText(q)).includes('Not quite.'));
await pick(q, 'Only loop 1');
check('q-inplace: only loop 1, and the explanation names the prefix sum', (await qText(q)).includes('Correct.') && (await qText(q)).includes('prefix sum') && (await q.$$eval('a', (a) => a.map((x) => x.getAttribute('href')))).includes('#/ch11'));
await pause(700); // saves are debounced
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));

// ================================================================ layout
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
check('at phone width the page itself does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), String(await page.evaluate(() => document.documentElement.scrollWidth)));
// The drawing scrolls inside its panel at this width; the picked cell must be scrolled into view.
const inView = (cell) =>
  page.evaluate((sel, cell) => {
    const g = document.querySelector(`${sel} g[data-cell="${cell}"]`);
    const panel = g.closest('.panel');
    const a = g.getBoundingClientRect();
    const b = panel.getBoundingClientRect();
    return { scrolls: panel.scrollWidth > panel.clientWidth + 1, ok: a.left >= b.left - 1 && a.right <= b.right + 1 };
  }, E, cell);
await chooseOp('grayscale');
await page.$eval(`${E} g[data-cell="out:7"]`, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(120);
let inViewNow = await inView('out:7');
check('at phone width the drawing scrolls in its panel and keeps the picked output in view', inViewNow.scrolls && inViewNow.ok, JSON.stringify(inViewNow));
await chooseOp('blur');
await page.$eval(`${E} g[data-cell="out:15"]`, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(120);
inViewNow = await inView('out:15');
check('and follows the pick to the far end of the array', inViewNow.ok, JSON.stringify(inViewNow));
if (out) {
  await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}
// The animation switches to its narrow layout: labels above the rows, drawn at a legible scale.
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pressScene('5. Loop 2 backwards');
await seek(1000);
const narrowStage = await page.$eval(S + ' .stage svg', (svg) => {
  const vb = svg.viewBox.baseVal.width;
  const scale = svg.getBoundingClientRect().width / vb;
  const cell = svg.querySelector('g > g > text');
  return { layout: svg.dataset.layout, scale, font: cell ? cell.getBoundingClientRect().height : 0, right: svg.getBoundingClientRect().right, page: window.innerWidth };
});
check('at phone width the animation uses its narrow layout at a scale of 0.75 or more, inside the page', narrowStage.layout === 'narrow' && narrowStage.scale >= 0.75 && narrowStage.right <= narrowStage.page, JSON.stringify(narrowStage));
st = await stage();
check('and draws the same numbers from the model', st.rows['x in memory now'].cells.join() === rev2.join() && st.text.includes(`all 8 threads done: ${differ2} of 8 results differ from the loop`), JSON.stringify(st.rows['x in memory now']));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-scene-narrow.png' });
}

const expected404 = (u) => /\/content\/ch02\/narration\/(manifest\.json|order-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
