// Browser check for chapter 1, "Start here": the questions, the prediction and the
// "Spend a power budget" explorer. Everything the explorer shows is compared with
// the model it runs on (src/sim/designs.ts, imported here directly).
// Usage: node e2e/ch01-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { design, finish, race, regions, sweep } from '../src/sim/designs.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and keep what would have been saved.
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    saved.push(r.postData() ?? '');
    r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  } else r.continue();
});
// The narration manifest for this chapter may not exist yet; that 404 is expected.
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const nf = (v) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });
const B = 'section.bench[aria-label="Spend a power budget"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.trim());
const outText = (name) => text(`${B} [data-out="${name}"]`);
const stat = (name) => text(`${B} [data-stat="${name}"]`);
const setInput = async (label, value) => {
  await page.$eval(`${B} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const setChains = (n) => setInput('Independent chains (exact)', n);
const setLength = (n) => setInput('Operations in a chain', n);
const pickDesign = async (which, label) => {
  await page.evaluate((B, which, label) => {
    const group = document.querySelector(`${B} [role=group][aria-label="${which}"]`);
    [...group.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click();
  }, B, which, label);
  await pause(80);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const choose = async (q, i) => {
  await (await q.$$('input[type=radio]'))[i].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const verdictFor = (r) =>
  r.winner === 'tie' ? `A tie: both finish at ${nf(r.a.last)}.` : `Design ${r.winner.toUpperCase()} finishes first, ${nf(r.factor)} times sooner.`;
const describe = (d) => `${nf(d.units)} unit${d.units === 1 ? '' : 's'}, speed ${d.speed}`;

const BUDGET = 64;
let a = design(BUDGET, 2);
let b = design(BUDGET, 0);

await goto(page, 'ch01/start');
await page.waitForSelector(B);

// ---- the page itself
const body = await text('.prose');
check('the book\'s three H100 figures are shown as the book\'s example', body.includes("The book's example") && body.includes('34, 67 and 1,979 TFLOPS'));
check('the cost rule is labelled as the book\'s argument and unverified', body.includes("the book's argument") && body.includes('unverified'));
check('the redrawn figure names both designs', (await text('figure')).includes('Latency-oriented') && (await text('figure')).includes('Throughput-oriented'));
check('the figure draws 4 and 64 arithmetic units', (await page.$$('figure svg rect[rx]')).length === a.units + b.units, String((await page.$$('figure svg rect[rx]')).length));
check('HIP and Metal are placed in a note marked as not in the book', (await text('.note.beyond')).includes('HIP is CUDA\'s programming model on AMD GPUs') && (await text('.note.beyond')).includes('share system memory'));
check('the model\'s simplifications are stated', (await text('.note.caveat')).includes('our extrapolation') && (await text('.note.caveat')).includes('not a device'));

// ---- calibration question and the spec-sheet question
let q = await question('batches of 32');
check('the calibration question is labelled as prior knowledge', (await q.evaluate((e) => e.querySelector('.kind').textContent)) === 'From what you already know');
let t = await choose(q, 2);
check('batching: "same time, 32 times the rate" is rejected with its reason', t.includes('Not quite.') && t.includes('not free'), t.slice(-300));
t = await choose(q, 0);
check('batching: longer wait and more requests per second is accepted', t.includes('Correct.') && t.includes('§22.5'));

q = await question('step of about 30 times');
t = await choose(q, 0);
check('16-bit: "halving the precision again" is rejected with the 2x pattern', t.includes('Not quite.') && t.includes('134'), t.slice(-300));
t = await choose(q, 1);
check('16-bit: a different unit plus sparsity is accepted', t.includes('Correct.') && t.includes('With sparsity'));

// ---- prediction
const P = await question('How many times sooner does B finish?');
const crowd = race(a, b, 1024, 100);
check('the prediction states the two designs from the model', (await P.evaluate((e) => e.textContent)).includes(`Design A has ${a.units} units, each ${a.speed} times faster`));
check('nothing is revealed before a prediction is committed', !(await P.evaluate((e) => e.textContent)).includes('waves'));
await (await P.$('input[type=number]')).type('16');
await (await P.$('button.primary')).click();
await pause(150);
t = await P.evaluate((e) => e.textContent);
check('the reveal gives the model\'s factor', t.includes('You predicted 16 times sooner.') && t.includes(`${nf(crowd.factor)} times.`) && crowd.factor === 4, t);
check('the reveal counts the waves of both designs', t.includes(`take ${nf(crowd.b.waves)} waves`) && t.includes(`needs ${nf(crowd.a.waves)} waves`), t);
check('the reveal answers the guess of 16', t.includes('ratio of the unit counts'), t);
check('the reveal sends the reader to the explorer', t.includes('Set the explorer below to 1,024 chains'));

// ---- explorer: the default view is one chain
const pressed = (which) => page.$eval(`${B} [role=group][aria-label="${which}"] button[aria-pressed=true]`, (e) => e.textContent.trim());
check('default designs are 4 fast units against 64 plain ones', (await pressed('Design A')) === describe(a) && (await pressed('Design B')) === describe(b), `${await pressed('Design A')} | ${await pressed('Design B')}`);
let r = race(a, b, 1, 100);
check('one chain: the fast design wins by its speed factor', (await outText('verdict')) === verdictFor(r) && r.winner === 'a' && r.factor === 4, await outText('verdict'));
check('one chain: first and last results match the model', (await stat('a-first')) === nf(r.a.first) && (await stat('a-last')) === nf(r.a.last) && (await stat('b-first')) === nf(r.b.first) && (await stat('b-last')) === nf(r.b.last),
  [await stat('a-first'), await stat('a-last'), await stat('b-first'), await stat('b-last')].join(' '));
check('peak rates are 16 and 64', (await stat('a-peak')) === nf(a.throughput) && (await stat('b-peak')) === nf(b.throughput) && a.throughput === 16 && b.throughput === 64);
check('one chain: delivered rates are 4 and 1', (await stat('a-delivered')) === nf(r.a.delivered) && (await stat('b-delivered')) === nf(r.b.delivered), `${await stat('a-delivered')} ${await stat('b-delivered')}`);
check('one chain: 63 of the 64 plain units are never used', (await outText('b-used')) === `1 unit working, ${r.b.idle} never used. 1 wave.` && r.b.idle === 63, await outText('b-used'));
const units = (k, busyOnly) => page.$$eval(`${B} [data-design="${k}"] svg[aria-label^="Floor plan"] rect[rx]`, (rs, busyOnly) => rs.filter((x) => !busyOnly || x.getAttribute('fill') === 'var(--ink-2)').length, busyOnly);
check('floor plans draw 4 and 64 units, one of each working', (await units('a')) === 4 && (await units('b')) === 64 && (await units('a', true)) === 1 && (await units('b', true)) === 1);
const segments = (k) => page.$$eval(`${B} [data-design="${k}"] svg[aria-label^="Which of"] rect`, (rs) => rs.length - 1);
check('each strip draws one segment for the one chain', (await segments('a')) === 1 && (await segments('b')) === 1);
const sentence = (regs) =>
  regs.length === 1
    ? 'These two designs are the same, so every number of chains is a tie.'
    : `For these two designs, at any chain length: ${regs.map((g) => {
        const span = g.from === g.to ? `at ${nf(g.from)}` : `from ${nf(g.from)} to ${nf(g.to)}`;
        return g.winner === 'tie' ? `a tie ${span}` : `${g.winner === 'a' ? 'A' : 'B'} leads ${span}`;
      }).join(', ')} chains, the most this explorer runs.`;
let regs = regions(sweep(a, b, 4096, 100));
check('the lead is summarised from the model: A to 12, tie 13 to 16, B from 17', (await outText('regions')) === sentence(regs) && (await outText('regions')).includes('A leads from 1 to 12, a tie from 13 to 16, B leads from 17 to 4,096'), await outText('regions'));
if (out) await page.screenshot({ path: out + '-one-chain.png', fullPage: true });

// ---- the prediction's case, and the factor turning round
await setChains(1024);
r = race(a, b, 1024, 100);
check('1,024 chains: the many units win by 4', (await outText('verdict')) === verdictFor(r) && r.winner === 'b' && r.factor === 4, await outText('verdict'));
check('1,024 chains: waves and last results match the model', (await stat('a-last')) === nf(r.a.last) && (await stat('b-last')) === nf(r.b.last) && (await outText('a-used')) === `Every unit gets a chain. ${nf(r.a.waves)} waves.`,
  `${await stat('a-last')} ${await stat('b-last')} ${await outText('a-used')}`);
check('1,024 chains: both designs deliver their peak rate', (await stat('a-delivered')) === nf(a.throughput) && (await stat('b-delivered')) === nf(b.throughput));
const marks = await page.$$eval(`${B} svg[aria-label^="Time to the last result"] text`, (ts) => ts.map((x) => x.textContent));
check('the plot marks the current point with both finishing times', marks.includes(`A: ${nf(r.a.last)}`) && marks.includes(`B: ${nf(r.b.last)}`), marks.join('|'));
check('the plot draws one line per design', (await page.$$(`${B} svg[aria-label^="Time to the last result"] polyline`)).length === 2);
if (out) await page.screenshot({ path: out + '-crowd.png', fullPage: true });

// ---- where the lead changes hands
for (const n of [12, 13, 16, 17, 64]) {
  await setChains(n);
  r = race(a, b, n, 100);
  check(`${n} chains: ${r.winner === 'tie' ? 'a tie' : r.winner.toUpperCase() + ' leads by ' + nf(r.factor)}`, (await outText('verdict')) === verdictFor(r), await outText('verdict'));
}
await setChains(17);
r = race(a, b, 17, 100);
check('17 chains: the fast design\'s strip draws 17 segments', (await segments('a')) === 17, String(await segments('a')));
check('17 chains is where B first leads, by 1.25', r.winner === 'b' && r.factor === 1.25 && (await outText('verdict')).includes('Design B finishes first, 1.25 times sooner.'));
check('the tie region is shaded in the plot', (await page.$$(`${B} svg[aria-label^="Time to the last result"] rect`)).length === 2);

// ---- chain length never changes the winner or the factor
for (const len of [1, 37, 1000]) {
  await setLength(len);
  const rl = race(a, b, 17, len);
  check(`length ${len}: same winner and factor, times scaled`, (await outText('verdict')) === verdictFor(rl) && rl.factor === r.factor && (await stat('a-last')) === nf(rl.a.last) && (await stat('b-last')) === nf(rl.b.last),
    `${await outText('verdict')} ${await stat('a-last')} ${await stat('b-last')}`);
  check(`length ${len}: the lead changes hands at the same chain counts`, (await outText('regions')) === sentence(regs));
}
await setLength(100);

// ---- the log slider
await setInput('Independent chains (log scale)', 100);
check('the log slider at 100 of 120 is 1,024 chains', (await text(`${B} label.field`)).startsWith('Independent chains = 1,024'), await text(`${B} label.field`));

// ---- other designs
await pickDesign('Design A', '1 unit, speed 8');
a = design(BUDGET, 3);
r = race(a, b, 1024, 100);
check('1 unit of speed 8 against 64 plain ones, 1,024 chains: B by 8', (await outText('verdict')) === verdictFor(r) && r.factor === 8 && r.winner === 'b', await outText('verdict'));
regs = regions(sweep(a, b, 4096, 100));
check('its summary of who leads comes from the model too', (await outText('regions')) === sentence(regs), await outText('regions'));
check('its floor plan is one unit that draws the whole budget', (await units('a')) === 1 && (await text(`${B} [data-design="a"]`)).includes('Each unit draws 64 of the 64'));
await setChains(4096);
check('4,096 chains on one unit: the strip still draws, as waves', (await segments('a')) >= 1 && (await stat('a-last')) === nf(finish(a, 4096, 100).last), String(await segments('a')));
if (out) await page.screenshot({ path: out + '-one-unit.png', fullPage: true });
await pickDesign('Design A', '64 units, speed 1');
check('two identical designs tie everywhere', (await outText('verdict')).startsWith('A tie: both finish at') && (await outText('regions')) === sentence([{ winner: 'tie', from: 1, to: 4096 }]), await outText('regions'));
await pickDesign('Design A', '16 units, speed 2');
a = design(BUDGET, 1);
await setChains(40);
r = race(a, b, 40, 100);
check('16 units of speed 2 against 64 plain ones, 40 chains: matches the model', (await outText('verdict')) === verdictFor(r) && (await stat('a-shape')) === `${a.units} × ${a.speed}`, await outText('verdict'));
await pickDesign('Design A', '4 units, speed 4');
a = design(BUDGET, 2);

// ---- the numeric question, checked in the explorer first
await setChains(80);
const two = finish(b, 80, 100);
check('80 chains of 100 on 64 plain units finish at 200, in two waves', (await stat('b-last')) === '200' && two.last === 200 && (await outText('b-used')) === 'Every unit gets a chain. 2 waves.', `${await stat('b-last')} ${await outText('b-used')}`);
check('80 chains: the plain design draws 80 segments', (await segments('b')) === 80, String(await segments('b')));
q = await question('80 independent chains of 100 operations');
await (await q.$('input[type=number]')).type('100');
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question rejects one wave', (await q.evaluate((e) => e.textContent)).includes('Not quite. The answer is 200 time units.'));
await (await q.$('input[type=number]')).evaluate((el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '200');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(60);
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question accepts two waves', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- the two remaining questions
q = await question('which design finishes each job first');
const jobX = race(a, b, 8, 1000);
const jobY = race(a, b, 1024, 5);
check('pieces, not arithmetic: job X has more operations than Y and fewer chains', 8 * 1000 > 1024 * 5 && jobX.winner === 'a' && jobY.winner === 'b');
t = await choose(q, 0);
check('pieces, not arithmetic: "both on A" is rejected with the model\'s times for Y', t.includes('Not quite.') && t.includes(`done at ${nf(jobY.a.last)}`) && t.includes(`done at ${nf(jobY.b.last)}`), t.slice(-300));
t = await choose(q, 1);
check('pieces, not arithmetic: "both on B" is rejected with the model\'s times for X', t.includes('Not quite.') && t.includes(`done at ${nf(jobX.b.last)}`) && t.includes(`done at ${nf(jobX.a.last)}`), t.slice(-300));
t = await choose(q, 2);
check('pieces, not arithmetic: "X on A, Y on B" is accepted', t.includes('Correct.') && t.includes('Count the independent pieces') && t.includes('the job with more arithmetic goes to the latency design'));
await setChains(8);
await setLength(1000);
check('the explorer confirms job X: A first, by 2', (await outText('verdict')) === verdictFor(jobX) && jobX.factor === 2, await outText('verdict'));
await setChains(1024);
await setLength(5);
check('the explorer confirms job Y: B first, by 4', (await outText('verdict')) === verdictFor(jobY) && jobY.factor === 4, await outText('verdict'));
await setLength(100);
q = await question('token by token');
t = await choose(q, 2);
check('tokens: "a GPU cannot help" is rejected', t.includes('Not quite.') && t.includes('inside one turn of the loop'));
t = await choose(q, 1);
check('tokens: independence inside each step is accepted, and marked as ours', t.includes('Correct.') && t.includes('This mapping is ours') && t.includes('our analysis'));

// ---- phone width: the drawings are made at their size, so their text stays legible
const minFont = (sel) =>
  page.$$eval(sel, (svgs) =>
    Math.min(
      ...svgs.flatMap((svg) => {
        const vb = svg.viewBox.baseVal;
        const k = svg.getBoundingClientRect().width / (vb && vb.width ? vb.width : svg.getBoundingClientRect().width);
        return [...svg.querySelectorAll('text')].map((t) => Number(t.getAttribute('fontSize') ?? t.getAttribute('font-size') ?? 12) * k);
      }),
    ),
  );
check('at desktop width the redrawn figure is one drawing', (await page.$eval('figure', (f) => f.dataset.stacked)) === '0' && (await page.$$('figure svg')).length === 1);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const phone = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
check('at 390 px the page does not scroll sideways', phone.scroll <= phone.client, JSON.stringify(phone));
check('at 390 px the figure stacks its two chips, still drawing 4 and 64 units', (await page.$eval('figure', (f) => f.dataset.stacked)) === '1' && (await page.$$('figure svg')).length === 2 && (await page.$$('figure svg rect[rx]')).length === a.units + b.units);
const figFont = await minFont('figure svg');
check(`at 390 px the figure's smallest text is at least 10 px (${figFont.toFixed(1)})`, figFont >= 10, String(figFont));
const plotW = await page.$eval(`${B} svg[aria-label^="Time to the last result"]`, (s) => s.getBoundingClientRect().width);
const plotFont = await minFont(`${B} svg[aria-label^="Time to the last result"]`);
check(`at 390 px the plot is drawn to fit (${Math.round(plotW)} px) with text of at least 10 px (${plotFont.toFixed(1)})`, plotW <= 360 && plotW >= 260 && plotFont >= 10, `${plotW} ${plotFont}`);
const plotLabels = await page.$$eval(`${B} svg[aria-label^="Time to the last result"]`, (svgs) => {
  const svg = svgs[0];
  const box = svg.getBoundingClientRect();
  return [...svg.querySelectorAll('text')].filter((t) => { const r = t.getBoundingClientRect(); return r.left < box.left - 0.5 || r.right > box.right + 0.5; }).map((t) => t.textContent);
});
check('at 390 px no label of the plot runs off its edge', plotLabels.length === 0, plotLabels.join(' | '));
await setChains(4096);
const edgeLabels = await page.$$eval(`${B} svg[aria-label^="Time to the last result"]`, (svgs) => {
  const svg = svgs[0];
  const box = svg.getBoundingClientRect();
  return [...svg.querySelectorAll('text')].filter((t) => { const r = t.getBoundingClientRect(); return r.left < box.left - 0.5 || r.right > box.right + 0.5; }).map((t) => t.textContent);
});
check('at 390 px and 4,096 chains the labels of the last point turn inwards', edgeLabels.length === 0, edgeLabels.join(' | '));
if (out) await page.screenshot({ path: out + '-phone.png', fullPage: true });
await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
await pause(300);

// ---- progress would have been saved under the stable ids
await pause(900);
const last = saved[saved.length - 1] ?? '';
const ids = ['ch01/start/q-batch', 'ch01/start/q-sixteenbit', 'ch01/start/q-chainlen', 'ch01/start/q-tokens', 'ch01/start/q-oneunitmore', 'ch01/start/p-crowd'];
check('answers and the prediction are saved under their ids, and only to the intercepted request', ids.every((id) => last.includes(`"${id}"`)), ids.filter((id) => !last.includes(`"${id}"`)).join(' '));

// ---- errors
const expected404 = (u) => /\/content\/ch01\/narration\/(manifest\.json|two-designs-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
