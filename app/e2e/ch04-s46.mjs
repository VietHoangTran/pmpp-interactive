// Browser check for section 4.6, warp scheduling and latency tolerance: the explorer
// "How many warps hide a wait?" against the model (src/sim/latency.ts, imported here
// directly), the prediction and its reveal, four questions, the margin notes, and the
// page at a narrow width. The narrated animation has its own check,
// e2e/ch04-scene-latency.mjs.
// Usage: node e2e/ch04-s46.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { PRESETS, estimate, idleCycles, period, simulate, utilisation, warpsNeeded } from '../src/sim/latency.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const pct = (f) => Math.round(f * 1000) / 10;
const E = 'section.bench[aria-label="How many warps hide a wait?"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const setRange = async (label, v) => {
  await page.$eval(`${E} input[type=range][aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const preset = async (key) => {
  await page.$eval(`${E} button[data-preset="${key}"]`, (b) => b.click());
  await pause(80);
};
/** What the explorer shows, read from its outputs and its drawing. */
const shown = async () => ({
  util: Number(await attr(`${E} [data-out=util]`, 'data-value')),
  estimate: Number(await attr(`${E} [data-out=estimate]`, 'data-value')),
  needed: Number(await text(`${E} [data-out=needed]`)),
  idle: Number(await attr(`${E} svg[data-rows]`, 'data-idle')),
  ready: Number(await attr(`${E} svg[data-rows]`, 'data-ready')),
  rows: Number(await attr(`${E} svg[data-rows]`, 'data-rows')),
  cycles: Number(await attr(`${E} svg[data-rows]`, 'data-cycles')),
  verdict: await text(`${E} [data-out=verdict]`),
  panel: await text(`${E} [aria-live]`),
});
/** What the model says for the same inputs, drawn over as many cycles as the explorer draws. */
const model = (warps, work, latency, slots) => {
  const cfg = { warps, work, latency, slots };
  const cycles = Math.max(48, Math.min(300, 3 * period(cfg)));
  const tl = simulate(cfg, cycles);
  return {
    util: pct(utilisation(cfg)),
    estimate: pct(estimate(cfg)),
    needed: warpsNeeded(work, latency, slots),
    idle: idleCycles(tl),
    ready: tl.rows.reduce((s, row) => s + row.filter((c) => c === 'ready').length, 0),
    rows: warps,
    cycles,
  };
};
const agrees = (s, m) => ['util', 'estimate', 'needed', 'idle', 'ready', 'rows', 'cycles'].every((k) => s[k] === m[k]);
const show = (s, m) => JSON.stringify({ shown: { ...s, panel: undefined, verdict: undefined }, model: m });
const q = (words) => page.evaluateHandle((w) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(w)), words);

await goto(page, 'ch04/4.6');
await page.waitForSelector(E);

// ---- the page around the interactives
const notes = await page.$$eval('aside.note', (a) => a.map((n) => n.textContent.replace(/\s+/g, ' ').trim()));
check('every book claim carries its pages: §4.6 pp. 82–83, p. 83, pp. 83–84, p. 84, pp. 84–85', ['pp. 82–83', 'p. 83', 'pp. 83–84', 'p. 84', 'pp. 84–85'].every((p) => notes.some((n) => n.startsWith('Book §4.6, ' + p))), notes.filter((n) => n.startsWith('Book')).join(' | '));
check('vendor facts are dated and labelled (Today for NVIDIA; Not in the book for HIP, Metal and the bridge)', notes.some((n) => n.startsWith('Today') && n.includes('A11')) && notes.some((n) => n.includes('B8') && n.includes('unverified')) && notes.some((n) => n.includes('B10')), notes.join(' | '));
check('the model says what it leaves out', notes.some((n) => n.startsWith('What this model leaves out') && n.includes('unlimited bandwidth') && n.includes('NVIDIA does not document')));
check('the scene "Hiding a wait" sits in this section', !!(await page.$('section.scene[aria-label="Hiding a wait"]')));
check('the section links on to 4.7', !!(await page.$('a[href="#/ch04/4.7"]')));

// ---- the explorer, step by step
let s = await shown();
let m = model(4, PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency, 1);
check('starts at 4 warps of the load-heavy preset: 25% busy, 16 needed, as the model says', agrees(s, m) && s.util === 25 && s.needed === 16, show(s, m));
check('the estimate is written out without doubled brackets', s.panel.includes('min(1, R × c / (c + L)) = min(1, 4 × 2 / 32) = 25%') && !s.panel.includes('(('), s.panel);
check('below the threshold the verdict counts the warps short', s.verdict.startsWith('12 warps short.') && s.verdict.includes('75% of its cycles'), s.verdict);
check('below the threshold no warp is kept waiting for the unit after the cold start', s.ready === m.ready && s.verdict.includes('Once the cold start has passed'), s.verdict);
if (out) await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-explorer-4.png' });

for (const R of [1, 8, 15]) {
  await setRange('Resident warps R', R);
  s = await shown();
  m = model(R, 2, 30, 1);
  check(`R = ${R}: every warp adds its full share, ${m.util}%`, agrees(s, m) && s.util === pct(R / 16), show(s, m));
}
await setRange('Resident warps R', 16);
s = await shown();
m = model(16, 2, 30, 1);
check('R = 16: the unit never idles, even from the cold start, and the verdict says exactly enough', agrees(s, m) && s.util === 100 && s.idle === 0 && s.verdict.startsWith('Exactly enough.'), show(s, m));
await setRange('Resident warps R', 24);
s = await shown();
m = model(24, 2, 30, 1);
check('R = 24: still 100%; the 8 extra warps only queue (ready, not chosen, drawn grey)', agrees(s, m) && s.util === 100 && s.ready > 0 && s.verdict.startsWith('8 warps more than needed.'), show(s, m));
if (out) await page.screenshot({ path: out + '-explorer-24.png' });

await preset('compute');
s = await shown();
m = model(24, PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency, 1);
check('long compute between loads, same 24 warps: 2 are enough, 22 are spare', agrees(s, m) && s.needed === 2 && s.verdict.startsWith('22 warps more than needed.'), show(s, m));
check('the compute preset is marked as chosen', (await attr(`${E} button[data-preset=compute]`, 'aria-pressed')) === 'true' && (await attr(`${E} button[data-preset=load]`, 'aria-pressed')) === 'false');
await setRange('Resident warps R', 1);
s = await shown();
m = model(1, 30, 30, 1);
check('one warp of the compute-heavy kernel keeps the unit half busy', agrees(s, m) && s.util === 50, show(s, m));

await preset('book');
s = await shown();
m = model(64, 30, 30, 4);
check("the book's example SM (64 warps, 4 slots) with the compute-heavy kernel: 8 needed, 100%", agrees(s, m) && s.needed === 8 && s.util === 100 && s.panel.includes('slots × (c + L)'), show(s, m));
await preset('load');
s = await shown();
m = model(64, 2, 30, 4);
check("the book's example SM with a load every other instruction: exactly 64 needed, the 16 to 1 of p. 85", agrees(s, m) && s.needed === 64 && s.util === 100 && s.verdict.startsWith('Exactly enough.'), show(s, m));
await setRange('Memory latency L, in cycles', 31);
s = await shown();
m = model(64, 2, 31, 4);
check('one more cycle of latency and 64 warps on 4 slots fall short', agrees(s, m) && s.util < 100 && s.needed > 64, show(s, m));
await setRange('Issue slots', 1);
await setRange('Cycles of work between memory accesses c', 7);
await setRange('Memory latency L, in cycles', 13);
await setRange('Resident warps R', 2);
s = await shown();
m = model(2, 7, 13, 1);
check('any slider setting matches the model (R 2, c 7, L 13): 70%', agrees(s, m) && s.util === 70, show(s, m));

// ---- the prediction
const P = await q('How many warps have to be resident to keep one execution unit busy');
await (await P.$('input[type=number]')).type('12');
await page.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click(), P);
await pause(150);
const reveal = await P.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal gives 16 and what 12 warps would have managed (75%)', reveal.includes('You predicted 12 warps.') && reveal.includes('16.') && reveal.includes('With 12 the unit would be busy 75% of the time.') && pct(utilisation({ warps: 12, work: 2, latency: 30, slots: 1 })) === 75, reveal);
const pred = await P.evaluate((e) => { const svg = e.querySelector('[data-out=predicted] svg'); return svg && { rows: +svg.dataset.rows, idle: +svg.dataset.idle }; });
check('and draws 16 warps from a cold start with no idle cycle', pred && pred.rows === 16 && pred.idle === 0 && idleCycles(simulate({ warps: 16, work: 2, latency: 30, slots: 1 }, 96)) === 0, JSON.stringify(pred));

// ---- questions
const answerNumeric = async (words, v) => {
  const h = await q(words);
  await (await h.$('input[type=number]')).type(String(v));
  await page.evaluate((e) => e.querySelector('button.primary').click(), h);
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const answerChoice = async (words, optionWords) => {
  const h = await q(words);
  const picked = await h.evaluate((e, w) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(w));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionWords);
  await pause(60);
  await page.evaluate((e) => e.querySelector('button.primary').click(), h);
  await pause(120);
  return { picked, text: await h.evaluate((e) => e.textContent) };
};
const util = await answerNumeric('8 warps are resident on a unit with one issue slot', 50);
check('q-util: 50% is accepted, and is the model\'s figure', util.includes('Correct.') && pct(utilisation({ warps: 8, work: 4, latency: 60, slots: 1 })) === 50, util.slice(0, 200));
let a = await answerChoice('A CPU pays for every context switch', 'stay in on-chip registers');
check('q-zero: the resident-state answer is right', a.picked && a.text.includes('Correct.'));
a = await answerChoice('What does latency tolerance let a GPU designer leave out', 'Each thread also finishes sooner');
check('q-area: a wrong option is marked wrong and says why', a.picked && a.text.includes('Not quite.') && a.text.includes("does not shorten any one thread's wait"), a.text.slice(0, 300));
a = await answerChoice('What does latency tolerance let a GPU designer leave out', 'A thread that waits on memory still waits just as long');
check('q-area: then the right one is accepted', a.picked && a.text.includes('Correct.'));
a = await answerChoice('Why can the first get away with so few warps?', 'compute for many cycles on values already in registers');
check('q-lowocc: the compute-per-load answer is right', a.picked && a.text.includes('Correct.'));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((E) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  bench: document.querySelector(E).getBoundingClientRect().right <= document.documentElement.clientWidth + 1,
}), E);
check('at a narrow width nothing overflows the page', narrow.page && narrow.bench, JSON.stringify(narrow));
if (out) await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

const expected404 = (u) => /\/content\/ch04\/narration\/(manifest\.json|latency-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
