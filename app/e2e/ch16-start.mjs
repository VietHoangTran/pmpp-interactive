// Browser check for chapter 16, "Before you start": the small table numbered by step
// (compared with the model it is drawn from, src/sim/wavefront.ts, imported here directly),
// the hook question, the five prerequisite questions, the margin notes, the links of the
// chapter map, and the page at phone width.
// Usage: node e2e/ch16-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { inputsOf, presetFronts } from '../src/sim/wavefront.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so that no
// question is prefilled whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  const input = await q.$('input[type=number]');
  // Clear through the native setter so React sees the change, then type.
  await input.evaluate((el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const q = await qHandle(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch16/start');
await page.waitForSelector('.q');

// ---------------------------------------------------------------- the small table against the model
const FIG = presetFronts('smith-waterman', 6, 6);
const steps = await page.$$eval('figure svg g[data-cell]', (g) => g.map((x) => Number(x.dataset.step)));
check('the table shows the model\'s step for every cell (r + c)', JSON.stringify(steps) === JSON.stringify(FIG.step), JSON.stringify(steps));
const front = FIG.sizes.indexOf(FIG.widest);
const outlined = await page.$$eval('figure svg g[data-cell] rect', (r) => r.filter((x) => Number(x.getAttribute('stroke-width')) > 2).length);
check(`the widest step (${front}) is outlined: ${FIG.widest} cells`, outlined === FIG.widest, String(outlined));
const arrows = await page.$$eval('figure svg line[data-input]', (l) => l.map((x) => x.dataset.input).sort());
check('the arrows are the three inputs of cell (3, 4)', JSON.stringify(arrows) === JSON.stringify(inputsOf(FIG, 3, 4).map(([r, c]) => `${r},${c}`).sort()), JSON.stringify(arrows));
const figText = await page.$eval('figure', (f) => f.textContent);
check('the legend gives the model\'s numbers', figText.includes(`step ${front},`) && figText.includes(`${FIG.widest} cells that need`) && figText.includes(`${FIG.steps} steps in all`), figText.slice(0, 300));

// ---------------------------------------------------------------- margin notes, labels, links
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
check('the page cites the opening and §16.1, pp. 373–374', notes.some((t) => t.includes('pp. 373–374')), notes.join(' | '));
check('the inference link is marked as not in the book', notes.some((t) => t.startsWith('Not in the book') && t.includes('group D')));
check('the project facts carry their check date and files', notes.some((t) => t.startsWith('Today') && t.includes('2026-11-20') && t.includes('LossCTC.cu') && t.includes('gated_delta.py')));
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('the hook comes first, before any theory', labels[0] === 'Before any theory', labels.join('|'));
for (const n of [2, 4, 5, 6, 11]) check(`a question labelled "From chapter ${n}"`, labels.includes(`From chapter ${n}`), labels.join('|'));
check('six questions on the page', labels.length === 6, String(labels.length));
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
for (const l of ['#/ch16/16.3', '#/ch16/16.6', '#/ch16/labs', '#/ch16/wrap', '#/review']) check(`the page links to ${l}`, links.includes(l), links.join('|'));
const prose = await page.$eval('.prose', (e) => e.textContent);
check('the page says plainly that the case studies are not inference operations', /are not\s+inference operations/.test(prose));
check('what is left out is named with pages', prose.includes('§16.5, pp. 382–384') && prose.includes('(p. 397)') && prose.includes('(pp. 397–399)'));

// ---------------------------------------------------------------- the hook
let t = await answerChoice('edit distance between two sequences', 1);
check('hook: "each row in parallel" is wrong, because a row is a chain', t.includes('Not quite.') && t.includes('A row is a chain'), t.slice(0, 400));
const HOOK = presetFronts('smith-waterman', 1000, 1000);
t = await answerChoice('edit distance between two sequences', 2);
check('hook: the anti-diagonals are right', t.includes('Correct.'), t.slice(0, 300));
check(`hook: the model's numbers, ${HOOK.steps} steps, widest ${HOOK.widest}, about ${Math.round(HOOK.mean)}`,
  t.includes('1,999 steps') && t.includes('the widest 1,000 cells') && t.includes(`about ${Math.round(HOOK.mean)} cells`) && HOOK.steps === 1999 && HOOK.widest === 1000, t.slice(0, 900));
if (out) await page.screenshot({ path: out + '-hook.png' });

// ---------------------------------------------------------------- the prerequisite questions
t = await answerChoice('Is that call needed for launch k + 1', 0);
check('stream order: "yes" is wrong, a stream runs in order', t.includes('Not quite.') && t.includes('Not within one stream'), t.slice(0, 400));
t = await answerChoice('Is that call needed for launch k + 1', 1);
check('stream order: "no, it only makes the host wait" is right, with Metal\'s encoder', t.includes('Correct.') && t.includes('hazard-tracked'), t.slice(0, 500));
t = await answerChoice('documented to work', 1);
check('exit barrier: "every backend" is wrong', t.includes('Not quite.') && t.includes('Apple silicon only'), t.slice(0, 400));
t = await answerChoice('documented to work', 0);
check('exit barrier: CUDA only, keep every thread to the barrier', t.includes('Correct.') && t.includes('never returns before its barrier'), t.slice(0, 500));
t = await answerChoice('that no thread has used before', 2);
check('one barrier: "two, as in tiled matmul" is wrong, nothing is reused', t.includes('Not quite.') && t.includes('the tiles are reused'), t.slice(0, 400));
t = await answerChoice('that no thread has used before', 1);
check('one barrier per step is right, and the page cites Figs. 16.9 and 16.16', t.includes('Correct.') && t.includes('Figs. 16.9 and 16.16'), t.slice(0, 600));
t = await answerNumeric('16*t + c', 32);
check('stride 16: 32 is wrong, the answer is 16', t.includes('Not quite. The answer is 16.'), t.slice(0, 300));
t = await answerNumeric('16*t + c', 16);
check('stride 16: 16 is right, two banks of 16 lanes', t.includes('Correct.') && t.includes('16 lanes on each of two'), t.slice(0, 400));
t = await answerChoice('from an atomic counter instead of using', 2);
check('counter: "blockIdx.x is not unique" is wrong', t.includes('Not quite.') && t.includes('It is unique'), t.slice(0, 400));
t = await answerChoice('from an atomic counter instead of using', 0);
check('counter: wait only for blocks that have started', t.includes('Correct.') && t.includes('keeps making progress'), t.slice(0, 500));
// The right option is not always first in the written order either.
const rightAt = await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelector('.opts')).map((q) => [...q.querySelectorAll('.opts label')].findIndex((l) => l.classList.contains('right'))));
check('the five choice questions have their right option in positions 3, 2, 1, 2 and 1', JSON.stringify(rightAt) === '[2,1,0,1,0]', JSON.stringify(rightAt));

await pause(700); // the app saves progress after a short delay
const last = saved.at(-1) ?? '';
check('answers were saved through PUT /api/progress (intercepted)',
  ['q-hook', 'q-stream-order', 'q-exit-barrier', 'q-raw-only', 'q-stride16', 'q-counter'].every((id) => last.includes('ch16/start/' + id)), last.slice(0, 300));
if (out) await page.screenshot({ path: out + '-page.png', fullPage: true });

// ---------------------------------------------------------------- themes and phone width
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
if (out) {
  await page.$eval('figure', (e) => e.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: out + '-figure-light.png' });
}
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
if (out) await page.screenshot({ path: out + '-figure-dark.png' });
await page.setViewport({ width: 390, height: 860, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('no horizontal scroll at 390 px', overflow <= 0, String(overflow));
const figW = await page.$eval('figure svg', (s) => s.getBoundingClientRect().width);
check('the figure shrinks to fit at 390 px', figW > 250 && figW <= 390, String(figW));
if (out) {
  await page.$eval('figure', (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
