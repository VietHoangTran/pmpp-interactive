// Browser check for chapter 16, "Summary and exercises": what to carry forward, the five
// questions of the mixed check (numbers taken from the same models the page imports), the
// exercises table, the two errata lists, what was left out, the project status read from
// /api/reports, links, margin notes and the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch16-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { presetFronts, stencil, wavefronts } from '../src/sim/wavefront.ts';
import { launches, stepsPerTile } from '../src/sim/wavetile.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') saved.push(r.postData() ?? '');
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch16/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises"', h1.includes('Summary and exercises'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('nothing on the page says it is being written', !/being written/i.test(text0));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => !['P', 'UL', 'OL', 'ASIDE'].includes(a.previousElementSibling?.tagName ?? '')).map((a) => a.textContent.slice(0, 60)));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(' | '));
const carry = await page.evaluate(() => {
  const h = [...document.querySelectorAll('.prose h2')].find((x) => x.textContent.includes('carry forward'));
  let e = h?.nextElementSibling;
  while (e && e.tagName !== 'UL') e = e.nextElementSibling;
  return e ? e.querySelectorAll('li').length : 0;
});
check('seven points to carry forward', carry === 7, String(carry));
check('carry-forward names relaunch, barrier, flag, two buffers, 2W − 1 and 3T − 1',
  ['a relaunch when the whole grid', 'a block barrier between steps', 'a flag when a tile waits', 'two buffers', 'W steps instead of 2W − 1', '3T − 1 launches'].every((s) => text0.includes(s)));
check('the inference bridge names CTC, RWKV, Mamba and the gated delta rule', ['CTC', 'RWKV', 'Mamba', 'gated delta rule'].every((s) => text0.includes(s)));
check('the mixed check carries a "Not in the book" note', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('H.264'))));

// ---- the mixed check
const qh = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerNumber = async (part, value) => {
  const h = await qh(part);
  await h.evaluate((e, v) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const answerChoice = async (part, option) => {
  const h = await qh(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return ok && (await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
const qs = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent));
check('five questions in the mixed check', qs.length === 5, qs.join(', '));

// H.264 tiles: the model numbers the tile grid under the tile-level stencil W, N, NE.
const T = 4;
const h264 = wavefronts(T, T, stencil(['W', 'N', 'NE']));
const sw = presetFronts('smith-waterman', T, T);
let v = await answerNumber('the H.264 pattern on page', 2 * T - 1);
check(`q-h264-tiles: ${2 * T - 1} (square tiles' fronts) is wrong, the answer is the model's ${h264.steps}`, v.includes(`Not quite. The answer is ${h264.steps} launches.`), v);
v = await answerNumber('the H.264 pattern on page', h264.steps);
check(`q-h264-tiles: correct, with ${h264.steps} launches of at most ${h264.widest} against ${sw.steps} of up to ${sw.widest}`,
  v.includes('Correct.') && v.includes(`${h264.steps} launches of at most ${h264.widest} tiles`) && v.includes(`${sw.steps} launches of up to ${sw.widest}`) && v.includes('tile_col = d − 2·tile_row'), v);

v = await answerChoice('How many', 'One, after the writes');
check('q-stencil-barriers: one barrier is wrong, with the write-after-read reason', v && v.includes('Not quite.') && v.includes('overwrites it'), v);
v = await answerChoice('How many', 'Two: one between');
check('q-stencil-barriers: two is correct, and double buffering is named', v && v.includes('Correct.') && v.includes('double buffering'), v);

v = await answerChoice('Viterbi decoding', 'Yes, as long as blocks take their work');
check('q-viterbi: the counter answer is wrong, with its reason', v && v.includes('Not quite.') && v.includes('does not shrink what a block waits for'), v);
v = await answerChoice('Viterbi decoding', 'No: each block of step t needs');
check('q-viterbi: correct, naming grid.sync() and CTC', v && v.includes('Correct.') && v.includes('grid.sync()') && v.includes('CTC'), v);

v = await answerChoice('causal attention layer', 'Yes: row i needs the rows above it');
check('q-causal: "rows are fronts" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('needs no result of another row'), v);
v = await answerChoice('causal attention layer', 'No: no score reads another score');
check('q-causal: correct, naming Mamba or RWKV as a real wavefront', v && v.includes('Correct.') && v.includes('Mamba or RWKV'), v);

// Exercise 2: rectangular tiles of 16 rows by 64 columns over L_seq = 1000.
const rows = Math.ceil(1000 / 16), cols = Math.ceil(1000 / 64), rect = rows + cols - 1;
const sq16 = launches({ Lseq: 1000, W: 16, scheme: 'square' }), sq64 = launches({ Lseq: 1000, W: 64, scheme: 'square' });
v = await answerNumber('rectangular tiles', sq16);
check(`q-ex2: ${sq16} (square 16 tiles) is wrong, the answer is ${rect}`, v.includes(`Not quite. The answer is ${rect} launches.`), v);
v = await answerNumber('rectangular tiles', rect);
check(`q-ex2: correct, ${rows} + ${cols} − 1, ${16 + 64 - 1} steps, square 16: ${sq16}, square 64: ${sq64} and ${stepsPerTile({ W: 64, scheme: 'square' })} steps`,
  v.includes('Correct.') && v.includes(`= ${rows} tile rows`) && v.includes(`= ${cols} tile columns`) && v.includes(`${rect} launches of at most ${Math.min(rows, cols)} tiles`) &&
  v.includes(`${16 + 64 - 1} of them`) && v.includes(`Square tiles of 16 need ${sq16} launches`) && v.includes(`of 64, ${sq64} launches`) &&
  v.includes(`${stepsPerTile({ W: 64, scheme: 'square' })} steps per tile`), v);
await pause(600);
const answers = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-h264-tiles', 'q-stencil-barriers', 'q-viterbi', 'q-causal', 'q-ex2'].map((s) => `ch16/wrap/${s}`);
check('every answer went to the intercepted /api/progress under its id', ids.every((id) => answers[id]), Object.keys(answers).join(', '));
check('  each records two attempts, the last one correct', ids.every((id) => answers[id]?.attempts === 2 && answers[id]?.correct === true),
  JSON.stringify(ids.map((id) => [answers[id]?.attempts, answers[id]?.correct])));
await shot('check', '.q');

// ---- exercises, errata, what was left out
const ex = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) => trs.map((tr) => ({ n: tr.children[0].textContent.trim(), where: tr.children[2].textContent, links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('six exercises and the two set in the text are mapped', ex.map((e) => e.n).join() === '1,2,3,4,5,6,p. 384,p. 386', ex.map((e) => e.n).join());
check('  each says where it is worked or that it is not built', ex.every((e) => e.links.length || e.where.includes('question above')), JSON.stringify(ex));
check('  the in-text exercises point at the lab', ex.slice(6).every((e) => e.links.includes('#/ch16/labs')));
const errata = await page.$$eval('[data-list="errata"] li', (li) => li.map((x) => x.textContent));
check('the wording errata cover Fig. 16.5, Eq. 16.2 names, the index flip, Fig. 16.14(a), p. 391, p. 398, p. 377 and p. 396',
  errata.length === 8 && ['Fig. 16.5', 'Eq. 16.2', 'p. 382', 'Fig. 16.14(a)', 'p. 391', 'p. 398', 'p. 377', 'p. 396'].every((s) => errata.some((e) => e.includes(s))), String(errata.length));
const code = await page.$$eval('[data-list="code"] li', (li) => li.map((x) => x.textContent));
check('the code errata cover INFINITY, the early return, unsigned arithmetic, the store test, pad() and undefined names',
  code.length === 6 && ['INFINITY', 'returns out-of-range threads', 'unsigned', 'q >= 1 && q < L', 'pad()', 'never define'].every((s) => code.some((e) => e.includes(s))), String(code.length));
check('INT_MAX / 2 is the course\'s sentinel', text0.includes('INT_MAX / 2'));
check('what was left out names traceback, Fig. 16.13, the sidebar, DPX and the references',
  ['traceback', 'Fig. 16.13', 'p. 397', 'DPX as a question only', 'origin of hypertiling'].every((s) => text0.includes(s)));

// ---- project status
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const pass = [...real, 'emu'].filter((b) => labStatus(reports[b], 'ch16/lab1_wavefront', 'work') === 'pass');
const lab = await page.$eval('[data-lab161]', (e) => ({ v: e.dataset.lab161, text: e.textContent }));
check(`the lab status is read from the reports (${pass.join(',') || 'none'})`, lab.v === (pass.join(',') || 'none'), JSON.stringify(lab));
check('the project status says there is no milestone and the lab is optional', text0.includes('adds no milestone') && text0.includes('Lab 16.1 is optional'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch16/16.3', '#/ch16/16.6', '#/ch16/labs', '#/ch17/start', '#/ch18/18.6', '#/ch20/start']);
check(`every link goes to a known route (${links.join(' ')})`, links.every((l) => known.has(l)) && links.length === known.size, links.filter((l) => !known.has(l)).join(', '));
for (const target of ['ch16/16.3', 'ch16/16.6', 'ch16/labs', 'ch17/start', 'ch18/18.6', 'ch20/start']) {
  await goto(page, target);
  const t = await page.$eval('.page h1', (e) => e.textContent).catch(() => '');
  check(`the link to ${target} opens a page`, t.length > 2, t);
}

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch16/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
