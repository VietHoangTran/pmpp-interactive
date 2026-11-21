// Browser check for module X2's summary: the carry-forward list, the mixed check (five
// questions, the choices answered wrong then right, the number recomputed here), the map of
// the module's tasks with links that open written pages, where the sources disagree, and the
// M9 status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/x2-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every question is
// unanswered whatever the reader has already done.
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
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const question = async (part) => (await page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(part)), part)).asElement();
const qText = (q) => q.evaluate((e) => e.textContent);
const press = (q, label) => q.evaluate((e, label) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click(), label);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(v));
  await press(q, 'Check');
  await pause(120);
};
const pick = async (q, index) => {
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), index);
  await pause(40);
  await press(q, 'Check');
  await pause(120);
};

await goto(page, 'x2/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('[data-m9-own="loading"]'), { timeout: 30000 });
const body = () => page.$eval('.prose', (e) => e.textContent);
const M = await page.evaluate(async () => {
  const t = await import('/src/sim/qtraffic.ts');
  const d = await import('/src/sim/qdot.ts');
  return { q4: t.byteRatio('q4_0'), bound: d.BOUND_Q8Q8, blocks: Math.floor(d.INT32_MAX / d.BOUND_Q8Q8) };
});

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary"', h1.trim().endsWith('Summary'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The module's exercises", 'Where the sources disagree', 'What this module leaves out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('.prose > ul', (uls) => [...uls[0].children].map((li) => li.textContent));
check('the carry-forward list has seven points, from bytes per weight to logits and tokens', carry.length === 7 &&
  carry[0].includes('bytes per weight') && carry[1].includes('Fuse the dequantise') && carry[3].includes('2-byte aligned') && carry[5].includes('never half') && carry[6].includes('logits and its tokens'), JSON.stringify(carry.map((c) => c.slice(0, 30))));
const notes = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.textContent));
check("the module is labelled as ours, with the book's anchor §A.5, pp. 592–593", notes.some((t) => t.includes("Ours, not the book's")) && notes.some((t) => t.includes('§A.5, pp. 592–593')));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the page', !/\d+(\.\d+)? (GB\/s|KiB)\b|\d+\.\d+ ms\b/.test(text0));

// ---- the five questions: the choices answered wrong then right, the number once
check('five questions, none answered at the start', (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);
let q = await question('You switch the cache to llama.cpp');
const kv = 8 / (4 + (4 * 34) / 64);
// A numeric question takes one answer and then shows the right one, so it is answered once, right.
await typeAnswer(q, kv.toFixed(2));
check(`kv-cache: ${kv.toFixed(2)}, from 4 + 4 × 34/64 = 6.125 GB against 8`, (await qText(q)).includes('Correct.') &&
  (await qText(q)).includes('2.125 GB') && (await qText(q)).includes(`8 / 6.125 ≈ ${kv.toFixed(2)}`), await qText(q));
q = await question('You design your own 4-bit format');
await pick(q, 1);
check('own layout: "every load is aligned" is wrong, 16 divides 20i + 4 only every fourth block', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('only when i is 3, 7, 11'));
await pick(q, 2);
check('own layout: "as with Q4_0" is wrong, a 4-byte scale keeps codes at a multiple of 4', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('2 mod 4 in even blocks'));
await pick(q, 0);
check('own layout: 4-byte loads everywhere, 16-byte every fourth; emulators do not check alignment', (await qText(q)).includes('Correct.') && (await qText(q)).includes('they do not check alignment'));
q = await question('You port lab X2.3');
await pick(q, 1);
check(`row-int: overflow is wrong, an int32 holds about ${M.blocks.toLocaleString('en-US')} worst-case blocks`, (await qText(q)).includes('Not quite.') &&
  (await qText(q)).includes(`${M.bound.toLocaleString('en-US')}`) && (await qText(q)).includes(`about ${M.blocks.toLocaleString('en-US')} worst-case blocks`), await qText(q));
await pick(q, 3);
check('row-int: the FP16 scale is not the problem; the other blocks\' scales are', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('the other 127 blocks have other scales'));
await pick(q, 0);
check('row-int: different units; integer inside a block, float across blocks', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Integer inside a block,'));
q = await question("PyTorch's int4 kernel");
await pick(q, 1);
check("torch zero: \"the zero is always 0\" is wrong, it is a float added after scaling", (await qText(q)).includes('Not quite.') && (await qText(q)).includes('a float added after scaling'));
await pick(q, 2);
check("torch zero: the integer zero point is MPSGraph's convention", (await qText(q)).includes('Not quite.') && (await qText(q)).includes("MPSGraph's convention"));
await pick(q, 0);
check('torch zero: fold the zero per group through Σx; four conventions', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Four projects, four conventions'));
q = await question('512-token prompt');
await pick(q, 1);
check('prefill: the GEMV per 8 tokens reads every weight 64 times', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('reads every weight 64 times'));
await pick(q, 2);
check(`prefill: a whole float32 copy is ${M.q4.toFixed(1)} times the Q4_0 bytes`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${M.q4.toFixed(1)} times the bytes`));
await pick(q, 0);
check("prefill: a tile at a time into shared memory, as llama.cpp's mul_mm", (await qText(q)).includes('Correct.') && (await qText(q)).includes('mul_mm'));
await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-kv-cache', 'q-own-layout', 'q-row-int', 'q-torch-zero', 'q-prefill'].map((s) => 'x2/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this module you have answered'));
await shot('questions', '.q');

// ---- the module's tasks
const exRows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ task: tr.children[0].textContent.trim(), lab: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the task table says plainly the book has none, and has twelve rows', text0.includes('The book has no exercises for this module') && exRows.length === 12, String(exRows.length));
const sections = ['X2.1', 'X2.2', 'X2.3', 'X2.4', 'X2.5', 'X2.6', 'X2.7'];
check('every section of the module is linked from the task table', sections.every((s) => exRows.some((r) => r.links.includes(`#/x2/${s}`))), JSON.stringify(exRows.map((r) => r.links)));
check('every lab of the module is named in the task table', ['lab X2.1', 'lab X2.2', 'lab X2.3', 'lab X2.4'].every((l) => exRows.some((r) => r.lab.includes(l))));

// ---- where the sources disagree
const disagree = await page.$$eval('.prose > ul', (uls) => [...uls[1].children].map((li) => li.querySelector('b')?.textContent));
check('the disagreements are grouped: offset, MLX docs and code, nibble order, integer dot, PyTorch page',
  JSON.stringify(disagree) === JSON.stringify(['The offset.', "MLX's docs and MLX's code.", 'Nibble order.', 'The integer dot.', "PyTorch's own page."]), JSON.stringify(disagree));
check('the page says the module has no PDF errata, and where its own facts file and design were wrong',
  text0.includes('This module has no PDF to be wrong') && text0.includes('where our own facts file and design were wrong') &&
  /at 2 mod 4 in the even\s+blocks/.test(text0) && text0.includes('exactly 133,144 fit') && text0.includes('0x8000'));
check('the disagreements are dated and sourced to the facts file', notes.some((t) => t.includes('Checked 2026-11-20 in the facts pass (notes/today/x2-facts.md')));

// ---- the project status, from the reports
const NAME = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = run ? run.results.filter((x) => x.lab === id && x.variant === variant && x.tests.length) : [];
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const realB = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passingOn = (id) => [...realB, 'emu'].filter((b) => labStatus(reports[b], id, 'work') === 'pass');
const own = passingOn('x2/lab4_m9');
const ownText = await page.$eval('[data-m9-own]', (e) => ({ v: e.dataset.m9Own, t: e.textContent }));
check(`the M9 line reads the reports: ${own.join(',') || 'none'}`, ownText.v === (own.join(',') || 'none') &&
  (Object.keys(reports).length === 0 ? ownText.t.includes('No lab results') : own.length ? ownText.t.includes(own.map(NAME).join(' and ')) : ownText.t.includes('does not pass on any backend')), JSON.stringify(ownText));
const DIRS = { '2.3': 'ch02/lab3_elementwise', '10.3': 'ch10/lab3_softmax_argmax', '22.1': 'ch22/lab1_gemv', '22.2': 'ch22/lab2_batching', 'X2.1': 'x2/lab1_q8gemv', 'X2.2': 'x2/lab2_q4gemv', 'X2.4': 'x2/lab4_m9' };
const CH = { '2.3': 'ch02', '10.3': 'ch10', '22.1': 'ch22', '22.2': 'ch22', 'X2.1': 'x2', 'X2.2': 'x2', 'X2.4': 'x2' };
const m9 = await page.$$eval('table[data-table="m9"] tbody tr', (trs) => trs.map((tr) => ({ num: tr.dataset.m9, pass: tr.dataset.pass, href: tr.querySelector('a').getAttribute('href') })));
check('the M9 table lists the seven labs whose code runs in the step, each linked to its labs page', m9.length === 7 && m9.every((r) => DIRS[r.num] && r.href === `#/${CH[r.num]}/labs`), JSON.stringify(m9));
check("each M9 row reads the reader's own status from the reports", m9.every((r) => r.pass === (passingOn(DIRS[r.num]).join(',') || 'none')), JSON.stringify(m9.map((r) => [r.num, r.pass])));
check('the next module is named, not linked', text0.includes('module X3, Reading real backends') &&
  !(await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href')?.startsWith('#/x3')))));
check("M10 is described in X3's words, and the GGUF claim is narrowed to llama.cpp's files",
  text0.includes('generated Llama-style model run end to end') && text0.includes('GGUF files made') && !text0.includes('published checkpoint'));
check('the page names what the module leaves out and points at the facts file\'s "Not checked" list',
  text0.includes('What this module leaves out') && ['GPTQ', 'AWQ', 'Marlin', 'KV caches'].every((t) => text0.includes(t)) && notes.some((t) => t.includes('Not checked') && t.includes('x2-facts.md')));
await shot('project', 'table[data-table="m9"]');

// ---- margin notes, links, narrow screen
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: `${out}-narrow.png` });
await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// Section pages with narrated scenes ask for a manifest and clips that are not recorded yet.
const real404 = errors.filter((e) => !/narration\/(manifest\.json|[a-z-]+-\d+\.mp3)/.test(e) && !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (the missing narration files of linked sections allowed)', real404.length === 0, real404.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
