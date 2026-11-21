// Browser check for module X3's summary: the carry-forward list, the mixed check (four
// questions, the choices answered wrong then right, the number once, with the permutation
// recomputed here from llama.cpp's rule), the map of the module's tasks with links that open
// written pages, where the sources disagree, and the M10 status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/x3-wrap.mjs [screenshot-prefix] [--dark]
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
// Options may be shown in another order than they are written: pick by the start of the label.
const pick = async (q, start) => {
  const found = await q.evaluate((e, start) => {
    const label = [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(start));
    if (!label) return false;
    label.querySelector('input[type=radio]').click();
    return true;
  }, start);
  if (!found) throw new Error('no option starting ' + start);
  await pause(40);
  await press(q, 'Check');
  await pause(120);
};

await goto(page, 'x3/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('[data-m10-own="loading"]'), { timeout: 30000 });
const body = () => page.$eval('.prose', (e) => e.textContent);

// llama.cpp's permute, as conversion/llama.py writes it (x3-facts B6): reshape the rows of each
// head to (2, half), swap the axes, flatten. Applied to K with n_head_kv.
const permuteRow = (row, rows, nHeadKv) => {
  const per = rows / nHeadKv;
  const half = per / 2;
  const head = Math.floor(row / per);
  const r = row % per;
  const k = Math.floor(r / half);
  const j = r % half;
  return head * per + 2 * j + k;
};
const target = permuteRow(100, 192, 3);
const partner = permuteRow(68, 192, 3);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary"', h1.trim().endsWith('Summary'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The module's exercises", 'Where the sources disagree', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('.prose > ul', (uls) => [...uls[0].children].map((li) => li.textContent));
check('the carry-forward list has six points, from the one place an op becomes a call to the margin rule', carry.length === 6 &&
  carry[0].includes('one place where an op becomes a backend call') && carry[1].includes('follow the string') && carry[3].includes('The shape picks the kernel') &&
  carry[5].includes('top-two margin'), JSON.stringify(carry.map((c) => c.slice(0, 30))));
const notes = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.textContent));
check("the module is labelled as ours, dated to the facts pass", notes.some((t) => t.includes("Ours, not the book's") && t.includes('2026-11-20')));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the page', !/\d+(\.\d+)? (GB\/s|KiB|GiB)\b|\d+\.\d+ ms\b|tokens\/s/.test(text0));

// ---- the four questions
check('four questions, none answered at the start', (await page.$$('.q')).length === 4 && (await page.$$('.q .verdict')).length === 0);
let q = await question("SmolLM2-135M's k_proj.weight");
check(`permute: the answer the page holds is the rule's, row ${target}`, target === 73);
await typeAnswer(q, target);
check(`permute: ${target} is right; the partner row 68 goes to ${partner}; 9 heads would not reshape`, (await qText(q)).includes('Correct.') &&
  (await qText(q)).includes(`64 + 9 = ${target}`) && (await qText(q)).includes(`goes to row ${partner}`) && (await qText(q)).includes('192 / 9 / 2 is not a whole number'), await qText(q));
q = await question('one of 16 sequences in a batch');
await pick(q, 'In a batched decode');
check('batch flip: a leaking cache is wrong, it would change every step', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('at every step'));
await pick(q, 'Greedy decoding still draws');
check('batch flip: greedy draws no random number', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('no random number is drawn'));
await pick(q, 'At batch 16');
check('batch flip: another kernel, another order, a close step; up to 8 tokens', (await qText(q)).includes('Correct.') && (await qText(q)).includes('vector kernels up to 8') && (await qText(q)).includes('not associative'));
q = await question('whose output is exactly 0 in almost every element');
await pick(q, 'The tool compares');
check('nmse: the tool compares with the CPU by default', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('CPU backend'));
await pick(q, 'The 1e-7 threshold');
check('nmse: 1e-7 is the per-op default', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('per-op default'));
await pick(q, 'NMSE divides');
check("nmse: the denominator is the reference's own size", (await qText(q)).includes('Correct.') && (await qText(q)).includes('divides by the sum of a²'));
q = await question('torch.nn.functional.rms_norm');
await pick(q, 'Whether the model is really on CUDA');
check('rms dtype: the fused kernel is registered for CUDA too', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('CUDA: _fused_rms_norm_cuda'));
await pick(q, 'Whether the rows are longer');
check("rms dtype: 2,048 is softmax's limit", (await qText(q)).includes('Not quite.') && (await qText(q)).includes('softmax'));
await pick(q, 'The dtypes');
check('rms dtype: the warning, and the composite of ordinary ops', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Cannot dispatch to fused') && (await qText(q)).includes('composite'));
await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-permute-row', 'q-batch-flip', 'q-nmse-zero', 'q-rms-dtype'].map((s) => 'x3/wrap/' + s);
check('all four answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 4);
check('the running count of answers appears after answering', (await body()).includes('So far in this module you have answered'));
// The correct option should not give itself away by length.
const longest = await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelectorAll('input[type=radio]').length).map((q) => {
  const ls = [...q.querySelectorAll('label')].map((l) => l.textContent.trim().length);
  return ls;
}));
console.log('note  option lengths per choice question: ' + JSON.stringify(longest));
await shot('questions', '.q');

// ---- the module's tasks
const exRows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ task: tr.children[0].textContent.trim(), lab: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the task table says plainly the book has none, and has eleven rows', text0.includes('The book has no exercises for this module') && exRows.length === 11, String(exRows.length));
const sections = ['X3.1', 'X3.2', 'X3.3', 'X3.4', 'X3.5', 'X3.6', 'X3.7', 'X3.8', 'X3.9'];
check('every section of the module is linked from the task table', sections.every((s) => exRows.some((r) => r.links.includes(`#/x3/${s}`))), JSON.stringify(exRows.map((r) => r.links)));
check('both labs are named in the task table', ['lab X3.1', 'lab X3.2'].every((l) => exRows.some((r) => r.lab.includes(l))));
check('the plan goal and the real-checkpoint gap are stated', text0.includes('find the kernel for a given op in each of three real codebases') && exRows.at(-1).lab.includes('unverified'));

// ---- where the sources disagree
const disagree = await page.$$eval('.prose > ul', (uls) => [...uls[1].children].map((li) => li.querySelector('b')?.textContent));
check('the disagreements are grouped', JSON.stringify(disagree) === JSON.stringify(['Where the kernel is.', 'One project, two backends.', 'One kernel per op.', 'RoPE.', 'Ties and AMD.', 'Formats.']), JSON.stringify(disagree));
check('the page says the module has no PDF errata, and where our own plan and design were wrong', text0.includes('This module has no PDF to be wrong') &&
  text0.includes('where our own plan and design were wrong') && text0.includes('small real checkpoint'));
check('the disagreements are dated and sourced to the facts file', notes.some((t) => t.includes('Checked 2026-11-20 in the facts pass (notes/today/x3-facts.md')));

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
const own = passingOn('x3/lab2_tinyllama');
const ownText = await page.$eval('[data-m10-own]', (e) => ({ v: e.dataset.m10Own, t: e.textContent }));
check(`the M10 line reads the reports: ${own.join(',') || 'none'}`, ownText.v === (own.join(',') || 'none') &&
  (Object.keys(reports).length === 0 ? ownText.t.includes('No lab results') : own.length ? ownText.t.includes(own.map(NAME).join(' and ')) : ownText.t.includes('does not pass on any backend')), JSON.stringify(ownText));
// The pieces lab X3.2's provided/pieces.cmake copies: every row's lab directory must exist.
const DIRS = { '2.3': 'ch02/lab3_elementwise', '6.4': 'ch06/lab4_linear', '10.3': 'ch10/lab3_softmax_argmax', '20.4': 'ch20/lab4_decode', '22.2': 'ch22/lab2_batching',
  'X3.1': 'x3/lab1_llama_pieces', '15.6': 'ch15/lab6_linear', '20.5': 'ch20/lab5_gqa_decode', '22.1': 'ch22/lab1_gemv', 'X1.3': 'x1/lab3_sampler', 'X2.4': 'x2/lab4_m9', 'X3.2': 'x3/lab2_tinyllama' };
const readme = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('x3/lab2_tinyllama/README.md')).then((r) => (r.ok ? r.text() : '')));
const m10 = await page.$$eval('table[data-table="m10"] tbody tr', (trs) => trs.map((tr) => ({ num: tr.dataset.m10, pass: tr.dataset.pass, href: tr.querySelector('a').getAttribute('href'), need: tr.children[2].textContent })));
check('the M10 table has the twelve labs, required ones first, each linked to its labs page', m10.length === 12 && m10.every((r) => DIRS[r.num] && r.href === `#/${DIRS[r.num].split('/')[0]}/labs`) &&
  m10.filter((r) => r.need === 'required').length === 6, JSON.stringify(m10));
{
  const flat = readme.replace(/\s+/g, ' ');
  const req = m10.filter((r) => r.need === 'required').map((r) => r.num);
  const opt = m10.filter((r) => r.need === 'if present').map((r) => r.num);
  const list = (ns) => ns.slice(0, -1).join(', ') + ' and ' + ns.at(-1);
  check(`the required (${req.join(', ')}) and optional (${opt.join(', ')}) pieces are the ones lab X3.2's README names`,
    flat.includes(`labs ${list(req)} (required) and ${list(opt)} (if present)`), flat.slice(flat.indexOf('copies the reference'), flat.indexOf('copies the reference') + 160));
}
check("each M10 row reads the reader's own status from the reports", m10.every((r) => r.pass === (passingOn(DIRS[r.num]).join(',') || 'none')), JSON.stringify(m10.map((r) => [r.num, r.pass])));
check('M10 says it runs with M9 through an adapter, and how that is checked', 
  text0.includes("with M9's quantised layer in its linear-layer slot") && text0.includes('provided adapter') &&
  text0.includes('same quantised weights') && !text0.includes('do not meet yet'));
check("chapter 24's summary page is named with its pages", text0.includes("chapter 24, the book's closing chapter (pp. 577–580)") &&
  (await page.$('a[href$="ch24/wrap"]')) !== null);
await shot('project', 'table[data-table="m10"]');

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

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
