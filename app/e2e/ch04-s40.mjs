// Browser check for chapter 4, "Before you start": the chapter map, the bridge to
// framework sources, the vocabulary table across vendors, the statement of what the
// emulators cannot show, and the five prerequisite questions. The page has no
// simulator; the expected answers are recomputed here from the chapter 3 grid model
// (src/sim/grid.ts), imported directly.
// Usage: node e2e/ch04-s40.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { coverage, dim3, globalCoord, sizeGrid, workingInBlock } from '../src/sim/grid.ts';

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

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const clean = (s) => s.replace(/\s+/g, ' ').trim();
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
};
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};

await goto(page, 'ch04/start');
await page.waitForSelector('.q');

// ---------------------------------------------------------------- the map of the chapter
const prose = clean(await page.$eval('.prose', (e) => e.textContent));
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
const sections = ['4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8'];
check('the chapter map links to sections 4.1 to 4.8, the labs and review', [...sections.map((s) => '#/ch04/' + s), '#/ch04/labs', '#/review'].every((h) => links.includes(h)), links.join(' '));
const order = sections.map((s) => links.indexOf('#/ch04/' + s));
check('and names them in the order of the book\'s pages', order.every((v, i) => i === 0 || v > order[i - 1]), order.join(','));
const heads = await page.$$eval('.prose h2', (h) => h.map((x) => x.textContent));
check('the page has its five parts under sentence-case headings', ['Where this sits under a framework call', 'Three words, three vendors', 'What you will measure', 'Before you start'].every((h) => heads.includes(h)), heads.join(' | '));
check('chapters 5 and 6 are named as what comes next', prose.includes('chapter 5 is the memory on the chip') && prose.includes('chapter 6 the memory off it'));

// ---------------------------------------------------------------- the bridge
const codes = await page.$$eval('.prose code', (c) => c.map((x) => x.textContent));
check('the bridge names torch.matmul and the three constants read in the projects', ['torch.matmul', 'C10_WARP_SIZE', '#define WARP_SIZE 32', 'constexpr int SIMD_SIZE = 32'].every((c) => codes.includes(c)), codes.join(' | '));
const beyond = await page.$$eval('.note', (ns) => ns.map((n) => n.textContent.replace(/\s+/g, ' ').trim()).filter((t) => t.startsWith('Not in the book')));
check('the bridge is marked as not in the book, with its facts items', beyond.some((t) => t.includes('PyTorch, llama.cpp or MLX') && t.includes('D1, D3, D4')), beyond.join(' || '));

// ---------------------------------------------------------------- the vocabulary table
const T = 'table[aria-label="Vocabulary across vendors"]';
const head = await page.$$eval(T + ' thead th', (th) => th.map((x) => x.textContent));
const rows = await page.$$eval(T + ' tbody tr', (trs) => trs.map((tr) => [...tr.querySelectorAll('td')].map((td) => td.textContent)));
check('the table has a column per vendor', head.join('|') === 'What it is|CUDA (the book)|HIP on AMD|Metal', head.join('|'));
check('three rows: block, SM, warp', rows.length === 3 && rows[0][1] === 'block' && rows[1][1].includes('SM') && rows[2][1] === 'warp', JSON.stringify(rows.map((r) => r[1])));
check('block row: workgroup on AMD, threadgroup on Metal', rows[0][2].includes('workgroup') && rows[0][3] === 'threadgroup', rows[0].join('|'));
check('SM row: compute unit on AMD, not documented on Metal', rows[1][2] === 'compute unit (CU)' && rows[1][3] === 'not documented', rows[1].join('|'));
check('warp row: wavefront on AMD, SIMD-group on Metal', rows[2][2].includes('wavefront') && rows[2][3] === 'SIMD-group', rows[2].join('|'));
check('the table is labelled as ours, with its facts items', beyond.some((t) => t.includes('Setting the columns side by side is ours') && t.includes('B1, B2, C2')), beyond.join(' || '));
const fits = await page.$eval(T, (t) => t.getBoundingClientRect().right <= document.documentElement.clientWidth);
check('the table fits the page width', fits);
if (out) {
  await page.$eval(T, (e) => e.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: out + '-table.png' });
}

// ---------------------------------------------------------------- what the emulators cannot show
check('four labs are announced', prose.includes('Four labs go with the chapter'));
check('the page says plainly that the emulators cannot show divergence or occupancy costs', prose.includes('They do not run the threads of a warp in step and they do not wait on memory') && prose.includes('cannot show what divergence or low occupancy costs'));
check('no device number is asserted for the reader', !/your device has \d/.test(prose) && prose.includes('python3 labs/pmpp.py doctor'));

// ---------------------------------------------------------------- five prerequisite questions
const kinds = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('five prerequisite questions, labelled by the chapter they come from', kinds.join('|') === 'From chapter 3|From chapter 3|From chapter 3|From chapter 2|From chapter 1', kinds.join('|'));

// q-surplus: 1000 elements in blocks of 256.
const c = coverage(1000, 256, sizeGrid(1000, 256, 'round-up'));
let q = await question('A vector of 1000 elements');
await typeAnswer(q, c.threads);
check('surplus: the thread count is rejected and the model\'s surplus is shown', (await qText(q)).includes(`Not quite. The answer is ${c.surplus} threads.`) && c.surplus === 24, await qText(q));
await typeAnswer(q, c.surplus);
check('surplus: 24 is accepted, with 4 blocks and 1024 threads in the explanation', (await qText(q)).includes('Correct.') && (await qText(q)).includes(`= ${c.blocks} blocks`) && (await qText(q)).includes(`${c.threads} threads`), await qText(q));

// q-region4: the corner block of the 62 x 76 picture.
const cols = workingInBlock(76, 16, 4);
const rws = workingInBlock(62, 16, 3);
q = await question('bottom-right block');
await typeAnswer(q, 256);
check('corner block: a full block is rejected', (await qText(q)).includes(`Not quite. The answer is ${cols * rws} threads.`) && cols * rws === 168, await qText(q));
await typeAnswer(q, cols * rws);
check('corner block: 12 columns times 14 rows = 168 is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${cols} × ${rws} = ${cols * rws}`), await qText(q));

// q-flat: thread (x 3, y 2) of block (x 1, y 0).
const g = globalCoord(dim3(1, 0, 0), dim3(16, 16, 1), dim3(3, 2, 0));
const flat = g.y * 76 + g.x;
q = await question('What is its flat index');
await typeAnswer(q, g.x * 76 + g.y);
check('flat index: rows and columns swapped is rejected', (await qText(q)).includes(`Not quite. The answer is ${flat}.`) && flat === 171, await qText(q));
await typeAnswer(q, flat);
check('flat index: 171 is accepted, as row 2 and column 19', (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${g.y} × 76 + ${g.x} = ${flat}`), await qText(q));

// q-order and q-throughput.
q = await question('which of them runs first');
await pick(q, 'Block 0 runs first');
check('order: "index order" is marked wrong with its reason', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Nothing in the model says so'), await qText(q));
await pick(q, 'Nothing. The runtime may run them in either order, or together.');
check('order: "nothing" is accepted, sourced to chapter 2', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Source: ch. 2 §2.6, p. 39'), await qText(q));
q = await question('Why is that a sound trade for a GPU');
await pick(q, 'Simple cores run each thread faster');
check('throughput: "each thread runs faster" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Each thread runs slower'), await qText(q));
await pick(q, 'Caches and predictors mostly cut one thread');
check('throughput: the throughput answer is accepted and points at section 4.6', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Section 4.6 shows the mechanism'), await qText(q));
if (out) {
  await q.evaluate((e) => e.scrollIntoView({ block: 'end' }));
  await page.screenshot({ path: out + '-questions.png' });
}

await pause(700);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
