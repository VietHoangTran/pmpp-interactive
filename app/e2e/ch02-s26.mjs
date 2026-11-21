// Browser check for section 2.6, "Calling a kernel": the launch calculator, the bug hunt,
// the order explorer, the narrated animation, the three listings, the predictions and the
// questions. Everything asserted is computed from the model the page runs on.
// Usage: node e2e/ch02-s26.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { tabStatus as sharedTabStatus } from './labstatus.mjs';
import {
  BOOK_BLOCK,
  BOOK_NS,
  EXPRESSIONS,
  ORDERS,
  SCENE,
  ZERO_BLOCKS,
  blockCount,
  ceilDiv,
  ex9,
  launchWith,
  makeData,
  runKernel,
  schedule,
  showValue,
  smallestFailing,
  smallestWasting,
  verdict,
  waveCount,
} from '../src/sim/launch.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
// E2E_DARK=1 and E2E_WIDTH=<px> run the same checks in the dark theme or at another width.
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000, dark: process.env.E2E_DARK === '1' });
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const exists = async (sel) => (await page.$(sel)) !== null;
const num = (v) => v.toLocaleString('en-US');
/** With a screenshot prefix given: scroll the interactive into view and save a picture of it. */
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return done;
};
/** Set an input, range or select the way a person would, so React sees the change. */
const setValue = async (sel, value) => {
  await page.$eval(
    sel,
    (e, value) => {
      const proto = e.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(e, value);
      e.dispatchEvent(new Event(e.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    },
    String(value),
  );
  await pause(80);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(150);
};
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(150);
};

await goto(page, 'ch02/2.6');
const CALC = 'Launch calculator';
const HUNT = 'Find the n that breaks it';
const ORDER = 'Any order, any device';
await page.waitForSelector(bench(CALC));
await page.waitForFunction(() => document.querySelectorAll('.code pre').length >= 3, { timeout: 15000 });
await pause(300);

// ---------------------------------------------------------------- listings
const lab = (rel) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8').replace(/\n$/, '').split('\n');
const cu = lab('ch02/lab1_vecadd/solution/kernel.cu');
const mm = lab('ch02/lab1_vecadd/solution/launch.mm');
const ex9File = lab('ch02/snippets/ex9_foo.cu');
const listings = await page.$$eval('.code', (figs) =>
  figs.map((f) => ({
    tab: f.querySelector('.tabs button')?.textContent ?? '',
    status: f.querySelector('.status')?.textContent ?? '',
    lines: [...f.querySelectorAll('pre .ln')].map((l) => ({ n: Number(l.dataset.n), hl: l.classList.contains('hl'), text: l.textContent.replace(/\n$/, '') })),
  })),
);
const byTab = (t) => listings.find((l) => l.tab === t);
const reports = await (await fetch(BASE + '/api/reports')).json();
const passed = (backend) => {
  const rs = (reports[backend]?.results ?? []).filter((x) => x.lab === 'ch02/lab1_vecadd' && x.variant === 'solution' && x.tests.length);
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
};
// LabKit's solutionTabStatus, from the shared helper, with this machine's runs and its CUDA
// front-end check (GET /api/frontend-check).
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
const tabStatus = (labId, file) => sharedTabStatus(reports, labId, file, frontend);

const host = byTab('CUDA and HIP');
const firstCode = cu.findIndex((l) => !l.startsWith('//'));
check('the host function is the lab 2.1 solution, every line after its opening comment, with the file\'s line numbers',
  !!host && host.lines.length === cu.length - firstCode && host.lines.every((l, i) => l.text === cu[firstCode + i] && l.n === firstCode + i + 1),
  host ? `${host.lines.length} lines shown, first ${host.lines[0]?.n}` : 'no listing');
const launchAt = cu.findIndex((l) => l.includes('vecAddKernel<<<')) + 1;
const hostHl = host.lines.filter((l) => l.hl);
check('the launch and the two lines after it are marked', hostHl.length === 3 && hostHl[0].n === launchAt && hostHl[0].text.includes('vecAddKernel<<<numBlocks, threadsPerBlock>>>')
  && hostHl[1].text.includes('CHECK(cudaGetLastError());') && hostHl[2].text.includes('CHECK(cudaDeviceSynchronize());'), JSON.stringify(hostHl));
const wantCu = tabStatus('ch02/lab1_vecadd', 'kernel.cu');
check('its status uses the fixed vocabulary and follows this machine\'s recorded runs', host.status === wantCu, `${host.status} / expected ${wantCu}`);

const metalHost = byTab('Metal host');
const mmFrom = mm.findIndex((l) => l.startsWith('int vecAdd('));
check('the Metal listing is vecAdd from the lab 2.1 launch.mm, with the file\'s line numbers',
  !!metalHost && metalHost.lines.length === mm.length - mmFrom && metalHost.lines.every((l, i) => l.text === mm[mmFrom + i] && l.n === mmFrom + i + 1));
check('the Metal status is "verified on Metal" only when this machine recorded a passing run', metalHost.status === tabStatus('ch02/lab1_vecadd', 'launch.mm'), metalHost.status);
const needles = [['id<MTLCommandQueue> queue'], ['[queue commandBuffer]'], ['computeCommandEncoder]'], ['setComputePipelineState:'], ['setBuffer:'], ['dispatchThreads:'], ['endEncoding]', 'commandBuffer commit]'], ['waitUntilCompleted]'], ['commandBuffer.status']];
const wantLines = needles.map((ns) => ns.flatMap((x) => mm.flatMap((l, i) => (l.includes(x) ? [i + 1] : []))));
const tableLines = await page.$$eval('table[data-metal-steps] tbody td[data-lines]', (tds) => tds.map((t) => t.dataset.lines));
const queueCell = await page.$eval('table[data-metal-steps] tbody td[data-lines]', (t) => t.textContent.trim());
check('the Metal table has a row for the queue (a parameter of vecAdd) and one for each of the 8 steps, with their lines in launch.mm',
  tableLines.length === 9 && wantLines.every((w, i) => w.length > 0 && tableLines[i] === w.join(',')) && queueCell === `${wantLines[0][0]}, a parameter`,
  tableLines.join(' | ') + ' / ' + queueCell);
check('the marked lines of the Metal listing are exactly the lines the table names',
  JSON.stringify(metalHost.lines.filter((l) => l.hl).map((l) => l.n)) === JSON.stringify(wantLines.flat().sort((a, b) => a - b)));
const tableText = await text('table[data-metal-steps]');
check('the table pairs the chevrons with dispatchThreads and the synchronise with waitUntilCompleted',
  tableText.includes('<<<numBlocks, threadsPerBlock>>>') && tableText.includes('dispatchThreads:threadsPerThreadgroup:') && tableText.includes('waitUntilCompleted') && tableText.includes('setBytes:length:atIndex:'));

const e9 = byTab('CUDA');
check('exercise 9 is shown as labs/ch02/snippets/ex9_foo.cu, every line', !!e9 && e9.lines.length === ex9File.length && e9.lines.every((l, i) => l.text === ex9File[i]));
check('its index line, assignment and launch are marked, and it is front-end-checked', JSON.stringify(e9.lines.filter((l) => l.hl).map((l) => l.n)) === '[2,4,9]' && e9.status === 'front-end-checked', e9.status);

// ---------------------------------------------------------------- first prediction
const b1000 = launchWith('book-ceil', 1000, BOOK_BLOCK);
const P = await question('How many threads does the launch create?');
await (await P.$('input[type=number]')).type('1000');
await (await P.$('button.primary')).click();
await pause(150);
let t = await qText(P);
check(`the prediction reveals ${num(b1000.threads)} threads in ${b1000.blocks} blocks, ${b1000.idle} idle, from the model`,
  t.includes(`It is ${num(b1000.threads)}.`) && t.includes(`${b1000.blocks} blocks`) && t.includes(`The other ${b1000.idle} have an index`), t);
check('a prediction of 1000 is told why a grid cannot have 1000 threads', t.includes('A grid cannot have exactly 1000 threads'), t);

// ---------------------------------------------------------------- A: launch calculator
const calcRows = () =>
  page.$$eval(bench(CALC) + ' tr[data-expr]', (rows) =>
    rows.map((r) => ({ key: r.dataset.expr, blocks: Number(r.dataset.blocks), lost: Number(r.dataset.lost), empty: Number(r.dataset.empty), code: r.querySelector('code').textContent, checked: r.querySelector('input').checked })),
  );
const outText = (name) => page.$eval(`${bench(CALC)} [data-out="${name}"]`, (e) => e.textContent.trim());
const steps = () => page.$$eval(bench(CALC) + ' [data-eval] li', (ls) => ls.map((l) => l.textContent));
const selectExpr = async (key) => {
  await page.$eval(`${bench(CALC)} tr[data-expr="${key}"] input`, (e) => e.click());
  await pause(80);
};
const typeN = async (v) => setValue(bench(CALC) + ' input[aria-label="n"]', v);
/** Everything the calculator shows for one expression, against the model. */
const calcMatches = async (key, n, B) => {
  const c = blockCount(key, n, B);
  const r = launchWith(key, n, B);
  const got = {
    blocks: await outText('blocks'), needed: await outText('needed'), threads: await outText('threads'), working: await outText('working'),
    idle: await outText('idle'), lost: await outText('lost'), empty: await outText('empty'), statement: await outText('statement'),
  };
  const want = {
    blocks: num(c.blocks), needed: num(c.needed), threads: num(r.threads), working: num(r.working), idle: num(r.idle),
    lost: num(r.unprocessedCount), empty: num(r.emptyBlocks),
    statement: `vecAddKernel<<<${EXPRESSIONS.find((e) => e.key === key).source(B)}, ${B}>>>(A_d, B_d, C_d, n);`,
  };
  const st = await steps();
  const stepsOk = st.length === c.steps.length && c.steps.every((s, i) => st[i].includes(`${s.text} is ${showValue(s.value, s.type)}`) && st[i].includes(s.rule));
  return { ok: JSON.stringify(got) === JSON.stringify(want) && stepsOk, detail: JSON.stringify({ got, want, st }) };
};

let rows = await calcRows();
check('five expressions are offered, written with the block size as a literal', rows.length === 5 && EXPRESSIONS.every((e, i) => rows[i].key === e.key && rows[i].code === e.source(BOOK_BLOCK)), JSON.stringify(rows.map((r) => r.code)));
check('it opens on the integer ceiling at the book\'s n = 1000', rows[0].checked && (await page.$eval(bench(CALC) + ' input[aria-label="n"]', (e) => e.value)) === '1000');
let m = await calcMatches('int-ceil', 1000, BOOK_BLOCK);
check('n = 1000: 4 blocks, 1,024 threads, 1,000 pass the guard, 24 stopped, with the two evaluation steps', m.ok, m.detail);

const wantBook = BOOK_NS.map((n) => blockCount('book-ceil', n, BOOK_BLOCK).blocks);
const gotBook = [];
for (const n of BOOK_NS) {
  await clickText(CALC, num(n));
  rows = await calcRows();
  gotBook.push([rows.find((r) => r.key === 'int-ceil').blocks, rows.find((r) => r.key === 'book-ceil').blocks]);
}
check(`the book's four lengths give ${wantBook.join(', ')} blocks with both right forms`, gotBook.every((g, i) => g[0] === wantBook[i] && g[1] === wantBook[i]) && JSON.stringify(wantBook) === '[4,3,16,7813]', JSON.stringify(gotBook));
m = await calcMatches('int-ceil', 2000000, BOOK_BLOCK);
check('n = 2,000,000 is evaluated in full: 7,813 blocks and 128 threads stopped', m.ok && (await outText('idle')) === '128', m.detail);

await clickText(CALC, num(1000));
await selectExpr('book-ceil');
m = await calcMatches('book-ceil', 1000, BOOK_BLOCK);
check('the book\'s form: 1000 / 256.0 is 3.90625, ceil gives 4.0, converted to 4 blocks', m.ok && (await steps())[0].includes('1000 / 256.0 is 3.90625'), m.detail);
await selectExpr('ceil-trunc');
m = await calcMatches('ceil-trunc', 1000, BOOK_BLOCK);
const bad = launchWith('ceil-trunc', 1000, BOOK_BLOCK);
check('ceil(n / 256): the int division gives 3 before ceil sees it; 232 elements never processed', m.ok && bad.blocks === 3 && bad.unprocessedCount === 232 && (await steps())[1].includes('ceil(3) is 3.0'), m.detail);
t = await outText('summary');
check('the summary names the elements never written and marks the launch wrong', t.includes('wrong for this n') && t.includes('elements 768 to 999 are never written'), t);
check('the bar has a red tail and the enlarged view shows block 2 launched and block 3 never created',
  (await exists(bench(CALC) + ' [data-tail="lost"]')) &&
  JSON.stringify(await page.$$eval(bench(CALC) + ' [data-coverage] [data-block]', (b) => b.map((x) => [x.dataset.block, x.dataset.launched]))) === '[["2","yes"],["3","no"]]' &&
  (await text(bench(CALC) + ' [data-coverage]')).includes('no thread exists for elements 768 to 999'));
await shot('calc-bug', bench(CALC));

// On, just under and just over a multiple of the block size.
const around = [];
for (const n of [1024, 1025, 1023]) {
  if (n === 1024) await typeN(1024);
  else if (n === 1025) await page.$eval(bench(CALC) + ' button[aria-label="n plus 1"]', (b) => b.click());
  else { await page.$eval(bench(CALC) + ' button[aria-label="n minus 1"]', (b) => b.click()); await pause(40); await page.$eval(bench(CALC) + ' button[aria-label="n minus 1"]', (b) => b.click()); }
  await pause(80);
  rows = await calcRows();
  around.push(rows.every((r) => { const w = launchWith(r.key, n, BOOK_BLOCK); return r.blocks === w.blocks && r.lost === w.unprocessedCount && r.empty === w.emptyBlocks; }) ? rows : null);
}
check('typing 1024, then +1, then −1 twice: every row follows the model', around.every(Boolean));
check('at n = 1024 nothing is lost by any form and n / 256 + 1 has a block with no work', !!around[0] && around[0].every((r) => r.lost === 0) && around[0].find((r) => r.key === 'plus-one').empty === 1 && around[0].filter((r) => r.empty).length === 1);
check('one element past a multiple, n / 256 loses 1; one short, it loses 255', !!around[1] && around[1].find((r) => r.key === 'trunc').lost === 1 && around[2].find((r) => r.key === 'trunc').lost === 255);
await typeN(1024);
await selectExpr('plus-one');
m = await calcMatches('plus-one', 1024, BOOK_BLOCK);
t = await text(bench(CALC) + ' [data-coverage]');
check('n / 256 + 1 at 1024: 5 blocks, correct, with a whole block of 256 threads the guard stops', m.ok && (await outText('summary')).includes('correct, with waste') && t.includes('launched, no thread works') && t.includes('all 256 threads are stopped by the guard'), m.detail + t);

// Below one block: zero blocks.
await typeN(100);
await selectExpr('trunc');
m = await calcMatches('trunc', 100, BOOK_BLOCK);
const zero = await page.$$eval(bench(CALC) + ' [data-zero] tr', (rs) => rs.map((r) => [r.dataset.backend, ...[...r.querySelectorAll('td')].map((c) => c.textContent.trim())]));
check('n = 100 with n / 256: 0 blocks, all 100 elements unprocessed', m.ok && (await outText('blocks')) === '0' && (await outText('lost')) === '100', m.detail);
check('a grid of 0 blocks: HIP error documented, emulator error, CUDA not documented',
  zero.length === 3 && ZERO_BLOCKS.every((z, i) => zero[i][0] === z.backend && zero[i][1] === z.name && zero[i][2] === z.outcome && zero[i][3] === z.detail) &&
  zero.find((z) => z[0] === 'cuda')[2] === 'not documented' && zero.find((z) => z[0] === 'hip')[3] === 'hipErrorInvalidValue', JSON.stringify(zero));
await shot('calc-zero', bench(CALC));
await selectExpr('int-ceil');
check('with a right form the zero-block panel is gone', !(await exists(bench(CALC) + ' [data-zero]')));

// Other block sizes, the slider and a bad entry.
await typeN(1000);
await setValue(bench(CALC) + ' input[aria-label="threads per block"]', 128);
m = await calcMatches('int-ceil', 1000, 128);
check('threads per block 128: (n + 128 - 1) / 128 gives 8 blocks for n = 1000', m.ok && (await outText('blocks')) === '8' && (await outText('B')) === '128', m.detail);
await setValue(bench(CALC) + ' input[aria-label="n slider"]', 2048);
m = await calcMatches('int-ceil', 2048, 128);
check('dragging n to 2048 updates the number box and the result', m.ok && (await page.$eval(bench(CALC) + ' input[aria-label="n"]', (e) => e.value)) === '2048', m.detail);
await typeN('0');
check('n = 0 is refused with a message and the last valid n stays', (await text(bench(CALC) + ' [role="alert"]')).includes('Still showing n = 2,048') && (await outText('working')) === '2,048');
await setValue(bench(CALC) + ' input[aria-label="threads per block"]', BOOK_BLOCK);
await typeN(1000);
check('a valid n clears the message', !(await exists(bench(CALC) + ' [role="alert"]')));

// ---------------------------------------------------------------- B: find the n that breaks it
const card = (key) => `${bench(HUNT)} [data-hunt="${key}"]`;
const runHunt = async (key, a, b) => {
  await setValue(`${card(key)} input[aria-label="passing n for ${key}"]`, a);
  await setValue(`${card(key)} input[aria-label="smallest bad n for ${key}"]`, b);
  await page.$eval(card(key) + ' button.primary', (e) => e.click());
  await pause(120);
  return {
    solved: await page.$eval(card(key), (e) => e.dataset.solved),
    pass: await text(card(key) + ' [data-feedback="pass"]'),
    bad: await text(card(key) + ' [data-feedback="bad"]'),
  };
};
const statements = await page.$$eval(bench(HUNT) + ' [data-hunt] p code', (c) => c.map((x) => x.textContent));
check('three launches are shown as code, each with block size 256',
  JSON.stringify(statements) === JSON.stringify(['n / 256', 'ceil(n / 256)', 'n / 256 + 1'].map((e) => `vecAddKernel<<<${e}, 256>>>(A_d, B_d, C_d, n);`)), statements.join(' | '));
check('nothing is revealed before a run', !(await exists(bench(HUNT) + ' [data-feedback]')) && (await text(bench(HUNT) + ' [data-hunt-score]')).includes('0 of 3 found'));
check('the model\'s targets: n = 1 for the first two, n = 256 for the third',
  smallestFailing('trunc', 256) === 1 && smallestFailing('ceil-trunc', 256) === 1 && smallestFailing('plus-one', 256) === null && smallestWasting('plus-one', 256) === 256);

let h = await runHunt('trunc', 1000, 1000);
const v1000 = verdict('trunc', 1000, 256);
check('n / 256 at n = 1000: the "passing" guess fails, and 1000 fails but is not the smallest', h.solved === 'false' && !v1000.passes && h.pass.startsWith('fails') && h.pass.includes('elements 768 to 999 are never written') && h.bad.includes('not the smallest') && h.bad.includes('A smaller n fails too'), JSON.stringify(h));
h = await runHunt('trunc', 512, 300);
check('512 passes (a multiple of 256); 300 fails but is still not the smallest', h.solved === 'false' && h.pass.startsWith('passes') && h.pass.includes('512 is a multiple of 256') && h.bad.includes('not the smallest'), JSON.stringify(h));
h = await runHunt('trunc', 512, 512);
check('an n that passes is not accepted as a failure', h.solved === 'false' && h.bad.includes('not this one') && h.bad.includes('The test passes at this n'), JSON.stringify(h));
h = await runHunt('trunc', 512, 1);
check('n = 1 is the smallest failure: 0 blocks, nothing launched', h.solved === 'true' && h.bad.startsWith('smallest') && h.bad.includes('0 blocks, 0 threads; element 0 is never written') && h.bad.includes('1 / 256 is 0 in integers'), JSON.stringify(h));
check('the solved card shows what the launch covered', (await text(card('trunc') + ' [data-coverage]')).includes('no thread exists for element 0') && (await text(card('trunc'))).includes('found'));
h = await runHunt('ceil-trunc', 256, 1);
check('ceil(n / 256) has the same answers', h.solved === 'true' && verdict('ceil-trunc', 256, 256).passes && !verdict('ceil-trunc', 1, 256).passes, JSON.stringify(h));
h = await runHunt('plus-one', 1000, 255);
check('n / 256 + 1: every n passes, and 255 wastes nothing', h.solved === 'false' && h.pass.startsWith('passes') && h.pass.includes('So does every other n') && h.bad.includes('No block is without work at this n'), JSON.stringify(h));
h = await runHunt('plus-one', 1000, 512);
check('512 wastes a block but is not the smallest', h.solved === 'false' && verdict('plus-one', 512, 256).wastes && h.bad.includes('A smaller n wastes one too'), JSON.stringify(h));
h = await runHunt('plus-one', 1000, 256);
check('n = 256 is the smallest: 2 blocks, one with no work', h.solved === 'true' && h.bad.includes('2 blocks, 512 threads') && h.bad.includes('256 / 256 + 1 is 2'), JSON.stringify(h));
t = await text(bench(HUNT));
const at1024 = ['trunc', 'ceil-trunc', 'plus-one'].map((k) => verdict(k, 1024, 256));
check('with all three found, the lesson is drawn: a test at n = 1,024 passes for every one of them',
  t.includes('3 of 3 found') && (await exists(bench(HUNT) + ' [data-hunt-done]')) && at1024.every((v) => v.passes) &&
  t.includes(`n / 256 gives ${at1024[0].blocks} blocks`) && t.includes(`n / 256 + 1 gives ${at1024[2].blocks} blocks`) && t.includes('n = 1, 31, 32, 33, 255, 256, 257, 1000 and 4097'), t.slice(-600));
await shot('hunt', bench(HUNT));

// ---------------------------------------------------------------- questions of the first half
let q = await question('What happens?');
await pick(q, '4 blocks are launched and C is correct:');
t = await qText(q);
check('integer-division question: the tempting answer is refused with its reason', t.includes('Not quite.') && t.includes('That is what ceil(n/256.0) does'), t);
await pick(q, `${bad.blocks} blocks are launched,`);
t = await qText(q);
check('and the right one names elements 768 to 999', t.includes('Correct.') && t.includes('elements 768 to 999') && t.includes('A test with n = 1024 would pass'), t);
q = await question('A vector of 8000 elements');
await typeAnswer(q, launchWith('int-ceil', 8000, 1024).threads);
check('book exercise 4: 8192 threads', (await qText(q)).includes('Correct.') && launchWith('int-ceil', 8000, 1024).threads === 8192);
q = await question('A launch asks for 4096 threads per block');
await pick(q, 'Nothing more.');
check('launch-check question: relying on the next cudaMemcpy is refused, and a refused launch may never be reported', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('may never be reported at all') && (await qText(q)).includes('lab 2.4'));
await pick(q, 'CHECK(cudaGetLastError());');
check('cudaGetLastError immediately after the launch is accepted', (await qText(q)).includes('Correct.'));

// ---------------------------------------------------------------- the narrated animation
const SC = 'section.scene';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const scripts = [];
for (let k = 1; k <= 7; k++) scripts.push(strip(fs.readFileSync(path.join(repo, 'content/ch02/narration', `one-launch-${k}.txt`), 'utf8')));
const manifest = await fetch(BASE + '/content/ch02/narration/manifest.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), SC, label);
  await pause(150);
};
const seek = async (v) => {
  await page.$eval(SC + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const drawn = () => page.$$eval(SC + ' .stage svg text', (ts) => ts.map((x) => x.textContent.trim()).filter(Boolean));
const has = (list, s) => list.some((x) => x.includes(s));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), SC, word);
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
check('the animation is titled "One launch, start to finish" and has seven parts', (await page.$eval(SC, (e) => e.getAttribute('aria-label'))) === 'One launch, start to finish' && (await page.$$(SC + ' .seg button')).length === 7);
check('seven scripts, each 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(' '));
check('it has a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const svgW = await page.$eval(SC + ' .stage svg', (s) => s.getBoundingClientRect().width);
check('the drawing keeps a legible width (600 px or more, scrolling sideways in its frame on a phone)', svgW >= 600, String(svgW));
await (await box('captions')).click();
await pause(150);
const sc = SCENE;
const count = blockCount('int-ceil', sc.n, sc.B);
const good = launchWith('int-ceil', sc.n, sc.B);
const sched = schedule(count.blocks, sc.S, 'shuffled', sc.seed);
const badScene = launchWith('trunc', sc.n, sc.B);
const captionsOk = [];
const LABELS = ['The statement', 'It returns', 'The grid', 'Two at a time', 'The guard', 'Only now', 'The bug'];
const frames = [];
for (let k = 0; k < 7; k++) {
  await press(`${k + 1}. ${LABELS[k]}`);
  captionsOk.push((await page.$eval(SC + ' .caption', (e) => e.textContent.trim())) === scripts[k]);
  await seek(1000);
  frames.push(await drawn());
  await shot(`scene-${k + 1}`, SC);
}
check('each part shows its own script as the caption', captionsOk.every(Boolean), JSON.stringify(captionsOk));
check('part 1: the configuration is evaluated first, 22 + 8 - 1 = 29 and 29 / 8 = 3, then the arguments',
  has(frames[0], `${count.steps[0].text} = ${count.steps[0].value}`) && has(frames[0], `${count.steps[1].text} = ${count.steps[1].value}`) && has(frames[0], '2. the arguments') && has(frames[0], `${count.blocks} blocks of ${sc.B} threads`), frames[0].join('|'));
check('part 2: the statement has returned and no thread has run', has(frames[1], 'the statement has returned') && has(frames[1], 'No thread has run yet.') && has(frames[1], 'configuration accepted'), frames[1].join('|'));
check(`part 3: ${good.blocks} blocks × ${sc.B} threads = ${good.threads} threads`, has(frames[2], `${good.blocks} blocks × ${sc.B} threads = ${good.threads} threads`) && has(frames[2], 'in the queue'), frames[2].join('|'));
check(`part 4: the device takes block ${sched.waves[0][0]} and block ${sched.waves[0][1]} first`, has(frames[3], `took block ${sched.waves[0][0]} and block ${sched.waves[0][1]}`) && JSON.stringify(sched.waves) === '[[2,0],[1]]', frames[3].join('|'));
check('part 5: the threads with i = 22 and 23 stop, then block 1 runs', has(frames[4], 'threads with i = 22 and 23 fail the test and stop') && has(frames[4], `took block ${sched.waves[1][0]}`), frames[4].join('|'));
check('part 6: the blocking call returns only now, and every order gives one result', has(frames[5], 'the blocking call returns now') && has(frames[5], 'different results: 1'), frames[5].join('|'));
check(`part 7: n / 8 gives ${badScene.blocks} blocks; block 2 is never created and elements 16 to 21 are never written`,
  has(frames[6], `${badScene.blocks} blocks, ${badScene.threads} threads, ${badScene.unprocessedCount} elements never written`) && has(frames[6], 'never created') && has(frames[6], 'elements 16 to 21: no thread exists for them'), frames[6].join('|'));
if (manifest) check('the manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(SC + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
// Runs by itself with the sound off.
const soundBox = await box('sound');
if (await soundBox.evaluate((e) => e.checked)) await soundBox.click();
await press('1. The statement');
await press('Play');
await pause(1500);
const moved = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
await press('Pause');
check('pressing Play moves the animation', moved > 20, String(moved));
await (await box('captions')).click();

// ---------------------------------------------------------------- C: any order, any device
const o = (name) => page.$eval(`${bench(ORDER)} [data-out="${name}"]`, (e) => e.textContent.trim());
const cells = () => page.$$eval(bench(ORDER) + ' [data-c]', (cs) => cs.map((c) => ({ i: Number(c.dataset.c), state: c.dataset.state, text: c.textContent.trim() })).sort((a, b) => a.i - b.i));
const setOrder = async (name) => { await page.$eval(`${bench(ORDER)} button[data-order="${name}"]`, (b) => b.click()); await pause(80); };
const setKernel = async (kind) => { await page.$eval(`${bench(ORDER)} button[data-kernel="${kind}"]`, (b) => b.click()); await pause(80); };
const setS = (S) => setValue(bench(ORDER) + ' input[aria-label="S"]', S);
const orderText = (s) => s.waves.map((w) => w.join(' ')).join('  |  ');
/** Run the current setting to the end and compare everything shown with the model. */
const runAndCompare = async (kind, n, B, S, order, seed) => {
  const blocks = ceilDiv(n, B);
  const s = schedule(blocks, S, order, seed);
  const run = runKernel(kind, s, makeData(n, B));
  await clickText(ORDER, 'Run to the end');
  const cs = await cells();
  const verdictText = await o('verdict');
  const ok =
    (await o('blocks')) === String(blocks) && (await o('waves')) === `${waveCount(blocks, S)} wave${waveCount(blocks, S) === 1 ? '' : 's'}` &&
    (await o('order')) === orderText(s).trim() && (await o('progress')) === `wave ${s.waves.length} of ${s.waves.length}` &&
    cs.length === n && cs.every((c) => c.text === String(run.final[c.i]) && c.state === (run.wrong.includes(c.i) ? 'wrong' : 'written')) &&
    (run.matches ? verdictText === "C equals the sequential loop's result" : verdictText === `${run.wrong.length} of ${n} elements differ from the sequential loop`);
  return { ok, run, detail: JSON.stringify({ verdictText, order: await o('order'), want: orderText(s), wrong: run.wrong.length }) };
};

await page.$eval(bench(ORDER), (e) => e.scrollIntoView({ block: 'start' }));
check('it opens on the animation\'s launch size and kernel: n = 22, 2 blocks at once, shuffled', (await o('n')) === '22' && (await o('S')) === '2' && (await o('progress')) === 'wave 0 of 3');
check('before anything runs every element of C is unwritten', (await cells()).every((c) => c.state === 'unwritten' && c.text === '·') && (await text(bench(ORDER) + ' [data-out="status"]')).includes('Nothing has run yet.'));
// One wave at a time.
const s0 = schedule(6, 2, 'shuffled', SCENE.seed);
const run0 = runKernel('independent', s0, makeData(22, 4));
await clickText(ORDER, 'Next wave');
let cs = await cells();
check(`after one wave only blocks ${s0.waves[0].join(' and ')} have written, in whole blocks`,
  cs.every((c) => (run0.written[0].includes(c.i) ? c.state === 'written' && c.text === String(run0.after[0][c.i]) : c.state === 'unwritten')) && (await text(bench(ORDER) + ' [data-out="status"]')).includes(`Wave 1: blocks ${s0.waves[0].join(' and ')} ran together.`));
await shot('order-wave1', bench(ORDER));
let r = await runAndCompare('independent', 22, 4, 2, 'shuffled', SCENE.seed);
check('shuffled, 2 at a time: C equals the sequential result and every cell shows the model\'s sum', r.ok && r.run.matches, r.detail);
check('the loop row shows what a sequential loop gives', JSON.stringify(await page.$$eval(bench(ORDER) + ' [data-want]', (w) => w.map((x) => [Number(x.dataset.want), x.textContent.trim()]).sort((a, b) => a[0] - b[0]).map((x) => x[1]))) === JSON.stringify(r.run.expected.map(String)));
await clickText(ORDER, 'Reset');
check('Reset empties C again', (await cells()).every((c) => c.state === 'unwritten') && (await o('progress')) === 'wave 0 of 3');

const independent = [];
for (const [order, S] of [['descending', 1], ['ascending', 8], ['descending', 3], ['shuffled', 5]]) {
  await setOrder(order);
  await setS(S);
  r = await runAndCompare('independent', 22, 4, S, order, SCENE.seed);
  independent.push(r.ok && r.run.matches && (await o('waves')) === `${waveCount(6, S)} wave${waveCount(6, S) === 1 ? '' : 's'}`);
}
check('first kernel: descending one at a time (6 waves), ascending 8 at once (1 wave), and two more settings all give the same C', independent.every(Boolean), JSON.stringify(independent));
await clickText(ORDER, 'Reshuffle');
r = await runAndCompare('independent', 22, 4, 5, 'shuffled', SCENE.seed + 1);
check('Reshuffle draws a new order, the one the model gives for the next seed', r.ok && orderText(schedule(6, 5, 'shuffled', SCENE.seed + 1)) !== orderText(schedule(6, 5, 'shuffled', SCENE.seed)), r.detail);

// The kernel that reads its neighbour's result.
await setKernel('chained');
await setOrder('ascending');
await setS(1);
r = await runAndCompare('chained', 22, 4, 1, 'ascending', SCENE.seed + 1);
check('second kernel, ascending and one block at a time: equals the loop (a running sum)', r.ok && r.run.matches, r.detail);
await setS(2);
r = await runAndCompare('chained', 22, 4, 2, 'ascending', SCENE.seed + 1);
check('ascending but two at a time: wrong elements are marked and counted', r.ok && !r.run.matches && (await cells()).filter((c) => c.state === 'wrong').length === r.run.wrong.length && r.run.wrong.length > 0, r.detail);
await shot('order-chained', bench(ORDER));
await setOrder('descending');
await setS(1);
r = await runAndCompare('chained', 22, 4, 1, 'descending', SCENE.seed + 1);
check('descending, one at a time: also wrong', r.ok && !r.run.matches, r.detail);
// Every setting at once.
const grid = (kind, n, B, seed) => ORDERS.map((ord) => Array.from({ length: 8 }, (_, i) => (runKernel(kind, schedule(ceilDiv(n, B), i + 1, ord, seed), makeData(n, B)).matches ? 'same' : 'differs')));
await clickText(ORDER, 'Try all 24 settings');
let shown = await page.$$eval(bench(ORDER) + ' table[data-all] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)));
let want = grid('chained', 22, 4, SCENE.seed + 1);
check('all 24 settings for the second kernel: only ascending with S = 1 is the same', JSON.stringify(shown) === JSON.stringify(want) && want.flat().filter((x) => x === 'same').length === 1 && want[0][0] === 'same' && (await o('all-summary')).startsWith('1 of 24'), JSON.stringify(shown));
await setKernel('independent');
shown = await page.$$eval(bench(ORDER) + ' table[data-all] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)));
want = grid('independent', 22, 4, SCENE.seed + 1);
check('all 24 settings for the first kernel: every one the same', JSON.stringify(shown) === JSON.stringify(want) && want.flat().every((x) => x === 'same') && (await o('all-summary')).startsWith('24 of 24'), JSON.stringify(shown));
// Other sizes, and Play.
await setValue(bench(ORDER) + ' select[aria-label="threads per block"]', 2);
await setValue(bench(ORDER) + ' input[aria-label="n"]', 24);
await setOrder('descending');
await setS(4);
check('blocks of 2 and n = 24: 12 blocks, 3 waves of 4', (await o('blocks')) === '12' && (await o('waves')) === '3 waves' && ceilDiv(24, 2) === 12);
await clickText(ORDER, 'Play');
await pause(3 * 650 + 900);
cs = await cells();
const runP = runKernel('independent', schedule(12, 4, 'descending', SCENE.seed + 1), makeData(24, 2));
check('Play runs the waves by itself to the end', (await o('progress')) === 'wave 3 of 3' && cs.every((c) => c.text === String(runP.final[c.i])) && (await o('verdict')) === "C equals the sequential loop's result");
await setValue(bench(ORDER) + ' select[aria-label="threads per block"]', 3);
await setValue(bench(ORDER) + ' input[aria-label="n"]', 32);
check('blocks of 3 and n = 32: 11 blocks, the last with 1 thread stopped by the guard', (await o('blocks')) === '11' && (await text(bench(ORDER) + ' [data-strip]')).includes('1 thread stopped by the guard') && ceilDiv(32, 3) === 11);

// ---------------------------------------------------------------- questions of the second half
q = await question('Which kernel body gives the same result whatever order');
await pick(q, 'Both.');
check('order question: "the runtime runs blocks in index order" is refused', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('No such order is promised'));
await pick(q, 'C[i] = A[i] + B[i]');
check('order question: the independent body is accepted', (await qText(q)).includes('Correct.'));
q = await question('executes four times as many blocks at once');
await pick(q, 'Nothing.');
check('scalability question: nothing has to change', (await qText(q)).includes('Correct.'));
q = await question('What did the first timer measure?');
await pick(q, 'The kernel.');
check('timer question: "the kernel" is refused', (await qText(q)).includes('Not quite.'));
await pick(q, 'Handing the launch to the runtime.');
check('timer question: the launch, not the kernel', (await qText(q)).includes('Correct.') && (await qText(q)).includes('asynchronous with respect to the host thread'));

// ---------------------------------------------------------------- book exercise 9
const e = ex9();
const predict = async (part, value) => {
  const p = await question(part);
  await (await p.$('input[type=number]')).type(String(value));
  await (await p.$('button.primary')).click();
  await pause(150);
  return qText(p);
};
t = await predict('Part (c)', e.blocks);
check(`exercise 9 (c): ${num(e.blocks)} blocks, with the division shown`, e.blocks === 1563 && t.includes('That is it.') && t.includes('200000 + 128 - 1 = 200,127') && t.includes('1,563'), t);
t = await predict('Part (b)', e.N);
check(`exercise 9 (b): a prediction of ${num(e.N)} is corrected to ${num(e.threads)}`, e.threads === 200064 && t.includes(`It is ${num(e.threads)}.`) && t.includes('Not 200,000'), t);
t = await predict('Part (d)', e.line2);
check(`exercise 9 (d): ${num(e.line2)} threads compute an index`, e.line2 === 200064 && t.includes('That is it.'), t);
t = await predict('Part (e)', e.line2);
check(`exercise 9 (e): ${num(e.line4)} threads reach the assignment, the other ${e.threads - e.line4} stop`, e.line4 === 200000 && t.includes(`It is ${num(e.line4)}.`) && t.includes(`The other ${e.threads - e.line4}, all in the last block`), t);

// ---------------------------------------------------------------- the page as a whole
const page_ = await text('.page');
check('the page states its sources: book pages and the facts items', page_.includes('§2.6, pp. 37–39') && page_.includes('notes/today/ch02-facts.md, A6') && page_.includes('notes/today/ch02-facts.md, A18'));
check('each simulation says what it leaves out', (await page.$$('.note.caveat')).length === 3);
const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('the page itself never scrolls sideways at this width', sideways <= 0, `${sideways} px too wide`);
check('answers and predictions were sent to the (intercepted) progress API', puts > 0, String(puts));
// A missing narration manifest (audio is generated centrally, later) is the only 404 allowed.
const expected404 = (u) => /\/content\/ch02\/narration\/(manifest\.json|one-launch-\d\.mp3)$/.test(u);
const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
