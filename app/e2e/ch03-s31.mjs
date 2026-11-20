// Browser check for section 3.1: the sizing explorer, the grid explorer, the find-the-thread drill,
// the prediction, the tables computed from the model, and the questions.
// Usage: node e2e/ch03-s31.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import {
  BOOK_RULE,
  CUDA_DOCUMENTED,
  FIG31,
  FORMULAS,
  blockCount,
  checkLaunch,
  coverage,
  describeViolation,
  dim3,
  drillTargets,
  formatBlockLabel,
  formatDim3,
  formatThreadLabel,
  linearThread,
  linearThreadWorking,
  sizeGrid,
  sizeGridMinusOne,
  threadFromLabel,
  threadsPerBlock,
  totalThreads,
} from '../src/sim/grid.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const exists = async (sel) => (await page.$(sel)) !== null;
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
  await pause(120);
};
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch03/3.1');
const G = 'Grid explorer';
const F = 'Find the thread';
const Z = 'Four ways to size a grid';
await page.waitForSelector(bench(G));
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].filter((p) => !p.textContent.includes('loading')).length >= 4);

// ---------------------------------------------------------------- listings
const listings = await page.$$eval('.code pre', (ps) => ps.map((p) => p.textContent));
for (const name of ['launch_1d.cu', 'launch_fig31.cu', 'launch_checked.cu']) {
  const file = fs.readFileSync(path.join(repo, 'labs/ch03/snippets', name), 'utf8').trimEnd().split('\n');
  check(`a listing shows labs/ch03/snippets/${name}, every line`, listings.some((l) => file.every((ln) => l.includes(ln))));
}
check('the llama.cpp quote is shown with its two caps', listings.some((l) => l.includes('std::min(ne1, 65535), std::min(ne2 * ne3, 65535)')));
const statuses = await page.$$eval('.code .status', (s) => s.map((x) => x.textContent));
check('our three listings are marked front-end-checked and the quote carries no status', statuses.length === 3 && statuses.every((s) => s === 'front-end-checked'), statuses.join('|'));

// ---------------------------------------------------------------- prediction
const total = Number(totalThreads(FIG31.grid, FIG31.block));
const P = await question('How many threads does the launch on lines 15 and 16 create?');
await (await P.$('input[type=number]')).type(String(threadsPerBlock(FIG31.block)));
await (await P.$('button.primary')).click();
await pause(150);
let t = await qText(P);
check(`prediction reveals ${total} threads, from the model`, t.includes(`${total}.`) && t.includes(`= ${blockCount(FIG31.grid)} blocks`) && t.includes(`${threadsPerBlock(FIG31.block)} threads`), t);
check('prediction of one block gets the matching feedback', t.includes('Sixteen is one block.'), t);

// ---------------------------------------------------------------- four ways to size a grid
const sizingRows = () =>
  page.$$eval(bench(Z) + ' tbody tr', (rows) =>
    rows.map((r) => ({
      key: r.dataset.formula,
      blocks: Number(r.dataset.blocks),
      threads: Number(r.dataset.threads),
      surplus: Number(r.dataset.surplus),
      uncovered: Number(r.dataset.uncovered),
      idle: Number(r.dataset.idle),
      cells: [...r.querySelectorAll('td.num')].map((c) => c.textContent),
    })),
  );
const sizingMatches = async (nn, b) => {
  const rows = await sizingRows();
  const bad = [];
  FORMULAS.forEach((f, i) => {
    const c = coverage(nn, b, sizeGrid(nn, b, f.key));
    const r = rows[i];
    if (!r || r.key !== f.key || r.blocks !== c.blocks || r.threads !== c.threads || r.surplus !== c.surplus || r.uncovered !== c.uncovered || r.idle !== c.idleBlocks)
      bad.push(`${f.key}: page ${JSON.stringify(r)} model ${JSON.stringify(c)}`);
    else if (r.cells.join('|') !== [c.blocks, c.threads, c.surplus, c.uncovered, c.idleBlocks].map(n).join('|')) bad.push(`${f.key}: cells ${r.cells.join('|')}`);
  });
  return bad;
};
let bad = await sizingMatches(1000, 256);
check("sizing: opens on the book's n = 1000, b = 256 and every row matches the model", bad.length === 0, bad.join(' ; '));
let rows = await sizingRows();
check("sizing: the book's 4 blocks for n = 1000 (p. 46), and truncation's 3", rows.map((r) => r.blocks).join() === '3,4,4,4', rows.map((r) => r.blocks).join());
check('sizing: truncation leaves 232 elements, marked as a failure', (await page.$$eval(bench(Z) + ' tr[data-formula="truncate"] .chip.fail', (c) => c.map((x) => x.textContent))).join() === '232');
check('sizing: the summary names the remainder', (await text(bench(Z) + ' [data-out="sizing-summary"]')).includes('256 does not divide 1,000: 232 elements are left over.'));
await clickText(Z, 'n = 4,000, b = 256');
rows = await sizingRows();
bad = await sizingMatches(4000, 256);
check("sizing: preset n = 4000 gives the book's 16 blocks", rows.map((r) => r.blocks).join() === '15,16,16,16' && bad.length === 0, rows.map((r) => r.blocks).join() + ' ' + bad.join(' ; '));
// b divides n: the discovery the task asks for.
await setValue(bench(Z) + ' input[aria-label="n"]', 1024);
bad = await sizingMatches(1024, 256);
rows = await sizingRows();
check('sizing: at n = 1024 truncation is right and n / b + 1 has a block with no work', bad.length === 0 && rows[0].uncovered === 0 && rows[1].idle === 1 && rows[1].blocks === 5 && rows[2].idle === 0, JSON.stringify(rows));
check('sizing: the summary says so', (await text(bench(Z) + ' [data-out="sizing-summary"]')).includes('256 divides 1,024 exactly.'));
check('sizing: no failure chip when everything is covered', !(await exists(bench(Z) + ' .chip.fail')));
await page.click(bench(Z) + ' button[aria-label="n plus 1"]');
await pause(80);
bad = await sizingMatches(1025, 256);
rows = await sizingRows();
check('sizing: one element more and truncation drops exactly that element', bad.length === 0 && rows[0].uncovered === 1 && rows[1].idle === 0 && rows[2].surplus === 255, JSON.stringify(rows));
await setValue(bench(Z) + ' input[aria-label="b"]', 96);
await setValue(bench(Z) + ' input[aria-label="n"]', 4999);
bad = await sizingMatches(4999, 96);
rows = await sizingRows();
check('sizing: another point (n = 4999, b = 96) matches the model, and the last two formulas agree', bad.length === 0 && rows[2].blocks === rows[3].blocks, bad.join(' ; '));
const segments = await page.$$eval(bench(Z) + ' tr[data-formula="round-up"] td:last-child > div > div', (d) => d.length);
check('sizing: the bar draws one segment per block and the n marker', segments === rows[2].blocks + 1, `${segments} vs ${rows[2].blocks}`);
if (out) await page.screenshot({ path: out + '-sizing.png' });

// ---------------------------------------------------------------- grid explorer
const outv = (name) => page.$eval(`${bench(G)} [data-out="${name}"]`, (e) => e.dataset.value);
const verdict = () => page.$eval(`${bench(G)} [data-out="verdict"]`, (e) => ({ legal: e.dataset.legal, text: e.textContent }));
const violations = () => page.$$eval(`${bench(G)} [data-out="violation"]`, (v) => v.map((x) => x.textContent));
const preset = async (key) => {
  await page.click(`${bench(G)} button[data-preset="${key}"]`);
  await pause(80);
};
const judgeBy = (key) => setValue(`${bench(G)} select[aria-label="Judge the launch by"]`, key);
const setDim = (name, axis, v) => setValue(`${bench(G)} input[aria-label="${name}.${axis}"]`, v);

check('explorer: opens on Fig. 3.1 with 4 blocks of 16 threads, 64 in all', (await outv('blocks')) === String(blockCount(FIG31.grid)) && (await outv('per-block')) === String(threadsPerBlock(FIG31.block)) && (await outv('threads')) === String(total));
check('explorer: the launch statement is written in constructor order', (await text(bench(G) + ' [data-out="statement"]')).includes('dim3 dimGrid(2, 2, 1);\ndim3 dimBlock(4, 2, 2);'));
const blockNames = await page.$$eval(bench(G) + ' button[data-block]', (b) => b.map((x) => x.textContent));
check("explorer: four blocks, named (y,x) as the book names them", blockNames.join(' ') === '(0,0) (0,1) (1,0) (1,1)', blockNames.join(' '));
check('explorer: the block at x 0, y 1 is selected and titled Block (1,0)', (await page.$eval(bench(G) + ' button[data-block="0-1-0"]', (b) => b.getAttribute('aria-pressed'))) === 'true' && (await text(bench(G) + ' [data-out="block-title"]')).includes('Block (1,0).') && (await text(bench(G) + ' [data-out="block-title"]')).includes('(0, 1, 0)'));
const numbers = await page.$$eval(bench(G) + ' button[data-thread]', (b) => b.map((x) => [x.dataset.thread, x.textContent]));
check(
  'explorer: 16 thread buttons, each showing its number with x fastest (model)',
  numbers.length === 16 && numbers.every(([k, v]) => { const [x, y, z] = k.split('-').map(Number); return Number(v) === linearThread({ x, y, z }, FIG31.block); }),
  JSON.stringify(numbers),
);
// The book's own example: Thread (1,0,2) of Block (1,0).
const bt = threadFromLabel([1, 0, 2]);
await page.click(`${bench(G)} button[data-thread="${bt.x}-${bt.y}-${bt.z}"]`);
await pause(80);
check("explorer: clicking x 2, y 0, z 1 shows the book's label Thread (1,0,2) of Block (1,0)", (await text(bench(G) + ' [data-out="label"]')) === `${formatThreadLabel(bt)} of ${formatBlockLabel({ x: 0, y: 1, z: 0 }, FIG31.grid)}`);
check('explorer: and the same thread in constructor order', (await text(bench(G) + ' [data-out="threadIdx"]')) === formatDim3(bt) && (await text(bench(G) + ' [data-out="blockIdx"]')) === '(0, 1, 0)');
check('explorer: and its number in the block with the working, 10', (await text(bench(G) + ' [data-out="linear"]')) === linearThreadWorking(bt, FIG31.block) && linearThread(bt, FIG31.block) === 10, await text(bench(G) + ' [data-out="linear"]'));
if (out) await page.screenshot({ path: out + '-explorer.png' });
await page.click(bench(G) + ' button[data-block="1-0-0"]');
await pause(80);
check('explorer: picking the block at x 1, y 0 retitles the panel Block (0,1) and clears the thread', (await text(bench(G) + ' [data-out="block-title"]')).includes('Block (0,1).') && (await text(bench(G) + ' [data-out="thread"]')).includes('Click a thread'));

// Limits: the book's rule against CUDA's documented limits.
let v = await verdict();
check("explorer: Fig. 3.1 is accepted under the book's rule", v.legal === 'true' && v.text.includes('accepted'));
await preset('b1-1-128');
v = await verdict();
check("explorer: (1,1,128) is accepted by the book's rule, which tests no single dimension", v.legal === 'true' && v.text.includes('The book gives no cap for a single dimension'), v.text);
check('explorer: it is 128 threads in a block', (await outv('per-block')) === '128');
check('explorer: four z layers of it are drawn, with a note that the drawing is partial', (await page.$$(bench(G) + ' button[data-thread]')).length === 4 && (await text(bench(G) + ' [data-out="threads-cut"]')).includes('of 1 × 1 × 128'));
await judgeBy('cuda-doc');
v = await verdict();
const zcap = checkLaunch(FIG31.grid, dim3(1, 1, 128), CUDA_DOCUMENTED);
check("explorer: under CUDA's documented limits it is rejected for the z cap, in the model's words", v.legal === 'false' && zcap.violations.length === 1 && (await violations()).join('|') === describeViolation(zcap.violations[0]), (await violations()).join('|'));
check('explorer: the error a real launch would set is named', (await text(bench(G) + ' [data-out="error"]')).includes('cudaErrorInvalidConfiguration (9)') && (await text(bench(G) + ' [data-out="error"]')).includes('cudaGetLastError()'));
if (out) await page.screenshot({ path: out + '-explorer-zcap.png' });

const shapes = [[512, 1, 1], [8, 16, 4], [32, 16, 2], [32, 32, 2]];
for (const key of ['book', 'cuda-doc']) {
  await judgeBy(key);
  const got = [];
  for (const [x, y, z] of shapes) {
    await preset(`b${x}-${y}-${z}`);
    got.push((await verdict()).legal);
  }
  const want = shapes.map(([x, y, z]) => String(checkLaunch(FIG31.grid, dim3(x, y, z), key === 'book' ? BOOK_RULE : CUDA_DOCUMENTED).legal));
  check(`explorer: the book's four shapes under ${key}: allowed, allowed, allowed, rejected`, got.join() === want.join() && got.join() === 'true,true,true,false', got.join());
}
check('explorer: (32,32,2) fails on the product, 2,048 against 1,024', (await violations()).join('|') === 'A block would hold 2,048 threads. The limit is 1,024.', (await violations()).join('|'));
// Limit sets read from this machine's reports, whatever they are.
const choices = await page.$$eval(bench(G) + ' select[aria-label="Judge the launch by"] option', (o) => o.map((x) => [x.value, x.textContent]));
check('explorer: the list starts with the book and CUDA and names no device of its own', choices[0][0] === 'book' && choices[1][0] === 'cuda-doc' && choices.slice(2).every(([, l]) => l.startsWith('Yours: ') || l.includes('emulator')), JSON.stringify(choices));
for (const [key, label] of choices.slice(2)) {
  await judgeBy(key);
  v = await verdict();
  const foot = await text(bench(G) + ' > footer');
  if (key === 'metal')
    check(`explorer: by "${label}" the per-dimension caps pass and the page says the total is per kernel`, v.legal === 'true' && v.text.includes('maxTotalThreadsPerThreadgroup') && v.text.includes('Apple documents no limit on the size of a grid'), v.text);
  else if (label.includes('emulator'))
    check(`explorer: by "${label}" (32,32,2) is rejected and the page says what the emulator enforces`, v.legal === 'false' && v.text.includes('only for a block of more than 1024 threads') && foot.includes('not a GPU'), v.text);
  else check(`explorer: by "${label}" there is a verdict and the footer says where the numbers come from`, ['true', 'false'].includes(v.legal) && foot.includes('./verify.sh'), v.text);
}
await judgeBy('cuda-doc');

// A 1D launch too large to draw in full.
await preset('p46');
check('explorer: 32 blocks of 128 threads is 4,096 threads (p. 46)', (await outv('threads')) === String(totalThreads(dim3(32), dim3(128))) && (await outv('threads')) === '4096');
check('explorer: the drawing is partial and says so', (await text(bench(G) + ' [data-out="blocks-cut"]')).includes('of 32 × 1 × 1') && (await text(bench(G) + ' [data-out="threads-cut"]')).includes('of 128 × 1 × 1'));
check('explorer: 1D blocks keep the two-part label', (await page.$$eval(bench(G) + ' button[data-block]', (b) => b.map((x) => x.textContent))).slice(0, 3).join(' ') === '(0,0) (0,1) (0,2)');

// Grid limits, zeros, a 3D grid, and bad input.
await setDim('gridDim', 'y', 65536);
v = await verdict();
check('explorer: gridDim.y = 65536 breaks the grid cap', v.legal === 'false' && (await violations()).join('|') === describeViolation(checkLaunch(dim3(32, 65536, 1), dim3(128), CUDA_DOCUMENTED).violations[0]), (await violations()).join('|'));
check('explorer: a count past 2^53 is still exact', (await (async () => { await setDim('gridDim', 'x', 2147483647); await setDim('gridDim', 'y', 65535); await setDim('gridDim', 'z', 65535); return outv('blocks'); })()) === String(blockCount(dim3(2147483647, 65535, 65535))));
check('explorer: a grid at its documented limits is accepted', (await verdict()).legal === 'true');
await setDim('gridDim', 'x', 2);
await setDim('gridDim', 'y', 0);
await setDim('gridDim', 'z', 1);
v = await verdict();
check('explorer: a zero dimension is rejected and nothing is drawn', v.legal === 'false' && v.text.includes('gridDim.y is 0. Every dimension must be at least 1.') && (await exists(bench(G) + ' [data-out="nothing"]')) && !(await exists(bench(G) + ' button[data-block]')), v.text);
await setDim('gridDim', 'y', '');
v = await verdict();
check('explorer: an empty field gives no verdict and says what a field takes', v.legal === 'none' && (await exists(bench(G) + ' [data-out="invalid"]')));
await setDim('gridDim', 'y', 2);
await setDim('gridDim', 'z', 2);
await preset('b8-16-4');
const names3 = await page.$$eval(bench(G) + ' button[data-block]', (b) => b.map((x) => x.textContent));
check('explorer: a grid with two z layers draws 8 blocks with three-part labels (z,y,x)', names3.length === 8 && names3.includes('(1,0,1)') && names3[1] === formatBlockLabel({ x: 1, y: 0, z: 0 }, dim3(2, 2, 2)).replace('Block ', ''), names3.join(' '));
await page.click(bench(G) + ' button[data-thread="3-5-2"]');
await pause(80);
check('explorer: in an (8,16,4) block the thread at x 3, y 5, z 2 is number 299', (await text(bench(G) + ' [data-out="linear"]')) === linearThreadWorking({ x: 3, y: 5, z: 2 }, dim3(8, 16, 4)) && linearThread({ x: 3, y: 5, z: 2 }, dim3(8, 16, 4)) === 299, await text(bench(G) + ' [data-out="linear"]'));
await preset('fig31');
check('explorer: the Fig. 3.1 preset restores 64 threads', (await outv('threads')) === '64');

// ---------------------------------------------------------------- tables computed from the model
const shapeRows = await page.$$eval('table[data-table="shapes"] tbody tr', (r) => r.map((x) => [x.dataset.shape, x.dataset.book, x.dataset.cuda, x.textContent]));
const shapeWant = [...shapes, [1, 1, 128]].map(([x, y, z]) => [`${x}-${y}-${z}`, String(checkLaunch(dim3(), dim3(x, y, z), BOOK_RULE).legal), String(checkLaunch(dim3(), dim3(x, y, z), CUDA_DOCUMENTED).legal)]);
check('shapes table: five shapes with the model verdicts under both rules', JSON.stringify(shapeRows.map((r) => r.slice(0, 3))) === JSON.stringify(shapeWant), JSON.stringify(shapeRows.map((r) => r.slice(0, 3))));
check('shapes table: (1,1,128) is the only row where the two rules disagree', shapeRows.filter((r) => r[1] !== r[2]).map((r) => r[0]).join() === '1-1-128');
check('shapes table: the z cap is given as the reason', shapeRows[4][3].includes('blockDim.z is 128. The cap for that dimension of a block is 64.'));
const limitRows = await page.$$eval('table[data-table="limits"] tbody tr', (r) => r.map((x) => [x.dataset.limits, x.textContent]));
check('limits table: the book states no per-dimension block cap; NVIDIA states 1,024, 1,024, 64', limitRows[0][0] === 'book' && limitRows[0][1].includes('not stated') && limitRows[1][0] === 'cuda-doc' && limitRows[1][1].includes('1,024, 1,024, 64'), JSON.stringify(limitRows.slice(0, 2)));
check('limits table: one row per limit set offered in the explorer', limitRows.map((r) => r[0]).join() === choices.map((c) => c[0]).join(), limitRows.map((r) => r[0]).join() + ' vs ' + choices.map((c) => c[0]).join());
const backendRows = await page.$$eval('table[data-table="backends"] tbody tr', (r) => r.map((x) => x.textContent));
check('backend table: six rows; Metal uses MTLSize and has no documented grid limit or error', backendRows.length === 6 && backendRows[0].includes('MTLSize') && backendRows[4].includes('not documented') && backendRows[5].includes('hipErrorInvalidConfiguration') && backendRows[5].endsWith('not documented'), backendRows.join(' || '));

// ---------------------------------------------------------------- find the thread
const cellOf = (tg) => `${bench(F)} button[data-cell="${tg.block.x}-${tg.block.y}-${tg.block.z}-${tg.thread.x}-${tg.thread.y}-${tg.thread.z}"]`;
const targets = drillTargets(FIG31.grid, FIG31.block, 3, 1);
const ask = (tg) => `${formatThreadLabel(tg.thread)} of ${formatBlockLabel(tg.block, FIG31.grid)}`;
check('drill: 64 unnamed cells', (await page.$$eval(bench(F) + ' button[data-cell]', (b) => b.filter((x) => x.textContent === '').length)) === 64);
check("drill: the first target is the book's Thread (1,0,2) of Block (1,0)", (await text(bench(F) + ' [data-out="target"]')).includes(`Target 1 of 3: click ${ask(targets[0])}.`) && ask(targets[0]) === 'Thread (1,0,2) of Block (1,0)', await text(bench(F) + ' [data-out="target"]'));
// The trap: the block label read as (x, y).
const trap = { block: { x: 1, y: 0, z: 0 }, thread: targets[0].thread };
await page.click(cellOf(trap));
await pause(80);
let fb = await text(bench(F) + ' [data-out="feedback"]');
check('drill: the block read as (x, y) is named for what it is, with the rule', fb.includes(`That is ${ask(trap)}.`) && fb.includes('A block label is (y, x): its first number is y.') && fb.includes('The thread is right.'), fb);
check('drill: the wrong cell is marked as a miss', (await page.$eval(cellOf(trap), (b) => b.dataset.state)) === 'missed' && (await page.$eval(bench(F) + ' [data-out="score"]', (e) => e.dataset.misses)) === '1');
// A wrong thread in the right block.
const wrongThread = { block: targets[0].block, thread: { x: 1, y: 0, z: 0 } };
await page.click(cellOf(wrongThread));
await pause(80);
fb = await text(bench(F) + ' [data-out="feedback"]');
check('drill: a wrong thread in the right block says so', fb.includes('The block is right.') && fb.includes('A thread label is (z, y, x)'), fb);
if (out) await page.screenshot({ path: out + '-drill-miss.png' });
for (let i = 0; i < targets.length; i++) {
  check(`drill: target ${i + 1} is the model's (${ask(targets[i])})`, (await text(bench(F) + ' [data-out="target"]')).includes(`Target ${i + 1} of 3: click ${ask(targets[i])}.`), await text(bench(F) + ' [data-out="target"]'));
  await page.click(cellOf(targets[i]));
  await pause(80);
  fb = await text(bench(F) + ' [data-out="feedback"]');
  check(`drill: target ${i + 1} found, with its constructor-order indices`, fb.includes('Found.') && fb.includes(`threadIdx is ${formatDim3(targets[i].thread)}`) && fb.includes(`blockIdx is ${formatDim3(targets[i].block)}`) && (await page.$eval(cellOf(targets[i]), (b) => b.dataset.state)) === 'found', fb);
  if (i < targets.length - 1) await clickText(F, 'Next target');
}
check('drill: the score is 3 found with 2 misses', (await text(bench(F) + ' [data-out="score"]')) === 'Found 3 of 3, with 2 misses.', await text(bench(F) + ' [data-out="score"]'));
if (out) await page.screenshot({ path: out + '-drill-done.png' });
check('drill: a new round can be started', await clickText(F, 'Three new targets'));
const round2 = drillTargets(FIG31.grid, FIG31.block, 3, 2);
check('drill: the new round resets the score and starts again', (await text(bench(F) + ' [data-out="score"]')) === 'Found 0 of 3, with 0 misses.' && (await text(bench(F) + ' [data-out="target"]')).includes(`Target 1 of 3: click ${ask(round2[0])}.`));
await page.click(cellOf(round2[0]));
await clickText(F, 'Next target');
check("drill: the second round's second target comes from the model with the next seed", (await text(bench(F) + ' [data-out="target"]')).includes(`Target 2 of 3: click ${ask(round2[1])}.`), await text(bench(F) + ' [data-out="target"]'));

// ---------------------------------------------------------------- questions
let q = await question("What number does the book's");
await typeAnswer(q, 17);
check('q-label: the label read in constructor order (17) is rejected', (await qText(q)).includes('Not quite. The answer is 10.'));
await typeAnswer(q, linearThread(threadFromLabel([1, 0, 2]), dim3(4, 2, 2)));
check('q-label: 10 is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('2 + 0×4 + 1×8 = 10'));
q = await question('which of these block shapes is rejected');
await pick(q, 'dim3(32, 16, 2)');
check('q-zcap: (32,16,2) is marked wrong, with the reason', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Exactly 1024 threads'));
await pick(q, 'dim3(1, 1, 128)');
check('q-zcap: (1,1,128) is accepted', (await qText(q)).includes('Correct.'));
q = await question('It never calls cudaGetLastError()');
await pick(q, 'The launch line throws an exception');
check('q-silent: an exception is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('throws nothing'));
await pick(q, 'The kernel never runs');
check('q-silent: the silent failure is accepted, and labelled unverified on device', (await qText(q)).includes('Correct.') && (await qText(q)).includes('unverified on device'));
q = await question('What is the Metal host code for the same');
await pick(q, 'MTLSizeMake(42)');
check('q-metal-size: one argument is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('no defaults'));
await pick(q, 'MTLSizeMake(42, 1, 1)');
check('q-metal-size: three arguments with ones is accepted', (await qText(q)).includes('Correct.'));
const ex3 = Number(totalThreads(dim3(sizeGridMinusOne(300, 16), sizeGridMinusOne(150, 32)), dim3(16, 32)));
q = await question('dim3 bd(16, 32);');
await typeAnswer(q, 95);
check('q-ex3-threads: the number of blocks is not the answer', (await qText(q)).includes(`Not quite. The answer is ${ex3} threads.`), await qText(q));
await typeAnswer(q, ex3);
check(`q-ex3-threads: ${ex3} is accepted and the three steps are shown`, ex3 === 48640 && (await qText(q)).includes('Correct.') && (await qText(q)).includes('(19, 5)') && (await qText(q)).includes('95 × 512 = 48,640'), await qText(q));

await pause(700);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
