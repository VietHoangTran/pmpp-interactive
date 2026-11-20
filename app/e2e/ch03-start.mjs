// Browser check for chapter 3, "Before you start": the chapter 2 listing and the four prerequisite questions.
// Usage: node e2e/ch03-start.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { coverage, dim3, globalCoord, sizeGrid } from '../src/sim/grid.ts';

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

await goto(page, 'ch03/start');
await page.waitForFunction(() => document.querySelector('.code pre')?.textContent.includes('vecAddKernel<<<'));

// The listing is the checked snippet file, line for line.
const shown = await page.$eval('.code pre', (e) => e.textContent);
const file = fs.readFileSync(path.join(repo, 'labs/ch03/snippets/vec_add_1d.cu'), 'utf8').trimEnd().split('\n');
check('the listing is labs/ch03/snippets/vec_add_1d.cu, every line', file.every((l) => shown.includes(l)), file.find((l) => !shown.includes(l)));
const hl = await page.$$eval('.code pre .ln.hl', (ls) => ls.map((l) => l.textContent));
check('the mapping and the guard are the two marked lines', hl.length === 2 && hl[0].includes('blockIdx.x * blockDim.x + threadIdx.x') && hl[1].includes('if (i < n)'), hl.join('|'));
check('the listing states its verification status', (await page.$eval('.code .status', (e) => e.textContent)) === 'front-end-checked');
check('the colour legend names the two tiers this chapter uses', (await page.$$eval('.note .chip', (c) => c.map((x) => x.textContent))).join('|') === 'register|global memory');

// Links to the rest of the chapter.
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
check('links to sections 3.1 to 3.4, the labs and review', ['#/ch03/3.1', '#/ch03/3.2', '#/ch03/3.3', '#/ch03/3.4', '#/ch03/labs', '#/review'].every((h) => links.includes(h)), links.join(' '));

// Four questions, all labelled as chapter 2 material.
const kinds = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('four prerequisite questions, each labelled "From chapter 2"', kinds.length === 4 && kinds.every((k) => k === 'From chapter 2'), kinds.join('|'));

// q-globalindex: the model's value is accepted, a neighbour is not.
const index = globalCoord(dim3(3, 0, 0), dim3(256), dim3(10, 0, 0)).x;
let q = await question('blockIdx.x = 3');
await typeAnswer(q, index + 256);
check('global index: one block too far is rejected and the answer shown is the model value', (await qText(q)).includes(`Not quite. The answer is ${index}.`), await qText(q));
await typeAnswer(q, index);
check(`global index: ${index} is accepted`, (await qText(q)).includes('Correct.') && (await qText(q)).includes('3 × 256 + 10 = 778'));

// q-surplus: 8 blocks of 1024 for 8000 elements.
const c = coverage(8000, 1024, sizeGrid(8000, 1024, 'round-up'));
q = await question('8000 elements');
await typeAnswer(q, c.threads);
check('surplus: the thread count is not the answer', (await qText(q)).includes(`Not quite. The answer is ${c.surplus} threads.`), await qText(q));
await typeAnswer(q, c.surplus);
check(`surplus: ${c.surplus} is accepted and the explanation carries the model numbers`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${c.blocks} blocks`) && (await qText(q)).includes('8,192 threads'));

// q-bytes and q-order.
q = await question('size argument of cudaMalloc');
await pick(q, 'w * h * 3 * sizeof(float)');
check('bytes: the float size is marked wrong with its reason', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('unsigned char, not float'));
await pick(q, '(void**)&Pin_d');
check('bytes: the pointer argument is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('first argument'));
await q.evaluate((e) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim() === 'w * h * 3').querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('bytes: w * h * 3 is accepted', (await qText(q)).includes('Correct.'));
q = await question('block 0 finishing');
await pick(q, 'Only on a small GPU');
check('order: the small-GPU answer is marked wrong', (await qText(q)).includes('Not quite.'));
await pick(q, 'No. Blocks may execute in any order.');
check('order: "no" is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('writes only its own output element'));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

await pause(700);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
