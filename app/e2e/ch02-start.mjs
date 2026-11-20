// Browser check for chapter 2, "Before you start": the kernel of lab 2.1 shown as
// the destination, the road map, the portability note and the four prerequisite
// questions from chapter 1.
// Usage: node e2e/ch02-start.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { ceiling, speedup } from '../src/sim/amdahl.ts';

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

await goto(page, 'ch02/start');
await page.waitForFunction(() => document.querySelector('.code pre')?.textContent.includes('__global__'));

// ---- the kernel is the lab file, cut down to the kernel
const file = fs.readFileSync(path.join(repo, 'labs/ch02/lab1_vecadd/solution/kernel.cu'), 'utf8').split('\n');
const at = file.findIndex((l) => l.startsWith('__global__'));
let first = at;
while (first > 0 && file[first - 1].startsWith('//')) first--;
const last = file.findIndex((l, i) => i > at && l.startsWith('}'));
const want = file.slice(first, last + 1);
const shown = await page.$$eval('.code pre .ln', (ls) => ls.map((l) => ({ n: Number(l.dataset.n), text: l.textContent.replace(/\n$/, '') })));
check('the listing is the kernel of labs/ch02/lab1_vecadd/solution/kernel.cu, line for line', shown.length === want.length && want.every((l, i) => shown[i].text === l), `${shown.length} lines shown, ${want.length} wanted`);
check('its line numbers are the ones in the file', shown[0].n === first + 1 && shown[shown.length - 1].n === last + 1, `${shown[0].n} to ${shown[shown.length - 1].n}`);
check('only the kernel is shown: no host code, and no loop', !shown.some((l) => /cudaMalloc|cudaMemcpy|vecAdd\(/.test(l.text)) && !shown.some((l) => /\bfor\b|\bwhile\b/.test(l.text.replace(/\/\/.*$/, ''))));
check('the kernel has the index, the guard and the one line of work', shown.some((l) => l.text.includes('int i = threadIdx.x + blockDim.x * blockIdx.x;')) && shown.some((l) => l.text.includes('if (i < n)')) && shown.some((l) => l.text.includes('C[i] = A[i] + B[i];')));
const globals = await page.$$eval('.code pre .v-global', (s) => [...new Set(s.map((x) => x.textContent))].sort());
const regs = await page.$$eval('.code pre .v-reg', (s) => [...new Set(s.map((x) => x.textContent))]);
check('A, B and C carry the colour of global memory and i the colour of a register', globals.join() === 'A,B,C' && regs.join() === 'i', `${globals} / ${regs}`);
const caption = await page.$eval('.code .caption', (e) => e.textContent);
check('the caption says what it is: Fig. 2.10 with const added, from lab 2.1', caption.includes('Fig. 2.10') && caption.includes('const') && caption.includes('labs/ch02/lab1_vecadd/solution/kernel.cu'), caption);
if (out) await page.screenshot({ path: out + '-top.png' });

// ---- margin notes
const chips = await page.$$eval('.note .chip', (c) => c.map((x) => x.textContent));
check('the colour legend names the two tiers this chapter uses', chips.join('|') === 'register|global memory', chips.join('|'));
check('every margin note directly follows text, never the listing', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName))));
check('there is no Today note on this page', (await page.$$('.note.today')).length === 0);
const beyond = await page.$$eval('.note.beyond', (n) => n.map((x) => x.textContent));
check('the PyTorch bridge is labelled as ours and names what was read', beyond.some((t) => t.includes('The PyTorch connection is ours') && t.includes('cudaMalloc') && t.includes('tensor.to("cuda")')), beyond.join(' || '));
check('the HIP and Metal paragraph is labelled as not in the book', beyond.some((t) => t.includes('The book covers CUDA only')));
const prose = await page.$eval('.prose', (e) => e.textContent);
check('the portability headline: hip for cuda, and no copies with a shared Metal buffer', prose.includes('in place of cuda') && prose.includes('shared storage') && prose.includes('the two copy steps have nothing to do'));

// ---- road map and links
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
const sections = ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7'];
check('the road map links to every section from 2.1 to 2.7', sections.every((s) => links.includes('#/ch02/' + s)), links.join(' '));
check('it links to the labs page and to review', links.includes('#/ch02/labs') && links.includes('#/review'));
check('the four labs are listed', ['2.1,', '2.2,', '2.3,', '2.4,'].every((s) => prose.includes(s)) && prose.includes('milestone M0'));

// ---- the four questions from chapter 1
const qs = await page.$$eval('.q', (q) => q.map((e) => e.querySelector('.kind').textContent));
check('four questions, each marked as coming from chapter 1', qs.length === 4 && qs.every((k) => k === 'From chapter 1'), qs.join('|'));

let q = await question('largest overall speed-up');
await typeAnswer(q, 9);
check('q-amdahl: 9 is rejected and the answer given', (await qText(q)).includes('Not quite. The answer is 10'));
await typeAnswer(q, Math.round(ceiling(0.9)));
let t = await qText(q);
check('q-amdahl: the model’s ceiling for 90% is accepted', t.includes('Correct.') && Math.round(ceiling(0.9)) === 10, t.slice(-200));
check('q-amdahl: the explanation uses the book’s own case from the same model', t.includes(speedup(0.3, 100).toFixed(2)) && t.includes(ceiling(0.3).toFixed(2)) && speedup(0.3, 100).toFixed(2) === '1.42' && ceiling(0.3).toFixed(2) === '1.43', t.slice(-400));
check('q-amdahl: it says which part is ours and where the copies sit', t.includes('The decode-step framing is ours') && t.includes('10% side'));

q = await question('Which workload belongs on the GPU side');
await pick(q, 'One long chain');
t = await qText(q);
check('q-design: a chain is rejected, with the reason', t.includes('Not quite.') && t.includes('only the speed of a single thread matters'), t.slice(-300));
await pick(q, 'The same small computation');
t = await qText(q);
check('q-design: independent elements is accepted and points to section 2.1', t.includes('Correct.') && t.includes('data parallelism') && (await q.$$eval('a', (a) => a.map((x) => x.getAttribute('href')))).includes('#/ch02/2.1'));

q = await question('names four difficulties');
await pick(q, 'Synchronisation overhead');
check('q-challenge: synchronisation is rejected', (await qText(q)).includes('No thread needs anything from another thread'));
await pick(q, 'Memory-bound execution');
t = await qText(q);
check('q-challenge: memory-bound is accepted, with 12 bytes per addition and the link to 5.1', t.includes('Correct.') && t.includes('12') && (await q.$$eval('a', (a) => a.map((x) => x.getAttribute('href')))).includes('#/ch05/5.1'), t.slice(-300));

q = await question('200 s on system A');
await typeAnswer(q, 20);
t = await qText(q);
check('q-speedup: 20 is accepted, with the caution about the baseline', t.includes('Correct.') && t.includes('poor CPU baseline'), t.slice(-300));
await pause(700); // saves are debounced
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---- layout
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
check('at phone width the page itself does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), String(await page.evaluate(() => document.documentElement.scrollWidth)));
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
