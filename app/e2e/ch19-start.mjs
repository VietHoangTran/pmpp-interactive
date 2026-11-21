// Browser check for chapter 19, "Before you start": structure and margin notes, the chapter
// map and the lab line, the bridge to PyTorch and llama.cpp, the five prerequisite questions
// (answers computed from the tested models, wrong answers first, and each explanation linking
// back to the section it came from), links, the page at 390 px, and no errors.
// Usage: APP_URL=http://localhost:<port> node e2e/ch19-start.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { offset, phases, printedPhases } from '../src/sim/convlib.ts';
import { basicIntensity } from '../src/sim/convai.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
// Leave no trace in the real progress file, and start from an empty one.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') puts++;
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
  if (sel) await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch19/start');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Before you start" of chapter 19', h1.includes('Before you start'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What the book sets out to do', 'The chapter at a glance', 'Underneath torch.nn.functional.conv2d', 'Before you start'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no em dashes in the prose', !text0.includes('—'));
check('opens with why: no convolution in a decoder block, patch embeddings and audio encoders', text0.includes('decoder-only transformer block contains no convolution') && text0.includes('patch embedding') && text0.includes('Whisper-style'));
check('names the lasting idea and its source (p. 464, Chellapilla)', text0.includes('p. 464') && text0.includes('Chellapilla') && text0.includes('one heavily tuned GEMM routine'));
check('says plainly that the promised optimisations come down to the GEMM formulation', text0.includes('The one optimisation the chapter develops is the GEMM formulation'));
check('the bridge names PyTorch\'s slow path as §19.3, labelled as not in the book', text0.includes('slow CUDA path of its own') && text0.includes('section 19.3 nearly line for line') && text0.includes('Not in the book'));
check('the porting sentence is there, with the HIP limits', text0.includes('set up to build for HIP unchanged through') && text0.includes('not been compiled or run on AMD hardware') && text0.includes('cooperative groups') && text0.includes('threadgroup_barrier'));
const today = norm(await page.$$eval('.prose > .note.today', (ns) => ns.map((n) => n.textContent).join(' ')));
check('a Today note on llama.cpp\'s patch embedding and Whisper encoder, dated', today.includes('ggml_conv_2d') && today.includes('conv1d') && today.includes('Checked 2026-11-20'), today);
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const sources = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.textContent.startsWith('Book')).map((n) => n.textContent));
check('book sources carry pages (pp. 453–454 and §19.3)', sources.some((s) => s.includes('453–454')) && sources.some((s) => s.includes('§19.3')), sources.join(' | '));

// ---- the chapter map and the lab
const mapLinks = await page.$$eval('table[data-map] a', (a) => a.map((x) => x.getAttribute('href')));
check('the map links 19.1 to 19.4', JSON.stringify(mapLinks) === JSON.stringify(['#/ch19/19.1', '#/ch19/19.2', '#/ch19/19.3', '#/ch19/19.4']), mapLinks.join(' '));
const lab = norm(await page.$eval('p[data-lab]', (e) => e.textContent));
check('the lab line names lab 19.1, the direct kernel and the M2 GEMM, and links the labs page', lab.includes('Lab 19.1') && lab.includes('M2 GEMM') && lab.includes('direct') && (await page.$('p[data-lab] a[href="#/ch19/labs"]')) !== null, lab);
await shot('top', '.prose');

// ---- the five questions
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerNumber = async (part, value) => {
  const h = await q(part);
  await h.evaluate((e, v) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => ({ text: e.querySelector('.verdict')?.textContent ?? '', links: [...e.querySelectorAll('.verdict a')].map((a) => a.getAttribute('href')) }));
};
const answerChoice = async (part, option) => {
  const h = await q(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e, ok) => ({
    ok,
    text: (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? ''),
    links: [...e.querySelectorAll('.verdict a')].map((a) => a.getAttribute('href')),
  }), ok);
};
check('five questions on the page', (await page.$$('.q')).length === 5, String((await page.$$('.q')).length));

const idx = offset('NCHW', { C: 3, H: 32, W: 32 }, 1, 2, 5, 7);
let v = await answerNumber('flat index of', 5280);
check('q-index4d: a wrong index is marked wrong, the answer shown, and the explanation links ch. 3 §3.2', v.text.includes('Not quite.') && v.text.includes(String(idx)) && v.links.includes('#/ch03/3.2'), JSON.stringify(v));
v = await answerNumber('flat index of', idx);
check(`q-index4d: ${idx} (from the model) is correct and points at 19.2 and 19.3`, v.text.includes('Correct.') && v.links.includes('#/ch19/19.2') && v.links.includes('#/ch19/19.3'), JSON.stringify(v));

v = await answerChoice('how many times is each element of', 'Once: after the first block');
check('q-reuse: "once" is wrong, with its reason, and links ch. 5 §5.3 and ch. 15 §15.2', v.ok && v.text.includes('Not quite.') && v.text.includes('That is the ideal') && v.links.includes('#/ch05/5.3') && v.links.includes('#/ch15/15.2'), JSON.stringify(v));
v = await answerChoice('how many times is each element of', 'once by each block in its row');
check(`q-reuse: ceil(l/T) is correct, with LeNet C1's ${phases(784, 16)} fetches`, v.ok && v.text.includes('Correct.') && v.text.includes(`fetched ${phases(784, 16)} times`), JSON.stringify(v));

v = await answerNumber('How many times does its phase loop run', printedPhases(25, 16));
check('q-phases: the truncated count (1) is wrong, and the explanation links ch. 5 §5.5', v.text.includes('Not quite.') && v.text.includes(`The answer is ${phases(25, 16)}`) && v.links.includes('#/ch05/5.5'), JSON.stringify(v));
v = await answerNumber('How many times does its phase loop run', phases(25, 16));
check('q-phases: 2 is correct, and Fig. 19.11\'s dropped 9 terms are named', v.text.includes('Correct.') && v.text.includes('drops 9 of the 25 terms') && v.links.includes('#/ch19/19.3'), JSON.stringify(v));

v = await answerNumber('FLOP does its inner loop do', 0.5);
check('q-convratio: 0.5 (the constant-memory figure) is wrong, and the explanation links ch. 7 §7.3', v.text.includes('Not quite.') && v.links.includes('#/ch07/7.3'), JSON.stringify(v));
v = await answerNumber('FLOP does its inner loop do', basicIntensity(4));
check(`q-convratio: ${basicIntensity(4)} FLOP/B is correct and ties to 19.2`, v.text.includes('Correct.') && v.links.includes('#/ch19/19.2'), JSON.stringify(v));

v = await answerChoice('flip the filter', 'PyTorch flips it, as signal processing does');
check('q-xcorr: "PyTorch flips" is wrong, with its reason, and links ch. 7 §7.1', v.ok && v.text.includes('Not quite.') && v.text.includes('cross-correlation') && v.links.includes('#/ch07/7.1'), JSON.stringify(v));
v = await answerChoice('flip the filter', 'Neither does:');
check('q-xcorr: "neither" is correct and points to 19.4', v.ok && v.text.includes('Correct.') && v.links.includes('#/ch19/19.4'), JSON.stringify(v));
await pause(600);
check('answers were saved through PUT /api/progress (intercepted)', puts > 0, String(puts));
await shot('questions', '.q');

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch19/19.1', '#/ch19/19.2', '#/ch19/19.3', '#/ch19/19.4', '#/ch19/labs', '#/review', '#/ch03/3.2', '#/ch05/5.3', '#/ch05/5.5', '#/ch07/7.1', '#/ch07/7.3', '#/ch07/7.5', '#/ch15/15.2']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch19/start');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-map]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
