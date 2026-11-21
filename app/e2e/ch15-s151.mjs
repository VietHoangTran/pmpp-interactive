// Browser check for section 15.1: the "One product, three spellings" explorer and its
// questions. Every value it asserts is computed from the model the page runs on
// (src/sim/gemmlayout.ts), imported here directly.
// Usage: node e2e/ch15-s151.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  OPERATIONS, agreement, blasCall, editBlas, editMps, exampleBuffers, mapOperation, mpsCall, reference, runBlas, runMps,
} from '../src/sim/gemmlayout.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const B = 'section.bench[aria-label="One product, three spellings"]';
const flat = (s) => s.replace(/\s+/g, ' ').trim();

const setRange = async (label, v) => {
  await page.$eval(`${B} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setOp = async (id) => {
  await page.select(`${B} select[aria-label="operation"]`, id);
  await pause(80);
};
const setBox = async (sel, on) => {
  const box = await page.$(sel);
  if ((await box.evaluate((e) => e.checked)) !== on) await box.click();
  await pause(80);
};
const cells = (m) => page.$$eval(`${B} [data-matrix="${m}"] [data-cell]`, (es) => es.map((e) => ({ at: e.dataset.cell, v: e.textContent.trim(), ok: e.dataset.ok })));
const verdict = (panel) => page.$eval(`${B} [data-panel="${panel}"] [data-verdict]`, (e) => e.dataset.verdict);
const callText = async (which) => flat(await page.$eval(`${B} [data-call="${which}"]`, (e) => e.textContent));

// What the model says the page should show for one operation, shape and set of edits.
function expected(id, shape, edits = {}, flipRight = false) {
  const map = mapOperation(id, shape);
  const bufs = exampleBuffers(map);
  const ref = reference(map, bufs);
  const call = editBlas(blasCall(map), edits);
  const blas = runBlas(call, bufs);
  const blasOk = agreement(map, blas.C, ref);
  const mps = runMps(editMps(mpsCall(map), flipRight), bufs);
  const mpsOk = agreement(map, mps.C, ref);
  const v = (r, ok) => (r.readOutside || r.writeOutside ? 'outside' : ok.every(Boolean) ? 'agree' : 'wrong');
  return { map, ref, call, blasOk, blasVerdict: v(blas, blasOk), mpsVerdict: v(mps, mpsOk), mpsCall: editMps(mpsCall(map), flipRight) };
}

await goto(page, 'ch15/15.1');
await page.waitForSelector(B);

// Default: nn.Linear at 3 x 4 x 5.
let shape = { M: 3, N: 4, K: 5 };
let e = expected('linear', shape);
let text = await callText('blas');
check('default operation is nn.Linear and its call is (T, N, out, batch, in, W, in, x, in, y, out)',
  text.includes('CUBLAS_OP_T, CUBLAS_OP_N') && text.includes('out, batch, in') && text.includes('&alpha, W, in') && text.includes('x, in') && text.includes('&beta, y, out'), text);
const ref = await cells('ref');
check('reference cells are the model\'s row-major product', ref.length === 12 && ref.every((c, i) => Number(c.v) === e.ref[i]), JSON.stringify(ref.map((c) => c.v)));
check('BLAS call agrees cell for cell', (await verdict('blas')) === 'agree' && (await cells('blas')).every((c) => c.ok === '1'));
check('MPS call agrees cell for cell', (await verdict('mps')) === 'agree' && (await cells('mps')).every((c) => c.ok === '1'));
text = await callText('mps');
check('MPS call has transposeRight YES and the layer\'s names', text.includes('transposeRight: YES') && text.includes('resultRows: batch') && text.includes('interiorColumns: in'), text);
check('W is described as the book\'s B stored column-major', (await page.$eval(`${B} [data-bstore]`, (e) => e.dataset.bstore)) === 'column');
const W = await cells('B');
check('W is drawn as stored: out rows of in values', W.length === 20 && W[W.length - 1].at === '3,4', W.at(-1)?.at);

// Numbers instead of names.
await setBox(`${B} label.check input:not([data-edit])`, true);
text = await callText('blas');
check('with numbers the call reads (T, N, 4, 3, 5, W, 5, x, 5, y, 4)', text.includes('4, 3, 5') && text.includes('W, 5') && text.includes('x, 5') && text.includes('y, 4'), text);
await setBox(`${B} label.check input:not([data-edit])`, false);

// Every operation: transa from the model, and both calls agree.
for (const op of OPERATIONS) {
  await setOp(op.id);
  e = expected(op.id, shape);
  text = await callText('blas');
  const transa = e.call.transa === 'T' ? 'CUBLAS_OP_T' : 'CUBLAS_OP_N';
  check(`${op.id}: transa is ${e.call.transa} and the operand order is ${e.call.symbolic.a} then ${e.call.symbolic.b}`,
    text.startsWith(`cublasSgemm(handle, ${transa}, CUBLAS_OP_N`) && text.includes(`&alpha, ${e.call.symbolic.a}, ${e.call.symbolic.lda}`), text);
  check(`${op.id}: both calls agree with the reference`, (await verdict('blas')) === 'agree' && (await verdict('mps')) === 'agree');
  check(`${op.id}: MPS transposeRight is ${e.mpsCall.transposeRight ? 'YES' : 'NO'}`, (await callText('mps')).includes(`transposeRight: ${e.mpsCall.transposeRight ? 'YES' : 'NO'}`));
}

// Q Kᵀ has the same call shape as nn.Linear.
await setOp('qkt');
check('Q Kᵀ: K is stored like a weight (column-major B)', (await page.$eval(`${B} [data-bstore]`, (e) => e.dataset.bstore)) === 'column');
if (out) await page.screenshot({ path: out + '-qkt.png' });

// Each edit breaks the call, exactly as the model says.
await setOp('linear');
for (const edit of ['swap', 'flipTransa', 'mnBook']) {
  await setBox(`${B} input[data-edit="${edit}"]`, true);
  e = expected('linear', shape, { [edit]: true });
  const v = await verdict('blas');
  const marked = (await cells('blas')).map((c) => c.ok);
  check(`edit "${edit}": the page shows "${e.blasVerdict}" as the model does`, v === e.blasVerdict && v !== 'agree', v);
  if (v === 'wrong') check(`edit "${edit}": the wrong cells are the model's`, marked.every((m, i) => (m === '1') === e.blasOk[i]), JSON.stringify(marked));
  if (out) await page.screenshot({ path: `${out}-edit-${edit}.png` });
  await setBox(`${B} input[data-edit="${edit}"]`, false);
}
check('with every edit off the call agrees again', (await verdict('blas')) === 'agree');

// On the book's product, flipping transa stays inside the buffers and gives wrong numbers.
await setOp('book');
await setBox(`${B} input[data-edit="flipTransa"]`, true);
e = expected('book', shape, { flipTransa: true });
const marks = (await cells('blas')).map((c) => c.ok);
check('book with transa flipped: wrong values, in the cells the model marks', e.blasVerdict === 'wrong' && (await verdict('blas')) === 'wrong' && marks.every((m, i) => (m === '1') === e.blasOk[i]), JSON.stringify(marks));
if (out) await page.screenshot({ path: out + '-book-flip.png' });
await setBox(`${B} input[data-edit="flipTransa"]`, false);
await setOp('linear');

await setBox(`${B} input[data-edit="flipRight"]`, true);
e = expected('linear', shape, {}, true);
check('flipping transposeRight breaks the MPS call as the model says', (await verdict('mps')) === e.mpsVerdict && e.mpsVerdict !== 'agree', await verdict('mps'));
await setBox(`${B} input[data-edit="flipRight"]`, false);

// Shapes: the sliders change the book's M, N, K.
await setRange('book M', 2);
await setRange('book K', 6);
shape = { M: 2, N: 4, K: 6 };
e = expected('linear', shape);
const r2 = await cells('ref');
check('after M = 2, K = 6 the reference is 2 x 4 and matches the model', r2.length === 8 && r2.every((c, i) => Number(c.v) === e.ref[i]), JSON.stringify(r2.map((c) => c.v)));
check('and the calls still agree', (await verdict('blas')) === 'agree' && (await verdict('mps')) === 'agree');
check('dims readout follows', flat(await page.$eval(`${B} [data-dims]`, (e) => e.textContent)) === 'batch = 2, out = 4, in = 6');

// Questions.
// In this section's questions the correct option is listed first.
const answer = async (needle, pickCorrect) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  const labels = await q.$$('.opts label');
  await labels[pickCorrect ? 0 : 1].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
check('q-linear: the right mapping is accepted', (await answer('Which mapping onto the', true)).includes('Correct.'));
check('q-qkt: Q is not the weight-like operand', (await answer('Which operand plays the role of', false)).includes('Not quite.'));
check('q-rowmajor-trick: the transpose explanation is accepted', (await answer('leave', true)).includes('Correct.'));

if (out) await page.screenshot({ path: out + '-end.png', fullPage: false });
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
