// Browser check for section 19.4, the cuDNN library: the output-size prediction and its
// reveal, the "Describe a layer to a library" explorer (names, output size, GEMM shapes,
// duplication, limits, flashcards), the "Which kernel would they run?" explorer (prediction
// gate, three presets, the rules of llama.cpp, PyTorch's Slow2d and MLX as the layer changes),
// the algorithm and backend tables, the eight questions, links, the page at 390 px, and no
// errors. Expected values come from the tested model, src/sim/convlib.ts.
// Usage: APP_URL=http://localhost:<port> node e2e/ch19-s194.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import {
  CARDS, CUDNN_FWD_ALGOS, DISPATCH_PRESETS, PRESETS, bookGemm, duplication, libraryGemm, llamaCuda, mlx, outputDim,
  shape, slow2d, unfoldSize,
} from '../src/sim/convlib.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const D = 'section.bench[aria-label="Describe a layer to a library"]';
const X = 'section.bench[aria-label="Which kernel would they run?"]';
const shot = async (name, sel) => {
  if (!out) return;
  if (sel) await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const txt = async (sel) => norm(await page.$eval(sel, (e) => e.textContent));
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const clickIn = async (scope, sel) => {
  await page.$eval(`${scope} ${sel}`, (e) => e.click());
  await pause(80);
};
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(`${scope} button`)].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setField = async (k, v) => {
  await page.$eval(`${D} input[data-field="${k}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(50);
};
const setLayer = async (l) => {
  for (const k of ['N', 'C', 'H', 'W', 'M', 'K', 'pad', 'stride', 'dilation']) await setField(k, l[k]);
  await pause(60);
};
const setPreset = async (id) => {
  await page.$eval(`${D} select[aria-label="Preset"]`, (s, id) => {
    s.value = id;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, id);
  await pause(80);
};
const layerOf = (id) => PRESETS.find((p) => p.id === id).layer;
const nameValue = (sym) => attr(`${D} tr[data-name="${sym}"] td[data-value]`, 'data-value').then(Number);
const nameChapter = (sym) => attr(`${D} tr[data-name="${sym}"] td[data-chapter]`, 'data-chapter');
const gemmKey = (g) => `${g.rows}x${g.inner}x${g.cols}x${g.count}`;
/** Everything the describe explorer shows for a layer, checked against the model. */
const describeMatches = async (l) => {
  const sh = shape(l);
  const got = {
    P: Number(await attr(`${D} [data-outsize]`, 'data-p')),
    Q: Number(await attr(`${D} [data-outsize]`, 'data-q')),
    K: await nameValue('K'),
    R: await nameValue('R'),
    u: await nameValue('u'),
    pad: await nameValue('pad_h'),
    book: sh.valid ? await attr(`${D} [data-book-gemm]`, 'data-book-gemm') : '',
    lib: sh.valid ? await attr(`${D} [data-lib-gemm]`, 'data-lib-gemm') : '',
    ratio: sh.valid ? Number(await attr(`${D} [data-unfold]`, 'data-ratio')) : 0,
  };
  const g = libraryGemm(l);
  const ok =
    got.P === sh.P && got.Q === sh.Q && got.K === l.M && got.R === l.K && got.u === l.stride && got.pad === l.pad &&
    (!sh.valid || (got.book === gemmKey(bookGemm(l)) && got.lib === `${g.rows}x${g.inner}x${g.cols}` && Math.abs(got.ratio - duplication(l)) < 1e-12));
  return { ok, got };
};

await goto(page, 'ch19/19.4');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is 19.4, the cuDNN library', h1.includes('The cuDNN library'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What cuDNN is', 'Who calls it', 'Two sets of names', 'The output size', 'The algorithms', 'How the matrix multiplication runs',
  'The last step: layout', 'Choosing an algorithm in practice', 'On AMD: MIOpen', 'On Apple: three routes, no menu', 'What real backends run', 'Where the backends differ'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('no em dashes in the prose', !text0.includes('—'));
check('the book\'s API is named as the legacy API, deprecated since cuDNN 9.0', text0.includes('legacy API') && text0.includes('deprecated since cuDNN 9.0'));
check('the p. 472 slip is named', text0.includes('One slip on p. 472'));
check('the padding correction: zero rows, ghost cells, changes the output size', text0.includes('chapter 7’s ghost cells') && text0.includes('so they change the output size'));
check('the Tan et al. comparison is marked unverified', text0.includes('Unverified: cuDNN documents no kernel internals'));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const todays = await page.$$eval('.prose > .note.today', (ns) => ns.map((n) => n.textContent));
check(`every Today note is dated 2026-11-20 and cites a facts item (${todays.length})`, todays.length >= 8 && todays.every((t) => t.includes('Checked 2026-11-20 (ch19-facts')), todays.map((t) => t.slice(-50)).join(' | '));

// ---- the prediction, before the explorer
const PR = '.q:has(input[aria-label="your prediction"])';
check('the explorer starts on LeNet C1, not on the predicted layer', (await page.$eval(`${D} select[aria-label="Preset"]`, (s) => s.value)) === 'lenet-c1');
const stem = layerOf('stem');
const want = outputDim(stem.H, stem.K, stem.pad, stem.stride, stem.dilation);
check('nothing is revealed before committing', (await page.$('[data-outsize-answer]')) === null);
await page.type(`${PR} input[aria-label="your prediction"]`, String(stem.H - stem.K + 1));
await clickText(PR, 'Commit prediction');
await page.waitForSelector('[data-outsize-answer]');
const rev = await txt('[data-outsize-answer]');
check(`the reveal gives P = ${want} from cuDNN's formula and explains the H − K + 1 guess`, Number(await attr('[data-outsize-answer]', 'data-outsize-answer')) === want && rev.includes(`= ${want}`) && rev.includes('H − K + 1') && rev.includes('You predicted') === false, rev);
check('the committed value is echoed', norm(await page.$eval('.q .verdict.info', (e) => e.textContent)).includes(`You predicted ${stem.H - stem.K + 1} rows`));
await page.$eval('[data-load-stem]', (b) => b.click());
await pause(150);
check('"Load this layer" puts the stem layer in the explorer', (await page.$eval(`${D} select[aria-label="Preset"]`, (s) => s.value)) === 'stem' && Number(await attr(`${D} [data-outsize]`, 'data-p')) === want);
const steps = await txt(`${D} [data-axis="P"] [data-steps]`);
check('the explorer shows the arithmetic: 1 + 223 / 2 = 112', steps.includes('1 + 223 / 2 = 112'), steps);
check('the chapter\'s formula is flagged as wrong for this layer', (await attr(`${D} [data-chapter-formula]`, 'data-chapter-formula')) === 'differs');
check('PyTorch\'s and llama.cpp\'s formulas give the same', (await txt(`${D} [data-others]`)).includes(`gives ${want} and llama.cpp’s gives ${want}`));

// ---- describe explorer
let r = await describeMatches(stem);
check('stem: names, P, Q, both GEMM orientations and the duplication match the model', r.ok, JSON.stringify(r.got));
check('cuDNN\'s K row is the chapter\'s M, and R is the chapter\'s K', (await nameChapter('K')) === 'M' && (await nameChapter('R')) === 'K' && (await nameChapter('u')) === '');
await shot('describe', D);
// change P without touching H or K
const noPad = { ...stem, pad: 0 };
await setField('pad', 0);
r = await describeMatches(noPad);
check(`padding alone changes P: pad 0 gives ${shape(noPad).P}`, r.ok && r.got.P !== want, JSON.stringify(r.got));
// stride equal to the filter ends duplication
const patch = layerOf('patch');
await setPreset('patch');
r = await describeMatches(patch);
check('patch embedding: stride 16 on 224 gives 14 × 14 and duplication exactly 1', r.ok && r.got.P === 14 && r.got.ratio === 1, JSON.stringify(r.got));
check('  and the panel says no pixel is copied twice', (await txt(`${D} [data-dup]`)).includes('No pixel is copied twice'));
// a typed layer, with H ≠ W and dilation
const typed = { N: 2, C: 16, H: 40, W: 30, M: 24, K: 3, pad: 1, stride: 1, dilation: 2 };
await setLayer(typed);
r = await describeMatches(typed);
check(`a typed layer (H 40, W 30, dilation 2): P ${shape(typed).P}, Q ${shape(typed).Q}, all panels match`, r.ok, JSON.stringify(r.got));
check('  the preset box says it is your own layer', (await page.$eval(`${D} select[aria-label="Preset"]`, (s) => s.value)) === '');
check('  Q gets its own line when W differs from H', (await page.$(`${D} [data-axis="Q"]`)) !== null);
// over the Metal floor
const big = { N: 8, C: 64, H: 224, W: 224, M: 64, K: 3, pad: 0, stride: 1, dilation: 1 };
await setLayer(big);
const u = unfoldSize(big);
check(`8 × 64 channels at 224, 3 × 3: ${u.batch.toLocaleString('en-US')} elements is over Metal's 256 MB floor, under int32`, (await attr(`${D} [data-limits]`, 'data-limits')) === 'metal' && u.overMetalFloor && !u.overInt32);
// a filter that does not fit
await setLayer({ N: 1, C: 1, H: 4, W: 4, M: 1, K: 7, pad: 0, stride: 1, dilation: 1 });
check('a 7 × 7 filter on a 4 × 4 input: no output, and no GEMM panels', (await page.$(`${D} [data-invalid]`)) !== null && (await page.$(`${D} [data-gemm]`)) === null);
// clamping
await setField('K', 0);
check('K = 0 is clamped to 1', (await nameValue('R')) === 1);

// flashcards
await clickText(D, 'Flashcards');
check('flashcard mode shows the first card', (await attr(`${D} [data-card]`, 'data-card')) === CARDS[0].id);
const wrong0 = CARDS[0].options.find((o) => o !== CARDS[0].answer);
await clickIn(D, `[data-option="${wrong0}"]`);
let verdict = await attr(`${D} [data-verdict]`, 'data-verdict');
check(`card 1: "${wrong0}" for the chapter's M is wrong and the answer ${CARDS[0].answer} is shown`, verdict === 'wrong' && (await txt(`${D} [data-verdict]`)).includes(`No: ${CARDS[0].answer}`));
await clickText(D, 'Next card');
let allRight = true;
for (let i = 1; i < CARDS.length; i++) {
  const id = await attr(`${D} [data-card]`, 'data-card');
  if (id !== CARDS[i].id) allRight = false;
  await clickIn(D, `[data-option="${CARDS[i].answer}"]`);
  if ((await attr(`${D} [data-verdict]`, 'data-verdict')) !== 'right') allRight = false;
  if (i === CARDS.length - 1) await shot('cards', D);
  if (i < CARDS.length - 1) await clickText(D, 'Next card');
}
check(`cards 2 to ${CARDS.length}: each model answer is accepted`, allRight);
check(`the score reads ${CARDS.length - 1} of ${CARDS.length}`, (await txt(`${D} [data-score]`)) === `${CARDS.length - 1} of ${CARDS.length} right so far.`, await txt(`${D} [data-score]`));
check('the last card offers to start again', await clickText(D, 'Start again'));
check('  and starts at card 1 with a clean score', (await attr(`${D} [data-card]`, 'data-card')) === CARDS[0].id && (await txt(`${D} [data-score]`)) === '0 of 0 right so far.');
await clickText(D, 'Explore');

// ---- dispatch explorer
check('nothing is revealed before the first prediction', (await page.$(`${X} [data-not-started]`)) !== null && (await page.$(`${X} [data-backend]`)) === null);
const backend = async (l) => ({
  llama: await attr(`${X} [data-backend="llama"] [data-choice]`, 'data-choice'),
  holds: await page.$$eval(`${X} [data-backend="llama"] [data-check]`, (li) => li.map((x) => x.dataset.holds === '1')),
  cols: await attr(`${X} [data-columns]`, 'data-columns'),
  mlx: await attr(`${X} [data-mlx]`, 'data-mlx'),
  ok: true,
});
const expected = (l) => ({
  llama: llamaCuda(l).path,
  holds: llamaCuda(l).checks.map((c) => c.holds),
  cols: !slow2d(l).modelled ? 'unknown' : slow2d(l).requiresColumns ? 'yes' : 'no',
  mlx: mlx(l).path,
});
const same = (a, b) => a.llama === b.llama && JSON.stringify(a.holds) === JSON.stringify(b.holds) && a.cols === b.cols && a.mlx === b.mlx;
const GUESS = { 'lenet-c1': 'direct', tiny: 'direct', pointwise: 'cublas' };
for (const p of DISPATCH_PRESETS) {
  await clickIn(X, `[data-preset="${p.id}"]`);
  const gated = (await page.$(`${X} [data-predict]`)) !== null && (await page.$(`${X} [data-backend]`)) === null;
  check(`${p.label}: the prediction comes first and hides the backends`, gated);
  check(`  the describe explorer above now shows the same layer`, (await describeMatches(p.layer)).ok);
  if (p.id === 'lenet-c1') await shot('predict', X);
  await clickIn(X, `[data-guess="${GUESS[p.id]}"]`);
  const right = GUESS[p.id] === llamaCuda(p.layer).path;
  check(`  guessing "${GUESS[p.id]}" is marked ${right ? 'right' : 'wrong'}`, (await attr(`${X} [data-guess-verdict]`, 'data-guess-verdict')) === (right ? 'right' : 'wrong'));
  const got = await backend(p.layer);
  check(`  llama.cpp runs ${p.expect}; its five conditions, Slow2d and MLX match the model`, got.llama === p.expect && same(got, expected(p.layer)), JSON.stringify(got));
  if (p.id === 'lenet-c1') await shot('dispatch', X);
}
check('the Apple card says no algorithm is exposed', (await txt(`${X} [data-apple]`)).includes('No algorithm is exposed or documented'));
// after the first prediction, edits in the explorer above update the backends directly
const w3 = { N: 1, C: 64, H: 56, W: 56, M: 64, K: 3, pad: 1, stride: 1, dilation: 1 };
await setLayer(w3);
let got = await backend(w3);
check('a padded 3 × 3, 64 → 64 layer: implicit GEMM, columns, an MLX Winograd candidate', same(got, expected(w3)) && got.mlx === 'winograd-candidate' && got.llama === 'implicit', JSON.stringify(got));
await setField('dilation', 2);
got = await backend({ ...w3, dilation: 2 });
check('with dilation 2 Slow2d is reported as not modelled, and MLX says dilation is not in its facts', got.cols === 'unknown' && (await txt(`${X} [data-mlx]`)).includes('does not say whether this test looks at dilation'), JSON.stringify(got));
const both = { N: 1, C: 3, H: 8, W: 8, M: 4, K: 1, pad: 0, stride: 1, dilation: 1 };
await setLayer(both);
got = await backend(both);
check('a small 1 × 1 layer meets both llama.cpp tests, and the explorer will not guess the order', got.llama === 'cublas-or-direct' && (await txt(`${X} [data-backend="llama"]`)).includes('will not guess'), JSON.stringify(got));

// ---- tables
const algos = await page.$$eval('table[data-algos] tr[data-algo]', (t) => t.map((r) => ({ name: r.dataset.algo, hip: r.lastElementChild.textContent })));
check('the algorithm table lists cuDNN\'s eight in the model\'s order', JSON.stringify(algos.map((a) => a.name)) === JSON.stringify(CUDNN_FWD_ALGOS.map((a) => a.name)));
check('  with "none listed" exactly where HIPIFY has no MIOpen counterpart', algos.every((a, i) => (CUDNN_FWD_ALGOS[i].hipify == null) === (a.hip === 'none listed')));
const differ = await page.$$eval('table[data-differ] tr[data-row]', (t) => t.map((r) => r.dataset.row));
check('the backend table has the seven library rows', JSON.stringify(differ) === JSON.stringify(['Vendor library', 'Forward algorithms', 'Choosing one', 'Explicit im2col as an API', 'Layouts', 'Filter flip', 'Precision trap against an FP32 reference']), differ.join(' | '));

// ---- questions
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
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
  return ok && (await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
const QS = [
  ['K = 64', 'K = 64 output maps, M = 3', 'M = 64 output maps, a 3 × 3', 'keeps cuDNN’s letters'],
  ['better memory alignment', 'It rounds each row', 'It adds pad_h zero rows', 'Padding of that kind exists inside cuDNN'],
  ['argues against on p. 467', 'IMPLICIT_GEMM; the workspace holds tiles', 'GEMM; the workspace holds the unfolded input matrix.', 'never forms the matrix'],
  ['needs a workspace that', 'A copy of the unfolded input', 'Precomputed indices into the input', 'That is GEMM, the explicit algorithm'],
  ['differs from', 'Whether cuDNN flipped the filter', 'Whether cuDNN ran the convolution in TF32', 'PyTorch selects cross-correlation'],
  ['Find step', 'Both make results deterministic', 'Both time several algorithms', 'separate flag'],
  ['receives a 1 × 1 convolution', 'The direct kernel, since', 'A cuBLAS GEMM directly on the input.', 'C·K·K = 256 is not below 64'],
];
for (const [part, wrongOpt, rightOpt, why] of QS) {
  const w = await answerChoice(part, wrongOpt);
  const c = await answerChoice(part, rightOpt);
  check(`question "${part}": a wrong option shows its reason, the right one is accepted`, w && w.includes('Not quite.') && w.includes(why) && c && c.includes('Correct.'), `${w} || ${c}`);
}
await pause(600);
check('answers were saved through PUT /api/progress (intercepted)', puts > 0, String(puts));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch15/15.8', '#/ch19/labs', '#/ch19/wrap']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch19/19.4');
let overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone-describe', D);
await clickIn(X, '[data-preset="lenet-c1"]');
await clickIn(X, '[data-guess="implicit"]');
overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('  nor with the dispatch explorer open', overflow <= 1, String(overflow));
await shot('phone-dispatch', X);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
