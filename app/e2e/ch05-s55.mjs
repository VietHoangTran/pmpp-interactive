// node e2e/s55.mjs
// Drives section 5.5 (boundary checks) in the running app: the boundary explorer and the
// rectangular-matrix exercise. Leaves no trace: writes to /api/progress are answered here
// and never reach the server.
import { open, goto } from './browser.mjs';

const { browser, page, errors } = await open();
await page.setRequestInterception(true);
page.on('request', (req) => {
  if (req.method() === 'PUT' && req.url().includes('/api/progress'))
    req.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  else req.continue();
});

let failed = 0;
const ok = (cond, msg) => {
  console.log((cond ? 'ok    ' : 'FAIL  ') + msg);
  if (!cond) failed++;
};
const eq = (got, want, msg) => ok(JSON.stringify(got) === JSON.stringify(want), `${msg}${JSON.stringify(got) === JSON.stringify(want) ? '' : `: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section[aria-label="${title}"]`;

async function pick(root, group, text) {
  await page.evaluate(
    (root, group, text) => {
      const b = [...document.querySelectorAll(`${root} [aria-label="${group}"] button`)].find((x) => x.textContent.trim() === text);
      if (!b) throw new Error(`no button "${text}" in group "${group}"`);
      b.click();
    },
    root, group, text,
  );
  await sleep(60);
}
async function click(sel) {
  await page.evaluate((s) => {
    const el = document.querySelector(s);
    if (!el) throw new Error('nothing matches ' + s);
    el.click();
  }, sel);
  await sleep(60);
}
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? null, sel);
const attr = (sel, name) => page.evaluate((s, n) => document.querySelector(s)?.getAttribute(n) ?? null, sel, name);

const question = (snippet) =>
  page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((q) => q.querySelector('.prompt')?.textContent.includes(t)), snippet);
async function answerNumeric(snippet, value) {
  const q = await question(snippet);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await sleep(80);
  return q.evaluate((el) => el.querySelector('.verdict')?.textContent ?? '');
}
async function answerChoice(snippet, index) {
  const q = await question(snippet);
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await sleep(80);
  return q.evaluate((el) => el.querySelector('.verdict')?.textContent ?? '');
}

try {
  await goto(page, 'ch05/5.5');

  // ---------------------------------------------------------------- the prediction
  {
    const q = await question('Which index of the array does it read');
    const input = await q.$('input[aria-label="your prediction"]');
    if (input) {
      await input.type('3');
      await (await q.$('button.primary')).click();
      await sleep(80);
    }
    const reveal = await q.evaluate((el) => el.querySelector('.verdict')?.textContent ?? '');
    ok(reveal.includes('M[1][0]'), 'prediction reveals that index 3 is M[1][0]');
  }

  // ---------------------------------------------------------------- boundary explorer
  const B = bench('Boundary explorer');
  const OPTS = ['guardM', 'guardN', 'guardP', 'zeroFillM', 'zeroFillN'];
  /** Sets the five switches; checks first, because a zero-write is disabled without its check. */
  async function setOpts(on) {
    for (const k of OPTS) {
      const sel = `${B} input[data-opt="${k}"]`;
      const now = await page.evaluate((s) => document.querySelector(s).checked, sel);
      if (now !== on.includes(k)) {
        if (await page.evaluate((s) => document.querySelector(s).disabled, sel)) throw new Error(k + ' is disabled');
        await click(sel);
      }
    }
  }
  const verdict = async () => ({ status: await attr(`${B} [data-verdict]`, 'data-verdict'), text: await text(`${B} [data-verdict]`) });
  const cell = (id) =>
    page.evaluate(
      (r, id) => {
        const e = document.querySelector(`${r} [data-cell="${id}"]`);
        return e ? { ...e.dataset, text: e.textContent.trim() } : null;
      },
      B, id,
    );
  const load = (tx, ty, a) =>
    page.evaluate(
      (r, k, a) => {
        const td = document.querySelector(`${r} tr[data-thread="${k}"] td[data-load="${a}"]`);
        return td ? { access: td.dataset.access, text: td.textContent.replace(/\s+/g, ' ').trim() } : null;
      },
      B, `${tx}-${ty}`, a,
    );
  const wrongCells = () => page.evaluate((r) => document.querySelectorAll(`${r} [data-cell^="P-"][data-wrong="1"]`).length, B);
  const kinds = () => page.evaluate((r) => [...document.querySelectorAll(`${r} [data-kind]`)].map((e) => e.dataset.kind), B);

  // Starts on the book's example with everything off.
  eq(await page.evaluate((r) => [...document.querySelectorAll(`${r} input[data-opt]`)].map((e) => e.checked), B), [false, false, false, false, false], 'all five switches start off');
  ok((await text(`${B} > header`)).includes('fewest options'), 'the task is shown in the bench header');
  let v = await verdict();
  ok(v.status === 'wrong' && /P is wrong in 9 of 9 elements/.test(v.text), 'W=3, T=2, everything off: the verdict is "wrong"');
  ok(/fetched a wrong element/.test(v.text), 'the verdict mentions loads that fetched a wrong element');
  eq(await kinds(), ['wrong-element', 'outside-read', 'bad-store'], 'violations are grouped by kind');
  ok((await text(`${B} [data-kind="wrong-element"]`)).includes('asked for M[0][3], which does not exist, and got M[1][0] instead'), 'the first wrong-element message is the M[0][3] to M[1][0] wrap');
  eq(await wrongCells(), 9, 'all nine elements of P are marked wrong');
  eq((await cell('P-0-0')).text, 'NaN', 'a poisoned element is shown as NaN');

  // The padded grid.
  eq(await page.evaluate((r) => ['M', 'N', 'P'].map((a) => [document.querySelectorAll(`${r} [data-cell^="${a}-"]`).length, document.querySelectorAll(`${r} [data-cell^="${a}-"][data-exists="0"]`).length]), B),
    [[16, 7], [16, 7], [16, 7]], 'each 3×3 matrix is drawn on a 4×4 grid with 7 positions marked as not part of it');

  // Block (0,0), phase 1: the book's Fig. 5.11 situation (preselected).
  ok((await text(`${B} [data-readout="selection"]`)).includes('Block (x 0, y 0), phase 1: 4 of 8'), 'block (0,0), phase 1 is selected: 4 of 8 loads out of range');
  let l = await load(1, 0, 'M');
  ok(l.access === 'wrap' && l.text.includes('M[0][3]') && l.text.includes('fetched M[1][0]'), 'thread tx=1, ty=0 asks for M[0][3] and is shown fetching M[1][0]');
  l = await load(0, 1, 'N');
  ok(l.access === 'outside' && l.text.includes('N[3][0]') && l.text.includes('beyond the end of the array'), 'thread tx=0, ty=1 asks for N[3][0], beyond the end of the array');
  eq((await load(0, 0, 'M')).access, 'ok', 'thread tx=0, ty=0 loads M[0][2] normally');
  let c = await cell('M-0-3');
  ok(c.exists === '0' && c.access === 'wrap', 'M position (0,3) is off the matrix and marked as a wrapping load');
  eq((await cell('M-1-0')).got, '1', 'M[1][0] is marked as the element actually fetched');
  eq((await cell('N-3-0')).access, 'outside', 'N position (3,0) is marked as outside the array');

  // Block (1,1), phase 0: not only the last phase (Fig. 5.12), and the thread with no output that still loads.
  await click(`${B} button[aria-label="block x 1, y 1"]`);
  await pick(B, 'Phase', '0');
  ok((await text(`${B} [data-readout="selection"]`)).includes('Block (x 1, y 1), phase 0: 4 of 8'), 'block (1,1), phase 0: 4 of 8 loads out of range');
  eq([(await load(0, 1, 'M')).access, (await load(1, 1, 'M')).access, (await load(1, 0, 'N')).access, (await load(1, 1, 'N')).access], ['outside', 'outside', 'wrap', 'wrap'],
    'block (1,1), phase 0: two M loads leave the array, two N loads wrap');
  l = await load(1, 0, 'M');
  ok(l.access === 'ok' && l.text.includes('M[2][1]'), 'thread tx=1, ty=0 of block (1,1) loads M[2][1]');
  eq(await attr(`${B} tr[data-thread="1-0"] td[data-owns]`, 'data-owns'), '0', '... although it owns no element of P');

  // Checks without zero-writes: loads are skipped, slots keep what they had, P is still wrong.
  await setOpts(['guardM', 'guardN', 'guardP']);
  await click(`${B} [data-preset="book"]`);
  v = await verdict();
  ok(v.status === 'wrong' && /nothing had filled/.test(v.text), 'three checks but no zero-writes: still wrong, slots used that nothing filled');
  l = await load(1, 0, 'M');
  ok(l.access === 'skipped' && l.text.includes('slot left as it was'), 'the M[0][3] load is skipped and its slot left as it was');
  eq((await cell('M-0-3')).access, 'skipped', 'M position (0,3) is marked as skipped');

  // All five on: correct at every width from 2 to 7 with both tiles.
  await setOpts(OPTS);
  l = await load(1, 0, 'M');
  ok(l.access === 'skipped' && l.text.includes('slot set to 0'), 'with the zero-write on, the skipped load sets its slot to 0');
  const all = [];
  for (const t of [2, 3])
    for (const w of [2, 3, 4, 5, 6, 7]) {
      await pick(B, 'Tile T', String(t));
      await pick(B, 'Width W', String(w));
      const r = await verdict();
      all.push(`W=${w},T=${t}:${r.status}:${await wrongCells()}`);
    }
  eq(all, [2, 3].flatMap((t) => [2, 3, 4, 5, 6, 7].map((w) => `W=${w},T=${t}:correct:0`)), 'all five options on: correct for every W from 2 to 7 with T = 2 and T = 3');
  const sweep = () => page.evaluate((r) => [...document.querySelectorAll(`${r} [data-sweep]`)].map((e) => `${e.dataset.sweep}:${e.dataset.status}`), B);
  ok((await sweep()).length === 14 && (await sweep()).every((s) => s.endsWith(':correct')), 'the sweep strip shows all 14 cases (W = 1 to 7, both tiles) correct');

  // One zero-write only, either side: right on the book's example, NaN when the matrix is narrower than a tile.
  for (const [only, other] of [['zeroFillN', 'Mds'], ['zeroFillM', 'Nds']]) {
    await setOpts(['guardM', 'guardN', 'guardP', only]);
    await click(`${B} [data-preset="book"]`);
    v = await verdict();
    ok(v.status === 'lucky' && /P is correct, but not safely/.test(v.text) && /nothing had filled/.test(v.text) && (await wrongCells()) === 0,
      `only ${only}, W=3, T=2: P is correct, and the unfilled ${other} slots are still reported`);
    await click(`${B} [data-preset="narrow"]`);
    v = await verdict();
    ok(v.status === 'wrong' && /P is wrong in 1 of 1/.test(v.text) && (await cell('P-0-0')).text === 'NaN', `only ${only}, W=1, T=2: P[0][0] is NaN`);
    eq(await attr(`${B} [data-sweep="2-3"]`, 'data-status'), 'wrong', `only ${only}: the sweep also shows W=2, T=3 wrong`);
    eq(await attr(`${B} [data-sweep="3-2"]`, 'data-status'), 'lucky', `only ${only}: the sweep shows W=3, T=2 as correct but flagged`);
  }
  // The sweep buttons navigate.
  await click(`${B} [data-sweep="5-3"]`);
  ok((await text(`${B} [data-verdict]`)).includes('W = 5, T = 3'), 'clicking a sweep entry opens that width and tile');

  // Without the store check, a thread with no element of P writes over a real one.
  await setOpts(['guardM', 'guardN', 'zeroFillM', 'zeroFillN']);
  await click(`${B} [data-preset="book"]`);
  v = await verdict();
  ok(v.status === 'wrong' && /own no element of P/.test(v.text), 'no check on the store: P is wrong, stores from threads that own no element');

  // ---------------------------------------------------------------- rectangular exercise
  const R = bench('Which dimension did Width mean?');
  const ANSWERS = ['j', 'k', 'k', 'l', 'k', 'l', 'l', 'k', 'l', 'j'];
  const selects = await page.$$(`${R} select`);
  eq(selects.length, 10, 'ten uses of Width to map');
  const results = () => page.evaluate((r) => [...document.querySelectorAll(`${r} tr[data-use]`)].map((e) => e.dataset.result), R);
  const kernelShown = () => page.evaluate(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('gemmKernel')));

  for (let i = 0; i < 10; i++) await selects[i].select(i === 5 ? 'k' : ANSWERS[i]); // one deliberate mistake: the stride of B
  await sleep(60);
  eq(await results(), Array(10).fill(''), 'no feedback before Check');
  await page.evaluate((r) => [...document.querySelectorAll(`${r} button`)].find((b) => b.textContent.trim() === 'Check').click(), R);
  await sleep(80);
  eq(await results(), ANSWERS.map((_, i) => (i === 5 ? 'wrong' : 'right')), 'per-row feedback: nine right, the B stride wrong');
  ok((await text(`${R} tr[data-use="5"]`)).includes('How long is a row of B?'), 'the wrong row gets a nudge, not the answer');
  ok((await text(`${R} [data-readout="rect"]`)).includes('9 of 10'), 'summary says 9 of 10');
  ok(!(await kernelShown()), 'the reference kernel stays hidden');

  await selects[5].select('l');
  await sleep(60);
  eq((await results())[5], '', 'changing an answer clears its feedback until the next check');
  await page.evaluate((r) => [...document.querySelectorAll(`${r} button`)].find((b) => b.textContent.trim() === 'Check').click(), R);
  await sleep(80);
  eq(await results(), Array(10).fill('right'), 'all ten rows right');
  ok((await text(`${R} [data-readout="rect"]`)).includes('All 10 right'), 'success message: all 10 right');
  await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('gemmKernel')), { timeout: 5000 });
  ok(await kernelShown(), 'the lab 5.3 reference kernel is shown once the mapping is right');
  const marked = await page.evaluate(() => {
    const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('gemmKernel'));
    return [...f.querySelectorAll('.ln.hl')].map((e) => e.textContent.trim());
  });
  ok(marked.some((x) => x.startsWith('As[ty][tx] = (Row < j && aCol < k)')) && marked.some((x) => x.startsWith('C[Row * l + Col]')) && marked.length >= 8,
    `its lines that use j, k or l are marked (${marked.length} lines)`);

  // ---------------------------------------------------------------- questions
  ok(/Correct\./.test(await answerChoice('A colleague tidies', 1)), 'question: no early return, the thread still loads for its neighbours');
  ok(/Not quite/.test(await answerChoice('A colleague tidies', 0)), 'question: "contributes nothing" is rejected');
  ok(/Correct\./.test(await answerChoice('Two 5×5 matrices', 1)), 'question: row overrun is silent, column overrun leaves the array');
  ok(/Correct\./.test(await answerNumeric('3×3 matrices, 2×2 tiles, no bounds checks', 4)), 'question: 4 of block (1,1)\'s 8 loads in phase 0 are out of range');
} catch (e) {
  ok(false, 'script error: ' + (e?.stack ?? e));
}

for (const e of errors) ok(false, e);
await browser.close();
console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
