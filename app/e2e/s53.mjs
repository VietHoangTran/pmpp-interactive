// node e2e/s53.mjs
// Drives section 5.3 (tiling) in the running app and asserts what its two interactives show.
// Leaves no trace: writes to /api/progress are answered here and never reach the server.
import fs from 'node:fs';
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

/** Clicks the button with this exact text inside the labelled group of a bench. */
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
/** Clicks the first button of a bench whose text or aria-label is exactly this. */
async function press(root, name) {
  await page.evaluate(
    (root, name) => {
      const b = [...document.querySelectorAll(`${root} button`)].find((x) => x.textContent.trim() === name || x.getAttribute('aria-label') === name);
      if (!b) throw new Error(`no button "${name}"`);
      b.click();
    },
    root, name,
  );
  await sleep(60);
}
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? null, sel);

/** The question block (.q) whose prompt contains this text. */
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
  await goto(page, 'ch05/5.3');

  // ---------------------------------------------------------------- the prediction
  {
    const q = await question('8×8 matrices, 4×4 blocks');
    const input = await q.$('input[aria-label="your prediction"]');
    if (input) {
      await input.type('16');
      await (await q.$('button.primary')).click();
      await sleep(80);
    }
    const reveal = await q.evaluate((el) => el.querySelector('.verdict')?.textContent ?? '');
    ok(/Four\./.test(reveal), 'prediction reveals the answer (four) once a number is committed');
  }

  // ---------------------------------------------------------------- who reads what
  const B1 = bench('Who reads what');
  const totals = () =>
    page.evaluate((r) => Object.fromEntries([...document.querySelectorAll(`${r} [data-total]`)].map((e) => [e.dataset.total, Number(e.textContent)])), B1);
  /** Cells of M or N that the selection reads, as "r-c:count". */
  const reads = (a) =>
    page.evaluate(
      (r, a) => [...document.querySelectorAll(`${r} [data-cell^="${a}-"]`)].filter((e) => e.dataset.count !== '0').map((e) => `${e.dataset.cell.slice(2)}:${e.dataset.count}`),
      B1, a,
    );
  const range = (n) => Array.from({ length: n }, (_, i) => i);

  await press(B1, 'P[0][1]');
  eq(await reads('M'), range(4).map((c) => `0-${c}:1`), 'W=4, one thread, P[0][1]: row 0 of M is highlighted, each element once');
  eq(await reads('N'), range(4).map((r) => `${r}-1:1`), 'W=4, one thread, P[0][1]: column 1 of N is highlighted, each element once');
  eq(await totals(), { loads: 8, distinct: 8, ratio: 1 }, 'one thread: 8 loads, 8 distinct elements, ratio 1');

  await pick(B1, 'Show the reads of', 'its whole block');
  eq(await reads('M'), [0, 1].flatMap((r) => range(4).map((c) => `${r}-${c}:2`)), 'W=4, block 2, block (0,0): rows 0 and 1 of M are each read by 2 threads');
  eq(await reads('N'), range(4).flatMap((r) => [0, 1].map((c) => `${r}-${c}:2`)), 'W=4, block 2, block (0,0): columns 0 and 1 of N are each read by 2 threads');
  eq(await totals(), { loads: 32, distinct: 16, ratio: 2 }, 'W=4, block 2: 32 loads for 16 distinct elements, ratio 2');
  ok((await text(`${B1} [data-cell="M-0-0"]`)).includes('×2'), 'the count is drawn on the element (×2)');

  await pick(B1, 'Width W', '8');
  ok(await page.evaluate((r) => [...document.querySelectorAll(`${r} [aria-label="Block size"] button`)].find((b) => b.textContent.trim() === '3').disabled, B1),
    'W=8: block size 3 is not offered (not a divisor)');
  await pick(B1, 'Block size', '4');
  await press(B1, 'P[5][2]');
  eq(await reads('M'), [4, 5, 6, 7].flatMap((r) => range(8).map((c) => `${r}-${c}:4`)), 'W=8, block 4, clicking P[5][2] selects block (x 0, y 1): rows 4 to 7 of M, each element ×4');
  eq(await reads('N'), range(8).flatMap((r) => range(4).map((c) => `${r}-${c}:4`)), 'W=8, block 4, block (x 0, y 1): columns 0 to 3 of N, each element ×4');
  eq(await totals(), { loads: 256, distinct: 64, ratio: 4 }, 'W=8, block 4: 256 loads for 64 distinct elements, ratio 4 (the prediction)');
  ok((await text(`${B1} [data-readout="who"]`)).includes('Block (x 0, y 1)'), 'the readout names the block (x 0, y 1)');
  eq(await page.evaluate((r) => [...document.querySelectorAll(`${r} button[aria-label^="P["][aria-pressed="true"]`)].length, B1), 16, 'all 16 elements of the selected block are marked in P');

  for (const [w, sizes] of [[4, [2, 4]], [6, [2, 3]], [8, [2, 4]]]) {
    await pick(B1, 'Width W', String(w));
    for (const b of sizes) {
      await pick(B1, 'Block size', String(b));
      await press(B1, `P[${w - 1}][${w - 1}]`);
      eq(await totals(), { loads: 2 * b * b * w, distinct: 2 * b * w, ratio: b }, `W=${w}, block ${b}, last block: ${2 * b * b * w} loads, ${2 * b * w} distinct, ratio ${b}`);
    }
  }
  await pick(B1, 'Width W', '6');
  await pick(B1, 'Show the reads of', 'one thread');
  await press(B1, 'P[4][1]');
  eq(await reads('M'), range(6).map((c) => `4-${c}:1`), 'W=6, one thread, P[4][1]: row 4 of M');
  eq(await reads('N'), range(6).map((r) => `${r}-1:1`), 'W=6, one thread, P[4][1]: column 1 of N');
  eq(await totals(), { loads: 12, distinct: 12, ratio: 1 }, 'W=6, one thread: 12 loads, ratio 1');

  // ---------------------------------------------------------------- phases
  const B2 = bench('Phases');
  const row = (tx, ty) =>
    page.evaluate(
      (r, k) => {
        const tr = document.querySelector(`${r} tr[data-thread="${k}"]`);
        return tr ? Object.fromEntries([...tr.querySelectorAll('[data-col]')].map((e) => [e.dataset.col, e.textContent.replace(/\s+/g, ' ').trim()])) : null;
      },
      B2, `${tx}-${ty}`,
    );
  const inTile = (a) =>
    page.evaluate((r, a) => [...document.querySelectorAll(`${r} [data-cell^="${a}-"][data-in-tile="1"]`)].map((e) => e.dataset.cell), B2, a);
  const slots = () =>
    page.evaluate((r) => [...document.querySelectorAll(`${r} [data-slot]`)].map((e) => ({ slot: e.dataset.slot, value: Number(e.dataset.value), writes: Number(e.dataset.writes), text: e.textContent })), B2);
  const disabled = (name) =>
    page.evaluate((r, n) => [...document.querySelectorAll(`${r} button`)].find((b) => b.textContent.trim() === n).disabled, B2, name);
  const phaseText = () => text(`${B2} [data-readout="phase"]`);
  /** M × N worked out here from the numbers drawn in the two global matrices. */
  const product = () =>
    page.evaluate((r) => {
      const grid = (a) => [...document.querySelectorAll(`${r} [data-cell^="${a}-"]`)].map((e) => Number(e.querySelector('b').textContent));
      const M = grid('M'), N = grid('N'), w = Math.sqrt(M.length);
      return { M, N, w };
    }, B2);

  ok((await phaseText()).startsWith('Phase 0 of 2'), '4×4, tile 2 starts at phase 0 of 2');
  ok(await disabled('Previous phase'), 'Previous is disabled at phase 0');
  eq((await row(1, 0)).m, 'M[0][1] → Mds[0][1]', 'phase 0: thread tx=1, ty=0 loads M[0][1] into Mds[0][1]');
  eq(await inTile('M'), ['M-0-0', 'M-0-1', 'M-1-0', 'M-1-1'], 'phase 0: the M tile is rows 0-1, columns 0-1');
  const first = await slots();
  eq(first.length, 8, 'two 2×2 shared tiles: 8 slots');
  ok(first.every((s) => s.writes === 1), 'phase 0: every slot has been written once');

  await press(B2, 'Next phase');
  ok((await phaseText()).startsWith('Phase 1 of 2'), 'Next goes to phase 1');
  const r10 = await row(1, 0);
  eq(r10.m, 'M[0][3] → Mds[0][1]', 'phase 1: thread tx=1, ty=0 loads M[0][3] into Mds[0][1]');
  eq(r10.n, 'N[2][1] → Nds[0][1]', 'phase 1: thread tx=1, ty=0 loads N[2][1] into Nds[0][1]');
  eq(await inTile('M'), ['M-0-2', 'M-0-3', 'M-1-2', 'M-1-3'], 'phase 1: the M tile moved one tile to the right');
  eq(await inTile('N'), ['N-2-0', 'N-2-1', 'N-3-0', 'N-3-1'], 'phase 1: the N tile moved one tile down');
  const second = await slots();
  eq(second.map((s) => s.slot), first.map((s) => s.slot), 'phase 1 uses the same 8 slots');
  ok(second.every((s) => s.writes === 2), 'phase 1: every slot has been written twice (overwritten)');
  const s01 = second.find((s) => s.slot === 'Mds-0-1');
  const old01 = first.find((s) => s.slot === 'Mds-0-1');
  ok(s01.value !== old01.value && s01.text.includes('from M[0][3]') && s01.text.includes('was ' + old01.value),
    `Mds[0][1] now holds M[0][3] = ${s01.value} and shows the value it replaced (${old01.value})`);
  ok(await disabled('Next phase'), 'Next is disabled at the last phase');

  // The running sums at the last phase must be M × N, recomputed here from the numbers on screen.
  {
    const { M, N, w } = await product();
    const mine = (r, c) => range(w).reduce((s, k) => s + M[r * w + k] * N[k * w + c], 0);
    const shown = [];
    const want = [];
    for (const [tx, ty] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const sum = (await row(tx, ty)).sum;
      shown.push(sum.replace(/^.*= (\d+) = (P\[\d\]\[\d\])$/, '$2=$1'));
      want.push(`P[${ty}][${tx}]=${mine(ty, tx)}`);
    }
    eq(shown, want, 'after the last phase each running sum equals its element of M × N (recomputed from the matrices shown)');
  }

  await press(B2, 'Previous phase');
  ok((await phaseText()).startsWith('Phase 0 of 2'), 'Previous goes back to phase 0');
  eq((await slots()).find((s) => s.slot === 'Mds-0-1').writes, 1, 'back in phase 0 the slot shows one write');

  await pick(B2, 'Problem', '6×6, tile 2');
  ok((await phaseText()).startsWith('Phase 0 of 3'), '6×6, tile 2: three phases, starting at 0');
  await press(B2, 'Next phase');
  await press(B2, 'Next phase');
  ok((await phaseText()).startsWith('Phase 2 of 3'), 'stepped to phase 2');
  const r01 = await row(0, 1);
  eq([r01.m, r01.n], ['M[1][4] → Mds[1][0]', 'N[5][0] → Nds[1][0]'], '6×6, tile 2, phase 2: thread tx=0, ty=1 loads M[1][4] and N[5][0] into slot [1][0]');
  const third = await slots();
  ok(third.length === 8 && third.every((s) => s.writes === 3), 'still 8 slots, each written three times');

  await pick(B2, 'Problem', '6×6, tile 3');
  ok((await phaseText()).startsWith('Phase 0 of 2'), '6×6, tile 3: two phases, and the stepper restarts at 0');
  eq((await slots()).length, 18, 'two 3×3 shared tiles: 18 slots');
  eq(await page.evaluate((r) => document.querySelectorAll(`${r} tr[data-thread]`).length, B2), 9, 'nine threads listed');
  eq((await row(2, 1)).n, 'N[1][2] → Nds[1][2]', '6×6, tile 3, phase 0: thread tx=2, ty=1 loads N[1][2] into Nds[1][2]');

  // ---------------------------------------------------------------- questions
  ok(/Correct\./.test(await answerNumeric('Two 96×96 matrices', 6)), 'question: 96×96 with 16×16 tiles, 6 requests per element is accepted');
  ok(/Not quite/.test(await answerNumeric('Two 96×96 matrices', 96)), 'question: the untiled count (96) is rejected');
  ok(/Correct\./.test(await answerChoice('Matrix addition', 2)), 'question: matrix addition has nothing to share');
  ok(/Correct\./.test(await answerNumeric('8×8 matrices, 4×4 tiles, every block', 256)), 'question: 8×8 with 4×4 tiles, 256 loads in total is accepted');

  // ---------------------------------------------------------------- the narrated scene is mounted in the right place
  {
    const src = fs.readFileSync(new URL('../src/chapters/ch05/S53.tsx', import.meta.url), 'utf8');
    const at = src.indexOf('<WhyTilingScene />');
    ok(at > 0 && at < src.indexOf('<PhasesBench />', at), 'source: the why-tiling scene sits before the Phases interactive');
  }
} catch (e) {
  ok(false, 'script error: ' + (e?.stack ?? e));
}

for (const e of errors) ok(false, e);
await browser.close();
console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
