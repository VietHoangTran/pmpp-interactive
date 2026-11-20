// Browser check for section 20.5 (Flash attention): both narrated scenes, the online-softmax
// stepper, the four explorers (fit, causal tiles, lane ownership, banks), the kernel stepper
// with its defects, the outcomes table, the lab listing, two questions, and the page at
// phone width. Every number asserted is computed from the tested models
// (src/sim/onlinesoftmax.ts, flashplan.ts, flashkernel.ts).
// Usage: node e2e/ch20-s205.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
// The models the page runs on. Node strips the types itself; on a Node too old for that,
// take the same modules as the dev server compiles them for the browser.
const load = async (p) => {
  try {
    return await import('../' + p);
  } catch {
    const code = await (await fetch(BASE + '/' + p)).text();
    return import('data:text/javascript;charset=utf-8,' + encodeURIComponent(code));
  }
};
const os = await load('src/sim/onlinesoftmax.ts');
const fk = await load('src/sim/flashkernel.ts');
// flashplan.ts imports its neighbours without file extensions, which Node does not resolve;
// its expected values are computed in the page, from the module the dev server serves.

const out = process.argv[2];
const LAB_FILES = ['kernel.cu', 'kernel.metal', 'launch.mm'].map((f) => `ch20/lab2_flash/solution/${f}`);
const missingLab = LAB_FILES.filter((f) => !fs.existsSync(path.join(repo, 'labs', f)));

const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one.
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1}' });
});
const notFound = [];
page.on('response', (r) => { if (r.status() === 404) notFound.push(decodeURIComponent(r.url())); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickIn = async (scope, label, exact = true) => {
  const done = await page.evaluate(
    (scope, label, exact) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => (exact ? b.textContent.trim() === label : b.textContent.includes(label)));
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope, label, exact,
  );
  await pause(40);
  return done;
};
const select = async (scope, aria, value) => {
  await page.evaluate(
    (scope, aria, value) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const s = root.querySelector(`select[aria-label="${aria}"]`);
      s.value = String(value);
      s.dispatchEvent(new Event('change', { bubbles: true }));
    },
    scope, aria, value,
  );
  await pause(60);
};
const setInput = async (scope, aria, value) => {
  await page.evaluate(
    (scope, aria, value) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const el = root.querySelector(`input[aria-label="${aria}"]`);
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    scope, aria, value,
  );
  await pause(60);
};
const clickUntil = async (scope, label, max = 400) => {
  for (let i = 0; i < max; i++) if (!(await clickIn(scope, label))) return i;
  return max;
};

await goto(page, 'ch20/20.5');
await page.waitForSelector(bench('Step one block'));
const fp = await page.evaluate(async () => {
  const m = await import('/src/sim/flashplan.ts');
  const t4096 = m.tileClasses(4096, 32, 32);
  const t100 = m.tileClasses(100, 32, 32);
  return {
    book: m.plan(m.BOOK).bytes.total,
    bc16: m.plan({ ...m.BOOK, bc: 16 }).bytes.total,
    d64: m.plan({ ...m.BOOK, d: 64 }).bytes.total,
    skip4096: t4096.counts.skip,
    frac4096: (t4096.counts.skip / t4096.total).toFixed(3),
    t100: { tr: t100.tr, tc: t100.tc, mask: t100.counts.mask, skip: t100.counts.skip },
    wrong5: m.wave64Broadcast(5, 64).filter((x) => !x.ok).map((x) => x.hw),
    src40w64: m.hipShuffleSource(40, 5, 64),
    src40w32: m.hipShuffleSource(40, 5, 32),
    pass32: m.ktStoreBanks(32, 0, false).passes,
    pass32p: m.ktStoreBanks(32, 0, true).passes,
    pass64p: m.ktStoreBanks(64, 0, true).passes,
  };
});

// ---------------------------------------------------------------- scene: online merge
{
  const S = 'section.scene[aria-label="Merging softmax one tile at a time"]';
  const seg = await page.$$(S + ' .seg button');
  check('online-merge: six parts', seg.length === 6, String(seg.length));
  const capBox = await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
  check('online-merge: captions option exists and is off by default', capBox === false);
  const script1 = fs.readFileSync(path.join(repo, 'content/ch20/narration/online-merge-1.txt'), 'utf8').trim();
  await pause(300);
  check('online-merge: with no audio the part 1 script is shown', (await text(S + ' .caption')) === script1.replace(/\s+/g, ' '));
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .seg button')][2].click(), S);
  await pause(120);
  await page.$eval(S + ' input[type=range]', (el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '500');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await pause(120);
  const scene = os.runOnline({ scores: os.DEMO_SCORES, V: os.V12, bc: 4, row: 11, order: 'ltr', init: 'neg-inf', form: 'deferred' });
  const svg = await text(S + ' svg');
  check('online-merge part 3: the max line reads 2.9', svg.includes('m = 2.9'), svg.slice(0, 200));
  check('online-merge part 3: the factor shown is the model’s', svg.includes('= ' + String(Number(scene.steps[1].aFactor.toPrecision(4)))), svg);
  if (out) await page.$eval(S, (e) => e.scrollIntoView()) , await page.screenshot({ path: out + '-scene-merge.png' });
}

// ---------------------------------------------------------------- online softmax stepper
{
  const B = 'Online softmax, one tile at a time';
  const cfg = { scores: os.presetScores('demo'), V: os.V12, bc: 4, row: 11, order: 'ltr', init: 'neg-inf', form: 'deferred' };
  const t = os.runOnline(cfg);
  const fv = (x) => (x === 1 ? '1' : String(Number(x.toPrecision(4))).replace('-', '−'));
  check('stepper starts with nothing merged', (await text(bench(B) + ' [data-progress]')) === 'tile 0 of 3 merged');
  await clickIn(B, 'Step');
  await clickIn(B, 'Step');
  check('step 2: the max rose to 2.9', (await text(bench(B) + ' [data-v="mAB"]')) === '2.9');
  check('step 2: the A-term factor is the model’s e^(1.6 − 2.9)', (await text(bench(B) + ' [data-v="aFactor"]')) === fv(t.steps[1].aFactor), await text(bench(B) + ' [data-v="aFactor"]'));
  check('step 2: narration says the running max rises', (await text(bench(B) + ' [data-narrate]')).includes('rises from 1.6 to 2.9'));
  await clickIn(B, 'Step');
  check('step 3: the factor is exactly 1', (await text(bench(B) + ' [data-v="aFactor"]')) === '1');
  check('step 3: final result equals exact attention', (await text(bench(B) + ' [data-final]')).includes('Equals exact attention over columns 0–11'));
  check('history has three rows, every check matches', (await page.$$eval(bench(B) + ' [data-history] [data-check="ok"]', (e) => e.length)) === 3);
  if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-online-stepper.png' });

  // every width and every order lands on the same answer
  let allOk = true;
  for (const w of [1, 2, 3, 6, 12])
    for (const order of ['ltr', 'rtl', 'shuffled']) {
      await page.evaluate((sel, w) => [...document.querySelectorAll(sel + ' [aria-label="Tile width"] button')].find((b) => b.textContent === String(w)).click(), bench(B), w);
      await pause(30);
      await select(B, 'Tile order', order);
      await clickUntil(B, 'Step', 20);
      const fin = await text(bench(B) + ' [data-final]');
      if (!fin.includes('Equals exact attention')) { allOk = false; console.log('   ', w, order, fin); }
    }
  check('every tile width and order gives exact attention', allOk);

  // the start value
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [aria-label="Tile width"] button')].find((b) => b.textContent === '4').click(), bench(B));
  await select(B, 'Tile order', 'rtl');
  await setInput(B, 'Causal row', 2);
  await select(B, 'Running max starts at', 'neg-inf');
  await clickIn(B, 'Step');
  check('row 2, right to left, minus infinity: first tile is fully masked and the factor is NaN', (await text(bench(B) + ' [data-v="aFactor"]')) === 'NaN');
  await clickUntil(B, 'Step', 10);
  check('… and the result is NaN', (await text(bench(B) + ' [data-final]')).includes('The result is NaN.'));
  const nanT = os.runOnline({ ...cfg, row: 2, order: 'rtl' });
  check('the model agrees it is NaN', nanT.nan === true);
  await select(B, 'Running max starts at', 'sentinel');
  await clickUntil(B, 'Step', 10);
  check('the same order from -FLT_MAX/2 is exact', (await text(bench(B) + ' [data-final]')).includes('Equals exact attention over columns 0–2'));
  await select(B, 'Form', 'normalised');
  await clickIn(B, 'Step');
  check('normalised form skips the empty tile', (await text(bench(B) + ' [data-narrate]')).includes('skips it'));
  await select(B, 'Data', 'offset');
  check('+100 data: e^(largest) overflows float32', (await text(bench(B))).includes('in float32 is ∞'));
}

// ---------------------------------------------------------------- fit the tiles
{
  const B = 'Fit the tiles';
  const v = (k) => text(bench(B) + ` [data-v="${k}"]`);
  check('fit: the book’s 37,376 bytes', fp.book === 37376 && (await v('total')) === fp.book.toLocaleString('en-US') + ' B', await v('total'));
  check('fit: the book’s 20 floats of registers', (await v('regs')) === '20 floats');
  const fitsOf = (id) => page.$eval(bench(B) + ` [data-limit="${id}"]`, (e) => e.dataset.fits);
  check('fit: CUDA 48 KB and AMD LDS fit', (await fitsOf('cuda')) === 'yes' && (await fitsOf('amd')) === 'yes');
  check('fit: Metal does not fit', (await fitsOf('metal')) === 'no');
  check('fit: largest B_c for the Metal limit is 16', (await text(bench(B) + ' [data-best-bc]')).endsWith('is 16.'));
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [aria-label="B_c"] button')].find((b) => b.textContent === '16').click(), bench(B));
  await pause(40);
  check('fit: B_c 16 is 18,688 bytes and fits Metal', fp.bc16 === 18688 && (await v('total')) === '18,688 B' && (await fitsOf('metal')) === 'yes');
  await page.evaluate((sel) => {
    [...document.querySelectorAll(sel + ' [aria-label="B_c"] button')].find((b) => b.textContent === '32').click();
    [...document.querySelectorAll(sel + ' [aria-label="Head dimension d"] button')].find((b) => b.textContent === '64').click();
  }, bench(B));
  await pause(40);
  check('fit: d 64 is 20,736 bytes', fp.d64 === 20736 && (await v('total')) === '20,736 B');
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [aria-label="B_r"] button')].find((b) => b.textContent === '8').click(), bench(B));
  await pause(40);
  check('fit: B_r 8 with 16 warps is refused', (await page.$(bench(B) + ' [data-problems]')) !== null && (await text(bench(B) + ' [data-problems]')).includes('whole number of rows'));
  await clickIn(B, 'The book’s example');
  check('fit: the book’s example restores 37,376', (await v('total')) === '37,376 B');
}

// ---------------------------------------------------------------- causal tiles
{
  const B = 'Which tiles does causality empty?';
  const cnt = (k) => page.$eval(bench(B) + ` [data-class="${k}"] td.num`, (e) => e.textContent);
  check('causal: 8,128 empty pairs at N = 4096', fp.skip4096 === 8128 && (await cnt('skip')) === '8,128');
  check('causal: the fraction is 0.496', fp.frac4096 === '0.496' && (await text(bench(B) + ' [data-fraction]')).includes('= 0.496 of the pairs'));
  await setInput(B, 'Sequence length exactly', 100);
  check('causal: N = 100 gives a 4 × 4 grid', (await page.$eval(bench(B) + ' svg', (e) => e.dataset.tilegrid)) === `${fp.t100.tr}x${fp.t100.tc}`);
  check('causal: counts follow the model at N = 100', (await cnt('mask')) === String(fp.t100.mask) && (await cnt('skip')) === String(fp.t100.skip));
  await setInput(B, 'Sequence length', 4096);
}

// ---------------------------------------------------------------- who holds Q element dd
{
  const B = 'Who holds Q element dd?';
  check('lanes: element 77 is in lane 13, slot 2', (await text(bench(B) + ' [data-owner]')).includes('lane 13, register slot 2'));
  check('lanes: lane 13 is the highlighted owner', (await page.$eval(bench(B) + ' [data-owner="yes"]', (e) => e.dataset.lane)) === '13');
  await setInput(B, 'Element dd', 5);
  check('lanes: element 5 is in lane 5, slot 0', (await text(bench(B) + ' [data-owner]')).includes('lane 5, register slot 0'));
  await clickIn(B, 'AMD wave64, book’s code, default width');
  const bad = await page.$$eval(bench(B) + ' [data-ok="no"]', (e) => e.map((x) => Number(x.dataset.hw)));
  check('lanes: default width, the 32 lanes of the second logical warp read the wrong lane', bad.length === 32 && bad.join() === fp.wrong5.join());
  check('lanes: hardware lane 40 receives lane 5', fp.src40w64 === 5 && (await text(bench(B) + ' [data-picked]')).includes('receives hardware lane 5, which belongs to the other'));
  if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-lanes-wave64.png' });
  await clickIn(B, 'AMD wave64, width 32');
  check('lanes: width 32, lane 40 receives 37 and nothing is wrong', fp.src40w32 === 37 && (await text(bench(B) + ' [data-picked]')).includes('receives hardware lane 37, the right one') && (await page.$$(bench(B) + ' [data-ok="no"]')).length === 0);
}

// ---------------------------------------------------------------- banks
{
  const B = 'Writing K^T with and without padding';
  const passes = () => page.$eval(bench(B) + ' [data-banks-summary]', (e) => e.dataset.passes);
  check('banks: B_c 32 unpadded is 32 passes', fp.pass32 === 32 && (await passes()) === '32');
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('padding')).querySelector('input').click(), bench(B));
  await pause(40);
  check('banks: padded, 1 pass', fp.pass32p === 1 && (await passes()) === '1');
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [aria-label="B_c"] button')].find((b) => b.textContent === '64').click(), bench(B));
  await pause(40);
  check('banks: B_c 64 padded still has 2 lanes per bank', fp.pass64p === 2 && (await passes()) === '2');
  await select(B, 'Banks', 'apple');
  check('banks: Apple documents no banks, nothing drawn', (await passes()) === 'none');
}

// ---------------------------------------------------------------- scene: flash block
{
  const S = 'section.scene[aria-label="One block\'s journey through the flash kernel"]';
  check('flash-block: six parts', (await page.$$(S + ' .seg button')).length === 6);
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .seg button')][4].click(), S);
  await pause(120);
  check('flash-block part 5: the empty tile is named', (await text(S + ' svg')).includes('tile 2 lies above the diagonal'));
  await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .seg button')][5].click(), S);
  await page.$eval(S + ' input[type=range]', (el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '1000');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await pause(120);
  const filled = await page.$$eval(S + ' svg [data-reg-row]', (g) => g.length);
  check('flash-block part 6: four rows of registers drawn', filled === 4);
  if (out) await page.$eval(S, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-scene-block.png' });
}

// ---------------------------------------------------------------- kernel stepper
{
  const B = 'Step one block';
  const narr = () => text(bench(B) + ' [data-narrate]');
  const tr = fk.runBlock(fk.TOY, fk.makeProblem(12, 8), 1, 'lockstep', 'together');
  check('kernel: step count is the model’s', (await text(bench(B) + ' [data-progress]')) === `step 0 of ${tr.steps.length}`);
  await clickIn(B, 'Step');
  check('kernel: step 1 loads Q rows 4 and 5', (await narr()).startsWith('Warp 0 loads its rows of Q (rows 4 and 5)'));
  check('kernel: the outline highlights load_Q', (await text(bench(B) + ' [data-outline] li.cur')).startsWith('load_Q'));
  await clickIn(B, 'Run to next barrier');
  check('kernel: the first barrier to open is the block barrier after tile 0’s loads', (await narr()).startsWith('The block barrier opens: tile 0'));
  await clickIn(B, 'Run to next barrier');
  check('kernel: the next is warp 0’s __syncwarp', (await narr()).startsWith("Warp 0's __syncwarp() opens"));
  await clickUntil(B, 'Next tile', 10);
  check('kernel: the block’s rows match exact attention', (await page.$eval(bench(B) + ' [data-block-result]', (e) => e.dataset.blockResult)) === 'ok');
  check('kernel: every row stored correctly', (await page.$$eval(bench(B) + ' [data-stored="ok"]', (e) => e.length)) === 4);
  if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-kernel-ok.png' });

  const defect = async (id, on) => {
    await page.evaluate((sel, id, on) => {
      const box = document.querySelector(sel + ` [data-defect="${id}"] input`);
      if (box.checked !== on) box.click();
    }, bench(B), id, on);
    await pause(80);
  };
  const runToEnd = () => clickUntil(B, 'Run to next barrier', 400);
  const result = () => page.$eval(bench(B) + ' [data-block-result]', (e) => e.dataset.blockResult);

  await defect('noSyncwarp', true);
  check('kernel: no __syncwarp, outline strikes it out', (await text(bench(B) + ' [data-outline]')).includes('(removed)'));
  check('kernel: no __syncwarp in lockstep offers no failure', !(await text(bench(B))).includes('Go to the first failure'));
  await runToEnd();
  check('kernel: no __syncwarp in lockstep is still correct', (await result()) === 'ok');
  await select(B, 'Lanes', 'apart');
  check('kernel: lanes apart, the missing barrier is found', await clickIn(B, 'Go to the first failure'));
  const n1 = await narr();
  check('kernel: the failure is a read of P before its lane wrote it', n1.includes('Failure:') && n1.includes('before lane'), n1);
  if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-kernel-fail.png' });
  await runToEnd();
  check('kernel: lanes apart without __syncwarp is wrong', (await result()) === 'wrong');
  await defect('noSyncwarp', false);
  await runToEnd();
  check('kernel: lanes apart with __syncwarp is correct', (await result()) === 'ok');

  await select(B, 'Lanes', 'lockstep');
  await defect('newMax', true);
  await runToEnd();
  const ntext = await text(bench(B) + ' [data-block-result]');
  check('kernel: returning the new max is wrong in lockstep, with no bad read', (await result()) === 'wrong' && ntext.includes('No bad read'), ntext);
  await defect('newMax', false);

  await defect('noBarrierAfter', true);
  await select(B, 'Warps', 'ahead');
  check('kernel: no barrier after the warp loop, warp 0 ahead: a failure to jump to', await clickIn(B, 'Go to the first failure'));
  check('kernel: … it is a read of a tile the next load overwrote', (await narr()).includes('already loaded the next tile over it'));
  await defect('noBarrierAfter', false);
  await select(B, 'Warps', 'together');

  await defect('negInfRtl', true);
  await select(B, 'Block', 0);
  await runToEnd();
  check('kernel: minus infinity, right to left: NaN in block 0', (await page.$$(bench(B) + ' [data-live="nan"]')).length > 0 && (await result()) === 'wrong');
  await defect('negInfRtl', false);
}

// ---------------------------------------------------------------- outcomes table
{
  const B = 'Every schedule at once';
  const prob = fk.makeProblem(12, 8);
  const rows = await page.$$eval(bench(B) + ' tbody tr', (rs) => rs.map((r) => [r.dataset.kernel, [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)]));
  const sched = [['lockstep', 'together'], ['lockstep', 'ahead'], ['apart', 'together'], ['apart', 'ahead']];
  let same = rows.length === 5;
  for (const [id, cells] of rows) {
    const cfg = fk.withDefects(fk.TOY, id === 'correct' ? [] : [id]);
    sched.forEach(([l, w], i) => {
      if ((fk.runKernel(cfg, prob, l, w).correct ? 'correct' : 'wrong') !== cells[i]) same = false;
    });
  }
  check('outcomes: every cell is the model’s verdict', same, JSON.stringify(rows));
  check('outcomes: the corrected kernel is correct everywhere', rows[0][1].every((c) => c === 'correct'));
  check('outcomes: the new-max slip is wrong everywhere', rows.find((r) => r[0] === 'newMax')[1].every((c) => c === 'wrong'));
}

// ---------------------------------------------------------------- listing and corrections
if (missingLab.length) {
  console.log(`ok   listing check skipped: ${missingLab.map((f) => 'labs/' + f).join(', ')} not written yet (the page loads them by their planned path)`);
} else {
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check('listing shows CUDA, Metal kernel and Metal host tabs', ['CUDA', 'Metal kernel', 'Metal host'].every((t) => tabs.includes(t)), tabs.join('|'));
  const cuText = fs.readFileSync(path.join(repo, 'labs', LAB_FILES[0]), 'utf8');
  const shown = await page.$eval('figure.code pre', (e) => e.textContent);
  check('the CUDA tab is lab 20.2’s file', shown.trim() === cuText.replace(/\n$/, '').trim());
}
check('the corrections table has nine rows', (await page.$$('[data-corrections] tbody tr')).length === 9);
check('the llama.cpp quote is shown with its status', (await page.$$eval('figure.code .status', (s) => s.map((x) => x.textContent))).some((t) => t.includes('quoted from llama.cpp')));

// ---------------------------------------------------------------- questions
{
  const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which lane holds element dd = 77')));
  await (await q.$('input[type=number]')).type('13');
  await (await q.$('button.primary')).click();
  await pause(100);
  check('numeric question: lane 13 is accepted', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
  const c = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('needs one warp barrier it does not have')));
  await c.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
  await (await c.$('button.primary')).click();
  await pause(100);
  check('choice question: the barrier between P and compute_O is accepted', (await c.evaluate((e) => e.textContent)).includes('Correct.'));
  const f = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('by what factor is the D you already have multiplied')));
  await (await f.$('input[type=number]')).type('0.135');
  await (await f.$('button.primary')).click();
  await pause(100);
  check('numeric question: 0.135 is accepted for e^(3 − 5)', (await f.evaluate((e) => e.textContent)).includes('Correct.'));
}

// ---------------------------------------------------------------- errors
// The only allowed errors are the 404s for lab files that do not exist yet.
const labMisses = notFound.filter((u) => u.includes('/api/lab-file') && missingLab.some((f) => u.includes(f))).length;
const real = [...errors];
for (let i = 0; i < labMisses; i++) {
  const k = real.findIndex((e) => e.includes('status of 404'));
  if (k >= 0) real.splice(k, 1);
}
check('no console or page errors (other than not-yet-written lab files)', real.length === 0, real.join(' | '));
await browser.close();

// ---------------------------------------------------------------- phone width, both themes
for (const dark of [false, true]) {
  const m = await open({ width: 390, height: 844, dark });
  await m.page.setRequestInterception(true);
  m.page.on('request', (r) => (r.method() === 'PUT' && r.url().includes('/api/progress') ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' }) : r.continue()));
  await goto(m.page, 'ch20/20.5');
  await m.page.evaluate((d) => { document.documentElement.dataset.theme = d ? 'dark' : 'light'; }, dark);
  await pause(300);
  const over = await m.page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check(`phone width (${dark ? 'dark' : 'light'}): the page does not scroll sideways`, over <= 0, String(over));
  if (out) {
    await m.page.$eval('section.bench[aria-label="Step one block"]', (e) => e.scrollIntoView());
    await m.page.screenshot({ path: out + `-phone-${dark ? 'dark' : 'light'}.png` });
  }
  await m.browser.close();
}

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
