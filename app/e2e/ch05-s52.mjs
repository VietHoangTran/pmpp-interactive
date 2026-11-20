// node e2e/s52.mjs
// Drives every interactive of section 5.2 ("Where data can live") against the running app
// and asserts on what it shows. Progress writes are intercepted, so a run leaves no trace.
import { open, goto } from './browser.mjs';

const { browser, page, errors } = await open();
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts += 1;
    r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  } else r.continue();
});

let failed = 0;
let passed = 0;
function check(name, ok, detail = '') {
  if (ok) passed += 1;
  else failed += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok || !detail ? '' : '\n       ' + detail}`);
}
const settle = (ms = 80) => new Promise((r) => setTimeout(r, ms));
const text = (sel) => page.$eval(sel, (el) => el.textContent.replace(/\s+/g, ' ').trim());
const exists = async (sel) => (await page.$(sel)) !== null;
// Replace the content of a number input the way a reader would: select all, type over it.
async function clearAndType(sel, value) {
  await page.focus(sel);
  await page.$eval(sel, (el) => el.select());
  await page.keyboard.press('Backspace');
  await page.type(sel, String(value));
  const got = await page.$eval(sel, (el) => el.value);
  if (got !== String(value)) throw new Error(`could not type ${value} into ${sel}: field holds "${got}"`);
}

try {
  await goto(page, 'ch05/5.2');
  check('section heading is shown', (await text('.page h1')).includes('Where data can live'));

  // ------------------------------------------------------------ memory map
  const MAP = 'section.bench[aria-label="Memory map"]';
  check('memory map bench is present', await exists(MAP));
  const looks = () =>
    page.$$eval(`${MAP} svg g[data-id]`, (gs) => Object.fromEntries(gs.map((g) => [g.dataset.id, g.dataset.look])));
  const pick = async (id) => {
    await page.click(`${MAP} button[data-pick="${id}"]`);
    await settle();
  };
  const expectLooks = async (name, want) => {
    const got = await looks();
    const bad = Object.entries(want).filter(([id, look]) => got[id] !== look);
    check(name, bad.length === 0, bad.map(([id, look]) => `${id}: expected ${look}, got ${got[id]}`).join('; '));
  };
  const detail = () => text(`${MAP} [data-map-detail]`);

  check('map has 17 selectable boxes, all real buttons', (await page.$$(`${MAP} button[data-pick]`)).length === 17);
  check('every box starts unselected', Object.values(await looks()).every((l) => l === 'plain'));
  check('detail panel starts with the instruction', (await detail()).includes('Nothing selected'));

  await pick('t01');
  await expectLooks('thread 1 of block 0: own registers, own local, own block shared, global are read+write', {
    t01: 'selected', r01: 'rw', l01: 'rw', s0: 'rw', g: 'rw',
  });
  await expectLooks('thread 1 of block 0: constant memory is read-only', { c: 'r' });
  await expectLooks("thread 1 of block 0: other threads' registers and local memory, and block 1's shared memory, are out of reach", {
    r00: 'none', r10: 'none', r11: 'none', l00: 'none', l10: 'none', l11: 'none', s1: 'none',
  });
  let d = await detail();
  check('detail names the thread and what it cannot reach',
    d.includes('Thread 1 of block 0') && d.includes('Cannot reach') && d.includes('including thread 0 of its own block') && d.includes('shared memory of block 1'), d);
  check('the picked button reports aria-pressed', (await page.$eval(`${MAP} button[data-pick="t01"]`, (b) => b.getAttribute('aria-pressed'))) === 'true');
  check('map shows "no access" and "read only" labels',
    await page.$eval(`${MAP} svg`, (s) => s.textContent.includes('no access') && s.textContent.includes('read only')));

  await pick('s1');
  await expectLooks('shared memory of block 1: only the two threads of block 1 can use it', {
    s1: 'selected', t10: 'rw', t11: 'rw', t00: 'none', t01: 'none', h: 'none',
  });
  d = await detail();
  check('detail says block 0 and the host are shut out', d.includes('Shared memory of block 1') && d.includes('by nobody else') && d.includes('the host sees neither'), d);

  await pick('r00');
  await expectLooks("registers of thread 0, block 0: only their owner, not even the block's other thread", {
    t00: 'rw', t01: 'none', t10: 'none', t11: 'none', h: 'none',
  });

  await pick('l11');
  await expectLooks('local memory of thread 1, block 1: only its owner', { t11: 'rw', t10: 'none', t00: 'none', t01: 'none', h: 'none' });
  check('detail says local memory sits inside global memory', (await detail()).includes('inside global memory'));

  await pick('g');
  await expectLooks('global memory: every thread and the host read and write', { t00: 'rw', t01: 'rw', t10: 'rw', t11: 'rw', h: 'rw' });

  await pick('c');
  await expectLooks('constant memory: threads read only, host reads and writes', { t00: 'r', t01: 'r', t10: 'r', t11: 'r', h: 'rw' });
  check('detail says no thread can write constant memory', (await detail()).includes('No thread can write it'));

  await pick('h');
  await expectLooks('host: reaches global and constant memory only', {
    h: 'selected', g: 'rw', c: 'rw', s0: 'none', s1: 'none', r00: 'none', r11: 'none', l00: 'none', l11: 'none',
  });
  await pick('h');
  check('picking the same box again clears the selection', Object.values(await looks()).every((l) => l === 'plain'));

  await page.focus(`${MAP} button[data-pick="t00"]`);
  await page.keyboard.press('Enter');
  await settle();
  await expectLooks('keyboard: Enter on a focused box selects it', { t00: 'selected', r00: 'rw', r01: 'none' });
  await page.keyboard.press('Space');
  await settle();
  check('keyboard: Space clears it', Object.values(await looks()).every((l) => l === 'plain'));

  // ------------------------------------------------------------ where does it live?
  const W = '[data-where]';
  check('classification bench is present', await exists(W));
  const tinted = (cls) => page.$$eval(`${W} .code .${cls}`, (els) => [...new Set(els.map((e) => e.textContent))].sort());
  const anyTint = async () => (await page.$$(`${W} .code .v-reg, ${W} .code .v-shared, ${W} .code .v-global, ${W} .code .v-const`)).length;
  const score = () => text(`${W} [data-where-score]`);
  const solvedAttr = (id) => page.$eval(`${W} [data-row="${id}"]`, (r) => r.dataset.solved);
  const feedback = (id) => text(`${W} [data-feedback="${id}"]`);
  const answer = async (id, mem, scope, copies) => {
    await page.select(`${W} select[aria-label="Memory for ${id}"]`, mem);
    await page.select(`${W} select[aria-label="Who can access ${id}"]`, scope);
    const input = `${W} input[aria-label="Copies of ${id}"]`;
    await clearAndType(input, copies);
    await page.click(`${W} button[aria-label="Check ${id}"]`);
    await settle();
  };

  check('kernel code states the launch: 48 blocks of 256 threads', (await text(`${W} .code pre`)).includes('smooth<<<48, 256>>>'));
  check('no name is coloured before anything is answered', (await anyTint()) === 0);
  check('score starts at 0 of 6', (await score()) === '0 of 6 placed');
  check('Check is disabled until all three answers are given', await page.$eval(`${W} button[aria-label="Check i"]`, (b) => b.disabled));

  await answer('i', 'shared', 'block', 48);
  let f = await feedback('i');
  check('i wrong on all three: memory feedback names the missing __shared__', f.includes('Nothing is shared unless it says __shared__'), f);
  check('i wrong: access feedback', f.includes('Each thread has its own value of i'), f);
  check('i wrong: count feedback says 48 is the number of blocks', f.includes('48 is the number of blocks'), f);
  check('i stays unsolved and uncoloured', (await solvedAttr('i')) === 'false' && (await anyTint()) === 0);

  await answer('i', 'register', 'thread', 12288);
  check('i right: row is marked solved', (await solvedAttr('i')) === 'true');
  check('i right: explanation appears', (await feedback('i')).includes('one copy per thread, each in a register'));
  check('i right: identifier i takes the register colour in the code', JSON.stringify(await tinted('v-reg')) === '["i"]');
  const sameColour = await page.evaluate((W) => {
    const span = document.querySelector(`${W} .code .v-reg`);
    const probe = document.createElement('span');
    probe.style.color = 'var(--reg)';
    document.body.appendChild(probe);
    const want = getComputedStyle(probe).color;
    probe.remove();
    return getComputedStyle(span).color === want;
  }, W);
  check('that colour is the --reg tier variable', sameColour);
  check('solved row locks its inputs', await page.$eval(`${W} select[aria-label="Memory for i"]`, (s) => s.disabled));
  check('score is 1 of 6', (await score()) === '1 of 6 placed');

  await answer('window', 'register', 'thread', 49152);
  f = await feedback('window');
  check('window as register: feedback gives the array rule and the constant-index exception', f.includes('goes to local memory by default') && f.includes('constant indices'), f);
  check('window: right access is acknowledged', f.includes('access is right'), f);
  check('window 49152: feedback says that counts floats', f.includes('That counts floats'), f);
  await answer('window', 'global', 'thread', 12288);
  f = await feedback('window');
  check('window as global: feedback says that is where it physically is, but the type is local', f.includes('where it physically is') && f.includes('count is right'), f);
  await answer('window', 'local', 'thread', 12288);
  check('window right: coloured as global (local memory is in global memory)', (await tinted('v-global')).includes('window'));

  await answer('blockGain', 'shared', 'block', 12288);
  f = await feedback('blockGain');
  check('blockGain 12288: feedback says that would be one per thread', f.includes('That would be one per thread') && f.includes('memory is right'), f);
  await answer('blockGain', 'shared', 'block', 48);
  check('blockGain right: shared colour', (await tinted('v-shared')).includes('blockGain'));

  await answer('stage', 'local', 'thread', 256);
  f = await feedback('stage');
  check('stage wrong: three specific messages',
    f.includes('__shared__ puts this one in shared memory') && f.includes('could not if the array were private') && f.includes('256 is how many threads use each copy'), f);
  await answer('stage', 'shared', 'block', 48);
  check('stage right: shared colour', JSON.stringify(await tinted('v-shared')) === '["blockGain","stage"]');

  await answer('in', 'constant', 'block', 48);
  f = await feedback('in');
  check('in as constant: feedback separates C++ const from constant memory', f.includes('const float* is a C++ promise') && f.includes('needs __constant__'), f);
  check('in per block: feedback', f.includes('Every block receives the same address') && f.includes('All 48 read the same array'), f);
  await answer('in', 'global', 'grid', 1);
  check('in right: global colour', JSON.stringify(await tinted('v-global')) === '["in","window"]');

  await answer('taps', 'global', 'grid', 4);
  f = await feedback('taps');
  check('taps as global: feedback says stored in global but cached', f.includes('stored in global memory but cached'), f);
  check('taps 4: feedback says four is the number of elements', f.includes('Four is the number of elements'), f);
  check('all-done summary is not shown at 5 of 6', !(await exists(`${W} [data-where-done]`)) && (await score()) === '5 of 6 placed');
  await answer('taps', 'constant', 'grid', 1);
  check('taps right: constant colour', JSON.stringify(await tinted('v-const')) === '["taps"]');
  check('score is 6 of 6 and the summary appears', (await score()) === '6 of 6 placed' && (await text(`${W} [data-where-done]`)).includes('All six placed'));

  await page.evaluate((W) => [...document.querySelectorAll(`${W} button`)].find((b) => b.textContent.trim() === 'Start again').click(), W);
  await settle();
  check('Start again clears colours, score and feedback',
    (await anyTint()) === 0 && (await score()) === '0 of 6 placed' && !(await exists(`${W} [data-feedback]`)) && !(await exists(`${W} [data-where-done]`)));

  // ------------------------------------------------------------ questions
  const tagQ = (needle, tag) =>
    page.evaluate(
      (needle, tag) => {
        const q = [...document.querySelectorAll('.q')].find((el) => el.querySelector('.prompt').textContent.includes(needle));
        if (!q) return false;
        q.setAttribute('data-e2e', tag);
        return true;
      },
      needle,
      tag,
    );
  check('four questions are on the page', (await page.$$('.q')).length === 4);
  check('question on thread-scope copies found', await tagQ('float acc = 0.0f;', 'thread'));
  check('question on block-scope copies found', await tagQ('__shared__ float stage[512];', 'block'));
  check('question on shared vs registers found', await tagQ('registers to spare', 'why'));
  check('question on unified memory found', await tagQ('shared storage mode', 'unified'));

  const typeAnswer = async (tag, value) => {
    const input = `.q[data-e2e="${tag}"] input[type="number"]`;
    await clearAndType(input, value);
    await page.click(`.q[data-e2e="${tag}"] button.primary`);
    await settle();
    return text(`.q[data-e2e="${tag}"] .verdict`);
  };
  let v = await typeAnswer('thread', 1000);
  check('thread-scope copies, wrong answer 1000: told the answer is 512000', v.includes('Not quite') && v.includes('512000') && v.includes('exercise 6'), v);
  v = await typeAnswer('thread', 512000);
  check('thread-scope copies, 512000: correct', v.startsWith('Correct.') && v.includes('1000 × 512'), v);
  v = await typeAnswer('block', 1000);
  check('block-scope copies, 1000: correct, with source', v.startsWith('Correct.') && v.includes('exercise 7'), v);

  const choose = async (tag, startsWith) => {
    const ok = await page.evaluate(
      (tag, startsWith) => {
        const label = [...document.querySelectorAll(`.q[data-e2e="${tag}"] .opts label`)].find((l) => l.textContent.trim().startsWith(startsWith));
        if (!label) return false;
        label.querySelector('input').click();
        return true;
      },
      tag,
      startsWith,
    );
    if (!ok) return 'option not found: ' + startsWith;
    await page.click(`.q[data-e2e="${tag}"] button.primary`);
    await settle();
    return text(`.q[data-e2e="${tag}"]`);
  };
  v = await choose('why', 'Reading shared memory is faster');
  check('shared vs registers, wrong option: "Not quite" with the reason for that option', v.includes('Not quite.') && v.includes('shared memory needs a load instruction'), v);
  v = await choose('why', 'A register belongs to one thread');
  check('shared vs registers, right option: correct, citing exercise 4', v.includes('Correct.') && v.includes('exercise 4'), v);
  v = await choose('unified', 'They disappear because CPU and GPU address the same allocation');
  check('unified memory, right option: correct, sourced to the facts file', v.includes('Correct.') && v.includes('ch05-facts.md, C6'), v);

  await settle(700); // progress is flushed 400 ms after the last answer
  check('answers went to /api/progress and were intercepted', puts > 0, `PUT count ${puts}`);

  // ------------------------------------------------------------ comparison table
  const cmp = await text('[data-compare]');
  check('comparison table has CUDA, HIP and Metal columns', ['CUDA (the book)', 'HIP', 'Metal'].every((h) => cmp.includes(h)));
  check('comparison table names the Metal barrier and the threadgroup address space',
    cmp.includes('threadgroup_barrier(mem_flags::mem_threadgroup)') && cmp.includes('[[threadgroup(n)]]'));
  check('comparison table covers transfer and limits', cmp.includes('shared storage mode') && cmp.includes('hipMemcpyHostToDevice') && cmp.includes('maxThreadgroupMemoryLength'));
  const dev = await text('[data-your-devices]');
  check('device panel shows reported values or says there are none', dev.includes('Reported') || dev.includes('No device report yet'), dev);
  const notes = await page.$$eval('.prose > aside.note', (els) => els.map((e) => e.textContent));
  check('margin notes: sources, two Today notes, a Beyond note', notes.filter((n) => n.startsWith('Book')).length >= 10 && notes.filter((n) => n.startsWith('Today')).length === 2 && notes.some((n) => n.includes('The book covers CUDA only')));
} catch (e) {
  failed += 1;
  console.log('FAIL script error: ' + (e && e.stack ? e.stack : e));
}

check('no console errors, page errors or failed requests', errors.length === 0, errors.join('\n       '));
console.log(`\n${passed} passed, ${failed} failed`);
await browser.close();
process.exit(failed ? 1 : 0);
