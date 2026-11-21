// Browser check for section 2.2: the two-lane timeline stepper (every switch, both
// sliders, the backend relabelling), the prediction, the three questions and the
// three-tab listing of lab 2.1. Everything the stepper shows is compared with the
// model it runs on (src/sim/timeline.ts, imported here directly).
// Usage: node e2e/ch02-s22.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { BOOK_PICTURE, overlap, program, readOutcomes, report, run, stmtText, waitCovers } from '../src/sim/timeline.ts';
import { elements } from '../src/sim/depend.ts';

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
const T = '[data-timeline]';
const press = async (label) => {
  const ok = await page.evaluate(
    (sel, label) => {
      const b = [...document.querySelectorAll(sel + ' button')].find((x) => x.textContent.trim() === label);
      if (!b || b.disabled) return false;
      b.click();
      return true;
    },
    T,
    label,
  );
  await pause(60);
  return ok;
};
const click = async (sel) => {
  await page.$eval(T + ' ' + sel, (e) => e.click());
  await pause(80);
};
const setRange = async (label, v) => {
  await page.$eval(
    `${T} input[type=range][aria-label="${label}"]`,
    (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    v,
  );
  await pause(80);
};
const dom = () =>
  page.evaluate((sel) => {
    const root = document.querySelector(sel);
    const o = (k) => root.querySelector(`[data-out="${k}"]`);
    return {
      done: +root.dataset.done,
      n: +root.dataset.n,
      time: o('time').textContent,
      grids: ['A', 'B'].map((g) => o('grid-' + g).dataset.state),
      reads: ['A', 'B'].map((g) => o('read-' + g).dataset.outcome),
      said: o('said').textContent,
      bad: o('said').classList.contains('bad'),
      stmts: [...root.querySelectorAll('tr[data-stmt]')].map((r) => ({
        kind: r.dataset.kind,
        ran: r.dataset.ran === 'true',
        text: r.querySelector('code').textContent,
        host: r.children[2].textContent,
        returned: r.children[3].textContent,
        cur: r.classList.contains('cur'),
      })),
      host: [...root.querySelectorAll('g[data-host]')].map((g) => ({ stmt: +g.dataset.host, kind: g.dataset.kind, state: g.dataset.state, from: +g.dataset.from, to: +g.dataset.to })),
      device: [...root.querySelectorAll('g[data-grid]')].map((g) => ({ grid: g.dataset.grid, state: g.dataset.state, start: +g.dataset.start, end: +g.dataset.end })),
      now: +root.querySelector('g[data-now]').dataset.now,
      overlap: o('overlap')?.textContent ?? null,
      deviceEnd: o('device-end')?.textContent ?? null,
      step: o('step').textContent.trim(),
    };
  }, T);

/** Does everything the stepper shows agree with the model for these options after `done` statements? */
function disagreements(d, opts, done, backend = 'cuda') {
  const prog = program(opts);
  const tl = run(prog);
  const rep = report(tl, done);
  const reads = readOutcomes(tl, done);
  const bad = [];
  const eq = (name, got, want) => {
    if (JSON.stringify(got) !== JSON.stringify(want)) bad.push(`${name}: shown ${JSON.stringify(got)}, model ${JSON.stringify(want)}`);
  };
  eq('steps', [d.done, d.n], [done, prog.length]);
  eq('time', d.time, `t = ${rep.time}`);
  eq('now', d.now, rep.time);
  eq('grid states', d.grids, rep.grids.map((g) => g.state));
  eq('read outcomes', d.reads, ['A', 'B'].map((g) => reads.find((r) => r.grid === g)?.outcome ?? 'not launched'));
  eq('statement text', d.stmts.map((s) => s.text), prog.map((s, i) => stmtText(s, backend, waitCovers(prog, i))));
  eq('statements run', d.stmts.map((s) => s.ran), prog.map((_, i) => i < done));
  eq('host column', d.stmts.map((s) => s.host), prog.map((_, i) => (i < done ? tl.host[i].state : i === done ? 'next' : '')));
  eq('returned column', d.stmts.map((s) => s.returned), prog.map((_, i) => (i < done ? `t = ${tl.host[i].to}` : '')));
  eq('host lane', d.host, tl.host.slice(0, done).map((h) => ({ stmt: h.stmt, kind: h.kind, state: h.state, from: h.from, to: h.to })));
  eq('device lane', d.device, tl.grids.filter((g) => g.stmt < done).map((g) => ({ grid: g.grid, state: rep.grids.find((x) => x.grid === g.grid).state, start: g.start, end: g.end })));
  if (done === prog.length) {
    eq('overlap', d.overlap, `${overlap(tl)} tick${overlap(tl) === 1 ? '' : 's'}`);
    eq('device end', d.deviceEnd, `t = ${tl.grids[tl.grids.length - 1].end}`);
  } else eq('no summary before the end', [d.overlap, d.deviceEnd], [null, null]);
  return bad;
}
/** Steps a freshly restarted program to its end, checking every step against the model. */
async function walk(opts, backend = 'cuda') {
  const n = program(opts).length;
  const bad = [];
  for (let done = 0; done <= n; done++) {
    bad.push(...disagreements(await dom(), opts, done, backend).map((b) => `after ${done}: ${b}`));
    if (done < n) await press('Step');
  }
  return bad;
}

await goto(page, 'ch02/2.2');
await page.waitForSelector(T);
await page.waitForFunction(() => document.querySelector('.code pre')?.textContent.includes('vecAddKernel<<<'));
await page.$eval('section.bench', (e) => e.scrollIntoView({ block: 'start' }));

// ---- notes around the stepper
const today = await page.$$eval('.note.today', (n) => n.map((x) => x.textContent));
check('a Today note says the launch is asynchronous and names the three waits', today.some((t) => t.includes('asynchronous') && t.includes('cudaDeviceSynchronize()') && t.includes('hipDeviceSynchronize()') && t.includes('waitUntilCompleted') && t.includes('Checked 2026-11-20')), today.join(' || '));
check('a Today note marks the thread-cost figures as the book’s claim', today.some((t) => t.includes('Could not be confirmed') && t.includes("book's claim")));
check('the stepper states what it leaves out, before the bench', await page.evaluate(() => {
  const note = [...document.querySelectorAll('.note.caveat')].find((n) => n.textContent.includes('Streams'));
  const bench = document.querySelector('section.bench');
  return !!note && !!(note.compareDocumentPosition(bench) & Node.DOCUMENT_POSITION_FOLLOWING) && note.textContent.includes('nothing is measured');
}));
check('every margin note directly follows text, never a bench or the listing', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName))));
check('the Metal side of the Today note: commit returns at once, waitUntilCompleted blocks', today.some((t) => t.includes('commit') && t.includes('returns at once')), today.join(' || '));
const proseText = await page.$eval('.prose', (e) => e.textContent);
const caveats = await page.$$eval('.note.caveat', (n) => n.map((x) => x.textContent));
check('one command buffer finishing before the next is called the model’s assumption for Metal, in the text and in the note',
  proseText.includes("is the model's assumption for Metal") && caveats.some((t) => t.includes('Apple documents only the order of the queue')));
const beyond = await page.$$eval('.note.beyond', (n) => n.map((x) => x.textContent));
check('the ordering note cites the hazard-tracking fact (C9) and the queue fact (C12)', beyond.some((t) => t.includes('C9') && t.includes('C12')), beyond.join(' || '));
check('the course’s HIP build is marked unverified', beyond.some((t) => t.includes('HIP build has not been compiled on AMD hardware: unverified')) && proseText.includes('is set up to compile'), beyond.join(' || '));

// ---- the opening program: a blocking call after each launch
const book = { ...BOOK_PICTURE };
let d = await dom();
check('it opens on the seven-statement program with nothing run', d.done === 0 && d.n === 7 && d.host.length === 0 && d.device.length === 0 && d.said.startsWith('Nothing has run yet'), JSON.stringify([d.done, d.n, d.said]));
let bad = await walk(book);
check('blocking call after each launch: every step agrees with the model', bad.length === 0, bad.slice(0, 3).join(' ; '));
await press('Restart');
await press('Step');
await press('Step');
d = await dom();
check('when the launch statement returns, grid A is queued and has not run', d.grids[0] === 'queued' && d.device[0].state === 'queued' && d.said.includes('Grid A is queued') && d.said.includes('none of its threads has run yet'), d.said);
check('reading A at that moment would find it not finished', d.reads[0] === 'not finished' && d.reads[1] === 'not launched');
await press('Step');
d = await dom();
const tlBook = run(program(book));
check('the blocking call is where the host waits, for as long as the grid needs', d.host[2].state === 'blocked' && d.host[2].to - d.host[2].from === tlBook.host[2].to - tlBook.host[2].from && d.said.includes(`blocked in cudaDeviceSynchronize for ${tlBook.host[2].to - tlBook.host[2].from} ticks`) && d.stmts[2].cur, d.said);
check('after it, A is safe to read', d.grids[0] === 'finished' && d.reads[0] === 'guaranteed' && (await page.$eval(`${T} [data-out="read-A"] .chip`, (e) => e.classList.contains('ok'))));
if (out) await page.screenshot({ path: out + '-book-step3.png' });
check('Back undoes one statement', (await press('Back')) && (await dom()).done === 2);
await press('Run to the end');
d = await dom();
check('at the end the read is guaranteed for both grids and nothing overlapped', d.done === 7 && d.reads.join() === 'guaranteed,guaranteed' && !d.bad && d.overlap === '0 ticks' && d.said.includes('on every run'), JSON.stringify([d.reads, d.overlap, d.said]));
check('Step and Run to the end are disabled at the end', !(await press('Step')) && !(await press('Run to the end')));
if (out) await page.screenshot({ path: out + '-book-end.png' });

// ---- more host work after each launch
const ahead = { ...book, waitAfterLaunch: false };
await click('button[data-follow="work"]');
d = await dom();
check('changing what follows a launch restarts a six-statement program with no blocking call', d.done === 0 && d.n === 6 && !d.stmts.some((s) => s.kind === 'wait'), JSON.stringify(d.stmts.map((s) => s.kind)));
bad = await walk(ahead);
check('more host work after each launch: every step agrees with the model', bad.length === 0, bad.slice(0, 3).join(' ; '));
d = await dom();
const tlAhead = run(program(ahead));
check('the host is never blocked and its work overlaps the grids', d.host.every((h) => h.state === 'running') && overlap(tlAhead) > 0 && d.overlap === `${overlap(tlAhead)} ticks`, d.overlap);
check('the device is still working after the host’s last statement', d.grids[1] === 'running' && Number(d.deviceEnd.replace('t = ', '')) > d.now, JSON.stringify([d.grids, d.deviceEnd, d.now]));
check('the read is flagged: A finished by timing only, B not finished', d.bad && d.reads.join() === 'finished by timing only,not finished' && d.said.includes('Grid B was still running') && d.said.includes('Nothing in the program makes that so'), d.said);
check('both outcomes are shown in the failure colour', (await page.$$eval(`${T} [data-out^="read-"] .chip`, (c) => c.every((x) => x.classList.contains('fail')))));
if (out) await page.screenshot({ path: out + '-ahead-end.png' });

// ---- the sliders: the same program, other timings
await setRange('Ticks each grid runs for', 1);
await setRange('Ticks each piece of host work takes', 12);
const lucky = { ...ahead, gridTicks: 1, hostTicks: 12 };
d = await dom();
bad = disagreements(d, lucky, 6);
check('the sliders keep the place and the picture follows the model', bad.length === 0 && (await page.$eval(`${T} [data-out="grid-ticks"]`, (e) => e.textContent)) === '1' && (await page.$eval(`${T} [data-out="host-ticks"]`, (e) => e.textContent)) === '12', bad.join(' ; '));
check('with short grids and long host work both grids have finished, and it is still not guaranteed', d.grids.join() === 'finished,finished' && d.reads.join() === 'finished by timing only,finished by timing only' && d.bad, JSON.stringify([d.grids, d.reads]));
await setRange('Ticks each grid runs for', 12);
await setRange('Ticks each piece of host work takes', 1);
const unlucky = { ...ahead, gridTicks: 12, hostTicks: 1 };
d = await dom();
bad = disagreements(d, unlucky, 6);
check('with long grids and short host work the same program reads two unfinished grids', bad.length === 0 && d.reads.join() === 'not finished,not finished' && d.said.includes('Grid B was still queued'), bad.join(' ; ') + d.said);
await setRange('Ticks each grid runs for', book.gridTicks);
await setRange('Ticks each piece of host work takes', book.hostTicks);

// ---- launch B straight after A
const b2bAhead = { ...ahead, backToBack: true };
await click('input[data-b2b]');
bad = await walk(b2bAhead);
check('B straight after A, no blocking call: every step agrees with the model', bad.length === 0, bad.slice(0, 3).join(' ; '));
await press('Restart');
for (let i = 0; i < 3; i++) await press('Step');
d = await dom();
check('B is launched while A is running, and waits its turn', d.grids.join() === 'running,queued' && d.said.includes('it will start when grid A has finished') && d.said.includes('The host waited for neither'), d.said);
check('on the device B starts exactly when A ends', d.device[1].start === d.device[0].end, JSON.stringify(d.device));
const b2b = { ...book, backToBack: true };
await click('button[data-follow="wait"]');
bad = await walk(b2b);
check('B straight after A, then one blocking call: every step agrees with the model', bad.length === 0, bad.slice(0, 3).join(' ; '));
d = await dom();
check('one blocking call after two launches covers both (question q-inorder)', d.n === 5 && d.reads.join() === 'guaranteed,guaranteed' && !d.bad && d.host[3].state === 'blocked' && d.host[3].to === d.device[1].end, JSON.stringify([d.reads, d.host[3], d.device[1]]));
await press('Back');
check('that blocking call waited for both grids', (await dom()).said.includes('until grid A and grid B had finished'), (await dom()).said);
await press('Step');
if (out) await page.screenshot({ path: out + '-b2b-end.png' });

// ---- the backend switch relabels and changes nothing else
const lanes = (x) => JSON.stringify([x.host, x.device, x.grids, x.reads, x.time]);
const cuda = await dom();
await click('button[data-backend="hip"]');
const hip = await dom();
check('HIP: same launch syntax, hipDeviceSynchronize, same picture', hip.stmts[1].text === 'KernelA<<<nBlk, nTid>>>(args);' && hip.stmts[3].text === 'hipDeviceSynchronize();' && lanes(hip) === lanes(cuda) && hip.done === cuda.done, JSON.stringify(hip.stmts.map((s) => s.text)));
await click('button[data-backend="metal"]');
const metal = await dom();
check('Metal: commit and waitUntilCompleted, same picture', metal.stmts[1].text === '[cmdA commit];' && metal.stmts[3].text === '[cmdA waitUntilCompleted]; [cmdB waitUntilCompleted];' && lanes(metal) === lanes(cuda), JSON.stringify(metal.stmts.map((s) => s.text)));
check('the statement text for Metal is the model’s', disagreements(metal, b2b, 5, 'metal').length === 0, disagreements(metal, b2b, 5, 'metal').join(' ; '));
await press('Back');
check('the sentence names the backend’s own wait', (await dom()).said.includes('blocked in waitUntilCompleted'), (await dom()).said);
await press('Back');
check('and calls the Metal launch a commit', (await dom()).said.startsWith('The commit returned at once'), (await dom()).said);
if (out) await page.screenshot({ path: out + '-metal.png' });
await click('button[data-backend="cuda"]');
check('Restart goes back to the first statement', (await press('Restart')) && (await dom()).done === 0);

// ---- the prediction
const threads = elements(8, 2048, 4096);
const predict = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first')));
check('the prediction asks for the thread count of a residual add', (await predict.evaluate((e) => e.textContent)).includes('hidden size 4,096'));
const before = puts;
if (await predict.$('input[type=number]')) {
  await (await predict.$('input[type=number]')).type(String(threads));
  await (await predict.$('button.primary')).click();
  await pause(700); // saves are debounced
}
const revealed = await predict.evaluate((e) => e.textContent);
check('committing reveals the count from the model', revealed.includes(threads.toLocaleString('en-US')) && threads === 67108864, revealed.slice(0, 200));
check('the commit was sent to the progress file, and intercepted', puts > before, `${puts} PUTs`);

// ---- questions
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
let q = await question('What is guaranteed to have');
await pick(q, 'The grid has started running');
let qt = await q.evaluate((e) => e.textContent);
check('q-returned: "has started" is rejected, with the reason', qt.includes('Not quite.') && qt.includes('or even start'), qt.slice(-300));
await pick(q, 'The launch has been handed to the runtime');
check('q-returned: "handed to the runtime" is accepted', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('There is no explicit wait anywhere');
await pick(q, 'Yes to both');
qt = await q.evaluate((e) => e.textContent);
check('q-inorder: yes to both, and the explanation says the last step is an inference', qt.includes('Correct.') && qt.includes('The last step is an inference'), qt.slice(-300));
q = await question('inside a host loop of 10');
await pick(q, '2 grids');
check('q-grid: 2 grids is rejected', (await q.evaluate((e) => e.textContent)).includes('A grid belongs to a launch'));
await pick(q, '11 grids, 1 host thread. Yes');
check('q-grid: 11 grids, 1 host thread, different kernels allowed', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- the listing: lab 2.1 on three backends
const read = (f) => fs.readFileSync(path.join(repo, 'labs/ch02/lab1_vecadd/solution', f), 'utf8').trimEnd().split('\n');
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has the three tabs', tabs.join('|') === 'CUDA and HIP: one file|Metal: the kernel|Metal: the host', tabs.join('|'));
const shownLines = () => page.$$eval('.code pre .ln', (ls) => ls.map((l) => l.textContent.replace(/\n$/, '')));
const cuFile = read('kernel.cu');
let shown = await shownLines();
check('tab 1 is labs/ch02/lab1_vecadd/solution/kernel.cu, line for line', shown.length === cuFile.length && cuFile.every((l, i) => shown[i] === l), `${shown.length} lines shown, ${cuFile.length} in the file`);
const marked = await page.$$eval('.code pre .ln.hl', (ls) => ls.map((l) => l.textContent.replace(/\n$/, '')));
const at = cuFile.findIndex((l) => l.startsWith('__global__'));
const end = cuFile.findIndex((l, i) => i > at && l.startsWith('}'));
check('the marked lines are the kernel: device code inside a file of host code', marked.some((l) => l.startsWith('__global__')) && marked[marked.length - 1] === '}' && marked.includes(cuFile[at + 1]) && marked.length >= end - at + 1 && marked.length < cuFile.length / 2 && !marked.some((l) => l.includes('cudaMalloc')), marked.join(' / '));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal: the kernel').click());
await pause(100);
shown = await shownLines();
const metalFile = read('kernel.metal');
check('tab 2 is kernel.metal, line for line', metalFile.every((l, i) => shown[i] === l) && shown.some((l) => l.includes('kernel void')), shown.slice(0, 3).join(' / '));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal: the host').click());
await pause(100);
shown = await shownLines();
const mmFile = read('launch.mm');
check('tab 3 is launch.mm, line for line', mmFile.every((l, i) => shown[i] === l) && shown.some((l) => l.includes('waitUntilCompleted')), shown.slice(0, 3).join(' / '));
check('the Metal tabs mark no lines', (await page.$$('.code pre .ln.hl')).length === 0);

// ---- links and layout
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
check('the page links to the labs and to section 2.3', links.includes('#/ch02/labs') && links.includes('#/ch02/2.3'), links.join(' '));
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
check('at phone width the page itself does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), String(await page.evaluate(() => document.documentElement.scrollWidth)));
// The timeline scrolls inside its panel at this width; "host is here" must stay in view as the program runs.
await press('Restart');
for (let i = 0; i < 6; i++) await press('Step');
const nowSeen = await page.evaluate((sel) => {
  const line = document.querySelector(`${sel} g[data-now] line`);
  const panel = line.closest('.panel');
  const a = line.getBoundingClientRect();
  const b = panel.getBoundingClientRect();
  return { scrolls: panel.scrollWidth > panel.clientWidth + 1, ok: a.left >= b.left - 1 && a.right <= b.right + 1 };
}, T);
check('at phone width the timeline scrolls in its panel and keeps "host is here" in view', nowSeen.scrolls && nowSeen.ok, JSON.stringify(nowSeen));
if (out) {
  await page.$eval('section.bench', (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
