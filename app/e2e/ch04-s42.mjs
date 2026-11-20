// Browser check for section 4.2: the prediction about waiting blocks, the stepper
// "Blocks onto SMs" (stepped event by event against the model), and the three
// questions. Every expected state is computed here by the model the page runs on
// (src/sim/blocksched.ts), imported directly.
// Usage: node e2e/ch04-s42.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  FIG42,
  capacity,
  completionOrder,
  equalDurations,
  schedule,
  stateAt,
  unevenDurations,
  waitingAtLaunch,
} from '../src/sim/blocksched.ts';

const out = process.argv[2];
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
const clean = (s) => s.replace(/\s+/g, ' ').trim();
const B = 'section.bench[aria-label="Blocks onto SMs"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (label, wait = 40) => {
  const done = await page.evaluate((B, label) => {
    const btn = [...document.querySelectorAll(B + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B, label);
  await pause(wait);
  return done;
};
const disabled = (label) => page.evaluate((B, label) => [...document.querySelectorAll(B + ' button')].find((b) => b.textContent.trim() === label).disabled, B, label);
const setSelect = async (aria, value) => {
  await page.evaluate((B, aria, value) => {
    const s = document.querySelector(`${B} select[aria-label="${aria}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, B, aria, value);
  await pause(120);
};
// A number input: React only sees a value set through the native setter.
const setInput = async (aria, value) => {
  await page.evaluate((B, aria, value) => {
    const el = document.querySelector(`${B} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, B, aria, value);
  await pause(120);
};
// Everything the bench shows about the device, read from the page.
const shown = () => page.evaluate((B) => {
  const root = document.querySelector(B);
  const chips = (sel) => [...root.querySelectorAll(sel + ' .chip')].map((c) => Number(c.textContent));
  const k = (name) => root.querySelector(`.counts [data-k="${name}"]`).textContent.trim();
  return {
    waiting: k('waiting'),
    resident: k('resident'),
    done: k('done'),
    pending: chips('.pending'),
    finished: chips('.finished'),
    slots: [...root.querySelectorAll('.panel.sm')].map((sm) => [...sm.querySelectorAll('[data-slot]')].map((d) => (d.dataset.block === '' ? null : Number(d.dataset.block)))),
    strips: [...root.querySelectorAll('g[data-strip]')].filter((g) => g.dataset.on === '1').map((g) => Number(g.dataset.strip)),
    stripWhere: Object.fromEntries([...root.querySelectorAll('g[data-strip]')].filter((g) => g.dataset.on === '1').map((g) => [g.dataset.strip, [...g.querySelectorAll('text')].pop().textContent])),
    event: root.querySelector('span[aria-live]').textContent.trim(),
    narrate: root.querySelector('.narrate').textContent.replace(/\s+/g, ' ').trim(),
  };
}, B);
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
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
const shot = async (name) => {
  if (!out) return;
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// Does what the page shows agree with the model after n events?
const agrees = (s, n, v) => {
  const st = stateAt(s, n);
  const started = s.events.slice(0, n).filter((e) => e.kind === 'assign').map((e) => e.block).sort((a, b) => a - b);
  return v.event === `event ${n} of ${s.events.length}`
    && v.waiting === `${st.pending.length} waiting`
    && v.resident === `${st.resident} of ${s.capacity} slots in use`
    && v.done === `${st.done.length} finished`
    && same(v.pending, st.pending) && same(v.finished, st.done) && same(v.slots, st.slots) && same(v.strips, started)
    && started.every((b) => v.stripWhere[b] === `SM ${s.placements[b].unit}`);
};

const BLOCKS = 20; // the grid of chapter 3's 62 x 76 picture
const cap = capacity(FIG42.units, FIG42.slotsPerUnit);
const wait0 = waitingAtLaunch(BLOCKS, cap);

await goto(page, 'ch04/4.2');
await page.waitForSelector(B);

// ---------------------------------------------------------------- the lesson
const prose = clean(await page.$eval('.prose', (e) => e.textContent));
check('the apparent erratum about "Section 4.6" is stated and points to 4.7', prose.includes('Apparent erratum') && prose.includes('"Section 4.6"') && (await page.$$eval('.note a', (a) => a.map((x) => x.getAttribute('href')))).includes('#/ch04/4.7'));
check('the stepper states what it simplifies', clean(await page.$eval('.note.caveat', (e) => e.textContent)).includes('Which free slot a block gets is our choice: no vendor documents the policy'));
check('NVIDIA\'s "in most cases" and the cluster facts are carried as dated notes', prose.includes('"in most cases"') && prose.includes('which AMD devices set it is unverified') && prose.includes('A7, B11, C3'));

// ---------------------------------------------------------------- prediction
let q = await question('How many blocks are waiting right after the launch');
await (await q.$('input[type=number]')).type(String(cap));
await (await q.$('button.primary')).click();
await pause(150);
let t = await qText(q);
check('prediction reveals 14 waiting: 20 blocks, 2 x 3 = 6 at once', t.includes(`${wait0}. The device runs ${FIG42.units} × ${FIG42.slotsPerUnit} = ${cap} blocks at once, so ${BLOCKS} − ${cap} = ${wait0} wait.`) && wait0 === 14, t);
check('and tells a reader who answered 6 which number that is', t.includes('6 is the number that start; the question was how many do not.'), t);

// ---------------------------------------------------------------- the stepper on the prediction's launch
let s = schedule({ blocks: BLOCKS, ...FIG42, durations: equalDurations(BLOCKS) });
let v = await shown();
check('it starts on the prediction\'s launch: 20 waiting, 2 SMs of 3 free slots, nothing started', agrees(s, 0, v) && v.slots.length === 2 && v.slots.every((u) => u.length === 3 && u.every((b) => b === null)) && v.waiting === '20 waiting', JSON.stringify(v));
check('the opening text gives the capacity from the model', v.narrate.includes(`The grid has ${BLOCKS} blocks`) && v.narrate.includes(`so ${cap} can be in progress at once`), v.narrate);
await clickText('Step');
v = await shown();
check('step 1 assigns block 0, whole, to SM 0', agrees(s, 1, v) && v.narrate === 'Time 0: block 0 is assigned to SM 0, slot 0. All of its threads go there together.', v.narrate);
await clickText('Fill the device');
v = await shown();
check('"Fill the device": 6 of 6 slots in use and 14 waiting, as predicted', agrees(s, cap, v) && v.waiting === `${wait0} waiting` && v.resident === '6 of 6 slots in use', JSON.stringify([v.waiting, v.resident]));
check('the first slot of every SM is filled before any second slot', same(v.slots, [[0, 2, 4], [1, 3, 5]]), JSON.stringify(v.slots));
check('with the device full, "Fill the device" is off', await disabled('Fill the device'));
await shot('full');
await clickText('Step');
v = await shown();
check('the next event is a completion that frees a slot', agrees(s, cap + 1, v) && v.narrate === 'Time 1: block 0 finishes on SM 0 and gives up its slot.' && v.resident === '5 of 6 slots in use', v.narrate);
await clickText('Step');
v = await shown();
check('and the next block on the list takes that slot at once', agrees(s, cap + 2, v) && v.narrate.startsWith('Time 1: block 6, next on the list, takes the slot on SM 0 that block 0 has just left.'), v.narrate);
await clickText('Back');
v = await shown();
check('"Back" undoes one event', agrees(s, cap + 1, v));
await clickText('Restart');
check('"Restart" returns to the launch', agrees(s, 0, await shown()));

// Every event, one by one: the page never shows more than the capacity, never a block twice.
let ok = true;
let worst = '';
let peak = 0;
for (let n = 1; n <= s.events.length; n++) {
  await clickText('Step', 15);
  v = await shown();
  const resident = v.slots.flat().filter((b) => b !== null);
  peak = Math.max(peak, resident.length);
  if (!agrees(s, n, v) || resident.length > cap || new Set(resident).size !== resident.length) {
    ok = false;
    worst = `event ${n}: ${JSON.stringify(v.slots)}`;
    break;
  }
}
check(`all ${s.events.length} events agree with the model, one by one`, ok && s.events.length === 2 * BLOCKS, worst);
check('the blocks in progress never exceed SMs x slots, and reach it', peak === cap, String(peak));
v = await shown();
check('equal blocks finish in the order they were launched', same(v.finished, completionOrder(s)) && same(v.finished, [...Array(BLOCKS).keys()]) && v.narrate.includes('They finished in the order they were launched.'), v.narrate);
check('at the end every strip is drawn, each on the one SM the model placed it on', v.strips.length === BLOCKS && s.placements.every((p) => v.stripWhere[p.block] === `SM ${p.unit}`));
check('at the end Step and Play are off', (await disabled('Step')) && (await disabled('Play')) && (await disabled('Run to the end')));

// ---------------------------------------------------------------- uneven block times
await clickText('uneven block times', 150);
s = schedule({ blocks: BLOCKS, ...FIG42, durations: unevenDurations(BLOCKS) });
check('changing the block times restarts the run', agrees(s, 0, await shown()));
await clickText('Run to the end', 150);
v = await shown();
const order = completionOrder(s);
check('uneven blocks finish in the model\'s order, which is not the launch order', agrees(s, s.events.length, v) && same(v.finished, order) && !same(order, [...Array(BLOCKS).keys()]) && v.narrate.includes('in a different order from the one they were launched in'), JSON.stringify(v.finished));
check('but the same set of blocks finishes', same([...v.finished].sort((a, b) => a - b), [...Array(BLOCKS).keys()]));
await clickText('Restart');
ok = true;
for (let n = 1; n <= s.events.length; n++) {
  await clickText('Step', 15);
  v = await shown();
  const resident = v.slots.flat().filter((b) => b !== null);
  if (!agrees(s, n, v) || resident.length > cap) { ok = false; worst = `event ${n}`; break; }
}
check('with uneven times every event still agrees with the model and the ceiling holds', ok, worst);
await clickText('Restart');
await clickText('Fill the device');
for (let i = 0; i < 7; i++) await clickText('Step', 15);
await shot('uneven');

// ---------------------------------------------------------------- other devices and grids
await clickText('equal block times', 150);
await setSelect('SMs', 4);
await setSelect('Blocks an SM holds at once', 1);
await setInput('Blocks in the grid', 8);
s = schedule({ blocks: 8, units: 4, slotsPerUnit: 1 });
v = await shown();
check('4 SMs of 1 slot and 8 blocks: the device is redrawn and the run restarts', agrees(s, 0, v) && v.slots.length === 4 && v.slots.every((u) => u.length === 1), JSON.stringify(v.slots));
await clickText('Fill the device');
v = await shown();
check('4 start and 4 wait, by the model', agrees(s, 4, v) && v.waiting === `${waitingAtLaunch(8, capacity(4, 1))} waiting` && v.waiting === '4 waiting', v.waiting);
await setInput('Blocks in the grid', 3);
s = schedule({ blocks: 3, units: 4, slotsPerUnit: 1 });
await clickText('Fill the device');
v = await shown();
check('a grid smaller than the device leaves a slot free and nothing waiting', agrees(s, 3, v) && v.waiting === '0 waiting' && v.resident === '3 of 4 slots in use' && v.slots.flat().filter((b) => b === null).length === 1, JSON.stringify(v.slots));
await setInput('Blocks in the grid', 99);
check('the block count is clamped to what the bench can draw', (await page.$eval(B + ' input[aria-label="Blocks in the grid"]', (e) => e.value)) === '24');

// Play advances by itself and can be paused.
await setSelect('SMs', 2);
await setSelect('Blocks an SM holds at once', 3);
await setInput('Blocks in the grid', BLOCKS);
s = schedule({ blocks: BLOCKS, ...FIG42 });
await clickText('Play');
await pause(1500);
await clickText('Pause', 100);
v = await shown();
const at = Number(v.event.match(/event (\d+) of/)[1]);
check('"Play" advances by itself and "Pause" stops it', at >= 2 && at < s.events.length && agrees(s, at, v), v.event);
await pause(600);
check('paused, it stays where it is', (await shown()).event === v.event);

// ---------------------------------------------------------------- questions
q = await question('How many blocks are still waiting on the runtime');
const w = waitingAtLaunch(1000, capacity(40, 6));
await typeAnswer(q, capacity(40, 6));
check('waiting question: the capacity is rejected and the model\'s answer shown', (await qText(q)).includes(`Not quite. The answer is ${w} blocks.`) && w === 760, await qText(q));
await typeAnswer(q, w);
check('waiting question: 760 is accepted, with 240 at once in the explanation', (await qText(q)).includes('Correct.') && (await qText(q)).includes('240 blocks can be in progress at once (40 × 6)'), await qText(q));
q = await question('never split one block across two SMs');
await pick(q, 'It could, but moving threads');
check('whole-block question: "a preference of the scheduler" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('It is a guarantee of the model'), await qText(q));
await pick(q, 'Its threads may meet at a barrier and share on-chip memory');
check('whole-block question: barrier and shared memory is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Source: ch. 4 §4.2, p. 69 and §4.3, pp. 72–73'), await qText(q));
q = await question('to turn the sums into a running total');
await pick(q, 'Correct once a __syncthreads() is placed before the read');
check('cross-block question: a barrier is marked wrong, it stops at the block edge', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('It cannot make one block wait for another'), await qText(q));
await pick(q, 'Wrong: nothing orders blocks');
check('cross-block question: "nothing orders blocks" is accepted and names chapter 11', (await qText(q)).includes('Correct.') && (await qText(q)).includes('scan algorithms of chapter 11'), await qText(q));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((B) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, bench: document.querySelector(B).getBoundingClientRect().width <= document.documentElement.clientWidth }), B);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
await shot('narrow');

await pause(600);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
