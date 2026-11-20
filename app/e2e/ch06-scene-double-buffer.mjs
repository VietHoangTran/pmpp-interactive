// Browser check for the narrated animation "Two buffers, one barrier" in section 6.7:
// six parts, captions are an option, the parts advance by themselves, it runs with the
// sound off, and every frame of parts 1 to 5 is a state of the model
// (src/sim/doublebuf.ts, imported here directly): the drawing names its kernel and
// step, and the slots and threads it shows must be the model's at that step.
// It passes with or without content/ch06/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch06-scene-double-buffer.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { KIND_LABEL, firstFailure, run, stateAt, tileBytes } from '../src/sim/doublebuf.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const S = 'section.scene[aria-label="Two buffers, one barrier"]';
const N = 6;
const LABELS = ['One buffer', 'No read barrier', 'No write barrier', 'Same slot', 'Two buffers', 'Back to tiling'];
const cfg = (variant) => ({ threads: 4, iterations: 2, variant });

const res = await fetch(BASE + '/content/ch06/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch06/narration/double-buffer-${n}.txt`)).text()));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().startsWith(label)).click(), S, label);
  await pause(150);
};
const mainButton = () => page.$eval(S + ' .row button.primary', (b) => b.textContent.trim());
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const current = () => page.$$eval(S + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused }));
const svgText = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
/** What the frame draws: its kernel and step, every slot and every thread. */
const drawn = () =>
  page.$eval(S + ' .stage svg', (svg) => ({
    variant: svg.dataset.variant ?? null,
    step: svg.dataset.step == null ? null : Number(svg.dataset.step),
    slots: [...svg.querySelectorAll('[data-slot]')].map((g) => ({ id: g.dataset.slot, value: g.dataset.value, version: Number(g.dataset.version) })),
    threads: [...svg.querySelectorAll('[data-thread]')].map((g) => ({ doing: g.dataset.doing, pointers: g.dataset.pointers })),
  }));
/** Compares a drawn frame with the model's state at the frame's own step. */
const matchesModel = (d) => {
  const tr = run(cfg(d.variant), 'in-order');
  const st = stateAt(tr, d.step);
  const slotsOk = d.slots.every((s) => {
    const [b, k] = s.id.split('-').map(Number);
    const m = st.buffers[b][k];
    return s.version === m.version && s.value === (Number.isNaN(m.value) ? '' : String(m.value));
  });
  const threadsOk = d.threads.every((t, i) => {
    const m = st.threads[i];
    const doing = m.done ? 'finished' : m.waiting ? 'waits' : m.last == null ? 'not started' : null;
    return (doing == null || t.doing === doing) && (!t.pointers || t.pointers === `${m.inBuf}${m.outBuf}`);
  });
  return { ok: slotsOk && threadsOk && d.slots.length >= 4 && d.threads.length === 4, tr, st };
};

await goto(page, 'ch06/6.7');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}[\]+]/.test(s)), scripts.find((s) => /[<>=×%/_(){}[\]+]/.test(s)));
const war = run(cfg('single-no-read-barrier'), 'in-order').violations[0];
const raw = run(cfg('single-no-write-barrier'), 'in-order').violations[0];
check('script 2 tells the model\'s first failure: thread 3 reads slot 0 after thread 0 wrote it', war.thread === 3 && war.kind === 'write-after-read' && scripts[1].includes('Thread 3 comes later and reads slot 0') && scripts[1].includes('write after read'));
check('script 3 tells the model\'s first failure: thread 0 reads slot 1 too early', raw.thread === 0 && raw.kind === 'read-after-write' && scripts[2].includes('reads slot 1 before thread 1 has written it'));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['double-buffer-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check("switching captions on shows part 1's script", (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));
await setBox('captions', false);
check('switching captions off works', !(await (await box('captions')).evaluate((e) => e.checked)) && (audible ? !(await hasCaption()) : await hasCaption()));
await setBox('captions', true);

// ---- runs with the sound off, and parts advance by themselves
await setBox('sound', false);
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(985);
await press('Resume');
await pause(2400);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every frame against the model
await press('1. One buffer');
let d;
for (const v of [0, 300, 600]) {
  await seek(v);
  d = await drawn();
  const m = matchesModel(d);
  check(`part 1 at ${v / 10}%: one buffer, both barriers, drawn exactly as the model's state at step ${d.step}`, d.variant === 'single-both' && m.ok, JSON.stringify(d));
}
await seek(1000);
d = await drawn();
let m = matchesModel(d);
check('part 1 ends with every thread finished and the model\'s final values', m.ok && d.step === m.tr.steps.length && d.threads.every((t) => t.doing === 'finished')
  && d.slots.filter((s) => s.id.startsWith('0-')).map((s) => Number(s.value)).join(',') === m.tr.reference.join(','), JSON.stringify(d.slots));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. No read barrier');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(400);
d = await drawn();
check('part 2 midway is a state of the kernel without the barrier after the read', d.variant === 'single-no-read-barrier' && matchesModel(d).ok && d.step < firstFailure(m.tr = run(cfg('single-no-read-barrier'), 'in-order')), JSON.stringify(d));
await seek(1000);
d = await drawn();
let t = await svgText();
check("part 2 ends on the model's first failure, a write-after-read, and says so", d.step === firstFailure(run(cfg('single-no-read-barrier'), 'in-order')) && matchesModel(d).ok
  && t.includes(`${KIND_LABEL['write-after-read']}: thread 3 still needed the initial value`), t.slice(-240));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. No write barrier');
await seek(1000);
d = await drawn();
t = await svgText();
check("part 3 ends on the model's first read-after-write", d.variant === 'single-no-write-barrier' && d.step === firstFailure(run(cfg('single-no-write-barrier'), 'in-order')) && matchesModel(d).ok
  && t.includes(`${KIND_LABEL['read-after-write']}: thread 0 read slot 1 in iteration 1`), t.slice(-240));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Same slot');
await seek(100);
check('part 4 starts from part 2\'s failure, still one buffer', (await drawn()).step === firstFailure(run(cfg('single-no-read-barrier'), 'in-order')) && !(await svgText()).includes('buffer B'));
await seek(1000);
d = await drawn();
check('part 4 ends with a second, empty buffer beside the first', (await svgText()).includes('a second buffer for the writes') && d.slots.filter((s) => s.id.startsWith('1-')).every((s) => s.version === -1 && s.value === ''), JSON.stringify(d.slots));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Two buffers');
for (const v of [200, 500, 800]) {
  await seek(v);
  d = await drawn();
  m = matchesModel(d);
  check(`part 5 at ${v / 10}%: two buffers, drawn exactly as the model's state at step ${d.step}, pointers included`, d.variant === 'double' && m.ok && d.slots.length === 8, JSON.stringify(d));
}
await seek(1000);
d = await drawn();
m = matchesModel(d);
const finalA = d.slots.filter((s) => s.id.startsWith('0-')).map((s) => Number(s.value));
check('part 5 ends correct: after two swaps every thread reads A, which holds the model\'s final values', m.ok && m.tr.correct && finalA.join(',') === m.tr.reference.join(',') && d.threads.every((x) => x.pointers === '01'), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Back to tiling');
await seek(1000);
t = await svgText();
check('part 6 ends with the shared-memory cost from the model and the lab', t.includes(`${tileBytes(32, 1).toLocaleString('en-US')} bytes`) && t.includes(`${tileBytes(32, 2).toLocaleString('en-US')} bytes`) && t.includes('lab 6.5') && t.includes('barrier 2'), t);
if (out) await page.screenshot({ path: out + '-part6.png' });

// ---- the end
await seek(985);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());
await press('Play again');
await pause(300);
check('playing again starts from part 1', (await current()) === 1 && (await mainButton()) === 'Pause');
await press('Pause');
await setBox('captions', false);

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|double-buffer-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
