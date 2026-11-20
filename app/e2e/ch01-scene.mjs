// Browser check for the narrated animation "One budget, two designs" on the
// chapter 1 start page: five parts, captions are an option, the parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with
// the model (src/sim/designs.ts, imported here directly).
// It passes with or without content/ch01/narration/manifest.json: with no audio
// the player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch01-scene.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { design, finish, race, regions, snapshot, sweep } from '../src/sim/designs.ts';

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
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const S = 'section.scene';
const N = 5;
const LABELS = ['One budget', 'One chain', 'Sixty-four chains', 'Where they cross', 'What it asks of you'];

const res = await fetch(BASE + '/content/ch01/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch01/narration/two-designs-${n}.txt`)).text()));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.trim());
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), S, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(150);
};
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const current = () => page.$$eval(S + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
const drawn = () => page.$$eval(S + ' .stage svg text', (ts) => ts.filter((t) => Number(getComputedStyle(t).opacity) > 0.9 && Number(t.getAttribute('opacity') ?? 1) > 0.9 && Number(t.closest('g[opacity]')?.getAttribute('opacity') ?? 1) > 0.9).map((t) => t.textContent.trim()));
const mainButton = () => page.$eval(S + ' button.primary', (e) => e.textContent.trim());
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused, t: a.currentTime }));
const has = (list, s) => list.some((x) => x.includes(s));

// The scene's two designs and its workloads, from the model.
const A = design(64, 2);
const B = design(64, 0);

await goto(page, 'ch01/start');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
check('five scripts exist, each 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(' '));
check('scripts use digits for numbers the model produces', scripts[1].includes('time 4') && scripts[1].includes('time 16') && scripts[1].includes('63 others') && scripts[2].includes('16 waves of 4'));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['two-designs-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script, without audio tags', (await hasCaption()) && (await caption()) === scripts[0] && !(await caption()).includes('['));
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
let t = await drawn();
check('before playing, part 1 shows the budget from the model', has(t, `a budget of ${A.budget} squares of power`), t.join('|'));
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(970);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part's drawing against the model
await press('1. One budget');
await seek(1000);
t = await drawn();
const steps = [0, 1, 2, 3].map((h) => design(64, h));
check('part 1 ends with the two designs the budget buys', has(t, `${A.units} units`) && has(t, `${B.units} units`) && has(t, `${A.unitPower} squares each`) && has(t, `speed ${A.speed}`), t.join('|'));
check('part 1 shows the rule applied 0 to 3 times', steps.every((d) => has(t, `speed ${d.speed}, power ${d.unitPower}`) && has(t, `the budget buys ${d.units}`)) && has(t, '2 times the speed of a unit costs 4 times the power'), t.join('|'));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. One chain');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
t = await drawn();
let r = race(A, B, 1, 16);
check('part 2: the fast unit is done at 4, the plain unit at 16', has(t, `done at ${r.a.last}`) && has(t, `done at ${r.b.last}`) && r.a.last === 4 && r.b.last === 16, t.join('|'));
check('part 2: 63 plain units are never used', has(t, `never used: ${r.b.idle} units`) && r.b.idle === 63, t.join('|'));
check('part 2: the fast design wins by the model\'s factor', has(t, `the fast unit finishes ${r.factor} times sooner`) && r.winner === 'a' && r.factor === 4, t.join('|'));
await seek(480);
t = await drawn();
const halfway = t.find((x) => x.startsWith('time '));
check('part 2 midway: the fast chain is finished, the plain one is not', has(t, `done at ${r.a.last}`) && !has(t, `done at ${r.b.last}`) && !!halfway, t.join('|'));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Sixty-four chains');
await seek(1000);
t = await drawn();
r = race(A, B, 64, 16);
check('part 3: first results at 4 and 16, last results at 64 and 16', has(t, `first result at ${r.a.first}`) && has(t, `last result at ${r.a.last}`) && has(t, `first result at ${r.b.first}`) && has(t, `last result at ${r.b.last}`) && r.a.last === 64 && r.b.last === 16, t.join('|'));
check('part 3: 16 waves against 1', has(t, `chains finished: 64 of 64, in ${r.a.waves} waves`) && has(t, 'chains finished: 64 of 64, in 1 wave') && r.a.waves === 16, t.join('|'));
check('part 3: the many units win by the model\'s factor', has(t, `64 chains: the many units finish ${r.factor} times sooner`) && r.winner === 'b' && r.factor === 4, t.join('|'));
await seek(500);
t = await drawn();
const now = Number(t.find((x) => x.startsWith('time ')).slice(5));
const left = t.find((x) => x.startsWith('chains finished') && x.includes('waves'));
const doneLeft = Number(left.match(/finished: (\d+) of/)[1]);
check('part 3 midway: the fast units have finished what the model says for that time', [snapshot(A, 64, 16, now).done, snapshot(A, 64, 16, now + 1).done].includes(doneLeft) && doneLeft > 0 && doneLeft < 64, `${left} at time ${now}`);
check('part 3 midway: the many units are all done', has(t, 'chains finished: 64 of 64, in 1 wave') && now >= finish(B, 64, 16).last, t.join('|'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Where they cross');
await seek(300);
t = await drawn();
const n = Number(t.find((x) => /^\d+ chains?$/.test(x)).split(' ')[0]);
const pt = sweep(A, B, 64, 16)[n - 1];
check('part 4 mid-sweep: both finishing times are the model\'s for that many chains', n > 16 && n < 64 && has(t, `fast units done at ${pt.a}`) && has(t, `many units done at ${pt.b}`) && has(t, 'many units lead'), t.join('|'));
await seek(1000);
t = await drawn();
const regs = regions(sweep(A, B, 64, 32));
const tie = regs.find((g) => g.winner === 'tie');
check('part 4: the tie from 13 to 16 chains is marked', has(t, `tie from ${tie.from} to ${tie.to} chains`) && tie.from === 13 && tie.to === 16, t.join('|'));
check('part 4: chains twice as long, the same crossing', has(t, 'chains of 32 operations') && has(t, 'fast units lead: 1 to 12') && has(t, 'many units lead: 17 to 64') && has(t, 'the same at every length')
  && JSON.stringify(regs) === JSON.stringify(regions(sweep(A, B, 64, 16))), t.join('|'));
check('part 4 draws both lines and the earlier, shorter pair', (await page.$$(S + ' .stage svg polyline')).length === 4);
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. What it asks of you');
await seek(1000);
t = await drawn();
const fb = finish(B, 64, 8);
const fa = finish(A, 64, 8);
check('part 5: 64 elements of an 8 by 8 product, one per unit, done at 8', has(t, 'P = M × N, 8 by 8: 64 elements') && has(t, '64 of 64 elements done') && has(t, `all done at ${fb.last}`) && fb.last === 8, t.join('|'));
check('part 5: the fast units would need until 32', has(t, `the ${A.units} fast units: at ${fa.last}`) && fa.last === 32, t.join('|'));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === 5, await mainButton());
await press('Play again');
await pause(300);
check('playing again starts from part 1', (await current()) === 1 && (await mainButton()) === 'Pause');
await press('Pause');

const expected404 = (u) => /\/content\/ch01\/narration\/(manifest\.json|two-designs-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
