// Browser check for the narrated animation "Growing a buffer without copying it" in
// section C.9: five parts, scripts written for the ear and consistent with the model,
// captions are an option (off by default, remembered), it runs with the sound off,
// parts advance by themselves, and every part's drawing agrees with the model
// (src/sim/vmmgrow.ts, imported here directly).
// It passes with or without content/appC/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: APP_URL=http://localhost:5173 node e2e/appC-scene-grow-buffer.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { grow } from '../src/sim/vmmgrow.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => {
  if (r.status() === 404) missing.push(r.url());
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const S = 'section.scene[aria-label="Growing a buffer without copying it"]';
const N = 5;
const LABELS = ['The usual way', 'Copies add up', 'Reserve', 'Map and grant', 'Grow in place'];

// ---- the model's numbers: 32 tokens, 4 to a chunk, as the scene runs them
const D = grow('double', { tokens: 32, bytesPerToken: 1, chunk: 4 });
const V = grow('vmm', { tokens: 32, bytesPerToken: 1, chunk: 4, reserve: 32 });
const moves = D.steps.filter((s) => s.moved);
const first = moves[0];

const res = await fetch(BASE + '/content/appC/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appC/narration/grow-buffer-${n}.txt`)).text()));

const hasCaption = () => page.$(S + ' .caption').then((e) => !!e);
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
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
const mainButton = () => page.$eval(S + ' button.primary', (e) => e.textContent.trim());
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused, t: a.currentTime }));
const drawn = () =>
  page.evaluate((S) => {
    const svg = document.querySelector(S + ' .stage svg');
    const g = (part) => svg.querySelector(`[data-part="${part}"]`);
    const d = g('double');
    const v = g('vmm');
    return {
      headline: g('headline').textContent.trim(),
      double: d ? { copied: Number(d.dataset.copied), moves: Number(d.dataset.moves), allocated: Number(d.dataset.allocated), dim: Number(d.getAttribute('opacity')) < 1 } : null,
      blocks: [...svg.querySelectorAll('[data-part="block"]')].map((b) => ({ base: Number(b.dataset.base), size: Number(b.dataset.size), state: b.dataset.state, held: Number(b.dataset.held) })),
      arrow: !!g('arrow'),
      vmm: v ? { reserved: Number(v.dataset.reserved), mapped: Number(v.dataset.mapped), copied: Number(v.dataset.copied), moves: Number(v.dataset.moves), granted: Number(v.dataset.granted), opacity: Number(v.getAttribute('opacity') ?? 1) } : null,
      pool: !!g('pool'),
      read: g('read')?.dataset.status ?? null,
    };
  }, S);

await goto(page, 'appC/C.9');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}+−#]/.test(s)), scripts.find((s) => /[<>=×%/_(){}+−#]/.test(s)));
check("part 1's script: 4 tokens, a buffer twice as large, room for 12 holding 4, from the model", scripts[0].includes(`holds ${first.from.size} tokens`) && first.to.size === 2 * first.from.size && scripts[0].includes(`room for ${first.during} tokens, holding ${first.copied}`), scripts[0]);
check("part 2's script: at token 9 copy 8, at 17 copy 16, 28 by token 32, from the model", moves.slice(1).every((m) => scripts[1].includes(`At token ${m.token} you copy ${m.copied}`)) && scripts[1].includes(`By token 32 you have copied ${D.copiedTotal}`), scripts[1]);
check("part 3's script: 8 chunks of 4 tokens", scripts[2].includes(`here ${V.reserved / 4} chunks of 4 tokens`));
check("part 5's script: 32 tokens, 8 chunks mapped", scripts[4].includes(`holds 32 tokens, ${V.steps[31].mapped} chunks are mapped`));
if (manifest) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['grow-buffer-1'];
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
await pause(1800);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(985);
await press('Resume');
await pause(2200);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part's drawing against the model
await press('1. The usual way');
await seek(100);
let d = await drawn();
check('part 1 early: one buffer of 4 slots, partly full, nothing copied', d.blocks.length === 1 && d.blocks[0].size === D.steps[0].to.size && d.blocks[0].held >= 1 && d.blocks[0].held <= 4 && d.double.copied === 0, JSON.stringify(d.blocks));
await seek(500);
d = await drawn();
check('part 1 midway: full, a new buffer of 8 elsewhere, both exist: 12 slots for 4 tokens', d.double.allocated === first.during && d.blocks.filter((b) => b.state !== 'freed').length === 2 && d.blocks.some((b) => b.state === 'old' && b.held === 4) && d.blocks.some((b) => b.state === 'live' && b.base === first.to.base && b.size === first.to.size), JSON.stringify(d));
await seek(720);
d = await drawn();
check('part 1, the copy: an arrow, and 4 tokens copied', d.arrow && d.double.copied === first.copied, JSON.stringify(d.double));
if (out) await page.screenshot({ path: out + '-scene-part1.png' });
await seek(1000);
d = await drawn();
check('part 1 ends: old buffer freed, the pointer has moved once', d.blocks.some((b) => b.state === 'freed' && b.base === first.from.base) && d.double.moves === 1 && d.headline.includes('base pointer has moved'), JSON.stringify(d));

await press('2. Copies add up');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
d = await drawn();
const last = D.steps[31];
check(`part 2 ends at token 32: ${D.copiedTotal} copied, ${D.pointerChanges} moves, as the model counts`, d.double.copied === D.copiedTotal && d.double.moves === D.pointerChanges && D.copiedTotal === 28, JSON.stringify(d.double));
check('three freed buffers and the live one where the model put it', d.blocks.filter((b) => b.state === 'freed').map((b) => b.base).join(',') === moves.map((m) => m.from.base).join(',') && d.blocks.some((b) => b.state === 'live' && b.base === last.to.base && b.size === last.to.size && b.held === 32), JSON.stringify(d.blocks));
if (out) await page.screenshot({ path: out + '-scene-part2.png' });

await press('3. Reserve');
await seek(800);
d = await drawn();
check('part 3: a reserved range of 32 slots with nothing mapped, the doubling run dimmed', d.vmm && d.vmm.reserved === 32 && d.vmm.mapped === 0 && d.vmm.opacity > 0.9 && d.double.dim, JSON.stringify(d.vmm));
check('part 3 headline: 8 chunks of addresses', d.headline.includes(`reserve ${V.reserved / 4} chunks`));

await press('4. Map and grant');
await seek(150);
d = await drawn();
check('part 4 first: a chunk with memory but no address', d.pool && d.vmm.mapped === 0);
await seek(400);
d = await drawn();
check('part 4 then: mapped at the start, no access', d.vmm.mapped === 1 && d.vmm.granted === 0 && !d.read);
await seek(620);
d = await drawn();
check('part 4: the read crashes before access is granted (the stepper model)', d.read === 'crash' && d.vmm.granted === 0);
await seek(900);
d = await drawn();
check('part 4 ends: access granted and the read works', d.read === 'ok' && d.vmm.granted === 1);
if (out) await page.screenshot({ path: out + '-scene-part4.png' });

await press('5. Grow in place');
await seek(500);
d = await drawn();
check('part 5 midway: chunks mapped as the model maps them, nothing copied, no move', d.vmm.mapped >= 3 && d.vmm.mapped <= 6 && d.vmm.copied === 0 && d.vmm.moves === 0, JSON.stringify(d.vmm));
await seek(1000);
d = await drawn();
check('part 5 ends: 8 chunks mapped, all granted, 28 copied against 0', d.vmm.mapped === V.steps[31].mapped && d.vmm.granted === 8 && d.headline.includes(`copied ${D.copiedTotal} against ${V.copiedTotal}`) && d.headline.includes(`moved ${D.pointerChanges} times against 0`), d.headline);
if (out) await page.screenshot({ path: out + '-scene-part5.png' });

// ---- the end
await seek(985);
await press('Resume');
await pause(2000);
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
  await page.screenshot({ path: out + '-scene-narrow.png' });
}

const expected404 = (u) => /\/content\/appC\/narration\/(manifest\.json|grow-buffer-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
