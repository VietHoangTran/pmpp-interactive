// Browser check for the narrated animation "One iteration with three streams" in section 23.4:
// five parts (four on Fig. 23.14, one setting its totals beside the NCCL and NVSHMEM loops'; the
// chapter is condensed, so those loops get no part of their own), captions are an option (off by default, remembered), parts advance by
// themselves, it runs with the sound off, every part's drawing agrees with the model
// (src/sim/haloloop.ts at DEFAULTS, imported directly), and the numbers the scripts say are
// the model's. It passes with or without content/ch23/narration/manifest.json.
// Usage: node e2e/ch23-scene-loop.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { DEFAULTS, simulate } from '../src/sim/haloloop.ts';

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
const r2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const S = 'section.scene[aria-label="One iteration with three streams"]';
const N = 5;
const LABELS = ['Insertion', 'Execution', 'Events join', 'The host waits', 'Totals'];

// ---- the model
const streams = simulate({ ...DEFAULTS, version: 'streams' });
const nccl = simulate({ ...DEFAULTS, version: 'nccl' });
const nvshmem = simulate({ ...DEFAULTS, version: 'nvshmem' });
const s1 = streams.iters[0];
const firstWait = streams.segs.find((s) => s.lane === 'host' && s.iter === 1 && s.kind === 'blocked').start;
const queued = streams.insertions.filter((q) => q.iter === 1 && q.at <= firstWait + 1e-9);

const res = await fetch(BASE + '/content/ch23/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch23/narration/loop-${n}.txt`)).text()));

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
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const all = (sel, f) => [...svg.querySelectorAll(sel)].map(f);
  return {
    beat: Number(svg.dataset.beat),
    headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '',
    tokens: all('[data-token]', (e) => [e.dataset.token, e.dataset.lane, e.dataset.at]),
    segs: all('[data-seg]', (e) => ({ kind: e.dataset.seg, lane: e.dataset.lane, start: e.dataset.start, end: e.dataset.end, line: e.dataset.line })),
    arrows: all('[data-arrow]', (e) => [e.dataset.arrow, e.dataset.from, e.dataset.to]),
    gaps: all('[data-gap]', (e) => e.dataset.gap),
    totals: all('[data-total]', (e) => [e.dataset.total, e.dataset.iterTime, e.dataset.blocked]),
    lanes: all('text', (e) => e.textContent).filter((t) => /stream$|^host$|exchange$/.test(t)),
  };
}, S);

await goto(page, 'ch23/23.4');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const words = scripts.map((s) => s.split(' ').length);
check('five scripts exist, each 35 to 60 words', scripts.length === N && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}#*]/.test(s)), scripts.find((s) => /[<>=×%/_(){}#*]/.test(s)));
const about = (x) => Math.round(x);
check(`script 1: the host queues ${queued.length} operations in about a quarter of a unit (${r2(firstWait)})`,
  scripts[0].includes(`queues ${queued.length} operations`) && scripts[0].includes('a quarter of one time unit') && Math.abs(firstWait - 0.25) < 0.05);
check(`script 2: boundary kernels end at about ${about(s1.bottom.end)}, the internal one at about ${about(s1.internal.end)}`,
  scripts[1].includes(`finish at about ${about(s1.bottom.end)}`) && scripts[1].includes(`until about ${about(s1.internal.end)}`) && about(s1.top.end) === about(s1.bottom.end));
check(`script 3: the copy runs at about ${about(s1.copy.start)}`, scripts[2].includes(`copy runs at about ${about(s1.copy.start)}`));
check(`script 4: the top exchange ends at about ${about(s1.exTop.end)}; the bottom one starts ${about(s1.bottomGap)} units late and outlasts the internal kernel`,
  scripts[3].includes(`until about ${about(s1.exTop.end)}`) && scripts[3].includes(`finished at about ${about(s1.bottom.end)}`) && scripts[3].includes(`starts ${about(s1.bottomGap)} units late`) && s1.exBottom.end > s1.internal.end);
check(`script 5: about 9 and a half (${r2(streams.iterTime)}) and about 7 and a half (${r2(nccl.iterTime)}, ${r2(nvshmem.iterTime)}); the host blocked most of each`,
  scripts[4].includes('about 9 and a half') && scripts[4].includes('about 7 and a half') && Math.abs(streams.iterTime - 9.5) < 0.1
  && Math.abs(nccl.iterTime - 7.5) < 0.1 && Math.abs(nvshmem.iterTime - 7.5) < 0.1 && nccl.hidden && nvshmem.hidden && !streams.hidden
  && [streams, nccl, nvshmem].every((r) => r.hostBlocked / r.iterTime > 0.9));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('five parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['loop-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check("switching captions on shows part 1's script", (await hasCaption()) && (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(500);
check('the captions choice is remembered after a reload', (await (await box('captions')).evaluate((e) => e.checked)) && (await hasCaption()));

// ---- runs with the sound off, and parts advance by themselves
await setBox('sound', false);
let d = await drawn();
check('before playing, part 1 has queued nothing', d.beat === 1 && d.tokens.length === 0, JSON.stringify(d.tokens));
await press('Play');
await pause(2000);
const moved = await slider();
check('with sound off the animation runs and the audio stays silent', moved > 30 && (await audio()).paused, JSON.stringify({ moved, audio: await audio() }));
await press('Pause');
const held = await slider();
await pause(500);
check('pause holds the position', Math.abs((await slider()) - held) <= 2 && (await mainButton()) === 'Resume');
await seek(985);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself', (await current()) === 2 && (await slider()) < 500, `part ${await current()}, slider ${await slider()}`);
check('and its caption follows', (await caption()) === scripts[1], await caption());
await press('Pause');

// ---- every part against the model
await press('1. Insertion');
await seek(1000);
d = await drawn();
check(`part 1: all ${queued.length} operations queued, each at the moment the model says, on its stream`,
  d.tokens.length === queued.length && d.tokens.every(([k, lane, at], i) => k === queued[i].kind && lane === queued[i].lane && at === r2(queued[i].at))
  && d.headline.includes(`${queued.length} of ${queued.length}`) && d.headline.includes(r2(firstWait)),
  JSON.stringify(d.tokens));
check('part 1 draws no execution at all', d.segs.every((s) => s.kind === 'call'), JSON.stringify(d.segs.map((s) => s.kind)));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Execution');
await seek(1000);
d = await drawn();
const k = (lane, kind = 'kernel') => d.segs.find((s) => s.lane === lane && s.kind === kind);
check(`part 2: top and bottom kernels end at ${r2(s1.top.end)} and ${r2(s1.bottom.end)}, the internal kernel at ${r2(s1.internal.end)}; no host waits, no exchanges yet`,
  k('top')?.end === r2(s1.top.end) && k('bottom')?.end === r2(s1.bottom.end) && k('internal')?.end === r2(s1.internal.end)
  && !d.segs.some((s) => s.kind === 'blocked' || s.lane === 'linkTop' || s.lane === 'linkBottom'),
  JSON.stringify(d.segs.filter((s) => s.kind === 'kernel')));
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Events join');
await seek(1000);
d = await drawn();
const arrow = (ev) => d.arrows.find((a) => a[0] === ev);
check(`part 3: computeTop (${r2(s1.top.end)}) and computeBottom (${r2(s1.bottom.end)}) lead to the waits that pass at ${r2(s1.internal.end)}; the copy follows`,
  JSON.stringify(arrow('computeTop')) === JSON.stringify(['computeTop', r2(s1.top.end), r2(s1.internal.end)])
  && JSON.stringify(arrow('computeBottom')) === JSON.stringify(['computeBottom', r2(s1.bottom.end), r2(s1.internal.end)])
  && k('internal', 'copy')?.start === r2(s1.copy.start) && d.headline.includes(r2(s1.copy.start)),
  JSON.stringify(d.arrows));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. The host waits');
await seek(1000);
d = await drawn();
const blocked = d.segs.filter((s) => s.lane === 'host' && s.kind === 'blocked').map((s) => s.line);
check(`part 4: the host waits in lines 27, 28–30, 31, 32–34, 37, 38; the bottom rows wait ${r2(s1.bottomGap)}`,
  JSON.stringify(blocked) === JSON.stringify(['27', '28–30', '31', '32–34', '37', '38']) && JSON.stringify(d.gaps) === JSON.stringify([r2(s1.bottomGap)])
  && k('linkBottom', 'exchange')?.start === r2(s1.exBottom.start) && d.headline.includes(`${r2(s1.bottomGap)} lost`),
  JSON.stringify({ blocked, gaps: d.gaps, h: d.headline }));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Totals');
await seek(1000);
d = await drawn();
check('part 5: the three versions\' iteration and host-blocked times are the model\'s',
  JSON.stringify(d.totals) === JSON.stringify([streams, nccl, nvshmem].map((r) => [r.input.version, r2(r.iterTime), r2(r.hostBlocked)])),
  JSON.stringify(d.totals));
if (out) await page.screenshot({ path: out + '-part5.png' });

// ---- the end
await seek(975);
await press('Resume');
await pause(2400);
check('after the last part it stops and offers to play again', (await mainButton()) === 'Play again' && (await current()) === N, await mainButton());

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await press('4. The host waits');
await seek(1000);
if (out) await page.screenshot({ path: out + '-part4-light.png' });
await setBox('captions', false);
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth }), S);
check('at a narrow width the scene fits the page (its drawing scrolls inside it)', narrow.page && narrow.scene, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch23\/narration\/(manifest\.json|[a-z-]+-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
