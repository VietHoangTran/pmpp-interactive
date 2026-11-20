// Browser check for the narrated animation in section 5.3: captions are the
// scripts, audio really plays, picture follows sound, and it runs muted too.
// Usage: node e2e/scene.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const S = 'section.scene';
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.trim());
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), S, label);
  await pause(120);
};
const audio = () => page.$eval(S + ' audio', (a) => ({ paused: a.paused, t: a.currentTime, src: a.src.split('/').pop(), dur: a.duration }));
const slider = () => page.$eval(S + ' input[type=range]', (e) => Number(e.value));

const manifest = await (await fetch(BASE + '/content/ch05/narration/manifest.json')).json();
const script = async (n) => strip(await (await fetch(`${BASE}/content/ch05/narration/why-tiling-${n}.txt`)).text());

await goto(page, 'ch05/5.3');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
check('all six clips are registered with a duration', [1, 2, 3, 4, 5, 6].every((n) => manifest.clips['why-tiling-' + n]?.duration > 5));
check('caption of part 1 is its script, without audio tags', (await caption()) === (await script(1)), await caption());
check('six parts are offered', (await page.$$(S + ' .seg button')).length === 6);

// Play: sound and picture advance together.
await press('Play');
await pause(2500);
let a = await audio();
check('audio for part 1 is playing', !a.paused && a.t > 1 && a.src === 'why-tiling-1.mp3', JSON.stringify(a));
check('audio length matches the manifest', Math.abs(a.dur - manifest.clips['why-tiling-1'].duration) < 0.3, `${a.dur} vs ${manifest.clips['why-tiling-1'].duration}`);
const pos = (await slider()) / 1000;
check('picture position follows the audio clock', Math.abs(pos - a.t / a.dur) < 0.06, `slider ${pos}, audio ${a.t / a.dur}`);
check('the drawing shows fetches being counted', (await page.$eval(S + ' svg', (e) => e.textContent)).includes('fetches from global memory so far'));
if (out) await page.screenshot({ path: out + '-part1.png' });

// Pause holds position.
await press('Pause');
const held = await slider();
await pause(600);
check('pause stops both', (await audio()).paused && Math.abs((await slider()) - held) <= 2);

// Jump to part 4.
await press('4. Phase 0');
check('jumping to part 4 shows its script', (await caption()) === (await script(4)), await caption());
await press('Resume');
await pause(9500);
a = await audio();
check('part 4 plays its own clip', !a.paused && a.src === 'why-tiling-4.mp3' && a.t > 7, JSON.stringify(a));
const svg = await page.$eval(S + ' svg', (e) => e.textContent);
check('halfway through phase 0 the shared tiles are drawn and a barrier is shown', svg.includes('Mds (shared)') && svg.includes('barrier'), svg.slice(0, 200));
if (out) await page.screenshot({ path: out + '-part4.png' });

// A part ends and the next begins by itself: move near the end of part 4 and wait.
await page.$eval(S + ' audio', (el) => { el.currentTime = el.duration - 1.2; });
await pause(2600);
a = await audio();
check('at the end of a part the next one starts by itself', a.src === 'why-tiling-5.mp3' && !a.paused, JSON.stringify(a));
check('and its caption follows', (await caption()) === (await script(5)));

// Muted: runs on the clock alone.
await press('Pause');
await page.$eval(S + ' label.check input', (e) => e.click());
await press('6. What changed');
await press('Resume');
await pause(2000);
a = await audio();
check('with sound off the audio stays silent but the animation still runs', a.paused && (await slider()) > 50, JSON.stringify({ a, slider: await slider() }));
if (out) await page.screenshot({ path: out + '-part6.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
