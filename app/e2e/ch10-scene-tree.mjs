// Browser check for the narrated animation "A reduction tree folding" in section 10.2:
// six parts, captions are an option (off by default, remembered), the parts advance by
// themselves, it runs with the sound off, and every part's drawing agrees with the model
// (src/sim/reducetree.ts, imported here directly).
// It passes with or without content/ch10/narration/manifest.json: with no audio the
// player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch10-scene-tree.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { FIG103B, TIE_ROW, buildTree, builderTree, disagreeingSet, f32hex, f32short, pairCombine, pairs, treeStats, ulpDistance } from '../src/sim/reducetree.ts';

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const S = 'section.scene[aria-label="A reduction tree folding"]';
const N = 6;
const LABELS = ['One adder', 'Pairs', 'Same work', 'Scale it up', 'Rounding', 'Ties'];

// ---- the model's numbers
const LOOP = builderTree(FIG103B, 'sum', 'loop');
const TREE = builderTree(FIG103B, 'sum', 'neighbours');
const BIG = treeStats(1024);
const F = disagreeingSet(8, 'uniform', 1);
const root = (rule, shape) => {
  const p = buildTree(pairs(TIE_ROW), pairCombine(rule), shape).root.value;
  return `${f32short(p.v)} #${p.i}`;
};

const res = await fetch(BASE + '/content/ch10/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch10/narration/tree-fold-${n}.txt`)).text()));

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
// What the stage draws, read from the SVG.
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const visible = (el) => {
    for (let e = el; e && e !== svg; e = e.parentElement) if (Number(e.getAttribute('opacity') ?? 1) < 0.45) return false;
    return true;
  };
  const trees = [...svg.querySelectorAll('g[data-part="tree"]')].filter(visible).map((t) => {
    const levels = [];
    for (const n of t.querySelectorAll('g[data-part="node"]')) (levels[Number(n.dataset.level)] ??= []).push(n.dataset.value);
    return { name: t.dataset.name, upto: Number(t.dataset.upto), levels };
  });
  const attr = (part, a) => [...svg.querySelectorAll(`[data-part="${part}"]`)].filter(visible).map((e) => e.getAttribute(a));
  return {
    headline: svg.querySelector('[data-part="headline"]').textContent.trim(),
    text: [...svg.querySelectorAll('text')].filter(visible).map((t) => t.textContent.trim()),
    trees,
    profiles: [...svg.querySelectorAll('g[data-part="profile"]')].filter(visible).map((g) => ({ name: g.dataset.name, ops: g.dataset.ops })),
    bars: attr('bar', 'data-ops').map(Number),
    sums: [...svg.querySelectorAll('[data-part="sum"]')].filter(visible).map((e) => ({ name: e.dataset.name, hex: e.dataset.hex, text: e.textContent.trim() })),
    floats: attr('float', 'data-part').length ? [...svg.querySelectorAll('[data-part="float"]')].filter(visible).map((e) => e.textContent.trim()) : [],
  };
}, S);
const has = (list, s) => list.some((x) => x.includes(s));

await goto(page, 'ch10/10.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(400);

// ---- scripts
const counts = scripts.map((s) => s.split(' ').length);
check('six scripts exist, each 35 to 60 words', scripts.length === N && counts.every((w) => w >= 35 && w <= 60), counts.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}+−#]/.test(s)), scripts.find((s) => /[<>=×%/_(){}+−#]/.test(s)));
const loopTotals = LOOP.levels.map((l) => l[0].value);
check('part 1\'s script reads the loop\'s running totals from the model', scripts[0].includes(`${loopTotals[0]}, then ${loopTotals.slice(1, -1).join(', ')}, and finally ${loopTotals[7]}`) && scripts[0].includes(`${LOOP.work} additions, in ${LOOP.span} steps`), scripts[0]);
const level1 = TREE.levels[1].map((n) => n.value);
check('part 2\'s script reads the tree\'s first level from the model', scripts[1].includes(`${TREE.opsPerStep[0]} adders work at once and produce ${level1.slice(0, 3).join(', ')} and ${level1[3]}`), scripts[1]);
check('part 2\'s script: then 16 and 28, then 44', scripts[1].includes(`produce ${TREE.levels[2].map((n) => n.value).join(' and ')}`) && scripts[1].includes(`produces ${TREE.root.value}`));
check('part 3\'s script: the same work, 7 steps against 3', scripts[2].includes(`${LOOP.work} additions`) && scripts[2].includes(`${LOOP.span} steps against ${TREE.span}`));
check('part 4\'s script: 512 adders, step 10, about 102, from the model', scripts[3].includes(`${BIG.n} values`) && scripts[3].includes(`${BIG.peak} adders`) && scripts[3].includes(`step ${BIG.span}`) && scripts[3].includes(`about ${Math.round(BIG.average)}`));
check('part 5\'s script: 8 float values; part 6\'s: 2 entries tie', scripts[4].includes(`${F.values.length} float values`) && scripts[5].includes('2 entries tie'));
if (manifest) {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV', JSON.stringify([manifest.model, manifest.voice]));
} else {
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(S + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
}

// ---- parts and captions
const labels = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['tree-fold-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0], (await hasCaption()) ? await caption() : 'no caption');
await setBox('captions', true);
check('switching captions on shows part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
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
await press('1. One adder');
await seek(500);
let d = await drawn();
let k = d.trees[0]?.upto ?? -1;
check('part 1 midway: the loop has done k steps and shows the model\'s running total', k > 0 && k < LOOP.span && d.headline === `the loop, step ${k} of ${LOOP.span}: running total ${LOOP.levels[k][0].value}` && d.profiles[0].ops === Array(k).fill(1).join(' '), JSON.stringify([d.headline, d.profiles]));
await seek(1000);
d = await drawn();
check('part 1 ends: 7 steps, one adder each, total 44', d.trees[0].upto === LOOP.span && same(d.trees[0].levels.slice(1).map((l) => l[0]), loopTotals.slice(1)) && d.profiles[0].ops === LOOP.opsPerStep.join(' ') && has(d.text, `running total: ${LOOP.root.value}`), JSON.stringify(d.trees[0]));
if (out) await page.screenshot({ path: out + '-part1.png' });

await press('2. Pairs');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
d = await drawn();
check('part 2 ends: the tree\'s levels are the model\'s, 11 5 13 15, 16 28, 44', same(d.trees[0].levels, TREE.levels.map((l) => l.map((n) => n.value))) && same(d.trees[0].levels[1], ['11', '5', '13', '15']), JSON.stringify(d.trees[0].levels));
check('and the busy adders per step are 4, 2, 1', d.profiles[0].ops === '4 2 1' && d.headline === 'the tree, step 3: 1 adder busy', d.headline);
if (out) await page.screenshot({ path: out + '-part2.png' });

await press('3. Same work');
await seek(300);
d = await drawn();
check('part 3: both profiles have 7 operations, in 7 steps and in 3', d.profiles.length === 2 && d.profiles[0].ops === LOOP.opsPerStep.join(' ') && d.profiles[1].ops === TREE.opsPerStep.join(' ') && has(d.text, `the loop: ${LOOP.work} operations in ${LOOP.span} steps`) && has(d.text, `the tree: ${TREE.work} operations in ${TREE.span} steps`) && d.headline === 'same work, shorter span', JSON.stringify(d.profiles));
await seek(1000);
d = await drawn();
check('part 3 ends on the general labels: N − 1 and log₂ N', has(d.text, `work: N − 1 = ${LOOP.work} for both`) && has(d.text, `span: log₂ N = ${TREE.span} for the tree`), d.text.join('|'));
if (out) await page.screenshot({ path: out + '-part3.png' });

await press('4. Scale it up');
await seek(1000);
d = await drawn();
check('part 4 ends: 10 bars for 1024 values, 512 down to 1, as the model counts', same(d.bars, BIG.opsPerStep) && BIG.opsPerStep.length === 10, JSON.stringify(d.bars));
check('with the average 102.3 against the peak 512', has(d.text, `average ${BIG.average.toFixed(1)}, peak ${BIG.peak}`) && BIG.average.toFixed(1) === '102.3' && d.headline === 'hardware sized for the peak idles most of the time', d.text.join('|'));
if (out) await page.screenshot({ path: out + '-part4.png' });

await press('5. Rounding');
await seek(150);
d = await drawn();
check('part 5 starts with whole numbers: loop and tree both 44', has(d.text, `the loop: ${LOOP.root.value}`) && has(d.text, `the tree: ${TREE.root.value}`) && d.headline === 'whole numbers: the loop and the tree agree', d.text.join('|'));
await seek(1000);
d = await drawn();
check('part 5 ends with the model\'s 8 float32 values', same(d.floats, F.values.map(f32short)), JSON.stringify(d.floats));
check('and their two sums, in hex, from the model, which differ', d.sums.length === 2 && d.sums[0].hex === f32hex(F.loop) && d.sums[1].hex === f32hex(F.tree) && d.sums[0].hex !== d.sums[1].hex && d.sums[0].text.includes(f32short(F.loop)), JSON.stringify(d.sums));
check('the headline gives the model\'s distance in float32 steps', d.headline === `the same 8 floats, regrouped: the sums are ${ulpDistance(F.loop, F.tree)} float32 steps apart`, d.headline);
if (out) await page.screenshot({ path: out + '-part5.png' });

await press('6. Ties');
await seek(250);
d = await drawn();
const rootOf = (name) => d.trees.find((t) => t.name === name).levels.at(-1)[0];
check('part 6 first: keep-left crowns different winners in the two brackets', rootOf('neighbours') === root('left', 'neighbours') && rootOf('halves') === root('left', 'halves') && rootOf('neighbours') !== rootOf('halves'), JSON.stringify([rootOf('neighbours'), rootOf('halves')]));
if (out) await page.screenshot({ path: out + '-part6a.png' });
await seek(900);
d = await drawn();
check('part 6 then: lower index wins a tie, the same winner in both', rootOf('neighbours') === root('lower', 'neighbours') && rootOf('halves') === rootOf('neighbours') && d.headline === `lower index wins a tie: #${root('lower', 'halves').split('#')[1]} in every bracket`, d.headline);
if (out) await page.screenshot({ path: out + '-part6b.png' });

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
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|tree-fold-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
