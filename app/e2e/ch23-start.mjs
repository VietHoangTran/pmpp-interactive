// Browser check for chapter 23's start page: the "Rows, halos and launches" explorer
// (every slider, a GPU pick, the before/after control, both toggles and the exercise
// preset), the prediction and the questions. Expected values come from the model,
// src/sim/halopart.ts, imported here directly.
// Usage: node e2e/ch23-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { EXERCISE1, bestTiles, coverage, launches, partition, tileSplit } from '../src/sim/halopart.ts';

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
const B = 'section.bench[aria-label="Rows, halos and launches"]';
const fmt = (v) => v.toLocaleString('en-US');
const text = (sel) => page.$eval(sel, (e) => e.textContent.trim()).catch(() => null);
const click = async (sel) => {
  const ok = await page.evaluate((sel) => {
    const e = document.querySelector(sel);
    if (!e || e.disabled) return false;
    e.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return true;
  }, sel);
  await pause(80);
  return ok;
};
const setRange = async (name, v) => {
  await page.evaluate(
    (sel, v) => {
      const e = document.querySelector(sel);
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, String(v));
      e.dispatchEvent(new Event('input', { bubbles: true }));
    },
    `${B} input[data-slider="${name}"]`,
    v,
  );
  await pause(80);
};
const toggle = async (name, on) => {
  await page.evaluate(
    (sel, on) => {
      const e = document.querySelector(sel);
      if (e.checked !== on) e.click();
    },
    `${B} input[data-toggle="${name}"]`,
    on,
  );
  await pause(100);
};
const readout = (k) => text(`${B} [data-readout="${k}"]`);
const rowRole = (r) => text(`${B} tr[data-row="${r}"] [data-role]`);
// The read-outs the explorer shows for one GPU must be the model's.
const readoutsMatch = async (p, name) => {
  const got = {
    owned: await readout('owned'),
    ny: await readout('ny'),
    oa: await readout('output-all'),
    ok: await readout('output-kernel'),
    halo: await readout('halo'),
    ba: await readout('boundary-all'),
    bk: await readout('boundary-kernel'),
    ia: await readout('internal-all'),
    ik: await readout('internal-kernel'),
    bytes: await readout('bytes'),
  };
  const want = {
    owned: fmt(p.owned),
    ny: fmt(p.ny),
    oa: fmt(p.output.all),
    ok: fmt(p.output.kernel),
    halo: fmt(p.halo),
    ba: fmt(p.boundary.all),
    bk: fmt(p.boundary.kernel),
    ia: fmt(p.internal.all),
    ik: fmt(p.internal.kernel),
    bytes: fmt(p.bytesSent),
  };
  check(name, JSON.stringify(got) === JSON.stringify(want), JSON.stringify({ got, want }));
};
// Screenshot of one element, for looking at the explorer in a given state.
const shoot = async (file, sel = B) => {
  if (!out) return;
  const el = await page.$(sel);
  await el.scrollIntoView();
  await pause(60);
  await el.screenshot({ path: `${out}-${file}.png` });
};
const question = (part) =>
  page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const answerNumeric = async (part, v) => {
  const q = await question(part);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (part, index) => {
  const q = await question(part);
  await q.evaluate((e, i) => e.querySelectorAll('.opts label input')[i].click(), index);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch23/start');
await page.waitForSelector(B);

// ---- default: 64 wide, 128 rows, 4 GPUs, GPU 1 (the split of q-halo-rows)
check('four bands, GPU 1 picked', (await page.$$(`${B} [data-domain] g[data-gpu]`)).length === 4 &&
  (await page.$eval(`${B} g[data-gpu="1"]`, (e) => e.getAttribute('aria-pressed'))) === 'true');
await readoutsMatch(partition(64, 128, 4, 1), 'default read-outs are the model\'s (GPU 1 of 4)');
check('GPU 1 owns global rows 32 to 63, ny = 34', (await text(`${B} [data-readouts] tbody tr:first-child td:first-child`)).includes('32 to 63') && (await readout('ny')) === '34');
check('row 0 is a copy of global row 31 from GPU 0', (await rowRole(0)).includes('global row 31, from GPU 0'), await rowRole(0));
check('row 33 is a copy of global row 64 from GPU 2', (await rowRole(33)).includes('global row 64, from GPU 2'), await rowRole(33));
check('row kinds: halo, boundary, internal', (await page.$$eval(`${B} tr[data-row]`, (rs) => rs.map((r) => r.dataset.kind).join(','))) ===
  'halo,boundary,internal,internal,boundary,halo');
check('one gap row for rows 3 to 30', ((await text(`${B} tr[data-gap]`)) || '').includes('Rows 3 to 30: 28 more internal rows'));
await shoot('default');

// ---- pick a GPU by clicking its band; the wrap-around
await click(`${B} g[data-gpu="3"]`);
check('clicking band 3 picks GPU 3', (await page.$eval(`${B} g[data-gpu="3"]`, (e) => e.getAttribute('aria-pressed'))) === 'true');
check('GPU 3: its bottom halo wraps to GPU 0', (await rowRole(33)).includes('from GPU 0 (by the wrap-around)'), await rowRole(33));
await click(`${B} g[data-gpu="0"]`);
check('GPU 0: its top halo wraps to GPU 3', (await rowRole(0)).includes('global row 127, from GPU 3 (by the wrap-around)'), await rowRole(0));

// ---- before / after the step
await click(`${B} button[data-phase="after"]`);
check('after the step the halo rows are stale', (await rowRole(0)).includes('stale') && (await rowRole(1)).includes('must reach GPU 3'), await rowRole(0));
await click(`${B} button[data-phase="before"]`);
check('before the step they are copies again', !(await rowRole(0)).includes('stale'));

// ---- sliders
await setRange('nx', 128);
check('nx slider shows 128', (await text(`${B} [data-value="nx"]`)) === '128');
await readoutsMatch(partition(128, 128, 4, 0), 'nx = 128: read-outs follow (halo 256, 1,024 bytes)');
await setRange('gpus', 16);
await setRange('rows', 520);
check('rows slider shows 520 and GPUs 16', (await text(`${B} [data-value="rows"]`)) === '520' && (await text(`${B} [data-value="gpus"]`)) === '16');
check('sixteen bands drawn', (await page.$$(`${B} [data-domain] g[data-gpu]`)).length === 16);
await click(`${B} g[data-gpu="15"]`);
await readoutsMatch(partition(128, 520, 16, 15), 'uneven split: the last GPU takes the remainder (40 rows, ny = 42)');
check('the last GPU owns 40 rows', (await readout('owned')) === '40' && (await readout('ny')) === '42');

// ---- the exercise preset
await click(`${B} button[data-preset="exercise1"]`);
check('preset sets 64, 512, 16', (await text(`${B} [data-value="nx"]`)) === '64' && (await text(`${B} [data-value="rows"]`)) === '512' && (await text(`${B} [data-value="gpus"]`)) === '16');
check('preset button shows pressed', (await page.$eval(`${B} button[data-preset="exercise1"]`, (e) => e.getAttribute('aria-pressed'))) === 'true');
const ex = partition(EXERCISE1.nx, EXERCISE1.rows, EXERCISE1.gpus, 15);
await readoutsMatch(ex, 'exercise read-outs are the model\'s');
check(
  'exercise 1a to 1e: 2048/1984, 128, 128/124, 1920/1860, 512',
  (await readout('output-all')) === '2,048' && (await readout('output-kernel')) === '1,984' && (await readout('halo')) === '128' &&
    (await readout('boundary-all')) === '128' && (await readout('boundary-kernel')) === '124' &&
    (await readout('internal-all')) === '1,920' && (await readout('internal-kernel')) === '1,860' && (await readout('bytes')) === '512',
);
check('exchange share 6.3% of the rows', ((await text(`${B} [data-share]`)) || '').includes('6.3%'));

// ---- the three launches
await toggle('launches', true);
const ls = launches(ex.ny, ex.nx);
for (const l of ls) {
  const row = `${B} [data-launches] tr[data-launch="${l.name}"]`;
  const got = await page.$eval(row, (r) => [r.querySelector('[data-offset]').dataset.offset, r.querySelector('[data-rows]').dataset.rows, r.querySelector('[data-writes]').dataset.writes, r.textContent]);
  check(`${l.name} launch: pointer ${l.pointer}, ${l.rows} rows, writes ${l.writes[0]}..${l.writes[l.writes.length - 1]}`,
    got[0] === String(l.offset) && got[1] === String(l.rows) && got[2] === l.writes.join(',') && got[3].includes(l.pointer), JSON.stringify(got));
}
check('three launches cover rows 1 to ny − 2 once', (await page.$eval(`${B} [data-coverage]`, (e) => e.dataset.coverage)) === 'exact' && coverage(ex.ny).exact);
check('row 1: written by thread row 0 of the top launch', (await text(`${B} tr[data-row="1"] td[data-launch="top"]`)) === 't0');
check('row 32: written by thread row 0 of the bottom launch', (await text(`${B} tr[data-row="32"] td[data-launch="bottom"]`)) === 't0');
check('row 2: written by thread row 0 of the internal launch', (await text(`${B} tr[data-row="2"] td[data-launch="internal"]`)) === 't0');
check('rows 3 to 30: internal thread rows 1 to 28', (await text(`${B} tr[data-gap] td[data-launch="internal"]`)) === 't1–t28');
check('the halo rows are read, never written', (await text(`${B} tr[data-row="0"] td[data-launch="top"]`)) === 'read' &&
  (await text(`${B} tr[data-row="33"] td[data-launch="bottom"]`)) === 'read');
await shoot('launches');

// ---- the GPU slider at its top, and one row per GPU
await setRange('gpus', 32);
const at32 = partition(64, 512, 32, 15);
await readoutsMatch(at32, '32 GPUs: same 512 bytes, 16 rows each');
check('exchange share doubles to 12.5%', ((await text(`${B} [data-share]`)) || '').includes('12.5%'));
await setRange('rows', 32);
check('one row per GPU: ny = 3', (await readout('ny')) === '3');
check('one row per GPU: row 1 written twice', (await page.$eval(`${B} [data-coverage]`, (e) => e.dataset.coverage)) === 'overlap' &&
  ((await text(`${B} [data-coverage]`)) || '').includes('row 1 twice') && !coverage(3).exact);
check('row 1 is both boundaries', (await rowRole(1)).startsWith('Top and bottom boundary'), await rowRole(1));
await shoot('onerow');

// ---- tiles
await click(`${B} button[data-preset="exercise1"]`);
await toggle('launches', false);
await toggle('tiles', true);
const best = bestTiles(64, 512, 16);
check('tiles: the tall exercise grid is best split by rows', (await page.$eval(`${B} [data-best]`, (e) => e.dataset.best)) === `${best.p}x${best.q}` && best.q === 1);
check('tiles: default arrangement is the row split itself', ((await text(`${B} [data-tileverdict]`)) || '').includes('row split itself'));
await click(`${B} button[data-pq="4x4"]`);
const t44 = tileSplit(64, 512, 4, 4);
check('4 × 4 on the exercise grid: 288 values, 4 neighbours, rows send less',
  (await text(`${B} [data-t="tiles-points"]`)) === fmt(t44.points) && (await text(`${B} [data-t="tiles-neighbours"]`)) === '4' &&
    (await text(`${B} [data-t="rows-points"]`)) === '128' && ((await text(`${B} [data-tileverdict]`)) || '').startsWith('Rows send less'));
await setRange('nx', 512);
const sq = tileSplit(512, 512, 4, 4);
check('a square grid: 4 × 4 sends 50% less than rows', (await text(`${B} [data-t="tiles-bytes"]`)) === fmt(sq.bytes) &&
  (await text(`${B} [data-t="rows-bytes"]`)) === fmt(tileSplit(512, 512, 16, 1).bytes) && ((await text(`${B} [data-tileverdict]`)) || '').includes('50% less'));
check('the square grid\'s best is 4 × 4', (await page.$eval(`${B} [data-best]`, (e) => e.dataset.best)) === '4x4');
await click(`${B} [data-tilespic] rect[data-tile="6"]`);
check('clicking a tile picks its GPU', (await page.$eval(`${B} g[data-gpu="6"]`, (e) => e.getAttribute('aria-pressed'))) === 'true');
await shoot('tiles', `${B} [data-tiles]`);
await setRange('gpus', 1);
check('one GPU: nothing to tile', (await page.$eval(`${B} [data-tiles]`, (e) => e.dataset.tiles)) === 'none');
check('one GPU: it is its own neighbour', ((await text(`${B} [data-share]`)) || '').includes('its own top and bottom neighbour') &&
  ((await readout('bytes')) === fmt(partition(512, 512, 1, 0).bytesSent)));

// ---- questions and the prediction
let t = await answerNumeric('GPU 1 owns global rows 32 to 63', 34);
check('q-halo-rows accepts 34', t.includes('Correct.'));
t = await answerNumeric('Book exercise 1e', 512);
check('q-ex1-bytes accepts 512', t.includes('Correct.'));
t = await answerChoice('What does t1 − t0 measure', 0);
check('q-queued: the queuing time is right', t.includes('Correct.'));
t = await answerChoice('Why must the blockReduce call sit outside', 2);
check('q-blockreduce: a wrong pick explains itself', t.includes('Not quite.') && t.includes('wrong results or a hang'));
t = await answerNumeric("Chapter 22’s example", 20);
check('q-amdahl accepts 20', t.includes('Correct.'));
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('now runs on 32 GPUs')));
await (await pq.$('input[type=number]')).type('256');
await (await pq.$('button.primary')).click();
await pause(150);
const reveal = await pq.evaluate((e) => e.textContent);
check('prediction reveal: 512 bytes, 32 to 16 rows, halving explained', reveal.includes('512 bytes') && reveal.includes('from 32 rows to 16') && reveal.includes('Halving'), reveal);

check('progress writes were intercepted', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
