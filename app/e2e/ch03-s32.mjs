// Browser check for section 3.2: the two predictions, the narrated scene "A grid
// over a picture", the thread map, the kernel stepper, the flattening explorer,
// the erratum hunt and the pixel tester. Expected numbers are the ones pinned in
// app/src/sim/mapping.test.ts.
// Usage: node e2e/ch03-s32.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const has = (sel) => page.$(sel).then((e) => !!e);
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setSelect = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};
// Range and number inputs: React only sees a value set through the native setter.
const setInput = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const el = document.querySelector(`${scope} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const setCheck = async (scope, label, on) => {
  await page.evaluate((scope, label, on) => {
    const box = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, label, on);
  await pause(120);
};
const question = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
const commit = async (needle, value) => {
  const q = await question(needle);
  await q.evaluate((e) => e.scrollIntoView({ block: 'center' }));
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q;
};
const shot = async (scope, name) => {
  if (!out) return;
  await page.$eval(scope, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch03/3.2');
const MAP = bench('The thread map');
await page.waitForSelector(MAP);

// ---------------------------------------------------------------- predictions
let q = await commit('How many threads have no pixel', 400);
let t = await q.evaluate((e) => e.textContent);
check('surplus prediction reveals 5120 threads, 4712 pixels, 408 surplus', t.includes('5120 threads for 4712 pixels leaves 408') && t.includes('off by 8'), t);
q = await commit('Which element of that array does Thread (0,0) of Block (1,0) write', 1216);
t = await q.evaluate((e) => e.textContent);
check('offset prediction reveals row 16 and element 1216', t.includes('row 1·16 + 0 = 16') && t.includes('element 1216') && t.includes('what you predicted'), t);
await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Select this thread in the map')).click());
await pause(200);
const panel = (k) => text(`${MAP} [data-panel="thread"] [data-k="${k}"]`);
check('its button selects Thread (0,0) of Block (1,0) in the map', (await panel('title')) === 'Thread (0,0) of Block (1,0)', await panel('title'));
check('the map panel gives grayOffset 1216 with the numbers substituted', (await panel('gray')) === 'grayOffset = row*width + col = 16*76 + 0 = 1216', await panel('gray'));
check('and rgbOffset 3648', (await panel('rgb')) === 'rgbOffset = grayOffset*3 = 3648', await panel('rgb'));

// ---------------------------------------------------------------- the narrated scene
const S = 'section.scene[aria-label="A grid over a picture"]';
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const script = async (n) => strip(await (await fetch(`${BASE}/content/ch03/narration/grid-over-picture-${n}.txt`)).text());
const sceneText = () => page.$eval(S + ' .stage svg', (e) => e.textContent);
const part = async (label, pos) => {
  await page.evaluate((S, label) => [...document.querySelectorAll(S + ' .seg button')].find((b) => b.textContent.includes(label)).click(), S, label);
  await pause(80);
  await setInput(S, 'Position in this part', pos);
};
check('six parts are offered', (await page.$$(S + ' .seg button')).length === 6);
check('there is a captions switch', (await text(S)).includes('captions'));
const scripts = await Promise.all([1, 2, 3, 4, 5, 6].map(script));
check('each script is 35 to 60 words', scripts.every((s) => { const n = s.split(' ').length; return n >= 35 && n <= 60; }), scripts.map((s) => s.split(' ').length).join(','));
const manifest = await fetch(BASE + '/content/ch03/narration/manifest.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
const audible = !!manifest?.clips?.['grid-over-picture-1'];
if (!audible) check('with no audio yet, part 1 shows its script as a caption', (await text(S + ' .caption')) === scripts[0], await text(S + ' .caption'));
await setCheck(S, 'captions', true);
check('with captions on, part 1 shows its script', (await text(S + ' .caption')) === scripts[0]);
await part('2. Blocks', 1000);
let sv = await sceneText();
check('part 2 ends with 20 blocks and 5120 threads laid down', sv.includes('blocks: 20') && sv.includes('threads: 20 × 256 = 5120'), sv);
check('part 2 caption is its script', (await text(S + ' .caption')) === scripts[1]);
await part('3. The overhang', 1000);
sv = await sceneText();
check('part 3 counts the overhang: 256 + 160 - 8 = 408', sv.includes('4 columns × 64 rows = 256') && sv.includes('2 rows × 80 columns = 160') && sv.includes('408 threads with no pixel'), sv);
await part('4. One thread', 1000);
sv = await sceneText();
check('part 4 works out row 16 and column 0', sv.includes('= 1 × 16 + 0 = 16') && sv.includes('= 0 × 16 + 0 = 0'), sv);
await part('5. Into one strip', 1000);
sv = await sceneText();
check('part 5 finds element 1216 and the red byte at 3648', sv.includes('16 × 76 + 0 = 1216') && sv.includes('3 × 1216 = 3648') && sv.includes('r 3648'), sv);
await shot(S, 'scene-5');
await part('6. Four kinds of block', 1000);
sv = await sceneText();
check('part 6 adds 12×256 + 3×192 + 4×224 + 1×168 to 4712', ['12 × 256 = 3072', '3 × 192 = 576', '4 × 224 = 896', '1 × 168 = 168', 'sum 4712 = 62 × 76'].every((x) => sv.includes(x)), sv);
await shot(S, 'scene-6');
await part('1. The picture', 0);
await clickText(S, 'Play');
await pause(1800);
const pos = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('pressing Play moves the animation', pos > 20, String(pos));
await clickText(S, 'Pause');
await setCheck(S, 'captions', false);

// ---------------------------------------------------------------- the thread map
const total = (k) => text(`${MAP} [data-total="${k}"]`);
check('totals for the book picture: 20 blocks, 5120 threads, 4712 pixels, 408 surplus',
  (await total('blocks')).includes('5 across × 4 down = 20') && (await total('threads')).includes('80 across × 64 down = 5120') &&
  (await total('pixels')).includes('76 across × 62 down = 4712') && (await total('surplus')).endsWith('408'),
  [await total('blocks'), await total('threads'), await total('pixels'), await total('surplus')].join(' | '));
const cellPoint = async (id, x, y) => {
  await page.$eval(`svg[data-lattice="${id}"]`, (e) => e.scrollIntoView({ block: 'center' }));
  await pause(100);
  return page.$eval(`svg[data-lattice="${id}"]`, (el, x, y) => {
    const r = el.getBoundingClientRect();
    const d = el.dataset;
    const k = r.width / Number(d.vbw);
    return { px: r.left + (Number(d.ml) + (x - Number(d.x0) + 0.5) * Number(d.cell)) * k, py: r.top + (Number(d.mt) + (y - Number(d.y0) + 0.5) * Number(d.cell)) * k };
  }, x, y);
};
const clickCell = async (id, x, y) => { const p = await cellPoint(id, x, y); await page.mouse.click(p.px, p.py); await pause(120); };
await clickCell('map', 37, 21);
check('clicking the thread at column 37, row 21 selects Thread (5,5) of Block (1,2)', (await panel('title')) === 'Thread (5,5) of Block (1,2)', await panel('title'));
check('its row is 1*16 + 5 = 21', (await panel('row')) === 'row = blockIdx.y*blockDim.y + threadIdx.y = 1*16 + 5 = 21', await panel('row'));
check('its column is 2*16 + 5 = 37', (await panel('col')) === 'col = blockIdx.x*blockDim.x + threadIdx.x = 2*16 + 5 = 37', await panel('col'));
check('its offsets are 1633 and 4899', (await panel('gray')).endsWith('= 1633') && (await panel('rgb')).endsWith('= 4899'));
check('the Pout strip lights slot 1633 and the Pin strip three bytes from 4899',
  (await page.$$eval(`${MAP} [data-panel="thread"] .mx span.hot`, (e) => e.map((x) => x.textContent))).join(',') === '1633,4899r,4900g,4901b');
await shot(MAP, 'map');
await clickText(MAP, 'a thread with no pixel');
check('a surplus thread fails the column clause: 76 < 76 is false', (await panel('guard')).includes('76 < 76 is false') && (await panel('guard')).includes('the thread stops'), await panel('guard'));
check('and has no offsets', await has(`${MAP} [data-panel="thread"] [data-k="nomem"]`) && !(await has(`${MAP} [data-panel="thread"] [data-k="gray"]`)));
await page.$eval('svg[data-lattice="map"]', (e) => e.focus());
await page.keyboard.press('ArrowLeft');
await pause(120);
check('the left arrow key moves to column 75, which works and writes Pout[75]', (await panel('gray')) === 'grayOffset = row*width + col = 0*76 + 75 = 75', await panel('gray'));
await page.keyboard.down('Shift');
await page.keyboard.press('ArrowDown');
await page.keyboard.up('Shift');
await pause(120);
check('Shift and the down arrow move one block down, to row 16', (await panel('row')).endsWith('= 16'), await panel('row'));
const rows = await page.$$eval(`${MAP} table.threads tbody tr[data-region]`, (r) => r.map((x) => [x.dataset.region, ...[...x.querySelectorAll('td')].map((c) => c.textContent)]));
check('four kinds of block: 12 × 256, 3 × 192, 4 × 224, 1 × 168',
  JSON.stringify(rows.slice(0, 4).map((r) => [r[0], r[2], parseInt(r[3]), r[4]])) === JSON.stringify([['1', '12', 256, '3072'], ['2', '3', 192, '576'], ['3', '4', 224, '896'], ['4', '1', 168, '168']]), JSON.stringify(rows));
check('their products add up to 4712, every pixel once', rows[4][2] === '4712' && (await text(`${MAP} [data-k="sum"]`)).includes('4712 = 62 × 76'), JSON.stringify(rows[4]));
const corner = await cellPoint('map', 70, 60);
await page.mouse.move(corner.px, corner.py);
await pause(150);
check('pointing at the corner block shows 168 of 256 threads working', (await text(`${MAP} [data-k="block"]`)).includes('Block (3,4) (under the pointer): 168 of 256 threads work, 12 across × 14 down'), await text(`${MAP} [data-k="block"]`));
await page.mouse.move(5, 5);

await setCheck(MAP, 'swap the grid extents', true);
check('swapped extents leave 744 pixels never written', (await text(`${MAP} [data-total="unreached"]`)) === '744 pixels never written' && (await has(`${MAP} [data-area="unreached"]`)));
check('and launch a kind of block that does no work', (await text(`${MAP} [data-k="sum"]`)).includes('short of 62 × 76 = 4712 by 744 pixels') && (await has(`${MAP} tr[data-region="idle"]`)));
await shot(MAP, 'map-swapped');
await page.evaluate((MAP) => {
  for (const [label, v] of [['Height (rows)', 48], ['Width (columns)', 48]]) {
    const el = [...document.querySelectorAll(MAP + ' label.field')].find((l) => l.textContent.startsWith(label)).querySelector('input');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
}, MAP);
await pause(150);
check('on a square 48 × 48 picture the swapped launch reaches every pixel', !(await has(`${MAP} [data-total="unreached"]`)) && (await total('pixels')).includes('48 across × 48 down = 2304'), await total('pixels'));
await setCheck(MAP, 'swap the grid extents', false);
await setSelect(MAP, 'Preset', 'book');

await setSelect(MAP, 'Launch style', 'threads');
check('Metal dispatchThreads creates exactly 76 × 62 = 4712 threads and no surplus', (await total('threads')).includes('76 across × 62 down = 4712') && (await total('surplus')).endsWith(' 0'), await total('threads'));
check('and the panel says the guard may be left out', await has(`${MAP} [data-k="noguard"]`));
await clickText(MAP, 'the last thread');
check('the last thread is in an edge group of (12, 14) and gets its position directly',
  (await panel('blockDim')).includes('this edge group is (12, 14)') && (await panel('gid')).includes('thread_position_in_grid = (75, 61)'), (await panel('blockDim')) + ' | ' + (await panel('gid')));
await shot(MAP, 'map-metal-threads');
await setSelect(MAP, 'Launch style', 'threadgroups');
check('Metal dispatchThreadgroups is back to 5120 threads and 408 surplus', (await total('threads')).includes('5120') && (await total('surplus')).endsWith('408') && (await total('blocks')).startsWith('threadgroups'));
check('with Metal names in the panel', (await panel('blockIdx')).startsWith('threadgroup_position_in_grid') && (await panel('threadIdx')).startsWith('thread_position_in_threadgroup'));
await setSelect(MAP, 'Launch style', 'blocks');

await setSelect(MAP, 'Preset', 'ex3');
check('exercise 3 preset: 19 × 5 = 95 blocks, 48640 threads', (await total('blocks')).includes('19 across × 5 down = 95') && (await total('threads')).includes('304 across × 160 down = 48640'), await total('blocks'));
check('and 45000 working threads', (await text(`${MAP} tr[data-region="sum"]`)).endsWith('45000'), await text(`${MAP} tr[data-region="sum"]`));
check('a large launch is drawn block by block with one block opened', (await has('svg[data-lattice="bmap"]')) && (await has('svg[data-lattice="inset"]')) && !(await has('svg[data-lattice="map"]')));
await clickCell('bmap', 18, 4);
check('picking the corner block of exercise 3 selects it', (await text(`${MAP} [data-k="block"]`)).includes('Block (4,18)'), await text(`${MAP} [data-k="block"]`));
await clickCell('inset', 18 * 16 + 11, 4 * 32 + 21);
check('picking a thread inside it gives row 149, column 299, the last element', (await panel('gray')) === 'grayOffset = row*width + col = 149*300 + 299 = 44999', await panel('gray'));
await shot(MAP, 'map-ex3');
await setSelect(MAP, 'Preset', 'large');
check('1500 × 2000 needs 125 × 94 = 11750 blocks', (await total('blocks')).includes('125 across × 94 down = 11750'), await total('blocks'));
await setSelect(MAP, 'Preset', 'small');
check('small preset: 16 × 12 = 192 threads, 62 surplus', (await total('threads')).includes('16 across × 12 down = 192') && (await total('surplus')).endsWith('62'), await total('threads'));
await setSelect(MAP, 'Block size', '16x32');
check('changing the block shape to 16 × 32 gives one block of 512 threads', (await total('blocks')).includes('1 across × 1 down = 1') && (await total('threads')).includes('16 across × 32 down = 512'), await total('threads'));
await setSelect(MAP, 'Preset', 'book');

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the kernel is shown for CUDA/HIP and Metal', tabs.includes('CUDA and HIP') && tabs.includes('Metal kernel'), tabs.join('|'));
const listing = await page.evaluate(() => [...document.querySelectorAll('figure.code')].find((f) => f.querySelector('.tabs')).querySelector('pre').textContent);
check('the corrected launch passes the output first', listing.includes('colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);'));
check('the corrected line casts the whole sum with the constants of the text', listing.includes('(unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);'));

// ---------------------------------------------------------------- the kernel stepper
const ST = bench('Step this thread through the kernel');
await page.$eval(ST, (e) => e.scrollIntoView({ block: 'start' }));
await clickText(ST, 'Thread (0,0) of Block (1,0)');
check('the stepper names its thread', (await text(`${ST} [data-k="who"]`)).startsWith('Thread (0,0) of Block (1,0)'), await text(`${ST} [data-k="who"]`));
await clickText(ST, 'Step');
let n = await text(ST + ' .narrate');
check('step 1 computes col with the numbers substituted', n === 'col = blockIdx.x*blockDim.x + threadIdx.x = 0*16 + 0 = 0.', n);
check('and highlights the col line of the kernel', (await text(ST + ' .code .ln.hl')).includes('int col = blockIdx.x * blockDim.x + threadIdx.x;'));
await clickText(ST, 'Step');
await clickText(ST, 'Step');
n = await text(ST + ' .narrate');
check('step 3 is the guard, and this thread passes', n.includes('0 < 76, true') && n.includes('16 < 62, true') && n.includes('has a pixel'), n);
await clickText(ST, 'Step');
n = await text(ST + ' .narrate');
check('step 4 computes grayOffset = 16*76 + 0 = 1216', n.includes('16*76 + 0 = 1216'), n);
await clickText(ST, 'Step');
await clickText(ST, 'Step');
n = await text(ST + ' .narrate');
check('step 6 reads the red byte at Pin[3648]', n.includes('r = Pin[rgbOffset] = Pin[3648] = 130'), n);
check('and lights that byte in Pin', (await page.$eval(`${ST} [data-pin="3648"]`, (e) => e.className)) === 'hot');
await clickText(ST, 'Run to the end');
check('at the end Pout[1216] holds 112 and the three bytes read were 130, 106, 100',
  (await text(`${ST} [data-pout="1216"]`)) === '112' && (await page.$$eval(`${ST} [data-pin]`, (e) => e.map((x) => x.textContent).join(','))) === '130,106,100',
  await text(`${ST} [data-k="mem"]`));
check('the registers show the float sum and the truncated byte',
  (await text(`${ST} tr[data-reg="sum"]`)).includes('112.49') && (await text(`${ST} tr[data-reg="byte"]`)).endsWith('112'), await text(`${ST} tr[data-reg="sum"]`));
check('the run took 12 steps', (await text(ST)).includes('step 12 of 12'));
await shot(ST, 'stepper');
check('the map follows the stepper to the same thread', (await panel('gray')).endsWith('= 1216'));
await clickText(ST, 'one past the right edge');
await clickText(ST, 'Run to the end');
n = await text(ST + ' .narrate');
check('a surplus thread stops after the guard, in 4 steps, touching no memory',
  (await text(ST)).includes('step 4 of 4') && n.includes('no pixel') && (await text(`${ST} [data-k="mem"]`)).includes('reads and writes nothing'), n);
await clickText(ST, 'Back');
n = await text(ST + ' .narrate');
check('its guard fails on the column clause and the row test is not evaluated', n.includes('76 < 76, false') && n.includes('not evaluated'), n);
check('the guard line is the one highlighted', (await text(ST + ' .code .ln.hl')).includes('if (col < width && row < height)'));

// ---------------------------------------------------------------- the flattening explorer
const FL = bench('Flatten an array');
await page.$eval(FL, (e) => e.scrollIntoView({ block: 'start' }));
const clickEl = async (sel) => { await page.$eval(sel, (e) => e.click()); await pause(100); };
const formula = () => text(`${FL} [data-k="formula"]`);
check('default 3 × 5 array: element [1,2] is at slot 1*5 + 2 = 7', (await formula()) === 'element [1,2] is at slot row*width + col = 1*5 + 2 = 7', await formula());
await clickEl(`${FL} [data-el="0,2,1"]`);
check('clicking element [2,1] lights slot 11', (await formula()).endsWith('2*5 + 1 = 11') && (await page.$eval(`${FL} [data-slot="11"]`, (e) => e.className)) === 'hot', await formula());
await clickEl(`${FL} [data-slot="4"]`);
check('clicking slot 4 gives back indices [0,4]', (await formula()).startsWith('element [0,4]'), await formula());
await setCheck(FL, 'show the strides', true);
check('the stride tuple of a (3, 5) array is (5, 1)', (await text(`${FL} [data-k="strides"]`)).startsWith('x.stride() for shape (3, 5) = (5, 1)'), await text(`${FL} [data-k="strides"]`));
await clickEl(`${FL} [data-el="0,1,1"]`);
const marks = await page.$$eval(`${FL} [data-strip] > div`, (d) => d.map((x) => x.lastElementChild.textContent).map((s, i) => (s ? `${i}:${s}` : '')).filter(Boolean));
check('one step along a column is +1 and one step along a row is +5', marks.join(' ') === '7:col+1 11:row+1', marks.join(' '));
await clickText(FL, 'column-major');
check('column-major: element [1,1] is at col*height + row = 1*3 + 1 = 4', (await formula()).endsWith('col*height + row = 1*3 + 1 = 4'), await formula());
const tr = await page.$$eval(`${FL} [data-grid="transpose"] span`, (e) => e.map((x) => x.textContent));
const st = await page.$$eval(`${FL} [data-slot]`, (e) => e.map((x) => x.textContent));
check('the transpose read row by row is the column-major strip', tr.join(' ') === st.join(' ') && st.slice(0, 4).join(' ') === '0,0 1,0 2,0 0,1', st.join(' '));
await shot(FL, 'flatten-colmajor');
await clickText(FL, 'wrong stride (a defect)');
await setInput(FL, 'Height', 2);
check('wrong stride on a 2 × 5 array: 3 slots written twice, 3 never', (await text(`${FL} [data-k="coverage"]`)).includes('3 slots written twice or more') && (await text(`${FL} [data-k="coverage"]`)).includes('3 slots never written'), await text(`${FL} [data-k="coverage"]`));
check('the collided slots are 2, 3 and 4', (await page.$$eval(`${FL} [data-slot].bad`, (e) => e.map((x) => x.dataset.slot).join(','))) === '2,3,4');
await shot(FL, 'flatten-wrong');
await setInput(FL, 'Height', 5);
await setInput(FL, 'Width', 2);
check('on a 5 × 2 array it writes 6 times past the end', (await text(`${FL} [data-k="coverage"]`)).includes('6 writes past the end') && (await page.$$(`${FL} [data-oob]`)).length === 6, await text(`${FL} [data-k="coverage"]`));
await setInput(FL, 'Height', 4);
await setInput(FL, 'Width', 4);
check('on a square 4 × 4 array the defect is invisible', (await text(`${FL} [data-k="coverage"]`)).includes('every slot written exactly once') && (await text(`${FL} [data-k="coverage"]`)).includes('cannot be seen'));
await clickText(FL, 'row-major');
await setInput(FL, 'Depth', 3);
await setInput(FL, 'Width', 5);
await clickEl(`${FL} [data-el="2,3,4"]`);
check('three planes of 4 × 5: element [2,3,4] is at 2*4*5 + 3*5 + 4 = 59, the last of 60 slots',
  (await formula()).endsWith('plane*height*width + row*width + col = 2*4*5 + 3*5 + 4 = 59') && (await page.$$(`${FL} [data-slot]`)).length === 60, await formula());
check('and the strides are (20, 5, 1)', (await text(`${FL} [data-k="strides"]`)).includes('for shape (3, 4, 5) = (20, 5, 1)'), await text(`${FL} [data-k="strides"]`));

// ---------------------------------------------------------------- the erratum hunt
const EH = bench('Find what is wrong in Fig. 3.4');
await page.$eval(EH, (e) => e.scrollIntoView({ block: 'start' }));
check('the printed kernel has 21 lines and line 19 is the printed store', (await page.$$(`${EH} [data-listing="k"] > div`)).length === 21 &&
  (await page.$eval(`${EH} [data-listing="k"] > div:nth-child(19)`, (e) => e.textContent)).includes('(unsigned char) 0.21f*r+0.71f*g+0.07f*b;'));
check('fourteen spots can be marked', (await page.$$(`${EH} [data-spot]`)).length === 14);
for (const id of ['cast', 'guard']) await clickEl(`${EH} [data-spot="${id}"]`);
await clickText(EH, 'Check my marks');
check('marking the cast and the guard scores 1 of 3', (await page.$eval(`${EH} [data-k="hunt"]`, (e) => e.dataset.found)) === '1' && (await text(`${EH} [data-k="hunt"]`)).includes('1 of 3 defects found'));
check('the guard is explained as not a defect', await has(`${EH} [data-alarm="guard"]`) && (await page.$eval(`${EH} [data-spot="guard"]`, (e) => e.dataset.state)) === 'wrong');
check('the defects not yet found are not given away', !(await has(`${EH} [data-finding="constants"]`)) && !(await has(`${EH} [data-finding="args"]`)));
await clickText(EH, 'Clear');
for (const id of ['cast', 'consts', 'args', 'name']) await clickEl(`${EH} [data-spot="${id}"]`);
await clickText(EH, 'Check my marks');
t = await text(`${EH} [data-k="hunt"]`);
const chips = await page.$$eval(`${EH} [data-finding] .chip`, (e) => e.map((x) => x.parentElement.dataset.finding + ':' + x.textContent).join(' '));
check('marking the cast, the constants and the call arguments scores 3 of 3', t.includes('3 of 3 defects found') && chips === 'cast:found constants:found args:found spelling:found', chips);
check('the spelling is listed as a fourth, unscored finding', (await text(`${EH} [data-finding="spelling"]`)).includes('Not scored'));
await shot(EH, 'hunt');
await clickText(EH, 'Clear');
await clickText(EH, 'Show the answers');
check('showing the answers lists all four findings and outlines the missed spots',
  (await page.$$(`${EH} [data-finding]`)).length === 4 && (await page.$$(`${EH} [data-spot][data-state="missed"]`)).length === 6);

// ---------------------------------------------------------------- the pixel tester
check('the pixel tester is closed until a prediction is made', (await has('[data-k="tester-closed"]')) && !(await has(bench('What each version stores'))));
q = await commit('What byte does line 19 as printed store', 0);
t = await q.evaluate((e) => e.textContent);
check('pure red: stored as 0, where 53 or 76 was meant', t.includes('It stores 0, as you predicted') && t.includes('would give 53') && t.includes('gives 76'), t);
const PT = bench('What each version stores');
await page.waitForSelector(PT);
const bytes = () => page.$$eval(`${PT} td[data-byte]`, (e) => e.map((x) => x.dataset.byte).join(','));
check('(200, 100, 50) is stored as 74, 116 and 124', (await bytes()) === '74,116,124', await bytes());
await clickText(PT, 'pure red');
check('pure red is stored as 0, 53 and 76', (await bytes()) === '0,53,76', await bytes());
await shot(PT, 'tester');
await clickText(PT, 'white');
check('white is stored as 198, 252 and 255', (await bytes()) === '198,252,255', await bytes());
await setInput(PT, 'r', 0);
await setInput(PT, 'g', 100);
await setInput(PT, 'b', 0);
check('the sliders work: (0, 100, 0) is stored as 71, 71 and 58', (await bytes()) === '71,71,58', await bytes());

// ---------------------------------------------------------------- questions
q = await commit('How many pixels are never written', 744);
check('the swapped-extents question accepts 744', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await commit('What is the flat offset of', 1008010);
check('the stride question accepts 1008010', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('What does it store?');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('Red contributes nothing')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('the precedence question accepts "red contributes nothing"', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('Which launch needs the range test');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith('Only dispatchThreads.')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('a wrong answer to the Metal launch question is marked and explained', (await q.evaluate((e) => e.textContent)).includes('Not quite.') && (await q.evaluate((e) => e.textContent)).includes('408'));
check('eight questions are asked on the page', (await page.$$eval('.q .kind', (k) => k.filter((x) => x.textContent !== 'Predict first').length)) === 8);

// A missing narration manifest (audio is generated centrally, later) is the only 404 allowed.
const expected404 = (u) => /\/content\/ch03\/narration\/(manifest\.json|grid-over-picture-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
