// Browser check for section X3.1: the op map (every op and project, against sim/backendmap.ts),
// the "Follow the call" stepper (every route walked, a wrong pick first), the narrated scene
// "One RMSNorm call, three codebases" (scripts, captions, drawings against the model), the two
// tables, and the questions.
// Usage: APP_URL=http://localhost:6721 node e2e/x3-sx31.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const click = async (sel) => {
  await page.$eval(sel, (b) => b.click());
  await pause(60);
};
const clickText = async (scope, text, exact = true) => {
  const ok = await page.evaluate((scope, text, exact) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => (exact ? x.textContent.trim() === text : x.textContent.includes(text)));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, text, exact);
  await pause(60);
  return ok;
};

await goto(page, 'x3/X3.1');
const M = await page.evaluate(async () => {
  const b = await import('/src/sim/backendmap.ts');
  const s = await import('/src/sim/x3sources.ts');
  return {
    ops: b.OP_IDS,
    projects: b.PROJECTS,
    cells: Object.fromEntries(b.OPS.map((o) => [o.id, o.cells])),
    commits: Object.fromEntries(s.PINNED.map((p) => [p.id, p.commit])),
    routes: Object.fromEntries(b.ROUTES.map((r) => [r.id, { hops: r.hops.map((h) => ({ right: h.options.find((o) => o.correct).label, wrong: h.options.find((o) => !o.correct).label, why: h.options.find((o) => !o.correct).why, found: h.found })), start: r.start }])),
    choices: b.ROUTE_CHOICES,
    fused: b.fuseGraph(b.ONE_CALL.graph, true),
    metal: b.metalNormName(b.ONE_CALL.graph, 1),
    entry: b.ENTRY_POINTS.length,
  };
});

// ---------------------------------------------------------------- op map
const OM = bench('Op map');
await page.waitForSelector(OM);
let mismatches = [];
let links = 0;
let badLinks = [];
for (const op of M.ops)
  for (const p of M.projects) {
    await click(`${OM} button[data-op="${op}"]`);
    await click(`${OM} button[data-project="${p}"]`);
    const cards = await page.$$eval(`${OM} [data-part=cells] section[data-backend]`, (cs) =>
      cs.map((c) => ({
        b: c.dataset.backend,
        file: c.dataset.file,
        tags: c.querySelector('[data-part=tags]').dataset.tags,
        links: [...c.querySelectorAll('a')].map((a) => a.href),
        nofile: !!c.querySelector('[data-part=nofile]'),
      })),
    );
    for (const c of cards) {
      const m = M.cells[op][p][c.b];
      if (c.file !== (m.file ?? '') || c.tags !== m.tags.join('|')) mismatches.push(`${op}/${p}/${c.b}`);
      if (!m.file && !m.tags.includes('no backend') && !c.nofile) mismatches.push(`${op}/${p}/${c.b}: no "no file" mark`);
      for (const h of c.links) {
        links++;
        if (h.includes('/blob/') || h.includes('/tree/')) {
          const pinnedHere = Object.values(M.commits).some((x) => h.includes('/' + x + '/'));
          const master = h.includes('/blob/master/') || h.includes('/blob/main/') || h.includes('/tree/main/');
          if (!pinnedHere && !(master && m.readAt === 'master')) badLinks.push(h);
        }
      }
    }
  }
check('op map: all 27 op and project choices show the model’s files and tags', mismatches.length === 0, mismatches.join(', '));
check('op map: every code link is a pinned permalink, or a master URL only for cells read at master', badLinks.length === 0 && links > 100, badLinks.slice(0, 3).join(' '));
await click(`${OM} button[data-op="add"]`);
await click(`${OM} button[data-project="pytorch"]`);
const addCuda = await page.$eval(`${OM} section[data-backend=cuda]`, (c) => c.textContent);
check('PyTorch add on CUDA: generated at build time, no file', addCuda.includes('generated at build time') && addCuda.includes('no file to open'));
await click(`${OM} button[data-op="silu"]`);
await click(`${OM} button[data-project="mlx"]`);
check('MLX: HIP column says no backend', (await page.$eval(`${OM} [data-readout=hip]`, (e) => e.textContent)).includes('no AMD backend'));
await click(`${OM} button[data-op="rmsnorm"]`);
await click(`${OM} button[data-project="llama.cpp"]`);
const files = await page.$$eval(`${OM} section[data-backend]`, (cs) => Object.fromEntries(cs.map((c) => [c.dataset.backend, c.dataset.file])));
check('llama.cpp RMSNorm: HIP is the CUDA file', files.hip === files.cuda && files.cuda.endsWith('norm.cu'));
if (out) await page.screenshot({ path: out + '-opmap.png' });

// ---------------------------------------------------------------- Follow the call
const FC = bench('Follow the call');
const optClick = async (label) => {
  const ok = await page.evaluate((FC, label) => {
    const b = [...document.querySelectorAll(FC + ' [data-part=hop] button')].find((x) => x.dataset.option === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, FC, label);
  await pause(60);
  return ok;
};
async function walk(routeId, label) {
  const r = M.routes[routeId];
  const start = await page.$eval(`${FC} [data-readout=start]`, (e) => e.textContent);
  check(`${label}: start shown`, start.includes(r.start), start);
  for (let i = 0; i < r.hops.length; i++) {
    const h = r.hops[i];
    if (i === 0) {
      await optClick(h.wrong);
      const why = await page.$eval(`${FC} [data-readout=why]`, (e) => e.textContent).catch(() => '');
      check(`${label}: a wrong first pick is explained and the hop stays open`, why.includes(h.why) && !(await page.$(`${FC} [data-readout=found]`)), why);
    }
    check(`${label}: hop ${i + 1}, the right option is offered`, await optClick(h.right));
    const found = await page.$eval(`${FC} [data-readout=found]`, (e) => ({ t: e.textContent, href: e.querySelector('a[data-part=permalink]').href }));
    if (!found.t.includes(h.found) || !Object.values(M.commits).some((c) => found.href.includes(c))) check(`${label}: hop ${i + 1} reveal`, false, found.href);
    await clickText(FC, i + 1 < r.hops.length ? 'Next hop' : 'Finish');
  }
  const end = await page.$eval(`${FC} [data-readout=end]`, (e) => e.textContent);
  check(`${label}: ends with the score, ${r.hops.length - 1} of ${r.hops.length} right first time`, end.includes(`Done: ${r.hops.length - 1} of ${r.hops.length}`), end);
  const trail = await page.$$eval(`${FC} [data-part=trail] li`, (l) => l.length);
  check(`${label}: the trail lists every hop`, trail === r.hops.length);
}
for (const c of M.choices) {
  await clickText(FC, c.label);
  if (c.variants) {
    await walk(c.variants.on, c.label + ' (fusion on)');
    await page.$eval(`${FC} label.check input`, (i) => i.click());
    await pause(80);
    await walk(c.variants.off, c.label + ' (fusion off)');
    await page.$eval(`${FC} label.check input`, (i) => i.click());
    await pause(80);
  } else await walk(c.id, c.label);
}
if (out) await page.screenshot({ path: out + '-follow.png' });

// ---------------------------------------------------------------- the two tables
check('entry-point table has a row per project', (await page.$$eval('table[data-part=entry-points] tbody tr', (r) => r.length)) === M.entry);
check('differences table has the three rows', (await page.$$eval('table[data-part=differ] tbody tr', (r) => r.length)) === 3);

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="One RMSNorm call, three codebases"]';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x3/narration/one-call-${n}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('scene: six scripts of 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scene: no symbols read aloud', scripts.every((s) => !/[_=<>{}*#/\\]/.test(s)), scripts.find((s) => /[_=<>{}*#/\\]/.test(s)));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const part = async (k) => {
  await page.evaluate((S, k) => [...document.querySelectorAll(S + ' .seg button')][k].click(), S, k);
  await pause(120);
};
const svg = () => page.$eval(S + ' .stage svg', (e) => e.textContent.replace(/\s+/g, ' '));
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
check('scene: six parts', (await page.$$eval(S + ' .seg button', (b) => b.length)) === 6);
check('scene: captions off by default', !(await page.$eval(S + ' label.check:last-of-type input', (i) => i.checked)));
check('scene: with no audio, the script is shown regardless', (await caption()) === scripts[0], await caption());
await seek(1000);
const t1 = await svg();
check('beat 1 draws the four graph nodes', ['GGML_OP_GET_ROWS', 'GGML_OP_RMS_NORM', 'GGML_OP_MUL', 'GGML_OP_MUL_MAT'].every((w) => t1.includes(w)), t1);
await part(1);
await seek(1000);
let t = await svg();
check('beat 2 ends with the fused call from fuseGraph and the plain call struck out', t.includes(M.fused[1].call) && t.includes('ggml_cuda_op_rms_norm') && t.includes('not reached'), t);
await part(2);
await seek(1000);
t = await svg();
check('beat 3 shows norm.cu and the HIP tag from the model', t.includes('norm.cu: rms_norm_f32, lines 127 to 272') && t.includes('same source as CUDA'), t);
await part(3);
await seek(1000);
t = await svg();
check('beat 4 has typed the whole Metal name and n_fuse', t.includes(`"${M.metal.name}"`) && t.includes(`n_fuse = ${M.metal.consumed}`), t);
await part(4);
await seek(1000);
t = await svg();
check('beat 5 matches the host_name instantiation with F = 2', t.includes(`F = ${M.metal.F}`) && t.includes('kernel_rms_norm_mul_add_f32'), t);
check('beat 5 caption is its script', (await caption()) === scripts[4]);
await part(5);
await seek(1000);
t = await svg();
check('beat 6 shows the three entry points', ['llama.cpp', 'MLX', 'PyTorch', 'eval_gpu', 'ggml_cuda_compute_forward'].every((w) => t.includes(w)), t);
if (out) await page.screenshot({ path: out + '-scene.png' });
// Play advances by itself (silent: no manifest).
await part(0);
await page.$eval(S + ' button.primary', (b) => b.click());
await pause(300);
check('scene: play starts the clock', (await page.$eval(S + ' button.primary', (b) => b.textContent.trim())) === 'Pause');
await page.$eval(S + ' button.primary', (b) => b.click());

// ---------------------------------------------------------------- questions
const answer = async (needle, index) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
check('q-no-add-file: right answer accepted', (await answer('find nothing. Why?', 0)).includes('Correct.'));
check('q-fusion-breakpoint: wrong answer explained', (await answer('a breakpoint in ggml_cuda_op_rms_norm', 2)).includes('Not quite.'));
check('q-metal-string: right answer accepted', (await answer('what do you search the', 0)).includes('Correct.'));
check('q-mlx-hip: right answer accepted', (await answer('through a hipify step', 0)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
